"""Turning a provider response into a validated typed action (PLAN.md 5.3).

PLAN 5.3's first acceptance criterion is that "provider responses become
validated typed actions", and its second is that "malformed outputs produce
bounded retries or a recorded failure". Both need the same thing first: the
ways a response can fail to become an action must be *distinguishable*, because
"the model is bad at this task", "the prompt does not describe the format",
"the model keeps choosing masked actions" and "the endpoint is down" call for
four different fixes and produce four identical-looking stalled runs otherwise.

So parsing returns one of two values, and a failure names its kind:

``unparseable``
    Nothing in the text identified an action at all. Usually a prompt problem.
``unknown_action``
    A name was given and the catalog does not contain it. The model invented an
    action -- often a plausible Factorio verb that this catalog does not expose.
``out_of_range``
    An integer was given outside the action space. The model is indexing a
    catalog it imagined rather than the one it was shown.
``illegal_action``
    A real, in-range action that the environment's mask currently forbids. The
    model ignored the legality list it was given; the environment would reject
    it, and PLAN section 2 forbids silently turning it into a different action.

A provider-level failure -- timeout, HTTP error, empty response -- is
deliberately *not* in this enum. It is carried by ``ModelReply.error`` and
recorded as ``provider_error``, because a call that never returned text is not
the model getting the format wrong.

Sequences (roadmap A2.3)
------------------------
A reply may carry one action or an ordered batch of up to
``MAX_ACTIONS_PER_SEQUENCE``. Both shapes land in the same :class:`ParsedSequence`,
so the one-action contract is the batch of length one rather than a separate
path, and every consumer that only wants the first action keeps working.

Only the **first** action is checked against the world here. Mask legality,
target handles and argument *values* are all properties of a state, and only
action 1 executes against the state the model was shown; A2.3 requires the loop
to validate again immediately before it dispatches each later action, which it
does by calling :func:`check_against_state` with a freshly derived mask. What is
checked here for every action is what does not depend on a state: that the verb
exists, that its index is in range, and that it declares no argument the catalog
does not know about.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

from factoriorl.agent.summary import LegalAction


class DecisionFailure(StrEnum):
    """Why a response did not become an action."""

    UNPARSEABLE = "unparseable"
    UNKNOWN_ACTION = "unknown_action"
    OUT_OF_RANGE = "out_of_range"
    ILLEGAL_ACTION = "illegal_action"
    #: A `target` was given for an action that acts on nothing, or naming a
    #: handle that is not in the observation. Distinct from `illegal_action`
    #: because the verb was fine and only the addressee was wrong.
    UNKNOWN_TARGET = "unknown_target"
    #: A parameterized action was chosen without every argument it needs.
    #: Distinct from `unparseable` because the verb and the format were both
    #: fine: `place_at` was named and `position` was not supplied.
    MISSING_ARGUMENT = "missing_argument"
    #: An argument was supplied whose value is not in the domain the
    #: observation offers. Distinct from `illegal_action` for the same reason
    #: `unknown_target` is: the verb was legal and only the value was wrong,
    #: and the two call for different corrections.
    BAD_ARGUMENT = "bad_argument"
    #: More actions in one reply than A2.3 permits. Its own kind rather than
    #: `unparseable` because the batch was well formed and only too long, and
    #: because the alternative -- executing the first eight -- would run a plan
    #: the model did not write and report it as the one it did.
    TOO_MANY_ACTIONS = "too_many_actions"
    #: The call itself failed. Not produced here; recorded by the loop so that
    #: one vocabulary covers every reason a decision did not happen.
    PROVIDER_ERROR = "provider_error"


#: Roadmap A2.3's cap on one batch. Eight rather than unbounded because every
#: action after the first is dispatched against a world the model never saw:
#: the loop re-validates each one, but a long plan simply has more chances to
#: be refused halfway, and a batch that stops at action seven of forty is a
#: worse record of intent than four batches of eight.
MAX_ACTIONS_PER_SEQUENCE = 8


@dataclass(frozen=True)
class ParsedAction:
    index: int
    key: str
    #: Whatever the model said about why, kept verbatim for the replay record.
    reason: str = ""
    #: Handle of the entity the model chose to act on, when it named one. The
    #: catalog binds `$target` to the *nearest* entity because a discrete index
    #: cannot carry an argument, and an agent that can see which chest it wants
    #: and can only say "the nearest one" is the defect this field removes.
    target: str | None = None
    #: Values for a `parameterized-v1` action's declared arguments. Empty for a
    #: discrete catalog, where the environment binds every reference itself.
    arguments: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "key": self.key,
            "reason": self.reason,
            "target": self.target,
            "arguments": dict(self.arguments),
        }


@dataclass(frozen=True)
class ParsedSequence:
    """The ordered batch one reply asked for. Length one is the old contract.

    `reason` is the batch-level justification. An action that carried its own
    `reason` keeps it, so a plan can say why it is a plan and each step can say
    what it is for, and a replay of a batch is not eight copies of one sentence.
    """

    actions: tuple[ParsedAction, ...]
    reason: str = ""
    #: The standing intention the model stated, if it stated one (roadmap
    #: A3.2). Distinct from `reason`, which justifies *this* reply: a plan
    #: outlives the turn that declared it and is carried forward in the prompt
    #: until it is superseded or stalls.
    plan: str = ""
    #: Something the model claims about the world and wants back later. Stored
    #: as the model's assertion, never as an observation -- A3.2 requires the
    #: two be kept apart, and the prompt renders them under different headings.
    note: str = ""

    def __len__(self) -> int:
        return len(self.actions)

    @property
    def first(self) -> ParsedAction:
        return self.actions[0]

    def to_dict(self) -> dict:
        return {
            "reason": self.reason,
            "plan": self.plan,
            "note": self.note,
            "actions": [a.to_dict() for a in self.actions],
        }


@dataclass(frozen=True)
class ParseFailure:
    failure: DecisionFailure
    detail: str
    #: What was extracted before validation rejected it, when anything was.
    candidate: str | None = None
    #: The action the reply asked for, when the reply named a real one and the
    #: *arguments* were what failed. Carried so a refusal can be counted like
    #: any other failed attempt (roadmap A3.3): an out-of-domain position never
    #: reaches the engine, so without this the most likely way an agent gets
    #: stuck -- proposing the same illegal argument over and over -- would be
    #: the one kind of failure nothing counted.
    action: ParsedAction | None = None
    #: The standing fields the reply carried. A plan is not an action: a model
    #: that states its intention and then picks a bad argument has still stated
    #: its intention, and discarding it would punish the wrong half of the reply.
    plan: str = ""
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "failure": self.failure.value,
            "detail": self.detail,
            "candidate": self.candidate,
            "action": self.action.to_dict() if self.action else None,
            "plan": self.plan,
            "note": self.note,
        }


#: A fenced or bare JSON object anywhere in the response. Models wrap JSON in
#: prose and in code fences no matter how the prompt is worded, so the parser
#: reads the *last* object in the text: when a model restates the format example
#: before answering, the first object is the example and taking it would execute
#: the instructions instead of the decision.
#: Kept for the flat case and for the tests that pin it, but extraction now
#: scans for balanced braces -- see `_json_objects`. `\{[^{}]*\}` cannot match a
#: nested object, so once `arguments` became an object the regex matched only
#: the *inner* `{...}`, which has no "action" key, and 37 perfectly well-formed
#: replies in a row were recorded as `unparseable`.
_OBJECT = re.compile(r"\{[^{}]*\}", re.DOTALL)

#: The fallback shape, for a model that answers in a line rather than JSON.
_LINE = re.compile(r"\baction\b\s*[:=]\s*\"?([A-Za-z0-9_\-]+)\"?", re.IGNORECASE)


def _json_objects(text: str) -> list[str]:
    """Every balanced `{...}` span in the text, outermost first.

    A brace scan rather than a regex, because the reply may nest: an
    `arguments` object inside the decision object is two levels, and no
    non-recursive pattern can bracket that.
    """
    spans: list[str] = []
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for index, character in enumerate(text or ""):
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            if depth == 0:
                start = index
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                spans.append(text[start : index + 1])
                start = -1
            elif depth < 0:
                depth = 0
    return spans


def _extract_steps(text: str) -> tuple[list[Any], str, dict] | ParseFailure | None:
    """Pull the requested action or actions out of a response.

    Returns the raw steps in order, the batch-level reason, the optional
    standing fields (`plan`, `note`), a
    :class:`ParseFailure` when a batch was found and refused outright, or
    ``None`` when nothing in the text identified an action at all.
    """
    for match in reversed(_json_objects(text or "")):
        try:
            payload = json.loads(match)
        except ValueError:
            continue
        if not isinstance(payload, dict):
            continue
        reason = str(payload.get("reason") or "")
        # Optional and additive. A reply that omits both is byte-identical in
        # effect to every reply sent before A3, which is what keeps the whole
        # existing fixture corpus valid.
        standing = {
            "plan": str(payload.get("plan") or "").strip(),
            "note": str(payload.get("note") or "").strip(),
        }
        batch = payload.get("actions")
        if isinstance(batch, list):
            if "action" in payload:
                # Both shapes in one object. There is no reading of this that is
                # not a guess about which the model meant, and guessing here
                # executes a plan nobody wrote -- the same class of defect as
                # letting `{"action": true}` through as action 1.
                return ParseFailure(
                    DecisionFailure.UNPARSEABLE,
                    "the reply carried both 'action' and 'actions'; send one or the other",
                    match,
                )
            if not batch:
                return ParseFailure(
                    DecisionFailure.UNPARSEABLE,
                    "'actions' was an empty list, so the reply asked for nothing",
                    match,
                )
            if len(batch) > MAX_ACTIONS_PER_SEQUENCE:
                # Refused, never trimmed. Executing the first eight of twelve
                # would run a different plan from the one recorded as requested,
                # and the run's own record would not show that it had happened.
                return ParseFailure(
                    DecisionFailure.TOO_MANY_ACTIONS,
                    f"{len(batch)} actions in one reply; at most "
                    f"{MAX_ACTIONS_PER_SEQUENCE} may be sent at once",
                    match,
                )
            return list(batch), reason, standing
        if "action" in payload:
            return [payload], reason, standing
    line = _LINE.search(text or "")
    if line:
        return [{"action": line.group(1)}], "", {"plan": "", "note": ""}
    return None


def _step_fields(step: Any) -> tuple[Any, str, str | None, dict] | ParseFailure:
    """One entry of a batch, normalised to the fields a single reply carries."""
    if isinstance(step, (int, str)) and not isinstance(step, bool):
        # `{"actions": [3, 7]}`. A prompt that asks for a list of actions gets
        # bare indices back often enough that refusing them would repeat the
        # `_OBJECT` defect: a reply whose intent is unambiguous, filed as
        # malformed because the parser wanted a different envelope.
        step = {"action": step}
    if not isinstance(step, dict):
        return ParseFailure(
            DecisionFailure.UNPARSEABLE,
            f"an entry was {type(step).__name__}, not an action object or an index",
            json.dumps(step, default=str),
        )
    if "action" not in step:
        return ParseFailure(
            DecisionFailure.UNPARSEABLE,
            "an entry had no 'action' field",
            json.dumps(step, sort_keys=True, default=str),
        )
    target = step.get("target")
    arguments = step.get("arguments")
    return (
        step["action"],
        str(step.get("reason") or ""),
        str(target) if isinstance(target, (str, int)) and str(target) else None,
        dict(arguments) if isinstance(arguments, dict) else {},
    )


def _extract(text: str) -> tuple[Any, str, str | None, dict] | None:
    """The single-action view of the same scan: the first action a reply names.

    Kept as its own entry point because the flat four-tuple is what the
    nested-object regression is pinned on, and that regression -- see `_OBJECT`
    -- cost 37 well-formed replies in a row.
    """
    steps = _extract_steps(text)
    if not isinstance(steps, tuple):
        return None
    fields = _step_fields(steps[0][0])
    return None if isinstance(fields, ParseFailure) else fields


#: Which observation-derived domain each argument name draws from. Mirrors
#: `factoriorl.catalog.ARGUMENT_DOMAINS`, imported rather than restated so the
#: agent and the environment cannot disagree about what a name means.
def _domain_of(argument: str, key: str = "") -> str | None:
    """Which domain an argument draws from, for the action that is asking.

    `item` means different things to `give_to` and `take_from` -- what you hold
    versus what the source holds -- and validating both against the character's
    inventory made collecting the first unit of a new product impossible.
    """
    from factoriorl.catalog import ARGUMENT_DOMAINS, ARGUMENT_DOMAINS_BY_KEY

    override = ARGUMENT_DOMAINS_BY_KEY.get(key, {}).get(argument)
    return override or ARGUMENT_DOMAINS.get(argument)


def _validate_arguments(
    key: str,
    supplied: dict,
    required: tuple[str, ...],
    domains: dict,
) -> ParseFailure | dict:
    """Check every declared argument against the domain the observation offers.

    Checked here rather than left to `env.step_arguments` so a wrong value is a
    *named decision failure the model can be corrected on*, inside the retry
    budget, instead of a `ValueError` that ends the episode. The environment
    still re-checks it: this narrows, it does not decide.
    """
    missing = [name for name in required if name not in supplied]
    if missing:
        return ParseFailure(
            DecisionFailure.MISSING_ARGUMENT,
            f"{key} needs {', '.join(required)}; missing {', '.join(missing)}",
            json.dumps(supplied, sort_keys=True),
        )
    extra = sorted(set(supplied) - set(required))
    if extra:
        return ParseFailure(
            DecisionFailure.BAD_ARGUMENT,
            f"{key} takes only {', '.join(required) or 'no arguments'}; "
            f"got {', '.join(extra)} as well",
            json.dumps(supplied, sort_keys=True),
        )
    cleaned: dict = {}
    for name in required:
        value = supplied[name]
        domain_name = _domain_of(name, key)
        legal_values = domains.get(domain_name) if domain_name else None
        if isinstance(value, list):
            # A position arrives as a JSON array and the domain holds lists.
            value = [float(v) for v in value]
        if legal_values is not None and value not in list(legal_values):
            shown = list(legal_values)[:6]
            return ParseFailure(
                DecisionFailure.BAD_ARGUMENT,
                f"{name}={value!r} is not one of the {len(list(legal_values))} legal "
                f"values for {domain_name} (e.g. {shown})",
                json.dumps({name: supplied[name]}, sort_keys=True),
            )
        cleaned[name] = value
    return cleaned


def _resolve_index(raw: Any, vocabulary: tuple[tuple[str, str], ...]) -> int | ParseFailure:
    """Which catalog slot the model named, whether by name or by index.

    State-independent, so a batch checks it for every action: an invented verb
    or an index off the end of the catalog is wrong now and will still be wrong
    when its turn comes, and saying so inside the retry budget is cheaper than
    dispatching six actions and then refusing the seventh.
    """
    keys = [key for key, _ in vocabulary]

    # An integer, or a string that is entirely digits: the model chose by index.
    index: int | None = None
    if isinstance(raw, bool):
        # `bool` is an `int` in Python, and `{"action": true}` reaching the
        # environment as action 1 would be a silent misexecution.
        return ParseFailure(DecisionFailure.UNPARSEABLE, "action was a boolean", str(raw))
    if isinstance(raw, int):
        index = raw
    elif isinstance(raw, str) and raw.strip().lstrip("-").isdigit():
        index = int(raw.strip())
    elif not isinstance(raw, str):
        return ParseFailure(
            DecisionFailure.UNPARSEABLE,
            f"action was {type(raw).__name__}, not a name or an index",
            str(raw),
        )

    if index is None:
        name = raw.strip()
        if name not in keys:
            return ParseFailure(
                DecisionFailure.UNKNOWN_ACTION,
                f"{name!r} is not in the action catalog",
                name,
            )
        index = keys.index(name)

    if not 0 <= index < len(vocabulary):
        return ParseFailure(
            DecisionFailure.OUT_OF_RANGE,
            f"index {index} is outside the action space of {len(vocabulary)}",
            str(index),
        )
    return index


def check_against_state(
    action: ParsedAction,
    legal: tuple[LegalAction, ...],
    *,
    targetable: frozenset[str] = frozenset(),
    handles: frozenset[str] = frozenset(),
    requires: dict[str, tuple[str, ...]] | None = None,
    domains: dict | None = None,
) -> ParsedAction | ParseFailure:
    """Everything about an action that is a fact about a *world*, not a catalog.

    Split out of `parse_action` because A2.3 requires the loop to re-run exactly
    this immediately before it dispatches action *N* of a batch. Action 1 is
    checked against the state the model was shown; action 2 is checked against
    the state action 1 left behind, and the two can disagree -- mining the last
    ore in reach masks `mine` out again, and a chest that has been emptied is
    still a handle but no longer a legal transfer. Sharing one function is what
    keeps the batch path from drifting into a second, weaker set of rules.
    """
    if action.index not in {a.index for a in legal}:
        return ParseFailure(
            DecisionFailure.ILLEGAL_ACTION,
            f"action {action.index} ({action.key}) is masked out in this state",
            action.key,
        )
    required = (requires or {}).get(action.key, ())
    if action.target is not None and required == ("handle",) and not action.arguments:
        # Legacy prompt wording advertised `target` generically. For the
        # unambiguous handle-only action shape, preserve the chosen game action
        # while normalizing it into the canonical arguments contract.
        action = replace(action, target=None, arguments={"handle": action.target})
    if action.target is not None:
        # An addressee is only meaningful for an action that acts on something,
        # and only if the thing is one the agent can currently see. Both are
        # refused rather than ignored: silently dropping the target would send
        # the action to the nearest entity instead, which is the behaviour the
        # model was trying to override.
        if action.key not in targetable:
            return ParseFailure(
                DecisionFailure.UNKNOWN_TARGET,
                f"{action.key} does not act on an entity, so it takes no target",
                action.target,
            )
        if action.target not in handles:
            return ParseFailure(
                DecisionFailure.UNKNOWN_TARGET,
                f"{action.target!r} is not a handle in the current observation",
                action.target,
            )
    if required or action.arguments:
        checked = _validate_arguments(action.key, action.arguments, required, domains or {})
        if isinstance(checked, ParseFailure):
            return checked
        return replace(action, arguments=checked)
    return action


def _at(failure: ParseFailure, position: int, total: int) -> ParseFailure:
    """Say which step of a batch was refused.

    A one-action reply -- still the common case -- keeps the message it always
    had, and a batch says `action 3 of 5` rather than leaving the model to guess
    which of its steps the correction was about.
    """
    if total <= 1:
        return failure
    return replace(failure, detail=f"action {position + 1} of {total}: {failure.detail}")


def parse_sequence(
    text: str,
    legal: tuple[LegalAction, ...],
    vocabulary: tuple[tuple[str, str], ...],
    *,
    targetable: frozenset[str] = frozenset(),
    handles: frozenset[str] = frozenset(),
    requires: dict[str, tuple[str, ...]] | None = None,
    domains: dict | None = None,
) -> ParsedSequence | ParseFailure:
    """Validate one response into the ordered batch it asked for.

    ``vocabulary`` is every action index that exists; ``legal`` is the subset
    the environment will currently accept. Both are needed to tell "you invented
    an action" from "that action exists but is not available right now", and
    that distinction is the difference between a prompt bug and a masking bug.

    Only ``actions[0]`` is checked against ``legal``, ``handles`` and
    ``domains`` -- see the module docstring. The loop calls
    :func:`check_against_state` again for each later action, against the world
    it is about to be dispatched into rather than the one it was planned in.
    """
    extracted = _extract_steps(text)
    if isinstance(extracted, ParseFailure):
        return extracted
    if extracted is None:
        return ParseFailure(
            DecisionFailure.UNPARSEABLE,
            "no JSON object with an 'action' or 'actions' field and no 'action:' line",
        )
    steps, reason, standing = extracted
    keys = [key for key, _ in vocabulary]
    actions: list[ParsedAction] = []
    for position, step in enumerate(steps):
        fields = _step_fields(step)
        if isinstance(fields, ParseFailure):
            return _standing(_at(fields, position, len(steps)), standing)
        raw, own_reason, target, supplied = fields
        index = _resolve_index(raw, vocabulary)
        if isinstance(index, ParseFailure):
            return _standing(_at(index, position, len(steps)), standing)
        action = ParsedAction(
            index=index,
            key=keys[index],
            reason=own_reason or reason,
            target=target,
            arguments=supplied,
        )
        if position == 0:
            checked = check_against_state(
                action,
                legal,
                targetable=targetable,
                handles=handles,
                requires=requires,
                domains=domains,
            )
            if isinstance(checked, ParseFailure):
                return _standing(replace(checked, action=checked.action or action), standing)
            action = checked
        actions.append(action)
    return ParsedSequence(
        tuple(actions),
        reason,
        plan=standing.get("plan", ""),
        note=standing.get("note", ""),
    )


def _standing(failure: ParseFailure, standing: dict) -> ParseFailure:
    """Carry a refused reply's plan and note through, so they are not lost."""
    return replace(failure, plan=standing.get("plan", ""), note=standing.get("note", ""))


def parse_action(
    text: str,
    legal: tuple[LegalAction, ...],
    vocabulary: tuple[tuple[str, str], ...],
    *,
    targetable: frozenset[str] = frozenset(),
    handles: frozenset[str] = frozenset(),
    requires: dict[str, tuple[str, ...]] | None = None,
    domains: dict | None = None,
) -> ParsedAction | ParseFailure:
    """The first action of a response, for callers that want exactly one.

    The one-action contract predates batching and several callers still hold
    it, so it stays a function rather than becoming each call site's job to
    reach into ``.actions[0]`` and get the failure case wrong.
    """
    outcome = parse_sequence(
        text,
        legal,
        vocabulary,
        targetable=targetable,
        handles=handles,
        requires=requires,
        domains=domains,
    )
    if isinstance(outcome, ParseFailure):
        return outcome
    return outcome.first
