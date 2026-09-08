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
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
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
    #: The call itself failed. Not produced here; recorded by the loop so that
    #: one vocabulary covers every reason a decision did not happen.
    PROVIDER_ERROR = "provider_error"


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
class ParseFailure:
    failure: DecisionFailure
    detail: str
    #: What was extracted before validation rejected it, when anything was.
    candidate: str | None = None

    def to_dict(self) -> dict:
        return {"failure": self.failure.value, "detail": self.detail, "candidate": self.candidate}


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


def _extract(text: str) -> tuple[Any, str, str | None, dict] | None:
    """Pull the action field out of a response, or ``None``."""
    for match in reversed(_json_objects(text or "")):
        try:
            payload = json.loads(match)
        except ValueError:
            continue
        if isinstance(payload, dict) and "action" in payload:
            target = payload.get("target")
            arguments = payload.get("arguments")
            return (
                payload["action"],
                str(payload.get("reason") or ""),
                str(target) if isinstance(target, (str, int)) and str(target) else None,
                dict(arguments) if isinstance(arguments, dict) else {},
            )
    line = _LINE.search(text or "")
    if line:
        return line.group(1), "", None, {}
    return None


#: Which observation-derived domain each argument name draws from. Mirrors
#: `factoriorl.catalog.ARGUMENT_DOMAINS`, imported rather than restated so the
#: agent and the environment cannot disagree about what a name means.
def _domain_of(argument: str) -> str | None:
    from factoriorl.catalog import ARGUMENT_DOMAINS

    return ARGUMENT_DOMAINS.get(argument)


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
        domain_name = _domain_of(name)
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
    """Validate one response against the catalog and the current mask.

    ``vocabulary`` is every action index that exists; ``legal`` is the subset
    the environment will currently accept. Both are needed to tell "you invented
    an action" from "that action exists but is not available right now", and
    that distinction is the difference between a prompt bug and a masking bug.
    """
    extracted = _extract(text)
    if extracted is None:
        return ParseFailure(
            DecisionFailure.UNPARSEABLE,
            "no JSON object with an 'action' field and no 'action:' line",
        )
    raw, reason, target, supplied = extracted
    legal_by_index = {a.index: a for a in legal}
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
    if index not in legal_by_index:
        return ParseFailure(
            DecisionFailure.ILLEGAL_ACTION,
            f"action {index} ({keys[index]}) is masked out in this state",
            keys[index],
        )
    key = keys[index]
    if target is not None:
        # An addressee is only meaningful for an action that acts on something,
        # and only if the thing is one the agent can currently see. Both are
        # refused rather than ignored: silently dropping the target would send
        # the action to the nearest entity instead, which is the behaviour the
        # model was trying to override.
        if key not in targetable:
            return ParseFailure(
                DecisionFailure.UNKNOWN_TARGET,
                f"{key} does not act on an entity, so it takes no target",
                target,
            )
        if target not in handles:
            return ParseFailure(
                DecisionFailure.UNKNOWN_TARGET,
                f"{target!r} is not a handle in the current observation",
                target,
            )
    required = (requires or {}).get(key, ())
    if required or supplied:
        checked = _validate_arguments(key, supplied, required, domains or {})
        if isinstance(checked, ParseFailure):
            return checked
        supplied = checked
    return ParsedAction(index=index, key=key, reason=reason, target=target, arguments=supplied)
