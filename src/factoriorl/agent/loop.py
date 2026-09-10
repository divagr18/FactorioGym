"""The provider-independent agent loop (PLAN.md 5.3).

Reset, summarise, ask, validate, step, repeat. The loop owns everything that
must be identical across providers -- the observation summary, the action
contract, the validation, the retry bound and the record -- and adapters own
nothing but the call. That split is what makes PLAN 5.3's "local and API models
use the same observation and action contracts" a property of the code rather
than a claim about it: an adapter has no way to reach the environment, and the
loop has no provider-specific branch.

A run writes its artifacts next to the training runs, under
``runtime/runs/<run_id>/``, with the same manifest schema
(:mod:`factoriorl.manifest`). An agent run and a policy run are then inspectable
the same way -- ``factoriorl runs list``, ``runs show`` and ``runs verify`` all
work on both -- which matters because Phase 5's demonstration has to be compared
against Phase 4's learning result.

Nothing here imports torch, stable-baselines3 or gymnasium. The environment side
of this repository is kept free of the training stack so that evaluation cannot
be changed by having torch installed, and an agent run is evaluation.
``test_agent_loop.py`` asserts it with the same subprocess import probe the
environment uses.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from factoriorl import assistance as assistance_module
from factoriorl import manifest as manifest_module
from factoriorl.agent.adapters import DEFAULT_MAX_TOKENS, ModelAdapter, ModelReply, ModelRequest
from factoriorl.agent.memory import Memory
from factoriorl.agent.parsing import (
    MAX_ACTIONS_PER_SEQUENCE,
    DecisionFailure,
    ParsedAction,
    ParsedSequence,
    ParseFailure,
    check_against_state,
    parse_sequence,
)
from factoriorl.agent.summary import (
    SUMMARY_ENCODING_VERSION,
    LegalAction,
    ObservationSummary,
    TaskBrief,
    action_vocabulary,
    argument_domains,
    argument_requirements,
    legal_actions,
    static_reference,
    summarise,
    targetable_actions,
    visible_handles,
)
from factoriorl.agent.transcript import Transcript

#: Provenance keys `_write_manifest` places itself. Everything else a caller
#: supplies is carried into `extra` instead of being dropped on the floor.
_PROVENANCE_CONSUMED = frozenset({"engine", "assistance", "seeds", "workers"})

#: How many times one decision may be asked for before the loop stops asking.
#: Declared here rather than buried in the loop because PLAN 5.3 requires the
#: retry bound to exist; three is one honest attempt plus two corrections, and a
#: model that cannot produce a catalog index after being told twice what was
#: wrong with its answer is not going to on the fourth try -- it is a prompt or
#: a model-choice problem, and spending the episode's wall clock on it hides
#: that.
MAX_ATTEMPTS_PER_DECISION = 3

#: How many consecutive decisions may fall back before the run is stopped and
#: recorded as failed. PLAN 5.3 permits "bounded retries **or** a recorded
#: failure"; this is the second half. Without it a broken endpoint produces a
#: 600-step episode of waiting that looks like a played episode in the results.
MAX_CONSECUTIVE_FALLBACKS = 5

#: Roadmap A2.3's ceiling on one batch, in wall-clock seconds. A batch is not
#: an escape from the run's own deadline: whichever of the two expires first
#: stops it, and the actions after that point are recorded as unexecuted rather
#: than dropped. Thirty seconds is generous against a measured decision -- each
#: action is one `step`/`step_arguments` round trip -- and small enough that a
#: worker which has stopped answering is noticed within one decision.
SEQUENCE_WALL_SECONDS = 30.0

#: Action statuses the environment reports for something that did not happen.
#: `running` is deliberately absent: a walk or a mine spans several decisions,
#: so an in-flight action is the normal case and stopping the batch on it would
#: make every movement plan a one-action plan.
_FAILED_STATUSES = frozenset({"failed", "rejected", "cancelled"})

#: Names the deliberation layer in a published manifest, the way
#: ``learn/train.py`` names ``skills-v1``. A result produced by a language model
#: over the primitive catalog is not comparable to one produced by a trained
#: policy, and a run that does not say which it was cannot be interpreted later.
DELIBERATION_PROFILE = "language-model-v1"

#: `$MAX` is substituted from `MAX_ACTIONS_PER_SEQUENCE` rather than typed here
#: as a digit: a prompt that invites nine actions and a parser that refuses them
#: is a rejection the model cannot act on, and the two numbers would drift the
#: first time the cap moved. An f-string cannot do it -- the format examples
#: below are full of braces.
SYSTEM_PROMPT = """\
You are controlling a single character in Factorio through a bounded action \
catalog. Each turn you receive the character's local observation and the exact \
list of actions the environment will currently accept, then choose one.

Rules:
- Choose exactly one action, by its numeric index, from the LEGAL ACTIONS list.
- An index that is not in that list will be rejected and you will be asked again.
- Entity and resource positions are shown as offsets from the character in \
tiles. The x axis grows east and the y axis grows south.
- Argument values under ARGUMENT VALUES are ABSOLUTE world coordinates, not \
offsets. Your own absolute position is the one under CHARACTER. To act on a \
tile you can see at offset (dx, dy), add that offset to your own position.
- One decision advances the world by a fixed interval, so movement, mining and \
crafting take several decisions to finish. Actions still in progress are listed \
under "in flight".
- You cannot see beyond the sensor radius. Entities marked REMEMBERED are what \
was there when you last saw them, not what is there now.

Some actions act on an entity. Those say "the nearest entity" in their \
description, and by default that is what they do. To act on a *particular* one \
instead, add its handle -- the [hN] shown beside it -- as "target".

Some actions need arguments you must supply. Those are marked "[needs: ...]" \
in the LEGAL ACTIONS list, and the legal values for each argument name are \
listed under ARGUMENT VALUES. Supply exactly the names it asks for, no more.

Reply with a single JSON object and nothing else:
{"action": <index>, "reason": "<one short sentence>"}
or, to choose which entity it acts on:
{"action": <index>, "target": "<hN>", "reason": "<one short sentence>"}
or, for an action marked [needs: ...]:
{"action": <index>, "arguments": {"<name>": <value>}, "reason": "<why>"}

When you already know the next few moves, you may send up to $MAX of them to \
run in order, using the same fields for each one:
{"actions": [{"action": <index>}, {"action": <index>, "target": "<hN>"}], \
"reason": "<why this sequence>"}
- They run one at a time, and each is checked again against the world the one \
before it left. The first one that is refused or fails stops the rest, and you \
will be told which ones ran, which one failed and which never started.
- Anything that already happened stays happened; nothing is undone.
- Only the first action is checked against the LEGAL ACTIONS list above. Later \
ones are checked when their turn comes, so a sequence that assumes a move has \
finished may stop early.
- Send one action when you are unsure, and a sequence when the steps do not \
depend on anything you cannot predict.\
""".replace("$MAX", str(MAX_ACTIONS_PER_SEQUENCE))

#: What the prompt above must **not** contain, asserted by a test. R3.2
#: requires that the reference build sequence is not embedded in runtime
#: assistance, and the honest way to keep that true is to name the phrases that
#: would violate it and check for them.
#:
#: The task's own `description` is fair game -- it is the objective, declared in
#: the spec and visible to every client. The *geometry* is not: which furnace
#: centre catches a drill's drop is what the reference solver had to measure on
#: an engine, and handing it over would make the baseline a test of
#: instruction-following instead of a baseline.
FORBIDDEN_IN_PROMPT = (
    "drop_position",
    "drop position",
    "drop tile",
    "y + 2",
    "y+2",
    "two tiles south",
    "(1, 3)",
)


def _prompt_digest() -> str:
    """Pin the prompt into the manifest.

    Two runs of the same task and model with different system prompts are not
    the same experiment, and the prompt is the part of an agent run most likely
    to be edited between them.
    """
    return manifest_module.config_digest(SYSTEM_PROMPT)


def _target_of(action_key: str, observation: dict) -> str | None:
    """Which entity an action acted on, derived from the observation alone.

    The catalog binds `$target` to the *nearest* entity, and the agent never
    names it, so without this a record of "gave 20 plates" cannot say where --
    and "already tried" degrades to "already did something". Resolved the same
    way the catalog resolves it, from the observation the agent was shown.
    """
    if not action_key.startswith(("give", "take", "transfer")):
        return None
    character = (observation.get("character") or {}).get("position") or [0.0, 0.0]
    nearest, best = None, None
    for entity in observation.get("entities") or []:
        point = entity.get("p") or entity.get("offset") or [0.0, 0.0]
        distance = abs(point[0] - character[0]) + abs(point[1] - character[1])
        if best is None or distance < best:
            nearest, best = entity.get("h") or entity.get("handle"), distance
    return nearest


@dataclass
class AgentConfig:
    """Everything about a run that is not the environment or the provider."""

    task_id: str
    episodes: int = 1
    max_attempts: int = MAX_ATTEMPTS_PER_DECISION
    max_consecutive_fallbacks: int = MAX_CONSECUTIVE_FALLBACKS
    max_tokens: int = DEFAULT_MAX_TOKENS
    #: Hard ceiling on decisions per episode, independent of the task's own
    #: budget. Present because a paid provider makes an unbounded episode an
    #: unbounded bill, and the task budget is chosen for RL training lengths.
    max_steps: int | None = None
    #: Record the full rendered summary with every decision. On by default: PLAN
    #: 5.5 requires that selecting an event shows the observation that produced
    #: it, and reconstructing the prompt afterwards from a stored observation is
    #: not the same evidence as the prompt that was actually sent.
    record_summaries: bool = True
    split: str = "val"
    shaping: bool = True
    #: Offer the temporally extended actions alongside the primitive catalog,
    #: as ``--skills`` does for training. The loop needs no change for this --
    #: ``action_vocabulary`` already appends whatever a wrapper exposes -- but
    #: the flag has to exist somewhere, because which action space an agent
    #: played in decides which random floor its result must be read against.
    #: The two are not interchangeable: `deliver`'s floor is 0.00 over
    #: primitives and 0.04 over skills, and before today's hardening it was 0.80.
    skills: bool = False
    #: Keep a record of what has been observed and tried, and render it into the
    #: prompt (PLAN 5.4). Per *episode*, never across them: memory that survived
    #: an episode boundary would carry one evaluation scene's contents into the
    #: next, which is contamination rather than competence.
    memory: bool = True
    #: Let the model name the entity an action acts on (PLAN 5.2's typed
    #: interactions). The catalog binds `$target` to the *nearest* entity
    #: because a discrete index cannot carry an argument, which is why a policy
    #: oscillated between two chests, one skill solved `navigate` 99 times in
    #: 100, and an agent fuelled a drill four times while the furnace two tiles
    #: away stayed empty. Recorded in the manifest: a result produced with
    #: addressing is not comparable to one produced without it.
    addressed_actions: bool = True
    #: When the model chooses to wait, repeat the wait for up to this many
    #: extra decisions without asking again, stopping early if the world
    #: materially changes or the episode ends. 0 disables it.
    #:
    #: This is an **assistance** and is recorded as one. It exists because
    #: `build_line`'s measurement window is 3,600 game ticks and a decision is
    #: 30, so a construction run is roughly 240 decisions of which ~200 are
    #: waiting for a furnace -- and a paid provider asked to re-read a 2,700
    #: character prompt 200 times to say "wait" again is measuring the price of
    #: patience, not the agent. It cannot substitute for a decision the model
    #: did not make: it only ever repeats `wait`, which the model just chose.
    wait_batch: int = 0
    run_prefix: str = "agent"
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "episodes": self.episodes,
            "max_attempts": self.max_attempts,
            "max_consecutive_fallbacks": self.max_consecutive_fallbacks,
            "max_tokens": self.max_tokens,
            "max_steps": self.max_steps,
            "split": self.split,
            "shaping": self.shaping,
            "skills": self.skills,
            "memory": self.memory,
            "addressed_actions": self.addressed_actions,
            "wait_batch": self.wait_batch,
        }


@dataclass
class Attempt:
    """One provider call, successful or not."""

    attempt: int
    reply: ModelReply
    outcome: ParsedAction | ParsedSequence | ParseFailure | None

    def to_dict(self) -> dict:
        body: dict[str, Any] = {"attempt": self.attempt, **self.reply.to_dict()}
        if isinstance(self.outcome, (ParsedAction, ParsedSequence)):
            actions = (
                list(self.outcome.actions)
                if isinstance(self.outcome, ParsedSequence)
                else [self.outcome]
            )
            # `parsed` keeps meaning one action, because that is what every
            # decisions.jsonl written before batching existed means by it.
            body["parsed"] = actions[0].to_dict()
            body["parsed_actions"] = [action.to_dict() for action in actions]
            body["failure"] = None
        elif isinstance(self.outcome, ParseFailure):
            body["parsed"] = None
            body.update(self.outcome.to_dict())
        else:
            # The call itself failed, so there was nothing to parse. Recorded
            # under the same key as a parse failure so one query answers "why
            # did this decision not happen", and distinguished by its value.
            body["parsed"] = None
            body["failure"] = DecisionFailure.PROVIDER_ERROR.value
            body["detail"] = self.reply.error
        return body


@dataclass
class Decision:
    """One decision: the prompt, every attempt at it, and what was executed.

    ``action_index``/``action_key``/``target``/``arguments`` name the **first**
    action of the sequence, and keep doing so now that a reply may carry up to
    eight (roadmap A2.3). They are scalars in every `decisions.jsonl` written so
    far, and `tools/replay.py` reads all four positionally -- it colours a
    timeline cell by `action_key`, filters on `place_at`, and prints
    `action_key #action_index` as "chosen". Redefining them as lists would leave
    every existing run unreadable by the same tool, so the sequence is *added*
    beside them in `requested` and `sequence` rather than folded into them.
    """

    episode: int
    step: int
    summary: ObservationSummary
    attempts: list[Attempt]
    action_index: int
    action_key: str
    resolution: str
    #: Handle the model named, when it named one. None means the catalog's
    #: default binding -- the nearest entity -- was used.
    target: str | None = None
    #: Values the model supplied for a parameterized action's arguments.
    arguments: dict = field(default_factory=dict)
    record_summary: bool = True
    result: dict = field(default_factory=dict)
    #: Every action the reply asked for, first included. Empty for a fallback,
    #: where no action was chosen at all.
    requested: list[ParsedAction] = field(default_factory=list)
    #: One record per requested action, in order, filled in by execution: what
    #: it was, whether it ran, and A2.4's three clocks. Written even for the
    #: actions that never started, because "requested but not executed" is the
    #: half of a partial failure that a count of completions cannot express.
    outcomes: list[dict] = field(default_factory=list)
    #: Why the sequence ended early, or None if every action ran.
    sequence_stopped: str | None = None

    @property
    def inference_ms(self) -> float:
        return sum(a.reply.latency_ms for a in self.attempts)

    @property
    def sequence(self) -> dict:
        """Completed, failed and unexecuted -- and they add up to the request.

        A2.3 asks for exactly this triple. It is derived from `outcomes` rather
        than counted during execution so the three can never drift apart from
        the per-action records they summarise.
        """
        counted = {"completed": 0, "failed": 0, "unexecuted": 0}
        for outcome in self.outcomes:
            status = outcome.get("status")
            if status in counted:
                counted[status] += 1
        return {"requested": len(self.outcomes), **counted, "stopped": self.sequence_stopped}

    def to_dict(self) -> dict:
        body = {
            "episode": self.episode,
            "step": self.step,
            "action_index": self.action_index,
            "action_key": self.action_key,
            # The whole semantic action, not just its verb: a replay of a
            # construction run has to show *where* the agent placed a machine,
            # and a bare catalog index cannot say.
            "arguments": dict(self.arguments),
            # Which entity it acted on, so a replay can tell an addressed action
            # from one that took the catalog's nearest-entity default.
            "target": self.target,
            "resolution": self.resolution,
            "inference_ms": round(self.inference_ms, 3),
            "attempts": [a.to_dict() for a in self.attempts],
            "legal_actions": [a.to_dict() for a in self.summary.actions],
            "result": self.result,
            # The four scalars above are the first of these. A reader that only
            # knows the one-action contract sees exactly what it always did.
            "actions": list(self.outcomes),
            "sequence": self.sequence,
        }
        if self.record_summary:
            body["prompt"] = self.summary.render()
            body["observation"] = self.summary.to_dict()
        return body


class AgentLoop:
    """Plays one task with one adapter and records what happened.

    The environment is duck-typed on purpose -- ``reset``, ``step``,
    ``action_masks``, ``catalog``, ``spec_`` and the wire observation. That is
    exactly the surface ``SkillEnv`` also presents, so a skill-augmented
    environment works here unchanged, and it is the surface an assisted profile
    will present too. It also means the loop is testable offline against a stub,
    which is what lets every test in this package run with no engine.
    """

    def __init__(
        self,
        env: Any,
        adapter: ModelAdapter,
        config: AgentConfig,
        *,
        run_id: str | None = None,
        run_dir: Path | None = None,
        provenance: dict | None = None,
        static_knowledge: str = "",
    ) -> None:
        self.env = env
        self.adapter = adapter
        self.config = config
        self.run_id = run_id or manifest_module.new_run_id(config.run_prefix)
        self.run_dir = run_dir or (manifest_module.runs_dir() / self.run_id)
        # Engine, worker and seed provenance the caller holds and the loop does
        # not: the loop never launches a worker, so claiming to know the engine
        # build would be inventing a provenance record.
        self.provenance = dict(provenance or {})
        self.brief = TaskBrief.from_spec(env.spec_)
        self.decisions: list[Decision] = []
        self.memory = Memory()
        #: Prototype data -- what things cost, what they make, how big they are.
        #: It goes in the transcript's static prefix rather than into each turn
        #: because it never changes, which makes it one cache miss and then a
        #: cache hit for the rest of the run. Measured on `deepseek-flash`, the
        #: whole block is ~12k tokens: about $0.004 the first time and $0.00007
        #: a turn afterwards. A single decision spent looking a recipe up costs
        #: more than that.
        self.static_knowledge = static_knowledge
        self.transcript = self._new_transcript()

    def _new_transcript(self) -> Transcript:
        """A fresh history that keeps the static prefix.

        The prefix survives an episode boundary deliberately: it is prototype
        data, identical in every world, so dropping it would throw away a cache
        entry to re-send bytes that were already correct. The *turns* are what
        must not cross a boundary, and they do not.
        """
        # Two invariant blocks: what the game's recipes cost, and what each verb
        # does. Both are the same on every turn, so both belong in the prefix
        # rather than in a message that is re-sent for the rest of the run.
        blocks = [block for block in (self.static_knowledge, static_reference(self.env)) if block]
        return Transcript(system=SYSTEM_PROMPT, static_prefix="\n\n".join(blocks))

    def _world_signature(self) -> tuple:
        """What must change before the model is asked again during a wait.

        Deliberately coarse and deliberately **observation-only**: entity
        identities and their working flags, plus the inventory. Reading
        production out of evaluator truth here would make the stopping rule a
        truth channel, which is the thing the whole observation boundary
        exists to prevent.
        """
        observation = self.env._observation
        entities = tuple(
            sorted(
                (str(record.get("h")), str(record.get("st")), bool(record.get("working")))
                for record in (observation.get("entities") or [])
                if record.get("h")
            )
        )
        inventory = tuple(sorted((observation.get("inventory") or {}).items()))
        return (entities, inventory)

    def _domains(self) -> dict:
        """The raw argument domains, for validation.

        Separate from the *rendered* domains the prompt shows: the prompt is
        allowed to summarise 121 placement candidates as a rule, and the
        validator is not.
        """
        if not any(getattr(t, "parameterized", False) for t in self.env.catalog.templates):
            return {}
        return self.env.argument_domains()

    def _addressing(self, observation: dict) -> tuple[frozenset[str], frozenset[str]]:
        """What may be addressed this step, and which handles exist."""
        if not self.config.addressed_actions:
            return frozenset(), frozenset()
        return targetable_actions(self.env), visible_handles(observation)

    def _execute(self, decision: Decision):
        """Run the decision's first action. The one-action entry point."""
        return self._dispatch(decision.action_index, decision.target, decision.arguments)

    def _dispatch(self, index: int, target: str | None, arguments: dict):
        """Run one action, addressed if the model named a target.

        Rebinding the catalog's own template is what keeps addressing from
        becoming a new capability: the verb, its item and its count are the
        ones the discrete action already carried, and only `$target` changes.
        A model cannot reach an action its catalog does not contain, and the
        action profile still decides what the engine accepts.

        A `parameterized-v1` action goes through `env.step_arguments`, which is
        the *same* entry point the RL adapter uses -- so R2.2's property that
        both clients issue equivalent semantic actions from matched states is
        not re-implemented here, it is shared. It is also what makes A2.3's
        "validate again before every action" free rather than something the
        batch has to arrange: `step_arguments` re-derives `argument_domains()`
        and `_context()` on the call, so action 2 of a sequence is bound and
        checked against the world action 1 left behind, never against the one
        the model was shown.
        """
        catalog = self.env.catalog
        if index < len(catalog.templates):
            template = catalog.templates[index]
            if getattr(template, "arguments", ()):
                return self.env.step_arguments(index, dict(arguments))
        if not target:
            return self.env.step(index)
        if index >= len(catalog.templates):
            # A skill, not a catalog template: skills resolve their own targets
            # and have no `$target` to rebind.
            return self.env.step(index)
        template = catalog.templates[index]
        context = {**self.env.unwrapped._context(), "target": target}
        return self.env.step_payload(template.bind(context), action_key=template.key)

    def _tick(self) -> int:
        """The observation clock, for A2.4's per-action freshness record."""
        return int((self.env._observation or {}).get("tick") or 0)

    def _revalidate(self, action: ParsedAction) -> ParsedAction | ParseFailure:
        """Check an action against the world it is about to be dispatched into.

        A2.3 requires this before *every* action of a batch, not only the first.
        The mask, the visible handles and the argument domains are all rederived
        from `self.env` here, so an action planned against the observation the
        model was shown is refused if the actions before it moved the world out
        from under it -- which is the only honest way to run a plan the model
        made without seeing the states it would run in.
        """
        observation = self.env._observation
        vocabulary = action_vocabulary(self.env)
        legal = legal_actions(vocabulary, self.env.action_masks())
        targetable, handles = self._addressing(observation)
        return check_against_state(
            action,
            legal,
            targetable=targetable,
            handles=handles,
            requires=argument_requirements(self.env),
            domains=self._domains(),
        )

    def _execute_sequence(
        self,
        decision: Decision,
        *,
        remaining: int,
        until: Callable[[], bool] | None = None,
    ) -> tuple[int, float, bool, bool, dict]:
        """Run the requested actions serially and record what became of each.

        Returns the number of actions the environment actually ran, the reward
        summed over them, the terminal flags, and the last `info` -- which is
        what the caller needs to keep accounting in decisions *and* in
        environment steps, since a batch of six is one decision and six steps.

        Three things stop it, and A2.3 names all three: the first action that is
        refused or fails, `SEQUENCE_WALL_SECONDS`, and the run's own `until`
        deadline. Nothing that already ran is undone -- there is no rollback in
        this environment and inventing one would mean issuing *more* game
        actions to reverse the agent's, which is not the same world state and
        would be recorded as the agent's own doing.
        """
        deadline = time.perf_counter() + SEQUENCE_WALL_SECONDS
        executed, total_reward = 0, 0.0
        terminated = truncated = False
        info: dict = {}
        for position, action in enumerate(decision.requested):
            record: dict[str, Any] = {
                "position": position,
                "index": action.index,
                "key": action.key,
                "target": action.target,
                "arguments": dict(action.arguments),
                "reason": action.reason,
            }
            decision.outcomes.append(record)
            if decision.sequence_stopped is not None:
                record["status"] = "unexecuted"
                continue
            if position >= remaining:
                # The task's own decision budget. Spending past it would let a
                # batch buy the episode steps it does not have.
                decision.sequence_stopped = "step_budget"
            elif position and time.perf_counter() >= deadline:
                decision.sequence_stopped = "sequence_deadline"
            elif position and until is not None and until():
                # Whichever comes first: the run's deadline is checked between
                # the actions of a batch as well as between decisions, so a
                # batch cannot run past the wall clock the run was given.
                decision.sequence_stopped = "run_deadline"
            if decision.sequence_stopped is not None:
                record["status"] = "unexecuted"
                continue
            if position:
                # Action 1 was already checked against this state by the parser.
                checked = self._revalidate(action)
                if isinstance(checked, ParseFailure):
                    decision.sequence_stopped = "refused"
                    record.update(
                        status="failed",
                        observation_tick=self._tick(),
                        failure=checked.failure.value,
                        detail=checked.detail,
                    )
                    continue
                action = checked
                record["arguments"] = dict(action.arguments)
            observation_tick = self._tick()
            at = time.perf_counter()
            _, reward, terminated, truncated, info = self._dispatch(
                action.index, action.target, action.arguments
            )
            executed += 1
            total_reward += float(reward)
            status = info.get("action_status")
            failed = status in _FAILED_STATUSES or bool(info.get("action_error"))
            record.update(
                status="failed" if failed else "completed",
                # A2.4: the tick the action was chosen against, the tick it
                # landed on, and the real time in between. Two of the three are
                # game clocks and the third is not, and a record carrying only
                # one of them cannot tell a slow provider from a slow world.
                observation_tick=observation_tick,
                execution_tick=self._tick(),
                elapsed_ms=round((time.perf_counter() - at) * 1000.0, 3),
                reward=round(float(reward), 4),
                action_status=status,
                action_error=info.get("action_error"),
            )
            if failed:
                decision.sequence_stopped = "action_failed"
            elif terminated or truncated:
                decision.sequence_stopped = "terminated" if terminated else "truncated"
        return executed, total_reward, terminated, truncated, info

    # ------------------------------------------------------------- deciding

    def _compose(self, summary: ObservationSummary) -> str:
        """The single user turn for this decision."""
        user = summary.render()
        if self.config.memory:
            # Appended rather than woven in, so the observation the model is
            # shown stays exactly the observation the environment published and
            # the remembered part is visibly separate from the current one.
            remembered = self.memory.render()
            if remembered:
                user = f"{user}\n\nWHAT YOU HAVE ALREADY SEEN AND TRIED\n{remembered}"
        return user

    def _ask(self) -> ModelReply:
        """Send the transcript exactly as it stands.

        Nothing is composed here. The body is whatever ``self.transcript``
        holds, which is what lets a retry after a *transport* failure re-send a
        byte-identical request: the call never reached the model, so there is
        nothing to append, and an identical body is a complete cache hit rather
        than a fresh charge at the miss rate.

        ``system`` and ``user`` are still filled from the transcript, so an
        adapter that cannot use a history -- and anything reading the request
        for a log -- still sees the current turn.
        """
        messages = self.transcript.record_sent()
        request = ModelRequest(
            system=messages[0]["content"],
            user=messages[-1]["content"],
            max_tokens=self.config.max_tokens,
            messages=tuple(messages),
        )
        return self.adapter.complete(request)

    def decide(
        self,
        summary: ObservationSummary,
        legal: tuple[LegalAction, ...],
        vocabulary: tuple[tuple[str, str], ...],
        *,
        episode: int,
        step: int,
        fallback_index: int,
        targetable: frozenset[str] = frozenset(),
        handles: frozenset[str] = frozenset(),
    ) -> Decision:
        """Ask until the answer validates or the bound is reached."""
        attempts: list[Attempt] = []
        # One user turn per decision, appended once. Retries within the
        # decision extend the history instead of rewriting this turn, so the
        # body sent on attempt N stays a strict prefix of the body sent on N+1.
        limit = max(self.config.max_attempts, 1)
        self.transcript.append_user(self._compose(summary))
        for number in range(1, limit + 1):
            reply = self._ask()
            if not reply.ok:
                # A failed call is recorded with its measured latency and does
                # not consume a *different* budget from a malformed answer: both
                # are ways this decision did not happen, and both are bounded by
                # the same attempt count so a flapping endpoint cannot loop.
                attempts.append(Attempt(number, reply, None))
                # Nothing is appended: the call never reached the model, so
                # there is no answer to record and no correction to make. The
                # next attempt re-sends a byte-identical body, which a
                # prefix-caching provider serves entirely from cache.
                continue
            # Everything recorded from the provider passes through the adapter's
            # redactor, because the adapter is the only object that holds the
            # credential and an endpoint that echoes its own auth header would
            # otherwise put it in the artifact verbatim.
            reply = ModelReply(
                text=self.adapter.redact(reply.text),
                latency_ms=reply.latency_ms,
                usage=reply.usage,
                meta=reply.meta,
            )
            outcome = parse_sequence(
                reply.text,
                legal,
                vocabulary,
                targetable=targetable,
                handles=handles,
                requires=summary.requires,
                domains=self._domains(),
            )
            attempts.append(Attempt(number, reply, outcome))
            if isinstance(outcome, ParsedSequence):
                # The accepted answer joins the history, so the next
                # decision's observation is appended after it and the cached
                # prefix keeps growing.
                self.transcript.append_assistant(reply.text)
                first = outcome.first
                return Decision(
                    episode=episode,
                    step=step,
                    summary=summary,
                    attempts=attempts,
                    # The first action, and it stays the first action: see the
                    # class docstring for what reads these four.
                    action_index=first.index,
                    action_key=first.key,
                    target=first.target,
                    arguments=first.arguments,
                    requested=list(outcome.actions),
                    resolution="model",
                    record_summary=self.config.record_summaries,
                )
            # The model's own answer, then why it was refused -- appended,
            # never spliced into a turn already sent. Telling it what was
            # wrong is what makes a retry different from a repeat; a loop that
            # re-sent an identical prompt once burned its whole retry budget
            # on the same malformed answer three times.
            self.transcript.append_assistant(reply.text)
            if number < limit:
                # No correction after the final attempt: nothing would read
                # it, and ending on the assistant turn keeps the roles
                # alternating for the next decision.
                self.transcript.append_user(
                    f"Your previous answer was rejected: {outcome.failure.value}: {outcome.detail}"
                )

        # The bound was reached. The environment still needs an action, and the
        # catalog guarantees a legal no-op exists in every state, so the loop
        # waits rather than crashing the episode -- but the decision is recorded
        # as a fallback, so it can never be read back as a choice the model made.
        fallback = ParsedAction(index=fallback_index, key=vocabulary[fallback_index][0])
        return Decision(
            episode=episode,
            step=step,
            summary=summary,
            attempts=attempts,
            action_index=fallback.index,
            action_key=fallback.key,
            # A sequence of one, so the executor has a single path. The
            # `resolution` is what says nobody chose it.
            requested=[fallback],
            resolution="fallback_wait",
            record_summary=self.config.record_summaries,
        )

    # -------------------------------------------------------------- running

    def _fallback_index(self, legal: tuple[LegalAction, ...]) -> int:
        """The no-op every catalog guarantees, or the first legal action.

        ``FactorioEnv.action_masks`` always leaves ``wait`` legal, so the first
        branch is the normal one; the second exists so a wrapper with a
        different action space cannot make the fallback path raise.
        """
        wait = self.env.catalog.wait_index
        if any(a.index == wait for a in legal):
            return wait
        return legal[0].index

    def run_episode(self, episode: int, *, until=None, on_decision=None) -> dict:
        """Reset, then play one segment for the whole budget."""
        self.env.reset()
        return self.run_segment(episode, fresh_memory=True, until=until, on_decision=on_decision)

    def run_segment(
        self,
        episode: int,
        *,
        first_step: int = 0,
        budget: int | None = None,
        fresh_memory: bool = False,
        until: Callable[[], bool] | None = None,
        on_decision: Callable[[], None] | None = None,
    ) -> dict:
        """Play decisions on an **already-reset** environment.

        Extracted so a driver that needs to interleave agent control with
        evaluator time -- commission, run a measurement window, inject a
        disruption, hand back for recovery -- can hand back to *this* loop
        instead of re-implementing the stepping.

        `tools/demonstration.py` re-implemented it, and the copy drifted twice:
        it omitted `arguments=`/`requires=` from `summarise`, so a
        `parameterized-v1` catalog would be prompted with no argument values at
        all, and it omitted `target=` from the memory record. Its own docstring
        records that the same re-implementation had already cost one measured
        defect (`unknown_target` x 50). One stepping path is the fix.

        `first_step` continues the decision numbering across a hand-back, so a
        replay reads as one episode rather than several starting at zero.
        `fresh_memory` is False by default because a hand-back is the *same*
        scene: what the agent saw before the intervention is still its own
        history. `run_episode` passes True, since a new episode is not.

        `until` is an **evaluator-side** stopping condition, checked after each
        decision: "the line is producing", "both machines hold fuel again". It
        may read truth, because it only decides when to stop asking -- it is
        never rendered into a prompt and the model is not told it exists. The
        segment records `stopped: "until"` when it fires, so a replay can tell
        a segment that reached its goal from one that ran out of budget.
        """
        started = time.perf_counter()
        total_reward = 0.0
        steps = 0
        consecutive_fallbacks = 0
        stopped = "budget"
        info: dict = {}
        # The same bound the batching must respect, resolved once: an unbounded
        # `wait_batch` could otherwise run past `max_steps` and spend an
        # episode's budget without asking the model anything.
        ceiling = min(
            self.env.spec_.max_decision_steps,
            self.config.max_steps if self.config.max_steps is not None else 1 << 30,
        )
        segment_budget = ceiling if budget is None else min(budget, ceiling)
        # Counted in absolute decisions, so a segment cannot spend more than the
        # task budget just because it started late.
        budget = max(0, min(segment_budget, ceiling - first_step))
        if fresh_memory:
            # A fresh record per episode. Carrying one over would put the
            # previous scene's containers into this scene's prompt.
            self.memory = Memory()
            # And a fresh transcript, for that reason and no other. Starting
            # over throws away a cached prefix, which is the expensive thing
            # in this design -- but a history carried across scenes would put
            # the previous scene's observations into this one's prompt, which
            # is contamination rather than competence. A single continuous
            # episode pays this once.
            self.transcript = self._new_transcript()
        while steps < budget:
            observation = self.env._observation
            if self.config.memory:
                self.memory.observe(first_step + steps, observation)
            mask = self.env.action_masks()
            vocabulary = action_vocabulary(self.env)
            legal = legal_actions(vocabulary, mask)
            summary = summarise(
                observation,
                brief=self.brief,
                actions=legal,
                step=first_step + steps,
                arguments=argument_domains(self.env),
                requires=argument_requirements(self.env),
            )
            targetable, handles = self._addressing(observation)
            decision = self.decide(
                summary,
                legal,
                vocabulary,
                episode=episode,
                step=first_step + steps,
                fallback_index=self._fallback_index(legal),
                targetable=targetable,
                handles=handles,
            )
            executed, reward, terminated, truncated, info = self._execute_sequence(
                decision, remaining=budget - steps, until=until
            )
            # In decisions the model made this is one; in environment steps it
            # is however many actions actually ran. Counting it as one would let
            # a batch of eight spend eight times the task's tick budget while
            # the segment believed it had used a single step.
            steps += executed
            total_reward += float(reward)
            batched = 0
            if (
                self.config.wait_batch
                # Only a bare `wait`. A batch that happens to start with one is
                # a plan the model wrote, and repeating its first action would
                # displace the rest of it.
                and len(decision.requested) == 1
                and decision.action_key == "wait"
                and decision.resolution == "model"
                and not (terminated or truncated)
            ):
                before = self._world_signature()
                for _ in range(self.config.wait_batch):
                    if steps >= budget:
                        break
                    _, extra_reward, terminated, truncated, info = self.env.step(
                        self.env.catalog.wait_index
                    )
                    steps += 1
                    batched += 1
                    total_reward += float(extra_reward)
                    if terminated or truncated:
                        break
                    if self._world_signature() != before:
                        # Something changed that the model should see: a machine
                        # appeared or stopped, or production moved. Stop
                        # repeating and give it back the decision.
                        break
            decision.result = {
                # Summed over the actions the sequence actually ran, so the
                # number still answers "what did this decision earn". For the
                # one-action case, which is still most of them, it is unchanged.
                "reward": round(float(reward), 4),
                # Environment steps this decision spent, which is the length of
                # the sequence minus whatever never ran.
                "steps": executed,
                # Waits the loop repeated without asking again, so a replay can
                # tell one decision from the game time it covered.
                "batched_waits": batched,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                # The environment's own verdict on the action, which is how an
                # accepted-but-rejected action (out of reach, no items) is told
                # apart from one that never reached the engine.
                "action_status": info.get("action_status"),
                "action_error": info.get("action_error"),
                "success": bool(info.get("success", False)),
                "infrastructure_failure": info.get("infrastructure_failure"),
                # An assisted action is one decision to the model and many to
                # the engine. PLAN 5.5 requires the replay to expand it, so the
                # primitives it actually issued travel with the decision rather
                # than being summarised into a count.
                "skill": info.get("skill"),
                "skill_outcome": info.get("skill_outcome"),
                "skill_steps": info.get("skill_steps"),
                "skill_trace": info.get("skill_trace"),
            }
            if self.config.memory:
                # Status and error only. `reward` and `success` sit above and
                # are deliberately not passed: memory is rendered back into the
                # prompt, so anything it reads, the model sees.
                #
                # One record per action that reached the environment, not one
                # per decision: a batch of six that stopped at the fourth is six
                # things the model asked for and four things that happened, and
                # a memory that says only "place_at" would tell it it had built
                # a line it did not build.
                ran = 0
                for outcome in decision.outcomes:
                    if outcome.get("status") not in ("completed", "failed"):
                        continue
                    self.memory.record_action(
                        first_step + steps - executed + ran,
                        outcome["key"],
                        # The addressee the model named, or -- for the catalog's
                        # nearest-entity default -- the one the observation this
                        # sequence was planned against resolves to.
                        target=outcome.get("target") or _target_of(outcome["key"], observation),
                        status=outcome.get("action_status"),
                        error=outcome.get("action_error") or outcome.get("failure"),
                    )
                    ran += 1
                self.memory.compact()
            self.decisions.append(decision)
            self._append_decision(decision)

            if not executed:
                # `steps` did not move, so the `while` would spin. Unreachable
                # while a decision carries at least one action -- the first is
                # exempt from every early stop -- and a spin is not a failure
                # mode worth discovering on a paid provider at decision 400.
                stopped = "model_failure"
                break
            if decision.resolution == "model":
                consecutive_fallbacks = 0
            else:
                consecutive_fallbacks += 1
                if consecutive_fallbacks >= self.config.max_consecutive_fallbacks:
                    stopped = "model_failure"
                    break
            if terminated or truncated:
                stopped = "terminated" if terminated else "truncated"
                break
            if on_decision is not None:
                # Evaluator-side work between decisions -- periodic checkpointing
                # is the reason it exists. Deliberately *not* folded into
                # `until`: that is a predicate, and a predicate with side effects
                # is a trap for whoever reads it next.
                on_decision()
            if until is not None and until():
                stopped = "until"
                break

        if stopped == "budget" and steps >= budget:
            # Distinguish "the run's own ceiling" from "the task's budget": a
            # driver that asked for 15 decisions and got 15 did not run out of
            # task, and reporting it as a truncation would misread the segment.
            stopped = (
                "step_ceiling"
                if self.config.max_steps is not None and first_step + steps >= self.config.max_steps
                else "segment_budget"
            )

        return {
            "episode": episode,
            "first_step": first_step,
            "steps": steps,
            "stopped": stopped,
            "total_reward": round(total_reward, 4),
            "success": bool(info.get("success", False)),
            # An infrastructure failure is not a task outcome (PLAN section 2),
            # so it is carried through to the result rather than counted as a
            # loss the model earned.
            "excluded_from_metrics": bool(info.get("excluded_from_metrics", False)),
            # Game time, not wall time: an agent run and a policy run are only
            # comparable on the clock the environment advances, and a loop that
            # stopped on its own step ceiling never sees the environment's
            # terminal `info` -- so this is read from the observation.
            "final_tick": int((self.env._observation or {}).get("tick") or 0),
            # The §12 metrics for this episode: time to first sustained output,
            # cumulative production and final rate, per simulated tick. R3.2
            # asks for throughput, and a decision count is not throughput.
            "production": self._production(),
            "wall_seconds": round(time.perf_counter() - started, 2),
        }

    def _production(self) -> dict | None:
        """The environment's own production metrics, if it keeps them."""
        metrics = getattr(self.env, "metrics", None)
        if metrics is None:
            return None
        try:
            return metrics.report()
        except Exception:  # noqa: BLE001 - a missing metric is not a failed run
            return None

    def run(
        self,
        *,
        until: Callable[[], bool] | None = None,
        on_decision: Callable[[], None] | None = None,
    ) -> dict:
        """Play ``config.episodes`` episodes and write the run artifact."""
        started = time.perf_counter()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(self.run_dir / "status.json", {"run_id": self.run_id, "state": "running"})
        self._write_manifest()
        status = {"run_id": self.run_id, "state": "running"}
        episodes: list[dict] = []
        try:
            for index in range(self.config.episodes):
                episodes.append(self.run_episode(index, until=until, on_decision=on_decision))
                if until is not None and until():
                    # A run-level stop -- a wall clock or a spend cap -- ends
                    # the run, not merely the episode it fired in.
                    break
            status = {"run_id": self.run_id, "state": "completed"}
        except Exception as exc:  # noqa: BLE001 - a failed run must stay inspectable
            import traceback

            status = {
                "run_id": self.run_id,
                "state": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            raise
        finally:
            result = {
                "run_id": self.run_id,
                # Where the replay is. The loop streams decisions to
                # `decisions.jsonl` and returns only aggregates, so a caller
                # that wants the decisions has to be told where they went.
                "run_dir": str(self.run_dir),
                "task": self.config.task_id,
                "episodes": episodes,
                "wall_seconds": round(time.perf_counter() - started, 2),
                **self.aggregate(),
            }
            self._write_json(self.run_dir / "result.json", result)
            self._write_json(self.run_dir / "status.json", status)
        return result

    # ------------------------------------------------------------ recording

    def aggregate(self) -> dict:
        """Latency, usage and failure counts across every decision made.

        Latency is summed over *calls*, not decisions: a decision that took
        three attempts cost three calls, and reporting only the successful one
        would understate what a retry-prone model costs.
        """
        latencies = [a.reply.latency_ms for d in self.decisions for a in d.attempts]
        failures: dict[str, int] = {}
        usage: dict[str, float] = {}
        for decision in self.decisions:
            for attempt in decision.attempts:
                body = attempt.to_dict()
                if body.get("failure"):
                    failures[body["failure"]] = failures.get(body["failure"], 0) + 1
                for key, value in (attempt.reply.usage or {}).items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        usage[key] = usage.get(key, 0) + value
        ordered = sorted(latencies)

        def percentile(fraction: float) -> float:
            if not ordered:
                return 0.0
            index = min(len(ordered) - 1, int(fraction * (len(ordered) - 1)))
            return round(ordered[index], 2)

        return {
            "decisions": len(self.decisions),
            "model_calls": len(latencies),
            "fallback_decisions": sum(1 for d in self.decisions if d.resolution != "model"),
            "failures": failures,
            "usage_totals": usage,
            "latency_ms": {
                "total": round(sum(latencies), 2),
                "mean": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
                "p50": percentile(0.5),
                "p95": percentile(0.95),
                "max": round(max(latencies), 2) if latencies else 0.0,
            },
        }

    def _write_json(self, path: Path, payload: Any) -> None:
        """Serialise, redact, then write.

        The redaction pass is the last gate before anything reaches disk: it
        runs over the *rendered* JSON, so a credential that arrived by any route
        -- an error body, a model that echoed its prompt, a future field nobody
        thought about -- is removed regardless of which key it landed under.
        """
        text = json.dumps(payload, indent=2, default=str)
        path.write_text(self.adapter.redact(text), encoding="utf-8")

    def _append_decision(self, decision: Decision) -> None:
        """One line per decision, appended as it happens.

        JSONL and append-as-you-go rather than one document at the end, because
        a run interrupted at decision 400 must leave 400 inspectable decisions;
        the training runs keep their curve the same way.
        """
        line = json.dumps(decision.to_dict(), default=str)
        with (self.run_dir / "decisions.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(self.adapter.redact(line) + "\n")

    def _write_manifest(self) -> None:
        """Written before the first decision, so an interrupted run still has one."""
        catalog = self.env.catalog
        spec = self.env.spec_
        payload = manifest_module.RunManifest(
            run_id=self.run_id,
            engine=self.provenance.get("engine") or {},
            task={
                "id": spec.id,
                "version": spec.version,
                "config_digest": manifest_module.config_digest(spec.to_dict()),
                "resolved": spec.to_dict(),
            },
            profiles={
                "observation": spec.observation_profile,
                "action": spec.action_profile,
                # `assisted-v1` is being implemented separately; until it exists
                # an agent run is honest about running the primitive profile
                # rather than claiming assistance it does not have.
                # The LLM client sees *every* published marker and selects for
                # itself, so it gets no goal-focus assistance even on a task
                # whose RL encoding does. Recorded distinctly, because both
                # clients previously wrote the same "none".
                "assistance": self.provenance.get("assistance", assistance_module.STATIC),
                "goal_encoding": None,
                "summary_encoding": SUMMARY_ENCODING_VERSION,
                "deliberation": DELIBERATION_PROFILE,
                "catalog": catalog.name,
                "catalog_digest": catalog.digest(),
                "resolved_catalog": list(catalog.keys()),
            },
            reward={
                "shaping_enabled": self.config.shaping,
                "config_digest": manifest_module.config_digest([r.name for r in spec.rewards]),
                "components": [
                    {"name": r.name, "kind": r.kind.value, "weight": r.weight, "shaping": r.shaping}
                    for r in spec.rewards
                ],
            },
            seeds=self.provenance.get("seeds") or {},
            budgets={
                "max_decision_steps": spec.max_decision_steps,
                "max_game_ticks": spec.max_game_ticks,
                "max_steps": self.config.max_steps,
                "max_attempts_per_decision": self.config.max_attempts,
            },
            workers=self.provenance.get("workers") or [],
            # `describe()` is contractually credential-free, and the write gate
            # redacts anyway.
            model={
                **self.adapter.describe(),
                "system_prompt_digest": _prompt_digest(),
                "max_tokens": self.config.max_tokens,
            },
            # Provenance this method does not itself consume, carried through
            # rather than dropped. It read exactly four keys -- engine,
            # assistance, seeds, workers -- so `run_world` could assemble a
            # `world` block, a parsed `starting_inventory` and a `clock` record
            # and have all three silently discarded. Which map, which starting
            # items, and whether the world ran while the model thought are
            # precisely what a later reader needs.
            extra={
                "config": self.config.to_dict(),
                **{
                    key: value
                    for key, value in self.provenance.items()
                    if key not in _PROVENANCE_CONSUMED
                },
                **self.config.extra,
            },
        ).to_dict()
        self._write_json(self.run_dir / "manifest.json", payload)
