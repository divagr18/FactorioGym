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
from factoriorl.agent.parsing import DecisionFailure, ParsedAction, ParseFailure, parse_action
from factoriorl.agent.summary import (
    SUMMARY_ENCODING_VERSION,
    LegalAction,
    ObservationSummary,
    TaskBrief,
    action_vocabulary,
    argument_domains,
    argument_requirements,
    legal_actions,
    summarise,
    targetable_actions,
    visible_handles,
)

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

#: Names the deliberation layer in a published manifest, the way
#: ``learn/train.py`` names ``skills-v1``. A result produced by a language model
#: over the primitive catalog is not comparable to one produced by a trained
#: policy, and a run that does not say which it was cannot be interpreted later.
DELIBERATION_PROFILE = "language-model-v1"

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
{"action": <index>, "arguments": {"<name>": <value>}, "reason": "<why>"}\
"""

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
    outcome: ParsedAction | ParseFailure | None

    def to_dict(self) -> dict:
        body: dict[str, Any] = {"attempt": self.attempt, **self.reply.to_dict()}
        if isinstance(self.outcome, ParsedAction):
            body["parsed"] = self.outcome.to_dict()
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
    """One decision: the prompt, every attempt at it, and what was executed."""

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

    @property
    def inference_ms(self) -> float:
        return sum(a.reply.latency_ms for a in self.attempts)

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
        """Run the decision, addressed if the model named a target.

        Rebinding the catalog's own template is what keeps addressing from
        becoming a new capability: the verb, its item and its count are the
        ones the discrete action already carried, and only `$target` changes.
        A model cannot reach an action its catalog does not contain, and the
        action profile still decides what the engine accepts.

        A `parameterized-v1` action goes through `env.step_arguments`, which is
        the *same* entry point the RL adapter uses -- so R2.2's property that
        both clients issue equivalent semantic actions from matched states is
        not re-implemented here, it is shared.
        """
        catalog = self.env.catalog
        if decision.action_index < len(catalog.templates):
            template = catalog.templates[decision.action_index]
            if getattr(template, "arguments", ()):
                return self.env.step_arguments(decision.action_index, dict(decision.arguments))
        if not decision.target:
            return self.env.step(decision.action_index)
        if decision.action_index >= len(catalog.templates):
            # A skill, not a catalog template: skills resolve their own targets
            # and have no `$target` to rebind.
            return self.env.step(decision.action_index)
        template = catalog.templates[decision.action_index]
        context = {**self.env.unwrapped._context(), "target": decision.target}
        return self.env.step_payload(template.bind(context), action_key=template.key)

    # ------------------------------------------------------------- deciding

    def _ask(self, summary: ObservationSummary, correction: str) -> ModelReply:
        user = summary.render()
        if self.config.memory:
            # Appended rather than woven in, so the observation the model is
            # shown stays exactly the observation the environment published and
            # the remembered part is visibly separate from the current one.
            remembered = self.memory.render()
            if remembered:
                user = f"{user}\n\nWHAT YOU HAVE ALREADY SEEN AND TRIED\n{remembered}"
        if correction:
            # Telling the model what was wrong is what makes a retry different
            # from a repeat. A loop that re-sent the identical prompt burned its
            # whole retry budget on the same malformed answer three times.
            user = f"{user}\n\nYour previous answer was rejected: {correction}\nAnswer again."
        request = ModelRequest(system=SYSTEM_PROMPT, user=user, max_tokens=self.config.max_tokens)
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
        correction = ""
        for number in range(1, max(self.config.max_attempts, 1) + 1):
            reply = self._ask(summary, correction)
            if not reply.ok:
                # A failed call is recorded with its measured latency and does
                # not consume a *different* budget from a malformed answer: both
                # are ways this decision did not happen, and both are bounded by
                # the same attempt count so a flapping endpoint cannot loop.
                attempts.append(Attempt(number, reply, None))
                correction = f"the previous request failed ({reply.error_kind})"
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
            outcome = parse_action(
                reply.text,
                legal,
                vocabulary,
                targetable=targetable,
                handles=handles,
                requires=summary.requires,
                domains=self._domains(),
            )
            attempts.append(Attempt(number, reply, outcome))
            if isinstance(outcome, ParsedAction):
                return Decision(
                    episode=episode,
                    step=step,
                    summary=summary,
                    attempts=attempts,
                    action_index=outcome.index,
                    action_key=outcome.key,
                    target=outcome.target,
                    arguments=outcome.arguments,
                    resolution="model",
                    record_summary=self.config.record_summaries,
                )
            correction = f"{outcome.failure.value}: {outcome.detail}"

        # The bound was reached. The environment still needs an action, and the
        # catalog guarantees a legal no-op exists in every state, so the loop
        # waits rather than crashing the episode -- but the decision is recorded
        # as a fallback, so it can never be read back as a choice the model made.
        return Decision(
            episode=episode,
            step=step,
            summary=summary,
            attempts=attempts,
            action_index=fallback_index,
            action_key=vocabulary[fallback_index][0],
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

    def run_episode(self, episode: int) -> dict:
        """Reset, then play one segment for the whole budget."""
        self.env.reset()
        return self.run_segment(episode, fresh_memory=True)

    def run_segment(
        self,
        episode: int,
        *,
        first_step: int = 0,
        budget: int | None = None,
        fresh_memory: bool = False,
        until: Callable[[], bool] | None = None,
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
            _, reward, terminated, truncated, info = self._execute(decision)
            steps += 1
            total_reward += float(reward)
            batched = 0
            if (
                self.config.wait_batch
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
                "reward": round(float(reward), 4),
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
                # Status and error only. `reward` and `success` sit two lines
                # above and are deliberately not passed: memory is rendered back
                # into the prompt, so anything it reads, the model sees.
                self.memory.record_action(
                    first_step + steps - 1,
                    decision.action_key,
                    target=_target_of(decision.action_key, observation),
                    status=info.get("action_status"),
                    error=info.get("action_error"),
                )
                self.memory.compact()
            self.decisions.append(decision)
            self._append_decision(decision)

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

    def run(self) -> dict:
        """Play ``config.episodes`` episodes and write the run artifact."""
        started = time.perf_counter()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(self.run_dir / "status.json", {"run_id": self.run_id, "state": "running"})
        self._write_manifest()
        status = {"run_id": self.run_id, "state": "running"}
        episodes: list[dict] = []
        try:
            for index in range(self.config.episodes):
                episodes.append(self.run_episode(index))
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
            extra={"config": self.config.to_dict(), **self.config.extra},
        ).to_dict()
        self._write_json(self.run_dir / "manifest.json", payload)
