"""Persistent agent memory (PLAN.md 5.4).

    Maintain explored terrain, last-seen entities, action outcomes, and the
    current plan separately from authoritative world state.

The requirement is not abstract. `gpt-5.6-luna` played `deliver` before the
objective was published and spent whole episodes on this:

    take | give | take | give | take | give | ...

Seventeen takes alternating with sixteen gives on the *same* container. That is
the correct strategy given what it could see -- deliver, read the reward, take
the plates back, try another -- and it cannot terminate, because nothing
recorded that a container had already been ruled out. Standing still leaves the
entity ranks unchanged, so the next decision faces an identical observation and
makes an identical choice.

What may enter memory, and what may not
---------------------------------------
Memory is written only from things the agent itself received: the observation it
was shown, and the *status* of the action it took. It is never written from
`reward`, `success`, `terminated` or anything else the evaluator computes from
ground truth.

That restriction is the whole of PLAN 5.4's last clause. Memory is rendered back
into the prompt, so a memory that ingested the evaluator's verdict would feed it
straight to the model, and "context compaction does not introduce evaluator
information" would be false by construction rather than by accident.
:func:`Fact.cite` makes every entry name the decision and the source it came
from, so an entry with no provenance cannot be written at all.

Staleness
---------
An entity the sensor can no longer see is not gone; it is unobserved. The two
are different claims and the model must be able to tell them apart, so a fact
carries the step it was last confirmed at and renders as stale once the current
observation stops confirming it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Sources a fact may cite. A fact built from anything else is a bug: those are
#: the only two channels the agent itself receives.
SOURCE_OBSERVATION = "observation"
SOURCE_ACTION = "action_result"
SOURCES = frozenset({SOURCE_OBSERVATION, SOURCE_ACTION})

#: Fields the evaluator computes and the agent never sees. Reading one into
#: memory would put it in the next prompt.
EVALUATOR_FIELDS = frozenset(
    {"reward", "success", "terminated", "truncated", "infrastructure_failure"}
)

#: How many attempt records to keep per target before the oldest are folded into
#: a count. Failures are never dropped entirely -- see `Memory.compact`.
ATTEMPTS_KEPT = 4


@dataclass
class Fact:
    """One remembered thing, and where it came from."""

    subject: str
    statement: str
    source: str
    step: int
    #: Last decision at which an observation still confirmed this. None for
    #: facts about events, which happened once and do not need reconfirming.
    confirmed_step: int | None = None

    def __post_init__(self) -> None:
        if self.source not in SOURCES:
            raise ValueError(f"a fact must cite {sorted(SOURCES)}, not {self.source!r}")

    def stale_at(self, step: int) -> bool:
        return self.confirmed_step is not None and self.confirmed_step < step

    def render(self, step: int) -> str:
        if self.stale_at(step):
            age = step - (self.confirmed_step or self.step)
            return f"{self.statement} (last seen {age} decisions ago, not confirmed since)"
        return self.statement


@dataclass
class Attempt:
    """Something the agent tried at a target, and what the environment said."""

    step: int
    action: str
    target: str | None
    status: str | None
    error: str | None

    @property
    def failed(self) -> bool:
        return bool(self.error) or (self.status not in (None, "completed"))


@dataclass
class Memory:
    """The agent's own record, kept apart from authoritative world state."""

    entities: dict[str, Fact] = field(default_factory=dict)
    attempts: list[Attempt] = field(default_factory=list)
    #: Plans the agent stated and how they ended. A failed plan stays here: PLAN
    #: 5.4 requires it, and it is the only thing that stops the agent proposing
    #: it again.
    plans: list[dict] = field(default_factory=list)
    step: int = 0

    # ------------------------------------------------------------- writing

    def observe(self, step: int, observation: dict) -> None:
        """Record what the observation shows. Nothing else is read from it."""
        self.step = step
        for entity in observation.get("entities") or []:
            handle = entity.get("h") or entity.get("handle")
            if not handle:
                continue
            position = entity.get("p") or entity.get("offset") or [0.0, 0.0]
            contents = entity.get("contents") or {}
            described = f"{entity.get('name')} [{handle}] at ({position[0]:.1f}, {position[1]:.1f})"
            if contents:
                described += " holding " + ", ".join(
                    f"{k} x{v}" for k, v in sorted(contents.items())
                )
            existing = self.entities.get(handle)
            self.entities[handle] = Fact(
                subject=handle,
                statement=described,
                source=SOURCE_OBSERVATION,
                step=existing.step if existing else step,
                confirmed_step=step,
            )

    def record_action(
        self,
        step: int,
        action: str,
        *,
        target: str | None = None,
        status: str | None = None,
        error: str | None = None,
    ) -> Attempt:
        """Record an action's *outcome*, never the evaluator's verdict on it."""
        attempt = Attempt(step=step, action=action, target=target, status=status, error=error)
        self.attempts.append(attempt)
        return attempt

    def record_plan(self, step: int, plan: str) -> None:
        self.plans.append({"step": step, "plan": plan, "outcome": "open"})

    def close_plan(self, step: int, outcome: str) -> None:
        for entry in reversed(self.plans):
            if entry["outcome"] == "open":
                entry["outcome"] = outcome
                entry["closed_step"] = step
                return

    # ------------------------------------------------------------- reading

    def ruled_out(self) -> list[str]:
        """Targets an action has already been tried on without effect.

        This is the memory that would have ended the take/give loop: a container
        the agent has already delivered to, and which the environment accepted
        without the episode ending, is not worth delivering to again.
        """
        seen: dict[str, int] = {}
        for attempt in self.attempts:
            if attempt.target and attempt.action.startswith(("give", "transfer")):
                seen[attempt.target] = seen.get(attempt.target, 0) + 1
        return sorted(target for target, count in seen.items() if count >= 1)

    def failures(self) -> list[Attempt]:
        return [a for a in self.attempts if a.failed]

    def compact(self, keep: int = ATTEMPTS_KEPT) -> None:
        """Bound the record without losing the parts that carry information.

        Successful repeats are the compressible half: ten completed walks say
        what one says. Failures are not -- a failed plan that disappears is a
        plan the agent will propose again -- so PLAN 5.4 keeps them and this
        drops only from the successful tail.
        """
        failed = [a for a in self.attempts if a.failed]
        succeeded = [a for a in self.attempts if not a.failed]
        self.attempts = sorted(failed + succeeded[-keep:], key=lambda a: a.step)

    def render(self) -> str:
        """The block added to the prompt. Contains no evaluator information."""
        lines: list[str] = []

        stale = [f for f in self.entities.values() if f.stale_at(self.step)]
        if stale:
            lines.append("REMEMBERED (not in the current observation)")
            for fact in sorted(stale, key=lambda f: f.subject):
                lines.append(f"  {fact.render(self.step)}")

        ruled = self.ruled_out()
        if ruled:
            lines.append("ALREADY TRIED")
            for target in ruled:
                attempts = [a for a in self.attempts if a.target == target]
                actions = ", ".join(sorted({a.action for a in attempts}))
                lines.append(f"  {target}: {actions} ({len(attempts)}x)")

        failures = self.failures()
        if failures:
            lines.append("REFUSED ACTIONS")
            for attempt in failures[-ATTEMPTS_KEPT:]:
                where = f" at {attempt.target}" if attempt.target else ""
                lines.append(
                    f"  decision {attempt.step}: {attempt.action}{where}"
                    f" -> {attempt.error or attempt.status}"
                )

        open_plans = [p for p in self.plans if p["outcome"] == "open"]
        closed = [p for p in self.plans if p["outcome"] != "open"]
        if open_plans:
            lines.append("CURRENT PLAN")
            lines.append(f"  {open_plans[-1]['plan']}")
        if closed:
            lines.append("PLANS THAT DID NOT WORK")
            for entry in closed[-ATTEMPTS_KEPT:]:
                lines.append(f"  {entry['plan']} -> {entry['outcome']}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "entities": {
                handle: {
                    "statement": fact.statement,
                    "source": fact.source,
                    "step": fact.step,
                    "confirmed_step": fact.confirmed_step,
                }
                for handle, fact in self.entities.items()
            },
            "attempts": [
                {
                    "step": a.step,
                    "action": a.action,
                    "target": a.target,
                    "status": a.status,
                    "error": a.error,
                }
                for a in self.attempts
            ],
            "plans": list(self.plans),
        }
