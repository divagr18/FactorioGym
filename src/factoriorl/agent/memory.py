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

#: Sources a fact may cite. A fact built from anything else is a bug.
#:
#: The first two are channels the environment gave the agent, and they are
#: authoritative in the narrow sense that the engine said them. `SOURCE_MODEL`
#: is not: it is something the model asserted and asked to keep (roadmap
#: A3.2). It is admitted here so that a claim can be *stored with its
#: provenance* rather than either discarded or, far worse, mixed in with what
#: was observed -- and `Memory.render` puts it under its own heading so the
#: distinction survives into the prompt, which is the only place it matters.
SOURCE_OBSERVATION = "observation"
SOURCE_ACTION = "action_result"
SOURCE_MODEL = "model"
SOURCES = frozenset({SOURCE_OBSERVATION, SOURCE_ACTION, SOURCE_MODEL})

#: Fields the evaluator computes and the agent never sees. Reading one into
#: memory would put it in the next prompt.
EVALUATOR_FIELDS = frozenset(
    {"reward", "success", "terminated", "truncated", "infrastructure_failure"}
)

#: How many attempt records to keep per target before the oldest are folded into
#: a count. Failures are never dropped entirely -- see `Memory.compact`.
ATTEMPTS_KEPT = 4

#: How many model-written notes to carry. Bounded because the model controls
#: this text completely: without a cap, an agent that writes a note every turn
#: grows its own prompt without limit, and in a re-sent history that is paid for
#: on every later turn.
NOTES_KEPT = 8

#: Statuses that mean "started and still going". The environment reports these
#: for every ongoing action, and they are outcomes rather than failures.
_ONGOING = frozenset({"running", "started", "accepted"})

#: How many times the *same* action, at the same target, with the same
#: arguments, may fail before the agent is told plainly that it is stuck
#: (roadmap A3.3). Three, because two is a coincidence and a fourth identical
#: refusal is a decision spent learning nothing.
STALL_THRESHOLD = 3

#: Longest refusal reason rendered, in characters. A domain refusal quotes its
#: legal values -- "not one of the 121 legal values for placements (e.g. ...)"
#: runs past 300 characters -- and the same sentence appearing under both
#: REPEATED FAILURE and REFUSED ACTIONS, on every turn, in a history that is
#: re-sent in full, is the exact cost A2 measured and removed elsewhere.
REASON_LIMIT = 120

#: Longest note kept, in characters. Truncated rather than refused -- a refused
#: note is a silent loss the agent cannot see, and a visibly clipped one is not.
NOTE_LIMIT = 240


def _clip(text: Any) -> str:
    """One line, bounded. A refusal reason is diagnostic, not a document."""
    single = " ".join(str(text or "").split())
    return single if len(single) <= REASON_LIMIT else single[: REASON_LIMIT - 1] + "…"


def _signature(action: str, target: str | None, arguments: dict) -> str:
    """The identity used to decide whether two failures are the same failure."""
    import json as _json

    return _json.dumps([action, target, sorted(arguments.items())], sort_keys=True, default=str)


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
    #: What the model supplied. Part of the identity of an attempt: `place_at`
    #: at one position and `place_at` at another are not the same thing tried
    #: twice, and counting them as one would either cry stall at an agent
    #: making progress or stay silent at one that is not.
    arguments: dict = field(default_factory=dict)

    @property
    def failed(self) -> bool:
        """Did this attempt *not happen*?

        `running` is deliberately not a failure. A walk, a mine and a craft all
        answer `running` and finish over the following decisions -- that is the
        normal shape of an ongoing action, and counting it as refused work
        filled the agent's own record with imaginary failures and pushed real
        ones out of the bounded list.
        """
        if self.status in _ONGOING:
            return False
        return bool(self.error) or (self.status not in (None, "completed"))

    @property
    def signature(self) -> str:
        """Identity for the repeated-failure count: verb, addressee, arguments."""
        return _signature(self.action, self.target, self.arguments)


@dataclass
class Memory:
    """The agent's own record, kept apart from authoritative world state."""

    entities: dict[str, Fact] = field(default_factory=dict)
    attempts: list[Attempt] = field(default_factory=list)
    #: Plans the agent stated and how they ended. A failed plan stays here: PLAN
    #: 5.4 requires it, and it is the only thing that stops the agent proposing
    #: it again.
    plans: list[dict] = field(default_factory=list)
    #: What the model asked to remember, in its own words. Never merged with
    #: `entities`, which is observation-sourced -- see `SOURCE_MODEL`.
    notes: list[Fact] = field(default_factory=list)
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
        arguments: dict | None = None,
    ) -> Attempt:
        """Record an action's *outcome*, never the evaluator's verdict on it."""
        attempt = Attempt(
            step=step,
            action=action,
            target=target,
            status=status,
            error=error,
            arguments=dict(arguments or {}),
        )
        self.attempts.append(attempt)
        return attempt

    def record_note(self, step: int, note: str) -> None:
        """Keep something the model claims, labelled as a claim.

        Consecutive repeats are dropped. A model that restates the same note
        every turn is not adding information, and eight copies of one sentence
        would evict seven real ones.
        """
        text = " ".join(str(note).split())[:NOTE_LIMIT]
        if not text:
            return
        if self.notes and self.notes[-1].statement == text:
            return
        self.notes.append(
            Fact(subject=f"note-{step}", statement=text, source=SOURCE_MODEL, step=step)
        )
        del self.notes[:-NOTES_KEPT]

    def record_plan(self, step: int, plan: str) -> None:
        """State a new plan, which supersedes whatever was open.

        Only one plan is open at a time. A model that states a second without
        closing the first has changed its mind, and recording both as current
        would put two contradictory intentions in the next prompt.
        """
        text = " ".join(str(plan).split())
        if not text:
            return
        open_plans = [entry for entry in self.plans if entry["outcome"] == "open"]
        if open_plans and open_plans[-1]["plan"] == text:
            # Restating the same plan is continuity, not a new plan.
            return
        self.close_plan(step, "superseded")
        self.plans.append({"step": step, "plan": text, "outcome": "open"})

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

    def repeated_failures(self, threshold: int = STALL_THRESHOLD) -> list[dict]:
        """Identical failures that have now happened `threshold` times or more.

        A3.3 wants the agent *told*, clearly, and then asked for a different
        plan. It explicitly does not want the loop quietly choosing something
        that works instead: that would make the run a measurement of this code
        rather than of the model, and the trace would not show it had happened.
        """
        # Counted *since the last success* for each signature. A three-strike
        # rule that never resets turns a temporary condition into a permanent
        # prohibition: an agent that failed to craft a furnace three times for
        # want of stone, then found stone and crafted one, was still being told
        # "sending it again will fail again".
        counted: dict[str, dict] = {}
        for attempt in self.attempts:
            if not attempt.failed:
                # A success clears the count for that exact signature, and only
                # that one.
                counted.pop(attempt.signature, None)
                continue
            entry = counted.setdefault(
                attempt.signature,
                {
                    "action": attempt.action,
                    "target": attempt.target,
                    "arguments": dict(attempt.arguments),
                    "count": 0,
                    "error": None,
                    "last_step": attempt.step,
                },
            )
            entry["count"] += 1
            entry["error"] = attempt.error or attempt.status
            entry["last_step"] = attempt.step
        return [e for e in counted.values() if e["count"] >= threshold]

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
        # The open plan and the notes are already bounded and are outstanding
        # work by definition, so compaction does not touch them. A3.2 requires
        # exactly that: "compact bounded older history without dropping
        # outstanding work or repeatedly encountered failures". Both halves of
        # that sentence are load-bearing and both are honoured above.
        del self.notes[:-NOTES_KEPT]

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

        if self.notes:
            # Deliberately its own heading, and deliberately worded. These are
            # the model's assertions; REMEMBERED above is what the sensor saw.
            # A3.2 requires the two be kept separate, and a separation that is
            # not visible in the prompt is not a separation.
            lines.append("THE AGENT'S OWN NOTES (unverified -- you wrote these)")
            for note in self.notes:
                lines.append(f"  decision {note.step}: {note.statement}")

        stuck = self.repeated_failures()
        stuck_signatures = set()
        if stuck:
            lines.append("REPEATED FAILURE -- this is not working")
            for entry in sorted(stuck, key=lambda e: -e["count"]):
                stuck_signatures.add(
                    _signature(entry["action"], entry["target"], entry["arguments"])
                )
                where = f" at {entry['target']}" if entry["target"] else ""
                if entry["arguments"]:
                    shown = ", ".join(f"{k}={v}" for k, v in sorted(entry["arguments"].items()))
                    where += f" with {shown}"
                lines.append(
                    f"  {entry['action']}{where} has failed {entry['count']} times: "
                    f"{_clip(entry['error'])}"
                )
            lines.append("  Nothing has changed since, so it will fail the same way.")
            lines.append("  Do something different, or change what it depends on first,")
            lines.append('  and say what in "plan".')

        # Anything already named above is not repeated here. The block above
        # says it more usefully -- with a count and an instruction -- and two
        # copies of one 300-character refusal is a turn's worth of tokens spent
        # saying the same thing twice.
        failures = [a for a in self.failures() if a.signature not in stuck_signatures]
        if failures:
            lines.append("REFUSED ACTIONS")
            seen: set[str] = set()
            for attempt in reversed(failures):
                if attempt.signature in seen:
                    continue
                seen.add(attempt.signature)
                if len(seen) > ATTEMPTS_KEPT:
                    break
                where = f" at {attempt.target}" if attempt.target else ""
                lines.append(
                    f"  decision {attempt.step}: {attempt.action}{where}"
                    f" -> {_clip(attempt.error or attempt.status)}"
                )

        open_plans = [p for p in self.plans if p["outcome"] == "open"]
        closed = [p for p in self.plans if p["outcome"] != "open"]
        if open_plans:
            lines.append("CURRENT PLAN")
            lines.append(f"  {open_plans[-1]['plan']}")
        if closed:
            # Not "plans that did not work", which is what this heading said
            # when nothing could reach it. A plan the model replaced of its own
            # accord was superseded, not defeated, and calling that a failure
            # in the prompt would teach the agent the wrong lesson about its
            # own history. The outcome is printed, so the two stay distinct.
            lines.append("EARLIER PLANS")
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
            "notes": [
                {"step": n.step, "statement": n.statement, "source": n.source} for n in self.notes
            ],
        }
