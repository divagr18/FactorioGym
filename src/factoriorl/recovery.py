"""When has a line stopped, and when has it come back? (R4.2)

R4.2: *"Define outage and recovery criteria before evaluating."* This module is
that definition, and it is a module rather than a paragraph so the criteria are
fixed before an arm runs and cannot be chosen after seeing one.

Three things the measurements already rule out, so the criteria are constrained
rather than invented:

**Fuel level is not an outage.** The Phase 5 disruption emptied both fuel
inventories at tick 4050 and the drill was still `working`, at full rate, for
1,170 ticks afterwards -- `docs/evidence/r4-burner-decay.json`, cross-checked
against 2,999,833 J / 150 kW = 1,199 ticks. A criterion reading `fuel == 0`
would have declared an outage that had not happened, which is what the old
report did.

**Machine status is not an outage either.** A *healthy* plate line reads
`working` on 7.5% of sampled ticks over 102,451 samples
(the project ledger): one burner drill outpaces one stone furnace, so the drill
sits in `waiting_for_space_in_destination` most of the time. An instantaneous
status check on a working line is false 92.5% of the time.

**A plate counter is not a recovery.** `tools/demonstration.py` already
recorded the false positive: *"the furnace keeps producing for a while on heat
it already had, so a plate counter says the line recovered when nothing has
been refuelled at all."*

So both criteria are windowed *production*, with declared durations and a
declared deadline, and `UNKNOWN` is a first-class answer rather than a
failure -- the same shape `effects.py` uses for a commanded action's expected
effect.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from factoriorl.effects import EffectState

#: Trailing window the rate is judged over, in ticks.
#:
#: 600 rather than 3,600: the decay curve reaches flat production about 1,200
#: ticks after the fuel goes, and a 3,600-tick window would still be averaging
#: over the run-on for another two windows after that. 600 is long enough that a
#: healthy line always has output in it -- the measured rate is 15 plates per
#: 3,600, so 2.5 per 600 -- and short enough to locate the transition.
RATE_WINDOW_TICKS = 600

#: Plates per `RATE_WINDOW_TICKS` a running line is expected to make. Measured,
#: not chosen: 15 per 3,600 ticks in steady state.
STEADY_PER_WINDOW = 2.5

#: At or below this, the line is not producing. Zero, deliberately: the decay
#: curve goes exactly flat once the drill's buffer drains, so nothing is gained
#: by a soft floor and a soft floor would fire during the run-on.
OUTAGE_AT_OR_BELOW = 0.0

#: How long output must stay at the floor before an outage is declared. One
#: window: a single empty window can be sampling noise at a transition, two
#: consecutive ones cannot.
OUTAGE_SUSTAINED_TICKS = 600

#: Output per window that counts as recovered. 80% of steady, so a line brought
#: back to a materially lower rate does not read as recovered, and a line
#: restored to full rate is not failed by one slow window.
RECOVERY_AT_LEAST = 0.8 * STEADY_PER_WINDOW

#: How long recovered output must hold. A full measurement window, so
#: "recovered" means the same thing `SUSTAINED_OUTPUT` means and not "one plate
#: appeared".
RECOVERY_SUSTAINED_TICKS = 3600


@dataclass(frozen=True)
class Criteria:
    """The declared thresholds, so a report carries what it was judged by."""

    item: str = "iron-plate"
    rate_window_ticks: int = RATE_WINDOW_TICKS
    outage_at_or_below: float = OUTAGE_AT_OR_BELOW
    outage_sustained_ticks: int = OUTAGE_SUSTAINED_TICKS
    recovery_at_least: float = RECOVERY_AT_LEAST
    recovery_sustained_ticks: int = RECOVERY_SUSTAINED_TICKS
    #: Beyond this many ticks after the disruption, an unfired criterion is
    #: `UNKNOWN` rather than false: the run ended before the question could be
    #: answered, and saying "no outage" would be a claim the data cannot make.
    deadline_ticks: int = 9000

    def to_dict(self) -> dict:
        return {
            "item": self.item,
            "rate_window_ticks": self.rate_window_ticks,
            "outage_at_or_below": self.outage_at_or_below,
            "outage_sustained_ticks": self.outage_sustained_ticks,
            "recovery_at_least": self.recovery_at_least,
            "recovery_sustained_ticks": self.recovery_sustained_ticks,
            "deadline_ticks": self.deadline_ticks,
            "steady_per_window": STEADY_PER_WINDOW,
            "basis": (
                "windowed production only. Fuel level and machine status are both "
                "unusable: a drill runs 1,170 ticks on stored energy after its fuel "
                "is taken, and a healthy line reads `working` on 7.5% of ticks."
            ),
        }


def rate_at(history: list[tuple[int, dict]], tick: int, criteria: Criteria) -> float | None:
    """Output in the `rate_window_ticks` ending at `tick`, or None if unsampled.

    None rather than 0.0 when the window has no sample at its start: an
    unobserved window cannot be reported as an idle one. Same reasoning as
    `Predicate.window_sampled`, and the same failure it prevents.
    """
    if not history:
        return None
    cutoff = tick - criteria.rate_window_ticks
    if cutoff < history[0][0]:
        return None
    baseline = None
    latest = None
    for sample_tick, counts in history:
        if sample_tick <= cutoff:
            baseline = counts
        if sample_tick <= tick:
            latest = counts
    if baseline is None or latest is None:
        return None
    return max(0.0, float(latest.get(criteria.item, 0)) - float(baseline.get(criteria.item, 0)))


@dataclass
class Verdict:
    """What the history says about one arm."""

    state: EffectState
    #: First tick at which the criterion had held for its full duration.
    at_tick: int | None = None
    detail: str = ""
    #: The windowed rate at each sampled tick, for the report to show.
    rates: list[tuple[int, float]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "state": self.state.value,
            "at_tick": self.at_tick,
            "detail": self.detail,
            "rates": [[t, round(r, 3)] for t, r in self.rates],
        }


def _sustained(
    history: list[tuple[int, dict]],
    criteria: Criteria,
    *,
    since: int,
    holds: Callable[[float], bool],
    duration: int,
) -> tuple[int | None, list[tuple[int, float]]]:
    """First tick from which `holds` was true continuously for `duration`."""
    rates: list[tuple[int, float]] = []
    run_start: int | None = None
    answer: int | None = None
    for tick, _counts in history:
        if tick < since:
            continue
        rate = rate_at(history, tick, criteria)
        if rate is None:
            run_start = None
            continue
        rates.append((tick, rate))
        if holds(rate):
            if run_start is None:
                run_start = tick
            elif answer is None and tick - run_start >= duration:
                answer = tick
        else:
            run_start = None
    return answer, rates


def judge_outage(
    history: list[tuple[int, dict]], criteria: Criteria, *, disrupted_at: int
) -> Verdict:
    """Did production actually stop after the disruption?"""
    at, rates = _sustained(
        history,
        criteria,
        since=disrupted_at,
        holds=lambda rate: rate <= criteria.outage_at_or_below,
        duration=criteria.outage_sustained_ticks,
    )
    if at is not None:
        return Verdict(
            EffectState.CONFIRMED,
            at,
            f"output stayed at or below {criteria.outage_at_or_below} for "
            f"{criteria.outage_sustained_ticks} ticks",
            rates,
        )
    latest = history[-1][0] if history else disrupted_at
    if latest - disrupted_at < criteria.deadline_ticks:
        return Verdict(
            EffectState.UNKNOWN,
            None,
            f"only {latest - disrupted_at} ticks observed after the disruption, "
            f"short of the {criteria.deadline_ticks}-tick deadline; the run ended "
            "before an outage could be ruled out",
            rates,
        )
    return Verdict(
        EffectState.REFUSED,
        None,
        "production never stopped for the declared duration, so no outage "
        "occurred -- whatever was done to the fuel",
        rates,
    )


def judge_recovery(
    history: list[tuple[int, dict]], criteria: Criteria, *, outage_at: int | None
) -> Verdict:
    """Did production come back, and stay back?

    `outage_at` is required: a recovery from an outage that never happened is
    not a recovery, and answering anyway is the error the old demonstration
    made. With no confirmed outage the verdict is `UNKNOWN`, not success.
    """
    if outage_at is None:
        return Verdict(
            EffectState.UNKNOWN,
            None,
            "no outage was confirmed, so there is nothing to have recovered from",
        )
    at, rates = _sustained(
        history,
        criteria,
        since=outage_at,
        holds=lambda rate: rate >= criteria.recovery_at_least,
        duration=criteria.recovery_sustained_ticks,
    )
    if at is not None:
        return Verdict(
            EffectState.CONFIRMED,
            at,
            f"output held at or above {criteria.recovery_at_least:.2f} per "
            f"{criteria.rate_window_ticks} ticks for "
            f"{criteria.recovery_sustained_ticks} ticks",
            rates,
        )
    latest = history[-1][0] if history else outage_at
    if latest - outage_at < criteria.recovery_sustained_ticks:
        return Verdict(
            EffectState.UNKNOWN,
            None,
            f"only {latest - outage_at} ticks observed after the outage, short of "
            f"the {criteria.recovery_sustained_ticks} a recovery must hold for",
            rates,
        )
    return Verdict(EffectState.REFUSED, None, "output never returned and held", rates)


def judge(
    history: list[tuple[int, dict]],
    criteria: Criteria | None = None,
    *,
    disrupted_at: int,
) -> dict:
    """Both verdicts for one arm, plus the criteria they were judged by."""
    criteria = criteria or Criteria()
    outage = judge_outage(history, criteria, disrupted_at=disrupted_at)
    recovery = judge_recovery(history, criteria, outage_at=outage.at_tick)
    return {
        "criteria": criteria.to_dict(),
        "disrupted_at": disrupted_at,
        "outage": outage.to_dict(),
        "recovery": recovery.to_dict(),
    }
