"""Benchmark metrics for a production task, denominated in simulated ticks.

`docs/research/book-synthesis-2026-09-09.md` §12 asks for time to first
sustained output, cumulative delivered production, and final output rate. It is
explicit that these are properties of *game time*, and that matters here for a
concrete reason: the goal vector normalises progress by decisions
(`env.py:439`), and `decision_ticks` differs per task and per action profile.
A rate per decision would make an agent that acts rarely look productive and
would not be comparable across profiles at all -- so nothing below may divide
by a decision count.

Kept beside `RewardAccountant` rather than inside it. These numbers describe
the run; they must not be able to change what the agent is paid, and a
recorder that returns no reward cannot.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from factoriorl.tasks.spec import (
    SAMPLE_TOLERANCE_TICKS,
    Predicate,
    PredicateKind,
    TaskSpec,
)

#: Rates are reported per this many ticks. One minute at 60 UPS, matching the
#: default `over_ticks` and the units the commissioning baseline was measured
#: in (`plate_line`: 15 plates per 3600 ticks).
RATE_TICKS = 3600


@dataclass
class ProductionMetrics:
    """Per-episode production record. Reset with the episode, never across one.

    Fed the same `(observation, truth)` the accountant sees, so it needs no
    engine access of its own and is testable without one.
    """

    items: tuple[str, ...]
    #: The window "sustained" is judged over, and the amount required in it.
    #: Taken from the task's own predicate where it declares one, so the metric
    #: and the success criterion cannot disagree about what sustained means.
    over_ticks: int = RATE_TICKS
    at_least: float = 1.0
    #: Widest gap allowed between consecutive samples covering the window.
    #: Mirrors `Predicate.max_sample_gap`; `for_task` reads the task's.
    max_sample_gap: int = SAMPLE_TOLERANCE_TICKS
    #: Which truth key the rates are computed over. A benchmark task measures
    #: `produced`, which is every item entering the force's inventory space, and
    #: that is what every existing result was measured against. An open world
    #: measures `machine_produced` -- roadmap A4.2 requires cumulative machine
    #: output be kept separate from handcrafting, and on a world where the agent
    #: bootstraps by hand the two are not close.
    source: str = "produced"
    #: Grow `items` from whatever the world actually produces. A task declares
    #: its items in its predicates; an open world has no predicates and produces
    #: whatever the agent decides to, so the set cannot be known in advance.
    discover: bool = False
    _samples: list[tuple[int, dict[str, float]]] = field(default_factory=list)
    _first_sustained: dict[str, int] = field(default_factory=dict)
    #: First tick each item was produced by a machine *at all*. Distinct from
    #: `_first_sustained`, which needs a full window above a rate -- A4.2 asks
    #: for "time to first machine production", which is a different question and
    #: was not answerable before.
    _first_output: dict[str, int] = field(default_factory=dict)
    #: `record` is called from the environment's own thread *and* from the
    #: wall-clock sampler (roadmap A4.3), which is the only source dense enough
    #: to fill a 60-second window in a realtime run. Two threads appending to
    #: one list is a race with a silent outcome -- a dropped sample looks
    #: exactly like a window nobody observed.
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @classmethod
    def for_task(cls, spec: TaskSpec) -> ProductionMetrics:
        """Read the items and the window off the task's own predicates."""
        items: list[str] = []
        over_ticks = RATE_TICKS
        at_least = 1.0
        max_sample_gap = SAMPLE_TOLERANCE_TICKS
        for predicate in spec.success:
            if predicate.kind not in (
                PredicateKind.PRODUCED,
                PredicateKind.SUSTAINED_OUTPUT,
            ):
                continue
            if predicate.item and predicate.item not in items:
                items.append(predicate.item)
            if predicate.kind is PredicateKind.SUSTAINED_OUTPUT:
                over_ticks = predicate.over_ticks
                at_least = predicate.at_least
                max_sample_gap = predicate.max_sample_gap
        return cls(
            items=tuple(items),
            over_ticks=over_ticks,
            at_least=at_least,
            max_sample_gap=max_sample_gap,
        )

    @classmethod
    def for_world(cls, mode: Any) -> ProductionMetrics:
        """Metrics for a world with no success predicate (roadmap A4.2).

        `for_task` reads its item list off `PRODUCED`/`SUSTAINED_OUTPUT`
        predicates. An open world declares none -- `open_world.spec_for` sets
        `success=()` -- so that path yields `items=()` and every field of
        `report()` comes back empty. Which is what an `open_factory` run would
        have reported: nothing at all, for the entire run.
        """
        del mode  # the window is the game's minute, not the world's business
        return cls(items=(), source="machine_produced", discover=True)

    # ---- recording ----------------------------------------------------
    def reset(self) -> None:
        self._samples = []
        self._first_sustained = {}
        self._first_output = {}

    def record(self, observation: dict, truth: dict) -> None:
        with self._lock:
            self._record(observation, truth)

    def _record(self, observation: dict, truth: dict) -> None:
        tick = int(observation.get("tick") or 0)
        counted = truth.get(self.source) or {}
        if self.discover:
            # Order is stable and additive: a new item joins the end and every
            # earlier sample is treated as zero for it, which is true.
            found = [name for name in sorted(counted) if name not in self.items]
            if found:
                self.items = (*self.items, *found)
        produced = {item: float(counted.get(item, 0)) for item in self.items}
        for item, count in produced.items():
            if count > 0 and item not in self._first_output:
                self._first_output[item] = tick
        # A step that advanced no ticks replaces its sample rather than adding
        # one, so a burst of zero-tick decisions cannot dilute a rate.
        if self._samples and self._samples[-1][0] == tick:
            self._samples[-1] = (tick, produced)
        else:
            self._samples.append((tick, produced))
        # The same rule the predicate applies, and for the same reason: before
        # a full window has elapsed the baseline falls back to the earliest
        # sample, so "sustained" would just mean "produced this much".
        if self._elapsed() >= self.over_ticks and self._sampled():
            for item in self.items:
                if item in self._first_sustained:
                    continue
                if self._window_output(item, tick) >= self.at_least:
                    self._first_sustained[item] = tick

    # ---- derivation ---------------------------------------------------
    def _elapsed(self) -> int:
        if len(self._samples) < 2:
            return 0
        return self._samples[-1][0] - self._samples[0][0]

    def _sampled(self) -> bool:
        """Whether the window was observed, not merely spanned.

        The same hole `Predicate.window_sampled` closes, and for the same
        reason: with no sample inside the window, output produced just before
        it is indistinguishable from output produced in it. Two samples 3,630
        ticks apart otherwise report a rate.
        """
        if len(self._samples) < 2:
            return False
        cutoff = self._samples[-1][0] - self.over_ticks
        start = 0
        for index, (tick, _counts) in enumerate(self._samples):
            if tick <= cutoff:
                start = index
            else:
                break
        ticks = [tick for tick, _counts in self._samples[start:]]
        if len(ticks) < 2:
            return False
        widest = max(b - a for a, b in zip(ticks[:-1], ticks[1:], strict=True))
        return widest <= self.max_sample_gap

    def _baseline(self, cutoff: int) -> dict[str, float]:
        """Counts at the last sample at or before `cutoff`.

        With no such sample the episode is younger than the window and the
        earliest sample is used -- the same rule `Predicate.window_output`
        applies, so the metric and the predicate agree on partial windows.
        """
        baseline = self._samples[0][1]
        for tick, counts in self._samples:
            if tick <= cutoff:
                baseline = counts
            else:
                break
        return baseline

    def _window_output(self, item: str, latest_tick: int) -> float:
        if not self._samples:
            return 0.0
        latest = self._samples[-1][1]
        baseline = self._baseline(latest_tick - self.over_ticks)
        return max(0.0, latest.get(item, 0.0) - baseline.get(item, 0.0))

    def report(self) -> dict:
        """§12's three metrics, plus what they were divided by.

        `rate_denominator` is stated in the payload rather than left to a
        reader's assumption: the same numbers divided by decisions would not be
        comparable across action profiles, and a report that does not say which
        it used cannot be checked.
        """
        if not self._samples:
            return {
                "rate_denominator": "simulated_ticks",
                "episode_ticks": 0,
                "ticks_to_first_sustained_output": dict.fromkeys(self.items),
                "ticks_to_first_output": dict.fromkeys(self.items),
                "cumulative_produced": dict.fromkeys(self.items, 0.0),
                "final_output_rate": dict.fromkeys(self.items, 0.0),
                "over_ticks": self.over_ticks,
                "counts": self.source,
                "samples": 0,
            }
        latest_tick, latest = self._samples[-1]
        first_tick = self._samples[0][0]
        elapsed = max(0, latest_tick - first_tick)
        span = min(self.over_ticks, elapsed) or 0
        rates: dict[str, float] = {}
        for item in self.items:
            output = self._window_output(item, latest_tick)
            # An episode shorter than one window is reported over what actually
            # elapsed, not extrapolated to a full window: a single plate in the
            # first 60 ticks is not "60 plates a minute".
            rates[item] = (output / span * RATE_TICKS) if span else 0.0
        return {
            "rate_denominator": "simulated_ticks",
            "episode_ticks": elapsed,
            "ticks_to_first_sustained_output": {
                item: self._first_sustained.get(item) for item in self.items
            },
            # A4.2 asks for "time to first machine production", which is not
            # the same question as time to first *sustained* output above: one
            # plate answers this and does not answer that.
            "ticks_to_first_output": {item: self._first_output.get(item) for item in self.items},
            "cumulative_produced": {item: latest.get(item, 0.0) for item in self.items},
            "final_output_rate": rates,
            "over_ticks": self.over_ticks,
            # Which truth key these came from. `produced` counts handcrafting
            # and mining too; `machine_produced` is the derived machine-only
            # column. A report that does not say which cannot be compared.
            "counts": self.source,
            "samples": len(self._samples),
        }


def sustained_predicates(spec: TaskSpec) -> tuple[Predicate, ...]:
    """The task's sustained-output predicates, if any."""
    return tuple(p for p in spec.success if p.kind is PredicateKind.SUSTAINED_OUTPUT)
