"""Outage and recovery, defined before the arms run (R4.2).

R4.2: *"Define outage and recovery criteria before evaluating."* These tests
pin what the criteria refuse, because each refusal corresponds to a mistake
already made in this repo:

* `fuel == 0` as an outage -- the drill ran 1,170 ticks on stored energy;
* machine status as an outage -- a healthy line reads `working` 7.5% of ticks;
* a plate counter as a recovery -- the furnace produces on residual heat;
* answering "recovered" when no outage was ever confirmed.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from factoriorl import recovery
from factoriorl.effects import EffectState

DECAY_EVIDENCE = (
    pathlib.Path(__file__).resolve().parents[2] / "docs" / "evidence" / "r4-burner-decay.json"
)


def _history(rows: list[tuple[int, float]]) -> list[tuple[int, dict]]:
    return [(tick, {"iron-plate": plates}) for tick, plates in rows]


def _steady(start: int, end: int, per_window: float, base: float = 0.0, step: int = 30):
    """A history producing `per_window` plates per `RATE_WINDOW_TICKS`."""
    rate = per_window / recovery.RATE_WINDOW_TICKS
    return _history([(t, base + rate * (t - start)) for t in range(start, end + 1, step)])


def _flat(start: int, end: int, plates: float, step: int = 30):
    return _history([(t, plates) for t in range(start, end + 1, step)])


class TestTheCriteriaAreDeclaredAndMeasured:
    def test_the_steady_rate_is_the_measured_one(self):
        """15 plates per 3,600 ticks, so 2.5 per 600."""
        assert recovery.STEADY_PER_WINDOW == pytest.approx(15 * 600 / 3600)

    def test_the_report_carries_what_it_was_judged_by(self):
        body = recovery.Criteria().to_dict()
        for key in (
            "rate_window_ticks",
            "outage_at_or_below",
            "outage_sustained_ticks",
            "recovery_at_least",
            "recovery_sustained_ticks",
            "deadline_ticks",
        ):
            assert key in body
        assert "windowed production only" in body["basis"]

    def test_recovery_demands_most_of_the_steady_rate(self):
        assert recovery.RECOVERY_AT_LEAST == pytest.approx(0.8 * recovery.STEADY_PER_WINDOW)


class TestAnUnsampledWindowIsNotAnIdleOne:
    def test_a_window_reaching_before_the_history_is_none(self):
        history = _steady(0, 300, 2.5)
        assert recovery.rate_at(history, 300, recovery.Criteria()) is None

    def test_an_unsampled_window_does_not_count_toward_an_outage(self):
        """Zero would report an unobserved window as an idle one, and a run of
        those would confirm an outage that was never seen."""
        history = _steady(0, 300, 2.5)
        verdict = recovery.judge_outage(history, recovery.Criteria(), disrupted_at=0)
        assert verdict.state is not EffectState.CONFIRMED
        assert verdict.rates == []


@pytest.fixture(scope="module")
def curve() -> dict:
    """The measured decay curve. Module-scoped and a plain function: a
    class-scoped fixture defined as a method is deprecated in pytest."""
    if not DECAY_EVIDENCE.is_file():
        pytest.skip("run tools/burner_decay.py first")
    return json.loads(DECAY_EVIDENCE.read_text(encoding="utf-8"))


class TestOutage:
    def test_a_line_that_keeps_running_is_not_an_outage(self):
        """The Phase 5 case: fuel gone, production unaffected.

        Run past the deadline, so the verdict is a positive "no outage" rather
        than "not observed long enough" -- both are non-confirmations, and only
        one of them is a finding.
        """
        history = _steady(0, 3600 + recovery.Criteria().deadline_ticks + 600, 2.5)
        verdict = recovery.judge_outage(history, recovery.Criteria(), disrupted_at=3600)
        assert verdict.state is EffectState.REFUSED
        assert "never stopped" in verdict.detail

    def test_a_line_that_stops_is_an_outage(self):
        history = _steady(0, 3600, 2.5) + _flat(3630, 9000, 15.0)
        verdict = recovery.judge_outage(history, recovery.Criteria(), disrupted_at=3600)
        assert verdict.state is EffectState.CONFIRMED
        assert verdict.at_tick is not None

    def test_it_is_not_declared_until_the_duration_has_held(self):
        """A single empty window at a transition is sampling noise."""
        history = _steady(0, 3600, 2.5) + _flat(3630, 4000, 15.0)
        verdict = recovery.judge_outage(history, recovery.Criteria(), disrupted_at=3600)
        assert verdict.state is EffectState.UNKNOWN

    def test_too_short_a_run_is_unknown_not_absent(self):
        """ "No outage" is a claim, and a run that ended early cannot make it."""
        history = _steady(0, 5000, 2.5)
        verdict = recovery.judge_outage(history, recovery.Criteria(), disrupted_at=3600)
        assert verdict.state is EffectState.UNKNOWN
        assert "deadline" in verdict.detail


class TestRecovery:
    def test_no_confirmed_outage_means_nothing_to_recover_from(self):
        """The old demonstration's error, as a test: it reported recovery from
        an outage it never established."""
        history = _steady(0, 10000, 2.5)
        verdict = recovery.judge_recovery(history, recovery.Criteria(), outage_at=None)
        assert verdict.state is EffectState.UNKNOWN
        assert "nothing to have recovered from" in verdict.detail

    def test_output_returning_and_holding_is_a_recovery(self):
        stopped = _flat(0, 1200, 0.0)
        back = _steady(1230, 6000, 2.5, base=0.0, step=30)
        verdict = recovery.judge_recovery(stopped + back, recovery.Criteria(), outage_at=1200)
        assert verdict.state is EffectState.CONFIRMED

    def test_one_plate_is_not_a_recovery(self):
        """A furnace burning off residual heat makes a plate or two."""
        history = _flat(0, 1200, 0.0) + _flat(1230, 6000, 2.0)
        verdict = recovery.judge_recovery(history, recovery.Criteria(), outage_at=1200)
        assert verdict.state is EffectState.REFUSED

    def test_a_materially_slower_line_is_not_recovered(self):
        history = _flat(0, 1200, 0.0) + _steady(1230, 6000, 1.0)
        verdict = recovery.judge_recovery(history, recovery.Criteria(), outage_at=1200)
        assert verdict.state is EffectState.REFUSED

    def test_output_that_returns_too_late_to_verify_is_unknown(self):
        history = _flat(0, 1200, 0.0) + _steady(1230, 2400, 2.5)
        verdict = recovery.judge_recovery(history, recovery.Criteria(), outage_at=1200)
        assert verdict.state is EffectState.UNKNOWN


class TestAgainstTheMeasuredDecayCurve:
    """The criteria have to give the right answer on real data, not only on
    fixtures shaped to suit them."""

    def _history(self, curve: dict):
        rows = curve["settle_curve"] + curve["decay_curve"]
        return [(r["tick"], {"iron-plate": r["plates"]}) for r in rows]

    def test_the_outage_is_confirmed(self, curve):
        verdict = recovery.judge_outage(
            self._history(curve), recovery.Criteria(), disrupted_at=curve["at_removal"]["tick"]
        )
        assert verdict.state is EffectState.CONFIRMED

    def test_it_is_confirmed_well_after_the_disruption(self, curve):
        """The number that unmakes the old claim: the line was still producing
        long after the fuel was taken, so an outage declared at removal time
        would have been declared on a running line."""
        removal = curve["at_removal"]["tick"]
        verdict = recovery.judge_outage(
            self._history(curve), recovery.Criteria(), disrupted_at=removal
        )
        assert verdict.at_tick - removal > 1000

    def test_the_demonstration_handed_back_before_any_outage_existed(self, curve):
        """It refuelled 90 ticks after removal. Under a declared criterion
        there was nothing to recover from at that point."""
        removal = curve["at_removal"]["tick"]
        verdict = recovery.judge_outage(
            self._history(curve), recovery.Criteria(), disrupted_at=removal
        )
        assert verdict.at_tick - removal > curve["findings"]["demonstration_recovery_ticks"]

    def test_recovery_is_refused_because_nothing_refuelled_it(self, curve):
        result = recovery.judge(self._history(curve), disrupted_at=curve["at_removal"]["tick"])
        assert result["recovery"]["state"] == EffectState.REFUSED.value

    def test_the_curve_agrees_with_the_machine_arithmetic(self, curve):
        found = curve["findings"]
        assert found["measured_agrees_with_arithmetic"], found
        assert found["line_was_still_running_when_the_agent_refuelled"]
