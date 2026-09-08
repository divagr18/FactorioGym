"""§12's benchmark metrics, and the denominator they must not use.

The recorder exists because the goal vector normalises progress by *decisions*
(`env.py`), `decision_ticks` differs per task, and `parameterized-v1` spends a
different number of decisions on the same physical work than `primitive-v1`
does. A rate per decision would rank an agent that acts rarely above one that
acts often for identical output, and would not be comparable across profiles
at all.
"""

from __future__ import annotations

from factoriorl.production import RATE_TICKS, ProductionMetrics
from factoriorl.tasks import all_tasks, get
from factoriorl.tasks.spec import (
    Predicate,
    PredicateKind,
    RewardComponent,
    RewardKind,
    TaskSpec,
)


def _feed(metrics, samples):
    metrics.reset()
    for tick, count in samples:
        metrics.record({"tick": tick}, {"produced": {"iron-plate": count}})
    return metrics.report()


def _metrics(**kwargs):
    kwargs.setdefault("items", ("iron-plate",))
    return ProductionMetrics(**kwargs)


def _steady(rate_per_window, ticks=7200, step=60):
    """A constant-rate history, `rate_per_window` per `RATE_TICKS`."""
    return [(t, int(rate_per_window * t / RATE_TICKS)) for t in range(0, ticks + 1, step)]


class TestTheDenominatorIsGameTime:
    def test_the_rate_does_not_change_with_decision_frequency(self):
        """The property the whole module exists for."""
        dense = _feed(_metrics(), _steady(15, step=30))
        sparse = _feed(_metrics(), _steady(15, step=600))
        assert dense["final_output_rate"]["iron-plate"] > 0
        assert (
            abs(
                dense["final_output_rate"]["iron-plate"] - sparse["final_output_rate"]["iron-plate"]
            )
            <= 1.0
        )

    def test_a_burst_of_zero_tick_decisions_cannot_dilute_a_rate(self):
        metrics = _metrics()
        metrics.reset()
        for tick, count in _steady(15):
            metrics.record({"tick": tick}, {"produced": {"iron-plate": count}})
        before = metrics.report()["final_output_rate"]["iron-plate"]
        final_tick, final_count = _steady(15)[-1]
        for _ in range(50):
            metrics.record({"tick": final_tick}, {"produced": {"iron-plate": final_count}})
        after = metrics.report()
        assert after["final_output_rate"]["iron-plate"] == before
        assert after["samples"] == len(_steady(15))

    def test_the_report_states_which_denominator_it_used(self):
        assert _feed(_metrics(), _steady(15))["rate_denominator"] == "simulated_ticks"
        assert _metrics().report()["rate_denominator"] == "simulated_ticks"

    def test_episode_ticks_is_game_time_not_a_step_count(self):
        report = _feed(_metrics(), _steady(15, ticks=7200, step=600))
        assert report["episode_ticks"] == 7200
        assert report["samples"] == 13


class TestTheThreeMetrics:
    def test_time_to_first_sustained_output(self):
        metrics = _metrics(at_least=5.0, over_ticks=RATE_TICKS)
        report = _feed(metrics, _steady(15))
        first = report["ticks_to_first_sustained_output"]["iron-plate"]
        assert first is not None
        # 5 plates at 15 per 3600 ticks arrives around tick 1200.
        assert 1100 <= first <= 1320

    def test_a_line_that_never_ran_reports_none_not_zero(self):
        report = _feed(_metrics(at_least=1.0), [(t, 0) for t in range(0, 3601, 600)])
        assert report["ticks_to_first_sustained_output"]["iron-plate"] is None
        assert report["cumulative_produced"]["iron-plate"] == 0.0

    def test_cumulative_production_is_the_final_counter(self):
        report = _feed(_metrics(), _steady(15))
        assert report["cumulative_produced"]["iron-plate"] == 30

    def test_a_burst_that_stopped_has_a_high_total_and_a_zero_final_rate(self):
        """The distinction §12 asks the benchmark to make."""
        burst = [(t, min(t, 1200) * 66 // 1200) for t in range(0, 7201, 60)]
        report = _feed(_metrics(), burst)
        assert report["cumulative_produced"]["iron-plate"] == 66
        assert report["final_output_rate"]["iron-plate"] == 0.0

        steady = _feed(_metrics(), _steady(15))
        assert steady["cumulative_produced"]["iron-plate"] < 66
        assert steady["final_output_rate"]["iron-plate"] > 0.0

    def test_a_short_episode_is_not_extrapolated_to_a_full_window(self):
        """One plate in 60 ticks is not 60 plates a minute."""
        report = _feed(_metrics(), [(0, 0), (60, 1)])
        assert report["final_output_rate"]["iron-plate"] == RATE_TICKS / 60


class TestEdges:
    def test_an_unrecorded_episode_reports_zeros_not_an_error(self):
        report = _metrics().report()
        assert report["episode_ticks"] == 0
        assert report["cumulative_produced"] == {"iron-plate": 0.0}
        assert report["final_output_rate"] == {"iron-plate": 0.0}

    def test_reset_does_not_carry_production_across_episodes(self):
        metrics = _metrics()
        _feed(metrics, _steady(15))
        assert _feed(metrics, [(0, 0)])["cumulative_produced"]["iron-plate"] == 0.0

    def test_a_task_naming_no_produced_item_records_nothing_and_still_reports(self):
        metrics = ProductionMetrics(items=())
        metrics.reset()
        metrics.record({"tick": 10}, {"produced": {"iron-plate": 5}})
        report = metrics.report()
        assert report["cumulative_produced"] == {}
        assert report["rate_denominator"] == "simulated_ticks"

    def test_a_counter_going_backwards_never_yields_a_negative_rate(self):
        report = _feed(_metrics(), [(0, 20), (7200, 5)])
        assert report["final_output_rate"]["iron-plate"] == 0.0


class TestConstructionFromATask:
    def test_the_window_comes_from_the_task_predicate(self):
        spec = TaskSpec(
            id="t",
            version="1.0.0",
            description="d",
            layout_families=(),
            success=(
                Predicate(
                    PredicateKind.SUSTAINED_OUTPUT,
                    item="iron-plate",
                    at_least=12.0,
                    over_ticks=1800,
                ),
            ),
            rewards=(RewardComponent(name="s", kind=RewardKind.SPARSE_SUCCESS),),
        )
        metrics = ProductionMetrics.for_task(spec)
        assert metrics.items == ("iron-plate",)
        assert metrics.over_ticks == 1800
        assert metrics.at_least == 12.0

    def test_a_produced_predicate_contributes_its_item_without_a_window(self):
        spec = TaskSpec(
            id="t",
            version="1.0.0",
            description="d",
            layout_families=(),
            success=(Predicate(PredicateKind.PRODUCED, item="iron-plate", at_least=3.0),),
            rewards=(),
        )
        metrics = ProductionMetrics.for_task(spec)
        assert metrics.items == ("iron-plate",)
        assert metrics.over_ticks == RATE_TICKS

    def test_every_registered_task_constructs(self):
        for name in all_tasks():
            metrics = ProductionMetrics.for_task(get(name).spec)
            assert metrics.report()["rate_denominator"] == "simulated_ticks"


def test_the_recorder_returns_no_reward():
    """It describes the run; it must not be able to change the payout."""
    assert not hasattr(ProductionMetrics, "step")
    assert not hasattr(ProductionMetrics, "total")
