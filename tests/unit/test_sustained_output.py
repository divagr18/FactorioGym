"""Sustained output, and predicates that can name an agent-built machine.

Two blockers R3 could not clear:

* `containers` and `working` are keyed on `scene.aliases`, bound when the scene
  is *installed*, so a machine the agent places has no name any predicate can
  use. `BUILT` and `ANY_WORKING` key on the prototype instead.
* `PRODUCED` is a monotone counter with no timestamp, so "30 plates, 15 in the
  final window" and "30 plates, line dead since tick 4000" are the same number.
  That is exactly the distinction the literature synthesis §12's gate is made of: "a
  policy that produces a small burst then stops cannot pass a
  sustained-operation objective".
"""

from __future__ import annotations

import pytest

from factoriorl.tasks.spec import Predicate, PredicateKind

WINDOW = 3600


def _sustained(at_least=10, over_ticks=WINDOW):
    return Predicate(
        PredicateKind.SUSTAINED_OUTPUT,
        item="iron-plate",
        at_least=at_least,
        over_ticks=over_ticks,
    )


def _history(rate_per_tick, ticks=8000, step=30, stop_at=None):
    """A `(tick, produced)` history at a constant rate, optionally stopping.

    `step` is 30 because that is what the environment actually produces -- one
    sample per `decision_ticks`. A coarser fixture is not a smaller version of
    a real history, it is an *under-sampled* one, and `window_sampled` refuses
    those on purpose.
    """
    out, total = [], 0.0
    for tick in range(0, ticks + 1, step):
        if stop_at is None or tick <= stop_at:
            total = rate_per_tick * min(tick, stop_at if stop_at else tick)
        out.append((tick, {"iron-plate": int(total)}))
    return out


class TestTheGateDiscriminates:
    def test_a_burst_that_stopped_fails_despite_a_larger_total(self):
        """The failure mode §12 names, and the reason `PRODUCED` is not enough."""
        burst = _history(200 / WINDOW, stop_at=1200)
        steady = _history(15 / WINDOW)
        burst_total = burst[-1][1]["iron-plate"]
        steady_total = steady[-1][1]["iron-plate"]

        assert burst_total > steady_total, "fixture must make the burst look better"
        assert not _sustained(at_least=10).evaluate({}, {"window": burst})
        assert _sustained(at_least=10).evaluate({}, {"window": steady})

    def test_produced_cannot_tell_them_apart(self):
        """Why a new kind was needed rather than a bigger threshold."""
        burst = _history(200 / WINDOW, stop_at=1200)
        steady = _history(15 / WINDOW)
        produced = Predicate(PredicateKind.PRODUCED, item="iron-plate", at_least=10)
        assert produced.evaluate({}, {"produced": burst[-1][1]})
        assert produced.evaluate({}, {"produced": steady[-1][1]})

    def test_output_is_measured_inside_the_window_only(self):
        history = _history(15 / WINDOW)
        wide = _sustained(over_ticks=WINDOW * 2).window_output({"window": history})
        narrow = _sustained(over_ticks=WINDOW // 2).window_output({"window": history})
        assert wide > narrow


class TestWindowEdges:
    def test_an_empty_history_is_zero_not_an_error(self):
        assert _sustained().window_output({}) == 0.0
        assert _sustained().window_output({"window": []}) == 0.0

    def test_an_episode_younger_than_its_window_cannot_pass_at_all(self):
        """Measured on a real engine: `build_line` passed at tick 2850 with 10
        plates in total, because the baseline falls back to the earliest sample
        before a window has elapsed. A line that then died would have produced
        the same 10 -- which is the case this predicate exists to reject."""
        young = [(0, {"iron-plate": 0}), (600, {"iron-plate": 4})]
        # The measurement is still a lower bound on the rate...
        assert _sustained(at_least=4).window_output({"window": young}) == 4.0
        # ...but the predicate refuses to answer on it.
        assert not _sustained(at_least=4).evaluate({}, {"window": young})
        assert not _sustained(at_least=10).evaluate({}, {"window": young})
        assert not _sustained().window_elapsed({"window": young})

    def test_it_passes_once_the_window_has_elapsed(self):
        history = _history(15 / WINDOW)
        assert _sustained().window_elapsed({"window": history})
        assert _sustained(at_least=10).evaluate({}, {"window": history})

    def test_a_single_sample_is_never_a_window(self):
        one = [(9999, {"iron-plate": 500})]
        assert not _sustained().window_elapsed({"window": one})
        assert not _sustained(at_least=1).evaluate({}, {"window": one})

    def test_a_counter_that_went_backwards_cannot_produce_negative_output(self):
        odd = [(0, {"iron-plate": 20}), (4000, {"iron-plate": 5})]
        assert _sustained().window_output({"window": odd}) == 0.0

    def test_an_unknown_item_is_zero(self):
        history = _history(15 / WINDOW)
        other = Predicate(
            PredicateKind.SUSTAINED_OUTPUT, item="copper-plate", at_least=1, over_ticks=WINDOW
        )
        assert other.window_output({"window": history}) == 0.0

    def test_the_baseline_is_the_last_sample_at_or_before_the_cutoff(self):
        history = [
            (0, {"iron-plate": 0}),
            (2000, {"iron-plate": 5}),
            (4000, {"iron-plate": 9}),
            (7600, {"iron-plate": 30}),
        ]
        # cutoff = 7600 - 3600 = 4000, so the baseline is the sample at 4000.
        assert _sustained().window_output({"window": history}) == 30 - 9


class TestAgentBuiltMachinery:
    def test_built_counts_what_the_agent_placed(self):
        built = Predicate(PredicateKind.BUILT, item="stone-furnace", at_least=2)
        assert built.evaluate({}, {"built": {"stone-furnace": 2}})
        assert not built.evaluate({}, {"built": {"stone-furnace": 1}})
        assert not built.evaluate({}, {})

    def test_any_working_needs_no_scene_alias(self):
        """The blocker: `ENTITY_WORKING` keys on an install-time alias."""
        aliased = Predicate(PredicateKind.ENTITY_WORKING, marker="drill")
        prototyped = Predicate(PredicateKind.ANY_WORKING, item="burner-mining-drill", at_least=1)
        # An agent-built drill: no alias, but it is running.
        truth = {"working": {}, "working_counts": {"burner-mining-drill": 1}}
        assert not aliased.evaluate({}, truth), "an alias cannot name an agent-built machine"
        assert prototyped.evaluate({}, truth)

    def test_any_working_requires_the_count(self):
        both = Predicate(PredicateKind.ANY_WORKING, item="stone-furnace", at_least=2)
        assert not both.evaluate({}, {"working_counts": {"stone-furnace": 1}})
        assert both.evaluate({}, {"working_counts": {"stone-furnace": 2}})

    def test_placed_and_built_are_different_questions(self):
        """A preplaced machine is placed but not built by the agent."""
        built = Predicate(PredicateKind.BUILT, item="stone-furnace", at_least=1)
        truth = {"placed_counts": {"stone-furnace": 1}, "built": {}}
        assert not built.evaluate({}, truth)


class TestExistingPredicatesAreUntouched:
    """Package 1 is additive; every current family reads the old fields."""

    @pytest.mark.parametrize(
        ("kind", "kwargs", "truth"),
        [
            (
                PredicateKind.CONTAINER_HOLDS,
                {"marker": "sink", "item": "iron-plate"},
                {"containers": {"sink": {"iron-plate": 1}}},
            ),
            (PredicateKind.ENTITY_WORKING, {"marker": "drill"}, {"working": {"drill": True}}),
            (PredicateKind.PRODUCED, {"item": "iron-plate"}, {"produced": {"iron-plate": 1}}),
        ],
    )
    def test_old_kinds_still_read_their_old_fields(self, kind, kwargs, truth):
        assert Predicate(kind, **kwargs).evaluate({}, truth)

    def test_the_new_fields_do_not_satisfy_an_old_predicate(self):
        working = Predicate(PredicateKind.ENTITY_WORKING, marker="drill")
        assert not working.evaluate({}, {"working_counts": {"burner-mining-drill": 5}})


class TestASettlingPeriod:
    """Why the window alone was not enough, measured by the R3 gate.

    A trailing window that begins at tick 0 spans the construction phase, so at
    tick 3600 the window still holds every plate the line ever made. The gate
    observed `build_line` accepted at tick 3600 with a drill that had been mined
    out 90 ticks earlier -- so at those parameters the criterion was doing
    almost nothing `PRODUCED` does not.
    """

    def _settled(self, at_least=10, over_ticks=WINDOW, not_before=2 * WINDOW):
        return Predicate(
            PredicateKind.SUSTAINED_OUTPUT,
            item="iron-plate",
            at_least=at_least,
            over_ticks=over_ticks,
            not_before_tick=not_before,
        )

    def _built_then_died(self, build_tick=600, death_tick=3510, end_tick=10710):
        """The exact shape the gate produced: built, ran, mined out, waited."""
        history, total = [], 0
        for tick in range(0, end_tick + 1, 30):
            if build_tick <= tick <= death_tick:
                total = int(15 * (tick - build_tick) / WINDOW)
            history.append((tick, {"iron-plate": total}))
        return history

    def test_the_unsettled_predicate_accepts_a_line_that_has_stopped(self):
        """The defect, stated as a test so the fix is not mistaken for taste."""
        history = self._built_then_died(end_tick=3600)
        loose = _sustained(at_least=10)
        assert loose.evaluate({}, {"window": history}), (
            "fixture must reproduce the accepted-while-dead case"
        )

    def test_the_settled_predicate_rejects_it(self):
        history = self._built_then_died(end_tick=3600)
        assert not self._settled().evaluate({}, {"window": history})

    def test_it_still_rejects_it_once_the_settling_period_has_passed(self):
        history = self._built_then_died()
        predicate = self._settled()
        assert predicate.settled({"window": history})
        assert predicate.window_output({"window": history}) == 0.0
        assert not predicate.evaluate({}, {"window": history})

    def test_a_line_that_keeps_running_is_accepted_after_settling(self):
        history, total = [], 0
        for tick in range(0, 10711, 30):
            total = int(15 * max(0, tick - 600) / WINDOW)
            history.append((tick, {"iron-plate": total}))
        assert self._settled().evaluate({}, {"window": history})

    def test_a_running_line_is_not_accepted_before_settling(self):
        history, total = [], 0
        for tick in range(0, 5001, 30):
            total = int(15 * max(0, tick - 600) / WINDOW)
            history.append((tick, {"iron-plate": total}))
        predicate = self._settled()
        assert not predicate.settled({"window": history})
        assert predicate.window_output({"window": history}) >= 10
        assert not predicate.evaluate({}, {"window": history})

    def test_no_history_is_never_settled(self):
        assert not self._settled().settled({})
        assert not self._settled().settled({"window": []})

    def test_a_predicate_declaring_no_settling_period_is_unaffected(self):
        """Every existing family leaves this at its default."""
        history = _history(15 / WINDOW)
        assert _sustained(at_least=10).not_before_tick == 0
        assert _sustained(at_least=10).evaluate({}, {"window": history})

    def test_build_line_declares_two_windows(self):
        from factoriorl.tasks import get

        predicate = next(
            p for p in get("build_line").spec.success if p.kind is PredicateKind.SUSTAINED_OUTPUT
        )
        assert predicate.not_before_tick == 2 * predicate.over_ticks


class TestAWindowMustBeSampledNotJustSpanned:
    """A window nothing observed cannot tell a rate from a jump.

    `window_elapsed` checked span only. Measured before this existed:

        history = [(450, 1 plate), (4080, 16 plates)]
        span 3630, samples 2 -> elapsed True, output 15.0, evaluate True

    Two samples 3,630 ticks apart satisfied a criterion built to reject a
    burst. The cause is not the baseline's position -- it sits a plausible 30
    ticks before the cutoff -- it is that no sample lies *inside* the window,
    so production from just before it is attributed to it.

    Unreachable while every stepping path sampled every `decision_ticks`;
    `FactorioEnv.advance` makes external advancement possible, which is what
    R4.1 adds, so it is checked.
    """

    #: The exact history a 3,600-tick external jump left in the recorded
    #: Phase 5 demonstration.
    JUMPED = [(450, {"iron-plate": 1}), (4080, {"iron-plate": 16})]

    def test_the_jump_still_spans_a_window(self):
        """The old check passes on it, which is why it was not enough."""
        assert _sustained().window_elapsed({"window": self.JUMPED})

    def test_but_it_was_never_sampled(self):
        assert not _sustained().window_sampled({"window": self.JUMPED})

    def test_so_it_is_refused(self):
        assert not _sustained(at_least=10).evaluate({}, {"window": self.JUMPED})

    def test_the_measurement_itself_is_unchanged(self):
        """`window_output` still reports what it sees; `evaluate` is what
        refuses. Keeping them separate is what makes the refusal legible."""
        assert _sustained().window_output({"window": self.JUMPED}) == 15.0

    def test_a_history_sampled_every_decision_passes(self):
        history = _history(15 / WINDOW)
        assert _sustained().window_sampled({"window": history})
        assert _sustained(at_least=10).evaluate({}, {"window": history})

    def test_one_gap_anywhere_in_the_window_is_enough_to_refuse(self):
        """Not just a two-sample history: a dense run with a single jump at
        the end is equally unable to place its production."""
        dense = _history(15 / WINDOW, ticks=3600)
        history = dense + [(7200, {"iron-plate": 45})]
        assert _sustained().window_elapsed({"window": history})
        assert not _sustained().window_sampled({"window": history})

    def test_a_gap_before_the_window_does_not_matter(self):
        """Only the samples covering the window are its evidence. A sparse
        prefix is irrelevant, and refusing on it would reject a legitimate
        history whose early part was recorded coarsely."""
        history = [(0, {"iron-plate": 0}), (3600, {"iron-plate": 0})] + [
            (t, {"iron-plate": int(15 * (t - 3600) / WINDOW)}) for t in range(3630, 7201, 30)
        ]
        assert _sustained().window_sampled({"window": history})

    def test_the_bound_is_declared_per_predicate(self):
        """A task stepping every 60 ticks cannot satisfy a 30-tick bound, so
        the bound is a field rather than a constant."""
        coarse = [(t, {"iron-plate": int(15 * t / WINDOW)}) for t in range(0, 7201, 60)]
        assert not _sustained().window_sampled({"window": coarse})
        declared = Predicate(
            PredicateKind.SUSTAINED_OUTPUT,
            item="iron-plate",
            at_least=10,
            over_ticks=WINDOW,
            max_sample_gap=60,
        )
        assert declared.window_sampled({"window": coarse})
        assert declared.evaluate({}, {"window": coarse})

    def test_a_bound_below_the_task_step_is_refused_at_load(self):
        """Otherwise every claim is silently rejected with nothing to say why."""
        import factoriorl.tasks as tasks_module
        from factoriorl.tasks import validate_all
        from factoriorl.tasks.spec import (
            Blueprint,
            LayoutFamily,
            RewardComponent,
            RewardKind,
            TaskSpec,
        )

        spec = TaskSpec(
            id="too_coarse",
            version="1.0.0",
            description="d",
            layout_families=(LayoutFamily("f", "train"), LayoutFamily("g", "test")),
            success=(
                Predicate(
                    PredicateKind.SUSTAINED_OUTPUT,
                    item="iron-plate",
                    at_least=10,
                    over_ticks=WINDOW,
                    max_sample_gap=15,
                ),
            ),
            rewards=(RewardComponent("s", RewardKind.SPARSE_SUCCESS),),
            decision_ticks=30,
        )
        import pytest as _pytest

        _monkey = _pytest.MonkeyPatch()
        try:
            _monkey.setattr(
                tasks_module,
                "all_tasks",
                lambda: {"too_coarse": tasks_module.RegisteredTask(spec, lambda f, r: Blueprint())},
            )
            report = validate_all(sample_seeds=1)
        finally:
            _monkey.undo()
        problems = report["tasks"]["too_coarse"]["problems"]
        assert any("max_sample_gap" in p and "decision_ticks" in p for p in problems), problems

    def test_every_registered_task_declares_a_workable_bound(self):
        from factoriorl.tasks import all_tasks, get

        for name in sorted(all_tasks()):
            spec = get(name).spec
            for predicate in spec.success:
                if predicate.kind is PredicateKind.SUSTAINED_OUTPUT:
                    assert predicate.max_sample_gap >= spec.decision_ticks, name
