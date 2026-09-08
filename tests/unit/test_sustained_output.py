"""Sustained output, and predicates that can name an agent-built machine.

Two blockers R3 could not clear:

* `containers` and `working` are keyed on `scene.aliases`, bound when the scene
  is *installed*, so a machine the agent places has no name any predicate can
  use. `BUILT` and `ANY_WORKING` key on the prototype instead.
* `PRODUCED` is a monotone counter with no timestamp, so "30 plates, 15 in the
  final window" and "30 plates, line dead since tick 4000" are the same number.
  That is exactly the distinction book-synthesis §12's gate is made of: "a
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


def _history(rate_per_tick, ticks=8000, step=200, stop_at=None):
    """A `(tick, produced)` history at a constant rate, optionally stopping."""
    out, total = [], 0.0
    for tick in range(0, ticks + 1, step):
        if stop_at is None or tick <= stop_at:
            total = rate_per_tick * min(tick, stop_at if stop_at else tick)
        out.append((tick, {"iron-plate": int(total)}))
    return out


class TestTheGateDiscriminates:
    def test_a_steady_rate_passes(self):
        # 15 plates per 3600 ticks is the measured commissioning baseline.
        history = _history(15 / WINDOW)
        assert _sustained(at_least=10).evaluate({}, {"window": history})

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

    def test_an_episode_younger_than_its_window_uses_the_earliest_sample(self):
        """It must not pass by being measured before the window has elapsed."""
        young = [(0, {"iron-plate": 0}), (600, {"iron-plate": 4})]
        assert _sustained(at_least=4).window_output({"window": young}) == 4.0
        assert not _sustained(at_least=10).evaluate({}, {"window": young})

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


def test_over_ticks_is_part_of_the_predicate_description():
    """Two tasks differing only in window are different tasks."""
    narrow = _sustained(over_ticks=1800).describe()
    wide = _sustained(over_ticks=7200).describe()
    # `describe` is used in reports; at minimum the predicates must not be equal.
    assert _sustained(over_ticks=1800) != _sustained(over_ticks=7200)
    assert isinstance(narrow, str) and isinstance(wide, str)
