"""Commanded, expected, observed -- three separate things.

`docs/research/book-synthesis-2026-09-09.md` §5: executing a command and
observing its intended effect are different, and its gate asks that a delay and
an actual failure be injected *separately* -- the monitor should wait through a
plausible delay and eventually report failure under its declared deadline.

Also: "issued fuel transfer" must not become "machine is fuelled" without a
successful result.
"""

from __future__ import annotations

import pytest

from factoriorl.effects import (
    EXPECTATIONS,
    ActionRecord,
    EffectKind,
    EffectState,
    Expectation,
    expectation_for,
    observe,
)


def _record(kind, deadline=0, detail=None, tick=100):
    return ActionRecord(
        action_key="k",
        commanded={"action": "place"},
        expectation=Expectation(kind, deadline, detail or {}),
        issued_tick=tick,
    )


class TestDelayAndFailureAreSeparate:
    """The gate's requirement, stated as three cases."""

    def test_it_waits_through_a_plausible_delay(self):
        record = _record(EffectKind.INVENTORY_FELL, 120, {"item": "iron-ore", "before": 5})
        observe(record, {"inventory": {"iron-ore": 5}}, 150)
        assert record.observed is EffectState.PENDING, "gave up inside its own window"

    def test_it_reports_failure_after_its_declared_deadline(self):
        record = _record(EffectKind.INVENTORY_FELL, 120, {"item": "iron-ore", "before": 5})
        observe(record, {"inventory": {"iron-ore": 5}}, 400)
        assert record.observed is EffectState.TIMED_OUT

    def test_a_refusal_is_not_a_timeout(self):
        """The engine declining is a different fact from the world not following.

        A controller that conflates them retries the wrong thing.
        """
        record = _record(EffectKind.ENTITY_PRESENT, 0, {"position": [3.5, 0.5]})
        record.refuse("collision", 100)
        assert record.observed is EffectState.REFUSED
        assert record.error == "collision"
        # And a settled record is not re-judged by a later observation.
        observe(record, {"entities": [{"p": [3.5, 0.5]}]}, 200)
        assert record.observed is EffectState.REFUSED


class TestOnlyTheDeclaredEffectIsChecked:
    def test_a_placement_confirms_presence(self):
        record = _record(EffectKind.ENTITY_PRESENT, 0, {"position": [3.5, 0.5]})
        observe(record, {"entities": [{"p": [3.5, 0.5], "d": 4}]}, 100)
        assert record.observed is EffectState.CONFIRMED

    def test_presence_says_nothing_about_facing(self):
        """§5: a completed placement does not establish orientation."""
        present = _record(EffectKind.ENTITY_PRESENT, 0, {"position": [3.5, 0.5]})
        facing = _record(EffectKind.ENTITY_FACING, 0, {"position": [3.5, 0.5], "direction": 4})
        world = {"entities": [{"p": [3.5, 0.5], "d": 0}]}  # placed, facing north
        observe(present, world, 100)
        observe(facing, world, 100)
        assert present.observed is EffectState.CONFIRMED
        assert facing.observed is EffectState.TIMED_OUT, "wrong facing must not confirm"

    def test_a_transfer_confirms_only_that_items_moved(self):
        record = _record(EffectKind.ITEMS_MOVED, 0, {"item": "coal", "before": 5})
        observe(record, {"inventory": {"coal": 4}}, 100)
        assert record.observed is EffectState.CONFIRMED

    def test_an_unchanged_inventory_does_not_confirm_a_transfer(self):
        record = _record(EffectKind.ITEMS_MOVED, 0, {"item": "coal", "before": 5})
        observe(record, {"inventory": {"coal": 5}}, 100)
        assert record.observed is not EffectState.CONFIRMED

    def test_movement_is_checked_against_the_previous_position(self):
        moved = _record(EffectKind.CHARACTER_MOVED, 60, {"position": [0.0, 0.0]})
        observe(moved, {"character": {"position": [4.45, 0.0]}}, 130)
        assert moved.observed is EffectState.CONFIRMED

        stuck = _record(EffectKind.CHARACTER_MOVED, 60, {"position": [0.0, 0.0]})
        observe(stuck, {"character": {"position": [0.0, 0.0]}}, 200)
        assert stuck.observed is EffectState.TIMED_OUT


class TestUnknownIsAFirstClassAnswer:
    def test_a_verb_with_no_observable_effect_reports_unknown(self):
        """Better than claiming a consequence nothing can check."""
        record = _record(EffectKind.NONE)
        observe(record, {}, 100)
        assert record.observed is EffectState.UNKNOWN

    def test_set_recipe_declares_no_local_effect(self):
        """A furnace auto-selects, so a confirmation would be a fiction."""
        assert expectation_for("set_recipe").kind is EffectKind.NONE

    def test_an_unmapped_verb_does_not_invent_an_expectation(self):
        assert expectation_for("no-such-verb").kind is EffectKind.NONE


class TestTheRecordShape:
    def test_predicted_and_observed_never_share_a_field(self):
        record = _record(EffectKind.ENTITY_PRESENT, 0, {"position": [1.5, 1.5]})
        payload = record.to_dict()
        assert payload["expected_effect"] == "entity_present"
        assert payload["observed"] == "pending"
        assert "expected_detail" in payload and "observed_tick" in payload

    def test_a_pending_record_is_not_settled(self):
        assert not _record(EffectKind.ENTITY_PRESENT, 99, {"position": [0, 0]}).settled

    @pytest.mark.parametrize("verb", sorted(EXPECTATIONS))
    def test_every_declared_verb_has_a_kind_and_a_nonnegative_deadline(self, verb):
        expectation = EXPECTATIONS[verb]
        assert isinstance(expectation.kind, EffectKind)
        assert expectation.deadline_ticks >= 0

    def test_a_delayed_verb_has_a_window_and_an_instant_one_does_not(self):
        """A monitor with a zero window on a slow verb calls every success a
        failure; one with a long window on an instant verb never reports."""
        assert expectation_for("mine").deadline_ticks > 0
        assert expectation_for("transfer").deadline_ticks == 0
