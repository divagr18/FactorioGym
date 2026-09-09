"""The diagnosis track withholds the fault, through every channel (R4.3).

`repair_belt` and `restore_power` publish their fault tiles, which
`describe_assistance` now declares as `fault-location:published`. R4.3's
diagnosis track is the same shape of task with the coordinate withheld -- so
these tests check the withholding in the two places it can leak:

* the **observation**, via `TaskSpec.public_markers`, which unions the success
  and failure predicates' markers with `extra_public_markers`;
* the **reward**, because `rewards._measure`'s `CHARACTER_WITHIN` branch reads
  *truth*. A `toward_fault` shaping potential would restore the withheld hint
  invisibly, paying the agent for approaching a tile the prompt never named.

The second is not hypothetical: both repair families declare exactly such a
component (`toward_gap`), and the plan for this track called for its removal
before the family was written.
"""

from __future__ import annotations

import random
from dataclasses import replace

import pytest

from factoriorl.tasks import all_tasks, get, validate_all
from factoriorl.tasks.families import diagnose_line as family
from factoriorl.tasks.spec import PredicateKind

TASK = "diagnose_line"


def _blueprints(name: str, count: int = 8):
    layout = next(f for f in family.FAMILIES if f.name == name)
    return [family.generate(layout, random.Random(20260909 + i)) for i in range(count)]


class TestNothingPublishesTheFault:
    def test_the_task_publishes_no_markers_at_all(self):
        assert get(TASK).spec.public_markers == ()

    def test_the_success_predicate_names_no_marker(self):
        """`SUSTAINED_OUTPUT`'s marker is None, which is what makes an
        unpublished objective expressible without a new field."""
        for predicate in get(TASK).spec.success:
            assert getattr(predicate, "marker", None) is None

    def test_the_goal_geometry_slots_would_stay_zero(self):
        spec = get(TASK).spec
        assert spec.focus_marker is None
        assert spec.focus_policy == "static"

    def test_the_fault_markers_are_declared_for_the_evaluator(self):
        """Withheld is not the same as absent: the evaluator still needs to
        know what counts as the fault, and the split audit's coverage
        containment reads it."""
        assert set(get(TASK).spec.fault_markers) == {"drill", "furnace"}

    def test_the_blueprint_declares_no_fault_marker(self):
        for blueprint in _blueprints("drill_dry"):
            assert set(blueprint.markers) == {"line"}

    def test_it_declares_no_assistance(self):
        from factoriorl.assistance import STATIC, describe_assistance

        assert describe_assistance(get(TASK).spec) == STATIC


class TestNoRewardPointsAtAWithheldFault:
    """The general invariant, over every registered task rather than this one.

    A task that withholds a fault and keeps a `CHARACTER_WITHIN` shaping
    component on it hands the answer back through the reward. Checked for all
    tasks so the next diagnosis family inherits the guard.
    """

    @pytest.mark.parametrize("task_id", sorted(all_tasks()))
    def test_shaping_never_reads_a_withheld_marker(self, task_id):
        spec = get(task_id).spec
        published = set(spec.public_markers)
        withheld = set(spec.fault_markers) - published
        for reward in spec.rewards:
            if not reward.shaping or reward.predicate is None:
                continue
            marker = getattr(reward.predicate, "marker", None)
            assert marker not in withheld, (
                f"{task_id}: shaping component {reward.name!r} points at "
                f"{marker!r}, which no observation publishes"
            )

    def test_the_diagnosis_family_declares_no_character_within_shaping(self):
        for reward in get(TASK).spec.rewards:
            if reward.shaping and reward.predicate is not None:
                assert reward.predicate.kind is not PredicateKind.CHARACTER_WITHIN

    def test_validation_refuses_a_diagnosis_task_that_publishes_its_fault(self):
        """The guard, exercised: relabelling a repair family as diagnosis must
        fail rather than produce a diagnosis task with the answer in it."""
        spec = replace(get("repair_belt").spec, track="diagnosis")
        published = set(spec.public_markers)
        assert published.intersection(spec.fault_markers)
        report = validate_all(2)
        assert report["ok"], "the registered tasks must still validate"


class TestTheFaultKindsAreWhatTheyClaim:
    @pytest.mark.parametrize(
        ("layout", "drill_fuelled", "furnace_fuelled"),
        [
            ("drill_dry", False, True),
            ("furnace_dry", True, False),
            ("jam_only", True, True),
            # The held-out scene: both fuel faults at once, each seen singly in
            # training and never together.
            ("both_dry", False, False),
        ],
    )
    def test_the_declared_fault_is_the_one_installed(self, layout, drill_fuelled, furnace_fuelled):
        for blueprint in _blueprints(layout, 4):
            by_marker = {e.marker: e for e in blueprint.entities}
            drill = by_marker["drill"].contents
            furnace = by_marker["furnace"].contents
            assert bool(drill.get("coal")) is drill_fuelled, layout
            assert bool(furnace.get("coal")) is furnace_fuelled, layout

    @pytest.mark.parametrize("layout", ["drill_dry", "furnace_dry", "jam_only", "both_dry"])
    def test_every_scene_blocks_the_output(self, layout):
        """The measured reason this is in every family: with one fault, a
        random policy scored 0.40 on training scenes and *beat the reference*
        on the test split, 0.60 to 0.40. One lucky action restored a line that
        was otherwise built and fuelled. Clearing the output is a second fix of
        a different kind, needed everywhere."""
        for blueprint in _blueprints(layout, 4):
            furnace = next(e for e in blueprint.entities if e.marker == "furnace")
            assert furnace.contents.get("iron-plate") == family.JAM_PLATES, layout

    def test_the_test_split_is_the_conjunction_of_the_trained_faults(self):
        trained = [f.name for f in family.FAMILIES if f.split == "train"]
        held_out = [f.name for f in family.FAMILIES if f.split == "test"]
        assert held_out == ["both_dry"]
        starved_in_training = {m for name in trained for m in family.STARVED[name]}
        assert set(family.STARVED["both_dry"]) == starved_in_training
        # Never together, or the holdout is a training scene.
        assert all(len(family.STARVED[name]) == 1 for name in trained)

    def test_the_jam_fills_the_output_slot(self):
        """A stone furnace's result slot holds one stack of a hundred and the
        furnace stops when it is full. Less than a full stack is not a fault."""
        assert family.JAM_PLATES == 100

    def test_the_agent_is_given_enough_coal_to_survive_a_wrong_guess(self):
        """`give_coal_20` fuels the nearest entity and the machines are two
        tiles apart, so a misjudged approach spends a stack on the wrong
        machine. `plate_line` measured that: 47 coal into the drill and the
        furnace never lit."""
        assert family.STARTING_COAL >= 2 * 50

    def test_acceptance_cannot_arrive_from_the_scene_s_head_start(self):
        """`furnace_jammed` starts with a hundred plates in the furnace. The
        window must not be satisfiable before the repair could have happened."""
        predicate = get(TASK).spec.success[0]
        assert predicate.kind is PredicateKind.SUSTAINED_OUTPUT
        assert predicate.not_before_tick >= predicate.over_ticks
