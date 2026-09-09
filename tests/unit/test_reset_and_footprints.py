"""R3 package 4: the two validators that existed only as a docstring sentence.

`validate_all`'s docstring claimed it checked "a reachable goal, and --
importantly -- the success predicate must be False at reset". Grep found only
the sentence. `footprint_conflicts` compared `round()` of one centre tile per
entity, so two adjacent 2x2 machines whose footprints overlap passed
validation -- the exact geometry `plate_line` reasons about by hand and
`build_line` had to measure on an engine.
"""

from __future__ import annotations

import random

import pytest

from factoriorl.tasks import all_tasks, get, validate_all
from factoriorl.tasks.spec import (
    ENTITY_TILE_SIZES,
    UNDECIDABLE_AT_RESET,
    Blueprint,
    EntitySpec,
    LayoutFamily,
    Predicate,
    PredicateKind,
    ResourceSpec,
    RewardComponent,
    RewardKind,
    TaskSpec,
    entity_tiles,
)

#: Read off a real engine (2.0.60) via `prototypes.entity[name].tile_width /
#: tile_height`, for every prototype the generators actually place.
MEASURED_FOOTPRINTS = {
    "burner-inserter": (1, 1),
    "burner-mining-drill": (2, 2),
    "electric-mining-drill": (3, 3),
    "small-electric-pole": (1, 1),
    "solar-panel": (3, 3),
    "stone-furnace": (2, 2),
    "stone-wall": (1, 1),
    "transport-belt": (1, 1),
    "wooden-chest": (1, 1),
}


class TestFootprints:
    @pytest.mark.parametrize(("name", "size"), sorted(MEASURED_FOOTPRINTS.items()))
    def test_the_table_matches_the_engine(self, name, size):
        assert ENTITY_TILE_SIZES.get(name, (1, 1)) == size

    def test_every_prototype_the_generators_place_is_covered(self):
        """A prototype missing from the table is silently 1x1 -- which is how
        `burner-mining-drill`, a 2x2, was treated before."""
        placed = set()
        for name in all_tasks():
            task = get(name)
            for family in task.spec.layout_families:
                for seed in range(6):
                    blueprint = task.generate(family, random.Random(seed))
                    placed |= {e.name for e in blueprint.entities}
        assert placed <= set(MEASURED_FOOTPRINTS), placed - set(MEASURED_FOOTPRINTS)

    def test_a_2x2_covers_four_tiles_around_its_centre(self):
        assert set(entity_tiles("stone-furnace", (1.0, 3.0))) == {
            (0, 2),
            (1, 2),
            (0, 3),
            (1, 3),
        }

    def test_a_3x3_is_centred_on_its_own_tile(self):
        tiles = set(entity_tiles("solar-panel", (0.0, 0.0)))
        assert len(tiles) == 9 and (0, 0) in tiles and (-1, -1) in tiles

    def test_two_adjacent_half_tile_centres_stay_distinct(self):
        """The trap `plate_line` documents: Python's `round()` is banker's
        rounding, so `round(-2.5)` and `round(-1.5)` are both -2 and a column
        of walls spaced on .5 boundaries collides with itself in pairs.
        `floor(p - w/2 + 0.5)` keeps them apart."""
        assert entity_tiles("stone-wall", (-2.5, 0.0)) == [(-3, 0)]
        assert entity_tiles("stone-wall", (-1.5, 0.0)) == [(-2, 0)]
        assert round(-2.5) == round(-1.5) == -2


class TestOverlapIsSeenNow:
    def test_two_adjacent_2x2_machines_that_overlap_are_reported(self):
        """The case the old check could not see: distinct centre tiles."""
        blueprint = Blueprint(
            entities=(
                EntitySpec("burner-mining-drill", (1.0, 1.0)),
                # Centre (1, 2) is a different tile from (1, 1), so the old
                # centre-tile comparison passed. The footprints share y = 1.
                EntitySpec("stone-furnace", (1.0, 2.0)),
            )
        )
        problems = blueprint.footprint_conflicts()
        assert problems, "an overlapping 2x2 pair was not reported"
        assert "stone-furnace" in problems[0]

    def test_the_alignment_build_line_actually_uses_is_not_reported(self):
        """Measured as buildable and productive: drill (1,1), furnace (1,3)."""
        blueprint = Blueprint(
            entities=(
                EntitySpec("burner-mining-drill", (1.0, 1.0), direction="south"),
                EntitySpec("stone-furnace", (1.0, 3.0)),
            )
        )
        assert blueprint.footprint_conflicts() == []

    def test_two_1x1_entities_on_one_tile_are_still_reported(self):
        blueprint = Blueprint(
            entities=(
                EntitySpec("wooden-chest", (0.5, 0.5)),
                EntitySpec("stone-wall", (0.5, 0.5)),
            )
        )
        assert blueprint.footprint_conflicts()

    def test_no_generator_produces_an_overlap_under_real_footprints(self):
        for name in all_tasks():
            task = get(name)
            for family in task.spec.layout_families:
                for seed in range(16):
                    blueprint = task.generate(family, random.Random(seed))
                    assert blueprint.footprint_conflicts() == [], f"{name}/{family.name}[{seed}]"


class TestInitialState:
    def test_the_cumulative_channels_are_empty(self):
        """Cleared by `world.clear_statistics` at every episode start, so this
        is empty by construction and not by assumption."""
        _, truth = Blueprint().initial_state()
        assert truth["produced"] == {}
        assert truth["built"] == {}
        assert truth["window"] == []

    def test_container_contents_come_from_the_blueprint(self):
        blueprint = Blueprint(
            entities=(
                EntitySpec("wooden-chest", (2.5, 0.5), marker="dst", contents={"iron-plate": 25}),
            )
        )
        _, truth = blueprint.initial_state()
        assert truth["containers"]["dst"] == {"iron-plate": 25}

    def test_a_marked_entity_publishes_its_position_as_a_marker(self):
        blueprint = Blueprint(entities=(EntitySpec("wooden-chest", (4.5, -1.5), marker="dst"),))
        _, truth = blueprint.initial_state()
        assert truth["markers"]["dst"] == [4.5, -1.5]

    def test_the_character_and_its_inventory_are_visible(self):
        blueprint = Blueprint(character_position=(3.0, 4.0), character_inventory={"coal": 7})
        observation, _ = blueprint.initial_state()
        assert observation["character"]["position"] == [3.0, 4.0]
        assert observation["inventory"] == {"coal": 7}

    def test_resources_appear_as_tiles_not_entities(self):
        blueprint = Blueprint(resources=(ResourceSpec("iron-ore", (1.0, 1.0)),))
        observation, _ = blueprint.initial_state()
        assert observation["resources"]["tiles"] == [{"name": "iron-ore", "p": [1.0, 1.0]}]
        assert observation["entities"] == []


class TestTheResetCheckActuallyRuns:
    def _scene_that_is_already_solved(self):
        return TaskSpec(
            id="already_solved",
            version="1.0.0",
            description="d",
            # `TaskSpec.track` has no valid default, so every fixture
            # declares one -- that is the point of the default being
            # refused rather than guessed.
            track="production",
            layout_families=(LayoutFamily("f", "train"), LayoutFamily("g", "test")),
            success=(
                Predicate(
                    PredicateKind.CONTAINER_HOLDS, marker="dst", item="iron-plate", at_least=20
                ),
            ),
            rewards=(RewardComponent("s", RewardKind.SPARSE_SUCCESS),),
        )

    def test_a_scene_that_already_satisfies_success_is_a_problem(self, monkeypatch):
        import factoriorl.tasks as tasks_module

        spec = self._scene_that_is_already_solved()

        def generate(family, rng):
            return Blueprint(
                entities=(
                    EntitySpec(
                        "wooden-chest",
                        (2.5, 0.5),
                        marker="dst",
                        contents={"iron-plate": 25},
                    ),
                )
            )

        monkeypatch.setattr(
            tasks_module,
            "all_tasks",
            lambda: {"already_solved": tasks_module.RegisteredTask(spec, generate)},
        )
        report = validate_all(sample_seeds=2)
        assert not report["ok"]
        problems = report["tasks"]["already_solved"]["problems"]
        assert any("already satisfied at reset" in p for p in problems), problems

    def test_the_same_scene_one_plate_short_is_accepted(self, monkeypatch):
        import factoriorl.tasks as tasks_module

        spec = self._scene_that_is_already_solved()

        def generate(family, rng):
            return Blueprint(
                entities=(
                    EntitySpec(
                        "wooden-chest",
                        (2.5, 0.5),
                        marker="dst",
                        contents={"iron-plate": 19},
                    ),
                )
            )

        monkeypatch.setattr(
            tasks_module,
            "all_tasks",
            lambda: {"already_solved": tasks_module.RegisteredTask(spec, generate)},
        )
        report = validate_all(sample_seeds=2)
        assert report["tasks"]["already_solved"]["problems"] == []

    def test_no_registered_task_is_solved_by_its_own_scene(self):
        report = validate_all(sample_seeds=16)
        for name, entry in report["tasks"].items():
            offenders = [p for p in entry["problems"] if "already satisfied" in p]
            assert offenders == [], f"{name}: {offenders}"


class TestTheCheckDeclaresWhatItCannotDecide:
    def test_working_is_named_as_undecidable(self):
        assert PredicateKind.ENTITY_WORKING in UNDECIDABLE_AT_RESET
        assert PredicateKind.ANY_WORKING in UNDECIDABLE_AT_RESET

    def test_the_production_kinds_are_decidable(self):
        for kind in (
            PredicateKind.PRODUCED,
            PredicateKind.SUSTAINED_OUTPUT,
            PredicateKind.BUILT,
            PredicateKind.CONTAINER_HOLDS,
            PredicateKind.INVENTORY_HOLDS,
            PredicateKind.CHARACTER_WITHIN,
        ):
            assert kind not in UNDECIDABLE_AT_RESET

    def test_the_report_says_which_tasks_the_check_covers(self):
        report = validate_all(sample_seeds=2)
        covered = {name for name, e in report["tasks"].items() if e["reset_check_covers_success"]}
        uncovered = set(report["tasks"]) - covered
        # `restore_power`'s only clause is `entity_working`, so the check is
        # not claimed for it. Every other task is covered.
        assert uncovered == {"restore_power"}, uncovered
        assert report["tasks"]["restore_power"]["undecidable_success"]

    def test_an_all_undecidable_task_is_not_silently_passed(self, monkeypatch):
        """It must be reported as uncovered, not counted as verified."""
        import factoriorl.tasks as tasks_module

        spec = TaskSpec(
            id="only_working",
            version="1.0.0",
            description="d",
            # `TaskSpec.track` has no valid default, so every fixture
            # declares one -- that is the point of the default being
            # refused rather than guessed.
            track="production",
            layout_families=(LayoutFamily("f", "train"), LayoutFamily("g", "test")),
            success=(Predicate(PredicateKind.ENTITY_WORKING, marker="drill"),),
            rewards=(RewardComponent("s", RewardKind.SPARSE_SUCCESS),),
        )
        monkeypatch.setattr(
            tasks_module,
            "all_tasks",
            lambda: {"only_working": tasks_module.RegisteredTask(spec, lambda f, r: Blueprint())},
        )
        report = validate_all(sample_seeds=1)
        entry = report["tasks"]["only_working"]
        assert entry["reset_check_covers_success"] is False
        assert entry["undecidable_success"]


class TestAMixedConjunctionIsJudgedSoundly:
    """One decidable clause that is False settles the whole conjunction.

    Reporting a mixed task as unverified would have called `build_line`
    unverified the moment it gained a `working` clause, while its `BUILT`
    clause is provably False at reset.
    """

    def _spec(self, *success):
        return TaskSpec(
            id="mixed",
            version="1.0.0",
            description="d",
            # `TaskSpec.track` has no valid default, so every fixture
            # declares one -- that is the point of the default being
            # refused rather than guessed.
            track="production",
            layout_families=(LayoutFamily("f", "train"), LayoutFamily("g", "test")),
            success=success,
            rewards=(RewardComponent("s", RewardKind.SPARSE_SUCCESS),),
        )

    def _report(self, monkeypatch, spec, generate):
        import factoriorl.tasks as tasks_module

        monkeypatch.setattr(
            tasks_module,
            "all_tasks",
            lambda: {"mixed": tasks_module.RegisteredTask(spec, generate)},
        )
        return validate_all(sample_seeds=2)["tasks"]["mixed"]

    def test_a_false_decidable_clause_makes_the_verdict_conclusive(self, monkeypatch):
        spec = self._spec(
            Predicate(PredicateKind.BUILT, item="stone-furnace", at_least=1),
            Predicate(PredicateKind.ANY_WORKING, item="stone-furnace", at_least=1),
        )
        entry = self._report(monkeypatch, spec, lambda f, r: Blueprint())
        assert entry["reset_check_covers_success"] is True
        assert entry["undecidable_success"]
        assert entry["problems"] == []

    def test_all_decidable_clauses_true_plus_an_undecidable_one_is_not_claimed(self, monkeypatch):
        """The scene may or may not be solved, and the check must say so."""
        spec = self._spec(
            Predicate(PredicateKind.CONTAINER_HOLDS, marker="dst", item="iron-plate", at_least=1),
            Predicate(PredicateKind.ANY_WORKING, item="stone-furnace", at_least=1),
        )

        def generate(family, rng):
            return Blueprint(
                entities=(
                    EntitySpec(
                        "wooden-chest", (2.5, 0.5), marker="dst", contents={"iron-plate": 5}
                    ),
                )
            )

        entry = self._report(monkeypatch, spec, generate)
        assert entry["reset_check_covers_success"] is False
        # Not a problem: unproven is not the same as proven-bad.
        assert entry["problems"] == []

    def test_all_clauses_decidable_and_true_is_still_a_problem(self, monkeypatch):
        spec = self._spec(
            Predicate(PredicateKind.CONTAINER_HOLDS, marker="dst", item="iron-plate", at_least=1),
        )

        def generate(family, rng):
            return Blueprint(
                entities=(
                    EntitySpec(
                        "wooden-chest", (2.5, 0.5), marker="dst", contents={"iron-plate": 5}
                    ),
                )
            )

        entry = self._report(monkeypatch, spec, generate)
        assert any("already satisfied at reset" in p for p in entry["problems"])
