"""Synthesis section 9: held-out *combinations* with covered marginals.

The existing overlap rule cannot express this and in fact inverts it: a 12-bin
histogram over {1} against one over {2} scores 0.0 overlap, which
`SEPARATION_MAX_OVERLAP` reads as good separation. Both distribution surprises
this repo had were exactly that shape, so the rule that catches them is set
containment.

The current generators pass, because v1.6.0 fixed both. So the rule's power is
demonstrated on synthetic generators reproducing the *documented pre-fix
shapes* -- otherwise "the check passes" would be indistinguishable from "the
check is vacuous".
"""

from __future__ import annotations

import hashlib
import json

import pytest

from factoriorl.seeding import SeedPlan
from factoriorl.tasks import all_tasks, coverage, get
from factoriorl.tasks.spec import (
    Blueprint,
    EntitySpec,
    LayoutFamily,
    Predicate,
    PredicateKind,
    RewardComponent,
    RewardKind,
    TaskSpec,
)

PLAN = SeedPlan(master=777, run_id="coverage-test")


class _Task:
    """A registered task without the registry."""

    def __init__(self, spec, generate):
        self.spec = spec
        self.generate = generate


def _chain(gap_index: int, poles: int = 6, gaps: tuple[int, ...] = ()) -> Blueprint:
    """A pole chain along y=0 with a drill at the end and holes at `gaps`."""
    holes = set(gaps) if gaps else {gap_index}
    entities = [
        EntitySpec("small-electric-pole", (float(4 * i), 0.0))
        for i in range(poles)
        if i not in holes
    ]
    entities.append(EntitySpec("electric-mining-drill", (float(4 * poles + 2), 0.0)))
    markers = {"gap": (float(4 * min(holes)), 0.0)}
    if len(holes) > 1:
        markers["gap2"] = (float(4 * sorted(holes)[1]), 0.0)
    return Blueprint(entities=tuple(entities), markers=markers)


def _spec(fault_markers=("gap", "gap2"), extra=("gap", "gap2")):
    return TaskSpec(
        id="synthetic",
        version="1.0.0",
        description="d",
        layout_families=(
            LayoutFamily("train_family", "train"),
            LayoutFamily("test_family", "test"),
        ),
        success=(Predicate(PredicateKind.ENTITY_WORKING, marker="drill"),),
        rewards=(RewardComponent("s", RewardKind.SPARSE_SUCCESS),),
        extra_public_markers=extra,
        fault_markers=fault_markers,
    )


class TestTheDescriptorSeesInteriorVersusTerminal:
    def test_an_interior_gap_is_flanked_on_both_sides(self):
        blueprint = _chain(gap_index=3)
        assert coverage.flanking_sides(blueprint, (12.0, 0.0)) == 2

    def test_a_terminal_gap_is_flanked_on_one_side(self):
        """The drill sits beyond the last pole, which is why counting *any*
        collinear entity scored this 2 and hid the whole distinction."""
        blueprint = _chain(gap_index=5)
        assert coverage.flanking_sides(blueprint, (20.0, 0.0)) == 1

    def test_the_real_generators_are_separated_too(self):
        """Not only the fixture: measured on `restore_power` itself."""
        import random

        task = get("restore_power")
        interior = task.generate(
            next(f for f in task.spec.layout_families if f.name == "pole_gap"),
            random.Random(3),
        )
        terminal = task.generate(
            next(f for f in task.spec.layout_families if f.name == "gap_near_drill"),
            random.Random(3),
        )
        assert coverage.coverage_cell(task, interior)[1] == 2
        assert coverage.coverage_cell(task, terminal)[1] == 1

    def test_a_fault_with_no_chain_beside_it_scores_zero(self):
        lone = Blueprint(entities=(), markers={"gap": (0.0, 0.0)})
        assert coverage.flanking_sides(lone, (0.0, 0.0)) == 0


class TestTheRuleReproducesBothDocumentedSurprises:
    def test_interior_training_against_a_terminal_holdout_fails(self):
        """`restore_power` before v1.6.0: train interior 600/600, test terminal
        600/600."""
        spec = _spec()

        def generate(family, rng):
            return _chain(gap_index=3 if family.split == "train" else 5)

        result = coverage.analyse(_Task(spec, generate), 20, PLAN)
        assert not result["contained"]
        assert not result["vacuous"]
        cell = result["held_out_cells"][0]["cell"]
        assert cell["fault_neighbours"] == 1
        assert result["held_out_cells"][0]["test_count"] == 20

    def test_one_fault_training_against_a_two_fault_holdout_fails(self):
        """`repair_belt` before v1.6.0: train one gap, test two."""
        spec = _spec()

        def generate(family, rng):
            if family.split == "train":
                return _chain(gap_index=2)
            return _chain(gap_index=2, gaps=(2, 3))

        result = coverage.analyse(_Task(spec, generate), 20, PLAN)
        assert not result["contained"]
        assert result["held_out_cells"][0]["cell"]["fault_count"] == 2

    def test_the_failure_names_a_scene_to_look_at(self):
        spec = _spec()

        def generate(family, rng):
            return _chain(gap_index=3 if family.split == "train" else 5)

        result = coverage.analyse(_Task(spec, generate), 8, PLAN)
        messages = coverage.failures("synthetic", result)
        assert len(messages) == 1
        example = result["held_out_cells"][0]["example"]
        assert example["blueprint_digest"] in messages[0]
        assert str(example["episode_index"]) in messages[0]

    def test_a_holdout_whose_combination_is_trained_on_passes(self):
        """The v1.6.0 shape: training samples both regimes."""
        spec = _spec()

        def generate(family, rng):
            if family.split == "train":
                # Both interior and terminal, the way v1.6.0 samples.
                return _chain(gap_index=5 if rng.random() < 0.3 else 3)
            return _chain(gap_index=5)

        result = coverage.analyse(_Task(spec, generate), 40, PLAN)
        assert result["contained"]
        trained = {tuple(row["cell"].values()) for row in result["train_cells"]}
        assert len(trained) == 2, "the fixture must train on both regimes"


class TestOverlapCouldNotHaveCaughtIt:
    def test_disjoint_singletons_score_perfect_separation(self):
        """Why this is a new rule and not a retuned threshold."""
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
        import generator_diagnostics as gd

        score = gd.overlap_coefficient([1.0] * 100, [2.0] * 100, bins=12)
        assert score == 0.0
        assert score <= gd.SEPARATION_MAX_OVERLAP, (
            "the separation rule reads total disjointness as good, which is the "
            "shape both surprises had"
        )


class TestTheRegisteredTasks:
    def test_every_task_is_analysed_without_error(self):
        for name in sorted(all_tasks()):
            result = coverage.analyse(get(name), 20, PLAN)
            assert "contained" in result and "vacuous" in result

    @pytest.mark.parametrize("name", ["repair_belt", "restore_power"])
    def test_the_two_fixed_families_are_contained_and_not_vacuous(self, name):
        """The v1.6.0 fixes, verified by the rule rather than by their commit
        messages."""
        result = coverage.analyse(get(name), 100, PLAN)
        assert not result["vacuous"], f"{name} must declare fault_markers"
        assert result["contained"], [c["cell"] for c in result["held_out_cells"]]

    @pytest.mark.parametrize("name", ["repair_belt", "restore_power"])
    def test_training_actually_covers_the_tested_cell_in_a_useful_share(self, name):
        """Containment with one training scene in a thousand would pass while
        teaching nothing, so the share is asserted, not just the set."""
        result = coverage.analyse(get(name), 200, PLAN)
        train = {tuple(r["cell"].values()): r["count"] for r in result["train_cells"]}
        total = sum(train.values())
        for row in result["test_cells"]:
            share = train[tuple(row["cell"].values())] / total
            assert share >= 0.10, f"{name} trains the tested cell in only {share:.1%}"

    def test_a_task_declaring_no_fault_markers_is_reported_as_vacuous(self):
        """A vacuous pass must not read as a verified one."""
        result = coverage.analyse(get("navigate"), 10, PLAN)
        assert result["vacuous"] and result["contained"]
        assert result["declared_fault_markers"] == []


class TestIdentityAndHygiene:
    def test_fault_markers_are_not_part_of_the_task_identity(self):
        """Measurement metadata, like `difficulty_marker`: declaring it must not
        bump a version or move a frozen digest."""
        for name in sorted(all_tasks()):
            assert "fault_markers" not in get(name).spec.to_dict()

    def test_the_digest_matches_the_one_the_environment_installs_by(self):
        from factoriorl.env import blueprint_digest

        payload = Blueprint(entities=(EntitySpec("wooden-chest", (0.5, 0.5)),)).to_dict()
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
        assert coverage._digest(payload) == expected == blueprint_digest(payload)

    def test_the_module_does_not_import_the_tools_directory(self):
        import pathlib

        source = pathlib.Path("src/factoriorl/tasks/coverage.py").read_text(encoding="utf-8")
        assert "generator_diagnostics" not in source.replace("`tools/generator_diagnostics.py`", "")
