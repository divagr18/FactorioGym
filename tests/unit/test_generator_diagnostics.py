"""Engine-free coverage for the split-content diagnostics (PLAN.md section 3).

The tool exists because a gate that compares layout-family *names* passed over
three real defects: two families that were byte-identical copies of another, and
one family whose generator had no branch at all. These tests protect the three
properties that detection rests on -- a digest that is stable across processes
and sensitive to content, an overlap measure with the right endpoints, and a
duplicate-family check that actually fires.

The duplicate case is built from a synthetic task rather than from the six
registered families on purpose: those definitions are being edited, and a
regression test that depends on today's family list would report a maintainer's
in-progress edit as this tool breaking.
"""

from __future__ import annotations

import random
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import generator_diagnostics as gd  # noqa: E402

from factoriorl.tasks import RegisteredTask  # noqa: E402
from factoriorl.tasks.spec import (  # noqa: E402
    Blueprint,
    EntitySpec,
    LayoutFamily,
    Predicate,
    PredicateKind,
    RewardComponent,
    RewardKind,
    TaskSpec,
)


def _blueprint(goal=(10.0, 0.0), extra=()) -> Blueprint:
    return Blueprint(
        entities=(EntitySpec("wooden-chest", goal, marker="goal"), *extra),
        character_position=(0.0, 0.0),
        markers={"goal": goal},
    )


# ------------------------------------------------------------------- digest


def test_digest_is_stable_across_processes():
    """`hash()` is randomised per PYTHONHASHSEED and this repo has been bitten by it.

    A digest that changes between processes would make "distinct scenes" a count
    of nothing, and -- worse -- would make the duplicate-family check pass or
    fail at random, which hides a defect instead of reporting one. Two fresh
    interpreters with different hash seeds must agree.
    """
    program = (
        "import sys; sys.path.insert(0, r'%s')\n"
        "import generator_diagnostics as gd\n"
        "from factoriorl.tasks.spec import Blueprint, EntitySpec\n"
        "bp = Blueprint(entities=(EntitySpec('wooden-chest', (3.0, 4.0), marker='goal'),),\n"
        "               markers={'goal': (3.0, 4.0)})\n"
        "print(gd.scene_digest(bp))\n" % (ROOT / "tools")
    )
    env_paths = str(ROOT / "src")
    digests = set()
    for seed in ("0", "1", "12345"):
        result = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            check=True,
            env={"PYTHONHASHSEED": seed, "PYTHONPATH": env_paths, "PATH": ""},
        )
        digests.add(result.stdout.strip())
    assert len(digests) == 1, f"digest is not process-stable: {digests}"
    assert digests.pop() == gd.scene_digest(
        Blueprint(
            entities=(EntitySpec("wooden-chest", (3.0, 4.0), marker="goal"),),
            markers={"goal": (3.0, 4.0)},
        )
    )


@pytest.mark.parametrize(
    "changed",
    [
        Blueprint(entities=(EntitySpec("wooden-chest", (3.0, 5.0)),)),
        Blueprint(entities=(EntitySpec("iron-chest", (3.0, 4.0)),)),
        Blueprint(entities=(EntitySpec("wooden-chest", (3.0, 4.0), direction="south"),)),
        Blueprint(entities=(EntitySpec("wooden-chest", (3.0, 4.0), contents={"coal": 1}),)),
        Blueprint(
            entities=(EntitySpec("wooden-chest", (3.0, 4.0)),),
            character_position=(1.0, 0.0),
        ),
        Blueprint(
            entities=(EntitySpec("wooden-chest", (3.0, 4.0)),),
            markers={"goal": (3.0, 4.0)},
        ),
    ],
)
def test_digest_is_sensitive_to_a_real_change(changed):
    """Position, name, direction, contents, spawn and markers all move the digest.

    A digest blind to any of these would count two genuinely different scenes as
    one and understate a family's scene count -- the opposite of the defect the
    tool hunts, and just as invisible.
    """
    base = Blueprint(entities=(EntitySpec("wooden-chest", (3.0, 4.0)),))
    assert gd.scene_digest(changed) != gd.scene_digest(base)


def test_digest_matches_the_digest_the_worker_installs_scenes_by():
    """`distinct_scenes` must mean "distinct scenes the environment would install".

    The tool recomputes the algorithm rather than importing `factoriorl.env`
    (which pulls in gymnasium and numpy), so the two can drift. This pins them.
    """
    pytest.importorskip("gymnasium")
    from factoriorl.env import blueprint_digest

    blueprint = _blueprint()
    assert gd.scene_digest(blueprint) == blueprint_digest(blueprint.to_dict())


# ------------------------------------------------------------------ overlap


def test_overlap_is_one_for_identical_distributions():
    values = [float(v) for v in range(200)]
    assert gd.overlap_coefficient(values, list(values)) == 1.0


def test_overlap_is_one_for_two_constant_samples_of_the_same_value():
    # Degenerate range: every observation is the same number. That is total
    # agreement on this descriptor; it is `distinct_scenes`, not the overlap,
    # that reports a generator with no variation at all.
    assert gd.overlap_coefficient([7.0] * 50, [7.0] * 50) == 1.0


def test_overlap_is_zero_for_disjoint_distributions():
    left = [float(v) for v in range(100)]
    right = [float(v) for v in range(1000, 1100)]
    assert gd.overlap_coefficient(left, right) == pytest.approx(0.0, abs=1e-9)


def test_overlap_is_symmetric_and_bounded():
    left = [float(v % 17) for v in range(300)]
    right = [float(v % 23) for v in range(300)]
    forward = gd.overlap_coefficient(left, right)
    backward = gd.overlap_coefficient(right, left)
    assert forward == backward
    assert 0.0 <= forward <= 1.0


def test_overlap_is_none_when_a_descriptor_was_never_measured():
    """None, not 1.0. An unmeasured descriptor reported as perfect agreement is
    exactly how an unchecked claim passes for a checked one."""
    assert gd.overlap_coefficient([None, None], [1.0, 2.0]) is None
    assert gd.overlap_coefficient([1.0, 2.0], []) is None


def test_overlap_ignores_unequal_sample_sizes():
    # Histograms are normalised before comparison, so 200 training samples
    # against 50 test samples is still an overlap of one when the two
    # distributions coincide.
    assert (
        gd.overlap_coefficient(
            [float(v % 10) for v in range(200)], [float(v % 10) for v in range(50)]
        )
        == 1.0
    )


def test_a_shifted_holdout_falls_below_the_parity_threshold():
    """The failure PLAN.md names: a holdout that is simply farther away."""
    train = [12.0 + (v % 60) / 10 for v in range(300)]
    harder = [v + 12.0 for v in train]
    assert gd.overlap_coefficient(train, harder) < gd.PARITY_MIN_OVERLAP


def test_same_distribution_sampling_noise_stays_above_the_parity_threshold():
    """The parity threshold must not fire on binning noise alone.

    Two independent samples of the same generator score below 1.0 purely because
    of finite binning; if that noise floor sat near the threshold the tool would
    report healthy splits as violations and be ignored.
    """
    rng = random.Random(4242)
    left = [rng.uniform(12.0, 30.0) for _ in range(200)]
    right = [rng.uniform(12.0, 30.0) for _ in range(200)]
    assert gd.overlap_coefficient(left, right) > gd.PARITY_MIN_OVERLAP


# ------------------------------------------------- duplicate layout families


def _synthetic_task(generate) -> RegisteredTask:
    """A task with two 'different' training families and one holdout."""
    spec = TaskSpec(
        id="synthetic",
        version="0.0.0",
        description="synthetic split fixture",
        layout_families=(
            LayoutFamily("plain", "train"),
            LayoutFamily("plain_far", "train"),
            LayoutFamily("screened", "test"),
        ),
        success=(Predicate(PredicateKind.CHARACTER_WITHIN, marker="goal", within=2.0),),
        rewards=(RewardComponent("arrived", RewardKind.SPARSE_SUCCESS, shaping=False),),
    )
    return RegisteredTask(spec=spec, generate=generate)


def test_two_identical_layout_families_are_detected():
    """The `gap_far == gap` and `pole_gap_far == pole_gap` regression.

    Both families were byte-identical copies of another: the name promised a
    distance variation the generator never implemented, so two of four declared
    families were the same content. A name comparison cannot see this; identical
    probe digests can.
    """

    def generate(family: LayoutFamily, rng) -> Blueprint:
        # `plain_far` promises a distance variation the generator never
        # implements, so it falls through to the `plain` layout unchanged.
        goal = (round(rng.uniform(12.0, 30.0), 1), 0.0)
        extra = ()
        if family.name == "screened":
            extra = (EntitySpec("stone-wall", (round(goal[0] / 2, 1), 0.0)),)
        return _blueprint(goal=goal, extra=extra)

    task = _synthetic_task(generate)
    plain = gd.probe_digests(task, LayoutFamily("plain", "train"))
    plain_far = gd.probe_digests(task, LayoutFamily("plain_far", "train"))
    assert plain == plain_far, "the fixture must actually be a duplicate"

    entry = gd.analyse_task_object(task, samples=40, plan=gd.SeedPlan(1, "test"))
    assert ["plain", "plain_far"] in entry["duplicate_families"]
    assert entry["verdict"] == "fail"
    assert any("identical content" in f for f in entry["failures"])


def test_genuinely_different_layout_families_are_not_flagged():
    """The check must not fire on families that differ, or it will be switched off."""

    def generate(family: LayoutFamily, rng) -> Blueprint:
        distance = rng.uniform(12.0, 30.0)
        if family.name == "plain_far":
            distance += 40.0
        goal = (round(distance, 1), 0.0)
        extra = ()
        if family.name == "screened":
            extra = tuple(
                EntitySpec("stone-wall", (goal[0] / 2, float(offset))) for offset in (-2, -1, 1, 2)
            )
        return _blueprint(goal=goal, extra=extra)

    task = _synthetic_task(generate)
    entry = gd.analyse_task_object(task, samples=40, plan=gd.SeedPlan(1, "test"))
    assert entry["duplicate_families"] == []


def test_a_holdout_admitting_one_scene_fails():
    """`restore_power.gap_near_drill` in miniature: a constant generator.

    "A holdout whose generator admits one scene is evaluated once, no matter how
    many episodes are run against it" (PLAN.md section 3).
    """

    def generate(family: LayoutFamily, rng) -> Blueprint:
        if family.name == "screened":
            return _blueprint(goal=(20.0, 0.0))
        return _blueprint(goal=(round(rng.uniform(12.0, 30.0), 1), 0.0))

    task = _synthetic_task(generate)
    entry = gd.analyse_task_object(task, samples=64, plan=gd.SeedPlan(1, "test"))
    assert entry["layout_families"]["screened"]["distinct_scenes"] == 1
    assert entry["verdict"] == "fail"
    assert any("distinct scene" in f for f in entry["failures"])


# ------------------------------------------------------- descriptor geometry


def test_path_length_routes_around_an_obstacle():
    """A wall on the direct line must cost tiles, or the detour factor is fiction."""
    wall = tuple(EntitySpec("stone-wall", (5.0, float(y))) for y in range(-6, 7))
    open_scene = _blueprint(goal=(10.0, 0.0))
    walled = _blueprint(goal=(10.0, 0.0), extra=wall)
    task = _synthetic_task(lambda family, rng: open_scene)
    assert gd.describe(task, walled)["path_length"] > gd.describe(task, open_scene)["path_length"]


def test_belts_are_walkable_and_do_not_block_a_route():
    """Transport belts carry the player rather than blocking it.

    Treating a belt line as a wall would report a fictitious detour around every
    repair_belt scene, which is a measurement defect masquerading as a task one.
    """
    belts = tuple(EntitySpec("transport-belt", (float(x), 0.0)) for x in range(1, 10))
    task = _synthetic_task(lambda family, rng: _blueprint())
    plain = gd.describe(task, _blueprint(goal=(10.0, 0.0)))
    lined = gd.describe(task, _blueprint(goal=(10.0, 0.0), extra=belts))
    assert lined["path_length"] == plain["path_length"]


def test_a_multi_tile_goal_entity_is_still_reachable():
    """A 3x3 drill's declared marker is its centre tile, which is inside the drill.

    Targeting the marker tile alone reports every such goal as unreachable, which
    would have made restore_power's whole difficulty column vanish.
    """
    drill = EntitySpec("electric-mining-drill", (12.0, 0.0), marker="goal")
    blueprint = Blueprint(
        entities=(drill,),
        character_position=(0.0, 0.0),
        markers={"goal": (12.0, 0.0)},
    )
    task = _synthetic_task(lambda family, rng: blueprint)
    assert gd.describe(task, blueprint)["path_length"] is not None


def test_difficulty_descriptors_are_none_when_no_success_predicate_names_a_marker():
    """mine_smelt and supply_furnace succeed on a production statistic.

    There is no declared position to measure a route to, and inventing one would
    be indistinguishable in the report from a measured number.
    """
    spec = TaskSpec(
        id="produced_only",
        version="0.0.0",
        description="succeeds on a production statistic",
        layout_families=(LayoutFamily("a", "train"), LayoutFamily("b", "test")),
        success=(Predicate(PredicateKind.PRODUCED, item="iron-plate", at_least=5),),
        rewards=(RewardComponent("made", RewardKind.SPARSE_SUCCESS, shaping=False),),
    )
    task = RegisteredTask(spec=spec, generate=lambda family, rng: _blueprint())
    values = gd.describe(task, _blueprint())
    for key in gd.DIFFICULTY_DESCRIPTORS:
        assert values[key] is None, key
    # ... and the structural half is still measured, so the report is not empty.
    assert values["obstacle_count"] == 1.0


def test_an_unmeasurable_difficulty_split_is_incomplete_not_passing():
    """A green verdict standing in for an unmeasured claim is the defect itself."""
    spec = TaskSpec(
        id="produced_only_2",
        version="0.0.0",
        description="succeeds on a production statistic",
        layout_families=(LayoutFamily("a", "train"), LayoutFamily("b", "test")),
        success=(Predicate(PredicateKind.PRODUCED, item="iron-plate", at_least=5),),
        rewards=(RewardComponent("made", RewardKind.SPARSE_SUCCESS, shaping=False),),
    )

    def generate(family: LayoutFamily, rng) -> Blueprint:
        goal = (round(rng.uniform(5.0, 30.0), 1), 0.0)
        extra = ()
        if family.name == "b":
            extra = tuple(EntitySpec("stone-wall", (float(x), 3.0)) for x in range(-10, 10))
        return _blueprint(goal=goal, extra=extra)

    entry = gd.analyse_task_object(
        RegisteredTask(spec=spec, generate=generate), samples=64, plan=gd.SeedPlan(1, "t")
    )
    assert entry["difficulty_parity"] == "unavailable"
    assert entry["verdict"] == "incomplete"
