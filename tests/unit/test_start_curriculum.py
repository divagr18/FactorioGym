"""Exploring starts must never reach a stream that gets reported.

A run may start a fraction of its *training* episodes beside the task's focus
marker, so that a family whose success is a conjunction -- `repair_belt` needs
the right tile and one of four belt facings -- actually reaches the placement
and can compare the facings. The risk is one-directional and is exactly the
contamination PLAN section 3 forbids: leaked into an evaluated stream, the
held-out number would describe an easier task than the benchmark claims.

The subtle case is the unfamiliar-seed row, measured with `Branch.EVAL` over
the *training* split -- so gating on the split alone is not enough, and this
pins the branch.

It is a run setting rather than a task field on purpose: it cannot change any
test-split scene, so it must not bump the task version or invalidate a frozen
holdout. That also makes curriculum-on and curriculum-off comparable against
the same holdout.
"""

from dataclasses import replace

import pytest

from factoriorl.env import FactorioEnv
from factoriorl.seeding import Branch, SeedPlan
from factoriorl.tasks import get
from factoriorl.tasks.spec import Blueprint, EntitySpec


class _Env(FactorioEnv):
    """Just the scene-selection half; no engine, no session."""

    def __init__(self, task, branch, split, start_curriculum=1 / 3):
        self.task = task
        self.spec_ = task.spec
        self.branch = branch
        self.split = split
        self.start_curriculum = float(start_curriculum)
        self.seed_plan = SeedPlan(master=7, run_id="curriculum-test")


def _blueprint():
    return Blueprint(
        entities=(EntitySpec(name="transport-belt", position=(12.5, -3.5)),),
        character_position=(0.0, 0.0),
        markers={"gap": (13.5, -3.5), "sink": (17.5, -3.5)},
    )


def _starts(branch, split, episodes=90, start_curriculum=1 / 3):
    env = _Env(get("repair_belt"), branch, split, start_curriculum)
    out = []
    for index in range(episodes):
        rng = env.seed_plan.generator_rng(branch, index)
        out.append(env._apply_start_curriculum(_blueprint(), rng).character_position)
    return out


def test_training_stream_moves_some_starts_to_the_gap():
    starts = _starts(Branch.TRAIN, "train")
    moved = [s for s in starts if s != (0.0, 0.0)]
    assert moved, "the curriculum never fired on the training stream"
    # Roughly the declared third; wide bounds because this is a real draw.
    assert 0.15 < len(moved) / len(starts) < 0.55
    for position in moved:
        assert abs(position[0] - 13.5) <= 3.0 and abs(position[1] + 3.5) <= 3.0


@pytest.mark.parametrize(
    ("branch", "split"),
    [
        (Branch.EVAL, "test"),
        (Branch.EVAL, "val"),
        # The one that a split-only guard would have let through.
        (Branch.EVAL, "train"),
        (Branch.TRAIN, "test"),
        (Branch.TRAIN, "val"),
    ],
)
def test_evaluated_streams_always_keep_the_canonical_start(branch, split):
    assert all(s == (0.0, 0.0) for s in _starts(branch, split))


def test_a_run_declaring_no_curriculum_is_untouched():
    """The default path, and every frozen digest, must be byte-identical."""
    assert all(s == (0.0, 0.0) for s in _starts(Branch.TRAIN, "train", start_curriculum=0.0))


def test_the_start_never_lands_on_an_entity():
    env = _Env(get("repair_belt"), Branch.TRAIN, "train")
    blueprint = replace(
        _blueprint(),
        entities=tuple(
            EntitySpec(name="transport-belt", position=(13.5, -3.5 + dy))
            for dy in (2.0, -2.0, 3.0, -3.0)
        )
        + (EntitySpec(name="transport-belt", position=(15.5, -3.5)),),
    )
    occupied = {(round(e.position[0], 1), round(e.position[1], 1)) for e in blueprint.entities}
    for index in range(60):
        rng = env.seed_plan.generator_rng(Branch.TRAIN, index)
        position = env._apply_start_curriculum(blueprint, rng).character_position
        assert (round(position[0], 1), round(position[1], 1)) not in occupied


def test_curriculum_is_recorded_in_the_run_config():
    from factoriorl.learn.train import TrainConfig

    assert TrainConfig(task_id="repair_belt").to_dict()["start_curriculum"] == 0.0
    recorded = TrainConfig(task_id="repair_belt", start_curriculum=1 / 3).to_dict()
    assert recorded["start_curriculum"] == pytest.approx(1 / 3)


def test_it_is_not_a_task_field_so_the_holdout_stays_valid():
    """A test-split scene cannot observe it, so it must not touch the digest."""
    resolved = get("repair_belt").spec.to_dict()
    assert "start_state_curriculum" not in resolved
    assert resolved["version"] == "1.5.0"
    assert resolved["focus_marker"] == "gap"
