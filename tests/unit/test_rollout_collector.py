from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "training"))

from rollout_artifacts import RolloutWriter
from rollout_collector import Sample, SequentialGroupCollector


class _Client:
    def reset(self, seed):
        return {
            "task_id": "construct_smelting_line",
            "scene": {"blueprint": "same", "episode_index": seed, "task_version": "v1"},
            "observation": {"seed": seed},
            "catalog": [],
            "argument_domains": {},
            "action_mask": [True],
        }

    def act(self, index, arguments):
        return {
            "task_id": "construct_smelting_line",
            "transition": {"reward": 0.0, "terminated": False, "truncated": False},
        }

    def finish(self):
        return {"verification": {"reward": 0.0}}


class _Sampler:
    policy_revision = "frozen-adapter"

    def sample(self, _state):
        return Sample([1], [2], [-0.1], 0, {})


def test_same_seed_group_writes_four_closed_attempts(tmp_path):
    writer = RolloutWriter(tmp_path / "run", run_id="r", task_version="v", policy_revision="p")
    digests = SequentialGroupCollector(_Client(), writer, _Sampler()).collect_group(
        seed=9, max_turns=1
    )
    assert len(digests) == 4
    assert len((tmp_path / "run" / "episodes.jsonl").read_text().splitlines()) == 4
