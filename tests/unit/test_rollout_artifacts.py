from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from training.rollout_artifacts import Episode, RolloutArtifactError, RolloutStep, RolloutWriter


def _episode(**changes):
    episode = Episode(
        episode_id="episode-1",
        task_id="construct_smelting_line",
        scene_id="frozen-scene-3",
        seed=3,
        policy_revision="gemma-e2b@adapter-sha",
        steps=[
            RolloutStep(
                turn=0,
                prompt_token_ids=[1, 2],
                completion_token_ids=[3, 4],
                behavior_logprobs=[-0.2, -0.3],
                tool_call={"index": 21, "arguments": {}},
                tool_result={"reward": 0.0, "terminated": False},
                reward=0.0,
            )
        ],
        terminated=True,
        truncated=False,
        termination_reason="verification_complete",
    )
    return Episode(**{**episode.__dict__, **changes})


def test_writer_records_recomputable_episode(tmp_path):
    writer = RolloutWriter(tmp_path / "run", run_id="r1", task_version="v1", policy_revision="p1")
    digest = writer.append(_episode())
    line = json.loads((tmp_path / "run" / "episodes.jsonl").read_text())
    assert line["sha256"] == digest
    assert line["episode"]["steps"][0]["completion_token_ids"] == [3, 4]
    assert line["episode"]["steps"][0]["behavior_logprobs"] == [-0.2, -0.3]


def test_writer_rejects_misaligned_sampled_token_probabilities(tmp_path):
    writer = RolloutWriter(tmp_path / "run", run_id="r1", task_version="v1", policy_revision="p1")
    invalid_step = RolloutStep(0, [1], [2, 3], [-0.1], None, None, 0.0)
    with pytest.raises(RolloutArtifactError, match="align"):
        writer.append(_episode(steps=[invalid_step]))


def test_writer_only_accepts_closed_episode(tmp_path):
    writer = RolloutWriter(tmp_path / "run", run_id="r1", task_version="v1", policy_revision="p1")
    with pytest.raises(RolloutArtifactError, match="closed"):
        writer.append(_episode(terminated=False, truncated=False))
