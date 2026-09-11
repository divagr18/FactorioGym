"""Append-only, replayable artifacts for multi-turn language-model rollouts.

The learner owns this file format rather than the Windows bridge.  A stored
episode contains exactly what is needed to recompute the sampled-token policy
log probabilities: prompt and completion token IDs, the behavior log
probabilities, public tool traffic, and a pinned policy revision.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


class RolloutArtifactError(ValueError):
    """The artifact is incomplete or inconsistent with an RL update."""


@dataclass(frozen=True)
class RolloutStep:
    """One model completion followed by at most one catalog action."""

    turn: int
    prompt_token_ids: list[int]
    completion_token_ids: list[int]
    behavior_logprobs: list[float]
    tool_call: dict[str, Any] | None
    tool_result: dict[str, Any] | None
    reward: float


@dataclass(frozen=True)
class Episode:
    """A fully closed episode, ready for grouped advantage computation."""

    episode_id: str
    task_id: str
    scene_id: str
    seed: int
    policy_revision: str
    steps: list[RolloutStep]
    terminated: bool
    truncated: bool
    termination_reason: str


def _validate_token_ids(name: str, token_ids: list[int], *, allow_empty: bool) -> None:
    if not isinstance(token_ids, list) or (not allow_empty and not token_ids):
        raise RolloutArtifactError(
            f"{name} must be a {'possibly empty ' if allow_empty else 'non-empty '}list"
        )
    if any(
        isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in token_ids
    ):
        raise RolloutArtifactError(f"{name} contains an invalid token id")


def validate_episode(episode: Episode) -> None:
    if not all(
        isinstance(value, str) and value
        for value in (
            episode.episode_id,
            episode.task_id,
            episode.scene_id,
            episode.policy_revision,
            episode.termination_reason,
        )
    ):
        raise RolloutArtifactError(
            "episode identity and termination fields must be non-empty strings"
        )
    if isinstance(episode.seed, bool) or not isinstance(episode.seed, int):
        raise RolloutArtifactError("seed must be an integer")
    if not episode.steps:
        raise RolloutArtifactError("an episode must contain at least one model step")
    if not (episode.terminated or episode.truncated):
        raise RolloutArtifactError("only closed episodes may enter a training artifact")
    for expected_turn, step in enumerate(episode.steps):
        if step.turn != expected_turn:
            raise RolloutArtifactError("turns must start at zero and be consecutive")
        _validate_token_ids("prompt_token_ids", step.prompt_token_ids, allow_empty=False)
        _validate_token_ids("completion_token_ids", step.completion_token_ids, allow_empty=False)
        if len(step.completion_token_ids) != len(step.behavior_logprobs):
            raise RolloutArtifactError("completion_token_ids and behavior_logprobs must align")
        if any(
            not isinstance(value, (int, float)) or not math.isfinite(float(value))
            for value in step.behavior_logprobs
        ):
            raise RolloutArtifactError("behavior_logprobs must be finite numbers")
        if not math.isfinite(float(step.reward)):
            raise RolloutArtifactError("reward must be finite")
        if step.tool_call is not None and not isinstance(step.tool_call, dict):
            raise RolloutArtifactError("tool_call must be an object or null")
        if step.tool_result is not None and not isinstance(step.tool_result, dict):
            raise RolloutArtifactError("tool_result must be an object or null")


class RolloutWriter:
    """Create a single immutable JSONL run directory without hidden globals."""

    def __init__(self, directory: Path, *, run_id: str, task_version: str, policy_revision: str):
        self.directory = directory
        self.path = directory / "episodes.jsonl"
        if directory.exists():
            raise RolloutArtifactError(f"run directory already exists: {directory}")
        directory.mkdir(parents=True)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "task_version": task_version,
            "policy_revision": policy_revision,
            "episodes_file": self.path.name,
        }
        (directory / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

    def append(self, episode: Episode) -> str:
        validate_episode(episode)
        encoded = json.dumps(asdict(episode), separators=(",", ":"), sort_keys=True)
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"episode": json.loads(encoded), "sha256": digest}, separators=(",", ":")
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        return digest
