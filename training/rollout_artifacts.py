"""Append-only, auditable artifacts for language-model rollout groups.

Schema two records the input the policy actually saw as well as token IDs. A
current source checkout must never be needed to discover which prompt produced
an on-policy sample.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2
LOGPROB_RECOMPUTE_TOLERANCE = 2e-3


class RolloutArtifactError(ValueError):
    """The artifact is incomplete or inconsistent with an RL update."""


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class RolloutStep:
    """One sampled completion and its public execution outcome."""

    turn: int
    prompt_token_ids: list[int]
    completion_token_ids: list[int]
    behavior_logprobs: list[float]
    tool_call: dict[str, Any] | None
    tool_result: dict[str, Any] | None
    reward: float
    messages: list[dict[str, str]] | None = None
    rendered_prompt: str = ""
    sampled_token_mask: list[bool] | None = None
    completion_text: str = ""
    parsed_action: dict[str, Any] | None = None
    parse_failure: dict[str, Any] | None = None
    logprob_recompute_max_abs_error: float = 0.0


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
    terminal_verification: dict[str, Any] | None = None
    terminal_reward: float = 0.0
    prompt_digest: str = ""


@dataclass(frozen=True)
class RolloutGroup:
    """Independent attempts from one frozen scene and policy revision."""

    group_id: str
    scene_id: str
    policy_revision: str
    episode_digests: list[str]
    rewards: list[float]


def _validate_token_ids(name: str, token_ids: list[int], *, allow_empty: bool) -> None:
    if not isinstance(token_ids, list) or (not allow_empty and not token_ids):
        raise RolloutArtifactError(
            f"{name} must be a {'possibly empty ' if allow_empty else 'non-empty '}list"
        )
    if any(
        isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in token_ids
    ):
        raise RolloutArtifactError(f"{name} contains an invalid token id")


def _validate_messages(messages: list[dict[str, str]] | None) -> None:
    if not messages or not all(
        isinstance(message, dict)
        and set(message) == {"role", "content"}
        and isinstance(message["role"], str)
        and isinstance(message["content"], str)
        for message in messages
    ):
        raise RolloutArtifactError("messages must be non-empty role/content records")


def validate_episode(episode: Episode) -> None:
    if not all(
        isinstance(value, str) and value
        for value in (
            episode.episode_id,
            episode.task_id,
            episode.scene_id,
            episode.policy_revision,
            episode.termination_reason,
            episode.prompt_digest,
        )
    ):
        raise RolloutArtifactError("episode identity, prompt, and termination fields must be strings")
    if isinstance(episode.seed, bool) or not isinstance(episode.seed, int):
        raise RolloutArtifactError("seed must be an integer")
    if not episode.steps:
        raise RolloutArtifactError("an episode must contain at least one model step")
    if not (episode.terminated or episode.truncated):
        raise RolloutArtifactError("only closed episodes may enter a training artifact")
    if not math.isfinite(float(episode.terminal_reward)):
        raise RolloutArtifactError("terminal_reward must be finite")
    if episode.terminal_verification is not None and not isinstance(episode.terminal_verification, dict):
        raise RolloutArtifactError("terminal_verification must be an object or null")
    for expected_turn, step in enumerate(episode.steps):
        if step.turn != expected_turn:
            raise RolloutArtifactError("turns must start at zero and be consecutive")
        _validate_messages(step.messages)
        if not isinstance(step.rendered_prompt, str) or not step.rendered_prompt:
            raise RolloutArtifactError("rendered_prompt must be a non-empty string")
        _validate_token_ids("prompt_token_ids", step.prompt_token_ids, allow_empty=False)
        _validate_token_ids("completion_token_ids", step.completion_token_ids, allow_empty=False)
        if len(step.completion_token_ids) != len(step.behavior_logprobs):
            raise RolloutArtifactError("completion_token_ids and behavior_logprobs must align")
        if step.sampled_token_mask is None or len(step.sampled_token_mask) != len(
            step.completion_token_ids
        ) or not all(step.sampled_token_mask):
            raise RolloutArtifactError("sampled_token_mask must select every generated token")
        if any(
            not isinstance(value, (int, float)) or not math.isfinite(float(value))
            for value in step.behavior_logprobs
        ):
            raise RolloutArtifactError("behavior_logprobs must be finite numbers")
        if not math.isfinite(float(step.logprob_recompute_max_abs_error)):
            raise RolloutArtifactError("logprob_recompute_max_abs_error must be finite")
        if not math.isfinite(float(step.reward)):
            raise RolloutArtifactError("reward must be finite")
        if step.tool_call is not None and not isinstance(step.tool_call, dict):
            raise RolloutArtifactError("tool_call must be an object or null")
        if step.tool_result is not None and not isinstance(step.tool_result, dict):
            raise RolloutArtifactError("tool_result must be an object or null")
        if step.parsed_action is not None and not isinstance(step.parsed_action, dict):
            raise RolloutArtifactError("parsed_action must be an object or null")
        if step.parse_failure is not None and not isinstance(step.parse_failure, dict):
            raise RolloutArtifactError("parse_failure must be an object or null")
        if step.parsed_action is not None and step.parse_failure is not None:
            raise RolloutArtifactError("a step cannot have both parsed_action and parse_failure")


def validate_group(group: RolloutGroup) -> None:
    if not all(
        isinstance(value, str) and value for value in (group.group_id, group.scene_id, group.policy_revision)
    ):
        raise RolloutArtifactError("group identity fields must be non-empty strings")
    if len(group.episode_digests) < 2 or len(group.episode_digests) != len(group.rewards):
        raise RolloutArtifactError("group members and rewards must align and contain at least two")
    if len(set(group.episode_digests)) != len(group.episode_digests):
        raise RolloutArtifactError("group contains a duplicate episode digest")
    if any(not math.isfinite(float(reward)) for reward in group.rewards):
        raise RolloutArtifactError("group rewards must be finite")


class RolloutWriter:
    """Create one immutable run directory without hidden global state."""

    def __init__(
        self,
        directory: Path,
        *,
        run_id: str,
        task_version: str,
        policy_revision: str,
        metadata: dict[str, Any] | None = None,
    ):
        self.directory = directory
        self.path = directory / "episodes.jsonl"
        self.groups_path = directory / "groups.jsonl"
        if directory.exists():
            raise RolloutArtifactError(f"run directory already exists: {directory}")
        directory.mkdir(parents=True)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "task_version": task_version,
            "policy_revision": policy_revision,
            "episodes_file": self.path.name,
            "groups_file": self.groups_path.name,
            "metadata": dict(metadata or {}),
        }
        (directory / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def _append(self, path: Path, key: str, value: Any) -> str:
        encoded = json.dumps(asdict(value), separators=(",", ":"), sort_keys=True)
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps({key: json.loads(encoded), "sha256": digest}, separators=(",", ":"))
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        return digest

    def append(self, episode: Episode) -> str:
        validate_episode(episode)
        return self._append(self.path, "episode", episode)

    def append_group(self, group: RolloutGroup) -> str:
        validate_group(group)
        return self._append(self.groups_path, "group", group)


def audit_run(directory: Path) -> dict[str, int]:
    """Validate hashes and schema of a written run without loading a model."""
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise RolloutArtifactError("only schema-v2 artifacts are eligible for training")
    counts = {"episodes": 0, "groups": 0}
    episodes: dict[str, Episode] = {}
    episodes_path = directory / "episodes.jsonl"
    if not episodes_path.exists():
        raise RolloutArtifactError("missing episodes.jsonl")
    for line in episodes_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if set(record) != {"episode", "sha256"} or record["sha256"] != _digest(record["episode"]):
            raise RolloutArtifactError("episodes.jsonl payload digest mismatch")
        body = dict(record["episode"])
        body["steps"] = [RolloutStep(**step) for step in body["steps"]]
        episode = Episode(**body)
        validate_episode(episode)
        if any(
            step.logprob_recompute_max_abs_error > LOGPROB_RECOMPUTE_TOLERANCE
            for step in episode.steps
        ):
            raise RolloutArtifactError("sampled-token logprob recomputation exceeded tolerance")
        episodes[record["sha256"]] = episode
        counts["episodes"] += 1
    groups_path = directory / "groups.jsonl"
    if not groups_path.exists():
        return counts
    for line in groups_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if set(record) != {"group", "sha256"} or record["sha256"] != _digest(record["group"]):
            raise RolloutArtifactError("groups.jsonl payload digest mismatch")
        group = RolloutGroup(**record["group"])
        validate_group(group)
        members = [episodes.get(digest) for digest in group.episode_digests]
        if any(member is None for member in members):
            raise RolloutArtifactError("group references an episode not present in this run")
        if any(member.scene_id != group.scene_id for member in members):
            raise RolloutArtifactError("group contains mixed scene identities")
        if any(member.policy_revision != group.policy_revision for member in members):
            raise RolloutArtifactError("group contains mixed policy revisions")
        if [member.terminal_reward for member in members] != group.rewards:
            raise RolloutArtifactError("group rewards do not match episode terminal rewards")
        counts["groups"] += 1
    return counts
