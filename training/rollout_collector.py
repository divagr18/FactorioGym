"""Sequential same-scene rollout collection for the first GRPO feasibility run."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Protocol

from bridge_client import BridgeClient, BridgeError
from rollout_artifacts import Episode, RolloutArtifactError, RolloutStep, RolloutWriter


class PolicySampler(Protocol):
    """A frozen sampler returning an executable action and sampled-token evidence."""

    policy_revision: str

    def sample(self, state: dict[str, Any]) -> Sample: ...


@dataclass(frozen=True)
class Sample:
    prompt_token_ids: list[int]
    completion_token_ids: list[int]
    behavior_logprobs: list[float]
    action_index: int
    arguments: dict[str, Any]


def public_scene_hash(state: dict[str, Any]) -> str:
    """Hash the installed scene identity, excluding reset-volatile entity handles."""
    payload = json.dumps(
        {"task_id": state["task_id"], "scene": state["scene"]},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class SequentialGroupCollector:
    def __init__(self, client: BridgeClient, writer: RolloutWriter, sampler: PolicySampler):
        self.client = client
        self.writer = writer
        self.sampler = sampler

    def collect_group(self, *, seed: int, group_size: int = 4, max_turns: int = 32) -> list[str]:
        if group_size < 2:
            raise RolloutArtifactError("a GRPO group needs at least two attempts")
        scene_hashes: list[str] = []
        episode_digests: list[str] = []
        for attempt in range(group_size):
            state = self.client.reset(seed)
            scene_hash = public_scene_hash(state)
            scene_hashes.append(scene_hash)
            if len(set(scene_hashes)) != 1:
                raise RolloutArtifactError("same-seed group reset produced different public scenes")
            steps: list[RolloutStep] = []
            terminated = truncated = False
            reason = "turn_limit"
            for turn in range(max_turns):
                sample = self.sampler.sample(state)
                try:
                    outcome = self.client.act(sample.action_index, sample.arguments)
                except BridgeError as exc:
                    steps.append(
                        RolloutStep(
                            turn,
                            sample.prompt_token_ids,
                            sample.completion_token_ids,
                            sample.behavior_logprobs,
                            {"index": sample.action_index, "arguments": sample.arguments},
                            {"bridge_error": str(exc)},
                            0.0,
                        )
                    )
                    terminated, truncated, reason = False, True, "invalid_tool_call"
                    break
                transition = outcome["transition"]
                steps.append(
                    RolloutStep(
                        turn,
                        sample.prompt_token_ids,
                        sample.completion_token_ids,
                        sample.behavior_logprobs,
                        {"index": sample.action_index, "arguments": sample.arguments},
                        transition,
                        float(transition["reward"]),
                    )
                )
                state = outcome
                terminated, truncated = (
                    bool(transition["terminated"]),
                    bool(transition["truncated"]),
                )
                if terminated or truncated:
                    reason = "environment_terminal"
                    break
            if not (terminated or truncated):
                finished = self.client.finish()
                terminated, truncated, reason = True, False, "verification_complete"
                if steps:
                    last = steps[-1]
                    steps[-1] = RolloutStep(
                        last.turn,
                        last.prompt_token_ids,
                        last.completion_token_ids,
                        last.behavior_logprobs,
                        last.tool_call,
                        {"transition": last.tool_result, "verification": finished["verification"]},
                        float(finished["verification"]["reward"]),
                    )
            episode = Episode(
                episode_id=f"seed-{seed}-attempt-{attempt}",
                task_id=state["task_id"],
                scene_id=scene_hash,
                seed=seed,
                policy_revision=self.sampler.policy_revision,
                steps=steps,
                terminated=terminated,
                truncated=truncated,
                termination_reason=reason,
            )
            episode_digests.append(self.writer.append(episode))
        return episode_digests
