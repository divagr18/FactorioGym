"""Sequential, same-scene collection for the first on-policy learner run."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from bridge_client import BridgeClient, BridgeError
from rollout_artifacts import Episode, RolloutArtifactError, RolloutGroup, RolloutStep, RolloutWriter


class PolicySampler(Protocol):
    """A frozen sampler returning native actions and sampled-token evidence."""

    policy_revision: str

    def sample(self, state: dict[str, Any]) -> "Sample": ...


@dataclass(frozen=True)
class Sample:
    prompt_token_ids: list[int]
    completion_token_ids: list[int]
    behavior_logprobs: list[float]
    action_index: int | None
    arguments: dict[str, Any]
    target: str | None = None
    messages: list[dict[str, str]] = field(default_factory=list)
    rendered_prompt: str = ""
    sampled_token_mask: list[bool] = field(default_factory=list)
    completion_text: str = ""
    parsed_action: dict[str, Any] | None = None
    parse_failure: dict[str, Any] | None = None
    prompt_digest: str = ""
    logprob_recompute_max_abs_error: float = 0.0


def public_scene_hash(state: dict[str, Any]) -> str:
    """Hash installed scene identity, excluding reset-volatile entity handles."""
    payload = json.dumps(
        {"task_id": state["task_id"], "scene": state["scene"]},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _step(sample: Sample, *, turn: int, tool_call: dict | None, tool_result: dict | None, reward: float) -> RolloutStep:
    return RolloutStep(
        turn=turn,
        prompt_token_ids=sample.prompt_token_ids,
        completion_token_ids=sample.completion_token_ids,
        behavior_logprobs=sample.behavior_logprobs,
        tool_call=tool_call,
        tool_result=tool_result,
        reward=reward,
        messages=sample.messages,
        rendered_prompt=sample.rendered_prompt,
        sampled_token_mask=sample.sampled_token_mask,
        completion_text=sample.completion_text,
        parsed_action=sample.parsed_action,
        parse_failure=sample.parse_failure,
        logprob_recompute_max_abs_error=sample.logprob_recompute_max_abs_error,
    )


class SequentialGroupCollector:
    """Collect frozen-policy attempts one at a time, resetting every member."""

    def __init__(self, client: BridgeClient, writer: RolloutWriter, sampler: PolicySampler):
        self.client = client
        self.writer = writer
        self.sampler = sampler

    def collect_group(self, *, seed: int, group_size: int = 4, max_turns: int = 32) -> list[str]:
        if group_size < 2:
            raise RolloutArtifactError("a GRPO group needs at least two attempts")
        scene_hashes: list[str] = []
        episode_digests: list[str] = []
        episode_rewards: list[float] = []
        for attempt in range(group_size):
            state = self.client.reset(seed)
            scene_hash = public_scene_hash(state)
            scene_hashes.append(scene_hash)
            if len(set(scene_hashes)) != 1:
                raise RolloutArtifactError("same-seed group reset produced different public scenes")
            steps: list[RolloutStep] = []
            terminated = truncated = False
            reason = "turn_limit"
            terminal_verification: dict[str, Any] | None = None
            terminal_reward = 0.0
            prompt_digest = ""
            for turn in range(max_turns):
                sample = self.sampler.sample(state)
                if sample.prompt_digest:
                    if prompt_digest and prompt_digest != sample.prompt_digest:
                        raise RolloutArtifactError("policy prompt changed inside one episode")
                    prompt_digest = sample.prompt_digest
                if sample.action_index is None:
                    steps.append(
                        _step(
                            sample,
                            turn=turn,
                            tool_call=None,
                            tool_result={"parse_failure": sample.parse_failure},
                            reward=0.0,
                        )
                    )
                    truncated, reason = True, "parse_failure"
                    break
                tool_call = {
                    "index": sample.action_index,
                    "arguments": sample.arguments,
                    "target": sample.target,
                }
                try:
                    outcome = (
                        self.client.act(sample.action_index, sample.arguments, sample.target)
                        if sample.target is not None
                        else self.client.act(sample.action_index, sample.arguments)
                    )
                except BridgeError as exc:
                    steps.append(
                        _step(
                            sample,
                            turn=turn,
                            tool_call=tool_call,
                            tool_result={"bridge_error": str(exc)},
                            reward=0.0,
                        )
                    )
                    truncated, reason = True, "bridge_error"
                    break
                transition = outcome["transition"]
                steps.append(
                    _step(
                        sample,
                        turn=turn,
                        tool_call=tool_call,
                        tool_result=transition,
                        reward=float(transition["reward"]),
                    )
                )
                state = outcome
                terminated, truncated = bool(transition["terminated"]), bool(transition["truncated"])
                if terminated or truncated:
                    reason = "environment_terminal"
                    break
            if not (terminated or truncated):
                finished = self.client.finish()
                terminal_verification = dict(finished["verification"])
                terminal_reward = float(terminal_verification["reward"])
                terminated, reason = True, "verification_complete"
            if not prompt_digest:
                raise RolloutArtifactError("sampler returned no prompt digest")
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
                terminal_verification=terminal_verification,
                terminal_reward=terminal_reward,
                prompt_digest=prompt_digest,
            )
            episode_digests.append(self.writer.append(episode))
            episode_rewards.append(terminal_reward)
        self.writer.append_group(
            RolloutGroup(
                group_id=f"seed-{seed}",
                scene_id=scene_hashes[0],
                policy_revision=self.sampler.policy_revision,
                episode_digests=episode_digests,
                rewards=episode_rewards,
            )
        )
        return episode_digests
