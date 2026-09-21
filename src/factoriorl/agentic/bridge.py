"""Policy-safe bridge between a Windows Factorio worker and a WSL learner.

The bridge deliberately has four operations: reset, observe, one catalog action,
and finish. It does not proxy arbitrary protocol requests, Lua, worker paths,
RCON credentials, evaluator truth, or save-state access.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from factoriorl.agent.policy import build_policy_turn
from factoriorl.agent.summary import TaskBrief, targetable_actions, visible_handles
from factoriorl.tasks import TaskConfigError


class BridgeRequestError(ValueError):
    """A caller named an operation or argument outside the public bridge API."""


@dataclass
class FactorioBridge:
    """Expose one already-created environment through a minimal typed contract."""

    env: Any
    task_id: str
    _ended: bool = False
    _scene: dict | None = None
    _episode_nonce: int = 0
    _turn: int = 0
    _receipt: dict | None = None

    def _catalog(self) -> list[dict]:
        return [
            {"index": index, "key": template.key, "arguments": list(template.arguments)}
            for index, template in enumerate(self.env.catalog.templates)
        ]

    def _state(self) -> dict:
        policy = build_policy_turn(
            self.env,
            brief=TaskBrief.from_spec(self.env.spec_),
            observation=self.env._observation,
            step=self._turn,
            receipt=self._receipt,
        )
        return {
            "task_id": self.task_id,
            "scene": self._scene,
            "observation": self.env._observation,
            "catalog": self._catalog(),
            "argument_domains": self.env.argument_domains(),
            "action_mask": [bool(value) for value in self.env.action_masks()],
            "ended": self._ended,
            "episode_nonce": self._episode_nonce,
            "policy": policy.to_public_dict(),
        }

    def reset(self, seed: int | None = None) -> dict:
        # A GRPO group needs independent attempts from one scene.  The public
        # bridge seed therefore names a stable Factorio scene index as well as
        # seeding Gym; ordinary environment users retain sequential resets.
        options = None if seed is None else {"scene_index": seed}
        _observation, info = self.env.reset(seed=seed, options=options)
        self._scene = {
            "blueprint": info["blueprint"],
            "episode_index": info["episode_index"],
            "task_version": info["task_version"],
        }
        self._ended = False
        self._episode_nonce += 1
        self._turn = 0
        self._receipt = None
        return self._state()

    def observe(self) -> dict:
        return self._state()

    def act(self, index: int, arguments: dict | None = None, target: str | None = None) -> dict:
        if self._ended:
            raise BridgeRequestError("episode has ended; reset before acting again")
        if not isinstance(index, int) or not 0 <= index < len(self.env.catalog.templates):
            raise BridgeRequestError(f"action index {index!r} is outside the registered catalog")
        if not bool(self.env.action_masks()[index]):
            raise BridgeRequestError(f"action {index} is not currently legal")
        supplied = arguments or {}
        if not isinstance(supplied, dict):
            raise BridgeRequestError("arguments must be a JSON object")
        template = self.env.catalog.templates[index]
        expected = set(template.arguments)
        if set(supplied) != expected:
            raise BridgeRequestError(
                f"{template.key} requires exactly {sorted(expected)}, got {sorted(supplied)}"
            )
        if target is not None:
            if template.key not in targetable_actions(self.env):
                raise BridgeRequestError(f"action {template.key} does not accept a target")
            if target not in visible_handles(self.env._observation):
                raise BridgeRequestError(
                    f"target {target!r} is not visible in the current observation"
                )
        try:
            if target is None:
                _observation, reward, terminated, truncated, info = self.env.step_arguments(
                    index, supplied
                )
            else:
                context = {**self.env.unwrapped._context(), "target": target}
                _observation, reward, terminated, truncated, info = self.env.step_payload(
                    template.bind(context), action_key=template.key
                )
        except ValueError as exc:
            raise BridgeRequestError(str(exc)) from exc
        self._ended = bool(terminated or truncated)
        self._turn += 1
        self._receipt = {
            "actions": [
                {
                    "position": 0,
                    "key": template.key,
                    "target": target,
                    "arguments": dict(supplied),
                    "status": "completed" if not info.get("action_error") else "failed",
                    "action_status": info.get("action_status"),
                    "action_error": info.get("action_error"),
                }
            ]
        }
        return {
            **self._state(),
            "transition": {
                "reward": reward,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "success": bool(info.get("success")),
                "action_key": info.get("action_key"),
                "action_status": info.get("action_status"),
                "action_error": info.get("action_error"),
            },
        }

    def finish(self) -> dict:
        if self._ended:
            raise BridgeRequestError("episode has ended; reset before finishing again")
        try:
            verification = self.env.run_verification()
        except TaskConfigError as exc:
            raise BridgeRequestError(str(exc)) from exc
        self._ended = True
        return {**self._state(), "verification": verification}
