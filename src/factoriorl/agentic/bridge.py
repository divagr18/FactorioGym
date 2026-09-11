"""Policy-safe bridge between a Windows Factorio worker and a WSL learner.

The bridge deliberately has four operations: reset, observe, one catalog action,
and finish. It does not proxy arbitrary protocol requests, Lua, worker paths,
RCON credentials, evaluator truth, or save-state access.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from factoriorl.tasks import TaskConfigError


class BridgeRequestError(ValueError):
    """A caller named an operation or argument outside the public bridge API."""


@dataclass
class FactorioBridge:
    """Expose one already-created environment through a minimal typed contract."""

    env: Any
    task_id: str
    _ended: bool = False

    def _catalog(self) -> list[dict]:
        return [
            {"index": index, "key": template.key, "arguments": list(template.arguments)}
            for index, template in enumerate(self.env.catalog.templates)
        ]

    def _state(self) -> dict:
        return {
            "task_id": self.task_id,
            "observation": self.env._observation,
            "catalog": self._catalog(),
            "argument_domains": self.env.argument_domains(),
            "action_mask": [bool(value) for value in self.env.action_masks()],
            "ended": self._ended,
        }

    def reset(self, seed: int | None = None) -> dict:
        # A GRPO group needs independent attempts from one scene.  The public
        # bridge seed therefore names a stable Factorio scene index as well as
        # seeding Gym; ordinary environment users retain sequential resets.
        options = None if seed is None else {"scene_index": seed}
        self.env.reset(seed=seed, options=options)
        self._ended = False
        return self._state()

    def observe(self) -> dict:
        return self._state()

    def act(self, index: int, arguments: dict | None = None) -> dict:
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
        try:
            _observation, reward, terminated, truncated, info = self.env.step_arguments(
                index, supplied
            )
        except ValueError as exc:
            raise BridgeRequestError(str(exc)) from exc
        self._ended = bool(terminated or truncated)
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
