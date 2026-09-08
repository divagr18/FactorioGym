"""Duration-aware advantage estimation (R1.3).

A skill spans k primitive steps, so under a primitive-time objective the
continuation is discounted by ``gamma**k``, not ``gamma``. SB3's
`RolloutBuffer.compute_returns_and_advantage` uses a scalar ``self.gamma`` in
two places per step, which is correct only when every action lasts one step.
"""

from __future__ import annotations

import numpy as np
import torch as th
from sb3_contrib.common.maskable.buffers import MaskableDictRolloutBuffer, MaskableRolloutBuffer


class _DurationMixin:
    """Per-transition durations, defaulting to 1 so nothing changes without them."""

    def reset(self) -> None:  # type: ignore[override]
        super().reset()
        self.durations = np.ones((self.buffer_size, self.n_envs), dtype=np.float32)

    def set_duration(self, position: int, durations: np.ndarray) -> None:
        if 0 <= position < self.buffer_size:
            self.durations[position] = np.maximum(np.asarray(durations, dtype=np.float32), 1.0)

    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
        last_values = last_values.clone().cpu().numpy().flatten()
        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones.astype(np.float32)
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            # gamma**k for a transition of duration k, and the trace decays per
            # primitive step too -- a declared choice, not an accident.
            discount = self.gamma ** self.durations[step]
            trace = (self.gamma * self.gae_lambda) ** self.durations[step]
            delta = self.rewards[step] + discount * next_values * next_non_terminal
            delta = delta - self.values[step]
            last_gae_lam = delta + trace * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        self.returns = self.advantages + self.values


class DurationAwareMaskableRolloutBuffer(_DurationMixin, MaskableRolloutBuffer):
    pass


class DurationAwareMaskableDictRolloutBuffer(_DurationMixin, MaskableDictRolloutBuffer):
    pass
