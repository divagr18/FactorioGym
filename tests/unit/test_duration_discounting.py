"""Grouping primitive steps into one decision must preserve the return (R1.3).

The objective is at primitive-step time: a skill of duration k contributes
``sum(gamma**i * r_i)`` and the learner continues at ``gamma**k``. Summing
undiscounted and discounting once per decision was a different time preference,
and left a residual on potential-based shaping.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch as th
from sb3_contrib.common.maskable.buffers import MaskableRolloutBuffer
from gymnasium import spaces

from factoriorl.learn.buffers import DurationAwareMaskableRolloutBuffer

GAMMA = 0.99
LAMBDA = 0.95


def _buffer(cls, size=4, envs=1):
    buffer = cls(
        size,
        spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
        spaces.Discrete(3),
        gamma=GAMMA,
        gae_lambda=LAMBDA,
        n_envs=envs,
    )
    buffer.reset()
    return buffer


def _fill(buffer, rewards, values, episode_starts):
    for step, (reward, value, start) in enumerate(
        zip(rewards, values, episode_starts, strict=True)
    ):
        buffer.rewards[step] = reward
        buffer.values[step] = value
        buffer.episode_starts[step] = start
    buffer.pos = buffer.buffer_size
    buffer.full = True


REWARDS = [0.5, -0.25, 1.0, 0.75]
VALUES = [0.1, 0.2, 0.3, 0.4]
STARTS = [1.0, 0.0, 0.0, 0.0]


def test_all_durations_one_reproduces_stock_gae_exactly():
    """The safety property: no behaviour change unless a duration says so."""
    stock = _buffer(MaskableRolloutBuffer)
    _fill(stock, REWARDS, VALUES, STARTS)
    stock.compute_returns_and_advantage(th.tensor([[0.6]]), np.array([False]))

    ours = _buffer(DurationAwareMaskableRolloutBuffer)
    _fill(ours, REWARDS, VALUES, STARTS)
    ours.compute_returns_and_advantage(th.tensor([[0.6]]), np.array([False]))

    assert np.allclose(ours.advantages, stock.advantages, atol=1e-6)
    assert np.allclose(ours.returns, stock.returns, atol=1e-6)


def test_a_longer_duration_discounts_the_continuation_more():
    short = _buffer(DurationAwareMaskableRolloutBuffer)
    _fill(short, REWARDS, VALUES, STARTS)
    short.compute_returns_and_advantage(th.tensor([[1.0]]), np.array([False]))

    long = _buffer(DurationAwareMaskableRolloutBuffer)
    _fill(long, REWARDS, VALUES, STARTS)
    long.set_duration(3, np.array([20.0]))
    long.compute_returns_and_advantage(th.tensor([[1.0]]), np.array([False]))

    # A positive bootstrap 20 primitive steps away is worth less than one step
    # away, so the final advantage must fall.
    assert long.advantages[3][0] < short.advantages[3][0]


def test_a_duration_below_one_cannot_inflate_the_continuation():
    buffer = _buffer(DurationAwareMaskableRolloutBuffer)
    _fill(buffer, REWARDS, VALUES, STARTS)
    buffer.set_duration(0, np.array([0.0]))
    assert buffer.durations[0][0] == 1.0


# ------------------------------------------- grouping preserves the return


def _discounted(rewards, gamma=GAMMA):
    return sum((gamma**i) * r for i, r in enumerate(rewards))


def _grouped_return(groups, gamma=GAMMA):
    """What the learner sees: one reward per decision, gamma**k continuation."""
    total, elapsed = 0.0, 0
    for group in groups:
        total += (gamma**elapsed) * _discounted(group, gamma)
        elapsed += len(group)
    return total


@pytest.mark.parametrize(
    "groups",
    [
        [[0.1], [0.2], [0.3], [0.4]],
        [[0.1, 0.2], [0.3, 0.4]],
        [[0.1, 0.2, 0.3, 0.4]],
        [[0.1], [0.2, 0.3, 0.4]],
        [[-0.001] * 12, [1.0]],
    ],
)
def test_grouping_transitions_preserves_the_discounted_return(groups):
    flat = [r for group in groups for r in group]
    assert _grouped_return(groups) == pytest.approx(_discounted(flat), abs=1e-12)


def test_the_old_convention_did_not_preserve_it():
    """Why this change was needed, stated as a measurement."""
    groups = [[-0.001] * 12, [1.0]]
    flat = [r for group in groups for r in group]

    old = 0.0
    for index, group in enumerate(groups):
        old += (GAMMA**index) * sum(group)  # undiscounted sum, one gamma/decision

    assert abs(old - _discounted(flat)) > 0.05
    assert _grouped_return(groups) == pytest.approx(_discounted(flat), abs=1e-12)


# ----------------------------------------- shaping telescopes under it


def test_potential_shaping_telescopes_across_a_grouped_skill():
    """Sum of F = gamma*phi' - phi must reduce to the endpoints only."""
    weight, phi = 0.5, [0.70, 0.75, 0.81, 0.86, 0.90]
    shaping = [weight * (GAMMA * phi[i + 1] - phi[i]) for i in range(len(phi) - 1)]

    # One skill spanning all four primitive steps.
    grouped = _grouped_return([shaping])
    flat = _discounted(shaping)
    assert grouped == pytest.approx(flat, abs=1e-12)

    # And the primitive-time sum is the telescoped identity, endpoints only.
    expected = weight * sum((GAMMA ** (i + 1)) * phi[i + 1] - (GAMMA**i) * phi[i] for i in range(4))
    assert flat == pytest.approx(expected, abs=1e-12)


def test_the_residual_the_old_convention_left():
    """The term that was charged to skills which made progress."""
    weight, phi = 0.5, [0.70, 0.75, 0.81, 0.86, 0.90]
    shaping = [weight * (GAMMA * phi[i + 1] - phi[i]) for i in range(len(phi) - 1)]

    undiscounted = sum(shaping)
    telescoped = weight * (GAMMA * phi[-1] - phi[0])
    residual = undiscounted - telescoped

    # Non-zero, and negative: it penalises approaching the goal.
    assert residual < 0
    assert residual == pytest.approx(weight * (GAMMA - 1.0) * sum(phi[1:-1]), abs=1e-12)
