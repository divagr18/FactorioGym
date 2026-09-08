"""A frozen evaluation must terminate, and must never substitute a scene.

The defect: `_reset_one` past the frozen bound returned `index = None` while
still playing a real episode, which could never become eligible, so
`evaluate_parallel`'s `while counted < episodes` spun on genuine engine
episodes indefinitely. There was no timeout at any level -- not in the loop,
not in `release_matrix`'s `subprocess.run` -- and `--eval-episodes 100` against
`episodes_per_task 100` left exactly zero slack, so one excluded episode was
enough.

The gate requires a failure injected at the **first**, **middle** and **final**
allocated scene, each terminating with either exact unique coverage after
bounded retry, or an explicit incomplete result. These tests drive the real
`evaluate_parallel` against a fake vec env, which needs no engine because the
function only calls `reset`, `action_masks` and `step`.
"""

from __future__ import annotations

import collections
import threading

import numpy as np
import pytest

from factoriorl.learn.train import evaluate_parallel
from factoriorl.vecenv import FactorioVecEnv

START = 3000
COUNT = 6
ACTIONS = 4


class _Model:
    """Always picks action 0; `evaluate_parallel` only needs `predict`."""

    def predict(self, observation, action_masks=None, deterministic=True):
        n = len(action_masks) if action_masks is not None else 1
        return np.zeros(n, dtype=np.int64), None


class FrozenVecEnv:
    """A frozen-index queue with the real requeue semantics, no engine.

    Mirrors `FactorioVecEnv`'s bounded-stream behaviour by *reusing* the real
    methods where they carry the logic under test -- `release_episode`,
    `pending_episodes`, `episode_attempts` are bound from the real class, so a
    change in their semantics breaks these tests rather than passing silently.
    """

    release_episode = FactorioVecEnv.release_episode
    pending_episodes = FactorioVecEnv.pending_episodes
    episode_attempts = FactorioVecEnv.episode_attempts

    def __init__(self, num_envs: int = 2, fail_scenes: tuple[int, ...] = (), fail_times: int = 99):
        self.num_envs = num_envs
        self._cursor_lock = threading.Lock()
        self._pending_indices = collections.deque(range(START, START + COUNT))
        self._attempts: collections.Counter = collections.Counter()
        self._issued: list[int] = []
        self.fail_scenes = set(fail_scenes)
        self.fail_times = fail_times
        self._failed_so_far: collections.Counter = collections.Counter()
        self._live: list[int | None] = [None] * num_envs
        self.scored_successes = 0

    # ---- what evaluate_parallel touches --------------------------------
    def reset(self):
        for slot in range(self.num_envs):
            self._assign(slot)
        return {"x": np.zeros((self.num_envs, 1), dtype=np.float32)}

    def action_masks(self):
        return np.ones((self.num_envs, ACTIONS), dtype=bool)

    def in_flight_episodes(self):
        return list(self._live)

    def step(self, actions):
        dones, infos, rewards = [], [], []
        for slot in range(self.num_envs):
            episode = self._live[slot]
            info: dict = {"episode_index": episode}
            if episode is not None and episode in self.fail_scenes:
                if self._failed_so_far[episode] < self.fail_times:
                    self._failed_so_far[episode] += 1
                    info["excluded_from_metrics"] = True
                    info["infrastructure_failure"] = "transport failure"
                else:
                    info["success"] = True
            elif episode is not None:
                info["success"] = True
            dones.append(True)
            rewards.append(0.0)
            infos.append(info)
        # `evaluate_parallel` releases the identity before the env resets, so
        # the requeue has to be visible to the next assignment -- mirroring
        # the real order in `step_wait`.
        self._pending_release = list(zip(range(self.num_envs), self._live, strict=True))
        out = (
            {"x": np.zeros((self.num_envs, 1), dtype=np.float32)},
            np.array(rewards, dtype=np.float32),
            np.array(dones),
            infos,
        )
        for slot in range(self.num_envs):
            self._assign(slot)
        return out

    def _assign(self, slot: int) -> None:
        with self._cursor_lock:
            index = self._pending_indices.popleft() if self._pending_indices else None
            if index is not None:
                self._issued.append(index)
                self._attempts[index] += 1
        self._live[slot] = index


def _evaluate(env, episodes=COUNT):
    return evaluate_parallel(
        env,
        _Model(),
        episodes,
        only_indices=set(range(START, START + COUNT)),
    )


# ------------------------------------------------------- the three gate cases


@pytest.mark.parametrize(
    ("label", "scene"),
    [("first", START), ("middle", START + COUNT // 2), ("final", START + COUNT - 1)],
)
def test_a_transient_failure_is_retried_and_coverage_stays_exact(label, scene):
    """Bounded retry of the *same* scene identity, then full coverage."""
    env = FrozenVecEnv(fail_scenes=(scene,), fail_times=1)
    row = _evaluate(env)

    assert row["incomplete_coverage"] is None, f"{label}: reported incomplete despite a retry"
    assert row["episodes"] == COUNT
    assert row["scored_episodes"] == list(range(START, START + COUNT))
    assert env.episode_attempts()[scene] == 2, "the same scene must be replayed, not skipped"


@pytest.mark.parametrize(
    ("label", "scene"),
    [("first", START), ("middle", START + COUNT // 2), ("final", START + COUNT - 1)],
)
def test_a_persistent_failure_terminates_with_explicit_incomplete_coverage(label, scene):
    """The gate: terminate and say so, rather than spin or substitute."""
    env = FrozenVecEnv(fail_scenes=(scene,), fail_times=99)
    row = _evaluate(env)

    assert row["incomplete_coverage"], f"{label}: no incomplete-coverage report"
    assert row["unrecovered_episodes"] == [scene]
    assert scene not in (row["scored_episodes"] or [])
    assert row["episodes"] == COUNT - 1
    # Retry limit enforced, not retried forever.
    assert env.episode_attempts()[scene] == 3
    # And crucially: no other scene was played in its place.
    assert set(row["scored_episodes"] or []) <= set(range(START, START + COUNT))


def test_no_scene_is_counted_twice():
    env = FrozenVecEnv(fail_scenes=(START + 1,), fail_times=1)
    row = _evaluate(env)
    scored = row["scored_episodes"] or []
    assert len(scored) == len(set(scored))


def test_a_failure_is_never_scored_as_an_agent_loss():
    """An excluded episode must not enter the denominator."""
    env = FrozenVecEnv(fail_scenes=(START + 2,), fail_times=99)
    row = _evaluate(env)
    # Five scenes all succeed; the sixth never scores. 5/5, not 5/6.
    assert row["episodes"] == COUNT - 1
    assert row["successes"] == COUNT - 1
    assert row["success_rate"] == 1.0


def test_a_clean_frozen_evaluation_is_unchanged():
    env = FrozenVecEnv()
    row = _evaluate(env)
    assert row["incomplete_coverage"] is None
    assert row["unrecovered_episodes"] is None
    assert row["episode_attempts"] is None
    assert row["episodes"] == COUNT
    assert row["scored_episodes"] == list(range(START, START + COUNT))


def test_the_fake_reuses_the_real_queue_methods():
    """If the real semantics change, these tests must break, not pass."""
    assert FrozenVecEnv.release_episode is FactorioVecEnv.release_episode
    assert FrozenVecEnv.pending_episodes is FactorioVecEnv.pending_episodes
    assert FrozenVecEnv.episode_attempts is FactorioVecEnv.episode_attempts
