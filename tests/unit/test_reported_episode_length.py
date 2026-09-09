"""The reported step count must be the env's, not the evaluation loop's.

`per_scene["steps"]` and `mean_episode_steps` came from an accumulator local to
`evaluate_parallel`, incremented once per `vec.step` and zeroed on each episode
boundary. That looks equivalent to the env's own `_steps` and is not: the
`EpisodeResetFailed` branch calls `vec.reset()` and `continue`, which zeroes
every env's `_steps` while skipping the `steps += 1` that would have tracked
it, so after one reset failure the two counters are permanently offset.

Why the env wins the tie. `env.py:744` compares `_steps` against
`max_decision_steps` to decide truncation, so it is the counter the
`truncated` flag is *about*. Reading anything else can publish a step count
that contradicts the flag sitting beside it in the same row -- which is what
happened: published rows carried counts as low as 12 alongside a truncation
flag, on a task whose only reachable truncation condition is 120 decisions.
A live episode confirmed the env side is sound (`info["steps"]` tracked
`env._steps` for all 120 decisions, truncation fired on the decision
condition, and the 15,000-tick budget was never approached at 30 ticks a
decision).

`loop_steps` is kept beside the corrected field rather than dropped, because a
disagreement between them means an episode boundary went unaccounted for in
the loop, and that is worth seeing rather than hiding.

Driven against a fake vec env: `evaluate_parallel` only calls `reset`,
`action_masks`, `step`, `pending_episodes` and `in_flight_episodes`, so no
engine is involved.
"""

from __future__ import annotations

import numpy as np

from factoriorl.learn.train import evaluate_parallel

START = 3000
ACTIONS = 3


class _Model:
    def predict(self, observation, action_masks=None, deterministic=True):
        n = len(action_masks) if action_masks is not None else 1
        return np.zeros(n, dtype=np.int64), None


class LengthVecEnv:
    """Plays each scene for a declared number of loop iterations.

    `reported_steps` is what the env claims in `info["steps"]`. Setting it
    different from the iteration count is the whole point: it is the only way
    to tell which counter the report actually reads.
    """

    def __init__(self, lengths: dict[int, int], reported: dict[int, int] | None = None):
        self.num_envs = 1
        self.lengths = lengths
        self.reported = reported or {}
        self._queue = list(lengths)
        self._live: int | None = None
        self._elapsed = 0

    def reset(self):
        self._next()
        return {"x": np.zeros((1, 1), dtype=np.float32)}

    def action_masks(self):
        return np.ones((1, ACTIONS), dtype=bool)

    def in_flight_episodes(self):
        return [self._live]

    def pending_episodes(self):
        return len(self._queue)

    def episode_attempts(self):
        return {}

    def _next(self):
        self._live = self._queue.pop(0) if self._queue else None
        self._elapsed = 0

    def step(self, actions):
        self._elapsed += 1
        episode = self._live
        done = episode is None or self._elapsed >= self.lengths.get(episode, 1)
        info: dict = {"episode_index": episode}
        if done and episode is not None:
            info["success"] = False
            info["TimeLimit.truncated"] = True
            info["steps"] = self.reported.get(episode, self._elapsed)
        out = (
            {"x": np.zeros((1, 1), dtype=np.float32)},
            np.array([0.0], dtype=np.float32),
            np.array([done]),
            [info],
        )
        if done:
            self._next()
        return out


def _run(env, episodes):
    return evaluate_parallel(env, _Model(), episodes, only_indices=set(env.lengths))


def test_the_reported_count_is_the_envs_not_the_loops():
    """The regression. The loop sees 4 iterations; the env says 120. A row that
    said 4 while flagging truncation on a 120-decision budget was the published
    contradiction."""
    env = LengthVecEnv(lengths={START: 4}, reported={START: 120})
    result = _run(env, 1)
    row = result["per_scene"][0]
    assert row["steps"] == 120
    assert row["loop_steps"] == 4


def test_the_loop_count_is_kept_so_a_mismatch_is_visible():
    """Dropping it would hide the fact that an episode boundary went
    unaccounted for, which is the symptom that found this."""
    env = LengthVecEnv(lengths={START: 3}, reported={START: 99})
    row = _run(env, 1)["per_scene"][0]
    assert row["loop_steps"] == 3 and row["steps"] == 99


def test_mean_episode_steps_follows_the_corrected_count():
    """The aggregate is built from the same list, so a fix that touched only
    `per_scene` would leave the summary wrong."""
    env = LengthVecEnv(
        lengths={START: 2, START + 1: 2},
        reported={START: 100, START + 1: 120},
    )
    result = _run(env, 2)
    assert result["mean_episode_steps"] == 110.0


def test_the_counts_agree_when_nothing_goes_wrong():
    """The common case must be unchanged: with no reset failure the loop and
    the env count the same decisions, and the fix must not perturb that."""
    env = LengthVecEnv(lengths={START: 5, START + 1: 7})
    result = _run(env, 2)
    for row in result["per_scene"]:
        assert row["steps"] == row["loop_steps"]
    assert sorted(row["steps"] for row in result["per_scene"]) == [5, 7]


def test_an_env_that_reports_no_step_count_falls_back_to_the_loop():
    """Older runs and any env that omits the key still get a number rather
    than a crash -- the fallback is the loop counter, which is what the field
    always was."""
    env = LengthVecEnv(lengths={START: 6})
    original = env.step

    def step(actions):
        observation, reward, done, infos = original(actions)
        for info in infos:
            info.pop("steps", None)
        return observation, reward, done, infos

    env.step = step
    row = _run(env, 1)["per_scene"][0]
    assert row["steps"] == 6 and row["loop_steps"] == 6
