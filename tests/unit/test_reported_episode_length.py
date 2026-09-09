"""Two step counts, both correct, counting different things.

`evaluate_parallel` tracks one number per `vec.step` -- the count of **policy
decisions** -- while `info["steps"]` is the environment's `_steps`, which counts
**primitive actions**. In the primitive action space they agree. Under skills
they do not: `SkillEnv.step` runs a whole skill through
`self.runner.run(skill)`, so one decision expands into many primitive steps and
a solved `deliver` episode reads 4 decisions against 18 primitive steps.

Conflating them cost an evening and a wrong published claim. `env.py:744`
compares `_steps` against `max_decision_steps`, so that budget is a *primitive*
budget despite its name, and a skills policy gets far fewer decisions than the
number suggests. Published rows showing truncation at 12-118 decisions against
a "120" budget were consistent all along; they looked like a defect only
because the two counts were assumed to be the same quantity.

So `steps` is the decision count -- "where did the attempt fail" is a question
about decisions the policy made -- and `primitive_steps` is recorded beside it,
because the ratio between them is what a reader needs to interpret either.

Driven against a fake vec env: `evaluate_parallel` only calls `reset`,
`action_masks`, `step`, `pending_episodes`, `in_flight_episodes` and
`episode_attempts`, so no engine is involved.
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
    """Plays each scene for a declared number of decisions.

    `primitive` is what the env claims in `info["steps"]`. Setting it different
    from the decision count is the whole point: it is how a skills expansion
    looks, and the only way to tell which count each field reports.
    """

    def __init__(self, decisions: dict[int, int], primitive: dict[int, int] | None = None):
        self.num_envs = 1
        self.decisions = decisions
        self.primitive = primitive or {}
        self._queue = list(decisions)
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
        done = episode is None or self._elapsed >= self.decisions.get(episode, 1)
        info: dict = {"episode_index": episode}
        if done and episode is not None:
            info["success"] = False
            info["TimeLimit.truncated"] = True
            info["steps"] = self.primitive.get(episode, self._elapsed)
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
    return evaluate_parallel(env, _Model(), episodes, only_indices=set(env.decisions))


def test_steps_is_the_decision_count_not_the_primitive_count():
    """The correction. A skills episode makes 4 decisions that expand into 18
    primitive steps, and `steps` must be the 4 -- reporting 18 answers a
    question nobody asked of a per-scene diagnostic."""
    env = LengthVecEnv(decisions={START: 4}, primitive={START: 18})
    row = _run(env, 1)["per_scene"][0]
    assert row["steps"] == 4
    assert row["primitive_steps"] == 18


def test_the_primitive_count_is_recorded_beside_it():
    """Without it, a 4 and a 120 in the same column are indistinguishable
    between action spaces, and the truncation flag looks inconsistent."""
    env = LengthVecEnv(decisions={START: 12}, primitive={START: 120})
    row = _run(env, 1)["per_scene"][0]
    assert row["steps"] == 12 and row["primitive_steps"] == 120


def test_the_two_agree_in_the_primitive_action_space():
    """One decision is one primitive step when no skill is involved, so the
    ratio is 1 and neither field is surprising."""
    env = LengthVecEnv(decisions={START: 5, START + 1: 7})
    result = _run(env, 2)
    for row in result["per_scene"]:
        assert row["steps"] == row["primitive_steps"]
    assert sorted(row["steps"] for row in result["per_scene"]) == [5, 7]


def test_mean_episode_steps_is_in_decisions():
    """The aggregate is built from the same list, so it has to report the same
    quantity as the field beside it."""
    env = LengthVecEnv(
        decisions={START: 4, START + 1: 6},
        primitive={START: 100, START + 1: 120},
    )
    assert _run(env, 2)["mean_episode_steps"] == 5.0


def test_an_env_reporting_no_primitive_count_falls_back_to_decisions():
    """Any env that omits the key still yields a number rather than a crash,
    and the fallback keeps the two fields consistent."""
    env = LengthVecEnv(decisions={START: 6})
    original = env.step

    def step(actions):
        observation, reward, done, infos = original(actions)
        for info in infos:
            info.pop("steps", None)
        return observation, reward, done, infos

    env.step = step
    row = _run(env, 1)["per_scene"][0]
    assert row["steps"] == 6 and row["primitive_steps"] == 6
