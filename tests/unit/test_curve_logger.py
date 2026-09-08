"""The curve logger must attribute episodes to the worker that finished them.

The first version summed rewards across the whole vector, counted vector steps
as episode steps, and read `infos[0]` for the success flag. On an eight-worker
run that reported a success in workers 1..7 as a zero, and two task families
were declared unlearnable off curves that in fact contained successes.
"""

import numpy as np
import pytest

from factoriorl.learn.train import CurveLogger


class _Locals(dict):
    """Stand-in for `BaseCallback.locals`."""


def _logger(tmp_path, num_envs):
    curve = CurveLogger(tmp_path / "curve.csv", started=0.0, num_envs=num_envs)
    curve.num_timesteps = 0
    return curve


def _step(curve, rewards, dones, infos):
    curve.locals = _Locals(rewards=np.asarray(rewards), dones=np.asarray(dones), infos=infos)
    curve.num_timesteps += len(rewards)
    curve._on_step()


def _blank(n):
    return [{} for _ in range(n)]


def test_success_in_a_late_worker_is_recorded(tmp_path):
    """The exact bug: only worker 0's flag was read."""
    curve = _logger(tmp_path, 4)
    _step(curve, [0.0] * 4, [False] * 4, _blank(4))
    infos = _blank(4)
    infos[3] = {"success": True, "layout_family": "gap_far"}
    _step(curve, [0.0, 0.0, 0.0, 1.0], [False, False, False, True], infos)

    assert len(curve.rows) == 1
    row = curve.rows[0]
    assert row["worker"] == 3
    assert row["success"] == 1
    assert row["layout_family"] == "gap_far"


def test_reward_is_per_episode_not_a_vector_sum(tmp_path):
    """Each worker's return is its own; a full vector must not be summed."""
    curve = _logger(tmp_path, 3)
    for _ in range(5):
        _step(curve, [-0.1, -0.2, -0.3], [False] * 3, _blank(3))
    infos = _blank(3)
    infos[1] = {"success": True}
    _step(curve, [0.0, 1.0, 0.0], [False, True, False], infos)

    assert len(curve.rows) == 1
    row = curve.rows[0]
    # worker 1 collected 5 x -0.2 then +1.0, and nothing from workers 0 and 2.
    assert row["episode_reward"] == pytest.approx(0.0, abs=1e-9)
    assert row["episode_steps"] == 6


def test_episode_steps_counts_env_steps_not_vector_steps(tmp_path):
    """A 300-step family used to report ~300/workers."""
    curve = _logger(tmp_path, 8)
    for _ in range(300):
        _step(curve, [0.0] * 8, [False] * 8, _blank(8))
    _step(curve, [0.0] * 8, [True] + [False] * 7, _blank(8))
    assert curve.rows[0]["episode_steps"] == 301


def test_simultaneous_dones_emit_one_row_each(tmp_path):
    curve = _logger(tmp_path, 4)
    infos = _blank(4)
    infos[0] = {"success": True}
    infos[2] = {"success": False}
    _step(curve, [1.0, 0.0, -1.0, 0.0], [True, False, True, False], infos)

    assert [row["worker"] for row in curve.rows] == [0, 2]
    assert [row["success"] for row in curve.rows] == [1, 0]


def test_accumulators_reset_independently(tmp_path):
    """A finished episode must not bleed reward into its worker's next one."""
    curve = _logger(tmp_path, 2)
    _step(curve, [5.0, 7.0], [True, False], _blank(2))
    _step(curve, [1.0, 1.0], [True, True], _blank(2))

    rows = curve.rows
    assert rows[0]["episode_reward"] == pytest.approx(5.0)
    assert rows[1]["episode_reward"] == pytest.approx(1.0)  # worker 0, fresh
    assert rows[2]["episode_reward"] == pytest.approx(8.0)  # worker 1, both steps


def test_summary_reports_totals_not_only_the_tail(tmp_path):
    """`final_train_success_rate` reported 24 successes as 0.0."""
    curve = _logger(tmp_path, 1)
    for index in range(50):
        info = {"success": index < 5}  # all successes early, none in the tail
        _step(curve, [0.0], [True], [info])

    summary = curve.summary()
    assert summary["episodes_scored"] == 50
    assert summary["train_successes"] == 5
    assert summary["train_success_rate"] == pytest.approx(0.1)
    assert summary["final_train_success_rate"] == pytest.approx(0.0)


def test_excluded_episodes_are_logged_but_not_scored(tmp_path):
    curve = _logger(tmp_path, 1)
    _step(curve, [0.0], [True], [{"success": True, "excluded_from_metrics": True}])
    _step(curve, [0.0], [True], [{"success": True}])

    summary = curve.summary()
    assert summary["episodes_logged"] == 2
    assert summary["episodes_scored"] == 1
    assert summary["train_successes"] == 1


def test_vector_wider_than_declared_grows_rather_than_misattributing(tmp_path):
    curve = _logger(tmp_path, 1)
    _step(curve, [0.0, 0.0, 3.0], [False, False, True], _blank(3))
    assert curve.rows[0]["worker"] == 2
    assert curve.rows[0]["episode_reward"] == pytest.approx(3.0)
