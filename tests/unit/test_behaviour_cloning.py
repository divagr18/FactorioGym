"""Behaviour cloning had no tests at all, and one gate nothing checked.

`learn/bc.py` has been run twice and published nothing, with no coverage. Two
of its properties are load-bearing enough that a silent regression in either
would invalidate every number the module produces.

**Only solved episodes contribute pairs.** A stuck solver produces a trajectory
of an agent failing, and cloning that teaches failing. The failures still have
to be *counted*, because a set whose solver succeeded half the time is a
different object from one where it always did -- so dropping them from the
labels while keeping them out of the outcome log would hide exactly the thing
that makes a demonstration set weak.

**Training traces and evaluation scenes are disjoint.** R5.3 requires it and
nothing verified it. The subtlety is what a scene *is*: it is a pure function of
(master, run_id, branch, index), so demonstrations from `Branch.TRAIN` at
indices 0..N and an evaluation row from `Branch.EVAL` over the same indices are
different scenes despite identical numbers. A check on indices alone would
report a collision that does not exist, and would miss the one that matters --
an evaluation row moved onto the demonstration branch.

These use fakes rather than an engine. `collect` needs only `reset`, an
`unwrapped` carrying `branch`/`split`/`_episode_index`, and a solver returning
something with `succeeded`; that is enough to pin the contract without a
Factorio session.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from factoriorl.learn.bc import Demonstrations, collect, disjointness


class FakeUnwrapped:
    def __init__(self, branch: str, split: str) -> None:
        self.branch = branch
        self.split = split
        self._episode_index = -1


class FakeEnv:
    """The smallest env `collect` actually uses."""

    def __init__(self, branch: str = "train", split: str = "train") -> None:
        self.unwrapped = FakeUnwrapped(branch, split)
        self.resets = 0

    def reset(self):
        self.unwrapped._episode_index += 1
        self.resets += 1
        return {}, {}


@dataclass
class FakeTrace:
    succeeded: bool
    stuck_reason: str | None = None


def solver(pattern: list[bool], pairs_each: int = 3):
    """A solver that succeeds or fails per episode, recording `pairs_each` steps."""
    calls = {"n": 0}

    def solve(recorder):
        index = calls["n"]
        calls["n"] += 1
        for step in range(pairs_each):
            recorder.pairs.append(({"grid": [step]}, step))
        outcome = pattern[index % len(pattern)]
        return FakeTrace(succeeded=outcome, stuck_reason=None if outcome else "no_path")

    return solve


# --- what gets cloned -------------------------------------------------------


def test_only_solved_episodes_contribute_labels():
    """Cloning a failure teaches failing, which is the one thing a
    demonstration set must not do."""
    env = FakeEnv()
    data = collect(env, solver([True, False, True, False]), episodes=4)
    assert data.episodes == 4
    assert data.solved == 2
    assert len(data) == 6, "3 pairs from each of the 2 solved episodes"


def test_failures_are_still_counted():
    """A set whose solver succeeded half the time is a different object from one
    where it always did, and the summary has to say which."""
    env = FakeEnv()
    data = collect(env, solver([True, False]), episodes=4)
    summary = data.summary()
    assert summary["episodes"] == 4 and summary["solved"] == 2
    assert summary["solved_rate"] == 0.5
    assert [o["solved"] for o in data.outcomes] == [True, False, True, False]
    assert data.outcomes[1]["stuck_reason"] == "no_path"


def test_an_all_failing_solver_yields_no_labels_rather_than_bad_ones():
    env = FakeEnv()
    data = collect(env, solver([False]), episodes=3)
    assert len(data) == 0 and data.episodes == 3 and data.solved == 0
    assert data.summary()["solved_rate"] == 0.0


def test_an_empty_set_reports_zero_instead_of_dividing_by_it():
    assert Demonstrations().summary()["solved_rate"] == 0.0
    assert Demonstrations().summary()["episode_index_range"] is None


# --- which scenes were demonstrated -----------------------------------------


def test_the_demonstrated_scenes_are_recorded_with_their_branch():
    """Half a scene's identity is its branch, so recording indices alone would
    leave the disjointness question unanswerable after the fact."""
    env = FakeEnv(branch="train", split="train")
    data = collect(env, solver([True]), episodes=3)
    assert data.scenes == [("train", 0), ("train", 1), ("train", 2)]
    assert data.split == "train"
    assert data.summary()["branches"] == ["train"]
    assert data.summary()["episode_index_range"] == [0, 2]


def test_first_index_moves_the_recorded_scenes_with_it():
    """`first_index` exists so demonstrations can be drawn from a range that
    avoids an evaluation row. If the recorded scenes did not move with it, the
    record would describe the wrong scenes."""
    env = FakeEnv()
    data = collect(env, solver([True]), episodes=2, first_index=500)
    assert data.scenes == [("train", 500), ("train", 501)]


def test_the_scene_is_read_off_the_env_not_counted_locally():
    """So a scene the env skipped is recorded as the one actually demonstrated,
    rather than as whatever a local counter assumed.

    The two readings only diverge when the env's index does not advance by one:
    counting locally from `first_index` would say 7 and 8, while reading the
    env says 8 and 10. Without a skipping env the test could not tell the
    implementations apart, and would have passed either way.
    """

    class SkippingEnv(FakeEnv):
        def reset(self):
            self.unwrapped._episode_index += 2
            return {}, {}

    env = SkippingEnv()
    data = collect(env, solver([True]), episodes=2, first_index=7)
    assert data.scenes == [("train", 8), ("train", 10)]


# --- R5.3's gate ------------------------------------------------------------


def test_the_same_indices_on_a_different_branch_are_not_an_overlap():
    """The whole reason the check is on (branch, index). Demonstrations come
    from Branch.TRAIN at 0..N and every evaluation row from Branch.EVAL over the
    same numbers; those are different scenes, and an index-only check would
    report a collision that does not exist."""
    data = Demonstrations(scenes=[("train", i) for i in range(10)])
    report = disjointness(data, {"seeds": {"branch": "eval", "episode_indices": list(range(10))}})
    assert report["disjoint"] is True
    assert report["rows"]["seeds"]["overlapping_scenes"] == []


def test_an_evaluation_row_on_the_demonstration_branch_is_caught():
    """The collision that would matter: a row moved onto the demonstration
    branch scores the policy on scenes it was trained to imitate."""
    data = Demonstrations(scenes=[("train", i) for i in range(10)])
    report = disjointness(data, {"seeds": {"branch": "train", "episode_indices": [8, 9, 10, 11]}})
    assert report["disjoint"] is False
    assert report["rows"]["seeds"]["overlapping_scenes"] == [8, 9]


def test_a_row_that_cannot_be_checked_is_not_reported_as_clean():
    """An unfalsifiable claim must not read as a satisfied one -- that is how a
    gate becomes decorative."""
    data = Demonstrations(scenes=[("train", 0)])
    report = disjointness(data, {"seeds": {"branch": "eval"}})
    assert report["rows"]["seeds"]["checked"] is False
    assert report["all_checked"] is False
    assert "cannot be determined" in report["rows"]["seeds"]["why"]


def test_no_rows_at_all_is_not_disjoint():
    """Vacuous truth is the failure mode here: `all()` over nothing is True, so
    a run that evaluated nothing would otherwise pass the gate."""
    data = Demonstrations(scenes=[("train", 0)])
    report = disjointness(data, {})
    assert report["disjoint"] is False
    assert report["all_checked"] is False


def test_every_row_is_checked_not_just_the_first():
    data = Demonstrations(scenes=[("train", 5)])
    report = disjointness(
        data,
        {
            "seeds": {"branch": "eval", "episode_indices": [5]},
            "val": {"branch": "eval", "episode_indices": [5]},
            "structures": {"branch": "train", "episode_indices": [5]},
        },
    )
    assert report["disjoint"] is False
    assert report["rows"]["structures"]["overlapping_scenes"] == [5]
    assert report["rows"]["seeds"]["disjoint"] is True
    assert report["rows"]["val"]["disjoint"] is True


def test_the_report_states_what_a_scene_is():
    """Because the next person to read this will reach for indices."""
    report = disjointness(Demonstrations(scenes=[("train", 0)]), {})
    assert "an index alone is not a scene" in report["scene_identity"]
    assert "R5.3" in report["requirement"]


@pytest.mark.parametrize("branch", ["train", "eval"])
def test_demonstrations_from_either_branch_are_recorded_faithfully(branch):
    env = FakeEnv(branch=branch, split="train")
    data = collect(env, solver([True]), episodes=2)
    assert {b for b, _ in data.scenes} == {branch}
