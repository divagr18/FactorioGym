"""The substitution rule in `tools/solution_fragility.py`.

Only the driver is covered. The measure's two real defects were conceptual --
no no-op control, and scoring the solver's plan shape rather than the task --
and no unit test would have caught either; the control arm and `--restarts`
did. What is worth pinning is the part a later edit could silently break: that
exactly one action is replaced, that the replacement is one the mask allowed,
and that the control arm substitutes `wait` or nothing at all.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from solution_fragility import PerturbingDriver  # noqa: E402


class _Template:
    def __init__(self, key: str) -> None:
        self.key = key


class _Catalog:
    def __init__(self, keys) -> None:
        self.templates = [_Template(k) for k in keys]


class _Env:
    """The narrowest thing `Driver.do` will accept."""

    def __init__(self, keys, legal) -> None:
        self.catalog = _Catalog(keys)
        self._legal = legal
        self.taken: list[str] = []

    def action_masks(self):
        return np.array([k in self._legal for k in (t.key for t in self.catalog.templates)])

    def step(self, action: int):
        self.taken.append(self.catalog.templates[action].key)
        return None, 0.0, False, False, {}


class _Trace:
    steps = 0
    stuck_reason = None
    last_error = None
    action_log: list[str] = []

    def __init__(self) -> None:
        self.action_log = []


def _driver(keys, legal, perturb_at, forced=None, seed=0):
    env = _Env(keys, legal)
    return env, PerturbingDriver(
        env, _Trace(), perturb_at, np.random.default_rng(seed), forced=forced
    )


def test_exactly_one_action_is_replaced() -> None:
    keys = ["a", "b", "c", "wait"]
    env, driver = _driver(keys, set(keys), perturb_at=2)
    for key in ("a", "b", "c"):
        driver.do(key)
    assert driver.replaced == "b"
    assert driver.injected in {"a", "c", "wait"}
    # The first and third calls are untouched, and "b" never happens.
    assert env.taken[0] == "a"
    assert env.taken[2] == "c"
    assert "b" not in env.taken


def test_the_substitute_is_drawn_from_what_the_mask_allowed() -> None:
    # "c" is masked out, so it must never be chosen: an illegal substitute
    # would measure the action guard rather than the task.
    keys = ["a", "b", "c", "wait"]
    for seed in range(12):
        env, driver = _driver(keys, {"a", "b", "wait"}, perturb_at=1, seed=seed)
        driver.do("b")
        assert driver.injected in {"a", "wait"}


def test_the_control_arm_substitutes_wait_or_nothing() -> None:
    keys = ["a", "b", "wait"]
    env, driver = _driver(keys, set(keys), perturb_at=1, forced="wait")
    driver.do("a")
    assert driver.injected == "wait"

    # Where `wait` is not legal the trial is not evidence either way, so the
    # original action is taken and `perturbed` stays false for the caller.
    env, driver = _driver(keys, {"a", "b"}, perturb_at=1, forced="wait")
    driver.do("a")
    assert driver.injected is None
    assert env.taken == ["a"]


@pytest.mark.parametrize("perturb_at", [1, 3])
def test_a_step_with_no_alternative_is_left_alone(perturb_at: int) -> None:
    keys = ["a", "b"]
    env, driver = _driver(keys, {"a"}, perturb_at=perturb_at)
    for _ in range(3):
        driver.do("a")
    assert driver.injected is None
    assert env.taken == ["a", "a", "a"]
