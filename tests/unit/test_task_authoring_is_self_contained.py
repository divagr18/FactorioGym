"""A new task family must not have to edit framework internals.

R6's gate ends with "...and author a small task without editing internals", and
the repository claimed this was already true. `tasks/families/__init__.py` says
"adding a task means dropping a module here", and `tasks/__init__.py` says
discovery is import-based so "adding a task never requires a reinstall". Both
are true about *registration* and were false about the reference solver.

`RegisteredTask` has carried a `solve` field since it was written and nothing
read it. Dispatch went through a hard-coded `SOLVERS` dict in
`tasks/reference.py`, so a family that dropped in a module and declared its own
solver got `no reference solution registered for <id>` -- which makes
`tools/solvability.py` mark it **defective** and fails `gate_phase3`'s 0.99
reference threshold. Authoring a task really did require editing internals.

These tests pin the fixed contract: a task's own `solve` is used, `SOLVERS`
remains the fallback for the ten that legitimately live there, and a task with
neither is reported as having neither rather than being silently skipped.
"""

from __future__ import annotations

import dataclasses

import pytest

from factoriorl import tasks as tasks_module
from factoriorl.tasks import RegisteredTask
from factoriorl.tasks.reference import SOLVERS, _solver_for


def test_a_family_can_supply_its_own_solver(monkeypatch):
    """The property that makes the authoring claim true. Nothing about this
    task is known to `reference.py`."""

    def my_solver(driver):
        return None

    fake = RegisteredTask(spec=object(), generate=lambda f, r: None, solve=my_solver)
    monkeypatch.setattr(tasks_module, "get", lambda task_id: fake)
    assert _solver_for("a_brand_new_family") is my_solver


def test_a_declared_solver_wins_over_the_builtin_table(monkeypatch):
    """So a family can override a shipped solver without editing the table --
    and so the precedence is not accidentally the other way round, which would
    make the field look wired up while doing nothing for any existing id."""

    def mine(driver):
        return None

    fake = RegisteredTask(spec=object(), generate=lambda f, r: None, solve=mine)
    monkeypatch.setattr(tasks_module, "get", lambda task_id: fake)
    assert _solver_for("navigate") is mine
    assert SOLVERS["navigate"] is not mine, "the table itself must be untouched"


def test_the_builtin_table_still_serves_tasks_that_declare_nothing():
    """The ten shipped solvers share `Driver`, `_belt_gap`, `_pole_gap` and the
    rest. Moving them into family modules would duplicate that machinery, so
    the table stays as the fallback rather than being replaced."""
    for task_id in sorted(SOLVERS):
        assert tasks_module.get(task_id).solve is None, (
            f"{task_id} declares its own solver, so this test is no longer "
            "checking the fallback path"
        )
        assert _solver_for(task_id) is SOLVERS[task_id]


def test_every_registered_task_resolves_to_a_solver():
    """The condition `tools/solvability.py` turns into "defective" and
    `gate_phase3` turns into a failure. Worth asserting directly, engine-free,
    rather than discovering it from a gate that needs a worker."""
    missing = [
        task_id for task_id in sorted(tasks_module.all_tasks()) if _solver_for(task_id) is None
    ]
    assert missing == [], f"no reference solver resolves for {missing}"


def test_a_task_with_neither_is_reported_as_having_neither(monkeypatch):
    """The message has to name both places a solver could have come from, or
    the next person edits the dict without noticing the field exists."""
    monkeypatch.setattr(
        tasks_module, "get", lambda task_id: (_ for _ in ()).throw(KeyError(task_id))
    )
    assert _solver_for("nothing_anywhere") is None


def test_an_unregistered_task_does_not_raise(monkeypatch):
    """`_solver_for` is called from `solve`, which already has a `None` branch
    with a better message. Raising here would replace that with a KeyError
    traceback from two frames deeper."""

    def boom(task_id):
        raise RuntimeError("registry exploded")

    monkeypatch.setattr(tasks_module, "get", boom)
    assert _solver_for("navigate") is SOLVERS["navigate"], (
        "a registry failure must fall back to the table, not propagate"
    )


def test_the_solve_field_is_actually_declarable():
    """Guards against the field being removed as dead code by someone who
    greps for readers and finds only this indirection."""
    task = RegisteredTask(spec=object(), generate=lambda f, r: None)
    assert task.solve is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        # frozen dataclass: a family declares `solve` at construction, not by
        # mutation, so authoring stays a single expression in one file.
        task.solve = lambda driver: None
