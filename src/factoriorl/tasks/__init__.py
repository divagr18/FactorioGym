"""Task registry (PLAN.md 3.1).

Discovery-based, not entry-point based: a family module dropped into
``factoriorl.tasks.families`` registers itself on import, so adding a task never
requires a reinstall and never touches worker management. The import-direction
test in ``tests/unit/test_task_isolation.py`` asserts the second half of that.

``validate_all()`` runs before any worker is launched, so a configuration error
fails at construction rather than at step 40,000.
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable
from dataclasses import dataclass

from factoriorl.tasks.spec import Blueprint, LayoutFamily, TaskSpec

#: family id -> (spec, generator, reference solution)
_REGISTRY: dict[str, RegisteredTask] = {}


@dataclass(frozen=True)
class RegisteredTask:
    spec: TaskSpec
    #: (family, rng) -> Blueprint
    generate: Callable[[LayoutFamily, object], Blueprint]
    #: Optional scripted solution: (env) -> bool. Used by the gate and the
    #: solvability suite, never importable from the training entrypoint.
    solve: Callable | None = None


def register(task: RegisteredTask) -> RegisteredTask:
    if task.spec.id in _REGISTRY:
        raise ValueError(f"duplicate task id: {task.spec.id}")
    _REGISTRY[task.spec.id] = task
    return task


def discover() -> dict[str, RegisteredTask]:
    """Import every family module so it can register itself."""
    from factoriorl.tasks import families

    for module in pkgutil.iter_modules(families.__path__):
        importlib.import_module(f"{families.__name__}.{module.name}")
    return dict(_REGISTRY)


def get(task_id: str) -> RegisteredTask:
    if not _REGISTRY:
        discover()
    if task_id not in _REGISTRY:
        raise TaskConfigError(f"unknown task: {task_id} (known: {sorted(_REGISTRY)})")
    return _REGISTRY[task_id]


def all_tasks() -> dict[str, RegisteredTask]:
    if not _REGISTRY:
        discover()
    return dict(_REGISTRY)


class TaskConfigError(ValueError):
    """A task is misconfigured. Raised before any worker is launched."""


def validate_all(sample_seeds: int = 16) -> dict:
    """Validate every registered task, reporting *all* problems at once.

    Includes an engine-free structural pass over sampled blueprints: in-box,
    non-overlapping footprints, a reachable goal, and -- importantly -- the
    success predicate must be False at reset, or the task is solved by its own
    initial conditions.
    """
    import random

    tasks = all_tasks()
    report: dict = {"tasks": {}, "ok": True}
    if not tasks:
        report["ok"] = False
        report["problems"] = ["no tasks registered"]
        return report

    for task_id, task in sorted(tasks.items()):
        problems: list[str] = []
        spec = task.spec

        if not spec.success:
            problems.append("no success predicate")
        sparse = [r for r in spec.rewards if r.kind.value == "sparse_success"]
        if len(sparse) != 1:
            problems.append(f"expected exactly one sparse_success reward, found {len(sparse)}")
        if spec.max_decision_steps <= 0 or spec.max_game_ticks <= 0:
            problems.append("budgets must be positive, or truncation is undefined")
        splits = {f.split for f in spec.layout_families}
        if "train" not in splits:
            problems.append("no train layout family")
        if "test" not in splits:
            problems.append("no held-out (test) layout family")

        for family in spec.layout_families:
            for index in range(sample_seeds):
                rng = random.Random(hash((task_id, family.name, index)) & 0xFFFFFFFF)
                try:
                    blueprint = task.generate(family, rng)
                except Exception as exc:  # noqa: BLE001 - report, do not raise
                    problems.append(f"{family.name}[{index}] generator raised: {exc}")
                    continue
                problems.extend(f"{family.name}[{index}] {p}" for p in blueprint.out_of_box())
                problems.extend(
                    f"{family.name}[{index}] {p}" for p in blueprint.footprint_conflicts()
                )

        report["tasks"][task_id] = {
            "version": spec.version,
            "families": [f.name for f in spec.layout_families],
            "problems": problems,
            "ok": not problems,
        }
        if problems:
            report["ok"] = False
    return report


__all__ = [
    "RegisteredTask",
    "TaskConfigError",
    "all_tasks",
    "discover",
    "get",
    "register",
    "validate_all",
]
