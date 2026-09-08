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

from factoriorl.tasks.spec import (
    UNDECIDABLE_AT_RESET,
    Blueprint,
    LayoutFamily,
    TaskSpec,
)

#: family id -> (spec, generator, reference solution)
_REGISTRY: dict[str, RegisteredTask] = {}

#: Mirrors ``MAX_ADVANCE_TICKS`` in ``mod/factoriorl/runtime.lua``: the mod
#: refuses to advance more ticks than this in one request. Checking it here
#: turns a per-step protocol error mid-episode into a task that fails
#: ``tasks validate`` before a worker is launched.
MAX_ADVANCE_TICKS = 36000


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

    Includes an engine-free structural pass over sampled blueprints: in-box
    positions, non-overlapping footprints over real prototype tile sizes, and
    the success predicate must be False at reset, or the task is solved by its
    own initial conditions.

    The reset check has limits, and they are reported rather than papered over.
    Five of the seven predicate kinds are fully determined by the blueprint:
    contents, inventory, character position, and the two cumulative counters
    that `world.clear_statistics` empties at every episode start. `working` is
    not -- it depends on fuel and power networks -- so a task whose success
    rests on it appears in the report under `undecidable_success`, and the
    check is not claimed for it.

    Reachability is *not* checked here; that is `tools/generator_diagnostics.py`,
    which owns the BFS and the padded bounding box. The previous version of this
    docstring claimed both a reachable goal and the reset check, and neither
    existed in the code -- grep found only the sentence.
    """
    import random
    import zlib

    from factoriorl import catalog as catalog_module

    def stable_seed(*parts: object) -> int:
        """Deterministic across processes.

        Python's built-in hash() is randomized per process (PYTHONHASHSEED), so
        seeding validation with it sampled different blueprints on every run and
        made the suite pass or fail at random -- which is worse than a failing
        check, because it hides one.
        """
        return zlib.crc32("|".join(str(p) for p in parts).encode()) & 0xFFFFFFFF

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
        if spec.decision_ticks <= 0 or spec.decision_ticks > MAX_ADVANCE_TICKS:
            problems.append(
                f"decision_ticks must be in [1, {MAX_ADVANCE_TICKS}], got "
                f"{spec.decision_ticks}; the mod refuses anything else at runtime"
            )
        elif spec.decision_ticks < catalog_module.LONG_MOVE_TICKS:
            # A move's duration is enforced by the mod (actions.lua sets
            # deadline_tick = game.tick + payload.ticks, inflight.lua stops the
            # walk there), independently of how long a decision lasts. Only
            # another move supersedes a running move, so with a decision
            # interval shorter than the longest stride the *next* action --
            # place, transfer, craft or wait -- executes while the character is
            # still walking. Placement binds to floor(position), so those
            # placements land on whatever tile the walk happened to reach and
            # nothing raises: episodes are corrupted non-deterministically and
            # the run merely looks unlucky. The invariant currently holds with
            # zero margin (30 == 30), which is exactly why it has to be
            # asserted rather than remembered.
            problems.append(
                f"decision_ticks {spec.decision_ticks} is shorter than the longest "
                f"catalog move stride ({catalog_module.LONG_MOVE_TICKS} ticks), so a "
                "move is still running when the next action executes"
            )
        splits = {f.split for f in spec.layout_families}
        if "train" not in splits:
            problems.append("no train layout family")
        if "test" not in splits:
            problems.append("no held-out (test) layout family")

        # A conjunction is already satisfied only if *every* clause is. Where a
        # clause cannot be decided engine-free, satisfying the rest is not
        # proof the scene is solved -- so those tasks are flagged as uncovered
        # and the conjunction over the decidable clauses is what is tested.
        decidable = [p for p in spec.success if p.kind not in UNDECIDABLE_AT_RESET]
        undecidable = [p for p in spec.success if p.kind in UNDECIDABLE_AT_RESET]

        for family in spec.layout_families:
            for index in range(sample_seeds):
                rng = random.Random(stable_seed(task_id, family.name, index))
                try:
                    blueprint = task.generate(family, rng)
                except Exception as exc:  # noqa: BLE001 - report, do not raise
                    problems.append(f"{family.name}[{index}] generator raised: {exc}")
                    continue
                problems.extend(f"{family.name}[{index}] {p}" for p in blueprint.out_of_box())
                problems.extend(
                    f"{family.name}[{index}] {p}" for p in blueprint.footprint_conflicts()
                )
                observation, truth = blueprint.initial_state()
                if decidable and all(p.evaluate(observation, truth) for p in decidable):
                    detail = ", ".join(p.describe() for p in decidable)
                    problems.append(
                        f"{family.name}[{index}] success already satisfied at reset "
                        f"by the scene's own contents: {detail}"
                        + ("" if not undecidable else " (plus undecidable clauses)")
                    )

        report["tasks"][task_id] = {
            "version": spec.version,
            "families": [f.name for f in spec.layout_families],
            "problems": problems,
            # Which success clauses the reset check could not decide, and so
            # did not verify. Empty for six of the seven tasks.
            "undecidable_success": [p.describe() for p in undecidable],
            "reset_check_covers_success": not undecidable,
            "ok": not problems,
        }
        if problems:
            report["ok"] = False
    return report


__all__ = [
    "MAX_ADVANCE_TICKS",
    "RegisteredTask",
    "TaskConfigError",
    "all_tasks",
    "discover",
    "get",
    "register",
    "validate_all",
]
