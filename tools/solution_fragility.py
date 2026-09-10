"""How many ways are there to solve this? Measured, not declared.

Factorion (github.com/beyarkay/factorion) draws a distinction this repository
needed and did not have: belt routing is **disjunctive** -- any connected path
works, so partial progress counts -- while crafting is **conjunctive** -- every
component must be simultaneously right for *any* reward. Their TRIAL lessons,
the conjunctive ones, score 0.11 at depth 1 and exactly 0 deeper.

It explains two of our results better than anything we had written. Behaviour
cloning scores **1.00** on `restore_power` and **0.17/0.13** on `repair_belt`,
same method, same demonstration count, both replicated across seeds. We had
been calling that "a property of the tasks" without naming the property.
`restore_power` asks for one placement at a published marker; `repair_belt`
asks the solver to infer the belt's modal row, place, *and* fix rotation, and
being right about two of three pays nothing.

Measuring it rather than labelling it
-------------------------------------
A `conjunctive: true` field on `TaskSpec` would be an assertion, and this
repository has been bitten repeatedly by assertions that drifted from the code
they described -- a `--holdout` flag that was never read, a docstring promising
a disruption check that did not exist. So this measures the property instead.

The operational definition: **a task is disjunctive to the extent that a wrong
move is recoverable.** Many solutions means a mistake leaves others reachable;
one solution means any deviation is fatal. So inject a single legal-but-wrong
action into a reference solve, let the solver carry on from the state that
produces, and see whether it still reaches success.

    fragility = P(reference solve fails | one wrong action injected)

Near 0 is disjunctive, near 1 is conjunctive. The reference solvers are the
right instrument because several already re-scan rather than replaying a fixed
plan -- `solve_repair_belt` loops up to six times re-finding the gap -- so a
recoverable task genuinely gets recovered rather than failing on a stale plan.

What this does not measure: whether a *policy* can find any of those solutions.
A disjunctive task can still be unlearnable. This is a property of the task, in
the same family as the random floor `tools/solvability.py` reports, and it is
worth reading beside that number rather than instead of it.

Run:
  uv run python tools/solution_fragility.py --tasks restore_power,repair_belt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from factoriorl.engine_config import resolve_game_speed  # noqa: E402
from factoriorl.env import FactorioEnv  # noqa: E402
from factoriorl.rcon import RCONClient  # noqa: E402
from factoriorl.seeding import Branch, SeedPlan  # noqa: E402
from factoriorl.session import WorkerSession  # noqa: E402
from factoriorl.tasks import all_tasks, get  # noqa: E402
from factoriorl.tasks.reference import Driver, SolveTrace, _solver_for  # noqa: E402
from factoriorl.worker import WorkerManager  # noqa: E402


class PerturbingDriver(Driver):
    """A `Driver` that substitutes one wrong action, then behaves normally.

    The substitution is drawn from the actions the *mask* permits, so it is a
    move the policy could legally have made -- not an illegal one the engine
    would refuse outright, which would measure the action guard rather than the
    task. The solver then continues from whatever state that produced.
    """

    def __init__(self, env, trace, perturb_at: int, rng) -> None:
        super().__init__(env, trace)
        self.perturb_at = perturb_at
        self.rng = rng
        self.calls = 0
        self.injected: str | None = None
        self.replaced: str | None = None

    def do(self, key: str, **arguments) -> bool:
        self.calls += 1
        if self.calls != self.perturb_at or self.injected is not None:
            return super().do(key, **arguments)
        mask = np.asarray(self.env.action_masks())
        legal = [
            template.key
            for i, template in enumerate(self.env.catalog.templates)
            if i < len(mask) and mask[i] and template.key != key
        ]
        if not legal:
            # Nothing else was legal here, so this step admits no perturbation
            # and the trial is not evidence either way. Recorded by the caller.
            return super().do(key, **arguments)
        self.replaced = key
        self.injected = str(self.rng.choice(legal))
        # Deliberately no arguments: a parameterised substitute would need a
        # value from the live domain, and the point is a plausible wrong move,
        # not a well-formed one.
        super().do(self.injected)
        return not (self.terminated or self.truncated)


def solve_length(task, session, plan, split: str, index: int) -> tuple[int, bool]:
    """Baseline: how many actions the unperturbed reference solve takes."""
    env = FactorioEnv(task, session, plan, branch=Branch.TRAIN, split=split)
    env._episode_index = index - 1
    env.reset()
    trace = SolveTrace(task=task.spec.id, budget=task.spec.max_decision_steps)
    driver = Driver(env, trace)
    solver = _solver_for(task.spec.id)
    solver(driver)
    # `trace.steps` is incremented inside `Driver.do`, so it is the action count
    # rather than a wall-clock or tick count.
    return trace.steps, driver.success


def trial(task, session, plan, split: str, index: int, perturb_at: int, rng) -> dict:
    env = FactorioEnv(task, session, plan, branch=Branch.TRAIN, split=split)
    env._episode_index = index - 1
    env.reset()
    trace = SolveTrace(task=task.spec.id, budget=task.spec.max_decision_steps)
    driver = PerturbingDriver(env, trace, perturb_at, rng)
    solver = _solver_for(task.spec.id)
    try:
        solver(driver)
    except Exception as exc:  # a solver may assume a state its plan no longer holds
        return {
            "perturb_at": perturb_at,
            "perturbed": driver.injected is not None,
            "recovered": False,
            "solver_error": f"{type(exc).__name__}: {str(exc)[:90]}",
        }
    return {
        "perturb_at": perturb_at,
        "perturbed": driver.injected is not None,
        "replaced": driver.replaced,
        "injected": driver.injected,
        "recovered": bool(driver.success),
    }


def measure(task, session, plan, split: str, episodes: int, per_episode: int, rng) -> dict:
    scenes = []
    for index in range(episodes):
        length, solved = solve_length(task, session, plan, split, index)
        if not solved or length < 2:
            scenes.append({"episode": index, "skipped": "reference did not solve it"})
            continue
        # Spread the injection points over the solve rather than clustering at
        # the start: a task can be forgiving early and unforgiving late.
        points = sorted({int(p) for p in np.linspace(1, length, per_episode, dtype=int)})
        trials = [trial(task, session, plan, split, index, p, rng) for p in points]
        usable = [t for t in trials if t["perturbed"]]
        scenes.append(
            {
                "episode": index,
                "reference_actions": length,
                "trials": len(usable),
                "recovered": sum(1 for t in usable if t["recovered"]),
                "detail": trials,
            }
        )
    usable = [s for s in scenes if s.get("trials")]
    total = sum(s["trials"] for s in usable)
    recovered = sum(s["recovered"] for s in usable)
    return {
        "episodes": len(usable),
        "trials": total,
        "recovered": recovered,
        "fragility": round(1 - recovered / total, 4) if total else None,
        "mean_reference_actions": (
            round(sum(s["reference_actions"] for s in usable) / len(usable), 1) if usable else None
        ),
        "scenes": scenes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tasks", default=",".join(sorted(all_tasks())))
    parser.add_argument("--splits", default="train,test")
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument(
        "--perturbations",
        type=int,
        default=5,
        help="injection points per scene, spread across the reference solve",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    task_ids = [t.strip() for t in args.tasks.split(",") if t.strip()]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    rng = np.random.default_rng(args.seed)

    manager = WorkerManager()
    handle = manager.launch("fragility")
    started = time.perf_counter()
    report: dict = {
        "measures": "P(reference solve fails | one legal-but-wrong action injected)",
        "reading": (
            "near 0 is disjunctive -- many solutions, a wrong move is recoverable. "
            "near 1 is conjunctive -- one solution, any deviation is fatal"
        ),
        "not_measured": (
            "whether a policy can find any of those solutions. A disjunctive task "
            "can still be unlearnable; read this beside the random floor, not "
            "instead of it"
        ),
        "credit": "framing from github.com/beyarkay/factorion",
        "tasks": {},
    }
    try:
        with RCONClient(handle.spec.rcon_endpoint, timeout=30.0) as client:
            client.lua(f"game.speed = {resolve_game_speed()} return game.speed")
        session = WorkerSession(handle, timeout=30.0)
        session.status()
        plan = SeedPlan(master=args.seed, run_id="fragility")

        for task_id in task_ids:
            task = get(task_id)
            entry: dict = {"version": task.spec.version, "splits": {}}
            for split in splits:
                if not task.spec.families(split):
                    continue
                result = measure(task, session, plan, split, args.episodes, args.perturbations, rng)
                entry["splits"][split] = result
                print(
                    f"{task_id:16s} {split:6s} fragility={result['fragility']} "
                    f"({result['recovered']}/{result['trials']} recovered, "
                    f"ref={result['mean_reference_actions']} actions)",
                    flush=True,
                )
            report["tasks"][task_id] = entry
    finally:
        manager.cleanup(handle)

    report["wall_seconds"] = round(time.perf_counter() - started, 1)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
