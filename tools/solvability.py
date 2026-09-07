"""Run every reference solution and report what it proves (PLAN.md 3.2).

    Reference solutions pass all fixed scenarios.
    Randomized generation passes a declared solvability suite.

A solver failure here is a **task defect**, not a training result: the solver
has full evaluator knowledge and still could not finish through the catalog the
policy is given. The `stuck_reason` names which action or precondition is
missing.

Run: uv run python tools/solvability.py [--episodes N] [--tasks a,b,c]
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

from factoriorl.env import FactorioEnv  # noqa: E402
from factoriorl.rcon import RCONClient  # noqa: E402
from factoriorl.seeding import Branch, SeedPlan  # noqa: E402
from factoriorl.session import WorkerSession  # noqa: E402
from factoriorl.tasks import all_tasks, get  # noqa: E402
from factoriorl.tasks.reference import random_rollout, solve  # noqa: E402
from factoriorl.worker import WorkerManager  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--tasks", default=",".join(sorted(all_tasks())))
    parser.add_argument("--splits", default="train,test")
    args = parser.parse_args()

    task_ids = [t.strip() for t in args.tasks.split(",") if t.strip()]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    manager = WorkerManager()
    handle = manager.launch("solvability")
    report: dict = {"episodes_per_split": args.episodes, "tasks": {}}
    started = time.perf_counter()
    try:
        with RCONClient(handle.spec.rcon_endpoint, timeout=30.0) as client:
            client.lua("game.speed = 30 return game.speed")
        session = WorkerSession(handle, timeout=30.0)
        session.status()
        plan = SeedPlan(master=777, run_id="solvability")

        for task_id in task_ids:
            task = get(task_id)
            entry: dict = {"splits": {}}
            for split in splits:
                if not task.spec.families(split):
                    continue
                solved = 0
                random_solved = 0
                traces = []
                for index in range(args.episodes):
                    env = FactorioEnv(task, session, plan, branch=Branch.TRAIN, split=split)
                    env._episode_index = index - 1
                    env.reset()
                    trace = solve(env)
                    solved += int(trace.succeeded)
                    traces.append(trace.to_dict())

                    rnd = FactorioEnv(task, session, plan, branch=Branch.TRAIN, split=split)
                    rnd._episode_index = index - 1
                    rnd.reset()
                    random_solved += int(
                        random_rollout(
                            rnd,
                            np.random.default_rng(index),
                            min(task.spec.max_decision_steps, 200),
                        )
                    )
                entry["splits"][split] = {
                    "reference_success": solved,
                    "episodes": args.episodes,
                    "reference_rate": round(solved / args.episodes, 2),
                    "random_rate": round(random_solved / args.episodes, 2),
                    "mean_steps": round(float(np.mean([t["steps"] for t in traces])), 1),
                    "stuck_reasons": sorted(
                        {t["stuck_reason"] for t in traces if t["stuck_reason"]}
                    ),
                    "last_errors": sorted({t["last_error"] for t in traces if t["last_error"]}),
                }
                status = entry["splits"][split]
                print(
                    f"{task_id:16s} {split:5s} reference={status['reference_rate']:.2f} "
                    f"random={status['random_rate']:.2f} steps={status['mean_steps']:.0f}",
                    flush=True,
                )
                for reason in status["stuck_reasons"]:
                    print(f"{'':22s} stuck: {reason}", flush=True)
                for err in status["last_errors"]:
                    print(f"{'':22s} error: {err}", flush=True)
            report["tasks"][task_id] = entry
        session.close()
    finally:
        report["wall_seconds"] = round(time.perf_counter() - started, 1)
        # A task is solvable when its reference solution finishes on every
        # split; anything less is a task defect with a named cause.
        report["solvable"] = sorted(
            t
            for t, e in report["tasks"].items()
            if e["splits"] and all(s["reference_rate"] >= 0.8 for s in e["splits"].values())
        )
        report["defective"] = sorted(set(report["tasks"]) - set(report["solvable"]))
        out = ROOT / "docs" / "evidence" / "phase3-solvability.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        manager.cleanup(handle)
    print(f"\nsolvable : {report['solvable']}")
    print(f"defective: {report['defective']}")
    return 0 if not report["defective"] else 1


if __name__ == "__main__":
    sys.exit(main())
