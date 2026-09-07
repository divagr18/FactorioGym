"""Run every reference solution and report what it proves (PLAN.md 3.2).

    Reference solutions pass all fixed scenarios.
    Randomized generation passes a declared solvability suite.

A solver failure here is a **task defect**, not a training result: the solver
has full evaluator knowledge and still could not finish through the catalog the
policy is given. The `stuck_reason` names which action or precondition is
missing.

Both random floors, and why there are two
-----------------------------------------
The same mistake was made four times in one day, in four disguises:
``mine_smelt``'s random floor went 0.00 to 0.80 when skills were added,
``supply_furnace``'s to 0.88, ``deliver``'s from 0.12 to 0.80, and a baseline
cache keyed on the catalog served a primitive floor to three skill arms and
turned a retracted headline into a published one.

Every one of those fell through the same hole, and this file was the hole: it
declared families solvable while measuring a floor over **primitive actions
only**. A family can be genuinely hard for a random walk over ``move_north``
and nearly free for a random walk over ``approach_entity_k``, and it is the
second number that decides whether an 80% result means anything -- 0.80 against
a floor of 0.80 demonstrates nothing at all. So the floor is measured in both
action spaces at the point where a family is declared solvable, which is the
point at which somebody decides to train on it.

The two ceilings are reported, not enforced. Making a high skill floor
*defective* would empty the solvable set and flip Phase 3 out of Accepted on
the strength of a threshold nobody has agreed to; the exit code stays tied to
reference solvability, and the discriminative-power finding is published beside
it for a human to act on.

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

from factoriorl import skills as skills_module  # noqa: E402
from factoriorl.env import FactorioEnv  # noqa: E402
from factoriorl.rcon import RCONClient  # noqa: E402
from factoriorl.seeding import Branch, SeedPlan  # noqa: E402
from factoriorl.session import WorkerSession  # noqa: E402
from factoriorl.tasks import all_tasks, get  # noqa: E402
from factoriorl.tasks.reference import random_rollout, solve  # noqa: E402
from factoriorl.worker import WorkerManager  # noqa: E402

#: A family whose random floor exceeds this in the action space a policy will
#: actually train on cannot carry an 80% acceptance claim: most of the result is
#: already available by chance. 0.10 is the ceiling `docs/research/ai-and-games.md`
#: argues for, kept identical for both spaces because the ceiling is a property
#: of what a benchmark result must mean, not of which actions produced it.
FLOOR_CEILING = 0.10


def measure_floor(task, session, plan, split: str, episodes: int, skills: bool) -> float:
    """Fraction of episodes a uniform random policy solves in this action space."""
    solved = 0
    for index in range(episodes):
        env = FactorioEnv(task, session, plan, branch=Branch.TRAIN, split=split)
        env._episode_index = index - 1
        wrapped = skills_module.SkillEnv(env) if skills else env
        wrapped.reset()
        solved += int(
            random_rollout(
                wrapped,
                np.random.default_rng(index),
                min(task.spec.max_decision_steps, 200),
            )
        )
    return solved / episodes


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
                traces = []
                for index in range(args.episodes):
                    env = FactorioEnv(task, session, plan, branch=Branch.TRAIN, split=split)
                    env._episode_index = index - 1
                    env.reset()
                    trace = solve(env)
                    solved += int(trace.succeeded)
                    traces.append(trace.to_dict())

                primitive_floor = measure_floor(
                    task, session, plan, split, args.episodes, skills=False
                )
                skill_floor = measure_floor(task, session, plan, split, args.episodes, skills=True)
                entry["splits"][split] = {
                    "reference_success": solved,
                    "episodes": args.episodes,
                    "reference_rate": round(solved / args.episodes, 2),
                    "random_rate": round(primitive_floor, 2),
                    "random_rate_skills": round(skill_floor, 2),
                    "discriminative": bool(
                        primitive_floor <= FLOOR_CEILING and skill_floor <= FLOOR_CEILING
                    ),
                    "mean_steps": round(float(np.mean([t["steps"] for t in traces])), 1),
                    "stuck_reasons": sorted(
                        {t["stuck_reason"] for t in traces if t["stuck_reason"]}
                    ),
                    "last_errors": sorted({t["last_error"] for t in traces if t["last_error"]}),
                }
                status = entry["splits"][split]
                print(
                    f"{task_id:16s} {split:5s} reference={status['reference_rate']:.2f} "
                    f"random={status['random_rate']:.2f} "
                    f"random+skills={status['random_rate_skills']:.2f} "
                    f"steps={status['mean_steps']:.0f}"
                    f"{'' if status['discriminative'] else '  <- floor too high'}",
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
        # Solvable and discriminative are different properties, and conflating
        # them is what let a family with a 0.80 skill-space floor be trained on
        # and published. This list does not affect the exit code; see the module
        # docstring for why the ceiling is reported rather than enforced.
        report["floor_ceiling"] = FLOOR_CEILING
        report["not_discriminative"] = sorted(
            t
            for t, e in report["tasks"].items()
            if e["splits"] and not all(s["discriminative"] for s in e["splits"].values())
        )
        out = ROOT / "docs" / "evidence" / "phase3-solvability.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        manager.cleanup(handle)
    print(f"\nsolvable : {report['solvable']}")
    print(f"defective: {report['defective']}")
    if report["not_discriminative"]:
        print(
            f"\nrandom floor above {FLOOR_CEILING:.2f} in at least one action space: "
            f"{report['not_discriminative']}\n"
            "  An 80% acceptance on these families is worth less than it reads: much of "
            "the rate is available by chance. This does not fail the run -- it is the "
            "number to quote a result against."
        )
    return 0 if not report["defective"] else 1


if __name__ == "__main__":
    sys.exit(main())
