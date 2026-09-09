"""Train a policy by cloning the scripted solver (PLAN.md 4.2 note, option C).

    The scripted solvers we already have make this the cheapest budget in the
    whole program -- a few thousand demonstration episodes, hours not weeks. It
    is also the only mechanism in the book that can make `restore_power`
    learnable.
    -- `docs/research/rl-theory.md`, R6, citing ABJKS Thm. 13.3/13.4.

This is a *separate training mode*, not a variant of the baseline. PLAN 4.2
forbids scripted-solution labels during ordinary PPO training, and that rule is
what makes the Phase 4 baseline mean anything, so a run from here writes
`training_mode: behaviour-cloning-v1` into its manifest and belongs in its own
results column. Comparing a cloned checkpoint against a PPO one without saying
which is which would be the same error as comparing a skills arm against a
primitive floor.

What it does
------------
1. Runs the reference solver for N episodes on the training split, recording the
   *encoded* observation and the catalog index the expert chose at every step.
   Only solved episodes contribute pairs.
2. Fits the policy network to those labels by cross-entropy.
3. Evaluates the result through the ordinary evaluation path -- same three rows,
   same frozen holdout if one is given, same random floor -- so the number is
   comparable to a baseline number even though the training was not.

Run: uv run python tools/clone_expert.py --task repair_belt --episodes 200
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--episodes", type=int, default=200, help="demonstration episodes")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--holdout", default=None)
    parser.add_argument("--skills", action="store_true")
    parser.add_argument("--prefix", default="bc")
    args = parser.parse_args()

    from factoriorl import manifest as manifest_module
    from factoriorl.env import FactorioEnv
    from factoriorl.learn.bc import (
        TRAINING_MODE,
        collect,
        disjointness,
        train_bc,
        write_report,
    )
    from factoriorl.learn.train import TRAIN_SPEED, _make_session, _wrap, evaluate
    from factoriorl.rcon import RCONClient
    from factoriorl.seeding import Branch, SeedPlan, seed_everything
    from factoriorl.skills import SkillEnv
    from factoriorl.tasks import get
    from factoriorl.tasks.reference import SOLVERS, solve
    from factoriorl.worker import WorkerManager

    # `solve` is the entry point that builds a Driver; SOLVERS is consulted only
    # so a task with no reference solution fails immediately rather than after
    # launching a worker.
    if SOLVERS.get(args.task) is None:
        raise SystemExit(f"{args.task} has no reference solution to clone")

    run_id = manifest_module.new_run_id(args.prefix)
    run_dir = manifest_module.runs_dir() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    seeded = seed_everything(args.seed)
    plan = SeedPlan(master=args.seed, run_id=run_id)
    task = get(args.task)

    manager = WorkerManager()
    started = time.perf_counter()
    handle, session = _make_session(manager, f"bc-{args.task}-{run_id[-8:]}")
    try:
        with RCONClient(handle.spec.rcon_endpoint, timeout=30.0) as client:
            client.lua(f"game.speed = {TRAIN_SPEED} return game.speed")

        def build(split: str, branch: Branch):
            env = FactorioEnv(task, session, plan, branch=branch, split=split)
            return SkillEnv(env) if args.skills else env

        print(f"collecting {args.episodes} demonstrations on {args.task}/train ...", flush=True)
        train_env = build("train", Branch.TRAIN)
        data = collect(train_env, solve, args.episodes)
        print(f"  {json.dumps(data.summary())}", flush=True)
        if not len(data):
            raise SystemExit(
                "the solver finished no episodes, so there is nothing to clone; run "
                "tools/solvability.py first"
            )

        model, report = train_bc(
            data,
            train_env.observation_space,
            train_env.action_space,
            epochs=args.epochs,
            seed=args.seed,
        )
        print(
            f"  fitted: loss {report['history'][0]['loss']:.4f} -> "
            f"{report['history'][-1]['loss']:.4f}, expert-action accuracy "
            f"{report['final_expert_action_accuracy']:.3f}",
            flush=True,
        )
        model.save(run_dir / "model")

        # `--holdout` was accepted and then never read: it was declared as an
        # argument and referenced nowhere else, while this module's docstring
        # promised "same frozen holdout if one is given". Every BC result was
        # therefore scored on the run's own scenes over the test split, not on
        # the frozen set, and was reported as though it were the latter -- the
        # same unpaired-comparison defect that got 4.4 withdrawn.
        frozen = None
        if args.holdout:
            frozen = json.loads(Path(args.holdout).read_text(encoding="utf-8"))
            entry = (frozen["holdout"].get("tasks") or {}).get(args.task)
            if entry is None:
                raise SystemExit(f"{args.holdout} does not cover task {args.task}")
            if entry["task_version"] != task.spec.version:
                raise SystemExit(
                    f"{args.holdout} froze {args.task} at v{entry['task_version']}; "
                    f"this tree is v{task.spec.version}. The scenes would differ"
                )

        rows: dict = {}
        for label, split, branch in (
            ("seeds", "train", Branch.EVAL),
            ("val", "val", Branch.EVAL),
            ("structures", "test", Branch.EVAL),
        ):
            if not task.spec.families(split):
                continue
            # The structural row is the published one, so it -- and only it --
            # moves onto the frozen plan and start index. The other two are
            # diagnostics for this run and stay on the run's own stream, which
            # is exactly what `train.py` does.
            row_plan, row_start = plan, -1
            if frozen is not None and split == "test":
                seeds_spec = frozen["holdout"]["seed_plan"]
                row_plan = SeedPlan(master=seeds_spec["master"], run_id=seeds_spec["run_id"])
                row_start = frozen["holdout"]["start_index"] - 1
            env = _wrap(
                FactorioEnv(task, session, row_plan, branch=branch, split=split), args.skills
            )
            env.unwrapped._episode_index = row_start
            rows[label] = evaluate(env, model, args.eval_episodes)
            if frozen is not None and split == "test":
                first = frozen["holdout"]["start_index"]
                rows[label]["holdout"] = {
                    "id": frozen["holdout"].get("holdout_id"),
                    "content_hash": frozen["content_hash"],
                    "start_index": first,
                    "episodes_per_task": frozen["holdout"]["episodes_per_task"],
                    "covers_frozen_set": (
                        args.eval_episodes >= frozen["holdout"]["episodes_per_task"]
                    ),
                }
            # Recorded so the disjointness check has something to check. The
            # env counts from -1 and reset advances it, so the row covers
            # 0..eval_episodes-1 on this branch.
            rows[label]["branch"] = branch.value
            rows[label]["episode_indices"] = list(range(args.eval_episodes))
            print(
                f"  {label:11s} rate={rows[label]['success_rate']:.2f} {rows[label]['wilson_95']}",
                flush=True,
            )

        # R5.3's gate, checked rather than asserted in prose. Demonstrations come
        # from Branch.TRAIN and every evaluation row from Branch.EVAL, so the
        # sets are disjoint by seed stream -- but nothing verified it, and moving
        # a row onto the demonstration branch would have silently scored the
        # policy on scenes it was trained to imitate.
        overlap = disjointness(data, rows)
        if not overlap["disjoint"]:
            raise SystemExit(
                "demonstration and evaluation scenes overlap, so the reported "
                f"numbers would score the policy on what it imitated: {json.dumps(overlap)}"
            )
        print(
            f"  disjoint: {overlap['demonstrated_scenes']} demonstrated scenes on "
            f"{overlap['demonstration_branches']}, no overlap with any evaluation row",
            flush=True,
        )

        report.update(
            {
                "run_id": run_id,
                "task": args.task,
                "task_version": task.spec.version,
                "skills": args.skills,
                "evaluation": rows,
                "wall_seconds": round(time.perf_counter() - started, 1),
                "seeds": {**plan.to_dict(), "seeded": seeded},
                # Stated here as well as in the module, because this file is what
                # a reader of the results will open.
                "not_comparable_to": (
                    "a PPO baseline run. PLAN 4.2 forbids scripted-solution labels "
                    "during ordinary training; this run is built from them."
                ),
                "demonstrations": data.summary(),
                "scene_disjointness": overlap,
            }
        )
        write_report(run_dir, report)
        session.close()
    finally:
        manager.cleanup(handle)

    print(f"\ntraining_mode: {TRAINING_MODE}")
    print(f"wrote {run_dir / 'bc.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
