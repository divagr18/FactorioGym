"""The release learning result: families x training seeds (PLAN.md 4.5).

    Run three training seeds and frozen held-out evaluation.
    At least three task families meet the agreed 80% threshold on the test
    (structural) split. At least one is production or repair. Evaluate 100
    held-out episodes per family per training seed. Publish the unfamiliar-seed
    rate on training layouts beside every structural rate.

Everything here exists to make one number quotable: the structural success rate
on a holdout nobody tuned against. Three obligations stand between a training
run and that number, and each is enforced rather than assumed.

**The holdout must be the frozen one.** ``freeze_holdout.py`` documents the trap
in ``RELEASE_RUNNER_CONTRACT``: an evaluation seed plan is built from a *fresh
per-run id*, so a run that does not opt in evaluates a different scene set while
its manifest still says "held-out success rate", and nothing downstream can tell.
This driver passes ``--holdout`` on every run and then checks each result's
``within_frozen_range`` flag, refusing to aggregate a run whose episodes fell
outside the frozen indices. A citation is not coverage.

**The candidate set must predate the first held-out read.** 4.5 forbids
selecting a family on the test split. The frozen file records the declaration
and its commit; this driver reads that declaration and will not run a family
absent from it. Choosing what to run from what scored well is precisely the
move the declaration exists to prevent, and it is easiest to make by accident
at aggregation time.

**A structural rate alone is not interpretable.** Two failures look identical in
a single number -- a policy that never learned, and a policy that learned the
training layouts and does not transfer. So every structural rate is published
beside that seed's unfamiliar-seed rate on training layouts (the ``seeds`` row),
and beside its own random floor over the action space it actually used. The
comparison that matters for 4.5's threshold is structural-versus-floor; the
comparison that diagnoses a miss is structural-versus-seeds.

Run:

    uv run python tools/release_matrix.py --tasks navigate,deliver,mine_smelt \\
        --seeds 1,2,3 --holdout docs/evidence/holdout_v1.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from factoriorl.profiling import COMMIT_HEADROOM, commit_status  # noqa: E402

PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
EVIDENCE = ROOT / "docs" / "evidence"

#: How long to wait for commit headroom before giving up on a cell. An
#: overnight run shares the machine with whatever else is open, and the first
#: attempt at this matrix lost six of nine cells to a memory wall that reported
#: itself as an empty error string: Windows refused to start the subprocess at
#: all, so there was no traceback, no run directory and nothing to diagnose from.
#: Waiting is better than failing, because the thing being waited on is usually
#: the previous cell's own teardown.
HEADROOM_WAIT_SECONDS = 600
HEADROOM_POLL_SECONDS = 20


def await_headroom() -> dict:
    """Block until commit use is under the ceiling, and report what was seen.

    Returns the last commit reading with ``proceeded`` saying whether headroom
    actually arrived. A cell that starts anyway would reproduce the empty-error
    failure this exists to replace, so the caller skips instead.
    """
    deadline = time.time() + HEADROOM_WAIT_SECONDS
    status = commit_status()
    waited = 0.0
    while status["used_fraction"] > COMMIT_HEADROOM and time.time() < deadline:
        print(
            f"    waiting for memory: {status['used_fraction']:.0%} of "
            f"{status['commit_limit_gb']:.1f} GB committed, ceiling {COMMIT_HEADROOM:.0%}",
            flush=True,
        )
        time.sleep(HEADROOM_POLL_SECONDS)
        waited += HEADROOM_POLL_SECONDS
        status = commit_status()
    status["waited_seconds"] = waited
    status["proceeded"] = status["used_fraction"] <= COMMIT_HEADROOM
    return status


THRESHOLD = 0.80

#: A family also has to be *discriminative*: if a uniform random policy over the
#: same action space already clears the bar, clearing it proves nothing. This is
#: the ceiling `docs/research/ai-and-games.md` argues for and `tools/solvability.py`
#: reports against.
#:
#: Declared here before any of this matrix's results were read, prompted by
#: `navigate`, whose random floor over skills measures 0.99 on the train, val
#: *and* test splits -- so the disqualification rests on a property of the task
#: and its action space, visible without touching the holdout, and not on a
#: held-out number. It makes acceptance strictly harder, never easier: PLAN
#: section 4 forbids relaxing a threshold, not tightening one.
FLOOR_CEILING = 0.10

# PLAN 4.5 requires at least one qualifying family to involve production or
# repair, and the task specs carry no category field, so the mapping is written
# here explicitly rather than inferred from a task id at aggregation time.
CATEGORY = {
    "navigate": "movement",
    "deliver": "logistics",
    "mine_smelt": "production",
    "supply_furnace": "production",
    "repair_belt": "repair",
    "restore_power": "repair",
}
QUALIFYING_CATEGORIES = {"production", "repair"}

# Longer horizons need more samples; a single budget across families would
# either waste hours on navigate or starve mine_smelt. Keyed on the family's own
# decision budget rather than a hand-written list, because the list had
# `restore_power` at 50,000 and `repair_belt` -- a *longer* episode, 300
# decisions against 250 -- at 25,000, which is 83 episodes minimum against
# deliver's 208. Whatever the release result turns out to be, it should not be
# an artefact of which names someone remembered to type.
LONG_HORIZON_DECISIONS = 250
LONG_HORIZON_STEPS = 50_000
FALLBACK_STEPS = 25_000


def steps_for(task_id: str) -> int:
    from factoriorl.tasks import get

    budget = get(task_id).spec.max_decision_steps
    return LONG_HORIZON_STEPS if budget >= LONG_HORIZON_DECISIONS else FALLBACK_STEPS


def _shown(path: Path) -> str:
    """Display path, without letting a relative --out abort a finished run.

    `relative_to` raises when the path is not under ROOT, and a relative --out
    is not -- so a completed matrix died on its own success message and exited
    1, which is also the code the tool uses for "not accepted".
    """
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def run_cell(task: str, seed: int, steps: int, episodes: int, holdout: Path, skills: bool) -> dict:
    command = [
        str(PYTHON),
        "-m",
        "factoriorl.cli",
        "train",
        "--task",
        task,
        "--steps",
        str(steps),
        "--seed",
        str(seed),
        "--eval-episodes",
        str(episodes),
        "--holdout",
        str(holdout),
        "--prefix",
        "release",
    ]
    if skills:
        command.append("--skills")
    started = time.perf_counter()
    process = subprocess.run(command, cwd=str(ROOT), capture_output=True, text=True, check=False)
    if process.returncode != 0:
        # The tail, not the head. A traceback ends with the exception type and
        # message and begins with frames from the interpreter's entry point, so
        # a head-truncated capture shows the least useful part -- which is what
        # turned a one-word lookup bug into an unreadable "FAILED:" line.
        return {
            "ok": False,
            "task": task,
            "seed": seed,
            "steps": steps,
            "error": process.stderr.strip()[-2000:],
        }
    result = json.loads(process.stdout[process.stdout.index("{") :])
    result["ok"] = True
    result["seed"] = seed
    result["wall_seconds"] = round(time.perf_counter() - started, 1)
    return result


def coverage_of(row: dict) -> dict:
    """Did this run actually touch the frozen episodes it cites?"""
    structures = (row.get("evaluation") or {}).get("structures") or {}
    return structures.get("holdout") or {}


def summarise(row: dict) -> dict:
    evaluation = row.get("evaluation") or {}
    structures = evaluation.get("structures") or {}
    seeds_row = evaluation.get("seeds") or {}
    holdout = coverage_of(row)
    return {
        "seed": row.get("seed"),
        "run_id": row.get("run_id"),
        "steps": row.get("total_steps"),
        "wall_seconds": row.get("wall_seconds"),
        "train_success_rate": row.get("final_train_success_rate"),
        "structural_success_rate": structures.get("success_rate"),
        "structural_wilson_95": structures.get("wilson_95"),
        "structural_random_floor": (structures.get("random_baseline") or {}).get("success_rate"),
        "unfamiliar_seed_success_rate": seeds_row.get("success_rate"),
        "unfamiliar_seed_wilson_95": seeds_row.get("wilson_95"),
        "holdout_within_frozen_range": holdout.get("within_frozen_range"),
        "holdout_episodes_touched": holdout.get("episodes_touched"),
        "holdout_content_hash": holdout.get("content_hash"),
    }


def diagnose(cells: list[dict]) -> str:
    """Say which of the two indistinguishable failures a miss actually is."""
    structural = [
        c["structural_success_rate"] for c in cells if c["structural_success_rate"] is not None
    ]
    familiar = [
        c["unfamiliar_seed_success_rate"]
        for c in cells
        if c["unfamiliar_seed_success_rate"] is not None
    ]
    if not structural or not familiar:
        return "no complete cell"
    mean_structural = sum(structural) / len(structural)
    mean_familiar = sum(familiar) / len(familiar)
    if mean_structural >= THRESHOLD:
        return "meets the threshold on the structural split"
    if mean_familiar < THRESHOLD:
        return (
            "did not learn: the policy misses the threshold on unfamiliar seeds over "
            "training layouts too, so the structural miss is not a transfer result"
        )
    return (
        "learned but did not transfer: the policy clears the threshold on training "
        "layouts under unfamiliar seeds and misses it on held-out structures"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="navigate,deliver,mine_smelt")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--holdout", default="docs/evidence/holdout_v1.json")
    parser.add_argument(
        "--no-skills",
        action="store_true",
        help="train over primitives only; most families cannot learn there",
    )
    parser.add_argument("--steps", type=int, default=None, help="override the per-family budget")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    holdout_path = (ROOT / args.holdout).resolve()
    frozen = json.loads(holdout_path.read_text(encoding="utf-8"))
    declared = (frozen.get("declaration") or {}).get("candidate_families")
    if not declared:
        print(
            "the frozen holdout declares no candidate families. PLAN 4.5 requires the "
            "candidate set to be declared before any held-out result is read:\n"
            "  uv run python tools/freeze_holdout.py --declare --candidates <ids>",
            file=sys.stderr,
        )
        return 2

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    undeclared = sorted(set(tasks) - set(declared))
    if undeclared:
        print(
            f"refusing to evaluate {undeclared}: not in the declared candidate set "
            f"{declared}. Selecting a family after seeing a held-out result is the "
            "move the declaration exists to prevent.",
            file=sys.stderr,
        )
        return 2

    seeds = [int(s) for s in args.seeds.split(",")]
    print(
        f"release matrix: {tasks} x seeds {seeds}, {args.eval_episodes} held-out episodes\n"
        f"holdout {frozen['holdout'].get('holdout_id')} "
        f"hash {frozen['content_hash'][:16]} "
        f"declared {declared} at {(frozen.get('declaration') or {}).get('declared_at')}\n",
        flush=True,
    )

    matrix: dict[str, list[dict]] = {}
    raw: list[dict] = []
    for task in tasks:
        steps = args.steps or steps_for(task)
        cells: list[dict] = []
        for seed in seeds:
            headroom = await_headroom()
            if not headroom["proceeded"]:
                row = {
                    "ok": False,
                    "task": task,
                    "seed": seed,
                    "steps": steps,
                    "skipped": "memory_ceiling",
                    "commit_before": headroom,
                    "error": (
                        f"skipped: commit use {headroom['used_fraction']:.0%} still above "
                        f"the {COMMIT_HEADROOM:.0%} ceiling after "
                        f"{HEADROOM_WAIT_SECONDS}s"
                    ),
                }
                raw.append(row)
                print(f"{task:14s} seed={seed} SKIPPED: {row['error']}", flush=True)
                continue
            row = run_cell(
                task, seed, steps, args.eval_episodes, holdout_path, skills=not args.no_skills
            )
            row["commit_before"] = headroom
            raw.append(row)
            if not row["ok"]:
                print(f"{task:14s} seed={seed} FAILED: {row['error'][-400:]}", flush=True)
                continue
            cell = summarise(row)
            cells.append(cell)
            covered = cell["holdout_within_frozen_range"]
            print(
                f"{task:14s} seed={seed} "
                f"structural={cell['structural_success_rate']:.2f} "
                f"{cell['structural_wilson_95']} "
                f"floor={cell['structural_random_floor']} "
                f"seeds-row={cell['unfamiliar_seed_success_rate']:.2f} "
                f"frozen={'yes' if covered else 'NO'} "
                f"({cell['wall_seconds']:.0f}s)",
                flush=True,
            )
        matrix[task] = cells

    families: dict[str, dict] = {}
    for task, cells in matrix.items():
        # A cell that missed the frozen range has not evaluated this holdout,
        # whatever its manifest says, so it must not enter the mean.
        covered = [c for c in cells if c["holdout_within_frozen_range"]]
        excluded = len(cells) - len(covered)
        rates = [
            c["structural_success_rate"]
            for c in covered
            if c["structural_success_rate"] is not None
        ]
        familiar = [
            c["unfamiliar_seed_success_rate"]
            for c in covered
            if c["unfamiliar_seed_success_rate"] is not None
        ]
        mean_structural = round(sum(rates) / len(rates), 4) if rates else None
        floors = [
            c["structural_random_floor"]
            for c in covered
            if c["structural_random_floor"] is not None
        ]
        mean_floor = round(sum(floors) / len(floors), 4) if floors else None
        clears = bool(mean_structural is not None and mean_structural >= THRESHOLD)
        discriminative = bool(mean_floor is not None and mean_floor <= FLOOR_CEILING)
        families[task] = {
            "category": CATEGORY.get(task, "unknown"),
            "seeds_completed": len(cells),
            "seeds_outside_frozen_range": excluded,
            "structural_success_rate_mean": mean_structural,
            "structural_success_rate_per_seed": rates,
            "random_floor_mean": mean_floor,
            "unfamiliar_seed_rate_mean": round(sum(familiar) / len(familiar), 4)
            if familiar
            else None,
            "clears_threshold": clears,
            "discriminative": discriminative,
            # Both, or the number is not evidence of anything.
            "meets_threshold": clears and discriminative,
            "diagnosis": diagnose(covered)
            if discriminative
            else (
                f"not discriminative: a random policy over the same action space scores "
                f"{mean_floor}, so clearing {THRESHOLD:.2f} here demonstrates nothing"
            ),
            "cells": cells,
        }

    passing = [t for t, f in families.items() if f["meets_threshold"]]
    qualifying = [t for t in passing if families[t]["category"] in QUALIFYING_CATEGORIES]
    accepted = len(passing) >= 3 and bool(qualifying)

    report = {
        "plan_section": "4.5",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "threshold": THRESHOLD,
        "holdout": {
            "id": frozen["holdout"].get("holdout_id"),
            "content_hash": frozen["content_hash"],
            "episodes_per_task": frozen["holdout"]["episodes_per_task"],
            "declared_candidates": declared,
            "declared_at": (frozen.get("declaration") or {}).get("declared_at"),
            "declared_at_commit": (frozen.get("declaration") or {}).get("declared_at_commit"),
        },
        "seeds": seeds,
        "eval_episodes": args.eval_episodes,
        "action_space": "primitive" if args.no_skills else "primitive+skills",
        "families": families,
        "acceptance": {
            "families_meeting_threshold": passing,
            "qualifying_production_or_repair": qualifying,
            "requires": (
                "at least three families at 0.80 whose random floor is at or below "
                f"{FLOOR_CEILING}, at least one of them production or repair"
            ),
            "floor_ceiling": FLOOR_CEILING,
            "accepted": accepted,
        },
        "runs": raw,
    }
    destination = Path(args.out) if args.out else EVIDENCE / "phase4-release-matrix.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n--- PLAN 4.5 acceptance")
    for task, family in families.items():
        mean_rate = family["structural_success_rate_mean"]
        shown = "n/a" if mean_rate is None else f"{mean_rate:.2f}"
        print(
            f"  {task:14s} {family['category']:10s} structural={shown} "
            f"floor={family['random_floor_mean']} "
            f"{'PASS' if family['meets_threshold'] else 'miss'} -- {family['diagnosis']}"
        )
    print(
        f"\n  {len(passing)}/3 families at {THRESHOLD:.2f}; "
        f"production-or-repair among them: {qualifying or 'none'}\n"
        f"  ACCEPTED: {accepted}\n"
        f"wrote {_shown(destination)}",
        flush=True,
    )
    return 0 if accepted else 1


if __name__ == "__main__":
    sys.exit(main())
