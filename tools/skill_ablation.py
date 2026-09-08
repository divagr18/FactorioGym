"""Flat versus skill-augmented action spaces (PLAN.md 4b.3).

    Ablate the skill layer on at least one family: identical training budget,
    identical evaluation episodes, one variable -- whether the temporally
    extended actions exist.

This ran ad hoc the first time and produced a headline that had to be retracted.
The retraction is the reason this file exists, so it is worth stating what went
wrong rather than only fixing it: the random-baseline cache keyed on the
resolved *catalog* digest, but skills are added by a wrapper around the
environment rather than by the catalog, so both arms of the ablation resolved to
the same cache key. The flat arm ran first, cached a floor measured over
primitive actions, and the skill arms then reported that floor as their own --
``deliver`` was published against 0.12 when its true skill-space floor was 0.80.

The cache key carries the action space now (``baselines.CACHE_VERSION`` 2), but
a key is a mechanism and this is a measurement, so the mechanism is checked
*here*, from the outside: if the two arms come back quoting an identical random
floor, that is reported as a suspected cache collision rather than accepted as a
result. A floor is a property of the action space; two different action spaces
agreeing to four decimal places on the same task is a coincidence worth
refusing.

What the comparison can and cannot say
--------------------------------------
Both arms are scored on the same evaluation episodes under the same success
predicate, so the *gap between the arms* is a paired measurement. What that gap
means depends on the floors: skills raise the random floor as well as the policy
(a random walk over ``approach_entity_k`` solves ``deliver`` far more often than
a random walk over ``move_north``), so a skills arm at 1.00 against a floor of
0.80 has demonstrated much less than the number alone suggests. The report
therefore leads with each arm's margin over its own floor, not with the rate.

Run: uv run python tools/skill_ablation.py --task deliver --steps 25000 --seeds 1,2
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
EVIDENCE = ROOT / "docs" / "evidence"


def run_arm(
    task: str, steps: int, seed: int, skills: bool, episodes: int, holdout: str | None
) -> dict:
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
        "--prefix",
        "sk" if skills else "flat",
    ]
    if skills:
        command.append("--skills")
    if holdout:
        command += ["--holdout", holdout]
    started = time.perf_counter()
    process = subprocess.run(command, cwd=str(ROOT), capture_output=True, text=True, check=False)
    if process.returncode != 0:
        return {
            "ok": False,
            "skills": skills,
            "seed": seed,
            "error": process.stderr.strip()[-600:],
        }
    result = json.loads(process.stdout[process.stdout.index("{") :])
    result["ok"] = True
    result["skills"] = skills
    result["seed"] = seed
    result["wall_seconds"] = round(time.perf_counter() - started, 1)
    return result


def floor_of(row: dict) -> float | None:
    baseline = (row.get("held_out") or {}).get("random_baseline") or {}
    value = baseline.get("success_rate")
    return None if value is None else float(value)


def rate_of(row: dict) -> float | None:
    value = (row.get("held_out") or {}).get("success_rate")
    return None if value is None else float(value)


def mean(rows: list[dict], getter) -> float | None:
    values = [v for v in (getter(r) for r in rows) if v is not None]
    return round(sum(values) / len(values), 4) if values else None


def interpret(
    flat_rate: float | None,
    skill_rate: float | None,
    margins: dict[str, float | None],
    collision: bool,
) -> str:
    if collision:
        return (
            "REFUSED: both arms report an identical random floor, which two different "
            "action spaces should not produce. Treat this as a baseline-cache collision "
            "(the failure that invalidated the first 4b result) until the floors are "
            "recomputed, and do not read the margins."
        )
    if flat_rate is None or skill_rate is None:
        return "inconclusive: an arm did not complete"
    if margins["skills"] is None or margins["flat"] is None:
        return "inconclusive: a random floor is missing, so no margin can be formed"
    if margins["skills"] - margins["flat"] > 0.1:
        return (
            "skills helped beyond what they gave away: the skill arm's margin over its "
            "own (higher) random floor exceeds the flat arm's margin over its floor, so "
            "the gain is not merely an easier action space"
        )
    if skill_rate - flat_rate > 0.1:
        return (
            "the skill arm scores higher but its floor rose by a comparable amount: at "
            "this budget the skill layer made the task easier to hit by chance about as "
            "much as it made it easier to learn"
        )
    return "no material difference between the arms at this budget"


def pairing(runs: list[dict]) -> dict:
    """Did every arm score the *same* episodes?

    An episode seed is ``blake2b(master | run_id | branch | index)`` and each run
    builds its plan with a fresh per-run id, so two arms both evaluating
    "episodes 0..24" evaluate two different scene draws. Every arm here reported
    touching indices 0,1,2... and none of them saw the same worlds.

    That is fatal to a comparison and invisible in its output: the arms differ by
    the variable under test *and* by the content they were scored on, and the
    smaller the real effect, the more of the reported difference is scene luck.
    Passing a frozen holdout makes both arms score one fixed set; this checks
    that it actually happened rather than trusting the flag.
    """
    sets = []
    for run in runs:
        if not run.get("ok"):
            continue
        scored = ((run.get("evaluation") or {}).get("structures") or {}).get("scored_episodes")
        sets.append(tuple(scored) if scored else None)
    if not sets:
        return {"paired": False, "reason": "no arm completed"}
    if any(s is None for s in sets):
        return {
            "paired": False,
            "reason": (
                "at least one arm recorded no scored episode set, which means it did not "
                "evaluate a frozen holdout; its scenes are a function of its own run id"
            ),
        }
    if len(set(sets)) != 1:
        return {
            "paired": False,
            "reason": (
                f"arms scored {len(set(sets))} different episode sets; the comparison "
                "confounds the variable under test with the scenes each arm was given"
            ),
        }
    return {"paired": True, "episodes": len(sets[0])}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="deliver")
    parser.add_argument("--steps", type=int, default=25_000)
    parser.add_argument("--seeds", default="1,2")
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument(
        "--holdout",
        default="docs/evidence/holdout_v3.json",
        help="frozen holdout both arms are scored on; '' to disable (unpaired)",
    )
    parser.add_argument("--out", default=None, help="report destination (default docs/evidence)")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    runs: list[dict] = []
    for seed in seeds:
        for skills in (False, True):
            row = run_arm(args.task, args.steps, seed, skills, args.eval_episodes, args.holdout)
            runs.append(row)
            label = "skills" if skills else "flat  "
            if row["ok"]:
                held = row["held_out"]
                floor = floor_of(row)
                shown = "n/a" if floor is None else f"{floor:.2f}"
                print(
                    f"{args.task} seed={seed} {label} "
                    f"train={row['final_train_success_rate']:.2f} "
                    f"held-out={held['success_rate']:.2f} {held.get('wilson_95')} "
                    f"floor={shown} ({row['wall_seconds']:.0f}s)",
                    flush=True,
                )
            else:
                print(f"{args.task} seed={seed} {label} FAILED: {row['error'][:300]}", flush=True)

    flat = [r for r in runs if r.get("ok") and not r["skills"]]
    skilled = [r for r in runs if r.get("ok") and r["skills"]]
    flat_rate, skill_rate = mean(flat, rate_of), mean(skilled, rate_of)
    flat_floor, skill_floor = mean(flat, floor_of), mean(skilled, floor_of)

    collision = (
        flat_floor is not None
        and skill_floor is not None
        and flat_floor > 0.0
        and abs(flat_floor - skill_floor) < 1e-9
    )
    margins = {
        "flat": None
        if flat_rate is None or flat_floor is None
        else round(flat_rate - flat_floor, 4),
        "skills": None
        if skill_rate is None or skill_floor is None
        else round(skill_rate - skill_floor, 4),
    }
    paired = pairing(runs)
    if not paired["paired"]:
        interpretation = (
            f"REFUSED: the arms are not paired ({paired['reason']}). Both arms must be "
            "scored on one fixed episode set or the gap between them includes the gap "
            "between their scene draws."
        )
    else:
        interpretation = interpret(flat_rate, skill_rate, margins, collision)

    report = {
        "plan_section": "4b.3",
        "task": args.task,
        "steps": args.steps,
        "seeds": seeds,
        "eval_episodes": args.eval_episodes,
        "holdout": args.holdout,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "arms": {
            "flat": {"held_out_mean": flat_rate, "random_floor_mean": flat_floor},
            "skills": {"held_out_mean": skill_rate, "random_floor_mean": skill_floor},
        },
        "margin_over_own_floor": margins,
        "suspected_baseline_cache_collision": collision,
        "pairing": paired,
        "interpretation": interpretation,
        "runs": runs,
    }
    destination = Path(args.out) if args.out else EVIDENCE / f"phase4b-ablation-{args.task}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n{interpretation}\nwrote {destination.relative_to(ROOT)}", flush=True)
    return 1 if collision else 0


if __name__ == "__main__":
    sys.exit(main())
