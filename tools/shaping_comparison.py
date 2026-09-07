"""Shaping dependence (PLAN.md 4.4).

    Compare sparse and shaped training on at least one tractable task.
    Both use the same success predicate.
    The report distinguishes faster learning from changed task difficulty.

The measurement that separates those two hypotheses is the one most easily
omitted: **both policies are evaluated under the sparse predicate only, on the
same held-out episodes**. If shaping merely sped learning up, the two agree at
convergence; if shaping changed what "success" means, they will not -- and the
success predicate is identical by construction here, because disabling shaping
zeroes the shaping weights and touches nothing else.

Run: uv run python tools/shaping_comparison.py --task supply_furnace --steps 40000
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


def run(task: str, steps: int, seed: int, shaped: bool) -> dict:
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
        "25",
        "--prefix",
        "shaped" if shaped else "sparse",
    ]
    if not shaped:
        command.append("--no-shaping")
    started = time.perf_counter()
    process = subprocess.run(command, cwd=str(ROOT), capture_output=True, text=True, check=False)
    if process.returncode != 0:
        return {"ok": False, "shaped": shaped, "seed": seed, "error": process.stderr.strip()[-500:]}
    result = json.loads(process.stdout[process.stdout.index("{") :])
    result["ok"] = True
    result["shaped"] = shaped
    result["seed"] = seed
    result["wall_seconds"] = round(time.perf_counter() - started, 1)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="navigate")
    parser.add_argument("--steps", type=int, default=40_000)
    parser.add_argument("--seeds", default="1,2")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    runs = []
    for seed in seeds:
        for shaped in (True, False):
            result = run(args.task, args.steps, seed, shaped)
            runs.append(result)
            label = "shaped" if shaped else "sparse"
            if result["ok"]:
                held = result["held_out"]
                print(
                    f"{args.task} seed={seed} {label:6s} "
                    f"train={result['final_train_success_rate']:.2f} "
                    f"held-out={held['success_rate']:.2f} {held['wilson_95']} "
                    f"({result['wall_seconds']:.0f}s)",
                    flush=True,
                )
            else:
                print(
                    f"{args.task} seed={seed} {label} FAILED: {result['error'][:200]}", flush=True
                )

    shaped_runs = [r for r in runs if r.get("ok") and r["shaped"]]
    sparse_runs = [r for r in runs if r.get("ok") and not r["shaped"]]

    def mean(rows, key):
        values = [r["held_out"][key] for r in rows]
        return round(sum(values) / len(values), 4) if values else None

    shaped_rate = mean(shaped_runs, "success_rate")
    sparse_rate = mean(sparse_runs, "success_rate")

    interpretation = "inconclusive"
    if shaped_rate is not None and sparse_rate is not None:
        if abs(shaped_rate - sparse_rate) < 0.1:
            interpretation = (
                "shaping did not change the outcome at this budget: both reach a "
                "comparable held-out rate under the same sparse predicate"
            )
        elif shaped_rate > sparse_rate:
            interpretation = (
                "shaping helped within this budget. Whether that is faster learning "
                "or an easier problem cannot be separated from the endpoint alone; "
                "the curves are in each run's curve.csv, and a longer sparse run is "
                "the test that distinguishes them"
            )
        else:
            interpretation = (
                "shaping hurt, which usually means the shaping signal points away "
                "from the success predicate -- a reward defect, not a tuning knob"
            )

    report = {
        "task": args.task,
        "steps": args.steps,
        "seeds": seeds,
        "shaped_held_out_mean": shaped_rate,
        "sparse_held_out_mean": sparse_rate,
        # Both evaluated under the same success predicate: disabling shaping
        # zeroes the shaping weights and changes nothing else.
        "same_success_predicate": True,
        "interpretation": interpretation,
        "runs": runs,
    }
    out = ROOT / "docs" / "evidence" / "phase4-shaping-comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nshaped={shaped_rate} sparse={sparse_rate}\n{interpretation}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
