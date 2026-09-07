"""Short training runs across every family, to find which actually learn.

PLAN 4.5 needs three families over the 80% bar with at least one production or
repair task. Committing a twelve-hour matrix before knowing which families are
learnable at their current difficulty would be guessing; this is the cheap
question that makes that an informed decision.

Runs are short on purpose. The output is not "did it reach 80%" but "is there
signal above the random baseline" -- the families that show signal earn the long
runs, and the ones that do not are a task-difficulty finding, not a training
failure.

Run: uv run python tools/feasibility_sweep.py [--steps N] [--concurrency K]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"

FAMILIES = ("deliver", "supply_furnace", "mine_smelt", "repair_belt", "restore_power")


def run_one(task: str, steps: int, seed: int) -> dict:
    started = time.perf_counter()
    process = subprocess.run(
        [
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
            "20",
            "--prefix",
            "sweep",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = round(time.perf_counter() - started, 1)
    if process.returncode != 0:
        return {
            "task": task,
            "ok": False,
            "wall_seconds": elapsed,
            "error": process.stderr.strip()[-600:],
        }
    payload = process.stdout[process.stdout.index("{") :]
    result = json.loads(payload)
    result["ok"] = True
    result["wall_seconds"] = elapsed
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=25_000)
    parser.add_argument("--seed", type=int, default=11)
    # Each run holds a Factorio worker plus a torch process, so concurrency is
    # bounded by commit charge, not by core count.
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--families", default=",".join(FAMILIES))
    args = parser.parse_args()

    families = tuple(f.strip() for f in args.families.split(",") if f.strip())
    started = time.perf_counter()
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(run_one, family, args.steps, args.seed) for family in families]
        for future in futures:
            result = future.result()
            results.append(result)
            if result["ok"]:
                held = result["held_out"]
                base = result["random_baseline"]
                print(
                    f"{result['task']:16s} train={result['final_train_success_rate']:.2f} "
                    f"held-out={held['success_rate']:.2f} {held['wilson_95']} "
                    f"random={base['success_rate']:.2f} "
                    f"({result['steps_per_second']:.1f} steps/s, {result['wall_seconds']:.0f}s)",
                    flush=True,
                )
            else:
                print(f"{result['task']:16s} FAILED: {result['error'][:200]}", flush=True)

    report = {
        "steps_per_run": args.steps,
        "seed": args.seed,
        "concurrency": args.concurrency,
        "wall_seconds": round(time.perf_counter() - started, 1),
        "runs": results,
    }
    # A family "shows signal" when its held-out lower bound clears the random
    # upper bound -- not merely when its point estimate is higher.
    for result in results:
        if not result.get("ok"):
            result["signal"] = False
            continue
        held = result["held_out"]["wilson_95"]
        base = result["random_baseline"]["wilson_95"]
        result["signal"] = held[0] > base[1]
    report["candidates"] = [r["task"] for r in results if r.get("signal")]

    out = ROOT / "docs" / "evidence" / "phase4-feasibility-sweep.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\ncandidates with signal above random: {report['candidates']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
