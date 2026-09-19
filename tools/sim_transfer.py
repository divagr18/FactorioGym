"""Evaluate a factory-sim policy on the real engine (M5 transfer).

The policy was trained only on the simulator. Here it plays the same task on
Factorio, through the same `ParameterizedEnv` any policy uses, on the training
split's families and on the held-out (test) family, and the success rates are
written beside the simulator's own numbers for the same checkpoint.

    uv run python tools/sim_transfer.py --policy ../factory-sim/runs/X/policy.ts \\
        --sim-report ../factory-sim/runs/X/final.json --episodes 32

Scenes are drawn from the EVAL seed branch of a fixed plan, so they are
disjoint from anything FactorioRL trains on, and every episode's blueprint
digest is recorded so a result names its scenes.
"""

from __future__ import annotations

import argparse
import json
import lzma
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from factoriorl.engine_config import resolve_game_speed  # noqa: E402
from factoriorl.env import FactorioEnv  # noqa: E402
from factoriorl.learn.sim_policy import SimPolicy  # noqa: E402
from factoriorl.parameterized import ParameterizedEnv  # noqa: E402
from factoriorl.rcon import RCONClient  # noqa: E402
from factoriorl.seeding import Branch, SeedPlan  # noqa: E402
from factoriorl.session import WorkerSession  # noqa: E402
from factoriorl.tasks import get  # noqa: E402
from factoriorl.worker import WorkerManager  # noqa: E402

OUT = ROOT / "docs" / "evidence" / "sim-transfer-m5.json"
#: Every episode's scene payload and action vectors, beside the report, so the
#: simulator can replay exactly what the engine was asked to do.
MASTER_SEED = 20260917


def wilson(successes: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 1.0]
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def state_fingerprint(env) -> dict:
    """The little of the engine's state a simulator must agree about.

    Position to the hundredth of a tile, what the character carries, and every
    machine with what is in it -- enough that a divergence in walking, in
    mining, or in how fast a furnace burns shows up here and nowhere else.
    """
    observation = env._observation
    character = observation.get("character") or {}
    position = character.get("position") or [0.0, 0.0]
    machines = []
    for record in observation.get("entities") or []:
        if record.get("type") == "item-entity":
            continue
        machines.append(
            {
                "p": [round(float(v), 2) for v in (record.get("p") or [0, 0])],
                "h": record.get("h"),
                "t": record.get("type"),
                "n": record.get("n") or record.get("name"),
                "c": {k: v for k, v in (record.get("contents") or {}).items() if v},
                "o": {k: v for k, v in (record.get("output") or {}).items() if v},
            }
        )
    return {
        "xy": [round(float(v), 2) for v in position],
        "inv": {k: v for k, v in (observation.get("inventory") or {}).items() if v},
        "m": sorted(machines, key=lambda r: (r["p"][0], r["p"][1])),
    }


def run_split(task, session, policy, split, episodes, greedy, replays, profile="v1") -> dict:
    plan = SeedPlan(master=MASTER_SEED, run_id="sim-transfer-m5")
    env = FactorioEnv(task, session, plan, branch=Branch.EVAL, split=split)
    captured: dict = {}
    install = env._install

    def capturing_install(blueprint):
        captured["payload"] = blueprint.to_dict(
            public_markers=env.spec_.public_markers,
            extra_tracked_items=env.spec_.extra_tracked_items,
        )
        return install(blueprint)

    env._install = capturing_install
    penv = ParameterizedEnv(env, profile=profile)
    rows = []
    for index in range(episodes):
        observation, reset_info = penv.reset(options={"scene_index": index})
        steps, total, decode_failures = 0, 0.0, 0
        vectors, rejected, fingerprints = [], [], []
        started = time.perf_counter()
        while True:
            # A fingerprint of the engine's world before each decision, so a
            # replay can find the first state the simulator gets wrong rather
            # than the first action it refuses. The two are not the same: a
            # refusal is visible, a slow furnace is not.
            fingerprints.append(state_fingerprint(env))
            action, _ = policy.predict(observation, penv.action_masks(), greedy)
            vectors.append([int(v) for v in action])
            observation, reward, terminated, truncated, info = penv.step(action)
            steps += 1
            total += reward
            # Which decisions this engine refused, not just how many. Replaying
            # the same vectors in the simulator reproduced four of eight
            # episodes exactly and diverged on the rest, and a count cannot say
            # where two backends first disagree.
            if info.get("decode_failure"):
                rejected.append(steps - 1)
            decode_failures += int(bool(info.get("decode_failure")))
            if terminated or truncated:
                break
        verification = info.get("verification") or {}
        rows.append(
            {
                "episode": index,
                "family": info.get("layout_family"),
                "blueprint": reset_info.get("blueprint"),
                "success": bool(info.get("success")),
                "excluded": bool(info.get("excluded_from_metrics")),
                "verified_output": verification.get("machine_output"),
                "return": round(total, 6),
                "steps": steps,
                "decode_failures": decode_failures,
                "seconds": round(time.perf_counter() - started, 1),
            }
        )
        print(json.dumps({"split": split, **rows[-1]}), flush=True)
        replays.append(
            {
                "split": split,
                "episode": index,
                "task": task.spec.id,
                "action_space": f"parameterized-{profile}",
                "decision_ticks": task.spec.decision_ticks,
                "max_decision_steps": task.spec.max_decision_steps,
                "construction_tick_limit": env.construction_tick_limit,
                "blueprint": captured["payload"],
                "vectors": vectors,
                "rejected": rejected,
                "fingerprints": fingerprints,
                "outcome": {
                    "success": rows[-1]["success"],
                    "verified_output": rows[-1]["verified_output"],
                    "return": rows[-1]["return"],
                    "excluded": rows[-1]["excluded"],
                },
            }
        )
    counted = [r for r in rows if not r["excluded"]]
    wins = sum(r["success"] for r in counted)
    return {
        "split": split,
        "greedy": greedy,
        "episodes": len(counted),
        "excluded": len(rows) - len(counted),
        "success": round(wins / max(1, len(counted)), 4),
        "success_ci95": wilson(wins, len(counted)),
        "by_family": {
            f: round(
                sum(r["success"] for r in counted if r["family"] == f)
                / max(1, sum(r["family"] == f for r in counted)),
                4,
            )
            for f in sorted({r["family"] for r in counted})
        },
        "episodes_detail": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--sim-report", type=Path, help="factory-sim final.json for the policy")
    parser.add_argument("--task", default="construct_smelting_line")
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--splits", default="train,test")
    parser.add_argument("--sampled", action="store_true", help="sample instead of greedy")
    parser.add_argument(
        "--profile",
        choices=("v1", "v2"),
        default="v1",
        help="the action-space profile the policy was trained against; a v2 "
        "policy read through v1 names a different entity and a different tile",
    )
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    task = get(args.task)
    policy = SimPolicy(args.policy)
    greedy = not args.sampled
    manager = WorkerManager()
    handle = manager.launch("sim-transfer")
    report: dict = {
        "about": "a factory-sim-trained policy evaluated on the real engine",
        "task": task.spec.id,
        "task_version": task.spec.version,
        "policy": policy.describe(),
        # Which meaning the target and placement indices carried. A result that
        # did not name it cannot be replayed, because the vectors are the same
        # shape under either profile.
        "action_space": f"parameterized-{args.profile}",
        "seed_plan": {"master": MASTER_SEED, "run_id": "sim-transfer-m5", "branch": "eval"},
        "engine": {k: v for k, v in manager.engine.to_dict().items() if k != "executable"},
        "real_engine": {},
    }
    if args.sim_report and args.sim_report.exists():
        report["simulator"] = json.loads(args.sim_report.read_text(encoding="utf-8"))
    try:
        with RCONClient(handle.spec.rcon_endpoint, timeout=30.0) as client:
            client.lua(f"game.speed = {resolve_game_speed()} return game.speed")
        session = WorkerSession(handle, timeout=120.0)
        session.status()
        replays: list[dict] = []
        for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
            if not task.spec.families(split):
                continue
            key = f"{split}_{'greedy' if greedy else 'sampled'}"
            report["real_engine"][key] = run_split(
                task, session, policy, split, args.episodes, greedy, replays, args.profile
            )
        session.close()
    finally:
        manager.cleanup(handle)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    episodes_path = args.out.with_name(args.out.stem + ".episodes.jsonl.xz")
    report["episodes_file"] = episodes_path.name
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", "utf-8")
    lines = "".join(json.dumps(r, sort_keys=True) + "\n" for r in replays)
    episodes_path.write_bytes(lzma.compress(lines.encode(), preset=9))
    summary = {k: v["success"] for k, v in report["real_engine"].items()}
    print(json.dumps({"real_engine": summary}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
