"""Does the verifier run by itself, and does it refuse a hand-fed furnace? (M1)

Two changes to `construct_smelting_line` are only claims until a real engine
agrees with them:

1. **Verification at the end of the budget.** `run_verification` used to run
   only when a driver called it -- the agentic bridge's `finish`, the
   evaluator-only `solve` -- so an RL episode that ran out of decisions was
   never measured and scored zero whatever it built. `FactorioEnv.step_payload`
   now runs it on the truncating step and ends the episode as terminated.
2. **Provenance.** `machine_produced` is `produced - handcrafted - mined`, which
   subtracts hand-mined *ore* but not the plates smelted from it. Version 1.1.0
   caps the counted plates by the iron ore machines mined inside the window.

The unit tests prove the logic against stubbed truth. This proves it against
Factorio, on one scene, with two arms that end the same way -- by running the
decision budget out, never by calling the verifier:

- **reference**: the evaluator's own drill-and-furnace construction
  (`tasks.reference._build_at`). Must terminate verified, and clear the target.
- **exploit**: a furnace and no drill, fuelled and fed hand-mined ore through
  the same catalog a policy has. Must show plates in the uncapped count and a
  scored output of zero.

Run:
  uv run python tools/probe_verification_provenance.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from factoriorl.engine_config import resolve_game_speed  # noqa: E402
from factoriorl.env import FactorioEnv  # noqa: E402
from factoriorl.rcon import RCONClient  # noqa: E402
from factoriorl.seeding import Branch, SeedPlan  # noqa: E402
from factoriorl.session import WorkerSession  # noqa: E402
from factoriorl.tasks import get  # noqa: E402
from factoriorl.tasks.reference import Driver, SolveTrace, _build_at  # noqa: E402
from factoriorl.worker import WorkerManager  # noqa: E402

TASK = "construct_smelting_line"
#: Enough ore for more plates than the target. At about 120 ticks per ore by
#: hand this is a few hundred ticks of mining, well inside the budget.
HAND_ORE = 14


def run_out_the_clock(env: FactorioEnv) -> dict:
    """Wait until the episode ends on its own, and report how it ended."""
    wait = env.catalog.wait_index
    steps = 0
    while True:
        _, reward, terminated, truncated, info = env.step(wait)
        steps += 1
        if terminated or truncated:
            verification = info.get("verification")
            return {
                "wait_steps": steps,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "final_step_reward": float(reward),
                "success": bool(info.get("success")),
                "verification": verification,
            }


def reference_arm(env: FactorioEnv) -> dict:
    trace = SolveTrace(task=TASK, budget=env.spec_.max_decision_steps)
    driver = Driver(env, trace)
    built = False
    for x, y in sorted(driver.resource_tiles()):
        if _build_at(driver, (x + 1, y + 1)):
            built = True
            break
        if driver.terminated or driver.truncated:
            break
    outcome = run_out_the_clock(env) if built else {}
    return {
        "built": built,
        "construction_steps": trace.steps,
        "stuck_reason": trace.stuck_reason,
        **outcome,
    }


def _nearest_ore_handle(env: FactorioEnv) -> str | None:
    here = (env._observation.get("character") or {}).get("position") or [0.0, 0.0]
    tiles = [
        tile
        for tile in (env._observation.get("resources") or {}).get("tiles") or []
        if tile.get("name") == "iron-ore" and tile.get("h")
    ]
    if not tiles:
        return None
    return str(min(tiles, key=lambda t: math.dist(t["p"], here))["h"])


def exploit_arm(env: FactorioEnv) -> dict:
    """A furnace, no drill, and ore fed at the last possible moment.

    The timing is the whole exploit. The window measures output *inside* it, so
    ore fed early is smelted during construction and scores nothing under any
    rule. Fed on the last construction decisions, fourteen ore takes about
    2,700 ticks to smelt -- inside the 3,600-tick window -- which is exactly the
    play 1.0.0 paid for in full.
    """
    trace = SolveTrace(task=TASK, budget=env.spec_.max_decision_steps)
    driver = Driver(env, trace)
    ore = driver.nearest_resource()
    if ore is None or not driver.walk_to(ore, tolerance=1.0):
        return {"error": f"could not reach ore: {trace.stuck_reason}"}

    # A furnace on any offered tile. No drill is ever placed.
    furnace = None
    for position in env.argument_domains()["placements"]:
        trace.last_error = None
        if not driver.do(
            "place_at", item="stone-furnace", position=list(position), direction="north"
        ):
            return {"error": f"episode ended while placing: {trace.stuck_reason}"}
        if not trace.last_error and driver.entities_of_type("furnace"):
            furnace = driver.entities_of_type("furnace")[0]
            break
    if furnace is None:
        return {"error": "no offered tile accepted a furnace"}
    handle = str(furnace["h"])

    trace.last_error = None
    driver.do("give_to", to=handle, item="coal", count=20)

    mining_steps = 0
    while driver.inventory().get("iron-ore", 0) < HAND_ORE:
        if driver.terminated or driver.truncated:
            return {"error": "budget ran out while hand-mining"}
        if env._observation.get("inflight"):
            driver.do("wait")
        else:
            target = _nearest_ore_handle(env)
            if target is None:
                return {"error": "no ore handle in view"}
            driver.do("mine_at", handle=target)
        mining_steps += 1
        if mining_steps > 400:
            return {"error": "hand-mining did not converge", "last_error": trace.last_error}
    held = driver.inventory().get("iron-ore", 0)

    # Hold the ore until the last construction decisions.
    last_tick = env.construction_tick_limit - 3 * env.spec_.decision_ticks
    last_step = env.spec_.max_decision_steps - 3
    while env._steps < last_step and int(env._observation.get("tick") or 0) < last_tick:
        if not driver.do("wait"):
            return {"error": "the episode ended before the ore could be fed"}

    driver.do("give_to", to=handle, item="iron-ore", count=20)
    fed = held - driver.inventory().get("iron-ore", 0)
    if driver.terminated or driver.truncated:
        return {"error": "feeding the furnace ended the episode before verification"}
    return {
        "drills_placed": len(driver.entities_of_type("mining-drill")),
        "hand_mined_ore": held,
        "ore_fed_to_furnace": fed,
        "fed_at_step": env._steps,
        "fed_at_tick": int(env._observation.get("tick") or 0),
        "construction_steps": trace.steps,
        **run_out_the_clock(env),
    }


def main() -> int:
    task = get(TASK)
    manager = WorkerManager()
    handle = manager.launch("provenance-probe")
    report: dict = {
        "task": TASK,
        "task_version": task.spec.version,
        "verification": task.spec.verification.to_dict(),
        "engine": manager.engine.to_dict(),
    }
    started = time.perf_counter()
    try:
        with RCONClient(handle.spec.rcon_endpoint, timeout=30.0) as client:
            client.lua(f"game.speed = {resolve_game_speed()} return game.speed")
        session = WorkerSession(handle, timeout=60.0)
        session.status()
        plan = SeedPlan(master=777, run_id="provenance-probe")

        for name, arm in (("reference", reference_arm), ("exploit", exploit_arm)):
            env = FactorioEnv(task, session, plan, branch=Branch.TRAIN, split="train")
            # The same scene for both arms.
            env._episode_index = -1
            env.reset()
            report[name] = arm(env)
            print(name, json.dumps(report[name], default=str)[:400], flush=True)
        session.close()
    finally:
        manager.cleanup(handle)
    report["wall_seconds"] = round(time.perf_counter() - started, 1)

    reference, exploit = report.get("reference", {}), report.get("exploit", {})
    ref_verification = reference.get("verification") or {}
    exp_verification = exploit.get("verification") or {}
    report["verdict"] = {
        "reference_built_a_line": bool(reference.get("built")),
        "reference_ended_by_budget_and_was_verified": bool(
            reference.get("terminated") and not reference.get("truncated") and ref_verification
        ),
        "reference_clears_the_target": bool(ref_verification.get("success")),
        "exploit_placed_no_drill": exploit.get("drills_placed") == 0,
        # Plates the old rule would have paid for, made inside the window.
        "exploit_furnace_made_plates_in_the_window": (exp_verification.get("uncapped_output") or 0)
        > 0,
        "exploit_ended_by_budget_and_was_verified": bool(
            exploit.get("terminated") and not exploit.get("truncated") and exp_verification
        ),
        "exploit_scores_nothing": exp_verification.get("machine_output") == 0
        and not exp_verification.get("success"),
    }
    out = ROOT / "docs" / "evidence" / "m1-verification-provenance.json"
    out.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report["verdict"], indent=2))
    print(f"wrote {out}")
    return 0 if all(report["verdict"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
