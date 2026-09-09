"""How long does a plate line keep producing after its fuel is taken? (R4.2)

R4.2: *"Removing inventory fuel need not stop a burner immediately. Measure the
actual production effect after accounting for stored energy and material in
transit. Define outage and recovery criteria before evaluating."*

The Phase 5 demonstration never measured this, and the omission is what makes
its recovery claim unsupportable. It emptied both fuel inventories at tick 4050
with `get_fuel_inventory():clear()` -- which does not touch `entity.energy` --
recorded `drill_status: 1` (**working**) with `drill_fuel: 0` at that same tick,
and then handed control back to the agent, which refuelled in **3 decisions =
90 ticks**. One coal is 4 MJ against a 150 kW drill, so a full item in the
burner's buffer runs on the order of 1,600 ticks. If the line never stopped,
`sustained_production_restored: true` measured nothing.

The repo had already noticed the effect twice without quantifying it:
`tools/demonstration.py`'s `refuelled()` records *"the furnace keeps producing
for a while on heat it already had, so a plate counter says the line recovered
when nothing has been refuelled at all"*, and the first "passing" run made two
plates in its second window *"while the drill sat at status 53, out of fuel"*.

So this measures the decay curve: fuel a line, reach steady state, empty the
fuel inventories, and sample until production stops.

Two things about how it samples, both load-bearing:

**It advances through `env.advance`, in `decision_ticks` chunks.** Not
`session.step` around the env and not an unpaused poll loop.
`tools/symmetry_probe.py` sets `game.tick_paused = false` and polls with
`time.sleep`, so its windows are not tick-aligned -- fine for independent
per-candidate measurements, wrong for anything whose timing is the result.

**Stored energy comes from truth, not the observation.** `world.truth()`
publishes `stored_energy` and `remaining_burning_fuel` per prototype; the
observation carries neither, so measuring this cannot change what a policy
sees.

Run: uv run python tools/burner_decay.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from factoriorl.engine_config import resolve_game_speed  # noqa: E402
from factoriorl.env import FactorioEnv  # noqa: E402
from factoriorl.paths import evidence_dir  # noqa: E402
from factoriorl.rcon import RCONClient  # noqa: E402
from factoriorl.seeding import Branch, SeedPlan  # noqa: E402
from factoriorl.session import WorkerSession  # noqa: E402
from factoriorl.tasks import get  # noqa: E402
from factoriorl.worker import WorkerManager  # noqa: E402

TASK = "plate_line"

#: Fuel each machine before the run, so the line reaches steady state without
#: needing an agent. Evaluator setup, declared: this is not an agent result.
FUEL_BOTH = """
local s = game.surfaces[1]
local fuelled = {}
for _, name in pairs({'burner-mining-drill', 'stone-furnace'}) do
  for _, e in pairs(s.find_entities_filtered({name = name})) do
    local inv = e.get_fuel_inventory()
    if inv then
      inv.insert({name = 'coal', count = %d})
      fuelled[#fuelled + 1] = name
    end
  end
end
return table.concat(fuelled, ',')
"""

#: The disruption, byte-identical in effect to the demonstration's: the fuel
#: *inventory* only. `entity.energy` is deliberately untouched -- that is the
#: whole point of the measurement.
EMPTY_FUEL = """
local s = game.surfaces[1]
local removed = {}
for _, name in pairs({'burner-mining-drill', 'stone-furnace'}) do
  for _, e in pairs(s.find_entities_filtered({name = name})) do
    local inv = e.get_fuel_inventory()
    if inv then
      local taken = 0
      for _, stack in pairs(inv.get_contents()) do taken = taken + stack.count end
      inv.clear()
      removed[#removed + 1] = name .. '=' .. taken
    end
  end
end
return 'tick=' .. game.tick .. '|' .. table.concat(removed, ',')
"""


def _sample(env: FactorioEnv) -> dict:
    """One row of the curve, from the observation and truth the env just took."""
    observation = env._observation
    truth = env._truth
    machines = {}
    for record in observation.get("entities") or []:
        kind = record.get("type")
        if kind in ("mining-drill", "furnace"):
            machines[kind] = {
                "status": record.get("st"),
                "working": bool(record.get("working")),
                "fuel": record.get("fuel") or {},
                "output": record.get("output") or {},
            }
    return {
        "tick": int(observation.get("tick") or 0),
        "plates": float((truth.get("produced") or {}).get("iron-plate", 0)),
        "working_counts": dict(truth.get("working_counts") or {}),
        # Evaluator-only, and the quantity the old measurement lacked.
        "stored_energy": {k: round(v, 1) for k, v in (truth.get("stored_energy") or {}).items()},
        "remaining_burning_fuel": {
            k: round(v, 1) for k, v in (truth.get("remaining_burning_fuel") or {}).items()
        },
        "machines": machines,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--settle-ticks",
        type=int,
        default=3600,
        help="run the fuelled line this long before removing its fuel, so the "
        "decay is measured from steady state rather than from start-up",
    )
    parser.add_argument(
        "--observe-ticks",
        type=int,
        default=7200,
        help="how long to keep sampling after the fuel is removed. Two windows: "
        "long enough that a line still producing at the end is a finding, not a "
        "truncation",
    )
    parser.add_argument("--coal", type=int, default=50)
    parser.add_argument("--out", default="r4-burner-decay.json")
    args = parser.parse_args()

    report: dict = {
        "experiment": "R4.2, first half: production after the fuel inventory is emptied",
        "task": TASK,
        "why": (
            "The Phase 5 demonstration removed fuel at tick 4050, recorded the drill "
            "still `working` with zero fuel, and refuelled 90 ticks later. Nothing "
            "measured whether the line ever stopped, so its recovery claim rests on "
            "an outage that may not have happened."
        ),
        "method": {
            "disruption": "get_fuel_inventory():clear() on both machines; entity.energy untouched",
            "advance": "env.advance, in decision_ticks chunks, tick-aligned through the typed path",
            "stored_energy_source": "world.truth(); the observation carries neither field",
        },
        "settle_ticks": args.settle_ticks,
        "observe_ticks": args.observe_ticks,
    }

    manager = WorkerManager()
    handle = manager.launch("burner-decay")
    session = None
    try:
        with RCONClient(handle.spec.rcon_endpoint, timeout=30.0) as client:
            client.lua(f"game.speed = {resolve_game_speed()} return game.speed")
        session = WorkerSession(handle, timeout=60.0)
        session.status()
        env = FactorioEnv(
            get(TASK),
            session,
            SeedPlan(master=20260909, run_id="burner-decay"),
            branch=Branch.TRAIN,
            split="train",
        )
        env._episode_index = -1
        env.reset()

        with RCONClient(handle.spec.rcon_endpoint, timeout=30.0) as client:
            report["fuelled"] = str(client.lua(FUEL_BOTH % args.coal))
        env.resync("evaluator fuelled both machines to reach steady state")

        # ---- steady state -------------------------------------------------
        print(f"settling for {args.settle_ticks} ticks...", flush=True)
        settle: list[dict] = [_sample(env)]
        advanced = 0
        while advanced < args.settle_ticks:
            advanced += env.advance(env.spec_.decision_ticks * 10)
            settle.append(_sample(env))
        report["steady_state"] = settle[-1]
        report["settle_curve"] = settle

        # ---- the disruption ----------------------------------------------
        with RCONClient(handle.spec.rcon_endpoint, timeout=30.0) as client:
            report["disruption"] = str(client.lua(EMPTY_FUEL))
        env.resync("fuel inventories emptied")
        at_removal = _sample(env)
        report["at_removal"] = at_removal
        print(f"fuel removed at tick {at_removal['tick']}", flush=True)

        # ---- the decay ----------------------------------------------------
        curve: list[dict] = [at_removal]
        advanced = 0
        while advanced < args.observe_ticks:
            advanced += env.advance(env.spec_.decision_ticks)
            curve.append(_sample(env))
        report["decay_curve"] = curve
        env_decision_ticks = env.spec_.decision_ticks
        session.close()
    finally:
        manager.cleanup(handle)

    # ---- what the curve says -------------------------------------------
    base = report["at_removal"]["plates"]
    last_growth = None
    for row in report["decay_curve"]:
        if row["plates"] > base:
            last_growth = row
            base = row["plates"]
    removal_tick = report["at_removal"]["tick"]
    final = report["decay_curve"][-1]
    report["findings"] = {
        "plates_at_removal": report["at_removal"]["plates"],
        "plates_at_end": final["plates"],
        "plates_after_removal": final["plates"] - report["at_removal"]["plates"],
        "last_tick_production_rose": last_growth["tick"] if last_growth else None,
        "ticks_producing_after_removal": (last_growth["tick"] - removal_tick if last_growth else 0),
        "still_producing_at_end": bool(
            final["plates"]
            > report["decay_curve"][max(0, len(report["decay_curve"]) - 30)]["plates"]
        ),
        "working_at_removal": report["at_removal"]["working_counts"],
        "stored_energy_at_removal": report["at_removal"]["stored_energy"],
    }
    # Cross-check the measurement against the machine's own numbers, so the
    # curve is not just asserted: a burner mining drill draws 150 kW, so the
    # energy left in the item it is burning divides into a tick count. If the
    # measured run-on disagrees with that, one of the two is wrong.
    DRILL_WATTS = 150_000
    joules = (report["at_removal"]["remaining_burning_fuel"] or {}).get("burner-mining-drill", 0.0)
    implied = int(joules / DRILL_WATTS * 60) if joules else 0
    measured = report["findings"]["ticks_producing_after_removal"] or 0
    report["findings"]["drill_remaining_burning_fuel_joules"] = joules
    report["findings"]["drill_watts"] = DRILL_WATTS
    report["findings"]["implied_run_on_ticks"] = implied
    report["findings"]["measured_agrees_with_arithmetic"] = bool(
        implied and abs(measured - implied) <= 2 * env_decision_ticks
    )

    demonstration_recovery_ticks = 90
    report["findings"]["demonstration_recovery_ticks"] = demonstration_recovery_ticks
    report["findings"]["line_was_still_running_when_the_agent_refuelled"] = bool(
        (report["findings"]["ticks_producing_after_removal"] or 0) > demonstration_recovery_ticks
    )

    path = evidence_dir() / args.out
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    found = report["findings"]
    print(
        f"\nproduced {found['plates_after_removal']:.0f} more plates after removal; "
        f"production last rose {found['ticks_producing_after_removal']} ticks after it",
        flush=True,
    )
    verdict = (
        "STILL RUNNING"
        if found["line_was_still_running_when_the_agent_refuelled"]
        else "already stopped"
    )
    print(
        f"the demonstration refuelled {demonstration_recovery_ticks} ticks after "
        f"removal, so the line was {verdict} when it did",
        flush=True,
    )
    print(f"wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
