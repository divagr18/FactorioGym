"""Gate A4.2: can machine output be told apart from handcrafting?

A4.2 requires cumulative machine production be kept separate from inventory
transfers and handcrafting. The engine will not do it: its item production
statistics count every item that enters the force's inventory space, so a plate
a furnace made, a plate a character hand-crafted and an ore a character mined
are the same number.

So it is derived -- `machine_produced = produced - handcrafted - mined` -- and
this measures whether the derivation actually holds against a running game. It
does the two things that must come out different:

  1. **Hand-craft** an iron gear wheel. Every gear must land in the by-hand
     column and none in the machine column.
  2. **Smelt** iron plates in a furnace. Those must land in the machine column.

Two different items, deliberately. An iron *plate* cannot be hand-crafted at
all -- its recipe is in the `smelting` category, which a character has no
crafting machine for -- so "hand-craft a plate and check it is not counted as
machine output" is a test of something that cannot happen. The first version of
this probe asked for exactly that and got zero plates, zero everything, and a
green column that meant nothing.

It also answers a question that cannot be reasoned out from documentation:
whether `on_player_crafted_item` and `on_player_mined_item` fire at all for a
character the mod controls. They are player events, and a controlled character
need not have a `LuaPlayer` behind it. The mod counts both ways and reports
which one moved.

Run:
  uv run python tools/probe_production.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

sys.path.insert(0, str(ROOT / "tools"))

from probe_bootstrap import Tools, mine, walk_to  # noqa: E402

from factoriorl import worlds  # noqa: E402
from factoriorl.freeplay import starting_inventory  # noqa: E402
from factoriorl.open_world import OpenWorldEnv  # noqa: E402
from factoriorl.rcon import RCONClient  # noqa: E402
from factoriorl.session import WorkerSession  # noqa: E402
from factoriorl.worker import WorkerManager  # noqa: E402


def truth(session: WorkerSession) -> dict:
    return session.truth().response.result or {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--out", default=str(ROOT / "docs" / "evidence" / "a4-production.json"))
    args = parser.parse_args()

    mode = worlds.get("open_factory")
    manager = WorkerManager()
    freeplay = starting_inventory(manager.engine.executable)
    report: dict = {
        "measures": "roadmap A4.2: machine output separated from handcrafting and mining",
        "world": mode.to_dict(),
        "engine": manager.engine.to_dict(),
    }
    started = time.perf_counter()

    handle = manager.launch("probe-a4", map_seed=args.seed, terrain=mode.terrain)
    try:
        with RCONClient(handle.spec.rcon_endpoint, timeout=60.0) as client:
            client.lua("game.speed = 1.0 return game.speed")
        session = WorkerSession(handle, timeout=60.0)
        session.status()
        session.configure(free_running=True)
        env = OpenWorldEnv(mode, session, inventory=freeplay["items"], seed=args.seed)
        env.reset()
        tools = Tools(env)

        # --- 1. mine, then hand-craft: nothing here is a machine ----------
        walk_to(tools, "iron-ore", report, "ore")
        mine(tools, "iron-ore", 8, report, "ore")
        report["after_mining"] = truth(session)

        # `count` draws from the `amounts` domain -- 1, 5, 20 -- like every
        # other argument. Asking for 2 is refused, which is the catalog working
        # correctly and the probe getting it wrong.
        report["handcraft"] = tools.do("craft_recipe", recipe="iron-gear-wheel", count=1)
        for _ in range(4):
            tools.do("wait_for", until="nothing_in_flight", seconds=10)
        report["after_handcraft"] = truth(session)
        report["gears_in_hand"] = tools.inventory("iron-gear-wheel")
        report["plates_in_hand"] = tools.inventory("iron-plate")

        # --- 2. smelt: this one *is* a machine ----------------------------
        report["placed_furnace"] = False
        if tools.inventory("stone-furnace"):
            for spot in tools.domain("placements")[:12]:
                placed = tools.do(
                    "place_at", item="stone-furnace", position=list(spot), direction="north"
                )
                if placed.get("status") == "completed":
                    report["placed_furnace"] = True
                    break
        furnace = tools.entity_handle("stone-furnace")
        if furnace:
            fuel = next((i for i in ("coal", "wood") if tools.inventory(i)), None)
            if fuel:
                tools.do("give_to", to=furnace, item=fuel, count=1)
            mine(tools, "iron-ore", 5, report, "ore2")
            held = tools.inventory("iron-ore")
            report["ore_to_furnace"] = tools.do(
                "give_to",
                to=furnace,
                item="iron-ore",
                # Largest legal amount the character is actually holding.
                count=max((n for n in tools.domain("amounts") if n <= held), default=1),
            )
            for _ in range(3):
                tools.do("wait_for", until="world_changes", seconds=30)
        report["after_smelting"] = truth(session)
        report["actions"] = tools.log
        session.close()
    finally:
        manager.cleanup(handle)

    mined_at = report.get("after_mining") or {}
    handcrafted_at = report.get("after_handcraft") or {}
    smelted_at = report.get("after_smelting") or {}

    def column(state: dict, key: str, item: str) -> float:
        return float((state.get(key) or {}).get(item, 0))

    report["wall_seconds"] = round(time.perf_counter() - started, 1)
    report["by_hand_source"] = smelted_at.get("by_hand_source")
    report["tracked_item_set"] = smelted_at.get("tracked_item_set")
    report["summary"] = {
        "ore_mined_by_hand": column(mined_at, "mined_by_hand", "iron-ore"),
        "ore_in_machine_column_after_mining": column(mined_at, "machine_produced", "iron-ore"),
        "gears_handcrafted": column(handcrafted_at, "handcrafted", "iron-gear-wheel"),
        "gears_in_machine_column": column(handcrafted_at, "machine_produced", "iron-gear-wheel"),
        "plates_in_machine_column_after_handcraft": column(
            handcrafted_at, "machine_produced", "iron-plate"
        ),
        "plates_in_machine_column_after_smelting": column(
            smelted_at, "machine_produced", "iron-plate"
        ),
        "plates_produced_total": column(smelted_at, "produced", "iron-plate"),
    }
    checks = report["summary"]
    report["verdict"] = {
        # Mining is not production by a machine, whatever the engine's counter says.
        "mining_is_counted_by_hand": checks["ore_mined_by_hand"] > 0,
        "mining_is_not_counted_as_machine_output": (
            checks["ore_in_machine_column_after_mining"] == 0
        ),
        # Handcrafting is not production by a machine either.
        "handcrafting_is_counted_by_hand": checks["gears_handcrafted"] > 0,
        "handcrafting_is_not_counted_as_machine_output": checks["gears_in_machine_column"] == 0,
        # And a furnace is.
        "smelting_is_counted_as_machine_output": (
            checks["plates_in_machine_column_after_smelting"]
            > checks["plates_in_machine_column_after_handcraft"]
        ),
        # The subtraction has to add up, or the three columns are decoration.
        "the_columns_add_up": (
            checks["plates_produced_total"] >= checks["plates_in_machine_column_after_smelting"]
        ),
        "the_tracked_set_is_live": smelted_at.get("tracked_item_set") == "live",
    }
    Path(args.out).write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report["verdict"], indent=2))
    print(json.dumps(report["summary"], indent=2))
    print(f"by-hand source: {report['by_hand_source']} | tracked set: {report['tracked_item_set']}")
    print(f"wrote {args.out}")
    return 0 if all(report["verdict"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
