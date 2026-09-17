"""Which tile a burner drill mines next, and when a trigger technology lands (M3).

Two things the golden traces showed but could not explain:

- **Drill tile order.** A burner drill takes ten ore from one of the four tiles
  under it and then moves to another. The smelting reference visited them
  top-right, bottom-left, top-left, bottom-right; the build_line reference
  went top-right, bottom-left, bottom-right. The difference could be the
  drill's position relative to a chunk boundary, the order the resource tiles
  were created in, or the facing. Six drills separate those: in one chunk and
  on a chunk corner, resources created x-first and y-first, and all four
  facings. Each runs long enough to visit every tile at least twice, and the
  tile it took each ore from is recorded.
- **Trigger research lag.** `steam-power` completes once 50 iron plates have
  been crafted, but the traces showed its recipes some ticks after the 50th
  plate. Four furnaces smelt past 50 plates, sampled every tick near the
  threshold, and the tick the plate count reached 50 is recorded beside the
  tick `steam-engine` became enabled.

Writes `docs/evidence/sim-mechanics-m3-drills.json`.

Run:
  uv run python tools/probe_drill_and_research.py
"""

from __future__ import annotations

import json
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
from factoriorl.worker import WorkerManager  # noqa: E402

OUT = ROOT / "docs" / "evidence" / "sim-mechanics-m3-drills.json"

#: 40 ore at 240 ticks each: every tile twice, whatever the order.
DRILL_TICKS = 10_000
DRILL_SAMPLE = 60

# name, drill centre, facing, creation order ("x" = x outer loop)
DRILLS = (
    ("in_chunk_south_x", (200, 200), "south", "x"),
    ("in_chunk_south_y", (210, 200), "south", "y"),
    ("chunk_corner_south_x", (224, 224), "south", "x"),
    ("in_chunk_north_x", (200, 210), "north", "x"),
    ("in_chunk_east_x", (210, 210), "east", "x"),
    ("in_chunk_west_x", (200, 220), "west", "x"),
    ("chunk_corner_north_x", (224, 256), "north", "x"),
)

DRILL_SETUP = """
local s = game.surfaces[1]
local specs = %s
storage.drills = {}
for _, spec in ipairs(specs) do
  local cx, cy = spec[2], spec[3]
  local tiles = {}
  if spec[5] == "x" then
    for dx = -1, 0 do for dy = -1, 0 do tiles[#tiles + 1] = {cx + dx, cy + dy} end end
  else
    for dy = -1, 0 do for dx = -1, 0 do tiles[#tiles + 1] = {cx + dx, cy + dy} end end
  end
  for _, t in ipairs(tiles) do
    s.create_entity({name = "iron-ore", position = {t[1] + 0.5, t[2] + 0.5}, amount = 10000})
  end
  local d = s.create_entity({name = "burner-mining-drill", position = {cx, cy},
    direction = defines.direction[spec[4]], force = "player"})
  d.insert({name = "coal", count = 50})
  local chest = s.create_entity({name = "iron-chest", position = d.drop_position,
    force = "player"})
  storage.drills[spec[1]] = {drill = d, chest = chest, cx = cx, cy = cy}
end
return true
"""

DRILL_SAMPLE_LUA = """
local s = game.surfaces[1]
local out = {tick = game.tick}
for name, rig in pairs(storage.drills) do
  local amounts = {}
  for dx = -1, 0 do for dy = -1, 0 do
    local r = s.find_entities_filtered({position = {rig.cx + dx + 0.5, rig.cy + dy + 0.5},
      radius = 0.1, type = "resource"})[1]
    amounts[#amounts + 1] = {dx, dy, r and r.amount or 0}
  end end
  out[name] = amounts
end
return out
"""

TRIGGER_SETUP = """
local s = game.surfaces[1]
storage.furnaces = {}
for k = 1, 4 do
  local f = s.create_entity({name = "stone-furnace", position = {140 + 3 * k, 140},
    force = "player"})
  f.insert({name = "coal", count = 5})
  f.get_inventory(defines.inventory.furnace_source).insert({name = "iron-ore", count = 20})
  storage.furnaces[k] = f
end
return game.tick
"""

TRIGGER_SAMPLE = """
local n = 0
for _, f in ipairs(storage.furnaces) do
  n = n + f.get_inventory(defines.inventory.furnace_result).get_item_count("iron-plate")
end
local force = game.forces.player
return {tick = game.tick, plates = n,
  researched = force.technologies["steam-power"].researched,
  engine_enabled = force.recipes["steam-engine"].enabled}
"""


def lua_specs() -> str:
    rows = [
        f'{{"{name}", {c[0]}, {c[1]}, "{facing}", "{order}"}}' for name, c, facing, order in DRILLS
    ]
    return "{" + ", ".join(rows) + "}"


def drill_sequence(samples: list[dict], name: str) -> list[list[int]]:
    """Which tile (dx, dy) lost ore between consecutive samples, in order."""
    out = []
    for before, after in zip(samples, samples[1:], strict=False):
        for (dx, dy, a), (_, _, b) in zip(before[name], after[name], strict=True):
            for _ in range(a - b):
                out.append([dx, dy])
    return out


def main() -> int:
    manager = WorkerManager()
    handle = manager.launch("drill-order")
    report: dict = {"engine": manager.engine.to_dict()}
    started = time.perf_counter()
    try:
        with RCONClient(handle.spec.rcon_endpoint, timeout=60.0) as client:
            client.lua(f"game.speed = {resolve_game_speed()} return game.speed")
            session = WorkerSession(handle, timeout=60.0)
            session.status()
            env = FactorioEnv(
                get("construct_smelting_line"),
                session,
                SeedPlan(master=41, run_id="drill-order"),
                branch=Branch.TRAIN,
                split="train",
            )
            env.reset(options={"scene_index": 0})

            # Trigger research, on a fresh counter.
            start = int(client.lua(TRIGGER_SETUP))
            samples = []
            while True:
                sample = client.lua(TRIGGER_SAMPLE)
                sample["t"] = sample["tick"] - start
                samples.append(sample)
                if sample["engine_enabled"] or sample["t"] > 4000:
                    break
                env.advance(1 if sample["plates"] >= 46 else 30)
            fifty = next((s for s in samples if s["plates"] >= 50), None)
            done = next((s for s in samples if s["engine_enabled"]), None)
            report["trigger"] = {
                "samples_near": [s for s in samples if s["plates"] >= 48],
                "fiftieth_plate_t": fifty and fifty["t"],
                "recipes_enabled_t": done and done["t"],
                "researched_at_enable": done and done["researched"],
            }
            print(
                "trigger",
                report["trigger"]["fiftieth_plate_t"],
                report["trigger"]["recipes_enabled_t"],
            )

            # Drill tile order.
            client.lua(DRILL_SETUP % lua_specs())
            samples = [client.lua(DRILL_SAMPLE_LUA)]
            for _ in range(DRILL_TICKS // DRILL_SAMPLE):
                env.advance(DRILL_SAMPLE)
                samples.append(client.lua(DRILL_SAMPLE_LUA))
            report["drills"] = {
                name: {
                    "centre": list(centre),
                    "facing": facing,
                    "creation_order": order,
                    "sequence": drill_sequence(samples, name),
                }
                for name, centre, facing, order in DRILLS
            }
            session.close()
    finally:
        manager.cleanup(handle)
    report["wall_seconds"] = round(time.perf_counter() - started, 1)
    OUT.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    for name, row in report["drills"].items():
        seq = row["sequence"]
        runs = []
        for tile in seq:
            if runs and runs[-1][0] == tile:
                runs[-1][1] += 1
            else:
                runs.append([tile, 1])
        print(f"{name:24} {[(tuple(t), n) for t, n in runs]}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
