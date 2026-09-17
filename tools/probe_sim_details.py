"""Measure the engine details the golden traces cannot pin down (M3, B0).

`tools/probe_sim_mechanics.py` measured the rates: walking speed, mining and
smelting intervals, fuel draw. The simulator also has to reproduce what happens
*between* decisions and at edges the recorded scenarios never touch. This
measures those against Factorio 2.0.60 and writes
`docs/evidence/sim-mechanics-m3.json`:

- **burner, tick by tick**: `energy`, `remaining_burning_fuel`, fuel count,
  mining or crafting progress and status for a drill and a furnace from the
  moment they are fuelled, across their first output and across the tick
  their last fuel runs out. This is the energy model: buffer size, warm-up,
  and run-on.
- **reach**: `can_reach_entity` swept outwards from a resource tile, a drill, a
  furnace and a wall along an axis and a diagonal, in 1/32-tile steps, beside
  the centre and bounding-box distances.
- **insertion**: `can_insert` and `insert` for the combinations `transfer` can
  reach: fuel into a full fuel slot, ore into a full or nearly full source,
  plates into each furnace inventory, ore into a drill.
- **collision**: prototype collision boxes, and the entity-sweep boundary of
  `find_entities_filtered{position, radius}` for 1x1 and 2x2 entities.
- **script-set state**: whether `walking_state` and `mining_state` read back
  what was written on the same tick.
- **number formatting**: how `helpers.table_to_json` writes doubles.
- **a drill blocked by a foreign pile**, and the source count at which a
  drill stops feeding an unfuelled furnace.

All rigs are raw Lua far from the scene. Time advances through `env.advance`.

Run:
  uv run python tools/probe_sim_details.py
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

OUT = ROOT / "docs" / "evidence" / "sim-mechanics-m3.json"

#: Ticks sampled one by one after fuelling: past the drill's 1,600 ticks on one
#: coal and the furnace's 2,667.
BURNER_TICKS = 2800
#: Only these relative-tick windows are written out; transitions and arrivals
#: are computed over every tick.
KEEP_WINDOWS = ((0, 20), (236, 250), (476, 490), (1592, 1610), (2300, 2312), (2658, 2676))

#: Rig origin, well outside every scene's 48-tile box.
OX, OY = 160, 160

PRELUDE = """
local s = game.surfaces[1]
local function try(fn) local ok, v = pcall(fn) if ok then return v end return nil end
local function g(v) if v == nil then return nil end return string.format("%.17g", v) end
local names = {}
for name, value in pairs(defines.entity_status) do names[value] = name end
local function status(e) return names[try(function() return e.status end)] end
local function count(inv, item) return inv and inv.get_item_count(item) or 0 end
storage.probe = storage.probe or {}
"""

# ---------------------------------------------------------------- burner

BURNER_SETUP = """
for _, e in pairs(s.find_entities_filtered({area = {{OX - 20, OY - 20}, {OX + 20, OY + 20}}})) do
  if e.valid and e.type ~= "character" then e.destroy() end
end
for dx = -1, 0 do for dy = -1, 0 do
  s.create_entity({name = "iron-ore", position = {OX + dx + 0.5, OY + dy + 0.5}, amount = 5000})
  s.create_entity({name = "iron-ore", position = {OX + 8 + dx + 0.5, OY + dy + 0.5},
    amount = 5000})
end end
-- Drill A drops on the ground (nothing at its drop position).
local a = s.create_entity({name = "burner-mining-drill", position = {OX, OY},
  direction = defines.direction.north, force = "player"})
-- Drill B drops onto a coal pile placed at its drop position beforehand.
local b = s.create_entity({name = "burner-mining-drill", position = {OX + 8, OY},
  direction = defines.direction.north, force = "player"})
s.spill_item_stack({position = b.drop_position, stack = {name = "coal", count = 1},
  enable_looted = false, allow_belts = false})
-- Furnace C: one coal, twelve ore, so it stops for ore before fuel.
local c = s.create_entity({name = "stone-furnace", position = {OX + 16, OY}, force = "player"})
c.get_inventory(defines.inventory.furnace_source).insert({name = "iron-ore", count = 12})
-- Drill D drops into a chest, so it keeps working until its one coal is gone.
for dx = -1, 0 do for dy = -1, 0 do
  s.create_entity({name = "iron-ore", position = {OX + dx + 0.5, OY + 8 + dy + 0.5},
    amount = 5000})
end end
local d = s.create_entity({name = "burner-mining-drill", position = {OX, OY + 8},
  direction = defines.direction.north, force = "player"})
local chest = s.create_entity({name = "wooden-chest", position = d.drop_position,
  force = "player"})
-- Furnace E: one coal, thirty ore, so it stops for fuel.
local e = s.create_entity({name = "stone-furnace", position = {OX + 16, OY + 8},
  force = "player"})
e.get_inventory(defines.inventory.furnace_source).insert({name = "iron-ore", count = 30})
storage.probe.a, storage.probe.b, storage.probe.c = a, b, c
storage.probe.d, storage.probe.e, storage.probe.chest = d, e, chest
local proto = function(e)
  local p = e.prototype
  local bp = try(function() return p.burner_prototype end)
  return {
    energy_usage = try(function() return p.energy_usage end),
    max_energy_usage = try(function() return p.max_energy_usage end),
    effectivity = bp and try(function() return bp.effectivity end),
    fuel_slots = bp and try(function() return bp.fuel_inventory_size end),
    burnt_slots = bp and try(function() return bp.burnt_inventory_size end),
    drop_position = try(function() return {e.drop_position.x, e.drop_position.y} end),
    position = {e.position.x, e.position.y},
  }
end
return {a = proto(a), b = proto(b), c = proto(c), d = proto(d), e = proto(e),
  chest = chest and {chest.position.x, chest.position.y},
  b_pile = #s.find_entities_filtered({position = b.drop_position, radius = 0.5,
    type = "item-entity"})}
"""

BURNER_FUEL = """
storage.probe.a.insert({name = "coal", count = 1})
storage.probe.b.insert({name = "coal", count = 1})
storage.probe.c.insert({name = "coal", count = 1})
storage.probe.d.insert({name = "coal", count = 1})
storage.probe.e.insert({name = "coal", count = 1})
return game.tick
"""

BURNER_SAMPLE = """
local function burner(e, progress)
  local b = e.burner
  local current = try(function() return b.currently_burning end)
  local current_name = current and try(function() return current.name.name end)
    or (current and try(function() return current.name end))
  return {
    status = status(e),
    energy = g(e.energy),
    remaining = g(b.remaining_burning_fuel),
    burning = type(current_name) == "string" and current_name or nil,
    heat = g(try(function() return b.heat end)),
    fuel = count(e.get_fuel_inventory(), "coal"),
    progress = g(progress(e)),
  }
end
local drill = function(e) return e.mining_progress end
local furnace = function(e) return e.crafting_progress end
local c = storage.probe.c
local piles = {}
for _, p in pairs(s.find_entities_filtered({area = {{OX - 3, OY - 4}, {OX + 11, OY + 1}},
    type = "item-entity"})) do
  piles[#piles + 1] = {p.stack.name, p.stack.count, p.position.x, p.position.y}
end
table.sort(piles, function(x, y)
  if x[3] ~= y[3] then return x[3] < y[3] end
  return x[4] < y[4]
end)
return {
  tick = game.tick,
  a = burner(storage.probe.a, drill),
  b = burner(storage.probe.b, drill),
  c = burner(c, furnace),
  d = burner(storage.probe.d, drill),
  e = burner(storage.probe.e, furnace),
  d_chest = count(storage.probe.chest.get_inventory(defines.inventory.chest), "iron-ore"),
  e_source = count(storage.probe.e.get_inventory(defines.inventory.furnace_source), "iron-ore"),
  e_result = count(storage.probe.e.get_inventory(defines.inventory.furnace_result),
    "iron-plate"),
  c_source = count(c.get_inventory(defines.inventory.furnace_source), "iron-ore"),
  c_result = count(c.get_inventory(defines.inventory.furnace_result), "iron-plate"),
  piles = piles,
}
"""

# ---------------------------------------------------------------- reach

REACH = """
local ox, oy = OX + 40, OY
for _, e in pairs(s.find_entities_filtered({area = {{ox - 20, oy - 20}, {ox + 20, oy + 20}}})) do
  if e.valid and e.type ~= "character" then e.destroy() end
end
local targets = {
  resource = s.create_entity({name = "iron-ore", position = {ox + 0.5, oy + 0.5}, amount = 10}),
  wall = s.create_entity({name = "stone-wall", position = {ox + 0.5, oy - 5.5}, force = "player"}),
  drill = s.create_entity({name = "burner-mining-drill", position = {ox, oy + 6},
    force = "player"}),
  furnace = s.create_entity({name = "stone-furnace", position = {ox, oy - 12},
    force = "player"}),
}
local ch = storage.frrl_character
local home = {ch.position.x, ch.position.y}
local out = {}
for name, t in pairs(targets) do
  local box = t.bounding_box
  local rows = {}
  for _, axis in ipairs({{1, 0}, {0.7071067811865476, 0.7071067811865476}}) do
    local last_true, first_false = nil, nil
    for step = 0, 32 * 14 do
      local d = step / 32
      local x = t.position.x + axis[1] * d
      local y = t.position.y + axis[2] * d
      ch.teleport({x, y})
      local p = ch.position
      local can = ch.can_reach_entity(t)
      local cd = math.sqrt((p.x - t.position.x) ^ 2 + (p.y - t.position.y) ^ 2)
      local bx = math.max(box.left_top.x - p.x, 0, p.x - box.right_bottom.x)
      local by = math.max(box.left_top.y - p.y, 0, p.y - box.right_bottom.y)
      local bd = math.sqrt(bx * bx + by * by)
      if can then last_true = {d, g(cd), g(bd)} elseif not first_false then
        first_false = {d, g(cd), g(bd)}
      end
    end
    rows[#rows + 1] = {axis = axis, last_true = last_true, first_false = first_false}
  end
  out[name] = {
    rows = rows,
    box = {box.left_top.x, box.left_top.y, box.right_bottom.x, box.right_bottom.y},
  }
end
ch.teleport(home)
out.reach_distance = ch.reach_distance
out.resource_reach_distance = ch.resource_reach_distance
out.build_distance = ch.build_distance
for _, t in pairs(targets) do if t.valid then t.destroy() end end
return out
"""

# ---------------------------------------------------------------- insertion

INSERTION = """
local ox, oy = OX + 60, OY
local function furnace(dx)
  return s.create_entity({name = "stone-furnace", position = {ox + dx, oy}, force = "player"})
end
local out = {}
local f = furnace(0)
local fuel = f.get_fuel_inventory()
fuel.insert({name = "coal", count = 50})
out.full_fuel = {can = fuel.can_insert({name = "coal", count = 1}),
  can_20 = fuel.can_insert({name = "coal", count = 20}),
  insert = fuel.insert({name = "coal", count = 20}), size = #fuel}
local src = f.get_inventory(defines.inventory.furnace_source)
out.source_size = #src
out.source_49 = {inserted_49 = src.insert({name = "iron-ore", count = 49}),
  can_20 = src.can_insert({name = "iron-ore", count = 20}),
  insert_20 = src.insert({name = "iron-ore", count = 20}),
  now = count(src, "iron-ore")}
local g2 = furnace(4)
local per = {}
for i, which in ipairs({1, 2, 3}) do
  local inv = g2.get_inventory(which)
  per[i] = {index = which, exists = inv ~= nil,
    plate = inv and inv.can_insert({name = "iron-plate", count = 1}) or false,
    ore = inv and inv.can_insert({name = "iron-ore", count = 1}) or false,
    coal = inv and inv.can_insert({name = "coal", count = 1}) or false}
end
out.furnace_inventories = per
local d = s.create_entity({name = "burner-mining-drill", position = {ox + 8, oy},
  force = "player"})
local dper = {}
for i, which in ipairs({1, 2, 3}) do
  local inv = d.get_inventory(which)
  dper[i] = {index = which, exists = inv ~= nil,
    ore = inv and inv.can_insert({name = "iron-ore", count = 1}) or false,
    coal = inv and inv.can_insert({name = "coal", count = 1}) or false}
end
out.drill_inventories = dper
for _, e in pairs({f, g2, d}) do e.destroy() end
return out
"""

# ---------------------------------------------------------------- collision, sweep

COLLISION = """
local out = {boxes = {}}
for _, name in ipairs({"character", "stone-wall", "burner-mining-drill", "stone-furnace",
    "iron-ore", "item-on-ground"}) do
  local p = prototypes.entity[name]
  if p then
    local b = p.collision_box
    local sel = p.selection_box
    out.boxes[name] = {
      collision = {b.left_top.x, b.left_top.y, b.right_bottom.x, b.right_bottom.y},
      selection = {sel.left_top.x, sel.left_top.y, sel.right_bottom.x, sel.right_bottom.y},
      tile_width = try(function() return p.tile_width end),
      tile_height = try(function() return p.tile_height end),
      mining_time = try(function() return p.mineable_properties.mining_time end),
      radius = try(function() return p.mining_drill_radius end),
      vector_to_place_result = try(function() return p.vector_to_place_result end),
    }
  end
end
local ox, oy = OX + 80, OY
local sweep = {}
for _, spec in ipairs({{"stone-wall", 0.5}, {"stone-furnace", 0}}) do
  for _, d in ipairs({31.5, 31.9, 32, 32.1, 32.4, 32.5, 32.6, 33, 33.5}) do
    local e = s.create_entity({name = spec[1], position = {ox + d, oy + spec[2]},
      force = "player"})
    if e then
      local found = s.find_entities_filtered({position = {ox, oy + spec[2]}, radius = 32,
        name = spec[1]})
      local hit = false
      for _, x in pairs(found) do if x == e then hit = true end end
      sweep[#sweep + 1] = {spec[1], d, e.position.x - ox, hit}
      e.destroy()
    end
  end
end
out.sweep = sweep
-- The same boundary from origins between tile centres, where a centre rule and
-- a bounding-box rule disagree.
local fine = {}
for _, spec in ipairs({{"stone-wall", {ox + 32.5, oy + 40.5}}, {"stone-furnace",
    {ox + 32, oy + 44}}, {"item-on-ground", {ox + 32.3, oy + 48.3}}}) do
  local e
  if spec[1] == "item-on-ground" then
    e = s.create_entity({name = "item-on-ground", position = spec[2],
      stack = {name = "iron-ore", count = 1}})
  else
    e = s.create_entity({name = spec[1], position = spec[2], force = "player"})
  end
  for step = 0, 32 do
    local origin = {ox + step / 32, e.position.y}
    local found = s.find_entities_filtered({position = origin, radius = 32,
      name = spec[1]})
    local hit = false
    for _, x in pairs(found) do if x == e then hit = true end end
    local cd = e.position.x - origin[1]
    fine[#fine + 1] = {spec[1], g(cd), hit}
  end
  e.destroy()
end
out.sweep_fine = fine
return out
"""

SCRIPTED = """
local ch = storage.frrl_character
local out = {}
ch.walking_state = {walking = true, direction = defines.direction.east}
out.walking_same_tick = {ch.walking_state.walking, ch.walking_state.direction}
ch.walking_state = {walking = false}
out.stopped_same_tick = {ch.walking_state.walking, ch.walking_state.direction}
out.mining_before = ch.mining_state.mining
local ore = s.create_entity({name = "iron-ore", position = {ch.position.x + 1.5,
  ch.position.y + 0.5}, amount = 5})
ch.update_selected_entity(ore.position)
ch.mining_state = {mining = true, position = ore.position}
out.mining_same_tick = ch.mining_state.mining
storage.probe.ore = ore
return out
"""

SCRIPTED_AFTER = """
local ch = storage.frrl_character
local out = {mining_next_tick = ch.mining_state.mining,
  progress = g(ch.character_mining_progress),
  walking_next_tick = ch.walking_state.walking}
ch.mining_state = {mining = false}
out.mining_after_stop = ch.mining_state.mining
if storage.probe.ore and storage.probe.ore.valid then storage.probe.ore.destroy() end
return out
"""

NUMBERS = """
return helpers.table_to_json({
  a = 0.1, b = 1 / 3, c = 0.24999999999999994, d = 0.5 + 2 ^ -52, e = 1e-7,
  f = 123456.789, g = 2 ^ 53, h = 1e21, i = -0.0, j = 5.0, k = 3.0000000000000004,
  l = 0.1484375, m = -460 / 256, n = 1e300, o = 12345678901234567890,
})
"""

# ---------------------------------------------------------------- terrain

#: Water within this many tiles of the origin. Every scene sits inside 48 tiles
#: and the sensor reaches 32 beyond the character, so this covers any tile an
#: observation of these tasks can report.
TERRAIN_RADIUS = 128

TERRAIN = """
local out = {water = {}, area_rule = {}}
for _, t in pairs(s.find_tiles_filtered({area = {{-R, -R}, {R, R}},
    collision_mask = "water_tile"})) do
  out.water[#out.water + 1] = {t.position.x, t.position.y}
end
-- Which tile columns a fractional area returns: every tile, one row.
for _, bounds in ipairs({{0, 1}, {0.25, 1.25}, {0.5, 1.5}, {0.75, 1.75}, {0.99, 1.01},
    {-0.25, 0.75}, {-1.5, -0.5}, {2, 2}, {2.5, 2.5}}) do
  local xs = {}
  for _, t in pairs(s.find_tiles_filtered({area = {{bounds[1], 0.5}, {bounds[2], 0.5}}})) do
    xs[#xs + 1] = t.position.x
  end
  local ys = {}
  for _, t in pairs(s.find_tiles_filtered({area = {{0.5, bounds[1]}, {0.5, bounds[2]}}})) do
    ys[#ys + 1] = t.position.y
  end
  out.area_rule[#out.area_rule + 1] = {bounds = bounds, xs = xs, ys = ys}
end
out.seed = s.map_gen_settings.seed
return out
"""

# ---------------------------------------------------------------- insertion limit

LIMIT_SETUP = """
local ox, oy = OX + 100, OY
for dx = -1, 0 do for dy = -1, 0 do
  s.create_entity({name = "iron-ore", position = {ox + dx + 0.5, oy + dy + 0.5}, amount = 5000})
end end
local d = s.create_entity({name = "burner-mining-drill", position = {ox, oy},
  direction = defines.direction.south, force = "player"})
d.insert({name = "coal", count = 50})
local f = s.create_entity({name = "stone-furnace", position = {ox, oy + 2}, force = "player"})
storage.probe.ld, storage.probe.lf = d, f
return {drop = {d.drop_position.x, d.drop_position.y}, furnace = {f.position.x, f.position.y}}
"""

LIMIT_SAMPLE = """
local d, f = storage.probe.ld, storage.probe.lf
return {tick = game.tick, status = status(d),
  source = count(f.get_inventory(defines.inventory.furnace_source), "iron-ore"),
  progress = g(d.mining_progress)}
"""


def lua(client: RCONClient, code: str):
    return client.lua(PRELUDE + code.replace("OX", str(OX)).replace("OY", str(OY)))


def arrivals(samples: list[dict], key: str) -> list[int]:
    """Relative ticks at which a counter went up."""
    out = []
    for before, after in zip(samples, samples[1:], strict=False):
        if after[key] > before[key]:
            out.append(after["t"])
    return out


def transitions(samples: list[dict], key: str) -> list[list]:
    out, previous = [], None
    for sample in samples:
        value = sample[key]["status"]
        if value != previous:
            out.append([sample["t"], value])
            previous = value
    return out


def main() -> int:
    manager = WorkerManager()
    handle = manager.launch("sim-details")
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
                SeedPlan(master=31, run_id="sim-details"),
                branch=Branch.TRAIN,
                split="train",
            )
            env.reset(options={"scene_index": 0})

            report["terrain"] = lua(client, TERRAIN.replace("R", str(TERRAIN_RADIUS)))
            report["collision"] = lua(client, COLLISION)
            report["reach"] = lua(client, REACH)
            report["insertion"] = lua(client, INSERTION)
            report["numbers"] = lua(client, NUMBERS)
            scripted = lua(client, SCRIPTED)
            env.advance(1)
            scripted.update(lua(client, SCRIPTED_AFTER))
            report["scripted_state"] = scripted

            # Burner rigs, tick by tick through first output, then again across
            # the tick each runs out of fuel.
            report["burner_rigs"] = lua(client, BURNER_SETUP)
            fuelled = int(lua(client, BURNER_FUEL))
            kept, every = [], [lua(client, BURNER_SAMPLE)]
            for _ in range(BURNER_TICKS):
                env.advance(1)
                every.append(lua(client, BURNER_SAMPLE))
            for sample in every:
                sample["t"] = sample["tick"] - fuelled
                if any(lo <= sample["t"] <= hi for lo, hi in KEEP_WINDOWS):
                    kept.append(sample)
            report["burner"] = {
                "fuelled_tick": fuelled,
                "kept_windows": KEEP_WINDOWS,
                "samples": kept,
                "transitions": {key: transitions(every, key) for key in ("a", "b", "c", "d", "e")},
                "d_chest_arrivals": arrivals(every, "d_chest"),
                "e_result_arrivals": arrivals(every, "e_result"),
            }

            report["insertion_limit_rig"] = lua(client, LIMIT_SETUP)
            limit = []
            blocked = 0
            for _ in range(2000):
                env.advance(60)
                sample = lua(client, LIMIT_SAMPLE)
                limit.append(sample)
                blocked = blocked + 1 if sample["status"] != "working" else 0
                if blocked >= 5:
                    break
            report["insertion_limit"] = {
                "samples": limit[-8:],
                "max_source": max(s["source"] for s in limit),
                "statuses": sorted({s["status"] for s in limit}),
            }
            session.close()
    finally:
        manager.cleanup(handle)
    report["wall_seconds"] = round(time.perf_counter() - started, 1)
    OUT.write_text(json.dumps(report, indent=0, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    summary = {
        "reach": {
            k: v["rows"] for k, v in report["reach"].items() if isinstance(v, dict) and "rows" in v
        },
        "insertion": report["insertion"],
        "scripted_state": report["scripted_state"],
        "numbers": report["numbers"],
        "sweep": report["collision"]["sweep"],
        "water_tiles": len(report["terrain"]["water"]),
        "area_rule": report["terrain"]["area_rule"],
        "burner_transitions": report["burner"]["transitions"],
        "d_chest_arrivals": report["burner"]["d_chest_arrivals"][:4],
        "e_result_arrivals": report["burner"]["e_result_arrivals"][:4],
        "insertion_limit": {k: report["insertion_limit"][k] for k in ("max_source", "statuses")},
    }
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
