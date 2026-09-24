"""Measure belts and burner inserters in the engine, tick by tick (M4, logistics).

factory-sim models drills, furnaces, walls and item piles. The next task family
needs transport belts, burner inserters and chests, and the simulator only adds
a mechanic after the engine has been measured doing it. This probe records the
raw per-tick state those mechanics produce in Factorio 2.0.60 and writes
`docs/evidence/sim-mechanics-m4-logistics.json`:

- **belt, straight**: plates fed at the back of a ten-belt run on both lanes as
  fast as the back accepts them, positions every tick until they compress at
  the end. Speed, spacing, the stop position, and how a line spans belts.
- **belt, curve**: one plate per lane through a right turn, so the inner and
  outer lane lengths show up as arrival ticks.
- **belt, sideload**: a belt feeding into the side of another, and which lane
  the items join and where.
- **drill onto a belt**: a fuelled burner drill whose drop position is a single
  belt tile: which lane, where on it, and when the drill waits because the
  spot is taken.
- **drill, belt, inserter**: a drill dropping onto a two-belt run whose end an
  inserter empties into a chest. Same-tick ordering of output and pickup.
- **inserter, chest to chest**: a fuelled burner inserter moving plates one at
  a time, hand state and energy every tick. Swing time, pickup and drop ticks,
  energy per swing.
- **inserter from a belt end**: ore stopped at the end of a belt picked into a
  stone furnace.
- **inserter from a moving belt**: a steady flow past an inserter, on the near
  lane and on the far lane. Whether and when it catches items decides whether
  the simulator models this or the task avoids it.
- **inserter from a furnace onto a belt**: result-slot pickup.
- **inserter onto a belt**: chest, inserter, belt running east, west and south,
  so the lane a drop lands on is read against the belt's direction.
- **fill limits**: how much ore and coal an inserter puts into a furnace before
  it stops.
- **inserter fuel**: burner inserters run to exhaustion from one coal and from
  one wood.
- **sideload phases**: one item per rig reaching a sideload at eight sub-tile
  phases, from each feed lane, to pin down where a sideloaded item lands.
- **update order**: drill into a chest, inserter out of it, built in both
  orders, so a same-tick output and pickup shows which entity ran first.
- **drill and inserter on one belt tile**: a drill dropping onto the tile an
  inserter picks from.
- **self-refuel**: an inserter given no fuel moving coal between chests, past
  the end of the fuel it is built with.
- **fill limit, fuelled furnace**: ore into a furnace that is smelting.
- **unblocking a drill**: at `UNBLOCK` the full single-belt drill rig gets a
  second belt, and the trace shows when the drill outputs again.

Everything samples the same tick, so update order within a tick can be read off
the traces. All rigs are raw Lua far from the scene. Time advances through
`env.advance`.

Belt item positions are in 1/256 tile, measured from the downstream end of the
belt's own line. Inserter `hand` is `held_stack_position` as map coordinates
times 256. `flow_belt` and `flow2_belt` hold belts 6-11 of 16 only.

`samples` is delta-encoded to keep the file small: sample 0 is complete, every
later sample holds `t`, `tick` and only what changed since the previous sample,
recursing into objects; keys that vanished are listed under `_gone`. `replay`
in this file rebuilds every tick.

Run:
  uv run python tools/probe_logistics.py
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

OUT = ROOT / "docs" / "evidence" / "sim-mechanics-m4-logistics.json"

#: Rig origin, well outside every scene's 48-tile box.
OX, OY = 160, 160
#: Ticks sampled one by one; long enough for the one-coal inserter to run dry,
#: a drill's single output belt to fill, and a belt run to compress.
TICKS = 3000
#: Relative tick at which the blocked drill's belt is extended by one belt.
UNBLOCK = 1500

PRELUDE = """
local s = game.surfaces[1]
local function try(fn) local ok, v = pcall(fn) if ok then return v end return nil end
local function g(v) if v == nil then return nil end return string.format("%.17g", v) end
local names = {}
for name, value in pairs(defines.entity_status) do names[value] = name end
local function status(e) return names[try(function() return e.status end)] end
local function count(inv, item) return inv and inv.get_item_count(item) or 0 end
storage.probe = storage.probe or {}
local D = defines.direction
"""

SETUP = """
s.request_to_generate_chunks({OX, OY}, 3)
s.force_generate_chunk_requests()
local area = {{OX - 40, OY - 40}, {OX + 40, OY + 40}}
for _, e in pairs(s.find_entities_filtered({area = area})) do
  if e.valid and e.type ~= "character" then e.destroy() end
end
local tiles = {}
for x = OX - 40, OX + 40 do for y = OY - 40, OY + 40 do
  tiles[#tiles + 1] = {name = "grass-1", position = {x, y}}
end end
s.set_tiles(tiles)
local P = {}
local built = {}
local function mk(spec)
  spec.force = spec.force or "player"
  local e = s.create_entity(spec)
  built[#built + 1] = {spec.name, e ~= nil}
  return e
end
local function belt(x, y, dir)
  return mk({name = "transport-belt", position = {OX + x + 0.5, OY + y + 0.5}, direction = dir})
end
local function chest(x, y)
  return mk({name = "wooden-chest", position = {OX + x + 0.5, OY + y + 0.5}})
end
-- An inserter's direction points at its pickup tile.
local function inserter(x, y, dir)
  return mk({name = "burner-inserter", position = {OX + x + 0.5, OY + y + 0.5}, direction = dir})
end
local function ore(x, y)
  for dx = -1, 0 do for dy = -1, 0 do
    s.create_entity({name = "iron-ore", position = {OX + x + dx + 0.5, OY + y + dy + 0.5},
      amount = 5000})
  end end
end

-- 1. straight: ten east belts at row -30, fed at the back in SAMPLE.
P.straight = {}
for i = 0, 9 do P.straight[#P.straight + 1] = belt(-30 + i, -30, D.east) end

-- 2. curve: four belts east, then four south (a right turn); one plate per lane.
P.curve = {}
for i = 0, 3 do P.curve[#P.curve + 1] = belt(-30 + i, -24, D.east) end
for i = 0, 3 do P.curve[#P.curve + 1] = belt(-26, -24 + i, D.south) end

-- 3. sideload: a north-running belt ending on the south side of an east run.
P.side_main = {}
for i = 0, 5 do P.side_main[#P.side_main + 1] = belt(-30 + i, -14, D.east) end
P.side_feed = {}
for i = 0, 3 do P.side_feed[#P.side_feed + 1] = belt(-28, -10 - i, D.north) end

-- 4. drill onto a single belt tile: the drill faces south onto an east belt.
ore(-10, -30)
P.drill = mk({name = "burner-mining-drill", position = {OX - 10, OY - 30}, direction = D.south})
local dp = P.drill.drop_position
P.drill_belt = {mk({name = "transport-belt", position = {math.floor(dp.x) + 0.5,
  math.floor(dp.y) + 0.5}, direction = D.east})}

-- 5. inserter chest -> chest.
P.c2c_src = chest(0, -31)
P.c2c_ins = inserter(0, -30, D.north)
P.c2c_dst = chest(0, -29)
P.c2c_src.insert({name = "iron-plate", count = 50})

-- 6. belt end -> inserter -> furnace; ore fed in SAMPLE, stopped at the belt end.
P.bend_belt = {}
for i = 0, 3 do P.bend_belt[#P.bend_belt + 1] = belt(8 + i, -30, D.east) end
P.bend_ins = inserter(11, -29, D.north)
P.bend_furnace = mk({name = "stone-furnace", position = {OX + 12, OY - 27}})

-- 7. moving belt past an inserter: east run fed every 20 ticks on the near lane
-- (right lane, 2, the south side) and a second run on the far lane (1).
P.flow_belt = {}
for i = 0, 15 do P.flow_belt[#P.flow_belt + 1] = belt(-30 + i, 0, D.east) end
P.flow_ins = inserter(-22, 1, D.north)
P.flow_chest = chest(-22, 2)
P.flow2_belt = {}
for i = 0, 15 do P.flow2_belt[#P.flow2_belt + 1] = belt(-30 + i, 5, D.east) end
P.flow2_ins = inserter(-22, 6, D.north)
P.flow2_chest = chest(-22, 7)

-- 8. furnace -> inserter -> belt: output plates onto an east belt.
P.fout_furnace = mk({name = "stone-furnace", position = {OX + 20, OY}})
P.fout_furnace.get_inventory(defines.inventory.furnace_source).insert(
  {name = "iron-ore", count = 20})
P.fout_furnace.insert({name = "coal", count = 2})
P.fout_ins = inserter(19, 1, D.north)
P.fout_belt = {}
for i = 0, 3 do P.fout_belt[#P.fout_belt + 1] = belt(19 + i, 2, D.east) end

-- 9. fill limits: chest of ore -> inserter -> unfuelled furnace; coal likewise.
P.fill_ore_chest = chest(0, 10)
P.fill_ore_chest.insert({name = "iron-ore", count = 100})
P.fill_ore_ins = inserter(0, 11, D.north)
P.fill_ore_furnace = mk({name = "stone-furnace", position = {OX + 1, OY + 13}})
P.fill_coal_chest = chest(8, 10)
P.fill_coal_chest.insert({name = "coal", count = 50})
P.fill_coal_ins = inserter(8, 11, D.north)
P.fill_coal_furnace = mk({name = "stone-furnace", position = {OX + 9, OY + 13}})

-- 10. fuel: inserters with one coal / one wood moving plates until they stop.
P.fuel_src = chest(16, 10)
P.fuel_src.insert({name = "iron-plate", count = 200})
P.fuel_ins = inserter(16, 11, D.north)
P.fuel_dst = chest(16, 12)
P.wood_src = chest(22, 10)
P.wood_src.insert({name = "iron-plate", count = 200})
P.wood_ins = inserter(22, 11, D.north)
P.wood_dst = chest(22, 12)

-- 11. drill -> two-belt run -> inserter at the end -> chest.
ore(-10, 0)
P.tick_drill = mk({name = "burner-mining-drill", position = {OX - 10, OY}, direction = D.south})
local tp = P.tick_drill.drop_position
local tx, ty = math.floor(tp.x), math.floor(tp.y)
P.tick_belt = {
  mk({name = "transport-belt", position = {tx + 0.5, ty + 0.5}, direction = D.east}),
  mk({name = "transport-belt", position = {tx + 1.5, ty + 0.5}, direction = D.east}),
}
P.tick_ins = mk({name = "burner-inserter", position = {tx + 1.5, ty + 1.5}, direction = D.north,
  force = "player"})
P.tick_chest = mk({name = "wooden-chest", position = {tx + 1.5, ty + 2.5}, force = "player"})

-- 12. drop onto belts: chest -> inserter (drops south) -> belt running E / W / S.
P.drop = {}
for i, dir in ipairs({D.east, D.west, D.south}) do
  local x = -30 + 6 * (i - 1)
  local c = chest(x, 20)
  c.insert({name = "iron-plate", count = 20})
  local r = {chest = c, ins = inserter(x, 21, D.north), belt = {}}
  if dir == D.south then
    for k = 0, 2 do r.belt[#r.belt + 1] = belt(x, 22 + k, dir) end
  elseif dir == D.east then
    for k = 0, 2 do r.belt[#r.belt + 1] = belt(x + k, 22, dir) end
  else
    for k = 0, 2 do r.belt[#r.belt + 1] = belt(x - k, 22, dir) end
  end
  P.drop[i] = r
end

-- 13. sideload phases: one item per rig, placed on a one-tile north feed at
-- 128 + k (1/256 tiles) from its front, k = 0..7, on lane 1 (row 30) or lane 2
-- (row 35); the feed ends on the south side of a two-belt east run.
P.phase = {}
for lane = 1, 2 do
  for k = 0, 7 do
    local x, y = -30 + 4 * k, 25 + 5 * lane
    local r = {lane = lane, k = k, main = {belt(x - 1, y, D.east), belt(x, y, D.east),
      belt(x + 1, y, D.east)}, feed = {belt(x, y + 1, D.north)}}
    r.feed[1].get_transport_line(lane).insert_at(0.5 + k / 256, {name = "copper-plate", count = 1})
    P.phase[#P.phase + 1] = r
  end
end

-- 14. update order: drill -> chest -> inserter -> chest, built in that order
-- (A) and in the reverse order (B).
local function order_rig(cx, cy, reverse)
  local r = {}
  local function drill()
    ore(cx, cy)
    r.drill = mk({name = "burner-mining-drill", position = {OX + cx, OY + cy}, direction = D.south})
  end
  local function rest()
    local list = {
      function() r.src = chest(cx, cy + 1) end,
      function() r.ins = inserter(cx, cy + 2, D.north) end,
      function() r.dst = chest(cx, cy + 3) end,
    }
    if reverse then for i = 3, 1, -1 do list[i]() end else for i = 1, 3 do list[i]() end end
  end
  if reverse then rest() drill() else drill() rest() end
  return r
end
P.order_a = order_rig(12, 31, false)
P.order_b = order_rig(18, 31, true)

-- 15. drill -> one belt tile <- inserter picking from that same tile -> chest.
ore(24, 31)
P.same_drill = mk({name = "burner-mining-drill", position = {OX + 24, OY + 31},
  direction = D.south})
P.same_belt = {belt(24, 32, D.east)}
P.same_ins = inserter(24, 33, D.north)
P.same_chest = chest(24, 34)

-- 16. self-refuel: an inserter given no fuel moving coal chest -> chest.
P.self_src = chest(30, 30)
P.self_src.insert({name = "coal", count = 50})
P.self_ins = inserter(30, 31, D.north)
P.self_dst = chest(30, 32)

-- 17. fill limit into a fuelled, working furnace.
P.hot_chest = chest(34, 30)
P.hot_chest.insert({name = "iron-ore", count = 50})
P.hot_ins = inserter(34, 31, D.north)
P.hot_furnace = mk({name = "stone-furnace", position = {OX + 35, OY + 33}})
P.hot_furnace.insert({name = "coal", count = 5})

storage.probe.P = P
storage.probe.flow_next = 0
storage.probe.fed = {straight = {0, 0}, bend = {0, 0}, side = {0, 0}, curve = {0, 0}}

local function proto(name)
  local p = prototypes.entity[name]
  local bp = try(function() return p.burner_prototype end)
  return {
    rotation_speed = try(function() return p.get_inserter_rotation_speed() end)
      or try(function() return p.inserter_rotation_speed end),
    extension_speed = try(function() return p.get_inserter_extension_speed() end)
      or try(function() return p.inserter_extension_speed end),
    chases_belt_items = try(function() return p.inserter_chases_belt_items end),
    belt_speed = try(function() return p.belt_speed end),
    max_energy_usage = try(function() return p.get_max_energy_usage() end)
      or try(function() return p.max_energy_usage end),
    energy_per_move = try(function() return p.energy_per_move end),
    fuel_slots = bp and try(function() return bp.fuel_inventory_size end),
    effectivity = bp and try(function() return bp.effectivity end),
    pickup = try(function() return p.inserter_pickup_position end),
    drop = try(function() return p.inserter_drop_position end),
    box = try(function() return p.collision_box end),
  }
end
local function xy(p) return {g(p.x), g(p.y)} end
local function insinfo(e)
  return {at = xy(e.position), pickup = xy(e.pickup_position), drop = xy(e.drop_position),
    direction = e.direction,
    pickup_target = e.pickup_target and e.pickup_target.name or nil,
    drop_target = e.drop_target and e.drop_target.name or nil}
end
local function lineinfo(belts)
  local out = {}
  for i, b in ipairs(belts) do
    local row = {at = xy(b.position), shape = try(function() return b.belt_shape end)}
    for lane = 1, 2 do
      local l = b.get_transport_line(lane)
      row[lane] = {length = g(l.line_length), segment = g(try(function()
        return l.total_segment_length end)),
        start = xy(l.get_line_item_position(0)),
        stop = xy(l.get_line_item_position(l.line_length)),
        same_as_first = l.line_equals(belts[1].get_transport_line(lane))}
    end
    out[i] = row
  end
  return out
end
local ins = {}
for _, key in pairs({"c2c_ins", "bend_ins", "flow_ins", "flow2_ins", "fout_ins", "fill_ore_ins",
    "fill_coal_ins", "fuel_ins", "wood_ins", "tick_ins"}) do
  ins[key] = insinfo(P[key])
end
for i, r in ipairs(P.drop) do ins["drop" .. i] = insinfo(r.ins) end
ins.order_a, ins.order_b = insinfo(P.order_a.ins), insinfo(P.order_b.ins)
ins.same_ins = insinfo(P.same_ins)
ins.self_ins, ins.hot_ins = insinfo(P.self_ins), insinfo(P.hot_ins)
local phase_lines = {}
for i, r in ipairs(P.phase) do
  phase_lines[i] = {lane = r.lane, k = r.k, main = lineinfo(r.main), feed = lineinfo(r.feed)}
end
local fuel_at_build = {}
for _, key in pairs({"c2c_ins", "fuel_ins", "wood_ins", "same_ins"}) do
  fuel_at_build[key] = try(function() return P[key].get_fuel_inventory().get_contents() end)
end
local burning_at_build = {}
local c = P.c2c_ins.burner
burning_at_build.c2c = {remaining = g(c.remaining_burning_fuel),
  item = try(function() return c.currently_burning.name.name end)}
return {
  built = built,
  prototypes = {["transport-belt"] = proto("transport-belt"),
    ["burner-inserter"] = proto("burner-inserter"),
    ["burner-mining-drill"] = proto("burner-mining-drill")},
  lines = {straight = lineinfo(P.straight), curve = lineinfo(P.curve),
    side_main = lineinfo(P.side_main), side_feed = lineinfo(P.side_feed),
    drill_belt = lineinfo(P.drill_belt), tick_belt = lineinfo(P.tick_belt),
    fout_belt = lineinfo(P.fout_belt), drop1 = lineinfo(P.drop[1].belt),
    drop2 = lineinfo(P.drop[2].belt), drop3 = lineinfo(P.drop[3].belt)},
  phase_lines = phase_lines,
  drill_drop = xy(dp),
  tick_drill_drop = xy(tp),
  order_drill_drop = {xy(P.order_a.drill.drop_position), xy(P.order_b.drill.drop_position)},
  same_drill_drop = xy(P.same_drill.drop_position),
  inserters = ins,
  fuel_at_build = fuel_at_build,
  burning_at_build = burning_at_build,
  energy_at_build = {c2c = g(P.c2c_ins.energy), fuel = g(P.fuel_ins.energy)},
}
"""

FUEL = """
local P = storage.probe.P
P.drill.insert({name = "coal", count = 5})
P.tick_drill.insert({name = "coal", count = 5})
P.same_drill.insert({name = "coal", count = 5})
for _, key in pairs({"c2c_ins", "bend_ins", "flow_ins", "flow2_ins", "fout_ins", "fill_ore_ins",
    "fill_coal_ins", "tick_ins", "same_ins", "hot_ins"}) do
  P[key].insert({name = "coal", count = 5})
end
for _, r in ipairs(P.drop) do r.ins.insert({name = "coal", count = 5}) end
for _, r in ipairs({P.order_a, P.order_b}) do
  r.drill.insert({name = "coal", count = 5})
  r.ins.insert({name = "coal", count = 5})
end
P.fuel_ins.insert({name = "coal", count = 1})
P.wood_ins.insert({name = "wood", count = 1})
storage.probe.t0 = game.tick
return game.tick
"""

# Sampled every tick. Feeding happens first, so an item inserted by script shows
# at the tick it went in. Belt lanes as (item, position, id) per belt entity;
# inserters as hand item, hand position, energy, burner fuel; containers as
# counts.
SAMPLE = """
local P = storage.probe.P
local fed = storage.probe.fed
local function feed(key, line, lane, limit, item)
  if fed[key][lane] < limit and line.can_insert_at_back() then
    if line.insert_at_back({name = item, count = 1}) then fed[key][lane] = fed[key][lane] + 1 end
  end
end
for lane = 1, 2 do
  feed("straight", P.straight[1].get_transport_line(lane), lane, 6, "iron-plate")
  feed("bend", P.bend_belt[1].get_transport_line(lane), lane, 4, "iron-ore")
  feed("side", P.side_feed[1].get_transport_line(lane), lane, 3, "copper-plate")
  feed("curve", P.curve[1].get_transport_line(lane), lane, 1, "iron-plate")
end
if game.tick >= storage.probe.flow_next then
  local near = P.flow_belt[1].get_transport_line(2)
  if near.can_insert_at_back() then near.insert_at_back({name = "iron-plate", count = 1}) end
  local far = P.flow2_belt[1].get_transport_line(1)
  if far.can_insert_at_back() then far.insert_at_back({name = "iron-plate", count = 1}) end
  storage.probe.flow_next = game.tick + 20
end
-- At t = UNBLOCK the blocked drill's single belt gets a second belt downstream.
if game.tick - storage.probe.t0 == UNBLOCK and not P.drill_belt[2] then
  local b = P.drill_belt[1]
  P.drill_belt[2] = s.create_entity({name = "transport-belt", position = {b.position.x + 1,
    b.position.y}, direction = D.east, force = "player"})
end
-- Positions in 1/256 tile when exact (they always have been), else a string.
local function p256(v)
  local q = v * 256
  if q == math.floor(q) then return math.floor(q) end
  return g(v)
end
local function lanes(belts, first, last)
  local out = {}
  for i = first or 1, last or #belts do
    local b = belts[i]
    local row = {}
    for lane = 1, 2 do
      local items = {}
      for _, it in pairs(b.get_transport_line(lane).get_detailed_contents()) do
        items[#items + 1] = {it.stack.name, p256(it.position), it.unique_id}
      end
      row[lane] = items
    end
    out[#out + 1] = row
  end
  return out
end
local function ins(e)
  local h = e.held_stack
  local hp = try(function() return e.held_stack_position end)
  local b = e.burner
  local cur = b and try(function() return b.currently_burning end)
  local cur_name = cur and (try(function() return cur.name.name end)
    or try(function() return cur.name end))
  local inv = e.get_fuel_inventory()
  return {
    held = h and h.valid_for_read and h.name or nil,
    hand = hp and {p256(hp.x), p256(hp.y)},
    energy = g(e.energy),
    remaining = b and g(b.remaining_burning_fuel),
    burning = type(cur_name) == "string" and cur_name or nil,
    coal = count(inv, "coal"), wood = count(inv, "wood"),
    status = status(e),
  }
end
local function fur(f)
  return {src = count(f.get_inventory(defines.inventory.furnace_source), "iron-ore"),
    fuel = count(f.get_fuel_inventory(), "coal"),
    res = count(f.get_inventory(defines.inventory.furnace_result), "iron-plate"),
    status = status(f), progress = g(f.crafting_progress)}
end
local function box(c, item) return count(c.get_inventory(defines.inventory.chest), item) end
local function drill(d)
  return {status = status(d), progress = g(d.mining_progress), energy = g(d.energy)}
end
local out = {}
for i, r in ipairs(P.drop) do out["drop" .. i] = {ins = ins(r.ins), belt = lanes(r.belt)} end
for i, r in ipairs(P.phase) do out["phase" .. i] = {main = lanes(r.main), feed = lanes(r.feed)} end
for _, key in pairs({"order_a", "order_b"}) do
  local r = P[key]
  out[key] = {drill = drill(r.drill), src = box(r.src, "iron-ore"), ins = ins(r.ins),
    dst = box(r.dst, "iron-ore")}
end
out.self = {ins = ins(P.self_ins), src = box(P.self_src, "coal"), dst = box(P.self_dst, "coal")}
out.hot = {ins = ins(P.hot_ins), furnace = fur(P.hot_furnace)}
out.same = {drill = drill(P.same_drill), belt = lanes(P.same_belt), ins = ins(P.same_ins),
  chest = box(P.same_chest, "iron-ore")}
local base = {
  tick = game.tick,
  straight = lanes(P.straight), curve = lanes(P.curve),
  side_main = lanes(P.side_main), side_feed = lanes(P.side_feed),
  drill = drill(P.drill), drill_belt = lanes(P.drill_belt),
  c2c = ins(P.c2c_ins), c2c_dst = box(P.c2c_dst, "iron-plate"),
  bend_belt = lanes(P.bend_belt), bend = ins(P.bend_ins), bend_furnace = fur(P.bend_furnace),
  flow_belt = lanes(P.flow_belt, 6, 11), flow = ins(P.flow_ins),
  flow_chest = box(P.flow_chest, "iron-plate"),
  flow2_belt = lanes(P.flow2_belt, 6, 11), flow2 = ins(P.flow2_ins),
  flow2_chest = box(P.flow2_chest, "iron-plate"),
  fout = ins(P.fout_ins), fout_furnace = fur(P.fout_furnace), fout_belt = lanes(P.fout_belt),
  fill_ore = ins(P.fill_ore_ins), fill_ore_furnace = fur(P.fill_ore_furnace),
  fill_coal = ins(P.fill_coal_ins), fill_coal_furnace = fur(P.fill_coal_furnace),
  fuel = ins(P.fuel_ins), fuel_dst = box(P.fuel_dst, "iron-plate"),
  wood = ins(P.wood_ins), wood_dst = box(P.wood_dst, "iron-plate"),
  tick_drill = drill(P.tick_drill), tick_belt = lanes(P.tick_belt), tick_ins = ins(P.tick_ins),
  tick_chest = box(P.tick_chest, "iron-ore"),
}
for k, v in pairs(base) do out[k] = v end
return out
"""


def lua(client: RCONClient, code: str):
    code = code.replace("OX", str(OX)).replace("OY", str(OY)).replace("UNBLOCK", str(UNBLOCK))
    return client.lua(PRELUDE + code)


def changes(samples: list[dict], pick) -> list[list]:
    """(relative tick, value) whenever `pick(sample)` changes."""
    out, previous = [], object()
    for sample in samples:
        value = pick(sample)
        if value != previous:
            out.append([sample["t"], value])
            previous = value
    return out


def diff(before, after):
    """What turns `before` into `after`: changed keys, recursing into dicts.

    Keys that disappeared (a Lua nil) are listed under `_gone`. Anything that is
    not a dict on both sides is replaced whole.
    """
    out = {}
    for key, value in after.items():
        old = before.get(key)
        if isinstance(value, dict) and isinstance(old, dict):
            inner = diff(old, value)
            if inner:
                out[key] = inner
        elif key not in before or old != value:
            out[key] = value
    gone = sorted(k for k in before if k not in after)
    if gone:
        out["_gone"] = gone
    return out


def undiff(before, change):
    """Apply one `diff` result; the inverse used to replay `samples`."""
    out = dict(before)
    for key in change.get("_gone", []):
        out.pop(key, None)
    for key, value in change.items():
        if key == "_gone":
            continue
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = undiff(out[key], value)
        else:
            out[key] = value
    return out


def delta(samples: list[dict]) -> list[dict]:
    """Sample 0 whole, then per sample only what changed (see `diff`)."""
    out, previous = [], {}
    for sample in samples:
        row = diff(previous, sample)
        row["t"], row["tick"] = sample["t"], sample["tick"]
        row.pop("_gone", None)
        out.append(row)
        previous = sample
    return out


def replay(rows: list[dict]) -> list[dict]:
    """Rebuild every full sample from `delta` output."""
    out, current = [], {}
    for row in rows:
        current = undiff(current, row)
        out.append(current)
    return out


def main() -> int:
    manager = WorkerManager()
    handle = manager.launch("logistics-probe")
    report: dict = {"engine": manager.engine.to_dict(), "origin": [OX, OY], "ticks": TICKS}
    started = time.perf_counter()
    try:
        with RCONClient(handle.spec.rcon_endpoint, timeout=60.0) as client:
            client.lua(f"game.speed = {resolve_game_speed()} return game.speed")
            session = WorkerSession(handle, timeout=60.0)
            session.status()
            env = FactorioEnv(
                get("construct_smelting_line"),
                session,
                SeedPlan(master=41, run_id="logistics-probe"),
                branch=Branch.TRAIN,
                split="train",
            )
            env.reset(options={"scene_index": 0})
            report["setup"] = lua(client, SETUP)
            fuelled = int(lua(client, FUEL))
            samples = [lua(client, SAMPLE)]
            for _ in range(TICKS):
                env.advance(1)
                samples.append(lua(client, SAMPLE))
            for sample in samples:
                sample["t"] = sample["tick"] - fuelled
            report["fuelled_tick"] = fuelled
            session.close()
    finally:
        manager.cleanup(handle)
    report["wall_seconds"] = round(time.perf_counter() - started, 1)
    report["samples_encoding"] = "delta: sample 0 whole, later samples only changed keys"
    report["samples"] = delta(samples)
    assert replay(report["samples"]) == samples
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KiB)")

    s = samples
    summary = {
        "built_failures": [b for b in report["setup"]["built"] if not b[1]],
        "c2c_hand": changes(s, lambda x: x["c2c"].get("held"))[:12],
        "c2c_delivered": changes(s, lambda x: x["c2c_dst"])[:8],
        "drill": changes(s, lambda x: x["drill"]["status"])[:6],
        "flow_caught": changes(s, lambda x: x["flow_chest"])[:8],
        "flow2_caught": changes(s, lambda x: x["flow2_chest"])[:8],
        "fout_hand": changes(s, lambda x: x["fout"].get("held"))[:8],
        "bend_furnace_src": changes(s, lambda x: x["bend_furnace"]["src"])[:8],
        "fill_ore_src": changes(s, lambda x: x["fill_ore_furnace"]["src"])[-3:],
        "fill_coal_fuel": changes(s, lambda x: x["fill_coal_furnace"]["fuel"])[-3:],
        "fuel_ins_status": changes(s, lambda x: x["fuel"]["status"])[:6],
        "wood_ins_status": changes(s, lambda x: x["wood"]["status"])[:6],
        "fuel_delivered_final": s[-1]["fuel_dst"],
        "wood_delivered_final": s[-1]["wood_dst"],
        "tick_chest": changes(s, lambda x: x["tick_chest"])[:6],
        "order_a_dst": changes(s, lambda x: x["order_a"]["dst"])[:4],
        "order_b_dst": changes(s, lambda x: x["order_b"]["dst"])[:4],
        "same_chest": changes(s, lambda x: x["same"]["chest"])[:4],
        "self_fuel": changes(
            s, lambda x: (x["self"]["ins"].get("burning"), x["self"]["ins"]["coal"])
        )[:6],
        "hot_src_max": max(x["hot"]["furnace"]["src"] for x in s),
        "drop_statuses": [
            changes(s, lambda x, i=i: x[f"drop{i}"]["ins"]["status"])[:4] for i in (1, 2, 3)
        ],
    }
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
