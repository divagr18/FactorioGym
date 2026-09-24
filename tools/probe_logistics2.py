"""Measure what the first logistics probe left open, tick by tick (M4, logistics 2).

`tools/probe_logistics.py` measured belts, burner inserters and chests well
enough to build them in factory-sim, and left a list of behaviours unmeasured.
This probe builds a rig for each and writes
`docs/evidence/sim-mechanics-m4-logistics2.json.xz`:

- **turns** (`turn_*`): left turns in two orientations and a right turn in a
  third, one item per lane, so lane lengths show as arrival ticks; the lines'
  own `line_length` and end positions are in `setup.lines`.
- **sideloads from the target's left** (`sl_left_<lane>_<k>`): one item per rig
  on a one-belt feed running south into the north side of an east run, at
  eight sub-tile phases, from each feed lane.
- **sideloads onto a turn, a belt end, a line start, head on, both sides**
  (`side_*`): which shape the belts take and where the items go.
- **simultaneous sideload arrivals** (`sim_*`): both feed lanes reaching the
  target on one tick, varying which lane got its item first, the feed length,
  which belts were built first, whether the target already carries items, and
  the side it comes from.
- **inserter drops onto turns** (`tdrop_*`) and **drill outputs onto turns**
  (`tdrill_*`), from each free side of a right and a left turn.
- **inserter pickups from turns and straight belt ends** (`tpick_*`,
  `spick_*`): items of distinct kinds stopped on both lanes, so each pickup
  names the lane and position it took.
- **ground** (`ground_*`): an inserter dropping onto empty ground and onto an
  item already there, and picking up from item piles at several offsets.
- **full chests** (`full_*`): a chest with no room, a chest with room for one,
  then room made by script at `RELEASE`.
- **fill limits** (`fill_*`): coal and wood into drills, inserters and a
  furnace; ore into a drill; copper ore, stone and plates into a furnace.
- **mixed chests** (`mix_*`): which item an inserter takes from a chest of
  several, into a chest, a furnace, a belt, a drill, and its own fuel slot.
- **update order** (`order_*`): several inserters reaching one chest or one
  slot on the same tick, built in different orders.
- **rotations and shape changes of loaded belts** (`rot_*`): a loaded turn
  rotated to straight, a loaded straight rotated into a turn, a turn's feeder
  removed or a feeder added behind it, read in the same script call before and
  after (`setup.rotations`) and then every tick.
- **energy running out** (`pe_*`): chest-to-chest inserters whose burning fuel
  is set so it runs out at 24 points of the cycle; refuelled at `REFUEL`.
- **self-refuel redirect** (`sr_*`): inserters moving coal whose burning fuel
  runs out at 24 points of the cycle, with one coal in the slot to load.
- **drill status after a change** (`dstat_*`): a blocked drill whose belt is
  extended, whose belt item is removed, whose chest gets room, whose ground pile
  is removed, near which an unrelated chest is built, whose belt is rotated;
  all at `UNBLOCK`, read in the same call.
- **the character**: placement checks under it (`setup.placement`), standing
  and walking on belts, straight, turning and at a belt end, then mining a
  chest, a loaded belt and two inserters (`character`, per tick).
- **prototypes** (`setup.prototypes`): collision boxes, mining times, inserter
  energy per movement and rotation.

All rigs are raw Lua far from the scene. Time advances through `env.advance`.
Belt item positions are in 1/256 tile from the downstream end of the belt's own
line, as `[name, position, unique_id]`; inserter `hp` is `held_stack_position`
as map coordinates times 256; ground items are `[name, count, x, y]` in 1/256.

`series` holds, per rig, `[t, record]` on every tick the rig's record changed
(the first sample is complete). A rig whose record is not listed at a tick is
unchanged. `character` the same.

Run:
  uv run python tools/probe_logistics2.py
"""

from __future__ import annotations

import json
import lzma
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

OUT = ROOT / "docs" / "evidence" / "sim-mechanics-m4-logistics2.json.xz"

#: Rig origin, well outside every scene's 48-tile box.
OX, OY = 200, 200
TICKS = 1500
#: Drill-status events and belt extensions.
UNBLOCK = 1100
#: Room made in the full chests.
RELEASE = 700
#: Coal given to the energy rigs once they stopped.
REFUEL = 1000
#: Rotation rigs stop being sampled after this many ticks.
ROT_TICKS = 40

PRELUDE = """
local s = game.surfaces[1]
local function try(fn) local ok, v = pcall(fn) if ok then return v end return nil end
local function g(v) if v == nil then return nil end return string.format("%.17g", v) end
local names = {}
for name, value in pairs(defines.entity_status) do names[value] = name end
local function status(e) return names[try(function() return e.status end)] end
local function p256(v)
  local q = v * 256
  if q == math.floor(q) then return math.floor(q) end
  return g(v)
end
storage.probe = storage.probe or {}
local D = defines.direction
local function line_items(line)
  local out = {}
  for _, it in pairs(line.get_detailed_contents()) do
    out[#out + 1] = {it.stack.name, p256(it.position), it.unique_id}
  end
  return out
end
local function belt_rec(b)
  if not b.valid then return "gone" end
  return {sh = b.belt_shape, d = b.direction, l = {line_items(b.get_transport_line(1)),
    line_items(b.get_transport_line(2))}}
end
local function inv_list(inv)
  local out = {}
  if not inv then return out end
  for i = 1, #inv do
    local st = inv[i]
    if st.valid_for_read then out[#out + 1] = {i, st.name, st.count} end
  end
  return out
end
local function ins_rec(e)
  if not e.valid then return "gone" end
  local h = e.held_stack
  local hp = try(function() return e.held_stack_position end)
  local b = e.burner
  local cur = b and try(function() return b.currently_burning end)
  local cur_name = cur and (try(function() return cur.name.name end)
    or try(function() return cur.name end))
  return {
    h = h and h.valid_for_read and h.name or nil,
    hc = h and h.valid_for_read and h.count or nil,
    hp = hp and {p256(hp.x), p256(hp.y)},
    e = g(e.energy),
    r = b and g(b.remaining_burning_fuel),
    bu = type(cur_name) == "string" and cur_name or nil,
    f = inv_list(e.get_fuel_inventory()),
    st = status(e),
  }
end
local function chest_rec(c)
  if not c.valid then return "gone" end
  return inv_list(c.get_inventory(defines.inventory.chest))
end
local function drill_rec(d)
  if not d.valid then return "gone" end
  return {st = status(d), pr = g(d.mining_progress), e = g(d.energy),
    f = inv_list(d.get_fuel_inventory())}
end
local function furnace_rec(f)
  if not f.valid then return "gone" end
  return {src = inv_list(f.get_inventory(defines.inventory.furnace_source)),
    f = inv_list(f.get_fuel_inventory()),
    res = inv_list(f.get_inventory(defines.inventory.furnace_result)),
    st = status(f), pr = g(f.crafting_progress)}
end
local function ground_rec(area)
  local out = {}
  for _, e in pairs(s.find_entities_filtered({area = area, type = "item-entity"})) do
    out[#out + 1] = {e.stack.name, e.stack.count, p256(e.position.x), p256(e.position.y)}
  end
  table.sort(out, function(a, b)
    if a[4] ~= b[4] then return tostring(a[4]) < tostring(b[4]) end
    if a[3] ~= b[3] then return tostring(a[3]) < tostring(b[3]) end
    return a[1] < b[1]
  end)
  return out
end
local function rig_rec(r)
  local out = {}
  if #r.belts > 0 then out.b = {} for i, b in ipairs(r.belts) do out.b[i] = belt_rec(b) end end
  if #r.ins > 0 then out.i = {} for i, e in ipairs(r.ins) do out.i[i] = ins_rec(e) end end
  if #r.chests > 0 then out.c = {} for i, c in ipairs(r.chests) do out.c[i] = chest_rec(c) end end
  if #r.drills > 0 then out.d = {} for i, d in ipairs(r.drills) do out.d[i] = drill_rec(d) end end
  if #r.furnaces > 0 then
    out.f = {} for i, f in ipairs(r.furnaces) do out.f[i] = furnace_rec(f) end
  end
  if r.ground then out.g = ground_rec(r.area) end
  return out
end
local function char_rec()
  local ch = storage.frrl_character
  if not (ch and ch.valid) then return "gone" end
  local ms = ch.mining_state
  local ws = ch.walking_state
  return {p = {p256(ch.position.x), p256(ch.position.y)}, w = ws.walking, wd = ws.direction,
    m = ms.mining, mp = g(ch.character_mining_progress),
    sel = ch.selected and ch.selected.name or nil,
    inv = inv_list(ch.get_inventory(defines.inventory.character_main))}
end
"""

SETUP = """
s.request_to_generate_chunks({OX, OY}, 4)
s.force_generate_chunk_requests()
local area = {{OX - 100, OY - 100}, {OX + 100, OY + 100}}
for _, e in pairs(s.find_entities_filtered({area = area})) do
  if e.valid and e.type ~= "character" then e.destroy() end
end
local tiles = {}
for x = OX - 100, OX + 100 do for y = OY - 100, OY + 100 do
  tiles[#tiles + 1] = {name = "grass-1", position = {x, y}}
end end
s.set_tiles(tiles)

local R, order = {}, {}
local built = {}
local cur = {x = -95, y = -95, h = 0}
local function slot(w, h)
  if cur.x + w > 95 then cur.x = -95 cur.y = cur.y + cur.h + 3 cur.h = 0 end
  local bx, by = cur.x, cur.y
  cur.x = cur.x + w + 3
  if h > cur.h then cur.h = h end
  return bx, by
end
local function rig(name, w, h, opts)
  local bx, by = slot(w, h)
  local r = {name = name, bx = bx, by = by, belts = {}, ins = {}, chests = {}, drills = {},
    furnaces = {}, feed = {}, until_t = opts and opts.until_t,
    ground = opts and opts.ground,
    area = {{OX + bx - 1, OY + by - 1}, {OX + bx + w + 1, OY + by + h + 1}}}
  R[name] = r
  order[#order + 1] = name
  return r
end
local function mk(spec)
  spec.force = spec.force or "player"
  local e = s.create_entity(spec)
  built[#built + 1] = {spec.name, e ~= nil}
  return e
end
local function at(r, x, y) return {OX + r.bx + x + 0.5, OY + r.by + y + 0.5} end
local function belt(r, x, y, dir)
  local b = mk({name = "transport-belt", position = at(r, x, y), direction = dir})
  r.belts[#r.belts + 1] = b
  return b
end
local function chest(r, x, y, items)
  local c = mk({name = "wooden-chest", position = at(r, x, y)})
  r.chests[#r.chests + 1] = c
  if items then
    local inv = c.get_inventory(defines.inventory.chest)
    for _, it in ipairs(items) do
      if it.slot then inv[it.slot].set_stack({name = it[1], count = it[2]})
      else c.insert({name = it[1], count = it[2]}) end
    end
  end
  return c
end
local function inserter(r, x, y, dir, coal)
  local e = mk({name = "burner-inserter", position = at(r, x, y), direction = dir})
  r.ins[#r.ins + 1] = e
  if coal and coal > 0 then e.insert({name = "coal", count = coal}) end
  return e
end
local function ore(r, cx, cy)
  -- the four tiles under a 2x2 machine centred on rig corner (cx, cy)
  for dx = -1, 0 do for dy = -1, 0 do
    s.create_entity({name = "iron-ore", position = {OX + r.bx + cx + dx + 0.5,
      OY + r.by + cy + dy + 0.5}, amount = 5000})
  end end
end
local function drill(r, cx, cy, dir, coal, on_ore)
  if on_ore ~= false then ore(r, cx, cy) end
  local d = mk({name = "burner-mining-drill", position = {OX + r.bx + cx, OY + r.by + cy},
    direction = dir})
  r.drills[#r.drills + 1] = d
  if coal and coal > 0 then d.insert({name = "coal", count = coal}) end
  return d
end
local function furnace(r, cx, cy)
  local f = mk({name = "stone-furnace", position = {OX + r.bx + cx, OY + r.by + cy}})
  r.furnaces[#r.furnaces + 1] = f
  return f
end
local function put(b, lane, pos, name)
  return b.get_transport_line(lane).insert_at(pos / 256, {name = name, count = 1})
end
local function feeder(r, b, lane, count, name)
  r.feed[#r.feed + 1] = {belt = b, lane = lane, left = count, item = name}
end
local events = {}
local function event(t, spec) spec.t = t events[#events + 1] = spec end
local function xy(p) return {g(p.x), g(p.y)} end
local function lineinfo(belts)
  local out = {}
  for i, b in ipairs(belts) do
    local row = {at = xy(b.position), shape = try(function() return b.belt_shape end),
      d = b.direction}
    for lane = 1, 2 do
      local l = b.get_transport_line(lane)
      row[lane] = {length = g(l.line_length),
        start = xy(l.get_line_item_position(0)),
        mid = xy(l.get_line_item_position(l.line_length / 2)),
        stop = xy(l.get_line_item_position(l.line_length))}
    end
    out[i] = row
  end
  return out
end
local lines = {}

-- ------------------------------------------------------------ turns
do
  local r = rig("turn_en_left", 5, 4)
  for i = 0, 3 do belt(r, i, 3, D.east) end
  belt(r, 4, 3, D.north)
  for i = 2, 0, -1 do belt(r, 4, i, D.north) end
  feeder(r, r.belts[1], 1, 1, "iron-plate") feeder(r, r.belts[1], 2, 1, "iron-plate")
  r = rig("turn_ws_left", 5, 4)
  for i = 4, 1, -1 do belt(r, i, 0, D.west) end
  belt(r, 0, 0, D.south)
  for i = 1, 3 do belt(r, 0, i, D.south) end
  feeder(r, r.belts[1], 1, 1, "iron-plate") feeder(r, r.belts[1], 2, 1, "iron-plate")
  r = rig("turn_ne_right", 4, 5)
  for i = 4, 1, -1 do belt(r, 0, i, D.north) end
  belt(r, 0, 0, D.east)
  for i = 1, 3 do belt(r, i, 0, D.east) end
  feeder(r, r.belts[1], 1, 1, "iron-plate") feeder(r, r.belts[1], 2, 1, "iron-plate")
  r = rig("turn_es_right", 5, 4)
  for i = 0, 3 do belt(r, i, 0, D.east) end
  belt(r, 4, 0, D.south)
  for i = 1, 3 do belt(r, 4, i, D.south) end
  feeder(r, r.belts[1], 1, 1, "iron-plate") feeder(r, r.belts[1], 2, 1, "iron-plate")
end

-- ------------------------------------------------------------ sideload from the left
for lane = 1, 2 do for k = 0, 7 do
  local r = rig("sl_left_" .. lane .. "_" .. k, 3, 2)
  belt(r, 0, 1, D.east) belt(r, 1, 1, D.east) belt(r, 2, 1, D.east)
  local f = belt(r, 1, 0, D.south)
  put(f, lane, 128 + k, "copper-plate")
end end

-- ------------------------------------------------------------ sideload shapes
do
  -- a second side feed into a turn: feeder W (east), T (south), feeder E (west)
  local r = rig("side_turn_two", 3, 4)
  local w = belt(r, 0, 0, D.east)
  belt(r, 1, 0, D.south)
  local e = belt(r, 2, 0, D.west)
  belt(r, 1, 1, D.south) belt(r, 1, 2, D.south) belt(r, 1, 3, D.south)
  for lane = 1, 2 do feeder(r, w, lane, 3, "iron-plate") feeder(r, e, lane, 3, "copper-plate") end
  -- a feeder added behind a turn
  r = rig("side_turn_behind", 3, 5)
  w = belt(r, 0, 1, D.east)
  belt(r, 1, 1, D.south)
  local n = belt(r, 1, 0, D.south)
  belt(r, 1, 2, D.south) belt(r, 1, 3, D.south) belt(r, 1, 4, D.south)
  for lane = 1, 2 do feeder(r, w, lane, 3, "iron-plate") feeder(r, n, lane, 3, "copper-plate") end
  -- sideload onto a belt with nothing downstream
  r = rig("side_end", 1, 4)
  belt(r, 0, 0, D.east)
  for i = 1, 3 do belt(r, 0, i, D.north) end
  for lane = 1, 2 do feeder(r, r.belts[4], lane, 4, "copper-plate") end
  -- sideload onto the first belt of a line
  r = rig("side_start", 3, 4)
  for i = 0, 2 do belt(r, i, 0, D.east) end
  for i = 1, 3 do belt(r, 0, i, D.north) end
  for lane = 1, 2 do feeder(r, r.belts[6], lane, 4, "copper-plate") end
  -- head on
  r = rig("side_headon", 4, 1)
  belt(r, 0, 0, D.east) belt(r, 1, 0, D.east) belt(r, 2, 0, D.west) belt(r, 3, 0, D.west)
  for lane = 1, 2 do
    feeder(r, r.belts[1], lane, 3, "iron-plate") feeder(r, r.belts[4], lane, 3, "copper-plate")
  end
  -- from both sides at once
  r = rig("side_both", 3, 5)
  for i = 0, 2 do belt(r, i, 2, D.east) end
  belt(r, 1, 1, D.south) belt(r, 1, 0, D.south)
  belt(r, 1, 3, D.north) belt(r, 1, 4, D.north)
  for lane = 1, 2 do
    feeder(r, r.belts[5], lane, 3, "iron-plate") feeder(r, r.belts[7], lane, 3, "copper-plate")
  end
end

-- ------------------------------------------------------------ simultaneous sideloads
-- main: three east belts; feed: `F` belts into the side of the middle one, index 1
-- next to the main. items: {lane, feed belt, position} in insertion order.
local function sim_rig(name, spec)
  local F = spec.F or 1
  local side = spec.side or "S"
  local r = rig(name, 3, F + 1)
  local my = side == "S" and 0 or F
  local main, feed = {}, {}
  local function build_main() for i = 0, 2 do main[#main + 1] = belt(r, i, my, D.east) end end
  local function build_feed()
    for k = 1, F do
      if side == "S" then feed[k] = belt(r, 1, my + k, D.north)
      else feed[k] = belt(r, 1, my - k, D.south) end
    end
  end
  if spec.feed_first then build_feed() build_main() else build_main() build_feed() end
  for _, m in ipairs(spec.main_items or {}) do put(main[m[2]], m[1], m[3], "iron-plate") end
  for _, it in ipairs(spec.items or {}) do put(feed[it[2]], it[1], it[3], "copper-plate") end
  for _, fd in ipairs(spec.feeders or {}) do feeder(r, feed[F], fd[1], fd[2], "copper-plate") end
  r.main, r.feedb = main, feed
end
sim_rig("sim_l1first", {items = {{1, 1, 128}, {2, 1, 128}}})
sim_rig("sim_l2first", {items = {{2, 1, 128}, {1, 1, 128}}})
sim_rig("sim_f4_l1first", {F = 4, items = {{1, 4, 128}, {2, 4, 128}}})
sim_rig("sim_f4_l2first", {F = 4, items = {{2, 4, 128}, {1, 4, 128}}})
sim_rig("sim_l1only", {items = {{1, 1, 128}}})
sim_rig("sim_l2only", {items = {{2, 1, 128}}})
sim_rig("sim_l1early", {items = {{1, 1, 120}, {2, 1, 128}}})
sim_rig("sim_l2early", {items = {{1, 1, 128}, {2, 1, 120}}})
sim_rig("sim_feedfirst_l1", {feed_first = true, items = {{1, 1, 128}, {2, 1, 128}}})
sim_rig("sim_feedfirst_l2", {feed_first = true, items = {{2, 1, 128}, {1, 1, 128}}})
sim_rig("sim_active_l1", {items = {{1, 1, 128}, {2, 1, 128}}, main_items = {{2, 1, 250}}})
sim_rig("sim_active_l2", {items = {{2, 1, 128}, {1, 1, 128}}, main_items = {{2, 1, 250}}})
sim_rig("sim_other_l1", {items = {{1, 1, 128}, {2, 1, 128}}, main_items = {{1, 1, 250}}})
sim_rig("sim_mainfirst_l1", {items = {{1, 1, 128}, {2, 1, 128}}, main_items = {{2, 3, 200}}})
sim_rig("sim_n_l1first", {side = "N", items = {{1, 1, 128}, {2, 1, 128}}})
sim_rig("sim_n_l2first", {side = "N", items = {{2, 1, 128}, {1, 1, 128}}})
sim_rig("sim_k3_l1first", {items = {{1, 1, 131}, {2, 1, 131}}})
sim_rig("sim_k3_l2first", {items = {{2, 1, 131}, {1, 1, 131}}})
sim_rig("sim_three_l1first", {F = 4, feeders = {{1, 3}, {2, 3}}})
sim_rig("sim_three_l2first", {F = 4, feeders = {{2, 3}, {1, 3}}})
sim_rig("sim_pairs_l1", {F = 2, items = {{1, 1, 128}, {1, 1, 200}, {2, 1, 128}, {2, 1, 200}}})
sim_rig("sim_pairs_l2", {F = 2, items = {{2, 1, 128}, {2, 1, 200}, {1, 1, 128}, {1, 1, 200}}})

-- ------------------------------------------------------------ turns: drops and pickups
-- T at (1, 2), fed from the west by W at (0, 2): facing south (right turn) or north (left).
local function turn_base(r, facing)
  belt(r, 0, 2, D.east)
  return belt(r, 1, 2, facing)
end
for _, tf in ipairs({{"r", D.south}, {"l", D.north}}) do
  -- inserter drops from the north, east and south side
  local r = rig("tdrop_" .. tf[1] .. "_n", 4, 5) turn_base(r, tf[2])
  chest(r, 1, 0, {{"iron-plate", 20}}) inserter(r, 1, 1, D.north, 5)
  r = rig("tdrop_" .. tf[1] .. "_e", 4, 5) turn_base(r, tf[2])
  chest(r, 3, 2, {{"iron-plate", 20}}) inserter(r, 2, 2, D.east, 5)
  r = rig("tdrop_" .. tf[1] .. "_s", 4, 5) turn_base(r, tf[2])
  chest(r, 1, 4, {{"iron-plate", 20}}) inserter(r, 1, 3, D.south, 5)
  -- drills: north side facing south, east side facing west, south side facing north
  r = rig("tdrill_" .. tf[1] .. "_n", 4, 5) turn_base(r, tf[2])
  drill(r, 1, 1, D.south, 5)  -- centre (1, 1): drops on tile (1, 2)
  r = rig("tdrill_" .. tf[1] .. "_e", 5, 5) turn_base(r, tf[2])
  drill(r, 3, 2, D.west, 5)   -- centre (3, 2): drops at (1.703, 2.5)
  r = rig("tdrill_" .. tf[1] .. "_s", 4, 6) turn_base(r, tf[2])
  drill(r, 2, 4, D.north, 5)  -- centre (2, 4): drops at (1.5, 2.703)
  -- pickups from items stopped on the turn
  for _, side in ipairs({"n", "e", "s"}) do
    r = rig("tpick_" .. tf[1] .. "_" .. side, 4, 5)
    local t = turn_base(r, tf[2])
    local outer = tf[1] == "r" and 1 or 2
    local inner = 3 - outer
    for k, name in ipairs({"iron-plate", "copper-plate", "iron-ore", "copper-ore", "stone"}) do
      put(t, outer, 64 * (k - 1), name)
    end
    put(t, inner, 0, "coal") put(t, inner, 64, "wood")
    if side == "n" then chest(r, 1, 0) inserter(r, 1, 1, D.south, 5)
    elseif side == "e" then chest(r, 3, 2) inserter(r, 2, 2, D.west, 5)
    else chest(r, 1, 4) inserter(r, 1, 3, D.north, 5) end
  end
end
-- pickups from items stopped at the end of a straight east belt, from each side
for _, side in ipairs({"n", "s", "e", "w"}) do
  local r = rig("spick_" .. side, 5, 5)
  local b = belt(r, 2, 2, D.east)
  for k, name in ipairs({"iron-plate", "copper-plate", "iron-ore", "copper-ore"}) do
    put(b, 1, 64 * (k - 1), name)
  end
  for k, name in ipairs({"coal", "wood", "stone", "iron-gear-wheel"}) do
    put(b, 2, 64 * (k - 1), name)
  end
  if side == "n" then chest(r, 2, 0) inserter(r, 2, 1, D.south, 5)
  elseif side == "s" then chest(r, 2, 4) inserter(r, 2, 3, D.north, 5)
  elseif side == "e" then chest(r, 4, 2) inserter(r, 3, 2, D.west, 5)
  else chest(r, 0, 2) inserter(r, 1, 2, D.east, 5) end
end

-- ------------------------------------------------------------ ground
do
  local r = rig("ground_drop", 1, 4, {ground = true})
  chest(r, 0, 0, {{"iron-plate", 20}}) inserter(r, 0, 1, D.north, 5)
  r = rig("ground_drop_taken", 1, 4, {ground = true})
  chest(r, 0, 0, {{"iron-plate", 20}})
  local e = inserter(r, 0, 1, D.north, 5)
  mk({name = "item-on-ground", position = e.drop_position, stack = {name = "copper-plate",
    count = 1}})
  local offsets = {center = {0, 0}, off = {0.3, 0.3}, edge = {-0.45, 0}, outside = {0, -0.6}}
  for key, o in pairs(offsets) do
    r = rig("ground_pick_" .. key, 1, 4, {ground = true})
    local p = at(r, 0, 0)
    mk({name = "item-on-ground", position = {p[1] + o[1], p[2] + o[2]},
      stack = {name = "iron-plate", count = 1}})
    inserter(r, 0, 1, D.north, 5) chest(r, 0, 2)
  end
  r = rig("ground_pick_multi", 1, 4, {ground = true})
  local p = at(r, 0, 0)
  mk({name = "item-on-ground", position = p, stack = {name = "iron-plate", count = 1}})
  mk({name = "item-on-ground", position = {p[1] + 0.3, p[2] - 0.3},
    stack = {name = "copper-plate", count = 1}})
  mk({name = "item-on-ground", position = {p[1] - 0.3, p[2] + 0.3},
    stack = {name = "iron-ore", count = 1}})
  inserter(r, 0, 1, D.north, 5) chest(r, 0, 2)
  r = rig("ground_pick_stack", 1, 4, {ground = true})
  mk({name = "item-on-ground", position = at(r, 0, 0), stack = {name = "iron-plate", count = 5}})
  inserter(r, 0, 1, D.north, 5) chest(r, 0, 2)
end

-- ------------------------------------------------------------ full chests
do
  local stone = {}
  for i = 1, 16 do stone[i] = {"stone", 50, slot = i} end
  local r = rig("full_stone", 1, 3)
  chest(r, 0, 0, {{"iron-plate", 20}}) inserter(r, 0, 1, D.north, 5) chest(r, 0, 2, stone)
  event(RELEASE, {kind = "clear_slot", rig = "full_stone", chest = 2, slot = 16})
  local room = {}
  for i = 1, 15 do room[i] = {"stone", 50, slot = i} end
  room[16] = {"iron-plate", 99, slot = 16}
  r = rig("full_room_one", 1, 3)
  chest(r, 0, 0, {{"iron-plate", 20}}) inserter(r, 0, 1, D.north, 5) chest(r, 0, 2, room)
  event(RELEASE, {kind = "take", rig = "full_room_one", chest = 2, name = "iron-plate", count = 1})
  local plates = {}
  for i = 1, 16 do plates[i] = {"iron-plate", 100, slot = i} end
  r = rig("full_plates", 1, 3)
  chest(r, 0, 0, {{"iron-plate", 20}}) inserter(r, 0, 1, D.north, 5) chest(r, 0, 2, plates)
  event(RELEASE, {kind = "take", rig = "full_plates", chest = 2, name = "iron-plate", count = 1})
end

-- ------------------------------------------------------------ fill limits
-- chest (0, 0) -> inserter (0, 1) facing north -> target on tile (0, 2)
local function fill_rig(name, item, count, target)
  local r = rig(name, 3, 4)
  chest(r, 0, 0, {{item, count}})
  inserter(r, 0, 1, D.north, 5)
  if target == "drill" then drill(r, 1, 3, D.north, 0, false)
  elseif target == "drill_ore" then drill(r, 1, 3, D.north, 0, true)
  elseif target == "furnace" then furnace(r, 1, 3)
  elseif target == "hot_furnace" then
    local f = furnace(r, 1, 3)
    f.get_inventory(defines.inventory.furnace_source).insert({name = "iron-ore", count = 30})
  elseif target == "inserter" then inserter(r, 0, 2, D.east, 0)
  elseif target == "chest" then chest(r, 0, 2) end
  return r
end
fill_rig("fill_coal_drill", "coal", 50, "drill")
fill_rig("fill_coal_drill_ore", "coal", 50, "drill_ore")
fill_rig("fill_coal_ins", "coal", 50, "inserter")
fill_rig("fill_wood_furnace", "wood", 50, "furnace")
fill_rig("fill_wood_drill", "wood", 50, "drill")
fill_rig("fill_wood_ins", "wood", 50, "inserter")
fill_rig("fill_coal_chest", "coal", 20, "chest")
fill_rig("fill_ore_drill", "iron-ore", 50, "drill_ore")
fill_rig("fill_copper_furnace", "copper-ore", 50, "furnace")
fill_rig("fill_stone_furnace", "stone", 50, "furnace")
fill_rig("fill_plate_furnace", "iron-plate", 50, "furnace")
fill_rig("fill_coal_hot_furnace", "coal", 50, "hot_furnace")

-- ------------------------------------------------------------ mixed chests
local function mix_rig(name, items, target, coal)
  local r = rig(name, 3, 4)
  chest(r, 0, 0, items)
  inserter(r, 0, 1, D.north, coal == nil and 5 or coal)
  if target == "chest" then chest(r, 0, 2)
  elseif target == "furnace" then furnace(r, 1, 3)
  elseif target == "drill" then drill(r, 1, 3, D.north, 0, false)
  elseif target == "belt" then belt(r, 0, 2, D.east) belt(r, 1, 2, D.east) belt(r, 2, 2, D.east)
  end
  return r
end
mix_rig("mix_plate_ore_coal_chest", {{"iron-plate", 5, slot = 1}, {"iron-ore", 5, slot = 2},
  {"coal", 5, slot = 3}}, "chest")
mix_rig("mix_ore_plate_chest", {{"iron-ore", 5, slot = 1}, {"iron-plate", 5, slot = 2}}, "chest")
mix_rig("mix_plate_ore_furnace", {{"iron-plate", 10, slot = 1}, {"iron-ore", 10, slot = 2}},
  "furnace")
mix_rig("mix_coal_ore_furnace", {{"coal", 10, slot = 1}, {"iron-ore", 10, slot = 2}}, "furnace")
mix_rig("mix_ore_coal_furnace", {{"iron-ore", 10, slot = 1}, {"coal", 10, slot = 2}}, "furnace")
mix_rig("mix_gaps_chest", {{"iron-ore", 5, slot = 3}, {"iron-plate", 5, slot = 7}}, "chest")
mix_rig("mix_plate_ore_belt", {{"iron-plate", 5, slot = 1}, {"iron-ore", 5, slot = 2}}, "belt")
mix_rig("mix_stone_coal_drill", {{"stone", 5, slot = 1}, {"coal", 5, slot = 2}}, "drill")
mix_rig("mix_plate_coal_self", {{"iron-plate", 5, slot = 1}, {"coal", 5, slot = 2}}, "chest", 0)
mix_rig("mix_split_chest", {{"iron-ore", 3, slot = 1}, {"iron-plate", 3, slot = 2},
  {"iron-ore", 3, slot = 3}}, "chest")

-- ------------------------------------------------------------ update order
local function order2(name, first)
  local r = rig(name, 3, 5)
  chest(r, 1, 2, {{"iron-plate", 1}})
  chest(r, 1, 0) chest(r, 1, 4)
  local function a() inserter(r, 1, 1, D.south, 5) end
  local function b() inserter(r, 1, 3, D.north, 5) end
  if first == "a" then a() b() else b() a() end
end
order2("order_pick_ab", "a")
order2("order_pick_ba", "b")
local function order4(name, sequence, plates)
  local r = rig(name, 5, 5)
  chest(r, 2, 2, {{"iron-plate", plates}})
  local spots = {n = {2, 1, D.south, 2, 0}, s = {2, 3, D.north, 2, 4}, e = {3, 2, D.west, 4, 2},
    w = {1, 2, D.east, 0, 2}}
  for _, key in ipairs(sequence) do
    local sp = spots[key]
    chest(r, sp[4], sp[5])
    inserter(r, sp[1], sp[2], sp[3], 5)
  end
end
order4("order4_nesw_1", {"n", "e", "s", "w"}, 1)
order4("order4_wsen_1", {"w", "s", "e", "n"}, 1)
order4("order4_ewns_1", {"e", "w", "n", "s"}, 1)
order4("order4_nesw_3", {"n", "e", "s", "w"}, 3)
order4("order4_wsen_3", {"w", "s", "e", "n"}, 3)
order4("order4_ewns_3", {"e", "w", "n", "s"}, 3)
local function order_drop(name, first)
  local r = rig(name, 3, 5)
  local room = {}
  for i = 1, 15 do room[i] = {"stone", 50, slot = i} end
  room[16] = {"iron-plate", 99, slot = 16}
  chest(r, 1, 2, room)
  chest(r, 1, 0, {{"iron-plate", 10}}) chest(r, 1, 4, {{"iron-plate", 10}})
  local function a() inserter(r, 1, 1, D.north, 5) end
  local function b() inserter(r, 1, 3, D.south, 5) end
  if first == "a" then a() b() else b() a() end
end
order_drop("order_drop_ab", "a")
order_drop("order_drop_ba", "b")
do
  -- three inserters with ore around one unfuelled furnace on tiles (1..2, 1..2)
  local r = rig("order_furnace3", 4, 5)
  furnace(r, 2, 2)
  chest(r, 1, -1, {{"iron-ore", 10}}) inserter(r, 1, 0, D.north, 5)
  chest(r, 2, -1, {{"iron-ore", 10}}) inserter(r, 2, 0, D.north, 5)
  chest(r, -1, 1, {{"iron-ore", 10}}) inserter(r, 0, 1, D.west, 5)
end

-- ------------------------------------------------------------ rotations of loaded belts
local rotations = {}
local function items_all(b)
  return {line_items(b.get_transport_line(1)), line_items(b.get_transport_line(2))}
end
local function load(b, len1, len2, j)
  local p = j
  while p < len1 do put(b, 1, p, "iron-plate") p = p + 64 end
  p = j
  while p < len2 do put(b, 2, p, "copper-plate") p = p + 64 end
end
local function rot_rig(name, facing, j, how)
  local r = rig(name, 2, 1, {until_t = ROT_TICKS})
  local w = belt(r, 0, 0, D.east)
  local t = belt(r, 1, 0, facing)
  local l1, l2 = 256, 256
  if t.belt_shape == "right" then l1, l2 = 295, 106 elseif t.belt_shape == "left" then
    l1, l2 = 106, 295 end
  load(t, l1, l2, j)
  local before = {shape = t.belt_shape, d = t.direction, l = items_all(t)}
  if how == "cw" then t.rotate({reverse = false})
  elseif how == "ccw" then t.rotate({reverse = true})
  elseif how == "mine_feeder" then w.destroy()
  elseif how == "add_behind" then
    local d = t.direction
    local dx, dy = 0, 0
    if d == D.south then dy = -1 elseif d == D.north then dy = 1 elseif d == D.east then dx = -1
    else dx = 1 end
    belt(r, 1 + dx, dy, d)
  end
  rotations[name] = {j = j, how = how, before = before,
    after = {shape = t.belt_shape, d = t.direction, l = items_all(t)}}
end
for j = 0, 63 do rot_rig("rot_r2s_ccw_" .. j, D.south, j, "ccw") end
for j = 0, 63 do rot_rig("rot_s2r_cw_" .. j, D.east, j, "cw") end
for j = 0, 63, 4 do rot_rig("rot_r2s_cw_" .. j, D.south, j, "cw") end
for j = 0, 63, 4 do rot_rig("rot_s2l_ccw_" .. j, D.east, j, "ccw") end
for j = 0, 63, 4 do rot_rig("rot_l2s_cw_" .. j, D.north, j, "cw") end
for j = 0, 63, 4 do rot_rig("rot_r_mine_" .. j, D.south, j, "mine_feeder") end
for j = 0, 63, 4 do rot_rig("rot_r_behind_" .. j, D.south, j, "add_behind") end
do
  -- a straight belt running south that a new feeder at its west turns
  for j = 0, 63, 4 do
    local name = "rot_s2r_side_" .. j
    local r = rig(name, 2, 1, {until_t = ROT_TICKS})
    local t = belt(r, 1, 0, D.south)
    load(t, 256, 256, j)
    local before = {shape = t.belt_shape, d = t.direction, l = items_all(t)}
    belt(r, 0, 0, D.east)
    rotations[name] = {j = j, how = "add_side", before = before,
      after = {shape = t.belt_shape, d = t.direction, l = items_all(t)}}
    -- keep the turned belt first in the rig's list
    r.belts = {t, r.belts[2]}
  end
end

-- ------------------------------------------------------------ energy running out
for i = 0, 23 do
  local r = rig("pe_" .. i, 1, 3)
  chest(r, 0, 0, {{"iron-plate", 50}}) inserter(r, 0, 1, D.north, 0) chest(r, 0, 2)
  r.ins[1].burner.remaining_burning_fuel = 40000 + 2800 * i
  event(REFUEL, {kind = "fuel", rig = r.name, ins = 1, name = "coal", count = 1})
end
for i = 0, 23 do
  local r = rig("sr_" .. i, 1, 3)
  chest(r, 0, 0, {{"coal", 50}}) inserter(r, 0, 1, D.north, 1) chest(r, 0, 2)
  r.ins[1].burner.remaining_burning_fuel = 30000 + 2800 * i
end

-- ------------------------------------------------------------ drill status after a change
do
  -- drill centred on (1, 1) facing south drops on tile (1, 2)
  local r = rig("dstat_extend", 3, 3) drill(r, 1, 1, D.south, 5) belt(r, 1, 2, D.east)
  event(UNBLOCK, {kind = "belt", rig = r.name, x = 2, y = 2, dir = D.east})
  r = rig("dstat_take_belt", 3, 3) drill(r, 1, 1, D.south, 5) belt(r, 1, 2, D.east)
  event(UNBLOCK, {kind = "belt_take", rig = r.name, belt = 1, lane = 1})
  r = rig("dstat_rotate", 3, 3) drill(r, 1, 1, D.south, 5) belt(r, 1, 2, D.east)
  event(UNBLOCK, {kind = "rotate", rig = r.name, belt = 1})
  r = rig("dstat_unrelated", 5, 3) drill(r, 1, 1, D.south, 5) belt(r, 1, 2, D.east)
  event(UNBLOCK, {kind = "chest", rig = r.name, x = 4, y = 2})
  local stone = {}
  for i = 1, 16 do stone[i] = {"stone", 50, slot = i} end
  r = rig("dstat_chest", 3, 3) drill(r, 1, 1, D.south, 5) chest(r, 1, 2, stone)
  event(UNBLOCK, {kind = "clear_slot", rig = r.name, chest = 1, slot = 16})
  r = rig("dstat_ground", 3, 3, {ground = true}) drill(r, 1, 1, D.south, 5)
  event(UNBLOCK, {kind = "clear_ground", rig = r.name})
end

-- ------------------------------------------------------------ follow-ups
-- straight belts rotated into straight belts: a lone loaded belt turned both ways
for j = 0, 63, 4 do
  for _, how in ipairs({"cw", "ccw"}) do
    local name = "rot_ss_" .. how .. "_" .. j
    local r = rig(name, 1, 1, {until_t = ROT_TICKS})
    local t = belt(r, 0, 0, D.east)
    load(t, 256, 256, j)
    local before = {shape = t.belt_shape, d = t.direction, l = items_all(t)}
    t.rotate({reverse = how == "ccw"})
    rotations[name] = {j = j, how = how, before = before,
      after = {shape = t.belt_shape, d = t.direction, l = items_all(t)}}
  end
end
-- two piles in one pickup tile, at mirrored offsets, for each facing of the inserter
do
  local facings = {{"n", D.north, 0, 1, 0, 2}, {"e", D.east, 0, 0, -1, 0},
    {"s", D.south, 0, -1, 0, -2}, {"w", D.west, 0, 0, 1, 0}}
  local pairs_ = {{"x", {0.3, 0}, {-0.3, 0}}, {"y", {0, 0.3}, {0, -0.3}},
    {"d", {0.3, 0.3}, {-0.3, -0.3}}, {"xr", {-0.3, 0}, {0.3, 0}}}
  for _, fc in ipairs(facings) do
    for _, pr in ipairs(pairs_) do
      local r = rig("gpair_" .. fc[1] .. "_" .. pr[1], 3, 5, {ground = true})
      -- pickup tile at (1, 2); the inserter one tile away, its drop chest beyond
      local ix, iy = 1 + (fc[5] ~= 0 and fc[5] or 0), 2 + (fc[4] ~= 0 and fc[4] or 0)
      if fc[1] == "e" then ix, iy = 0, 2 elseif fc[1] == "w" then ix, iy = 2, 2 end
      if fc[1] == "n" then ix, iy = 1, 3 elseif fc[1] == "s" then ix, iy = 1, 1 end
      local p = at(r, 1, 2)
      mk({name = "item-on-ground", position = {p[1] + pr[2][1], p[2] + pr[2][2]},
        stack = {name = "iron-plate", count = 1}})
      mk({name = "item-on-ground", position = {p[1] + pr[3][1], p[2] + pr[3][2]},
        stack = {name = "copper-plate", count = 1}})
      inserter(r, ix, iy, fc[2], 5)
      chest(r, ix + (ix - 1), iy + (iy - 2))
    end
  end
end
-- a waiting inserter whose drop spot is cleared, and whose held item is then taken
do
  local r = rig("ground_clear", 1, 4, {ground = true})
  chest(r, 0, 0, {{"iron-plate", 20}})
  local e = inserter(r, 0, 1, D.north, 5)
  mk({name = "item-on-ground", position = e.drop_position, stack = {name = "copper-plate",
    count = 1}})
  event(400, {kind = "clear_ground", rig = r.name})
end
-- three inserters asleep on one empty chest, given one plate at 100, 300 and 500
do
  local r = rig("wake3", 5, 5)
  chest(r, 2, 2)
  chest(r, 2, 0) inserter(r, 2, 1, D.south, 5)
  chest(r, 4, 2) inserter(r, 3, 2, D.west, 5)
  chest(r, 2, 4) inserter(r, 2, 3, D.north, 5)
  for _, t in ipairs({100, 300, 500}) do
    event(t, {kind = "put", rig = r.name, chest = 1, name = "iron-plate", count = 1})
  end
end
-- belt line segments: an inserter asleep at the end of a line, one item added at t=400
-- to belt k of the line; which adds wake it at once. Lines of east belts, a turn south
-- and south belts, built along the chain or in reverse.
local seg = {}
local function seg_rig(name, n_east, turn, n_south, k, reverse)
  local r = rig(name, n_east + 2, n_south + 4)
  local list = {}
  for i = 0, n_east - 1 do list[#list + 1] = {i, 0, D.east} end
  if turn then list[#list + 1] = {n_east, 0, D.south} end
  for j = 1, n_south do list[#list + 1] = {n_east, j, D.south} end
  local built_order = {}
  if reverse then for i = #list, 1, -1 do built_order[#built_order + 1] = list[i] end
  else built_order = list end
  local by_index = {}
  for _, b in ipairs(built_order) do
    local e = belt(r, b[1], b[2], b[3])
    for i, l in ipairs(list) do if l == b then by_index[i] = e end end
  end
  r.belts = {}
  for i = 1, #list do r.belts[i] = by_index[i] end
  local last = list[#list]
  if turn or n_south > 0 then
    chest(r, last[1], last[2] + 2) inserter(r, last[1], last[2] + 1, D.north, 5)
  else
    chest(r, last[1], 2) inserter(r, last[1], 1, D.north, 5)
  end
  event(400, {kind = "put_belt", rig = name, belt = k + 1, lane = 1, pos = 128})
  seg[name] = true
end
for k = 0, 8 do seg_rig("seg_a_" .. k, 5, true, 3, k, false) end
for k = 0, 8 do seg_rig("seg_ar_" .. k, 5, true, 3, k, true) end
for k = 0, 7 do seg_rig("seg_b_" .. k, 4, true, 3, k, false) end
for k = 0, 9 do seg_rig("seg_c_" .. k, 10, false, 0, k, false) end
for k = 0, 8 do seg_rig("seg_d_" .. k, 3, true, 5, k, false) end
-- the window an asleep inserter watches: an item put far upstream at t=400 while the
-- inserter swings a held item to its chest; it comes back to rest with the item still
-- far away, falls asleep, and wakes (refills) when the item comes close enough.
local function win_rig(name, n_east, turn, n_south, lane, across)
  local r = rig(name, n_east + 2, n_south + 4)
  for i = 0, n_east - 1 do belt(r, i, 0, D.east) end
  if turn then belt(r, n_east, 0, D.south) end
  for j = 1, n_south do belt(r, n_east, j, D.south) end
  if turn then
    chest(r, n_east, n_south + 2) inserter(r, n_east, n_south + 1, D.north, 5)
  elseif across then
    chest(r, n_east - 2, 2) inserter(r, n_east - 2, 1, D.north, 5)
  else
    chest(r, n_east + 1, 0) inserter(r, n_east, 0, D.west, 5)
  end
  event(400, {kind = "hold", rig = name, ins = 1})
  event(400, {kind = "put_belt", rig = name, belt = 1, lane = lane, pos = 255})
end
win_rig("win_s_near", 20, false, 0, 2, true)
win_rig("win_s_far", 20, false, 0, 1, true)
win_rig("win_s_end", 20, false, 0, 1, false)
win_rig("win_t_1", 10, true, 3, 1, false)
win_rig("win_t_2", 10, true, 3, 2, false)
-- a chain chest -> X -> C -> Y -> D: Y asleep on the empty C when X drops into it,
-- built in both orders
for _, first in ipairs({"x", "y"}) do
  local r = rig("chain_" .. first, 1, 5)
  chest(r, 0, 0, {{"iron-plate", 5}}) chest(r, 0, 2) chest(r, 0, 4)
  local function x() inserter(r, 0, 1, D.north, 5) end
  local function y() inserter(r, 0, 3, D.north, 5) end
  if first == "x" then x() y() else y() x() end
end
-- a woken inserter against an active one arriving the same tick: both pick from S;
-- A (built second) wins S's one plate at t=9, B falls asleep; A is back at S on t=85,
-- and a plate is put into S after t=84
do
  local r = rig("woken_vs_active", 3, 5)
  chest(r, 1, 2, {{"iron-plate", 1}})
  chest(r, 1, 4) inserter(r, 1, 3, D.north, 5)   -- B
  chest(r, 1, 0) inserter(r, 1, 1, D.south, 5)   -- A
  event(84, {kind = "put", rig = r.name, chest = 1, name = "iron-plate", count = 1})
  -- the same with the plate put after t=83, so B wakes a tick before A arrives
  r = rig("woken_early", 3, 5)
  chest(r, 1, 2, {{"iron-plate", 1}})
  chest(r, 1, 4) inserter(r, 1, 3, D.north, 5)
  chest(r, 1, 0) inserter(r, 1, 1, D.south, 5)
  event(83, {kind = "put", rig = r.name, chest = 1, name = "iron-plate", count = 1})
end
-- status corners
do
  -- coal into a furnace that has fuel and no ore
  local r = rig("stat_fueled_furnace", 3, 4)
  chest(r, 0, 0, {{"coal", 50}}) inserter(r, 0, 1, D.north, 5)
  local f = furnace(r, 1, 3) f.insert({name = "coal", count = 2})
  -- coal into an inserter that is moving plates
  r = rig("stat_busy_ins", 3, 4)
  chest(r, 0, 0, {{"coal", 50}}) inserter(r, 0, 1, D.north, 5)
  chest(r, 1, 2, {{"iron-plate", 50}}) inserter(r, 0, 2, D.east, 0) chest(r, -1, 2)
  -- plates into a chest whose every slot holds part of a stack of stone
  local part = {}
  for i = 1, 16 do part[i] = {"stone", 49, slot = i} end
  r = rig("stat_partial_stone", 1, 3)
  chest(r, 0, 0, {{"iron-plate", 20}}) inserter(r, 0, 1, D.north, 5) chest(r, 0, 2, part)
  -- the same, and the source has stone too (last slot)
  r = rig("stat_partial_stone_src", 1, 3)
  chest(r, 0, 0, {{"iron-plate", 20, slot = 1}, {"stone", 5, slot = 2}})
  inserter(r, 0, 1, D.north, 5) chest(r, 0, 2, part)
  -- a chest with room for stone only, and plates in the source's last slot
  r = rig("stat_room_stone", 1, 3)
  chest(r, 0, 0, {{"stone", 5, slot = 1}, {"iron-plate", 20, slot = 2}})
  inserter(r, 0, 1, D.north, 5) chest(r, 0, 2, part)
end
-- the character carried by a belt into a chest, and turns at several points
local char = {}
do
  local r = rig("char_chestend", 3, 1) belt(r, 0, 0, D.east) belt(r, 1, 0, D.east)
  chest(r, 2, 0)
  char.chestend = at(r, 0, 0)
  r = rig("char_rturn", 2, 6)
  belt(r, 0, 0, D.east) belt(r, 1, 0, D.south)
  for i = 1, 5 do belt(r, 1, i, D.south) end
  char.rturn = at(r, 1, 0)
  r = rig("char_lturn", 2, 6)
  belt(r, 0, 5, D.east) belt(r, 1, 5, D.north)
  for i = 4, 0, -1 do belt(r, 1, i, D.north) end
  char.lturn = at(r, 1, 5)
end
do
  local r = rig("char_row", 12, 1) for i = 0, 11 do belt(r, i, 0, D.east) end
  char.row = at(r, 3, 0)
  r = rig("char_turn", 2, 8)
  belt(r, 0, 0, D.east) belt(r, 1, 0, D.south)
  for i = 1, 7 do belt(r, 1, i, D.south) end
  char.turn = at(r, 1, 0)
  r = rig("char_north", 1, 10)
  for i = 9, 0, -1 do belt(r, 0, i, D.north) end
  for lane = 1, 2 do feeder(r, r.belts[1], lane, 20, "iron-plate") end
  char.lane1 = {at(r, 0, 8)[1] - 60 / 256, at(r, 0, 8)[2]}
  char.lane2 = {at(r, 0, 8)[1] + 60 / 256, at(r, 0, 8)[2]}
  r = rig("char_end", 2, 1) belt(r, 0, 0, D.east) belt(r, 1, 0, D.east)
  char.last = at(r, 1, 0)
  r = rig("char_free", 3, 3)
  char.free = at(r, 1, 1)
  -- mining: the character stands at (4, 4)
  r = rig("mine", 9, 10)
  char.mine = at(r, 4, 4)
  local c = chest(r, 1, 4, {{"iron-plate", 50, slot = 1}, {"iron-ore", 30, slot = 2},
    {"copper-plate", 10, slot = 3}, {"stone-brick", 4, slot = 5}, {"stone", 5, slot = 16}})
  local b = belt(r, 7, 4, D.east)
  put(b, 1, 0, "wood") put(b, 1, 64, "iron-gear-wheel") put(b, 1, 128, "iron-stick")
  put(b, 2, 0, "copper-cable") put(b, 2, 64, "pipe") put(b, 2, 128, "electronic-circuit")
  -- an inserter that holds copper ore: an item lies where it would drop it
  chest(r, 4, 9, {{"copper-ore", 10}})
  local hold = inserter(r, 4, 8, D.south, 3)
  mk({name = "item-on-ground", position = hold.drop_position, stack = {name = "stone",
    count = 1}})
  -- an idle inserter with fuel
  local idle = inserter(r, 4, 1, D.north, 3)
  storage.probe.mine_queue = {c, b, hold, idle}
end

-- placement under the character
local ch = storage.frrl_character
ch.get_inventory(defines.inventory.character_main).clear()
ch.teleport(char.free)
local placement = {}
for _, name in ipairs({"transport-belt", "wooden-chest", "burner-inserter"}) do
  local out = {}
  for _, o in ipairs({{0, 0}, {0.3, 0}, {0.6, 0}, {0.75, 0}, {0.9, 0}, {1.0, 0}}) do
    out[#out + 1] = {o[1], o[2], s.can_place_entity({name = name,
      position = {char.free[1] + o[1], char.free[2] + o[2]}, direction = D.east,
      force = "player", build_check_type = defines.build_check_type.manual})}
  end
  placement[name] = out
end
placement.character_box = {ch.prototype.collision_box.left_top.x,
  ch.prototype.collision_box.left_top.y}
ch.teleport(char.row)

local plan = {
  {t = 40, kind = "walk", dir = D.east}, {t = 60, kind = "walk", dir = D.west},
  {t = 90, kind = "stop"}, {t = 100, kind = "walk", dir = D.north}, {t = 115, kind = "stop"},
  {t = 130, kind = "teleport", at = char.turn}, {t = 200, kind = "teleport", at = char.lane1},
  {t = 240, kind = "teleport", at = char.lane2}, {t = 280, kind = "teleport", at = char.last},
  {t = 340, kind = "teleport", at = char.mine}, {t = 345, kind = "mine"},
  {t = 420, kind = "teleport", at = char.chestend},
}
local k = 0
for _, o in ipairs({{0, 0}, {0.25, 0}, {0, -0.25}, {-0.3, 0.3}, {0.3, 0.3}, {0.3, -0.3},
    {-0.3, -0.3}, {0.45, 0.45}}) do
  plan[#plan + 1] = {t = 500 + 40 * k, kind = "teleport",
    at = {char.rturn[1] + o[1], char.rturn[2] + o[2]}}
  k = k + 1
end
for _, o in ipairs({{0, 0}, {0.3, -0.3}, {-0.3, 0.3}}) do
  plan[#plan + 1] = {t = 500 + 40 * k, kind = "teleport",
    at = {char.lturn[1] + o[1], char.lturn[2] + o[2]}}
  k = k + 1
end
for _, p in ipairs(plan) do event(p.t, p) end

-- ------------------------------------------------------------ what was built
for _, name in ipairs({"turn_en_left", "turn_ws_left", "turn_ne_right", "turn_es_right",
    "side_turn_two", "side_turn_behind", "side_end", "side_start", "side_headon",
    "side_both", "tdrop_r_n", "tdrop_l_n", "char_turn"}) do
  lines[name] = lineinfo(R[name].belts)
end
for name in pairs(seg) do
  local out = {}
  local last = R[name].belts[#R[name].belts]
  for i, b in ipairs(R[name].belts) do
    local row = {}
    for lane = 1, 2 do
      local l = b.get_transport_line(lane)
      row[lane] = {seg = g(try(function() return l.total_segment_length end)),
        same_as_pickup = try(function() return l.line_equals(last.get_transport_line(lane)) end)}
    end
    out[i] = row
  end
  lines[name] = out
end
local targets = {}
for _, name in ipairs(order) do
  local r = R[name]
  for i, e in ipairs(r.ins) do
    targets[name .. "#" .. i] = {pickup = xy(e.pickup_position), drop = xy(e.drop_position),
      at = xy(e.position), direction = e.direction,
      pickup_target = e.pickup_target and e.pickup_target.name or nil,
      drop_target = e.drop_target and e.drop_target.name or nil}
  end
  for i, d in ipairs(r.drills) do
    targets[name .. "#d" .. i] = {drop = xy(d.drop_position), at = xy(d.position)}
  end
end

local function proto(name)
  local p = prototypes.entity[name]
  local mp = p.mineable_properties
  local cb = p.collision_box
  local out = {collision_box = {cb.left_top.x, cb.left_top.y, cb.right_bottom.x, cb.right_bottom.y},
    mining_time = mp and mp.mining_time, minable = mp and mp.minable}
  for _, key in ipairs({"energy_per_movement", "energy_per_rotation", "belt_speed",
      "inserter_rotation_speed", "inserter_extension_speed", "mining_speed",
      "inserter_stack_size_bonus", "filter_count"}) do
    out[key] = try(function() return p[key] end)
  end
  out.max_energy_usage = try(function() return p.get_max_energy_usage() end)
  return out
end
local protos = {}
for _, name in ipairs({"transport-belt", "wooden-chest", "burner-inserter",
    "burner-mining-drill", "stone-furnace", "item-on-ground", "character", "stone-wall"}) do
  protos[name] = proto(name)
end
local cp = prototypes.entity.character
protos.character.running_speed = try(function() return cp.running_speed end)
protos.character.mining_speed = try(function() return cp.mining_speed end)

storage.probe.R = R
storage.probe.order = order
storage.probe.events = events
storage.probe.prev = {}
storage.probe.t0 = game.tick
storage.probe.mining = false
local failed = {}
for _, b in ipairs(built) do if not b[2] then failed[#failed + 1] = b[1] end end
local bases = {}
for _, name in ipairs(order) do bases[name] = {R[name].bx, R[name].by} end
return {built = #built, failed = failed, rigs = order, bases = bases, lines = lines,
  targets = targets,
  rotations = rotations, placement = placement, prototypes = protos, t0 = game.tick}
"""

SAMPLE = """
local P = storage.probe
local t = game.tick - P.t0
local ch = storage.frrl_character
-- scripted events due at this tick, before anything is read
for _, ev in ipairs(P.events) do
  if ev.t == t then
    local r = ev.rig and P.R[ev.rig]
    local k = ev.kind
    if k == "clear_slot" then
      r.chests[ev.chest].get_inventory(defines.inventory.chest)[ev.slot].clear()
    elseif k == "take" then
      r.chests[ev.chest].remove_item({name = ev.name, count = ev.count})
    elseif k == "hold" then
      r.ins[ev.ins].held_stack.set_stack({name = "iron-plate", count = 1})
    elseif k == "put_belt" then
      r.belts[ev.belt].get_transport_line(ev.lane).insert_at(ev.pos / 256,
        {name = "iron-plate", count = 1})
    elseif k == "put" then
      r.chests[ev.chest].insert({name = ev.name, count = ev.count})
    elseif k == "fuel" then
      r.ins[ev.ins].insert({name = ev.name, count = ev.count})
    elseif k == "belt" then
      r.belts[#r.belts + 1] = s.create_entity({name = "transport-belt", force = "player",
        position = {OX + r.bx + ev.x + 0.5, OY + r.by + ev.y + 0.5}, direction = ev.dir})
    elseif k == "belt_take" then
      local line = r.belts[ev.belt].get_transport_line(ev.lane)
      local items = line.get_detailed_contents()
      local front = nil
      for _, it in pairs(items) do if front == nil or it.position < front.position then
        front = it end end
      if front and not try(function() front.stack.clear() return true end) then
        line.remove_item({name = front.stack.name, count = 1})
      end
    elseif k == "rotate" then
      r.belts[ev.belt].rotate({reverse = false})
    elseif k == "chest" then
      r.chests[#r.chests + 1] = s.create_entity({name = "wooden-chest", force = "player",
        position = {OX + r.bx + ev.x + 0.5, OY + r.by + ev.y + 0.5}})
    elseif k == "clear_ground" then
      for _, e in pairs(s.find_entities_filtered({area = r.area, type = "item-entity"})) do
        e.destroy()
      end
    elseif k == "walk" then
      ch.walking_state = {walking = true, direction = ev.dir}
    elseif k == "stop" then
      ch.walking_state = {walking = false}
    elseif k == "teleport" then
      ch.teleport(ev.at)
    elseif k == "mine" then
      P.mining = true
    end
  end
end
-- belt feeders
for _, name in ipairs(P.order) do
  local r = P.R[name]
  for _, fd in ipairs(r.feed) do
    if fd.left > 0 then
      local line = fd.belt.get_transport_line(fd.lane)
      if line.can_insert_at_back() and line.insert_at_back({name = fd.item, count = 1}) then
        fd.left = fd.left - 1
      end
    end
  end
end
-- the character's mining: the next valid target, re-asserted every tick
if P.mining then
  local target = nil
  for _, e in ipairs(P.mine_queue) do if e.valid then target = e break end end
  if target then
    ch.update_selected_entity(target.position)
    ch.mining_state = {mining = true, position = target.position}
  else
    ch.mining_state = {mining = false}
    P.mining = false
  end
end
local out = {}
for _, name in ipairs(P.order) do
  local r = P.R[name]
  if not r.until_t or t <= r.until_t then
    local text = helpers.table_to_json(rig_rec(r))
    if P.prev[name] ~= text then
      out[name] = text
      P.prev[name] = text
    end
  end
end
local text = helpers.table_to_json({char = char_rec()})
if P.prev_char ~= text then out._char = text P.prev_char = text end
return {t = t, tick = game.tick, rigs = out}
"""


def lua(client: RCONClient, code: str):
    for key, value in (("OX", OX), ("OY", OY), ("UNBLOCK", UNBLOCK), ("RELEASE", RELEASE),
                       ("REFUEL", REFUEL), ("ROT_TICKS", ROT_TICKS)):  # fmt: skip
        code = code.replace(key, str(value))
    return client.lua(PRELUDE + code)


def main() -> int:
    manager = WorkerManager()
    handle = manager.launch("logistics-probe-2")
    report: dict = {"engine": manager.engine.to_dict(), "origin": [OX, OY], "ticks": TICKS,
                    "events": {"unblock": UNBLOCK, "release": RELEASE, "refuel": REFUEL,
                               "rotation_ticks": ROT_TICKS}}  # fmt: skip
    series: dict[str, list] = {}
    character: list = []
    started = time.perf_counter()
    try:
        with RCONClient(handle.spec.rcon_endpoint, timeout=120.0) as client:
            client.lua(f"game.speed = {resolve_game_speed()} return game.speed")
            session = WorkerSession(handle, timeout=120.0)
            session.status()
            env = FactorioEnv(
                get("construct_smelting_line"),
                session,
                SeedPlan(master=41, run_id="logistics-probe-2"),
                branch=Branch.TRAIN,
                split="train",
            )
            env.reset(options={"scene_index": 0})
            report["setup"] = lua(client, SETUP)

            def take(sample):
                t = sample["t"]
                for name, text in (sample.get("rigs") or {}).items():
                    value = json.loads(text)
                    if name == "_char":
                        character.append([t, value["char"]])
                    else:
                        series.setdefault(name, []).append([t, value])

            take(lua(client, SAMPLE))
            for _ in range(TICKS):
                env.advance(1)
                take(lua(client, SAMPLE))
            session.close()
    finally:
        manager.cleanup(handle)
    report["wall_seconds"] = round(time.perf_counter() - started, 1)
    report["series"] = series
    report["character"] = character
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(lzma.compress(json.dumps(report, sort_keys=True).encode()))
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KiB), {report['wall_seconds']} s")
    print("built", report["setup"]["built"], "failed", report["setup"]["failed"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
