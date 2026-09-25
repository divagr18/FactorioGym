"""Measure hand-mining and reach on the engine, tick by tick (v3 decisions 1 and 2).

Two families, one evidence file each (`docs/evidence/handmine-<family>.json.xz`),
both raw Lua driven the way the mod drives the character:

**`mine`** -- rigs run one after another in one world, each on its own patch
of freshly painted grass with resource entities of a chosen amount. While a
rig's `mine` op is active the character's selection and `mining_state` are
re-asserted every tick at the tile, exactly as the mod's `POLLS.mine` does;
`stop` clears `mining_state` (the poller's own stop), `walk` sets
`walking_state` every tick until `halt`. Every tick it changes the reading is
logged:

- `p`: character position (1/256 tiles);
- `m`: `mining_state.mining`; `g`: `character_mining_progress` (%.17g);
- `w`: `walking_state.walking`;
- `inv`: main-inventory count of each tracked item, and `e`: empty stacks;
- `a`: each rig resource's amount, or -1 once it is gone;
- `sel`: the selected entity's name, or "";
- `piles`: item-on-ground entities in the rig's area, [x, y (1/256, from the
  base tile), item, count]; `built`: player-force entity names there;
- `prod`: the force's item production input count of each tracked item.

Rigs: every resource's time per item and amount decrease to exhaustion;
stopping and resuming, switching tile and back; walking while mining;
a tile beyond resource reach; a full inventory with and without a partial
stack of the mined item; a tile under a burner drill.

**`reach`** -- no ticks. For each entity type, one entity at a fixed centre
and the character teleported over a grid of sub-tile offsets (1/16 tile) and
whole-tile offsets up to 12, asking `can_reach_entity`; then, along rays at
1/256 resolution, the last accepted and first refused positions. The mod's
two other distances are code (`actions.lua`): a placement is refused when
`distance(character.position, position) > build_distance`, and a resource
mine when `distance(character.position, resource.position) >
resource_reach_distance`; the prototype values are recorded here.

Run (on the desktop or the laptop, about a minute each):
  uv run python tools/probe_handmine.py --family mine
  uv run python tools/probe_handmine.py --family reach
"""

from __future__ import annotations

import argparse
import json
import lzma
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs" / "evidence"

RESOURCES = ("iron-ore", "copper-ore", "coal", "stone")
TRACKED = (
    "iron-ore",
    "copper-ore",
    "coal",
    "stone",
    "wood",
    "iron-plate",
    "burner-mining-drill",
    "stone-furnace",
    "transport-belt",
    "burner-inserter",
    "wooden-chest",
    "stone-wall",
)


class Rig:
    """`base` tile; resources `(dx, dy, name, amount)`; character start and
    inventory; timed ops `[t, op, ...]`; `ticks` recorded."""

    def __init__(self, name, base, ticks, start=(0.5, 0.5), inventory=None, fill=None):
        self.name, self.base, self.ticks = name, base, ticks
        self.start, self.inventory, self.fill = start, inventory or {}, fill
        self.resources: list[list] = []
        self.entities: list[list] = []
        self.ops: list[list] = []

    def res(self, dx, dy, name, amount):
        self.resources.append([dx, dy, name, amount])
        return self

    def ent(self, name, x, y, d=0):
        self.entities.append([name, x, y, d])
        return self

    def op(self, t, *args):
        self.ops.append([t, *args])
        return self

    def to_dict(self):
        return {"name": self.name, "base": list(self.base), "ticks": self.ticks,
                "start": list(self.start), "inventory": self.inventory,
                "fill": self.fill or {}, "resources": self.resources,
                "entities": self.entities, "ops": self.ops}  # fmt: skip


def mine_rigs() -> list[Rig]:
    rigs = []
    # 1. Time per item and amount to exhaustion, per resource: 3 ore, mined
    #    to the end, then kept asking for 60 ticks.
    for k, name in enumerate(RESOURCES):
        r = Rig(f"time_{name}", (40 * k, 0), 560, start=(0.5, 0.5))
        r.res(2, 0, name, 3)
        r.op(1, "mine", 2, 0).op(500, "stop")
        rigs.append(r)
    # 1b. The start and first item more finely, from a larger deposit, with a
    #     second item started and stopped at 1/2.
    r = Rig("time_iron_big", (160, 0), 400, start=(0.5, 0.5))
    r.res(2, 0, "iron-ore", 1000)
    r.op(1, "mine", 2, 0).op(182, "stop")
    rigs.append(r)
    # 2. Stop and resume, and switch tile and back.
    r = Rig("stop_resume", (0, 40), 700, start=(0.5, 0.5))
    r.res(2, 0, "iron-ore", 50).res(-2, 0, "iron-ore", 50)
    r.op(1, "mine", 2, 0).op(61, "stop").op(121, "mine", 2, 0)  # 60 ticks idle
    r.op(300, "mine", -2, 0).op(360, "mine", 2, 0).op(600, "stop")
    rigs.append(r)
    # 2b. Stopped for one tick only, and re-asserted on the same tick as stop.
    r = Rig("stop_one_tick", (40, 40), 400, start=(0.5, 0.5))
    r.res(2, 0, "iron-ore", 50)
    r.op(1, "mine", 2, 0).op(40, "stop").op(41, "mine", 2, 0).op(300, "stop")
    rigs.append(r)
    # 3. Walking while mining: the walk starts mid-item and mining stays asked for.
    r = Rig("walk_while_mining", (80, 40), 500, start=(0.5, 0.5))
    r.res(2, 0, "iron-ore", 50)
    r.op(1, "mine", 2, 0).op(60, "walk", "north").op(70, "halt").op(400, "stop")
    rigs.append(r)
    # 3b. Mining asked for while already walking.
    r = Rig("mine_while_walking", (120, 40), 400, start=(0.5, 0.5))
    r.res(2, 0, "iron-ore", 50)
    r.op(1, "walk", "north").op(5, "mine", 2, 0).op(40, "halt").op(300, "stop")
    rigs.append(r)
    # 4. A tile beyond resource reach (the mod refuses it; the engine?).
    r = Rig("beyond_reach", (160, 40), 300, start=(0.5, 0.5))
    r.res(4, 0, "iron-ore", 50).res(3, 1, "iron-ore", 50)
    r.op(1, "mine", 4, 0).op(140, "mine", 3, 1).op(280, "stop")
    rigs.append(r)
    # 5. Inventory full: every slot wood, nothing to put ore into.
    r = Rig("full_no_room", (0, 80), 400, start=(0.5, 0.5), fill={"wood": 80})
    r.res(2, 0, "iron-ore", 50)
    r.op(1, "mine", 2, 0).op(350, "stop")
    rigs.append(r)
    # 5b. 79 slots wood and 49 iron ore in the last: one more fits, then none.
    r = Rig("full_partial_stack", (40, 80), 600, start=(0.5, 0.5),
            fill={"wood": 79}, inventory={"iron-ore": 49})  # fmt: skip
    r.res(2, 0, "iron-ore", 50)
    r.op(1, "mine", 2, 0).op(560, "stop")
    rigs.append(r)
    # 5c. Room for exactly one item in a fresh slot: 79 wood, 1 empty.
    r = Rig("full_last_slot", (80, 80), 400, start=(0.5, 0.5), fill={"wood": 79})
    r.res(2, 0, "coal", 50)
    r.op(1, "mine", 2, 0).op(360, "stop")
    rigs.append(r)
    # 6. The tile under a burner drill (drill centred on the corner at (3, 1)).
    r = Rig("under_drill", (120, 80), 300, start=(0.5, 0.5))
    for dx in (2, 3):
        for dy in (0, 1):
            r.res(dx, dy, "iron-ore", 50)
    r.ent("burner-mining-drill", 3, 1, 8)
    r.op(1, "mine", 2, 0).op(250, "stop")
    rigs.append(r)
    # 6b. The ore tile covered by each 1x1 entity and by a 2x2 one: what the
    #     mod's selection-then-mine at the tile centre actually mines.
    covers = [("transport-belt", 2.5, 0.5), ("wooden-chest", 2.5, 0.5),
              ("burner-inserter", 2.5, 0.5), ("stone-wall", 2.5, 0.5),
              ("stone-furnace", 3, 1), ("item-on-ground", 2.5, 0.5)]  # fmt: skip
    for k, (name, x, y) in enumerate(covers):
        r = Rig(f"cover_{name}", (20 * k, 120), 200, start=(0.5, 0.5))
        for dx in (2, 3):
            for dy in (0, 1):
                r.res(dx, dy, "iron-ore", 50)
        r.ent(name, x, y, 0)
        r.op(1, "mine", 2, 0).op(180, "stop")
        rigs.append(r)
    # 6c. A pile lying on a belt over ore; a pile off the tile centre (where a
    #     south drill drops, 0.297 into the tile); retrying after a pickup.
    r = Rig("cover_pile_on_belt", (120, 120), 200, start=(0.5, 0.5))
    r.res(2, 0, "iron-ore", 50).ent("transport-belt", 2.5, 0.5, 0)
    r.ent("item-on-ground", 2.5, 0.5, 0)
    r.op(1, "mine", 2, 0).op(180, "stop")
    rigs.append(r)
    r = Rig("cover_pile_off_centre", (0, 160), 300, start=(0.5, 0.5))
    r.res(2, 0, "iron-ore", 50).ent("item-on-ground", 2.5, 0.296875, 0)
    r.op(1, "mine", 2, 0).op(280, "stop")
    rigs.append(r)
    r = Rig("cover_then_retry", (20, 160), 300, start=(0.5, 0.5))
    r.res(2, 0, "iron-ore", 50).ent("wooden-chest", 2.5, 0.5, 0)
    r.op(1, "mine", 2, 0).op(60, "stop").op(61, "mine", 2, 0).op(280, "stop")
    rigs.append(r)
    # 6d. The character's own selection box over the ore tile: standing on it,
    #     and just south of it (the box reaches 1.4 tiles up).
    for k, start in enumerate(((2.5, 0.5), (2.5, 1.3), (2.5, 1.7))):
        r = Rig(f"self_cover_{k}", (40 + 20 * k, 160), 160, start=start)
        r.res(2, 0, "iron-ore", 50)
        r.op(1, "mine", 2, 0).op(150, "stop")
        rigs.append(r)
    # 5d. A full inventory mining a pile of what it cannot take.
    r = Rig("full_mine_pile", (140, 120), 200, start=(0.5, 0.5), fill={"wood": 80})
    r.ent("item-on-ground", 2.5, 0.5, 0)
    r.op(1, "mine", 2, 0).op(180, "stop")
    rigs.append(r)
    # 7. Sub-tile start positions: first-item time does not depend on them.
    for k, start in enumerate(((0.1, 0.9), (0.73, 0.21))):
        r = Rig(f"subtile_{k}", (160 + 20 * k, 80), 200, start=start)
        r.res(2, 0, "stone", 50)
        r.op(1, "mine", 2, 0).op(150, "stop")
        rigs.append(r)
    return rigs


# ---------------------------------------------------------------- engine side

PRE = r"""
local s = game.surfaces[1]
local ch = storage.frrl_character
assert(ch and ch.valid, "no character")
local function paint(x0, y0, x1, y1)
  local cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
  local r = math.ceil(math.max(x1 - x0, y1 - y0) / 64) + 2
  s.request_to_generate_chunks({cx, cy}, r)
  s.force_generate_chunk_requests()
  for _, e in pairs(s.find_entities_filtered({area = {{x0, y0}, {x1, y1}}})) do
    if e.valid and e ~= ch then e.destroy() end
  end
  local tiles = {}
  for x = x0, x1 - 1 do
    for y = y0, y1 - 1 do tiles[#tiles + 1] = {name = "grass-1", position = {x, y}} end
  end
  s.set_tiles(tiles)
end
local function g(v) return string.format("%.17g", v) end
"""

MINE = r"""
storage.hm = {log = {}, failed = {}}
local P = storage.hm
local DIR = {north = defines.direction.north, east = defines.direction.east,
             south = defines.direction.south, west = defines.direction.west}
for _, rig in ipairs(RIGS) do
  paint(rig.base[1] - 8, rig.base[2] - 8, rig.base[1] + 12, rig.base[2] + 12)
end
local order, start = {}, 0
for k, rig in ipairs(RIGS) do
  order[k] = {rig = rig, t0 = start, byt = {}}
  for _, op in ipairs(rig.ops) do
    order[k].byt[op[1]] = order[k].byt[op[1]] or {}
    table.insert(order[k].byt[op[1]], op)
  end
  start = start + rig.ticks + 1
end
local cur = {k = 0, res = {}, mine = nil, walk = nil}
local function setup(k)
  local rig = RIGS[k]
  local bx, by = rig.base[1], rig.base[2]
  ch.walking_state = {walking = false}
  ch.mining_state = {mining = false}
  ch.teleport({bx + rig.start[1], by + rig.start[2]})
  local inv = ch.get_main_inventory()
  inv.clear()
  local slot = 1
  for name, n in pairs(rig.fill) do
    for _ = 1, n do
      inv[slot].set_stack({name = name, count = prototypes.item[name].stack_size})
      slot = slot + 1
    end
  end
  for name, n in pairs(rig.inventory) do
    inv[slot].set_stack({name = name, count = n})
    slot = slot + 1
  end
  cur.res = {}
  for _, r in ipairs(rig.resources) do
    local e = s.create_entity({name = r[3], position = {bx + r[1] + 0.5, by + r[2] + 0.5},
                               amount = r[4]})
    assert(e, "resource " .. r[3])
    cur.res[#cur.res + 1] = e
  end
  for _, en in ipairs(rig.entities) do
    local e
    if en[1] == "item-on-ground" then
      e = s.create_entity({name = "item-on-ground", position = {bx + en[2], by + en[3]},
                           stack = {name = "iron-plate", count = 1}})
    else
      e = s.create_entity({name = en[1], position = {bx + en[2], by + en[3]},
                           direction = en[4], force = "player"})
    end
    if not e then P.failed[#P.failed + 1] = {rig.name, en[1], en[2], en[3]} end
  end
  cur.mine, cur.walk, cur.last = nil, nil, nil
end
local function apply(k, op)
  local rig = RIGS[k]
  local kind = op[2]
  if kind == "mine" then
    cur.mine = {rig.base[1] + op[3] + 0.5, rig.base[2] + op[4] + 0.5}
  elseif kind == "stop" then
    cur.mine = nil
    ch.mining_state = {mining = false}
  elseif kind == "walk" then
    cur.walk = DIR[op[3]]
  elseif kind == "halt" then
    cur.walk = nil
    ch.walking_state = {walking = false}
  else
    error("op " .. kind)
  end
end
local function reading()
  local inv = ch.get_main_inventory()
  local counts = {}
  for _, name in ipairs(TRACKED) do counts[#counts + 1] = inv.get_item_count(name) end
  local amounts = {}
  for i, e in ipairs(cur.res) do amounts[i] = e.valid and e.amount or -1 end
  local piles = {}
  local bx, by = RIGS[cur.k].base[1], RIGS[cur.k].base[2]
  for _, e in pairs(s.find_entities_filtered({area = {{bx - 8, by - 8}, {bx + 12, by + 12}},
                                             type = "item-entity"})) do
    piles[#piles + 1] = {math.floor(e.position.x * 256 + 0.5) - bx * 256,
                         math.floor(e.position.y * 256 + 0.5) - by * 256,
                         e.stack.name, e.stack.count}
  end
  table.sort(piles, function(a, b) return a[1] < b[1] or (a[1] == b[1] and a[2] < b[2]) end)
  local built = {}
  for _, e in pairs(s.find_entities_filtered({area = {{bx - 8, by - 8}, {bx + 12, by + 12}},
                                             force = "player"})) do
    if e ~= ch then built[#built + 1] = e.name end
  end
  table.sort(built)
  local stats = ch.force.get_item_production_statistics(s)
  local prod = {}
  for _, name in ipairs(TRACKED) do prod[#prod + 1] = stats.get_input_count(name) end
  local ms = ch.mining_state
  return {p = {math.floor(ch.position.x * 256 + 0.5), math.floor(ch.position.y * 256 + 0.5)},
          m = ms.mining and true or false, g = g(ch.character_mining_progress),
          w = ch.walking_state.walking and true or false, inv = counts,
          e = inv.count_empty_stacks(), a = amounts,
          sel = (ch.selected and ch.selected.valid) and ch.selected.name or "",
          piles = piles, built = built, prod = prod}
end
P.t0 = game.tick + 1
local old = script.get_event_handler(defines.events.on_tick)
local body
script.on_event(defines.events.on_tick, function(ev)
  if old then old(ev) end
  if P.last == game.tick or P.err then return end
  P.last = game.tick
  local ok, msg = pcall(body)
  if not ok then P.err = tostring(msg) end
end)
body = function()
  local T = game.tick - P.t0
  for k, o in ipairs(order) do
    local t = T - o.t0
    if t >= 0 and t <= o.rig.ticks then
      if t == 0 then cur.k = k setup(k) end
      for _, op in ipairs(o.byt[t] or {}) do apply(k, op) end
      if cur.walk then ch.walking_state = {walking = true, direction = cur.walk} end
      if cur.mine then
        ch.update_selected_entity(cur.mine)
        ch.mining_state = {mining = true, position = cur.mine}
      end
      local key = helpers.table_to_json(reading())
      if key ~= cur.last then
        cur.last = key
        P.log[#P.log + 1] = {k, t, key}
      end
    end
  end
end
return start
"""

REACH = r"""
local out = {types = {}, sub = SUB, offsets = OFFSETS}
paint(-40, -40, 40, 40)
local cx, cy = 0, 0
for _, spec in ipairs(TYPES) do
  local name, x, y = spec[1], spec[2], spec[3]
  local e
  if name == "item-on-ground" then
    e = s.create_entity({name = "item-on-ground", position = {x, y},
                         stack = {name = "iron-plate", count = 1}})
  else
    e = s.create_entity({name = name, position = {x, y}, force = "player"})
  end
  assert(e, "create " .. name)
  local grid = {}
  for si = 0, SUB * SUB - 1 do
    local sx, sy = (si % SUB) / SUB, math.floor(si / SUB) / SUB
    local bits = {}
    for dy = -OFFSETS, OFFSETS do
      for dx = -OFFSETS, OFFSETS do
        local ok = ch.teleport({math.floor(e.position.x) + dx + sx,
                                math.floor(e.position.y) + dy + sy})
        if not ok then bits[#bits + 1] = "x"
        else bits[#bits + 1] = ch.can_reach_entity(e) and "1" or "0" end
      end
    end
    grid[#grid + 1] = table.concat(bits)
  end
  local rays = {}
  for _, ray in ipairs(RAYS) do
    local ux, uy, oy = ray[1], ray[2], ray[3]
    local last_true, first_false = nil, nil
    for step = 256 * 8, 256 * 13 do
      local d = step / 256
      ch.teleport({e.position.x + ux * d, e.position.y + uy * d + oy})
      local ok = ch.can_reach_entity(e)
      if ok then last_true = {g(ch.position.x), g(ch.position.y)}
      elseif not first_false then first_false = {g(ch.position.x), g(ch.position.y)} end
    end
    rays[#rays + 1] = {ray = ray, last_true = last_true, first_false = first_false}
  end
  local box = e.bounding_box
  out.types[#out.types + 1] = {
    name = name, position = {g(e.position.x), g(e.position.y)},
    bounding_box = {g(box.left_top.x), g(box.left_top.y), g(box.right_bottom.x),
                    g(box.right_bottom.y)},
    collision_box = {g(e.prototype.collision_box.left_top.x),
                     g(e.prototype.collision_box.right_bottom.x)},
    selection_box = {g(e.prototype.selection_box.left_top.x),
                     g(e.prototype.selection_box.right_bottom.x)},
    grid = grid, rays = rays}
  e.destroy()
end
out.character = {build_distance = ch.build_distance, reach_distance = ch.reach_distance,
                 resource_reach_distance = ch.resource_reach_distance,
                 item_pickup_distance = ch.item_pickup_distance,
                 drop_item_distance = ch.drop_item_distance,
                 mining_speed = ch.prototype.mining_speed,
                 character_mining_speed_modifier = ch.character_mining_speed_modifier}
out.resources = {}
for _, name in ipairs(RESOURCES) do
  local p = prototypes.entity[name]
  out.resources[name] = {mining_time = p.mineable_properties.mining_time,
                         collision_box = {g(p.collision_box.left_top.x),
                                          g(p.collision_box.right_bottom.x)}}
end
return helpers.table_to_json(out)
"""

#: Entity types and centres for the reach sweep (1x1 on a tile centre, 2x2 on a corner).
REACH_TYPES = [
    ["wooden-chest", 0.5, 0.5],
    ["transport-belt", 0.5, 0.5],
    ["burner-inserter", 0.5, 0.5],
    ["stone-wall", 0.5, 0.5],
    ["stone-furnace", 1.0, 1.0],
    ["burner-mining-drill", 1.0, 1.0],
    ["item-on-ground", 0.5, 0.5],
]
#: Rays: unit direction, and a sideways offset, from the entity's centre.
REACH_RAYS = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [1, 0, 0.3], [1, 0, 0.7],
              [0.7071067811865476, 0.7071067811865476, 0], [0.6, 0.8, 0]]  # fmt: skip


def _lua_value(v) -> str:
    if v is None:
        return "nil"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, (list, tuple)):
        return "{" + ", ".join(_lua_value(x) for x in v) + "}"
    if isinstance(v, dict):
        return "{" + ", ".join(f"[{json.dumps(k)}] = {_lua_value(x)}" for k, x in v.items()) + "}"
    raise TypeError(v)


def _world(label: str, body: str, ticks: int | None):
    sys.path.insert(0, str(ROOT / "src"))
    from factoriorl.engine_config import resolve_game_speed
    from factoriorl.env import FactorioEnv
    from factoriorl.rcon import RCONClient
    from factoriorl.seeding import Branch, SeedPlan
    from factoriorl.session import WorkerSession
    from factoriorl.tasks import get
    from factoriorl.worker import WorkerManager

    manager = WorkerManager()
    handle = manager.launch(label)
    try:
        with RCONClient(handle.spec.rcon_endpoint, timeout=3600.0) as client:
            client.lua(f"game.speed = {resolve_game_speed()} return game.speed")
            session = WorkerSession(handle, timeout=3600.0)
            session.status()
            env = FactorioEnv(get("construct_smelting_line"), session,
                              SeedPlan(master=41, run_id=label), branch=Branch.TRAIN,
                              split="train")  # fmt: skip
            env.reset(options={"scene_index": 0})
            result = client.lua(body)
            rows, failed, notes = None, None, None
            if ticks is not None:
                env.advance(int(result) + 2)
                err = client.lua("return storage.hm.err or ''")
                if err:
                    raise RuntimeError(f"probe Lua failed: {err}")
                failed = client.lua("return storage.hm.failed")
                notes = client.lua("return storage.hm.notes or {}")
                n = client.lua("return #storage.hm.log")
                rows = []
                for a in range(1, n + 1, 200):
                    rows += client.lua(
                        f"local o = {{}} for i = {a}, math.min({a + 199}, #storage.hm.log) do "
                        "o[#o + 1] = storage.hm.log[i] end return o"
                    )
            session.close()
    finally:
        manager.cleanup(handle)
    return result, rows, failed, notes


def run_mine() -> dict:
    rigs = [r.to_dict() for r in mine_rigs()]
    body = (PRE + "local TRACKED = " + _lua_value(list(TRACKED)) + "\n"
            + "local RIGS = " + _lua_value(rigs) + "\n" + MINE)  # fmt: skip
    _, rows, failed, _ = _world("probe-handmine-mine", body, ticks=1)
    logs: dict = {r["name"]: [] for r in rigs}
    for k, t, key in rows:
        logs[rigs[k - 1]["name"]].append([t, json.loads(key)])
    return {
        "rigs": {r["name"]: r for r in rigs},
        "tracked": list(TRACKED),
        "log": logs,
        "create_failed": failed or [],
    }


def run_reach(sub: int, offsets: int) -> dict:
    body = (PRE + f"local SUB, OFFSETS = {sub}, {offsets}\n"
            + "local TYPES = " + _lua_value(REACH_TYPES) + "\n"
            + "local RAYS = " + _lua_value(REACH_RAYS) + "\n"
            + "local RESOURCES = " + _lua_value(list(RESOURCES)) + "\n" + REACH)  # fmt: skip
    result, _, _, _ = _world("probe-handmine-reach", body, ticks=None)
    return json.loads(result)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--family", choices=("mine", "reach"), required=True)
    ap.add_argument("--sub", type=int, default=16, help="reach: sub-tile steps per axis")
    ap.add_argument("--offsets", type=int, default=12, help="reach: whole-tile offset bound")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    started = time.perf_counter()
    doc = run_mine() if args.family == "mine" else run_reach(args.sub, args.offsets)
    from factoriorl.worker import WorkerManager

    doc["engine"] = WorkerManager().engine.to_dict()
    doc["wall_seconds"] = round(time.perf_counter() - started, 1)
    out = Path(args.out) if args.out else EVIDENCE / f"handmine-{args.family}.json.xz"
    out.write_bytes(lzma.compress(json.dumps(doc, sort_keys=True).encode()))
    print(f"wrote {out} ({out.stat().st_size // 1024} KiB), {doc['wall_seconds']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
