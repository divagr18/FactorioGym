"""Measure the inventory facts the v3 per-operation masks rest on, and taking fuel.

User decisions (docs/sim-logistics.md, 2026-09-25): v3 masks per operation
(option C), which need the character's slot count, stack sizes and what each
inventory accepts, read from the running engine rather than typed in; and
taking fuel out of a machine's fuel slot, measured before it is offered.

Families, one evidence file each (`docs/evidence/inventory-<family>.json.xz`):

- `prototypes`: for every `ITEMS_V3` item its stack size, fuel value and
  category, and what it places; for every entity it places, whether it takes a
  direction and turns (`LuaEntity.rotate`), its inventories and their sizes,
  whether each inventory accepts one of each item (`LuaInventory.can_insert`),
  and what mining it gives; the same for a ground pile; the character's main
  inventory size.
- `fuel`: drills, furnaces, inserters and a boiler, fuelled and running, with
  fuel taken out through the mod's own transfer (`actions.lua`, `H.transfer`:
  the first inventory that holds the item, `min(count, available)`, then an
  insert into the character's main inventory and whatever did not fit put
  back), mid-burn and after; and transfers with part of the room they need,
  both ways. Every tick whose reading changed is logged.

Run (under a minute each):
  uv run python tools/probe_inventory.py --family prototypes
  uv run python tools/probe_inventory.py --family fuel
"""

from __future__ import annotations

import argparse
import json
import lzma
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs" / "evidence"
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

from probe_handmine import PRE, _lua_value, _world  # noqa: E402

from factoriorl.encoders import ITEMS_V3  # noqa: E402

#: The inventories the mod's transfer tries, in its order (`actions.lua`).
INVENTORIES = (
    "chest",
    "fuel",
    "furnace_source",
    "furnace_result",
    "assembling_machine_input",
    "assembling_machine_output",
)

PROTOTYPES = r"""
paint(-60, -60, 60, 60)
local out = {items = {}, entities = {}, inventories = INVENTORIES}
local main = ch.get_main_inventory()
out.character = {main_slots = #main, empty_stacks = main.count_empty_stacks()}
for _, name in ipairs(ITEMS) do
  local p = prototypes.item[name]
  local rec = {stack_size = p.stack_size, fuel_value = p.fuel_value,
               fuel_category = p.fuel_category}
  if p.place_result then
    rec.place_result = p.place_result.name
    rec.place_type = p.place_result.type
  end
  out.items[name] = rec
end
local x = -50
local function describe(e)
  local rec = {name = e.name, type = e.type, supports_direction = e.supports_direction,
               inventories = {}}
  if e.supports_direction then
    local before = e.direction
    rec.rotated = e.rotate() and true or false
    rec.direction_after = e.direction
    if rec.rotated then e.rotate({reverse = true}) end
    rec.direction_back = e.direction
    rec.direction_before = before
  else
    rec.rotated = e.rotate() and true or false
  end
  -- `defines.inventory` gives one number several names (chest and fuel are
  -- both 1, furnace_source and assembling_machine_input 2, ...), so an
  -- inventory is recorded once, by its index, with every name that reached it.
  local by_index = {}
  for _, which in ipairs(INVENTORIES) do
    local ok, inv = pcall(function() return e.get_inventory(defines.inventory[which]) end)
    if ok and inv and #inv > 0 then
      local key = tostring(inv.index)
      if by_index[key] then
        table.insert(by_index[key].names, which)
      else
        local accepts, capacity = {}, {}
        for _, item in ipairs(ITEMS) do
          if inv.can_insert({name = item, count = 1}) then
            accepts[#accepts + 1] = item
            -- How many one insert puts into the empty inventory.
            capacity[item] = inv.insert({name = item, count = 100000})
            inv.clear()
          end
        end
        by_index[key] = {index = inv.index, names = {which}, size = #inv, accepts = accepts,
                         capacity = capacity}
      end
    end
  end
  rec.inventories = by_index
  local m = e.prototype.mineable_properties
  rec.minable = m and m.minable or false
  if m and m.products and m.products[1] then
    rec.mines_to = m.products[1].name
    rec.mining_time = m.mining_time
  end
  rec.burner = e.burner ~= nil
  return rec
end
for _, name in ipairs(ITEMS) do
  local p = prototypes.item[name]
  if p.place_result then
    local e = s.create_entity({name = p.place_result.name, position = {x, 0}, force = "player"})
    if e then
      out.entities[p.place_result.name] = describe(e)
      e.destroy()
    else
      out.entities[p.place_result.name] = {create_failed = true}
    end
    x = x + 8
  end
end
local pile = s.create_entity({name = "item-on-ground", position = {x, 0},
                              stack = {name = "iron-plate", count = 1}})
out.entities["item-on-ground"] = describe(pile)
pile.destroy()
return helpers.table_to_json(out)
"""

FUEL = r"""
storage.hm = {log = {}, failed = {}, notes = {}}
local P = storage.hm
local main = ch.get_main_inventory()
local CAND = {}
for i, which in ipairs(INVENTORIES) do CAND[i] = {which, defines.inventory[which]} end
local function inventory_of(e, item, removing)
  local first, first_name = nil, nil
  for _, c in ipairs(CAND) do
    local ok, inv = pcall(function() return e.get_inventory(c[2]) end)
    if ok and inv then
      first, first_name = first or inv, first_name or c[1]
      if removing then
        if inv.get_item_count(item) > 0 then return inv, c[1] end
      elseif inv.can_insert({name = item, count = 1}) then
        return inv, c[1]
      end
    end
  end
  return first, first_name
end
-- actions.lua `H.transfer`, from `from` to `to` (nil: the character).
local function transfer(from, to, item, count)
  local fi, fn, ti, tn
  if from then fi, fn = inventory_of(from, item, true) else fi, fn = main, "main" end
  if to then ti, tn = inventory_of(to, item, false) else ti, tn = main, "main" end
  local r = {from = fn, to = tn, item = item, count = count,
             from_index = fi and fi.index or nil, to_index = ti and ti.index or nil}
  if not fi or not ti then r.status = "rejected" r.code = "no_inventory" return r end
  r.available = fi.get_item_count(item)
  if r.available <= 0 then r.status = "rejected" r.code = "no_items" return r end
  r.wanted = math.min(count, r.available)
  r.can_insert = ti.can_insert({name = item, count = r.wanted})
  if not r.can_insert then r.status = "rejected" r.code = "no_space" return r end
  r.removed = fi.remove({name = item, count = r.wanted})
  r.inserted = ti.insert({name = item, count = r.removed})
  if r.inserted < r.removed then
    r.put_back = fi.insert({name = item, count = r.removed - r.inserted})
    r.status = "rejected" r.code = "no_space"
  else
    r.status = "completed"
  end
  return r
end
local rigs = {}
for k, rig in ipairs(RIGS) do
  local bx, by = rig.base[1], rig.base[2]
  paint(bx - 6, by - 6, bx + 10, by + 10)
  local R = {rig = rig, E = {}, last = nil}
  for _, r in ipairs(rig.resources) do
    s.create_entity({name = "iron-ore", position = {bx + r[1] + 0.5, by + r[2] + 0.5},
                     amount = 10000})
  end
  for _, en in ipairs(rig.entities) do
    local e = s.create_entity({name = en.name, position = {bx + en.x, by + en.y},
                               direction = en.d, force = "player"})
    if not e then
      P.failed[#P.failed + 1] = {rig.name, en.label}
    else
      for which, stacks in pairs(en.inv or {}) do
        local inv = e.get_inventory(defines.inventory[which])
        for _, st in ipairs(stacks) do inv.insert({name = st[1], count = st[2]}) end
      end
      R.E[en.label] = e
    end
  end
  rigs[k] = R
end
local function contents(inv)
  if not inv then return nil end
  local out = {}
  for i = 1, #inv do
    local st = inv[i]
    if st.valid_for_read then out[#out + 1] = {i, st.name, st.count} end
  end
  return out
end
local function g(v) return string.format("%.17g", v) end
local function reading(R)
  local ents = {}
  for label, e in pairs(R.E) do
    if e.valid then
      local rec = {st = e.status, inv = {}}
      for _, c in ipairs(CAND) do
        local ok, inv = pcall(function() return e.get_inventory(c[2]) end)
        if ok and inv then rec.inv[c[1]] = contents(inv) end
      end
      if e.burner then
        local b = e.burner
        local cb = b.currently_burning
        rec.burning = cb and (cb.name.name or cb.name) or nil
        rec.remaining = g(b.remaining_burning_fuel)
        rec.heat = g(b.heat)
      end
      local ok_p, prog = pcall(function() return e.mining_progress end)
      if ok_p and prog then rec.mp = g(prog) end
      local ok_c, cp = pcall(function() return e.crafting_progress end)
      if ok_c and cp then rec.cp = g(cp) end
      ents[label] = rec
    else
      ents[label] = "gone"
    end
  end
  local counts = {}
  for _, name in ipairs(TRACKED) do counts[#counts + 1] = main.get_item_count(name) end
  return {ents = ents, main = counts, empty = main.count_empty_stacks()}
end
local function set_main(spec)
  main.clear()
  local slot = 1
  for _, st in ipairs(spec) do
    main[slot].set_stack({name = st[1], count = st[2]})
    slot = slot + 1
  end
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
  local t = game.tick - P.t0
  for k, R in ipairs(rigs) do
    for _, op in ipairs(R.rig.ops) do
      if op[1] == t then
        set_main(op[6])
        local r
        if op[2] == "take" then
          r = transfer(R.E[op[3]], nil, op[4], op[5])
        else
          r = transfer(nil, R.E[op[3]], op[4], op[5])
        end
        r.main_after = {}
        for _, name in ipairs(TRACKED) do r.main_after[name] = main.get_item_count(name) end
        P.notes[#P.notes + 1] = {R.rig.name, t, op[2], r}
      end
    end
    if t <= R.rig.ticks then
      local key = helpers.table_to_json(reading(R))
      if key ~= R.last then
        R.last = key
        P.log[#P.log + 1] = {k, t, key}
      end
    end
  end
end
return TOTAL
"""

TRACKED = ("coal", "wood", "iron-ore", "iron-plate", "stone")
S = 8  # south
E = 4
N = 0


def wood(n: int) -> list:
    return [["wood", 100]] * n


def fuel_rigs() -> list[dict]:
    """Every rig runs at once, each on its own ground; a transfer sets the
    character's main inventory first (it is shared), and rig k's ticks are
    offset by k so no two rigs touch it on one tick."""
    rigs = []

    def rig(name, ticks, entities, ops, resources=()):
        k = len(rigs)
        ops = [[t + k, *rest] for t, *rest in ops]
        rigs.append({"name": name, "base": [-400 + 40 * (k % 10), -400 + 40 * (k // 10)],
                     "ticks": ticks + k, "entities": entities, "ops": ops,
                     "resources": list(resources)})  # fmt: skip

    def ent(label, name, x, y, d=N, **inv):
        return {"label": label, "name": name, "x": x, "y": y, "d": d, "inv": inv}

    ore = [(dx, dy) for dx in (0, 1) for dy in (0, 1)]
    room = wood(79)  # one slot free
    # A drill mining into a chest: fuel taken mid-burn, then more than is left,
    # then from an empty slot while the last coal burns.
    rig("drill", 1800,
        [ent("drill", "burner-mining-drill", 1, 1, S, fuel=[["coal", 5]]),
         ent("out", "wooden-chest", 0.5, 2.5)],
        [[30, "take", "drill", "coal", 1, room], [90, "take", "drill", "coal", 20, room],
         [120, "take", "drill", "coal", 1, room]], ore)  # fmt: skip
    rig("furnace", 2900,
        [ent("furnace", "stone-furnace", 1, 1, fuel=[["coal", 5]],
             furnace_source=[["iron-ore", 10]])],
        [[30, "take", "furnace", "coal", 1, room], [90, "take", "furnace", "coal", 20, room],
         [120, "take", "furnace", "iron-ore", 1, room],
         [150, "take", "furnace", "coal", 1, room]])  # fmt: skip
    rig(
        "inserter",
        2400,
        [
            ent("src", "wooden-chest", 0.5, 0.5, chest=[["iron-plate", 50]]),
            ent("ins", "burner-inserter", 1.5, 0.5, 12, fuel=[["coal", 2]]),
            ent("dst", "wooden-chest", 2.5, 0.5),
        ],
        [[30, "take", "ins", "coal", 1, room], [90, "take", "ins", "coal", 20, room]],
    )
    rig(
        "inserter_wood",
        600,
        [ent("ins", "burner-inserter", 0.5, 0.5, fuel=[["wood", 5]])],
        [[30, "take", "ins", "wood", 5, room], [60, "take", "ins", "wood", 5, room]],
    )
    # As built by script, no fuel given: nothing to take.
    rig(
        "inserter_unfuelled",
        120,
        [ent("ins", "burner-inserter", 0.5, 0.5)],
        [[30, "take", "ins", "wood", 1, room], [60, "take", "ins", "coal", 1, room]],
    )
    # A boiler, no water.
    rig(
        "boiler",
        300,
        [ent("boiler", "boiler", 1.5, 1, fuel=[["coal", 5]])],
        [[30, "take", "boiler", "coal", 1, room], [90, "take", "boiler", "coal", 20, room]],
    )
    # One coal: it starts burning at once and the slot is empty.
    rig(
        "furnace_one_coal",
        2900,
        [
            ent(
                "furnace",
                "stone-furnace",
                1,
                1,
                fuel=[["coal", 1]],
                furnace_source=[["iron-ore", 20]],
            )
        ],
        [[30, "take", "furnace", "coal", 1, room]],
    )
    # Part of the room: a fuel slot with room for 2, then full (the next
    # inventory that accepts coal is the result slot).
    rig(
        "give_partial",
        120,
        [ent("furnace", "stone-furnace", 1, 1, fuel=[["coal", 48]])],
        [
            [30, "give", "furnace", "coal", 5, [["coal", 50]]],
            [60, "give", "furnace", "coal", 20, [["coal", 50]]],
        ],
    )
    # A character with room for 3 coal (79 slots of wood, 47 coal), then none,
    # then room for 1 asked for 20.
    rig(
        "take_partial",
        150,
        [ent("chest", "wooden-chest", 0.5, 0.5, chest=[["coal", 30]])],
        [
            [30, "take", "chest", "coal", 5, [*wood(79), ["coal", 47]]],
            [60, "take", "chest", "coal", 1, wood(80)],
            [90, "take", "chest", "coal", 20, [*wood(79), ["coal", 49]]],
            [120, "take", "chest", "coal", 1, [*wood(79), ["coal", 49]]],
        ],
    )
    rig(
        "take_fuel_partial",
        150,
        [ent("drill", "burner-mining-drill", 1, 1, S, fuel=[["coal", 10]])],
        [[30, "take", "drill", "coal", 5, [*wood(79), ["coal", 47]]]],
        ore,
    )
    return rigs


def run_prototypes() -> dict:
    body = (PRE + "local ITEMS = " + _lua_value(list(ITEMS_V3)) + "\n"
            + "local INVENTORIES = " + _lua_value(list(INVENTORIES)) + "\n"
            + PROTOTYPES)  # fmt: skip
    result, _, _, _ = _world(f"probe-inv-proto-{os.getpid()}", body, ticks=None)
    return json.loads(result)


def run_fuel() -> dict:
    rigs = fuel_rigs()
    total = max(r["ticks"] for r in rigs) + 2
    body = (PRE + "local TRACKED = " + _lua_value(list(TRACKED)) + "\n"
            + "local INVENTORIES = " + _lua_value(list(INVENTORIES)) + "\n"
            + "local RIGS = " + _lua_value(rigs) + "\n"
            + f"local TOTAL = {total}\n" + FUEL)  # fmt: skip
    _, rows, failed, notes = _world(f"probe-inv-fuel-{os.getpid()}", body, ticks=1)
    logs: dict = {r["name"]: [] for r in rigs}
    for k, t, key in rows:
        logs[rigs[k - 1]["name"]].append([t, json.loads(key)])
    return {
        "rigs": {r["name"]: r for r in rigs},
        "tracked": list(TRACKED),
        "inventories": list(INVENTORIES),
        "log": logs,
        "transfers": notes or [],
        "create_failed": failed or [],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--family", choices=("prototypes", "fuel"), required=True)
    args = ap.parse_args()
    started = time.perf_counter()
    data = run_prototypes() if args.family == "prototypes" else run_fuel()
    data["wall_seconds"] = round(time.perf_counter() - started, 1)
    path = EVIDENCE / f"inventory-{args.family}.json.xz"
    payload = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    path.write_bytes(lzma.compress(payload, preset=9))
    print(f"wrote {path} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
