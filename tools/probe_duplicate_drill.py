"""Can two burner mining drills end up on one tile? Measured, not argued.

The transfer replay found the engine reporting two drills with distinct
handles at exactly (0, -2), immediately after a placement, where the simulator
had one. Factorio will not let a player do that, so either `can_place_entity`
is answering a question `create_entity` does not then ask, or the placement is
being applied twice.

This asks the engine directly, in four steps:

1. place a drill on ore;
2. ask `can_place_entity` for a second drill at the same position, with the
   same `build_check_type` the action handler uses;
3. call `create_entity` anyway, as the handler does;
4. count what is actually on the tile.

    uv run python tools/probe_duplicate_drill.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from factoriorl.engine_config import resolve_game_speed  # noqa: E402
from factoriorl.rcon import RCONClient  # noqa: E402
from factoriorl.worker import WorkerManager  # noqa: E402

OUT = ROOT / "docs" / "evidence" / "duplicate-drill.json"

#: Clear the area, lay ore, and put one drill down the way `actions.place` does.
SETUP = """
local srf = game.surfaces[1]
for _, e in pairs(srf.find_entities_filtered({ area = {{-8,-8},{8,8}} })) do
  if e.valid and e.type ~= "character" then e.destroy() end
end
for x = -3, 3 do for y = -3, 3 do
  if srf.can_place_entity({ name = "iron-ore", position = {x + 0.5, y + 0.5} }) then
    srf.create_entity({ name = "iron-ore", position = {x + 0.5, y + 0.5}, amount = 2000 })
  end
end end
local first = srf.create_entity({
  name = "burner-mining-drill", position = {0, -2},
  direction = defines.direction.south, force = "player" })
return helpers.table_to_json({ placed = first ~= nil and first.valid })
"""

#: The question the action handler asks, then the call it makes regardless.
ATTEMPT = """
local srf = game.surfaces[1]
local spec = {
  name = "burner-mining-drill", position = {0, -2},
  direction = defines.direction.south, force = "player",
  build_check_type = defines.build_check_type.manual,
}
local allowed = srf.can_place_entity(spec)
local ghost_allowed = srf.can_place_entity({
  name = "burner-mining-drill", position = {0, -2},
  direction = defines.direction.south, force = "player",
  build_check_type = defines.build_check_type.ghost_place,
})
local second = srf.create_entity({
  name = "burner-mining-drill", position = {0, -2},
  direction = defines.direction.south, force = "player" })
local drills = srf.find_entities_filtered({
  area = {{-2, -4}, {2, 0}}, name = "burner-mining-drill" })
local where = {}
for i, d in pairs(drills) do
  where[i] = { x = d.position.x, y = d.position.y, unit = d.unit_number, valid = d.valid }
end
return helpers.table_to_json({
  can_place_manual = allowed,
  can_place_ghost = ghost_allowed,
  create_entity_returned = second ~= nil and second.valid,
  drills_on_the_tile = #drills,
  drills = where,
})
"""


#: Does `create_entity` put the drill where it was asked, or snap it onto a
#: grid -- and if it snaps, does `can_place_entity` answer for the asked
#: position or the snapped one? A 2x2 entity only sits on even tile boundaries.
SNAP = """
local srf = game.surfaces[1]
local box_all = { {-8, -8}, {8, 8} }
for _, e in pairs(srf.find_entities_filtered({ area = box_all, name = "burner-mining-drill" })) do
  e.destroy()
end
local first = srf.create_entity({
  name = "burner-mining-drill", position = {0, -2},
  direction = defines.direction.south, force = "player" })
local out = {}
for i, p in pairs({ {0.5, -1.5}, {0.4, -2.4}, {1, -2}, {0.5, -2.5} }) do
  local spec = {
    name = "burner-mining-drill", position = p,
    direction = defines.direction.south, force = "player",
    build_check_type = defines.build_check_type.manual,
  }
  local allowed = srf.can_place_entity(spec)
  local made = nil
  if allowed then
    local e = srf.create_entity({
      name = "burner-mining-drill", position = p,
      direction = defines.direction.south, force = "player" })
    if e and e.valid then
      made = { x = e.position.x, y = e.position.y }
      e.destroy()
    end
  end
  out[i] = { asked = { x = p[1], y = p[2] }, can_place = allowed, landed_at = made }
end
out.first_at = { x = first.position.x, y = first.position.y }
return helpers.table_to_json(out)
"""


#: The guard `actions.place` now runs after `create_entity`. Does it see the
#: stacked entity? The bounding box is shrunk so that a neighbour merely
#: touching the edge does not read as an overlap, and resources and loose items
#: are skipped because neither blocks a build.
GUARD = """
local srf = game.surfaces[1]
local box_all = { {-8, -8}, {8, 8} }
for _, e in pairs(srf.find_entities_filtered({ area = box_all, name = "burner-mining-drill" })) do
  e.destroy()
end
local first = srf.create_entity({
  name = "burner-mining-drill", position = {0, -2},
  direction = defines.direction.south, force = "player" })
local created = srf.create_entity({
  name = "burner-mining-drill", position = {0, -2},
  direction = defines.direction.south, force = "player" })
local box = created.bounding_box
local shrunk = {
  { box.left_top.x + 0.05, box.left_top.y + 0.05 },
  { box.right_bottom.x - 0.05, box.right_bottom.y - 0.05 },
}
local caught, by = false, nil
for _, other in pairs(srf.find_entities_filtered({ area = shrunk })) do
  if other.valid and other ~= created and other.type ~= "character"
    and other.type ~= "item-entity" and other.type ~= "resource" then
    caught, by = true, other.name
  end
end
-- And the other half of the guard: a drill placed legally beside another must
-- NOT read as an overlap, or every honest placement would be rolled back.
local neighbour = srf.create_entity({
  name = "burner-mining-drill", position = {2, -2},
  direction = defines.direction.south, force = "player" })
local nbox = neighbour.bounding_box
local nshrunk = {
  { nbox.left_top.x + 0.05, nbox.left_top.y + 0.05 },
  { nbox.right_bottom.x - 0.05, nbox.right_bottom.y - 0.05 },
}
local false_alarm = false
for _, other in pairs(srf.find_entities_filtered({ area = nshrunk })) do
  if other.valid and other ~= neighbour and other.type ~= "character"
    and other.type ~= "item-entity" and other.type ~= "resource" then
    false_alarm = true
  end
end
return helpers.table_to_json({
  guard_catches_the_stack = caught,
  occupied_by = by,
  guard_allows_a_neighbour = not false_alarm,
})
"""


#: `place_at` offers tile centres and a 2x2 entity snaps to the integer centre
#: one up, so `can_place_entity` may be judging a different box than the one
#: `create_entity` then occupies. Sweep the neighbourhood of a standing drill
#: and look for a position the check allows and the creation overlaps.
SWEEP = """
local srf = game.surfaces[1]
local box_all = { {-8, -8}, {8, 8} }
for _, e in pairs(srf.find_entities_filtered({ area = box_all, name = "burner-mining-drill" })) do
  e.destroy()
end
local first = srf.create_entity({
  name = "burner-mining-drill", position = {0, -2},
  direction = defines.direction.south, force = "player" })
local fb = first.bounding_box
local out, n = {}, 0
for gx = -3, 3 do
  for gy = -5, 1 do
    local p = { gx + 0.5, gy + 0.5 }
    local allowed = srf.can_place_entity({
      name = "burner-mining-drill", position = p,
      direction = defines.direction.south, force = "player",
      build_check_type = defines.build_check_type.manual })
    if allowed then
      local e = srf.create_entity({
        name = "burner-mining-drill", position = p,
        direction = defines.direction.south, force = "player" })
      if e and e.valid then
        local b = e.bounding_box
        local overlaps = b.left_top.x < fb.right_bottom.x and b.right_bottom.x > fb.left_top.x
                     and b.left_top.y < fb.right_bottom.y and b.right_bottom.y > fb.left_top.y
        if overlaps then
          n = n + 1
          out[n] = { asked = { p[1], p[2] }, landed = { e.position.x, e.position.y } }
        end
        e.destroy()
      end
    end
  end
end
return helpers.table_to_json({
  drill_at = { first.position.x, first.position.y },
  allowed_but_overlapping = out,
  how_many = n,
})
"""


#: The request the guard actually caught in a live episode, replayed exactly.
#: The earlier sweep missed it because it only ever asked facing south.
BY_DIRECTION = """
local srf = game.surfaces[1]
local box_all = { {-8, -8}, {8, 8} }
for _, e in pairs(srf.find_entities_filtered({ area = box_all, name = "burner-mining-drill" })) do
  e.destroy()
end
local first = srf.create_entity({
  name = "burner-mining-drill", position = {0, -2},
  direction = defines.direction.south, force = "player" })
local out = {}
local dirs = {
  north = defines.direction.north, east = defines.direction.east,
  south = defines.direction.south, west = defines.direction.west,
}
for name, d in pairs(dirs) do
  local spec = {
    name = "burner-mining-drill", position = {-0.5, -2.5},
    direction = d, force = "player",
    build_check_type = defines.build_check_type.manual,
  }
  local allowed = srf.can_place_entity(spec)
  local landed = nil
  if allowed then
    local e = srf.create_entity({
      name = "burner-mining-drill", position = {-0.5, -2.5},
      direction = d, force = "player" })
    if e and e.valid then
      landed = { e.position.x, e.position.y }
      e.destroy()
    end
  end
  out[name] = { can_place = allowed, landed_at = landed }
end
return helpers.table_to_json({ standing_drill = { first.position.x, first.position.y }, asked = { -0.5, -2.5 }, by_direction = out })
"""


def main() -> int:
    manager = WorkerManager()
    handle = manager.launch("duplicate-drill")
    report: dict = {
        "about": "whether can_place_entity and create_entity agree about a second drill",
        "engine": {k: v for k, v in manager.engine.to_dict().items() if k != "executable"},
    }
    try:
        with RCONClient(handle.spec.rcon_endpoint, timeout=60.0) as client:
            client.lua(f"game.speed = {resolve_game_speed()} return game.speed")
            report["setup"] = json.loads(client.lua(SETUP))
            report["attempt"] = json.loads(client.lua(ATTEMPT))
            report["snap"] = json.loads(client.lua(SNAP))
            report["guard"] = json.loads(client.lua(GUARD))
            report["sweep"] = json.loads(client.lua(SWEEP))
            report["by_direction"] = json.loads(client.lua(BY_DIRECTION))
    finally:
        manager.cleanup(handle)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["by_direction"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
