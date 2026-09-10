"""What tiles does a placement actually cover, and what does a drill mine?

Run 7 placed a burner mining drill on the east edge of the iron patch. It asked
for `[-28.5, 0.5]`; the engine produced a machine covering `x -29..-28,
y 0..1`, of which exactly one tile held ore. The drill reported
`no_minable_resources` within seconds.

The agent could not have avoided that. The placement domain offers tile
centres, a burner drill is 2x2, and *nothing in the observation says which way
the other three tiles extend*. Nor does anything say that a mining drill mines
only what lies under its own footprint -- so "put the drill on an ore tile"
looks sufficient and is not.

Both facts are ordinary: a player holding a drill sees the build ghost and the
highlighted ore under it. Neither is a layout, a coordinate or a solution.

This measures rather than assumes, because the anchoring rule is exactly the
kind of thing that is easy to infer from two samples and get wrong for odd
sizes. For every placeable item in the open catalog it asks the engine:

  - the prototype's tile footprint;
  - the tiles a real entity built at a named position actually occupies, for a
    request on a tile centre;
  - for mining drills, the tile area the drill searches for resources.

Run:
  uv run python tools/probe_footprints.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from factoriorl import worlds  # noqa: E402
from factoriorl.rcon import RCONClient  # noqa: E402
from factoriorl.session import WorkerSession  # noqa: E402
from factoriorl.worker import WorkerManager  # noqa: E402

#: One of each shape the open catalog can place, chosen to cover odd and even
#: tile sizes -- the anchoring rule is only interesting where they differ.
ITEMS = [
    "burner-inserter",
    "stone-furnace",
    "burner-mining-drill",
    "wooden-chest",
    "iron-chest",
    "transport-belt",
    "electric-mining-drill",
    "assembling-machine-1",
    "boiler",
    "offshore-pump",
    "lab",
]

#: The tile centres a placement is offered at. Two of them, so a rule derived
#: from one cannot be an accident of that one.
POSITIONS = [(10.5, 10.5), (-7.5, -3.5)]

LUA = """
local out = {}
local surface = game.surfaces[1]
local force = game.forces.player
for _, name in ipairs(%s) do
  local proto = prototypes.entity[name]
  local row = { name = name }
  if proto == nil then
    row.missing = true
  else
    row.tile_width = proto.tile_width
    row.tile_height = proto.tile_height
    local ok_radius, radius = pcall(function() return proto.mining_drill_radius end)
    if ok_radius and radius then row.mining_drill_radius = radius end
    row.placements = {}
    for _, at in ipairs(%s) do
      local built = surface.create_entity({
        name = name, position = { at[1], at[2] }, force = force,
        direction = defines.direction.north, raise_built = false,
      })
      local entry = { asked = { at[1], at[2] } }
      if built == nil then
        entry.refused = true
      else
        local box = built.bounding_box
        entry.position = { built.position.x, built.position.y }
        entry.covers = {
          math.floor(box.left_top.x), math.floor(box.left_top.y),
          math.ceil(box.right_bottom.x) - 1, math.ceil(box.right_bottom.y) - 1,
        }
        local ok_area, area = pcall(function() return built.mining_target end)
        if row.mining_drill_radius then
          -- The tile square the drill searches, from the engine's own radius
          -- rather than from the bounding box: for a burner drill they agree
          -- and for an electric one they do not.
          local r = row.mining_drill_radius
          entry.mines = {
            math.floor(built.position.x - r), math.floor(built.position.y - r),
            math.ceil(built.position.x + r) - 1, math.ceil(built.position.y + r) - 1,
          }
        end
        if ok_area and area then entry.mining_target = area.name end
        built.destroy()
      end
      row.placements[#row.placements + 1] = entry
    end
  end
  out[#out + 1] = row
end
return helpers.table_to_json(out)
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--out", default=str(ROOT / "docs" / "evidence" / "footprints.json"))
    args = parser.parse_args()

    mode = worlds.get("open_factory")
    manager = WorkerManager()
    report: dict = {
        "measures": "which tiles a placement covers, and what a mining drill mines",
        "engine": manager.engine.to_dict(),
    }
    started = time.perf_counter()

    handle = manager.launch("probe-fp", map_seed=args.seed, terrain=mode.terrain)
    try:
        items = "{" + ",".join(f'"{name}"' for name in ITEMS) + "}"
        positions = "{" + ",".join(f"{{{x},{y}}}" for x, y in POSITIONS) + "}"
        with RCONClient(handle.spec.rcon_endpoint, timeout=90.0) as client:
            raw = client.lua(LUA % (items, positions))
        report["rows"] = json.loads(raw.strip())

        # And what the agent is actually told. `knowledge.placeable()` derives
        # the same footprint from `collision_box` and the parity rule rather
        # than by building anything -- deriving is the only option across ~200
        # prototypes -- so the derivation is checked against the builds above.
        # A reference table that disagrees with the engine is worse than no
        # table: this one used to say "footprint centred on the placement
        # position", which is false for every even-sized machine, and that is
        # what put run 7's drill on the edge of the patch.
        session = WorkerSession(handle, timeout=60.0)
        session.status()
        snapshot = session.knowledge().response.result or {}
        published = {str(row.get("name")): row for row in (snapshot.get("placeable") or [])}
        session.close()
    finally:
        manager.cleanup(handle)

    mismatches = []
    for row in report["rows"]:
        told = published.get(row["name"])
        if told is None:
            mismatches.append({"name": row["name"], "problem": "not published at all"})
            continue
        for entry in row.get("placements") or []:
            covers, asked = entry.get("covers"), entry.get("asked")
            if not covers or not asked:
                continue
            named = (int(asked[0] // 1), int(asked[1] // 1))
            offsets = [
                covers[0] - named[0],
                covers[1] - named[1],
                covers[2] - named[0],
                covers[3] - named[1],
            ]
            if list(told.get("covers") or []) != offsets:
                mismatches.append(
                    {
                        "name": row["name"],
                        "field": "covers",
                        "published": told.get("covers"),
                        "engine": offsets,
                    }
                )
            if entry.get("mines"):
                mined = [
                    entry["mines"][0] - named[0],
                    entry["mines"][1] - named[1],
                    entry["mines"][2] - named[0],
                    entry["mines"][3] - named[1],
                ]
                if list(told.get("mines") or []) != mined:
                    mismatches.append(
                        {
                            "name": row["name"],
                            "field": "mines",
                            "published": told.get("mines"),
                            "engine": mined,
                        }
                    )
    sampled = {row["name"] for row in report["rows"]}
    report["published"] = {name: published[name] for name in published if name in sampled}
    report["derivation_mismatches"] = mismatches

    # The rule, derived from the measurement rather than stated ahead of it:
    # for a request on tile centre (x + 0.5, y + 0.5), is the named tile always
    # the north-west corner of the footprint?
    nw_anchored, exceptions = 0, []
    for row in report["rows"]:
        for entry in row.get("placements") or []:
            covers, asked = entry.get("covers"), entry.get("asked")
            if not covers or not asked:
                continue
            named = (int(asked[0] // 1), int(asked[1] // 1))
            if (covers[0], covers[1]) == named:
                nw_anchored += 1
            else:
                exceptions.append(
                    {
                        "name": row["name"],
                        "size": [row.get("tile_width"), row.get("tile_height")],
                        "asked": asked,
                        "named_tile": list(named),
                        "covers": covers,
                    }
                )
    report["named_tile_is_north_west_corner"] = nw_anchored
    report["exceptions"] = exceptions
    report["wall_seconds"] = round(time.perf_counter() - started, 1)

    Path(args.out).write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    for row in report["rows"]:
        size = f"{row.get('tile_width')}x{row.get('tile_height')}"
        radius = row.get("mining_drill_radius")
        first = (row.get("placements") or [{}])[0]
        print(
            f"{row['name']:24} {size:>5}  covers {first.get('covers')}"
            + (f"  mines {first.get('mines')} (r={radius})" if radius else "")
        )
    print(f"\nnamed tile is the north-west corner: {nw_anchored} placements")
    print(f"named tile is the centre instead: {len(exceptions)} placements")
    print(
        "what the agent is told vs what the engine does: "
        + ("AGREES on every item" if not mismatches else json.dumps(mismatches, indent=2))
    )
    print(f"wrote {args.out}")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
