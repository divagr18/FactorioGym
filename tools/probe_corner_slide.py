"""Measure how a walking character slides around an obstacle's corner (M5).

A factory-sim policy's greedy trajectory diverged from the engine's on a
held-out `obstructed_patch` scene
(`docs/evidence/sim-transfer-m5.json`): walking west past a wall column, the
engine moved the character sideways by 50/256 of a tile and let it through,
while the simulator stopped it. Factorio nudges a character around a corner
when it only just overlaps an obstacle. The simulator does not, and the
recorded golden traces never walk into a corner.

Each rig builds one obstacle in open ground far from any scene:
- a single stone wall;
- the `obstructed_patch` column of five walls;
- a stone furnace.

For each rig the probe starts the character at a grid of offsets across the
obstacle, walks it at the obstacle, and records its position every tick. Two
directions, west and north, check that the rule is the same on both axes.
Offsets are in 1/256 tiles, the engine's position resolution.

Writes docs/evidence/sim-mechanics-m5-slide.json. With `--gaps` it runs the
second set instead, and writes sim-mechanics-m5-slide-gaps.json:
- what passes between two adjacent walls, two walls a tile apart, and two
  adjacent furnaces;
- which way a centred walker slides, in all four directions.

With `--creep`, writes sim-mechanics-m5-slide-creep.json:
- how far a blocked walker creeps each tick, from every start phase;
- the 131/256 gap between a wall and a furnace.

    uv run python tools/probe_corner_slide.py [--gaps | --creep]
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

OUT = ROOT / "docs" / "evidence" / "sim-mechanics-m5-slide.json"

#: Rig origin, far outside every scene's 48-tile box.
OX, OY = -160, 160
TICKS = 64
#: Start this many tiles from the obstacle's centre, then walk back at it.
DIRECTIONS = {"west": (6, 0), "north": (0, 6)}
#: All four, for the rigs that check which way a centred walker slides.
ALL_DIRECTIONS = {"west": (6, 0), "north": (0, 6), "east": (-6, 0), "south": (0, -6)}

#: Each rig: the entities it builds, as (prototype, dx, dy) from the origin,
#: and the start offsets across it, in 1/256 tiles from its centre.
RIGS = {
    "wall": ([("stone-wall", 0, 0)], list(range(-160, 161, 8))),
    # The obstructed_patch shape. Walking west at it, a slide would have to
    # clear the whole five-tile column.
    "wall_column": (
        [("stone-wall", 0, dy) for dy in range(-2, 3)],
        list(range(-768, 769, 32)),
    ),
    "furnace": ([("stone-furnace", 0, 0)], list(range(-256, 257, 16))),
}

#: Second probe (`--gaps`): what passes between two obstacles, and ties.
GAP_RIGS = {
    # Two adjacent walls: the gap between their boxes is 108/256, wider than
    # the character's 102, yet the column rig was blocked mid-gap. Offsets
    # across the gap at full resolution, relative to the pair's centre.
    "walls_adjacent": ([("stone-wall", 0, 0), ("stone-wall", 0, 1)], list(range(-70, 71, 2))),
    # One empty tile between them: a 364/256 gap.
    "walls_one_apart": ([("stone-wall", 0, 0), ("stone-wall", 0, 2)], list(range(-256, 257, 8))),
    # Two furnaces side by side: a 154/256 gap between their boxes.
    "furnaces_adjacent": (
        [("stone-furnace", 0, 0), ("stone-furnace", 0, 2)],
        list(range(-192, 193, 8)),
    ),
    # A centred walker, and just off centre, in all four directions.
    "wall_ties": ([("stone-wall", 0, 0)], [-4, -1, 0, 1, 4]),
    "furnace_ties": ([("stone-furnace", 0, 0)], [-1, 0, 1]),
}
TIE_RIGS = {"wall_ties", "furnace_ties"}

#: Third probe (`--creep`): how far a blocked walker creeps each tick, from
#: every start phase, where no slide is possible; and one more gap width.
CREEP_RIGS = {
    # Dead centre on a furnace: the needed slide (231) is over the limit.
    "furnace_creep": ([("stone-furnace", 0, 0)], [0]),
    # Mid-gap on two adjacent walls: blocked, and no slide.
    "walls_adjacent_creep": ([("stone-wall", 0, 0), ("stone-wall", 0, 1)], [0]),
    # A wall beside a furnace, centres 1.5 tiles apart: a 131/256 gap.
    "wall_furnace_gap": (
        [("stone-furnace", 0, 0), ("stone-wall", 0, 1)],
        list(range(-160, 161, 8)),
    ),
}
CREEP_RIGS_PHASED = {"furnace_creep", "walls_adjacent_creep"}
#: Start distance offsets for the phased rigs: one stride, unit by unit.
PHASES = list(range(38))

SETUP = """
local s = game.surfaces[1]
for _, e in pairs(s.find_entities_filtered({area = {{OX - 12, OY - 12}, {OX + 12, OY + 12}}})) do
  if e.valid and e.type ~= "character" then e.destroy() end
end
local placed = {}
for _, spec in ipairs(SPECS) do
  local e = s.create_entity({name = spec[1], position = {OX + spec[2], OY + spec[3]},
    force = "neutral"})
  placed[#placed + 1] = {e.name, e.position.x, e.position.y}
end
return placed
"""

START = """
local ch = storage.frrl_character
ch.walking_state = {walking = false}
local ok = ch.teleport({X, Y})
return {ok, ch.position.x, ch.position.y}
"""

WALK = """
local ch = storage.frrl_character
ch.walking_state = {walking = true, direction = defines.direction.DIR}
return {ch.position.x, ch.position.y}
"""

STOP = """
local ch = storage.frrl_character
ch.walking_state = {walking = false}
return {ch.position.x, ch.position.y}
"""


def walk_rig(client, env, rig, centre, offsets, directions=DIRECTIONS, phases=(0,)) -> None:
    """Walk at the rig from each offset (and start phase), in each direction."""
    cx, cy = centre
    for name, (away_x, away_y) in directions.items():
        runs = []
        for offset, phase in [(o, ph) for o in offsets for ph in phases]:
            if away_x:
                x, y = cx + away_x + phase / 256, cy + offset / 256
            else:
                x, y = cx + offset / 256, cy + away_y + phase / 256
            start = client.lua(START.replace("X", repr(x)).replace("Y", repr(y)))
            path = []
            for _ in range(TICKS):
                path.append(client.lua(WALK.replace("DIR", name)))
                env.advance(1)
            path.append(client.lua(STOP))
            runs.append({"offset_256": offset, "phase_256": phase, "start": start, "path": path})
        rig["runs"][name] = runs


def main() -> int:
    mode = "gaps" if "--gaps" in sys.argv else "creep" if "--creep" in sys.argv else ""
    rigs = {"gaps": GAP_RIGS, "creep": CREEP_RIGS}.get(mode, RIGS)
    out = OUT.with_name(f"sim-mechanics-m5-slide-{mode}.json") if mode else OUT
    manager = WorkerManager()
    handle = manager.launch("corner-slide")
    report: dict = {
        "engine": {k: v for k, v in manager.engine.to_dict().items() if k != "executable"},
        "rig_origin": [OX, OY],
        "ticks": TICKS,
        "start_distance": DIRECTIONS,
        "rigs": {},
    }
    started = time.perf_counter()
    try:
        with RCONClient(handle.spec.rcon_endpoint, timeout=60.0) as client:
            client.lua(f"game.speed = {resolve_game_speed()} return game.speed")
            session = WorkerSession(handle, timeout=60.0)
            session.status()
            env = FactorioEnv(
                get("construct_smelting_line"),
                session,
                SeedPlan(master=31, run_id="corner-slide"),
                branch=Branch.TRAIN,
                split="train",
            )
            env.reset(options={"scene_index": 0})
            for name, (specs, offsets) in rigs.items():
                lua_specs = "{" + ",".join(f'{{"{p}", {dx}, {dy}}}' for p, dx, dy in specs) + "}"
                code = SETUP.replace("SPECS", lua_specs)
                placed = client.lua(code.replace("OX", str(OX)).replace("OY", str(OY)))
                centre = (
                    sum(p[1] for p in placed) / len(placed),
                    sum(p[2] for p in placed) / len(placed),
                )
                rig = {"placed": placed, "centre": centre, "offsets_256": offsets, "runs": {}}
                report["rigs"][name] = rig
                directions = ALL_DIRECTIONS if name in TIE_RIGS else DIRECTIONS
                phases = PHASES if name in CREEP_RIGS_PHASED else (0,)
                walk_rig(client, env, rig, centre, offsets, directions, phases)
                # Written after every rig, so a run cut short keeps what it measured.
                out.write_text(json.dumps(report, indent=0, sort_keys=True) + "\n", "utf-8")
                print(f"rig {name} done", flush=True)
            session.close()
    finally:
        manager.cleanup(handle)
    report["wall_seconds"] = round(time.perf_counter() - started, 1)
    out.write_text(json.dumps(report, indent=0, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
