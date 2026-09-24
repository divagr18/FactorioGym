"""Measure how a burner inserter picks items off a moving belt, tick by tick.

`docs/sim-logistics.md` found belt pickup deterministic but left the chase
unexplained. This probe builds many small rigs, each one burner inserter taking
from a yellow belt into a wooden chest, and records every tick of every rig:

- hand position (`held_stack_position`, relative to the inserter, 1/256 tile),
- whether the hand holds an item,
- burner energy (remaining fuel + buffer, full precision), whose per-tick drop
  is the arm's movement cost,
- every item on the pickup belt and its two neighbours (unique id, lane,
  position in 1/256 tile).

Rig families (`--family`):

- `single`: one item reaching an idle inserter. Inserter facing N/E/S/W; belt
  crossing the pickup tile in either direction, running toward the inserter
  (items stop at the belt end on the pickup tile) or away from it; near or far
  lane; the item's sub-step phase (position mod 8) 0..7.
- `stream`: items fed every k ticks for a long run, so the inserter meets them
  at every point of its swing. Spacing 8..40 ticks, near/far/both lanes, all
  four facings, both crossing directions, plus queues at a belt end. Some rigs
  start with an item in hand so their first return swing is out of phase.
- `wake`: when a sleeping inserter's burner buffer is refilled, long after
  the belts were built (see `wake_rigs`).
- `loss`: an item the returning hand fails to catch, lost with the hand at
  many distances from it (see `loss_rigs`).
- `pair`: two inserters on opposite sides of one belt tile, both lanes fed
  (see `pair_rigs`).

`--only key=value,...` keeps a subset of a family, `--set key=json,...`
overrides rig fields, `--origin x,y` moves the whole grid and `--tag` names
the output file; they exist for the wake-timing experiments in
docs/sim-logistics.md ("Inserter belt pickup"), which found the timing depends
on world position. `INJECT_T` is the tick the `single` rigs get their item.

Rigs are described in a local frame: the inserter at (0, 0), its pickup tile at
(0, -1), the chest at (0, 1). `rotate` maps that frame to the world for each
facing. The `rigs` list in the output is the exact spec each rig was built from.

Output: `docs/evidence/inserter-chase-<family>.json.xz`, one JSON document:
`rigs`, `setup` (what was built, pivots), and `ticks`, a list of per-tick
records `[t, [rig0, rig1, ...]]` with each rig as
`[hx, hy, held, energy, status, items, buffer]` (`energy` is burner fuel
left plus the buffer, `buffer` the buffer alone), `items` a list of
`[belt, lane, id, pos]` (`belt` -1 upstream neighbour, 0 pickup belt, 1
downstream neighbour). Time is relative to the fuelling tick; the record for t
is the state after tick t ran.

Run (about 25 s, 3 min, 15 s and 30 s):
  uv run python tools/probe_inserter_chase.py --family single
  uv run python tools/probe_inserter_chase.py --family stream
  uv run python tools/probe_inserter_chase.py --family wake
  uv run python tools/probe_inserter_chase.py --family loss
"""

from __future__ import annotations

import argparse
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

EVIDENCE = ROOT / "docs" / "evidence"
#: Rig grid origin, far outside every scene's 48-tile box.
OX, OY = 200, 200
CELL = 10
COLS = 24
FACINGS = ["N", "E", "S", "W"]
#: Tick at which the single-item rigs get their item.
INJECT_T = 40


def rotate(facing: str, x: float, y: float) -> tuple[float, float]:
    """Local frame (pickup at (0, -1)) to world offset for an inserter facing `facing`."""
    if facing == "N":
        return x, y
    if facing == "E":
        return -y, x
    if facing == "S":
        return -x, -y
    return y, -x


def belt_layout(kind: str) -> tuple[list[tuple[int, int]], int, int]:
    """Local belt tiles in flow order, local direction index (0 N .. 3 W), pickup index."""
    if kind == "east":
        return [(x, -1) for x in range(-4, 3)], 1, 4
    if kind == "west":
        return [(x, -1) for x in range(4, -3, -1)], 3, 4
    if kind == "toward":
        return [(0, y) for y in range(-5, 0)], 2, 4
    if kind == "away":
        return [(0, y) for y in range(-1, -5, -1)], 0, 0
    raise ValueError(kind)


def near_lane(kind: str) -> int:
    """Transport-line index (1 left, 2 right of travel) nearer the inserter."""
    # east: left of travel is local north (far); west: left is local south (near).
    # toward/away: the belt runs through the pickup tile along the arm; neither
    # lane is nearer. Report lane 2 by convention.
    return {"east": 2, "west": 1, "toward": 2, "away": 2}[kind]


def single_rigs() -> list[dict]:
    rigs = []
    for facing in FACINGS:
        for kind in ("east", "west", "toward", "away"):
            for which in ("near", "far"):
                for phase in range(8):
                    lane = near_lane(kind) if which == "near" else 3 - near_lane(kind)
                    belt = 0
                    pos = 255 - phase
                    if kind == "away":
                        pos = 247 - phase  # on the pickup belt itself
                    rigs.append(
                        {
                            "family": "single",
                            "facing": facing,
                            "belt": kind,
                            "lane": lane,
                            "which": which,
                            "phase": phase,
                            "hold": False,
                            "inject": [{"t": INJECT_T, "belt": belt, "lane": lane, "pos": pos}],
                        }
                    )
    return rigs


def stream_rigs() -> list[dict]:
    rigs = []
    spacings = [8, 9, 11, 13, 16, 20, 24, 29, 35, 40]
    for facing in FACINGS:
        for kind in ("east", "west"):
            for which in ("near", "far", "both"):
                for i, every in enumerate(spacings):
                    phase = (i * 3) % 8
                    hold = (i % 2) == 1
                    lanes = (
                        [near_lane(kind), 3 - near_lane(kind)]
                        if which == "both"
                        else [near_lane(kind) if which == "near" else 3 - near_lane(kind)]
                    )
                    inject = []
                    for k, lane in enumerate(lanes):
                        inject.append(
                            {
                                "t": 40 + 5 * k + i,
                                "every": every,
                                "count": 400,
                                "belt": 0,
                                "lane": lane,
                                "pos": 255 - phase,
                            }
                        )
                    rigs.append(
                        {
                            "family": "stream",
                            "facing": facing,
                            "belt": kind,
                            "which": which,
                            "every": every,
                            "phase": phase,
                            "hold": hold,
                            "inject": inject,
                        }
                    )
        for which in ("near", "far", "both"):
            kind = "toward"
            lanes = [2, 1] if which == "both" else [2 if which == "near" else 1]
            inject = [
                {"t": 40 + 3 * k, "every": 13, "count": 400, "belt": 0, "lane": lane, "pos": 250}
                for k, lane in enumerate(lanes)
            ]
            rigs.append(
                {
                    "family": "stream",
                    "facing": facing,
                    "belt": kind,
                    "which": which,
                    "every": 13,
                    "phase": 5,
                    "hold": False,
                    "inject": inject,
                }
            )
    return rigs


def wake_rigs() -> list[dict]:
    """When a sleeping inserter's burner buffer gets refilled, long after build.

    A burner inserter that goes to sleep at the pickup point keeps whatever its
    buffer held after its last move; the first move after waking is limited to
    that. Cases, all after the post-build period (t >= 500):

    - `A`: asleep since the start; one item inserted upstream at t=600.
    - `B`: an item put in its hand at t=500; while it swings, an item goes on
      the belt upstream (t=520); it comes back and sleeps before that item
      arrives. Nothing else happens on the belt.
    - `C`: as B, plus a second item inserted far upstream at t=600, while it
      sleeps.
    - `D`: as B, plus an item inserted on an unconnected belt next to it.
    - `E`: as B, plus an item inserted on the last belt, downstream of the
      pickup belt, at t=600.
    """
    rigs = []
    for facing in ("N", "E"):
        for kind in ("east", "west"):
            for case in "ABCDE":
                for rep in range(4):
                    lane = near_lane(kind)
                    main = {
                        "t": 600 if case == "A" else 520,
                        "belt": 0,
                        "lane": lane,
                        "pos": 255 - rep,
                    }
                    inject = [main]
                    if case == "C":
                        inject.append({"t": 600, "belt": 0, "lane": 3 - lane, "pos": 255})
                    if case == "D":
                        inject.append({"t": 600, "belt": -1, "lane": 1, "pos": 128})
                    if case == "E":
                        inject.append({"t": 600, "belt": 6, "lane": lane, "pos": 128})
                    rigs.append(
                        {
                            "family": "wake",
                            "facing": facing,
                            "belt": kind,
                            "which": "near",
                            "phase": rep,
                            "case": case,
                            "hold": False,
                            "hold_at": None if case == "A" else 500,
                            "inject": inject,
                        }
                    )
    return rigs


def loss_rigs() -> list[dict]:
    """An item the returning hand fails to catch, lost at many hand positions.

    One item goes on the belt upstream at t=600 and crosses the pickup belt
    around t=726..758. An item is put in the hand at `hold_at`, 600..720 in
    steps of 2, so the hand is somewhere on its way back when the item leaves.
    Both lanes, both crossing directions, north-facing.
    """
    rigs = []
    for kind in ("east", "west"):
        for which in ("near", "far"):
            lane = near_lane(kind) if which == "near" else 3 - near_lane(kind)
            for h in range(600, 722, 2):
                rigs.append(
                    {
                        "family": "loss",
                        "facing": "N",
                        "belt": kind,
                        "which": which,
                        "phase": 0,
                        "hold": False,
                        "hold_at": h,
                        "inject": [{"t": 600, "belt": 0, "lane": lane, "pos": 255}],
                    }
                )
            # finer: every tick and every sub-step phase where the hand is about
            # one tile from the item as it leaves
            for h in range(683, 693):
                for phase in range(1, 8):
                    rigs.append(
                        {
                            "family": "loss",
                            "facing": "N",
                            "belt": kind,
                            "which": which,
                            "phase": phase,
                            "hold": False,
                            "hold_at": h,
                            "inject": [{"t": 600, "belt": 0, "lane": lane, "pos": 255 - phase}],
                        }
                    )
    return rigs


OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}


def pair_rigs() -> list[dict]:
    """Two inserters taking from one belt tile, one on each side.

    Rig 2k is the usual rig (inserter at (0, 0), belt through (0, -1)), fed on
    both lanes; rig 2k+1 is a second inserter at (0, -2) of rig 2k facing the
    other way onto the same belt tile, with its own chest at (0, -3). The
    second rig builds no belts of its own; its `layout` is rig 2k's belts in
    its own frame, `partner` the index of the other rig.
    """
    rigs = []
    for facing in ("N", "E"):
        for kind in ("east", "west"):
            for i, every in enumerate((8, 11, 16, 23, 35)):
                a = {
                    "family": "pair",
                    "facing": facing,
                    "belt": kind,
                    "which": "both",
                    "every": every,
                    "phase": i,
                    "hold": i % 2 == 1,
                    "inject": [
                        {
                            "t": 40 + 5 * k + i,
                            "every": every,
                            "count": 400,
                            "belt": 0,
                            "lane": lane,
                            "pos": 255 - i,
                        }
                        for k, lane in enumerate((1, 2))
                    ],
                }
                tiles, ldir, pick = belt_layout(kind)
                b = {
                    "family": "pair",
                    "facing": OPPOSITE[facing],
                    "belt": kind,
                    "which": "both",
                    "every": every,
                    "phase": i,
                    "hold": False,
                    "inject": [],
                    "partner_of": len(rigs),
                    "layout": {
                        "tiles": [[-x, -2 - y] for x, y in tiles],
                        "dir": (ldir + 2) % 4,
                        "pick": pick + 1,
                    },
                }
                a["partner"] = len(rigs) + 1
                b["partner"] = len(rigs)
                rigs += [a, b]
    return rigs


PRELUDE = """
local s = game.surfaces[1]
local function g(v) return string.format("%.17g", v) end
local D = defines.direction
local WORLD_DIR = {D.north, D.east, D.south, D.west}
"""

SETUP = """
local spec = helpers.json_to_table(SPEC_JSON)
local OX, OY, CELL, COLS = spec.ox, spec.oy, spec.cell, spec.cols
local rows = math.ceil(#spec.rigs / COLS)
s.request_to_generate_chunks({OX + COLS * CELL / 2, OY + rows * CELL / 2},
  math.ceil(math.max(COLS, rows) * CELL / 32) + 2)
s.force_generate_chunk_requests()
local area = {{OX - 12, OY - 12}, {OX + COLS * CELL + 12, OY + rows * CELL + 12}}
for _, e in pairs(s.find_entities_filtered({area = area})) do
  if e.valid and e.type ~= "character" then e.destroy() end
end
local tiles = {}
for x = area[1][1], area[2][1] do for y = area[1][2], area[2][2] do
  tiles[#tiles + 1] = {name = "grass-1", position = {x, y}}
end end
s.set_tiles(tiles)
local fdir = {N = 0, E = 1, S = 2, W = 3}
local function rot(f, x, y)
  if f == "N" then return x, y elseif f == "E" then return -y, x
  elseif f == "S" then return -x, -y else return y, -x end
end
storage.chase = {rigs = {}, t0 = nil}
local built, failed = 0, {}
for i, r in ipairs(spec.rigs) do
  local cx = OX + ((i - 1) % COLS) * CELL + 5
  local cy = OY + math.floor((i - 1) / COLS) * CELL + 5
  local share = r.partner_of and storage.chase.rigs[r.partner_of + 1] or nil
  if share then
    local wx, wy = rot(share.facing, 0, -2)
    cx, cy = share.cx + wx, share.cy + wy
  end
  local f = r.facing
  local function at(x, y) local wx, wy = rot(f, x, y) return {cx + wx + 0.5, cy + wy + 0.5} end
  local function mk(p)
    p.force = "player"
    local e = s.create_entity(p)
    if e then built = built + 1 else failed[#failed + 1] = {i, p.name} end
    return e
  end
  local belts = {}
  local ldir = r.layout.dir
  local wdir = WORLD_DIR[((ldir + fdir[f]) % 4) + 1]
  if share then
    belts = share.belts
  else
    for k, xy in ipairs(r.layout.tiles) do
      belts[k] = mk({name = "transport-belt", position = at(xy[1], xy[2]), direction = wdir})
    end
  end
  local ins = mk({name = "burner-inserter", position = at(0, 0),
    direction = WORLD_DIR[fdir[f] + 1]})
  local chest = mk({name = "wooden-chest", position = at(0, 1)})
  local pv = ins.position
  local extra = nil
  for _, inj in ipairs(r.inject) do
    if inj.belt == -1 and not extra then
      extra = mk({name = "transport-belt", position = at(3, 3), direction = wdir})
    end
  end
  storage.chase.rigs[i] = {ins = ins, belts = belts, pick = r.layout.pick, px = pv.x, py = pv.y,
    inject = r.inject, hold = r.hold, hold_at = r.hold_at, extra = extra, fed = {},
    cx = cx, cy = cy, facing = f}
end
local out = {}
for i, rr in ipairs(storage.chase.rigs) do
  local ins = rr.ins
  out[i] = {pivot = {g(rr.px), g(rr.py)},
    pickup = {g(ins.pickup_position.x), g(ins.pickup_position.y)},
    drop = {g(ins.drop_position.x), g(ins.drop_position.y)},
    pickup_target = ins.pickup_target and ins.pickup_target.name or nil,
    drop_target = ins.drop_target and ins.drop_target.name or nil,
    direction = ins.direction,
    belt_dir = rr.belts[1].direction}
end
return {built = built, failed = failed, rigs = out}
"""

FUEL = """
local st = storage.chase
for _, rr in ipairs(st.rigs) do
  rr.ins.insert({name = "coal", count = 5})
  if rr.hold then rr.ins.held_stack.set_stack({name = "iron-plate", count = 1}) end
end
st.t0 = game.tick
st.lines = {}
local status_names = {}
for name, value in pairs(defines.entity_status) do status_names[value] = name end
local function record()
  local t = game.tick - st.t0
  local rows = {}
  for i, rr in ipairs(st.rigs) do
    local ins = rr.ins
    local hp = ins.held_stack_position
    local hs = ins.held_stack
    local b = ins.burner
    local items = {}
    for d = -1, 1 do
      local belt = rr.belts[rr.pick + d]
      if belt and belt.valid then
        for lane = 1, 2 do
          for _, it in pairs(belt.get_transport_line(lane).get_detailed_contents()) do
            items[#items + 1] = string.format("[%d,%d,%d,%d]", d, lane, it.unique_id,
              math.floor(it.position * 256 + 0.5))
          end
        end
      end
    end
    rows[i] = string.format("[%d,%d,%d,\\"%s\\",%d,[%s],\\"%s\\"]",
      math.floor((hp.x - rr.px) * 256 + 0.5), math.floor((hp.y - rr.py) * 256 + 0.5),
      hs.valid_for_read and hs.count or 0,
      g(b.remaining_burning_fuel + ins.energy), ins.status or -1, table.concat(items, ","),
      g(ins.energy))
  end
  helpers.write_file(st.file, string.format("[%d,[%s]]\\n", t, table.concat(rows, ",")), true)
end
-- Feeds scheduled items. Runs in on_tick of tick T, i.e. between ticks T-1 and T.
local function feed()
  local t = game.tick - st.t0
  for _, rr in ipairs(st.rigs) do
    if rr.hold_at and t == rr.hold_at then
      rr.ins.held_stack.set_stack({name = "iron-plate", count = 1})
    end
    for k, inj in ipairs(rr.inject) do
      local n = rr.fed[k] or 0
      local due = inj.t + n * (inj.every or 0)
      if n < (inj.count or 1) and t >= due then
        local belt = inj.belt == -1 and rr.extra or rr.belts[inj.belt + 1]
        local line = belt.get_transport_line(inj.lane)
        if line.insert_at(inj.pos / 256, {name = "iron-plate", count = 1}) then
          rr.fed[k] = n + 1
        elseif not inj.every then
          rr.fed[k] = n + 1  -- a single insert that failed is not retried
        end
      end
    end
  end
end
st.record, st.feed = record, feed
local old = script.get_event_handler(defines.events.on_tick)
script.on_event(defines.events.on_tick, function(e)
  if old then old(e) end
  if st.last == game.tick then return end
  st.last = game.tick
  if st.on then
    record()  -- the state after tick game.tick - 1
    feed()
  end
end)
st.on = true
st.file = FILE
helpers.write_file(st.file, "", false)
return st.t0
"""


def run(
    family: str,
    ticks: int,
    only: str | None = None,
    tag: str = "",
    origin: tuple[int, int] = (OX, OY),
    overrides: str | None = None,
) -> Path:
    rigs = {
        "single": single_rigs,
        "stream": stream_rigs,
        "wake": wake_rigs,
        "loss": loss_rigs,
        "pair": pair_rigs,
    }[family]()
    if only:
        keep = dict(kv.split("=") for kv in only.split(","))
        rigs = [r for r in rigs if all(str(r.get(k)) == v for k, v in keep.items())]
    for kv in (overrides or "").split(","):
        if kv:
            k, v = kv.split("=")
            for r in rigs:
                r[k] = json.loads(v)
    for r in rigs:
        if "layout" not in r:
            tiles, ldir, pick = belt_layout(r["belt"])
            r["layout"] = {"tiles": [list(t) for t in tiles], "dir": ldir, "pick": pick + 1}
    spec = {"ox": origin[0], "oy": origin[1], "cell": CELL, "cols": COLS, "rigs": rigs}
    fname = f"chase-{family}.jsonl"
    manager = WorkerManager()
    handle = manager.launch(f"chase-{family}")
    started = time.perf_counter()
    try:
        with RCONClient(handle.spec.rcon_endpoint, timeout=300.0) as client:
            client.lua(f"game.speed = {resolve_game_speed()} return game.speed")
            session = WorkerSession(handle, timeout=300.0)
            session.status()
            env = FactorioEnv(
                get("construct_smelting_line"),
                session,
                SeedPlan(master=41, run_id=f"chase-{family}"),
                branch=Branch.TRAIN,
                split="train",
            )
            env.reset(options={"scene_index": 0})
            setup = client.lua(PRELUDE + SETUP.replace("SPEC_JSON", json.dumps(json.dumps(spec))))
            failed = list(setup["failed"])[:5] if setup["failed"] else []
            print("built", setup["built"], "failed", failed)
            fuel = FUEL.replace("FILE", json.dumps(fname))
            # The record written in on_tick of game tick T is the state after T - 1:
            # relabel below.
            client.lua(PRELUDE + fuel)
            env.advance(ticks + 1)
            session.close()
        raw = (handle.spec.write_data / "script-output" / fname).read_text(encoding="utf-8")
    finally:
        manager.cleanup(handle)
    lines = [json.loads(line) for line in raw.splitlines() if line.strip()]
    # on_tick(T) sees the state after tick T-1; the first line is on_tick of t0+1.
    records = [[t - 1, rows] for t, rows in lines]
    doc = {
        "engine": manager.engine.to_dict(),
        "family": family,
        "origin": list(origin),
        "rigs": rigs,
        "setup": setup,
        "wall_seconds": round(time.perf_counter() - started, 1),
        "record_format": "[t, [[hx, hy, held_count, energy_str, status, "
        "[[belt(-1,0,1), lane, unique_id, pos256], ...], buffer_str], ...]]",
        "ticks": records,
    }
    out = EVIDENCE / f"inserter-chase-{family}{tag}.json.xz"
    with lzma.open(out, "wt", encoding="utf-8") as fh:
        json.dump(doc, fh, separators=(",", ":"))
    print(f"wrote {out} ({out.stat().st_size // 1024} KiB), {len(records)} ticks, {len(rigs)} rigs")
    return out


def load(family: str) -> dict:
    with lzma.open(EVIDENCE / f"inserter-chase-{family}.json.xz", "rt", encoding="utf-8") as fh:
        return json.load(fh)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--family", choices=["single", "stream", "wake", "loss", "pair"], required=True
    )
    parser.add_argument("--ticks", type=int, default=None)
    parser.add_argument("--only", default=None, help="rig filter, e.g. facing=N,belt=east")
    parser.add_argument("--tag", default="", help="suffix for the output file name")
    parser.add_argument("--origin", default=f"{OX},{OY}")
    parser.add_argument("--set", default=None, help="override rig fields, e.g. hold=true")
    args = parser.parse_args()
    origin = tuple(int(v) for v in args.origin.split(","))
    ticks = (
        args.ticks
        or {"single": 400, "stream": 3000, "wake": 900, "loss": 900, "pair": 2000}[args.family]
    )
    run(args.family, ticks, args.only, args.tag, origin, args.set)
    return 0


if __name__ == "__main__":
    sys.exit(main())
