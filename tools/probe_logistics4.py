"""Measure the belt merge delay over whole areas, and settle two sideload details (M4, logistics 4).

The third logistics probe (tools/probe_logistics3.py) found that a belt lane's
line merges with its neighbours a fixed delay `d` after it is built, `d` a
function of the tile and the lane only, and measured it over the parity
scenes' area. factory-sim reproduces belt-line segments from a table of `d`;
this probe measures that table and checks it. docs/sim-logistics.md, "Fourth
probe", states the findings. Families (`--family`):

- `delaymap`: `d` for every tile and both lanes of a rectangle (`--rect
  x0,y0,x1,y1`, half-open, tiles), by the third probe's method: a lone belt
  at each measured tile (`--facing` 0, north, or 4, east), its downstream
  neighbour destroyed and rebuilt every tick so the neighbour's own delay
  never runs out, and the tick the two lines become equal. Six worlds
  (lattice offsets) cover the rectangle. A neighbour whose own tile has
  d = 1 runs out before it is rebuilt and the reading is then 1, so each
  rectangle is measured facing north and facing east and factory-sim's
  tools/make_belt_delay.py combines the two. Output
  `runtime/delaymap-<x0>_<y0>_<x1>_<y1>[-f4].json.xz` (large).
- `delaycheck`: the same measurement again, in fresh worlds, for a random
  sample of tiles (`--sample N` per rectangle of `--rects`, seed `--seed`):
  facing north twice, then east, south and west, to check determinism and
  that facing does not matter.
- `order`: two burner inserters acting on the same tick, built in both
  orders, in several arrangements: which acts first.
- `sleep`: a stopped line of young (and of old) belts freed by a script, an
  inserter or a new belt: when the items behind move again.
- `accept`: a sideload or a drop arriving next to an item moving along the
  target lane, at every relative position and phase: where it goes on.

Run (on the desktop: about 2.5 min per 256 x 256 facing; the others 20 to 60 s):
  uv run python tools/probe_logistics4.py --family delaymap --rect=-128,-128,128,128
  uv run python tools/probe_logistics4.py --family delaymap --rect=-128,-128,128,128 --facing 4
  (and the same for --rect=96,96,352,352 and --rect=296,456,340,480)
  uv run python tools/probe_logistics4.py --family delaycheck --sample 300
  uv run python tools/probe_logistics4.py --family order
  uv run python tools/probe_logistics4.py --family sleep
  uv run python tools/probe_logistics4.py --family accept
"""

from __future__ import annotations

import argparse
import json
import lzma
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
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
RUNTIME = ROOT / "runtime"

#: Common Lua: helpers and an area made ready for building (chunks generated,
#: entities other than resources removed, land tiles, characters moved out).
PRE = r"""
local s = game.surfaces[1]
local D = defines.direction
storage.p4 = {log = {}, out = {}}
local P = storage.p4
local function mk(spec)
  spec.force = spec.force or "player"
  local e = s.create_entity(spec)
  assert(e, "create failed " .. spec.name .. " at " .. spec.position[1] .. "," .. spec.position[2])
  return e
end
local function belt(x, y, d)
  return mk({name = "transport-belt", position = {x + 0.5, y + 0.5}, direction = d})
end
local function put(b, lane, pos, name)
  return b.get_transport_line(lane).insert_at(pos / 256, {name = name or "copper-plate", count = 1})
end
local function prepare(x0, y0, x1, y1)
  local cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
  local r = math.ceil(math.max(x1 - x0, y1 - y0) / 64) + 2
  s.request_to_generate_chunks({cx, cy}, r)
  s.force_generate_chunk_requests()
  for _, c in pairs(s.find_entities_filtered({type = "character"})) do
    c.teleport({x0 - 40, y1 + 40})
  end
  for _, e in pairs(s.find_entities_filtered({area = {{x0, y0}, {x1, y1}}})) do
    if e.valid and e.type ~= "character" and e.type ~= "resource" then e.destroy() end
  end
  local tiles = {}
  for x = x0, x1 - 1 do
    for y = y0, y1 - 1 do tiles[#tiles + 1] = {name = "grass-1", position = {x, y}} end
  end
  s.set_tiles(tiles)
end
local HOOK = nil
"""

POST = r"""
P.t0 = game.tick
local old = script.get_event_handler(defines.events.on_tick)
script.on_event(defines.events.on_tick, function(e)
  if old then old(e) end
  if P.last == game.tick then return end
  P.last = game.tick
  local t = game.tick - P.t0
  if HOOK then HOOK(t) end
end)
return P.extra or true
"""


def _run_world(body: str, ticks: int, label: str) -> dict:
    """Build `body` in a fresh world, run `ticks` ticks, and fetch `P.out` (a
    list of strings) and `P.log`."""
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
            extra = client.lua(PRE + body + POST)
            env.advance(ticks)
            n = client.lua("return #storage.p4.out")
            out = []
            for a in range(1, n + 1):
                out.append(client.lua(f"return storage.p4.out[{a}]"))
            n = client.lua("return #storage.p4.log")
            log = []
            for a in range(1, n + 1, 200):
                log += client.lua(
                    f"local o = {{}} for i = {a}, math.min({a + 199}, #storage.p4.log) do "
                    "o[#o + 1] = storage.p4.log[i] end return o"
                )
            session.close()
    finally:
        manager.cleanup(handle)
    return {"extra": extra, "out": out, "log": log}


def _run_many(jobs: list[tuple[str, str]], ticks: int, parallel: int) -> dict:
    out: dict = {}

    def one(k: int) -> None:
        name, body = jobs[k]
        started = time.perf_counter()
        out[name] = _run_world(body, ticks, f"probe4-{k:03d}")
        print(f"  {name}: {time.perf_counter() - started:.0f} s", flush=True)

    with ThreadPoolExecutor(parallel) as pool:
        list(pool.map(one, range(len(jobs))))
    return out


# ---------------------------------------------------------------- merge delay

#: Measure `d` at every tile of `specs` ({x, y, facing}): the belt at the tile
#: faces `facing`, its neighbour is the next tile downstream, destroyed and
#: rebuilt every tick until both lanes have merged. `P.out` gets, per tile in
#: order, "t1,t2" (lane 1 and lane 2), in strings of at most 2,000 tiles.
DELAY = r"""
local V = {[0] = {0, -1}, [4] = {1, 0}, [8] = {0, 1}, [12] = {-1, 0}}
local X, N, M1, M2 = {}, {}, {}, {}
for k, p in ipairs(specs) do X[k] = belt(p[1], p[2], p[3]) end
local left = #specs
HOOK = function(t)
  if left == 0 then return end
  for k, p in ipairs(specs) do
    if not (M1[k] and M2[k]) then
      local n = N[k]
      if n and n.valid then
        local x = X[k]
        if not M1[k] and x.get_transport_line(1).line_equals(n.get_transport_line(1)) then
          M1[k] = t
        end
        if not M2[k] and x.get_transport_line(2).line_equals(n.get_transport_line(2)) then
          M2[k] = t
        end
      end
      if M1[k] and M2[k] then
        left = left - 1
        if n and n.valid then n.destroy() end
      else
        if n and n.valid then n.destroy() end
        local v = V[p[3]]
        N[k] = belt(p[1] + v[1], p[2] + v[2], p[3])
      end
    end
  end
  if left == 0 or t == TICKS then
    local chunk = {}
    for k = 1, #specs do
      chunk[#chunk + 1] = (M1[k] or -1) .. "," .. (M2[k] or -1)
      if #chunk == 2000 or k == #specs then
        P.out[#P.out + 1] = table.concat(chunk, ";")
        chunk = {}
      end
    end
    left = 0
  end
end
"""


def _delay_job(specs_lua: str, prepare: tuple[int, int, int, int], ticks: int) -> str:
    x0, y0, x1, y1 = prepare
    return (
        f"prepare({x0}, {y0}, {x1}, {y1})\n"
        f"local TICKS = {ticks}\n"
        f"local specs = {specs_lua}\n" + DELAY
    )


#: Lattice steps (x, y) per facing: a belt and its downstream neighbour, with
#: a free tile before the next measured belt along the facing.
STEPS = {0: (2, 3), 4: (3, 2)}


def _lattice_lua(x0: int, y0: int, x1: int, y1: int, ox: int, oy: int, facing: int = 0) -> str:
    """Lua that builds the list of measured tiles of one world."""
    sx, sy = STEPS[facing]
    return (
        "(function() local o = {} "
        f"for x = {x0 + ox}, {x1 - 1}, {sx} do for y = {y0 + oy}, {y1 - 1}, {sy} do "
        f"o[#o + 1] = {{x, y, {facing}}} end end return o end)()"
    )


def _lattice(x0: int, y0: int, x1: int, y1: int, ox: int, oy: int,
             facing: int = 0) -> list[tuple[int, int]]:  # fmt: skip
    sx, sy = STEPS[facing]
    return [(x, y) for x in range(x0 + ox, x1, sx) for y in range(y0 + oy, y1, sy)]


def _parse(out: list[str]) -> list[tuple[int, int]]:
    vals = []
    for chunk in out:
        for pair in chunk.split(";"):
            a, b = pair.split(",")
            vals.append((int(a), int(b)))
    return vals


def family_delaymap(args) -> dict:
    x0, y0, x1, y1 = (int(v) for v in args.rect.split(","))
    ticks = args.ticks or 700
    facing = args.facing
    sx, sy = STEPS[facing]
    jobs = []
    for ox in range(sx):
        for oy in range(sy):
            body = _delay_job(_lattice_lua(x0, y0, x1, y1, ox, oy, facing),
                              (x0 - 3, y0 - 3, x1 + 3, y1 + 3), ticks)  # fmt: skip
            jobs.append((f"o{ox}{oy}", body))
    res = _run_many(jobs, ticks + 5, args.parallel)
    w, h = x1 - x0, y1 - y0
    lane1, lane2 = [-2] * (w * h), [-2] * (w * h)
    for ox in range(sx):
        for oy in range(sy):
            tiles = _lattice(x0, y0, x1, y1, ox, oy, facing)
            vals = _parse(res[f"o{ox}{oy}"]["out"])
            assert len(vals) == len(tiles), (len(vals), len(tiles))
            for (x, y), (a, b) in zip(tiles, vals, strict=True):
                lane1[(y - y0) * w + (x - x0)] = a
                lane2[(y - y0) * w + (x - x0)] = b
    missing = sum(v < 0 for v in lane1 + lane2)
    nb = {0: "(x, y-1)", 4: "(x+1, y)"}[facing]
    return {"rect": [x0, y0, x1, y1], "order": "row-major, y then x", "lane1": lane1,
            "lane2": lane2, "missing": missing, "ticks": ticks, "facing": facing,
            "method": f"belt facing {facing} per tile, downstream neighbour at {nb} rebuilt "
                      "every tick; first on_tick t (from the build command) at which "
                      "line_equals"}  # fmt: skip


def family_delaycheck(args) -> dict:
    """Re-measure a random sample of every rectangle given: north twice (two
    worlds each), then east, south and west once."""
    rng = random.Random(args.seed)
    rects = [tuple(int(v) for v in r.split(",")) for r in args.rects.split(";")]
    sample: list[tuple[int, int]] = []
    for x0, y0, x1, y1 in rects:
        sample += [(rng.randrange(x0, x1), rng.randrange(y0, y1)) for _ in range(args.sample)]
    sample = sorted(set(sample))
    ticks = args.ticks or 700
    # Measured tiles far apart: each at its own place in a sparse grid of
    # 4 x 4 cells, so a belt and its neighbour never touch another rig.
    cells = {}
    jobs = []
    for facing, reps in ((0, 2), (4, 1), (8, 1), (12, 1)):
        for rep in range(reps):
            name = f"f{facing}r{rep}"
            # a world can only build at the real tiles, so tiles that would
            # touch are split over several worlds
            groups: list[list[tuple[int, int]]] = []
            for tx, ty in sample:
                for g in groups:
                    if all(abs(tx - gx) > 2 or abs(ty - gy) > 2 for gx, gy in g):
                        g.append((tx, ty))
                        break
                else:
                    groups.append([(tx, ty)])
            for gi, g in enumerate(groups):
                spec = ",".join(f"{{{x},{y},{facing}}}" for x, y in g)
                xs, ys = [x for x, _ in g], [y for _, y in g]
                box = (min(xs) - 3, min(ys) - 3, max(xs) + 4, max(ys) + 4)
                jobs.append((f"{name}g{gi}", _delay_job("{" + spec + "}", box, ticks)))
                cells[f"{name}g{gi}"] = g
    res = _run_many(jobs, ticks + 5, args.parallel)
    table: dict = {}
    for job, g in cells.items():
        name = job.split("g")[0]
        vals = _parse(res[job]["out"])
        for (x, y), (a, b) in zip(g, vals, strict=True):
            table.setdefault(name, {})[f"{x},{y}"] = [a, b]
    return {"rects": rects, "seed": args.seed, "sample": sample, "measured": table}


# ---------------------------------------------------------------- order

#: Per-tick log of every item on the rig's belts (with the engine's ids) and
#: every chest's contents, and each inserter's held item.
REC = r"""
local function items(line)
  local o = {}
  for _, it in pairs(line.get_detailed_contents()) do
    o[#o + 1] = {math.floor(it.position * 256 + 0.5), it.unique_id, it.stack.name}
  end
  return o
end
local function inv(c)
  local o = {}
  local i = c.get_inventory(defines.inventory.chest)
  for k = 1, #i do
    if i[k].valid_for_read then o[#o + 1] = {k, i[k].name, i[k].count} end
  end
  return o
end
local function sample(t)
  local row = {t = t, b = {}, c = {}, i = {}}
  for k, b in ipairs(R.belts) do
    row.b[k] = {items(b.get_transport_line(1)), items(b.get_transport_line(2))}
  end
  for k, c in ipairs(R.chests) do row.c[k] = inv(c) end
  for k, e in ipairs(R.ins) do
    row.i[k] = e.held_stack.valid_for_read and e.held_stack.name or ""
  end
  return row
end
HOOK = function(t)
  if t <= TICKS then P.log[#P.log + 1] = sample(t) end
end
"""


def _order_rig(kind: str, order: str) -> str:
    """Two inserters W (west of the middle tile, facing west: picks from x-1,
    drops on the middle) and E (mirror), built in `order`. `kind`:

    - `belt`: the middle is a north feed belt (as `ins_*` in the third probe),
      each inserter fed from its own chest of copper;
    - `belt_pre`: the same with both chests built before either inserter;
    - `belt_nofuel`: the same, inserters not given coal (built-in wood only);
    - `chest1`: the middle is a chest with one free slot, each inserter fed
      from its own chest (one iron, one copper): who fills the slot;
    - `pick1`: the middle is a chest with one plate; W and E pick from it
      (facing east and west) and drop on their own belts;
    - `pick1_chest`: the same, dropping into their own chests (the second
      probe's `order_pick_*`, turned a quarter);
    - `vert`: N and S inserters dropping on both lanes of an east belt.
    """
    ox, oy = 200, 200
    lines = ["R = {belts = {}, chests = {}, ins = {}}"]
    add = lines.append

    def chest(x, y, item=None, n=50, full_but=None):
        s = f"do local c = mk({{name = 'wooden-chest', position = {{{x} + 0.5, {y} + 0.5}}}}) "
        if item:
            s += f"c.insert({{name = '{item}', count = {n}}}) "
        if full_but is not None:
            s += (
                "local i = c.get_inventory(defines.inventory.chest) "
                f"for k = 1, 16 - {full_but} do i[k].set_stack({{name = 'stone', count = 50}}) end "
            )
        s += "R.chests[#R.chests + 1] = c end"
        return s

    def ins(x, y, d, fuel=True):
        s = (f"do local e = mk({{name = 'burner-inserter', position = {{{x} + 0.5, {y} + 0.5}}, "
             f"direction = D.{d}}}) ")  # fmt: skip
        if fuel:
            s += "e.insert({name = 'coal', count = 2}) "
        s += "R.ins[#R.ins + 1] = e end"
        return s

    if kind in ("belt", "belt_pre", "belt_nofuel", "belt_old"):
        add(f"for i = 0, 2 do R.belts[#R.belts + 1] = belt({ox} + i, {oy}, D.east) end")
        add(f"R.belts[#R.belts + 1] = belt({ox} + 1, {oy} + 1, D.north)")
        fuel = kind != "belt_nofuel"
        side = {
            "W": (chest(ox - 1, oy + 1, "copper-plate"), ins(ox, oy + 1, "west", fuel)),
            "E": (chest(ox + 3, oy + 1, "iron-plate"), ins(ox + 2, oy + 1, "east", fuel)),
        }
        if kind == "belt_pre":
            add(side[order[0]][0])
            add(side[order[1]][0])
            add(side[order[0]][1])
            add(side[order[1]][1])
        else:
            for k in order:
                add(side[k][0])
                add(side[k][1])
    elif kind == "vert":
        # an east main, a west feed into its middle... simpler: one east belt
        # of three, inserters north and south of the middle belt dropping on it
        add(f"for i = 0, 2 do R.belts[#R.belts + 1] = belt({ox} + i, {oy}, D.east) end")
        side = {
            "N": (chest(ox + 1, oy - 2, "copper-plate"), ins(ox + 1, oy - 1, "north")),
            "S": (chest(ox + 1, oy + 2, "iron-plate"), ins(ox + 1, oy + 1, "south")),
        }
        for k in order:
            add(side[k][0])
            add(side[k][1])
    elif kind == "chest1":
        add(chest(ox + 1, oy, None, 0, full_but=1))
        side = {
            "W": (chest(ox - 1, oy, "copper-plate"), ins(ox, oy, "west")),
            "E": (chest(ox + 3, oy, "iron-plate"), ins(ox + 2, oy, "east")),
        }
        for k in order:
            add(side[k][0])
            add(side[k][1])
    elif kind == "pick1":
        add(chest(ox + 1, oy, "iron-plate", 1))
        side = {
            "W": (f"R.belts[#R.belts + 1] = belt({ox - 1}, {oy}, D.north)", ins(ox, oy, "east")),
            "E": (
                f"R.belts[#R.belts + 1] = belt({ox + 3}, {oy}, D.north)",
                ins(ox + 2, oy, "west"),
            ),
        }
        for k in order:
            add(side[k][0])
            add(side[k][1])
    elif kind == "pick1_chest":
        add(chest(ox + 1, oy, "iron-plate", 1))
        side = {
            "W": (chest(ox - 1, oy), ins(ox, oy, "east")),
            "E": (chest(ox + 3, oy), ins(ox + 2, oy, "west")),
        }
        for k in order:
            add(side[k][0])
            add(side[k][1])
    else:
        raise ValueError(kind)
    return "\n".join(lines) + "\n"


ORDER_KINDS = ("belt", "belt_pre", "belt_nofuel", "vert", "chest1", "pick1", "pick1_chest")


def family_order(args) -> dict:
    jobs = []
    for kind in ORDER_KINDS:
        orders = ("NS", "SN") if kind == "vert" else ("WE", "EW")
        for order in orders:
            body = (f"prepare(190, 190, 215, 215)\nlocal TICKS = {args.ticks or 120}\n"
                    + _order_rig(kind, order) + REC)  # fmt: skip
            jobs.append((f"{kind}_{order}", body))
    res = _run_many(jobs, (args.ticks or 120) + 5, args.parallel)
    return {name: {"log": r["log"]} for name, r in res.items()}


# ---------------------------------------------------------------- acceptance

#: One item arriving on a lane just behind or ahead of an item moving along it,
#: at every relative position and phase around the boundary, fresh belts:
#: - `side`: a three-belt east main at (X..X+2, Y) and a one-belt north feed at
#:   (X+1, Y+1) sideloading onto the main's lane 2; one item on feed lane `L`
#:   at `F` (16..23: it reaches the main in 2 or 3 ticks), one on the main's
#:   lane 2 at chain coordinate `C` from the main's end (C 256.. is belt 2);
#: - `drop`: a four-belt east main at (X..X+3, Y); a fuelled burner inserter
#:   at (X+2, Y+1) facing south, from a chest of iron, drops onto belt 3's
#:   lane 1 at t=47; one item on the main's lane 1 at chain coordinate `C`.
ACCEPT_SIDE = r"""
do
  local b = {}
  for i = 0, 2 do b[#b + 1] = belt(X + i, Y, D.east) end
  b[4] = belt(X + 1, Y + 1, D.north)
  put(b[4], L, F, "copper-plate")
  put(b[3 - math.floor(C / 256)], 2, C % 256, "iron-plate")
  RIGS[#RIGS + 1] = {name = NAME, belts = b}
end
"""
ACCEPT_DROP = r"""
do
  local b = {}
  for i = 0, 3 do b[#b + 1] = belt(X + i, Y, D.east) end
  local c = mk({name = "wooden-chest", position = {X + 2.5, Y + 2.5}})
  c.insert({name = "copper-plate", count = 5})
  local e = mk({name = "burner-inserter", position = {X + 2.5, Y + 1.5}, direction = D.south})
  e.insert({name = "coal", count = 2})
  put(b[4 - math.floor(C / 256)], 1, C % 256, "iron-plate")
  RIGS[#RIGS + 1] = {name = NAME, belts = b}
end
"""
ACCEPT_REC = r"""
local function items(line)
  local o = {}
  for _, it in pairs(line.get_detailed_contents()) do
    o[#o + 1] = {math.floor(it.position * 256 + 0.5), it.stack.name}
  end
  return o
end
HOOK = function(t)
  if t > TICKS then return end
  local row = {}
  for _, r in ipairs(RIGS) do
    local bb = {}
    for k, x in ipairs(r.belts) do
      bb[k] = {items(x.get_transport_line(1)), items(x.get_transport_line(2))}
    end
    row[#row + 1] = {r.name, bb}
  end
  P.log[#P.log + 1] = {t, row}
end
"""


def _accept_rigs() -> list[tuple[str, str, dict]]:
    rigs = []
    for lane, entry in ((1, 188), (2, 67)):
        for f in range(16, 24):
            # the feed item's entry, as a chain coordinate of the main at the
            # tick it arrives: 256 + entry, less the ticks it takes
            arrive = (f + 8) // 8
            centre = 256 + entry + 8 * arrive
            for c in range(centre - 40, centre + 41):
                rigs.append((f"side_l{lane}_f{f}_c{c}", "side", {"L": lane, "F": f, "C": c}))
    # the drop lands at 128 on belt 3 (chain coordinate 384) at t=47
    for c in range(384 + 8 * 47 - 40, 384 + 8 * 47 + 41):
        rigs.append((f"drop_c{c}", "drop", {"C": c}))
    return rigs


def family_accept(args) -> dict:
    rigs = _accept_rigs()
    per, jobs = 120, []
    for j in range(0, len(rigs), per):
        parts = ["prepare(110, 110, 340, 340)\nlocal TICKS = 70\nlocal RIGS = {}\n"]
        for n, (name, kind, v) in enumerate(rigs[j : j + per]):
            x, y = 112 + (n % 11) * 20, 112 + (n // 11) * 20
            vals = "".join(f"local {k} = {val}\n" for k, val in v.items())
            parts.append(f"do\nlocal X, Y, NAME = {x}, {y}, '{name}'\n{vals}"
                         + (ACCEPT_SIDE if kind == "side" else ACCEPT_DROP) + "end\n")  # fmt: skip
        parts.append(ACCEPT_REC)
        jobs.append((f"accept{j // per:02d}", "".join(parts)))
    res = _run_many(jobs, 75, args.parallel)
    out: dict = {"rigs": {name: {"kind": kind, **v} for name, kind, v in rigs}, "log": {}}
    for r in res.values():
        for t, row in r["log"]:
            for name, belts in row:
                out["log"].setdefault(name, []).append([t, belts])
    return out


# ---------------------------------------------------------------- sleep

#: Seven east belts at (306..312, 345), whose lanes merge late (the earliest
#: of their delays is 267), lane 1 filled 64 apart from the end, so every
#: item stands still; then the front is freed at t=`AT` by `HOW`:
#: `script` (LuaTransportLine.remove_item on the last belt), `ins` (an
#: inserter south of the last belt picks into a chest), `extend` (an eighth
#: belt built at the end). Freed at t=20 (young belts) or t=620 (every lane
#: merged long before).
SLEEP = r"""
R = {belts = {}, chests = {}, ins = {}}
for i = 0, 6 do R.belts[#R.belts + 1] = belt(306 + i, 345, D.east) end
for i = 1, 7 do
  for _, p in ipairs({0, 64, 128, 192}) do put(R.belts[i], 1, p, "iron-plate") end
end
local function free()
  if HOW == "script" then
    R.belts[7].get_transport_line(1).remove_item({name = "iron-plate", count = 1})
  elseif HOW == "ins" then
    local c = mk({name = "wooden-chest", position = {312.5, 347.5}})
    R.chests[1] = c
    local e = mk({name = "burner-inserter", position = {312.5, 346.5}, direction = D.north})
    e.insert({name = "coal", count = 5})
    R.ins[1] = e
  elseif HOW == "extend" then
    R.belts[8] = belt(313, 345, D.east)
  end
end
"""


def family_sleep(args) -> dict:
    jobs = []
    for how in ("script", "ins", "extend"):
        for at in (20, 620):
            rec = REC.replace("HOOK = function(t)\n",
                              "HOOK = function(t)\n  if t == AT then free() end\n")  # fmt: skip
            body = (f"prepare(296, 335, 330, 360)\nlocal TICKS = {at + 200}\n"
                    f"local HOW, AT = '{how}', {at}\n" + SLEEP + rec)  # fmt: skip
            jobs.append((f"{how}_{at}", body))
    res = _run_many(jobs, 830, args.parallel)
    return {name: {"log": r["log"]} for name, r in res.items()}


FAMILIES = {"delaymap": family_delaymap, "delaycheck": family_delaycheck,
            "order": family_order, "accept": family_accept, "sleep": family_sleep}  # fmt: skip


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--family", choices=sorted(FAMILIES), required=True)
    ap.add_argument("--rect", default="-128,-128,128,128", help="delaymap: x0,y0,x1,y1")
    ap.add_argument("--facing", type=int, default=0, choices=(0, 4), help="delaymap: 0 or 4")
    ap.add_argument("--rects", default="-128,-128,128,128;96,96,352,352",
                    help="delaycheck: rectangles to sample, ';'-separated")  # fmt: skip
    ap.add_argument("--sample", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260925)
    ap.add_argument("--ticks", type=int, default=0)
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    started = time.perf_counter()
    doc = FAMILIES[args.family](args)
    doc["engine"] = WorkerManager().engine.to_dict()
    doc["wall_seconds"] = round(time.perf_counter() - started, 1)
    if args.out:
        out = Path(args.out)
    elif args.family == "delaymap":
        RUNTIME.mkdir(exist_ok=True)
        tag = "" if args.facing == 0 else f"-f{args.facing}"
        out = RUNTIME / ("delaymap-" + args.rect.replace(",", "_") + tag + ".json.xz")
    else:
        out = EVIDENCE / f"logistics4-{args.family}.json.xz"
    out.write_bytes(lzma.compress(json.dumps(doc, sort_keys=True).encode()))
    print(f"wrote {out} ({out.stat().st_size // 1024} KiB), {doc['wall_seconds']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
