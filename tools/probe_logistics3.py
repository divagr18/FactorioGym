"""Measure belt-line segments, turn geometry and sideload order (M4, logistics 3).

The first two logistics probes left three behaviours unexplained: two items
sideloading onto one lane in the same tick (one moves 8/256 further), where
the arm aims for an item on a turn, and a burner inserter that sleeps with ore
a few belts upstream. This probe measures what lies under all three: the
engine's belt-line *segments* (lines of consecutive belts merged into one
object, visible as `LuaTransportLine.line_equals`), when they merge and split,
and the order the engine moves lines in. docs/sim-logistics.md, "Third probe",
states the findings. Families (`--family`):

- `turns`: `get_line_item_position` at every position of both lanes of all
  eight turns (four facings, left and right), as offsets from the belt centre
  in 1/256 tile.
- `delay`: the merge delay of every belt tile and lane in a rectangle
  (`--area x0,y0,x1,y1`, default the scene area around the origin). A lone
  north-facing belt is built at each measured tile and a neighbour north of it
  is destroyed and rebuilt every tick, so the neighbour's own delay never runs
  out; the tick the two lines become equal is the measured tile's delay.
  Six worlds (tile offsets 2 x 3) cover the rectangle.
- `segments`: installs the parity scenarios `logistics_smelting_chain`,
  `logistics_sideload_merge` and `logistics_belt_pickup` as the recorder does
  and logs, every tick it changes, which belts' lines are equal (per lane) and
  every inserter's buffer.
- `sideload`: one fresh world per rig: a three-belt east main, a north feed of
  F belts into its middle belt, one item on each feed lane placed on feed belt
  `where` at 128, in a chosen order, at a chosen tick; or two burner inserters
  dropping them (`ins_*`). Records every item on every belt every tick.

Output: `docs/evidence/logistics3-<family>.json.xz`.

Run (about 20 s, 30 s, 65 s and 100 s):
  uv run python tools/probe_logistics3.py --family turns
  uv run python tools/probe_logistics3.py --family delay
  uv run python tools/probe_logistics3.py --family segments
  uv run python tools/probe_logistics3.py --family sideload
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import lzma
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

#: Common Lua: helpers, a cleared grass area far from the scene, and the
#: per-tick recorder. `BODY` builds rigs into `P.rigs[name] = {belts = {...}}`
#: and may set `HOOK(t)` (called first every tick) and `P.extra`.
PRE = r"""
local s = game.surfaces[1]
local D = defines.direction
local function g(v) return string.format("%.17g", v) end
local function p256(v)
  local q = v * 256
  if q == math.floor(q) then return math.floor(q) end
  return g(v)
end
if not storage.p3init then
  s.request_to_generate_chunks({250, 250}, 9)
  s.force_generate_chunk_requests()
  for _, e in pairs(s.find_entities_filtered({area = {{100, 100}, {480, 480}}})) do
    if e.valid and e.type ~= "character" then e.destroy() end
  end
  local tiles = {}
  for x = 100, 480 do
    for y = 100, 480 do tiles[#tiles + 1] = {name = "grass-1", position = {x, y}} end
  end
  s.set_tiles(tiles)
  storage.p3init = true
end
storage.p3 = {log = {}, rigs = {}, merged = {}}
local P = storage.p3
local function mk(spec)
  spec.force = spec.force or "player"
  local e = s.create_entity(spec)
  assert(e, "create failed " .. spec.name)
  return e
end
local function belt(x, y, d)
  return mk({name = "transport-belt", position = {x + 0.5, y + 0.5}, direction = d})
end
local function put(b, lane, pos, name)
  return b.get_transport_line(lane).insert_at(pos / 256, {name = name or "copper-plate", count = 1})
end
local function line_items(line)
  local out = {}
  for _, it in pairs(line.get_detailed_contents()) do
    out[#out + 1] = {p256(it.position), it.unique_id}
  end
  return out
end
local function chest_with(x, y, item)
  mk({name = "wooden-chest", position = {x + 0.5, y + 0.5}}).insert({name = item, count = 50})
end
local function fuelled_inserter(x, y, d)
  mk({name = "burner-inserter", position = {x + 0.5, y + 0.5}, direction = d})
    .insert({name = "coal", count = 2})
end
local function clear(x0, y0, x1, y1)
  for _, e in pairs(s.find_entities_filtered({area = {{x0, y0}, {x1, y1}}})) do
    if e.valid and e.type ~= "character" and e.type ~= "resource" then e.destroy() end
  end
  local around = {{x0 - 30, y0 - 30}, {x1 + 30, y1 + 30}}
  for _, c in pairs(s.find_entities_filtered({area = around, type = "character"})) do
    c.teleport({x0 - 20, y1 + 20})
  end
end
local HOOK = nil
"""

POST = r"""
P.t0 = game.tick
local function sample()
  local row = {}
  for name, r in pairs(P.rigs) do
    local rr = {}
    for i, b in ipairs(r.belts or {}) do
      rr[i] = b.valid and {line_items(b.get_transport_line(1)),
        line_items(b.get_transport_line(2))} or "gone"
    end
    row[name] = rr
  end
  return row
end
local old = script.get_event_handler(defines.events.on_tick)
script.on_event(defines.events.on_tick, function(e)
  if old then old(e) end
  if P.last == game.tick then return end
  P.last = game.tick
  local t = game.tick - P.t0
  if HOOK then HOOK(t) end
  if P.record then P.log[#P.log + 1] = {t, sample()} end
end)
return P.extra or true
"""


def _run_world(body: str, ticks: int, label: str) -> dict:
    manager = WorkerManager()
    handle = manager.launch(label)
    try:
        with RCONClient(handle.spec.rcon_endpoint, timeout=900.0) as client:
            client.lua(f"game.speed = {resolve_game_speed()} return game.speed")
            session = WorkerSession(handle, timeout=900.0)
            session.status()
            env = FactorioEnv(get("construct_smelting_line"), session,
                              SeedPlan(master=41, run_id=label), branch=Branch.TRAIN,
                              split="train")  # fmt: skip
            env.reset(options={"scene_index": 0})
            extra = client.lua(PRE + body + POST)
            env.advance(ticks)
            n = client.lua("return #storage.p3.log")
            log = []
            for a in range(1, n + 1, 200):
                log += client.lua(
                    f"local o = {{}} for i = {a}, math.min({a + 199}, #storage.p3.log) do "
                    "o[#o + 1] = storage.p3.log[i] end return o"
                )
            merged = client.lua("return storage.p3.merged")
            session.close()
    finally:
        manager.cleanup(handle)
    return {"extra": extra, "log": log, "merged": merged}


def _run_many(jobs: list[tuple[str, str]], ticks: int, parallel: int) -> dict:
    out: dict = {}

    def one(k: int) -> None:
        name, body = jobs[k]
        out[name] = _run_world(body, ticks, f"probe3-{k:03d}")

    with ThreadPoolExecutor(parallel) as pool:
        list(pool.map(one, range(len(jobs))))
    return out


# ---------------------------------------------------------------- turns

TURNS = r"""
local out = {}
local V = {[0] = {0, -1}, [4] = {1, 0}, [8] = {0, 1}, [12] = {-1, 0}}
local k = 0
for _, F in ipairs({0, 4, 8, 12}) do
  for _, turn in ipairs({"right", "left"}) do
    local fd = turn == "right" and (F + 12) % 16 or (F + 4) % 16
    local bx, by = 150 + 5 * k, 150
    k = k + 1
    local t = belt(bx, by, F)
    belt(bx - V[fd][1], by - V[fd][2], fd)
    local rec = {shape = t.belt_shape, facing = F}
    for lane = 1, 2 do
      local l = t.get_transport_line(lane)
      local pts = {}
      local len = math.floor(l.line_length * 256 + 0.5)
      for p = 0, len do
        local q = l.get_line_item_position(p / 256)
        pts[#pts + 1] = {(q.x - t.position.x) * 256, (q.y - t.position.y) * 256}
      end
      rec["lane" .. lane] = pts
    end
    out[#out + 1] = rec
  end
end
P.extra = out
"""


def family_turns(args) -> dict:
    res = _run_world(TURNS, 1, "probe3-turns")
    return {"turns": res["extra"]}


# ---------------------------------------------------------------- merge delay


def delay_body(tiles: list[tuple[int, int]], clear: tuple[int, int, int, int] | None) -> str:
    spec = ",".join(f"{{{x},{y}}}" for x, y in tiles)
    wipe = ""
    if clear:
        x0, y0, x1, y1 = clear
        wipe = f"clear({x0 - 2}, {y0 - 2}, {x1 + 2}, {y1 + 2})\n"
    return (
        wipe
        + "local specs = {"
        + spec
        + "}"
        + """
local X, N = {}, {}
for k, p in ipairs(specs) do X[k] = belt(p[1], p[2], D.north) end
HOOK = function(t)
  for k, p in ipairs(specs) do
    if N[k] and N[k].valid then
      for lane = 1, 2 do
        local key = k * 2 + lane
        local here, there = X[k].get_transport_line(lane), N[k].get_transport_line(lane)
        if not P.merged[key] and here.line_equals(there) then
          P.merged[key] = {p[1], p[2], lane, t}
        end
      end
    end
    if not (P.merged[k * 2 + 1] and P.merged[k * 2 + 2]) then
      if N[k] and N[k].valid then N[k].destroy() end
      N[k] = belt(p[1], p[2] - 1, D.north)
    end
  end
end
"""
    )


def family_delay(args) -> dict:
    x0, y0, x1, y1 = (int(v) for v in args.area.split(","))
    jobs = []
    for ox in range(2):
        for oy in range(3):
            tiles = [(x, y) for x in range(x0 + ox, x1 + 1, 2) for y in range(y0 + oy, y1 + 1, 3)]
            jobs.append((f"o{ox}{oy}", delay_body(tiles, (x0, y0, x1, y1))))
    res = _run_many(jobs, args.ticks or 700, 6)
    table = {}
    for r in res.values():
        for x, y, lane, t in r["merged"].values():
            table[f"{x},{y},{lane}"] = t
    return {"area": [x0, y0, x1, y1], "delay": table}


# ---------------------------------------------------------------- segments

SEGMENTS = r"""
local s = game.surfaces[1]
local function by_pos(a, b)
  if a.position.y ~= b.position.y then return a.position.y < b.position.y end
  return a.position.x < b.position.x
end
local belts = s.find_entities_filtered({type = "transport-belt"})
table.sort(belts, by_pos)
local ins = s.find_entities_filtered({type = "inserter"})
table.sort(ins, by_pos)
storage.sg = {log = {}, t0 = game.tick}
local st = storage.sg
local function rec()
  local out = {}
  for i, b in ipairs(belts) do
    local r = {}
    for lane = 1, 2 do
      local l = b.get_transport_line(lane)
      local eq = {}
      for j, c in ipairs(belts) do
        if l.line_equals(c.get_transport_line(lane)) then eq[#eq + 1] = j end
      end
      r[lane] = table.concat(eq, ",")
    end
    out[i] = r
  end
  local e = {}
  for i, x in ipairs(ins) do e[i] = string.format("%.3f", x.energy) end
  return helpers.table_to_json({out, e})
end
local old = script.get_event_handler(defines.events.on_tick)
script.on_event(defines.events.on_tick, function(ev)
  if old then old(ev) end
  if st.last == game.tick then return end
  st.last = game.tick
  local r = rec()
  if r ~= st.prev then st.log[#st.log + 1] = {game.tick - st.t0, r} st.prev = r end
end)
local bp, ip = {}, {}
for i, b in ipairs(belts) do bp[i] = {b.position.x, b.position.y, b.direction, b.belt_shape} end
for i, e in ipairs(ins) do ip[i] = {e.position.x, e.position.y, e.direction} end
return {belts = bp, inserters = ip}
"""

SCENARIOS = ("logistics_smelting_chain", "logistics_sideload_merge", "logistics_belt_pickup")


def _segments_of(name: str, ticks: int) -> dict:
    spec = importlib.util.spec_from_file_location("rpt", ROOT / "tools" / "record_parity_trace.py")
    rpt = importlib.util.module_from_spec(spec)
    sys.modules["rpt"] = rpt
    spec.loader.exec_module(rpt)
    scenario = next(s for s in rpt.SCENARIOS if s.name == name)
    manager = WorkerManager()
    handle = manager.launch(f"probe3-{name[10:20]}")
    try:
        with RCONClient(handle.spec.rcon_endpoint, timeout=900.0) as client:
            client.lua(f"game.speed = {resolve_game_speed()} return game.speed")
            session = WorkerSession(handle, timeout=900.0)
            session.status()
            env, _ = rpt.make_env(scenario, session)
            env.reset(options={"scene_index": scenario.episode_index})
            info = client.lua(SEGMENTS)
            env.advance(ticks)
            log = client.lua("return storage.sg.log")
            session.close()
    finally:
        manager.cleanup(handle)
    return {"layout": info, "log": [[t, json.loads(r)] for t, r in log]}


def family_segments(args) -> dict:
    return {name: _segments_of(name, args.ticks or 1500) for name in SCENARIOS}


# ---------------------------------------------------------------- sideload order


def sideload_rig(ox: int, oy: int, feed: int, where: int, order: str, at: int = 0) -> str:
    """Main (ox..ox+2, oy) east; feed x = ox+1, y = oy+1..oy+feed, north. `order`
    is "12" (lane 1 item placed first) or "21"; placed at tick `at`."""
    first = int(order[0])
    return f"""
local main, feed = {{}}, {{}}
for i = 0, 2 do main[#main + 1] = belt({ox} + i, {oy}, D.east) end
for k = 1, {feed} do feed[k] = belt({ox} + 1, {oy} + k, D.north) end
local bl = {{main[1], main[2], main[3]}}
for k = 1, {feed} do bl[#bl + 1] = feed[k] end
P.rigs.a = {{belts = bl}}
P.record = true
local function go() put(feed[{where}], {first}, 128) put(feed[{where}], {3 - first}, 128) end
if {at} == 0 then go() else HOOK = function(t) if t == {at} then go() end end end
"""


def inserter_rig(ox: int, oy: int, feed: int, order: str) -> str:
    """Two burner inserters drop copper onto feed belt `feed` (the farthest),
    one from each side; `order` "WE" builds the west one first."""
    yk = oy + feed
    side = {
        "W": f"chest_with({ox - 1}, {yk}, 'copper-plate') fuelled_inserter({ox}, {yk}, D.west)",
        "E": f"chest_with({ox + 3}, {yk}, 'copper-plate') fuelled_inserter({ox + 2}, {yk}, D.east)",
    }
    return f"""
local bl = {{}}
for i = 0, 2 do bl[#bl + 1] = belt({ox} + i, {oy}, D.east) end
for k = 1, {feed} do bl[#bl + 1] = belt({ox} + 1, {oy} + k, D.north) end
{side[order[0]]}
{side[order[1]]}
P.rigs.a = {{belts = bl}}
P.record = true
"""


def sideload_jobs() -> list[tuple[str, str]]:
    jobs = []
    # feed length and where the items start, young belts, at (300, 300)
    for feed, where in ((1, 1), (2, 2), (3, 3), (4, 4), (4, 1), (4, 2)):
        for order in ("12", "21"):
            jobs.append(
                (f"young_f{feed}w{where}_{order}", sideload_rig(300, 300, feed, where, order))
            )
    # the same feed placed one tile to the side: the young result changes
    for feed, where in ((4, 1), (4, 2), (4, 3), (4, 4)):
        jobs.append((f"young_at183_f{feed}w{where}_12", sideload_rig(183, 113, feed, where, "12")))
    # belts 600 ticks old
    for feed, where in ((4, 1), (4, 2), (4, 3), (4, 4)):
        for ox in (183, 184):
            jobs.append(
                (f"old_at{ox}_f{feed}w{where}_12", sideload_rig(ox, 113, feed, where, "12", 600))
            )
    # inserter drops, both build orders
    for feed in (1, 2, 3):
        for order in ("WE", "EW"):
            jobs.append((f"ins_f{feed}_{order}", inserter_rig(200, 200, feed, order)))
    return jobs


def family_sideload(args) -> dict:
    jobs = sideload_jobs()
    res = _run_many(jobs, args.ticks or 760, 8)
    return {name: {"log": r["log"]} for name, r in res.items()}


FAMILIES = {"turns": family_turns, "delay": family_delay, "segments": family_segments,
            "sideload": family_sideload}  # fmt: skip


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--family", choices=sorted(FAMILIES), required=True)
    ap.add_argument("--area", default="-6,-22,35,6", help="delay: x0,y0,x1,y1 (tiles)")
    ap.add_argument("--ticks", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    started = time.perf_counter()
    doc = FAMILIES[args.family](args)
    doc["engine"] = WorkerManager().engine.to_dict()
    doc["wall_seconds"] = round(time.perf_counter() - started, 1)
    out = EVIDENCE / f"logistics3-{args.family}{('-' + args.tag) if args.tag else ''}.json.xz"
    out.write_bytes(lzma.compress(json.dumps(doc, sort_keys=True).encode()))
    print(f"wrote {out} ({out.stat().st_size // 1024} KiB), {doc['wall_seconds']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
