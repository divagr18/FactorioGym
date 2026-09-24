"""When does a sleeping burner inserter wake up for a belt? (companion to probe_inserter_chase)

A burner inserter idle at its pickup point, with nothing on the belt line it
picks from, goes to sleep and stops refilling its burner buffer. The buffer it
wakes with limits its first move, so the simulator must know when it wakes.
This probe keeps every inserter asleep, adds one item in some way, and records
the inserter's buffer (`energy`) every tick: a sleeping inserter's buffer
jumps back to full (2560 J) on the tick it wakes.

Groups (12 rigs each; a rig is 8 east belts, a north-facing
inserter under belt 5, a chest):

- `up`: item inserted on belt 1 (four belts upstream) at t=600.
- `long`: a 40-belt line, item inserted 35 belts upstream at t=600.
- `side`: item inserted at t=600 on a belt that sideloads into belt 2.
- `other`: item inserted at t=600 on an unconnected belt two tiles away.
- `down`: item inserted at t=600 on belt 8, downstream of the pickup belt.
- `drop`: at t=600 an item goes into a chest that a second inserter empties
  onto belt 1.
- `young`: everything built at t=0, item inserted on belt 1 at t=40 (belts 40
  ticks old).
- `young_belt_old_ins`: belts built at t=0, inserter built at t=400, item at
  t=440.
- `old_belt_young_ins`: inserter built at t=0, belts at t=400, item at t=440.
- `all_late`: everything built at t=400, item at t=440.

Output: `docs/evidence/inserter-wake.json` with, per rig, the ticks at which
the buffer changed (`[t, buffer, held]`). `t` is the tick after which the
state was read.

Run:
  uv run python tools/probe_inserter_wake.py
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

OUT = ROOT / "docs" / "evidence" / "inserter-wake.json"
TICKS = 1100

LUA = r"""
local s = game.surfaces[1]
local D = defines.direction
local OX, OY = 300, 300
s.request_to_generate_chunks({OX + 70, OY + 120}, 6)
s.force_generate_chunk_requests()
for _, e in pairs(s.find_entities_filtered({area = {{OX - 10, OY - 10}, {OX + 160, OY + 260}}})) do
  if e.type ~= "character" then e.destroy() end
end
storage.wk = {rigs = {}, log = {}}
local st = storage.wk
local function belt(x, y, d)
  return s.create_entity{name = "transport-belt", position = {x + 0.5, y + 0.5}, direction = d,
    force = "player"}
end
local function build_belts(r)
  r.belts = {}
  for i = 1, r.n do r.belts[i] = belt(r.x + i, r.y, D.east) end
  if r.kind == "side" then r.feed = belt(r.x + 2, r.y - 1, D.south) end
  if r.kind == "other" then r.other = belt(r.x + 2, r.y - 2, D.east) end
end
local function build_ins(r)
  r.ins = s.create_entity{name = "burner-inserter", position = {r.x + r.pick + 0.5, r.y + 1.5},
    direction = D.north, force = "player"}
  s.create_entity{name = "wooden-chest", position = {r.x + r.pick + 0.5, r.y + 2.5},
    force = "player"}
  r.ins.insert{name = "coal", count = 5}
  if r.kind == "drop" then
    r.src = s.create_entity{name = "wooden-chest", position = {r.x + 1.5, r.y - 1.5},
      force = "player"}
    r.dins = s.create_entity{name = "burner-inserter", position = {r.x + 1.5, r.y - 0.5},
      direction = D.north, force = "player"}
    r.dins.insert{name = "coal", count = 5}
  end
end
local groups = {"up", "long", "side", "other", "down", "drop", "young", "young_belt_old_ins",
  "old_belt_young_ins", "all_late"}
local row = 0
for _, kind in ipairs(groups) do
  for rep = 0, 11 do
    local r = {kind = kind, rep = rep, n = (kind == "long") and 40 or 8,
      pick = (kind == "long") and 36 or 5}
    r.x = OX + (kind == "long" and 0 or (rep % 3) * 12)
    r.y = OY + row * 5
    if kind == "long" or rep % 3 == 2 then row = row + 1 end
    local belts_at = (kind == "old_belt_young_ins" or kind == "all_late") and 400 or 0
    local ins_at = (kind == "young_belt_old_ins" or kind == "all_late") and 400 or 0
    r.belts_at, r.ins_at = belts_at, ins_at
    if belts_at == 0 then build_belts(r) end
    if ins_at == 0 then build_ins(r) end
    r.item_at = (kind == "young") and 40 or (belts_at + ins_at > 0 and 440 or 600)
    st.rigs[#st.rigs + 1] = r
  end
end
st.t0 = game.tick
local old = script.get_event_handler(defines.events.on_tick)
script.on_event(defines.events.on_tick, function(e)
  if old then old(e) end
  if st.last == game.tick then return end
  st.last = game.tick
  local t = game.tick - st.t0
  local rowv = {}
  for k, r in ipairs(st.rigs) do
    if r.ins and r.ins.valid then
      rowv[k] = {r.ins.energy, r.ins.held_stack.valid_for_read and 1 or 0}
    else
      rowv[k] = {-1, 0}
    end
    if t == r.belts_at and t > 0 then build_belts(r) end
    if t == r.ins_at and t > 0 then build_ins(r) end
    if t == r.item_at then
      local item = {name = "iron-plate", count = 1}
      if r.kind == "side" then r.feed.get_transport_line(1).insert_at(0.5, item)
      elseif r.kind == "other" then r.other.get_transport_line(2).insert_at(0.5, item)
      elseif r.kind == "down" then r.belts[8].get_transport_line(2).insert_at(0.5, item)
      elseif r.kind == "drop" then r.src.insert(item)
      else r.belts[1].get_transport_line(2).insert_at(0.5, item) end
    end
  end
  st.log[#st.log + 1] = {t - 1, rowv}
end)
local kinds = {}
for i, r in ipairs(st.rigs) do kinds[i] = r.kind end
return kinds
"""


def main() -> int:
    manager = WorkerManager()
    handle = manager.launch("inserter-wake")
    started = time.perf_counter()
    try:
        with RCONClient(handle.spec.rcon_endpoint, timeout=300.0) as client:
            client.lua(f"game.speed = {resolve_game_speed()} return game.speed")
            session = WorkerSession(handle, timeout=300.0)
            session.status()
            env = FactorioEnv(
                get("construct_smelting_line"),
                session,
                SeedPlan(master=41, run_id="inserter-wake"),
                branch=Branch.TRAIN,
                split="train",
            )
            env.reset(options={"scene_index": 0})
            kinds = client.lua(LUA)
            env.advance(TICKS)
            log = client.lua("return storage.wk.log")
            session.close()
    finally:
        manager.cleanup(handle)
    rigs = []
    for k, kind in enumerate(kinds):
        changes, prev = [], None
        for t, row in log:
            value = [round(row[k][0], 6), row[k][1]]
            if value != prev:
                changes.append([t] + value)
                prev = value
        rigs.append({"kind": kind, "changes": changes})
    doc = {
        "engine": manager.engine.to_dict(),
        "ticks": TICKS,
        "wall_seconds": round(time.perf_counter() - started, 1),
        "rigs": rigs,
    }
    OUT.write_text(json.dumps(doc, indent=0) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    for kind in dict.fromkeys(kinds):
        woke = []
        for r in rigs:
            if r["kind"] != kind:
                continue
            item_at = (
                40
                if kind == "young"
                else (
                    440 if kind in ("young_belt_old_ins", "old_belt_young_ins", "all_late") else 600
                )
            )
            w = next((c[0] for c in r["changes"] if c[0] >= item_at and c[1] == 2560), None)
            woke.append(None if w is None else w - item_at)
        print(f"{kind:>20}: ticks from item to refill {woke}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
