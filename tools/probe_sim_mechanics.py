"""Measure the early-game mechanics a simulator has to reproduce (M2).

The simulator plan lists what the engine has already pinned -- reach, placement
footprints, mining area, drill drop offsets, burner run-on -- and what it has
only assumed or never touched. This measures the second list, against Factorio
2.0.60 build 83512, so the C simulator reads its constants from evidence rather
than from comments:

**Read from prototypes**, which are authoritative for what they cover: stack
sizes, fuel values, mining and crafting speeds, energy use, inventory sizes,
recipe times, resource mining times, character running and mining speed.

**Measured tick by tick**, because each is engine behaviour rather than a single
prototype number:

- walking: distance per tick for each catalog stride, and whether it is an
  exact multiple of 1/256 (the hypothesis: 38/256 tiles per tick);
- hand-mining: ticks from issuing `mine` to the ore arriving;
- a burner drill emptying into a chest: ticks to its first ore, and the
  interval after that;
- a burner drill feeding an *unfuelled* furnace: how many ore it pushes in
  before it stops for space -- the insertion limit that sets a line's buffer;
- a stone furnace smelting: ticks to its first plate and the interval after;
- fuel drawn per tick by a working drill and a working furnace;
- the status each machine reports over time, as engine names.

Entities for the timing runs are created evaluator-side with raw Lua, off to the
side of the scene, and are never seen through the policy's action path. Time
advances through `env.advance`, one tick per sample, so every sample is
tick-aligned.

Run:
  uv run python tools/probe_sim_mechanics.py
"""

from __future__ import annotations

import json
import statistics
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

TASK = "construct_smelting_line"
TIMING_TICKS = 2400
#: The coarse phase that follows. Up to 60,000 ticks is room for a drill at one
#: ore per 240 ticks to fill any source slot a stack or two deep.
COARSE_TICKS = 60
COARSE_SAMPLES = 1000
BLOCKED_SAMPLES = 5
ITEMS = (
    "iron-ore", "copper-ore", "coal", "stone", "wood", "iron-plate", "copper-plate",
    "stone-brick", "iron-gear-wheel", "stone-furnace", "burner-mining-drill",
    "transport-belt", "burner-inserter", "iron-chest", "wooden-chest",
)  # fmt: skip
RECIPES = (
    "iron-plate", "copper-plate", "stone-brick", "iron-gear-wheel",
    "stone-furnace", "burner-mining-drill", "transport-belt", "burner-inserter",
    "iron-chest", "wooden-chest",
)  # fmt: skip
RESOURCES = ("iron-ore", "copper-ore", "coal", "stone")

PROTOTYPES_LUA = """
local function try(fn) local ok, v = pcall(fn) if ok then return v end return nil end
local out = {items = {}, recipes = {}, resources = {}, entities = {}}
for _, name in ipairs(%(items)s) do
  local p = prototypes.item[name]
  if p then out.items[name] = {stack_size = try(function() return p.stack_size end),
                               fuel_value = try(function() return p.fuel_value end)} end
end
for _, name in ipairs(%(recipes)s) do
  local r = prototypes.recipe[name]
  if r then
    local ingredients, products = {}, {}
    for _, i in ipairs(r.ingredients) do
      ingredients[#ingredients + 1] = {name = i.name, amount = i.amount}
    end
    for _, i in ipairs(r.products) do
      products[#products + 1] = {name = i.name, amount = i.amount}
    end
    out.recipes[name] = {energy = r.energy, category = r.category,
                         ingredients = ingredients, products = products}
  end
end
for _, name in ipairs(%(resources)s) do
  local p = prototypes.entity[name]
  if p then
    out.resources[name] = {
      mining_time = try(function() return p.mineable_properties.mining_time end),
    }
  end
end
for _, name in ipairs({"character", "burner-mining-drill", "stone-furnace"}) do
  local p = prototypes.entity[name]
  out.entities[name] = {
    running_speed = try(function() return p.running_speed end),
    mining_speed = try(function() return p.mining_speed end),
    crafting_speed = try(function() return p.get_crafting_speed() end)
      or try(function() return p.crafting_speed end),
    energy_usage = try(function() return p.energy_usage end),
    max_energy_usage = try(function() return p.get_max_energy_usage() end),
    fuel_inventory_size = try(function() return p.burner_prototype.fuel_inventory_size end),
    effectivity = try(function() return p.burner_prototype.effectivity end),
    mining_drill_radius = try(function() return p.mining_drill_radius end),
  }
end
local ch = storage.frrl_character
if ch and ch.valid then
  out.character_main_slots = try(function()
    return #ch.get_inventory(defines.inventory.character_main)
  end)
  out.character_reach = {
    build = try(function() return ch.build_distance end),
    reach = try(function() return ch.reach_distance end),
    resource = try(function() return ch.resource_reach_distance end),
    pickup = try(function() return ch.item_pickup_distance end),
  }
end
return helpers.table_to_json(out)
"""

#: Three rigs, placed off the scene's own ore so they cannot interfere with it:
#: a drill into a chest, a drill into an unfuelled furnace, and a lone furnace.
#: Each drill sits on ore it brings with it.
SETUP_LUA = """
local s = game.surfaces["nauvis"]
local force = game.forces.player
local ox, oy = %(ox)s, %(oy)s
local function ore(x, y)
  s.create_entity({name = "iron-ore", position = {x, y}, amount = 5000})
end
for dx = 0, 1 do
  for dy = 0, 1 do
    ore(ox + dx + 0.5, oy + dy + 0.5)
    ore(ox + 8 + dx + 0.5, oy + dy + 0.5)
  end
end
local rig = {}
rig.drill_a = s.create_entity({name = "burner-mining-drill", position = {ox + 1, oy + 1},
                               direction = defines.direction.south, force = force})
local drop_a = rig.drill_a.drop_position
rig.chest = s.create_entity({name = "iron-chest", position = drop_a, force = force})
rig.drill_b = s.create_entity({name = "burner-mining-drill", position = {ox + 9, oy + 1},
                               direction = defines.direction.south, force = force})
local drop_b = rig.drill_b.drop_position
rig.furnace_b = s.create_entity({name = "stone-furnace",
  position = {math.floor(drop_b.x) + 1, math.floor(drop_b.y) + 1}, force = force})
rig.furnace_c = s.create_entity({name = "stone-furnace",
  position = {ox + 17, oy + 1}, force = force})
storage.frrl_mechanics_probe = rig
local placed = {}
for k, e in pairs(rig) do placed[k] = e and e.valid and {e.position.x, e.position.y} or false end
placed.drop_a = {drop_a.x, drop_a.y}
placed.drop_b = {drop_b.x, drop_b.y}
-- An area around the drop point, as `observations.lua` resolves `drop_into`.
-- A bare `position` filter matches entity positions exactly, and reported
-- this rig as not catching when its ore was visibly arriving.
local caught = s.find_entities_filtered({
  area = {{drop_b.x - 0.1, drop_b.y - 0.1}, {drop_b.x + 0.1, drop_b.y + 0.1}},
  name = "stone-furnace"})[1]
placed.furnace_b_catches = caught ~= nil and caught == rig.furnace_b
return helpers.table_to_json(placed)
"""

START_LUA = """
local rig = storage.frrl_mechanics_probe
rig.drill_a.get_fuel_inventory().insert({name = "coal", count = 5})
-- A full stack: at 2,500 J a tick five coal is only 33 ore of work, short of
-- a limit a stack deep, and the drill would stop for fuel rather than space.
rig.drill_b.get_fuel_inventory().insert({name = "coal", count = 50})
rig.furnace_c.get_fuel_inventory().insert({name = "coal", count = 5})
local source = rig.furnace_c.get_inventory(defines.inventory.furnace_source)
source.insert({name = "iron-ore", count = 12})
return game.tick
"""

SAMPLE_LUA = """
local rig = storage.frrl_mechanics_probe
local names = {}
for k, v in pairs(defines.entity_status) do names[v] = k end
local function burner(e)
  local ok, b = pcall(function() return e.burner end)
  if not ok or not b then return nil end
  return {remaining = b.remaining_burning_fuel,
          fuel = e.get_fuel_inventory().get_item_count("coal")}
end
local function status(e)
  local ok, v = pcall(function() return e.status end)
  return ok and names[v] or nil
end
local function count(e, inventory, item)
  return e.get_inventory(inventory).get_item_count(item)
end
return helpers.table_to_json({
  tick = game.tick,
  chest_ore = count(rig.chest, defines.inventory.chest, "iron-ore"),
  drill_a = {status = status(rig.drill_a), burner = burner(rig.drill_a)},
  drill_b = {status = status(rig.drill_b), burner = burner(rig.drill_b)},
  furnace_b_source = count(rig.furnace_b, defines.inventory.furnace_source, "iron-ore"),
  furnace_c = {
    status = status(rig.furnace_c), burner = burner(rig.furnace_c),
    source = count(rig.furnace_c, defines.inventory.furnace_source, "iron-ore"),
    result = count(rig.furnace_c, defines.inventory.furnace_result, "iron-plate"),
  },
})
"""


def lua_list(names) -> str:
    return "{" + ",".join(json.dumps(n) for n in names) + "}"


def arrivals(samples: list[dict], read) -> list[int]:
    """Ticks at which a monotone count increased, one entry per unit."""
    ticks, previous = [], read(samples[0])
    for sample in samples[1:]:
        value = read(sample)
        for _ in range(max(0, value - previous)):
            ticks.append(sample["tick"])
        previous = value
    return ticks


def intervals(ticks: list[int]) -> dict:
    gaps = [b - a for a, b in zip(ticks, ticks[1:], strict=False)]
    return {
        "count": len(ticks),
        "first_tick_offset": None,
        "median_interval": statistics.median(gaps) if gaps else None,
        "intervals": sorted(set(gaps)),
    }


def transitions(samples: list[dict], read) -> list[list]:
    out, previous = [], object()
    for sample in samples:
        value = read(sample)
        if value != previous:
            out.append([sample["tick"], value])
            previous = value
    return out


def measure_walking(env: FactorioEnv) -> dict:
    keys = env.catalog.keys()
    result = {}
    for key in ("nudge_east", "step_east", "move_east", "nudge_west", "step_west", "move_west"):
        before = list((env._observation.get("character") or {}).get("position") or [0, 0])
        tick_before = int(env._observation.get("tick") or 0)
        env.step(keys.index(key))
        after = list((env._observation.get("character") or {}).get("position") or [0, 0])
        ticks = int(env._observation.get("tick") or 0) - tick_before
        template = env.catalog.templates[keys.index(key)]
        stride = int(template.payload.get("ticks") or 0)
        dx = after[0] - before[0]
        result[key] = {
            "stride_ticks": stride,
            "decision_ticks": ticks,
            "dx": dx,
            "per_stride_tick": dx / stride if stride else None,
            "dx_in_256ths": dx * 256,
            "exact_256ths": float(dx * 256).is_integer(),
        }
    return result


def measure_hand_mining(env: FactorioEnv, repeats: int = 3) -> dict:
    keys = env.catalog.keys()
    mine = keys.index("mine_at")
    tiles = (env._observation.get("resources") or {}).get("tiles") or []
    here = (env._observation.get("character") or {}).get("position") or [0, 0]
    reachable = [
        t for t in tiles if ((t["p"][0] - here[0]) ** 2 + (t["p"][1] - here[1]) ** 2) ** 0.5 <= 2.5
    ]
    if not reachable:
        return {"error": "no ore within resource reach of the character"}
    target = str(
        min(reachable, key=lambda t: (t["p"][0] - here[0]) ** 2 + (t["p"][1] - here[1]) ** 2)["h"]
    )
    durations = []
    for _ in range(repeats):
        have = int((env._observation.get("inventory") or {}).get("iron-ore", 0))
        start = int(env._observation.get("tick") or 0)
        env.step_arguments(mine, {"handle": target})
        while int((env._observation.get("inventory") or {}).get("iron-ore", 0)) <= have:
            env.advance(1)
            if int(env._observation.get("tick") or 0) - start > 1200:
                return {"error": "no ore arrived within 1200 ticks", "durations": durations}
        durations.append(int(env._observation.get("tick") or 0) - start)
        # Let the settled mine release its slot before issuing the next one.
        env.advance(env.spec_.decision_ticks)
    return {"ticks_from_issue_to_ore": durations}


def main() -> int:
    task = get(TASK)
    manager = WorkerManager()
    handle = manager.launch("sim-mechanics")
    report: dict = {"engine": manager.engine.to_dict(), "task": TASK, "timing_ticks": TIMING_TICKS}
    started = time.perf_counter()
    try:
        with RCONClient(handle.spec.rcon_endpoint, timeout=60.0) as client:
            client.lua(f"game.speed = {resolve_game_speed()} return game.speed")
            session = WorkerSession(handle, timeout=60.0)
            session.status()
            env = FactorioEnv(task, session, SeedPlan(master=777, run_id="sim-mechanics"),
                              branch=Branch.TRAIN, split="train")  # fmt: skip
            env._episode_index = -1
            env.reset()

            report["prototypes"] = json.loads(
                str(
                    client.lua(
                        PROTOTYPES_LUA
                        % {
                            "items": lua_list(ITEMS),
                            "recipes": lua_list(RECIPES),
                            "resources": lua_list(RESOURCES),
                        }
                    )
                ).strip()  # fmt: skip
            )
            report["walking"] = measure_walking(env)

            # Walk onto the task's ore before mining by hand.
            patch = (env._truth.get("markers") or {}).get("patch")
            if patch:
                from factoriorl.tasks.reference import Driver, SolveTrace

                driver = Driver(env, SolveTrace(task=TASK, budget=task.spec.max_decision_steps))
                driver.walk_to(tuple(patch), tolerance=1.0)
            report["hand_mining"] = measure_hand_mining(env)

            # Timing rigs, well away from the scene.
            report["rigs"] = json.loads(str(client.lua(SETUP_LUA % {"ox": 60, "oy": 60})).strip())
            start_tick = int(str(client.lua(START_LUA)).strip())
            samples = [json.loads(str(client.lua(SAMPLE_LUA)).strip())]
            for _ in range(TIMING_TICKS):
                env.advance(1)
                samples.append(json.loads(str(client.lua(SAMPLE_LUA)).strip()))
            # The first run stopped here and reported "9 ore" as the drill's
            # insertion limit into an unfuelled furnace. It was not a limit:
            # the drill makes one ore per 240 ticks, so 2,400 ticks is ten ore,
            # and its status never left `working`. Run on, coarsely, until it
            # has been blocked for several samples in a row.
            coarse = []
            blocked_run = 0
            for _ in range(COARSE_SAMPLES):
                env.advance(COARSE_TICKS)
                sample = json.loads(str(client.lua(SAMPLE_LUA)).strip())
                coarse.append(sample)
                blocked = sample["drill_b"]["status"] == "waiting_for_space_in_destination"
                blocked_run = blocked_run + 1 if blocked else 0
                if blocked_run >= BLOCKED_SAMPLES:
                    break
            session.close()
    finally:
        manager.cleanup(handle)

    drill_ore = arrivals(samples, lambda s: s["chest_ore"])
    plates = arrivals(samples, lambda s: s["furnace_c"]["result"])
    drill = intervals(drill_ore)
    drill["first_tick_offset"] = drill_ore[0] - start_tick if drill_ore else None
    furnace = intervals(plates)
    furnace["first_tick_offset"] = plates[0] - start_tick if plates else None

    def burn_rate(read, status) -> float | None:
        # Joules per tick over consecutive ticks the machine reported working:
        # the fall in remaining fuel plus the fuel value of coal taken from its
        # slot. The first run averaged over the whole window, idle ticks
        # included, and read a furnace that had finished its ore as 1,440.6
        # against a prototype 1,500.
        coal = report["prototypes"]["items"].get("coal", {}).get("fuel_value") or 0
        spent, ticks = 0.0, 0
        for before, after in zip(samples, samples[1:], strict=False):
            if status(before) != "working" or status(after) != "working":
                continue
            a, b = read(before), read(after)
            if not a or not b:
                continue
            spent += (a["remaining"] - b["remaining"]) + (a["fuel"] - b["fuel"]) * coal
            ticks += after["tick"] - before["tick"]
        return spent / ticks if ticks else None

    report["drill_into_chest"] = {
        **drill,
        "status_transitions": transitions(samples, lambda s: s["drill_a"]["status"]),
        "joules_per_tick": burn_rate(
            lambda s: s["drill_a"]["burner"], lambda s: s["drill_a"]["status"]
        ),
    }
    everything = samples + coarse
    held_when_blocked = [
        s["furnace_b_source"]
        for s in coarse
        if s["drill_b"]["status"] == "waiting_for_space_in_destination"
    ]
    report["drill_into_unfuelled_furnace"] = {
        "catches": report["rigs"].get("furnace_b_catches"),
        "blocked": blocked_run >= BLOCKED_SAMPLES,
        "source_ore_when_blocked": sorted(set(held_when_blocked)),
        "max_source_ore": max(s["furnace_b_source"] for s in everything),
        "ticks_observed": everything[-1]["tick"] - start_tick,
        "status_transitions": transitions(everything, lambda s: s["drill_b"]["status"]),
    }
    report["furnace_smelting"] = {
        **furnace,
        "status_transitions": transitions(samples, lambda s: s["furnace_c"]["status"]),
        "joules_per_tick": burn_rate(
            lambda s: s["furnace_c"]["burner"], lambda s: s["furnace_c"]["status"]
        ),
    }
    report["wall_seconds"] = round(time.perf_counter() - started, 1)
    walking = report["walking"]
    report["summary"] = {
        "walk_dx_per_tick": sorted(
            {round(w["per_stride_tick"], 7) for w in walking.values() if w["per_stride_tick"]}
        ),
        "walk_exact_256ths": all(w["exact_256ths"] for w in walking.values()),
        "hand_mining_ticks": report["hand_mining"].get("ticks_from_issue_to_ore"),
        "drill_first_ore_tick": drill["first_tick_offset"],
        "drill_ore_interval": drill["median_interval"],
        "drill_joules_per_tick": report["drill_into_chest"]["joules_per_tick"],
        "drill_to_unfuelled_furnace_blocked": report["drill_into_unfuelled_furnace"]["blocked"],
        "drill_to_unfuelled_furnace_limit": (
            report["drill_into_unfuelled_furnace"]["source_ore_when_blocked"]
        ),
        "furnace_first_plate_tick": furnace["first_tick_offset"],
        "furnace_plate_interval": furnace["median_interval"],
        "furnace_joules_per_tick": report["furnace_smelting"]["joules_per_tick"],
        "character_main_slots": report["prototypes"].get("character_main_slots"),
    }  # fmt: skip
    out = ROOT / "docs" / "evidence" / "sim-mechanics-m1.json"
    out.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, default=str))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
