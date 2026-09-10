-- Local structured observations (PLAN.md 2.3), assembled from the sensor
-- sweep, remembered observations, entity handles and the in-flight registry,
-- then filtered through the observation profile's declared key list.

local sensor = require("sensor")
local memory = require("memory")
local profiles = require("profiles")
local handles = require("handles")
local inflight = require("inflight")
-- `world` requires only `handles`, so this cannot close a require cycle.
local world = require("world")

local observations = {}

local function inventory_contents(inv)
  local out = {}
  if not inv then return out end
  for _, stack in pairs(inv.get_contents()) do
    out[stack.name] = (out[stack.name] or 0) + stack.count
  end
  return out
end

--- Recipe names the character's force can currently craft.
--
-- Capped and sorted so the list is bounded and order-independent: an
-- unbounded, order-varying list would make two identical scenes encode
-- differently between steps.
local RECIPE_CAP = 64
local function enabled_recipes()
  local force = game.forces.player
  if not force then return {} end
  local names = {}
  for name, recipe in pairs(force.recipes) do
    -- A recipe with no products cannot make anything, and `parameter-0`
    -- through `parameter-9` are exactly that: enabled, not hidden, zero
    -- ingredients and zero products, placeholders for parametrised
    -- blueprints. Measured on 2.0.60. They occupied 10 of the 22 observable
    -- recipes, so 45% of the `recipe` argument dimension was unusable by
    -- either client. Filtered on the property rather than the name prefix.
    local makes_something = recipe.products and #recipe.products > 0
    if recipe.enabled and not recipe.hidden and makes_something then
      names[#names + 1] = name
    end
  end
  table.sort(names)
  if #names > RECIPE_CAP then
    local capped = {}
    for index = 1, RECIPE_CAP do capped[index] = names[index] end
    return capped
  end
  return names
end

local function character_state(ch, observation_profile)
  if not ch or not ch.valid then return { present = false } end
  local record = {
    present = true,
    position = { ch.position.x, ch.position.y },
    walking = ch.walking_state.walking,
    direction = ch.walking_state.direction,
  }
  -- `encoders.encode` builds the self vector from position, walking,
  -- direction, mining and crafting alone; it never reads health or reach, so a
  -- slim profile would pay five numbers per step for values no policy can see.
  -- Dropping them cannot loosen a precondition either: `actions.lua` checks
  -- `ch.resource_reach_distance` and `ch.build_distance` off the character
  -- itself before returning `out_of_reach`, never off this block.
  if not observation_profile.slim then
    record.health = ch.health
    record.reach = {
      entity = ch.reach_distance,
      build = ch.build_distance,
      resource = ch.resource_reach_distance,
      pickup = ch.item_pickup_distance,
    }
  end
  if ch.mining_state and ch.mining_state.mining then
    record.mining = { active = true, progress = ch.character_mining_progress }
  end
  local queue = ch.crafting_queue
  if queue and #queue > 0 then
    local items = {}
    for _, item in pairs(queue) do
      items[#items + 1] = { recipe = item.recipe, count = item.count, index = item.index }
    end
    record.crafting = { queue = items, progress = ch.crafting_queue_progress }
  end
  return record
end

--- Technologies that can be started *right now*.
--
-- Not the whole tree, which is 196 entries and the single largest block the
-- snapshot could carry -- `local-v2` dropped `force` entirely for that reason.
-- This is the researchable frontier: prerequisites all met, not already
-- researched, and enabled. Early game that is a handful of names.
--
-- Sent because `research` was permanently masked without it.
-- `env.argument_domains` returned an empty `technologies` list, and an argument
-- whose domain the policy cannot see is not selectable, so the verb sat in the
-- action matrix unreachable. A frontier is also the honest amount to publish:
-- the full tree is static knowledge the agent is given separately, while *what
-- is available now* is world state.
local function researchable(force)
  local names = {}
  for name, tech in pairs(force.technologies) do
    if tech.enabled and not tech.researched then
      local ready = true
      for _, prerequisite in pairs(tech.prerequisites) do
        if not prerequisite.researched then
          ready = false
          break
        end
      end
      if ready then names[#names + 1] = name end
    end
  end
  table.sort(names)
  return names
end

local function force_state(force)
  local researched = {}
  for name, tech in pairs(force.technologies) do
    if tech.researched then researched[#researched + 1] = name end
  end
  table.sort(researched)
  return {
    researched = researched,
    current_research = force.current_research and force.current_research.name or nil,
    research_progress = force.research_progress,
  }
end

function observations.snapshot(state)
  local ch = storage.frrl_character
  local observation_profile = profiles.observation(state.observation_profile)
  local origin = (ch and ch.valid) and ch.position or { x = 0, y = 0 }
  local surface = game.surfaces["nauvis"]

  local swept = sensor.sweep(surface, origin, observation_profile)
  local remembered = observation_profile.memory
    and memory.update(swept.entities, origin, observation_profile.radius)
    or {}

  local task = storage.frrl_task or { transfers = 0, items_moved = 0 }
  local force_declared = profiles.declares(observation_profile, "force")

  -- The three blocks below are assembled here rather than inline because a
  -- slim profile omits sub-fields, and `profiles.filter` only reaches
  -- top-level keys -- a nested field can only be left out at the point it
  -- would be built.
  local sensor_block = {
    radius = observation_profile.radius,
    origin = { origin.x, origin.y },
  }
  if not observation_profile.slim then
    -- Diagnostics, not policy input. `truncated` reports that the entity cap
    -- clipped the sweep, and the two counters are growth proxies; the Phase 3
    -- gate reads its growth proxies from `world_digest`, not from here, so
    -- omitting them cannot blind the leak check. Note that `handles.count()`
    -- and `memory.count()` are not even called for a slim profile -- the point
    -- is to skip the work, not just the bytes.
    sensor_block.truncated = swept.truncated
    sensor_block.handles = handles.count()
    sensor_block.remembered = memory.count()
  end

  local terrain = { blocked = swept.blocked }
  local resources = { tiles = swept.resource_tiles }
  if not observation_profile.slim then
    -- `detail` is a constant of the profile, and the profile's name already
    -- travels in every observation under `profiles`, so this restates on every
    -- step something the reader can already look up once.
    terrain.detail = observation_profile.terrain_detail
    -- Per-patch aggregates. `encoders.encode` and every task predicate read
    -- `resources.tiles`; nothing reads `patches`.
    resources.patches = swept.patches
  end

  local snapshot = {
    episode_id = state.episode_id,
    tick = game.tick - state.episode_start_tick,
    absolute_tick = game.tick,
    profiles = profiles.metadata(state.observation_profile, state.action_profile),
    character = character_state(ch, observation_profile),
    inventory = inventory_contents(
      ch and ch.valid and ch.get_inventory(defines.inventory.character_main) or nil
    ),
    sensor = sensor_block,
    terrain = terrain,
    resources = resources,
    entities = swept.entities,
    remembered = remembered,
    -- Skipped entirely when the profile does not declare `force`: building it
    -- walks and sorts the whole technology list, which is the single most
    -- expensive block in the snapshot, and `profiles.filter` would only throw
    -- the result away afterwards.
    force = force_declared and force_state(ch and ch.force or game.forces["player"]) or nil,
    task = { transfers = task.transfers, items_moved = task.items_moved },
    -- Where the objective is. Two sources, chosen by the task at scene
    -- install: markers a success or failure predicate names, and markers the
    -- task declares in `extra_public_markers` -- which is how `repair_belt`
    -- and `restore_power` publish their fault tiles. This said "only markers
    -- a success or failure predicate names" until R4.6, which had been false
    -- since v1.6.0. Decoys are in neither source and never appear.
    goal = world.public_markers(),
    inflight = inflight.summary(),
    events = state.events or {},
    -- What the force can actually make. Never sent before, so `craft` and
    -- `set_recipe` had no observable vocabulary and a policy could not name a
    -- legal value for either. Enabled recipes only, so this states a
    -- capability rather than leaking the tech tree.
    recipes = enabled_recipes(),
    -- The researchable frontier, when the profile declares it. Filtered out
    -- for every benchmark profile, which neither declares nor needs it.
    researchable = profiles.declares(observation_profile, "researchable")
      and researchable(ch and ch.force or game.forces["player"])
      or nil,
  }
  return profiles.filter(snapshot, observation_profile)
end

return observations
