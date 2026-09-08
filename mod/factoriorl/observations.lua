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
    -- Where the objective is. Only markers a success or failure predicate
    -- names, chosen by the task at scene install; decoys never appear.
    goal = world.public_markers(),
    inflight = inflight.summary(),
    events = state.events or {},
  }
  return profiles.filter(snapshot, observation_profile)
end

return observations
