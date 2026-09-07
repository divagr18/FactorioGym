-- Local structured observations (PLAN.md 2.3), assembled from the sensor
-- sweep, remembered observations, entity handles and the in-flight registry,
-- then filtered through the observation profile's declared key list.

local sensor = require("sensor")
local memory = require("memory")
local profiles = require("profiles")
local handles = require("handles")
local inflight = require("inflight")

local observations = {}

local function inventory_contents(inv)
  local out = {}
  if not inv then return out end
  for _, stack in pairs(inv.get_contents()) do
    out[stack.name] = (out[stack.name] or 0) + stack.count
  end
  return out
end

local function character_state(ch)
  if not ch or not ch.valid then return { present = false } end
  local record = {
    present = true,
    position = { ch.position.x, ch.position.y },
    walking = ch.walking_state.walking,
    direction = ch.walking_state.direction,
    health = ch.health,
    reach = {
      entity = ch.reach_distance,
      build = ch.build_distance,
      resource = ch.resource_reach_distance,
      pickup = ch.item_pickup_distance,
    },
  }
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
  local snapshot = {
    episode_id = state.episode_id,
    tick = game.tick - state.episode_start_tick,
    absolute_tick = game.tick,
    profiles = profiles.metadata(state.observation_profile, state.action_profile),
    character = character_state(ch),
    inventory = inventory_contents(
      ch and ch.valid and ch.get_inventory(defines.inventory.character_main) or nil
    ),
    sensor = {
      radius = observation_profile.radius,
      origin = { origin.x, origin.y },
      truncated = swept.truncated,
      handles = handles.count(),
      remembered = memory.count(),
    },
    terrain = { detail = observation_profile.terrain_detail, blocked = swept.blocked },
    resources = { tiles = swept.resource_tiles, patches = swept.patches },
    entities = swept.entities,
    remembered = remembered,
    force = force_state(ch and ch.force or game.forces["player"]),
    task = { transfers = task.transfers, items_moved = task.items_moved },
    inflight = inflight.summary(),
    events = state.events or {},
  }
  return profiles.filter(snapshot, observation_profile)
end

return observations
