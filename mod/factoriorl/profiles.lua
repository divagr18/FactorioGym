-- Observation and action profiles (PLAN.md 2.5).
--
-- A profile is a name plus a version, echoed in every observation and every
-- describe response, so a result cannot omit which profile produced it.
--
-- The allowed observation keys are declared here and `observations.snapshot`
-- filters its output through them. Adding a field to an observation therefore
-- requires editing this declaration -- a visible, reviewable act rather than an
-- accident, which is the mechanism (not the convention) that keeps
-- evaluator-only information out of policy input.
--
-- `keys` filters top-level keys only, because `profiles.filter` walks one
-- level. A profile that also wants sub-blocks trimmed sets `slim = true`, and
-- `observations.snapshot` consults it where `character`, `sensor`, `resources`
-- and `terrain` are built. Splitting it this way keeps the filter mechanical:
-- it cannot silently miss a nested field it was never told about, because the
-- omission happens at the point of construction instead.

local profiles = {}

profiles.OBSERVATION = {
  ["local-v1"] = {
    name = "local-v1",
    version = 1,
    radius = 32,
    entity_cap = 256,
    resource_cap = 512,
    -- Well beyond the 2.7-tile resource reach, far enough for local planning,
    -- and short of the payload blow-up a whole patch would cause.
    resource_detail_radius = 12,
    terrain_detail = "mask",
    memory = true,
    keys = {
      "episode_id", "tick", "absolute_tick", "profiles", "character",
      "inventory", "sensor", "terrain", "resources", "entities",
      "remembered", "force", "task", "inflight", "events",
    },
  },
  -- Opt-in, and deliberately not the default: `local-v1` is pinned by the
  -- frozen protocol fixtures and by the engine gates, so it stays exactly as
  -- it is and a task asks for `local-v2` by name.
  --
  -- Same sensor contract as v1 -- identical radius, caps and detail radius, so
  -- a policy sees the same world -- carrying only what Python actually reads.
  -- The observation is ~32 KB per step, serialised by `helpers.table_to_json`
  -- on the engine's own thread (competing with tick time) and parsed again on
  -- every step; per-step latency degrades from 23.7 ms at one worker to 37.8 ms
  -- at eight. Every field here that no consumer reads is that cost paid for
  -- nothing.
  --
  -- Dropped relative to v1, each verified unread on the Python side by grep:
  --   * `force` -- researched / current_research / research_progress. The
  --     technology list is the single largest block and is rebuilt and sorted
  --     from scratch every observation.
  --   * `events` -- carried but never inspected.
  --   * `resources.patches`, `sensor.handles`, `sensor.remembered`,
  --     `sensor.truncated`, `character.health`, `character.reach` and
  --     `terrain.detail`, all handled by `slim` below since they are nested.
  -- `sensor.origin`, `sensor.radius`, `resources.tiles`, `terrain.blocked` and
  -- character position/walking/direction/mining/crafting all feed
  -- `encoders.encode` and must stay.
  ["local-v2"] = {
    name = "local-v2",
    version = 2,
    radius = 32,
    entity_cap = 256,
    resource_cap = 512,
    resource_detail_radius = 12,
    terrain_detail = "mask",
    memory = true,
    slim = true,
    keys = {
      "episode_id", "tick", "absolute_tick", "profiles", "character",
      "inventory", "sensor", "terrain", "resources", "entities",
      "remembered", "task", "inflight",
    },
  },
}

profiles.ACTION = {
  ["primitive-v1"] = {
    name = "primitive-v1",
    version = 1,
    assistance = "none",
    available = true,
    -- Recorded because PLAN.md section 2 requires any approximation to be
    -- documented in the action profile. The Phase 2.0 spike found mining and
    -- crafting are both native, so nothing here is approximated; placement is
    -- assembled only because build_from_cursor is LuaPlayer-only.
    drivers = {
      mining = "native",
      crafting = "native",
      placement = "assembled",
    },
    placement_requires_unlocked_recipe = true,
  },
  -- Declared now, unavailable until Phase 5, so the assisted contract is
  -- visible from Phase 2 and cannot be retro-fitted into primitive-v1 by
  -- accident.
  ["assisted-v1"] = {
    name = "assisted-v1",
    version = 1,
    assistance = "navigation+bounded-batches",
    available = false,
    drivers = {},
  },
}

profiles.DEFAULT_OBSERVATION = "local-v1"
profiles.DEFAULT_ACTION = "primitive-v1"

function profiles.observation(name)
  return profiles.OBSERVATION[name or profiles.DEFAULT_OBSERVATION]
end

function profiles.action(name)
  return profiles.ACTION[name or profiles.DEFAULT_ACTION]
end

--- Does this observation profile declare `key` as a top-level key?
---
--- `filter` already drops undeclared keys, so this exists only for blocks
--- expensive enough that building them and then discarding them is itself the
--- problem -- `force`, which walks and sorts the entire technology list every
--- observation. Asking before building keeps the declaration in `keys` the one
--- place that decides, rather than adding a second, drift-prone switch.
function profiles.declares(observation_profile, key)
  for _, declared in ipairs(observation_profile.keys) do
    if declared == key then return true end
  end
  return false
end

--- Keep only keys the observation profile declares.
function profiles.filter(snapshot, observation_profile)
  local allowed = {}
  for _, key in ipairs(observation_profile.keys) do allowed[key] = true end
  local out = {}
  for key, value in pairs(snapshot) do
    if allowed[key] then out[key] = value end
  end
  return out
end

function profiles.metadata(observation_name, action_name)
  local obs = profiles.observation(observation_name)
  local act = profiles.action(action_name)
  return {
    observation = obs.name,
    observation_version = obs.version,
    action = act.name,
    action_version = act.version,
    assistance = act.assistance,
    drivers = act.drivers,
  }
end

return profiles
