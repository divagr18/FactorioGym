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
