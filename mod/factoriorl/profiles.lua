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

local matrix = require("matrix")

local profiles = {}

--- The action catalog a profile offers, composed from the two ordered lists in
--- `matrix.lua`. Composing it here rather than writing it out twice is what
--- keeps a new assisted action from having to be added in two places, one of
--- which would eventually be forgotten.
local function action_catalog(include_assisted)
  local names = {}
  for _, name in ipairs(matrix.ORDER) do names[#names + 1] = name end
  if include_assisted then
    for _, name in ipairs(matrix.ASSISTED_ORDER) do names[#names + 1] = name end
  end
  return names
end

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
      "remembered", "force", "task", "inflight", "events", "goal",
    },
  },
  -- Opt-in, and deliberately not the default: `local-v1` is pinned by the
  -- frozen protocol fixtures and by the engine gates, so it stays exactly as
  -- it is and a task asks for `local-v2` by name.
  --
  -- Same sensor region as v1 -- identical radius and detail radius, so a policy
  -- sees the same world -- carrying only what Python actually reads. The
  -- observation is ~32 KB per step, serialised by `helpers.table_to_json`
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
  --
  -- The one place v2 does depart from v1's sensor contract is `entity_cap`, and
  -- it departs without changing what the policy sees. `encoders.encode` sorts
  -- `entities` + `remembered` by distance and keeps only the nearest 32, and
  -- `sensor.sweep` now orders the sweep by distance before capping it, so a
  -- visible entity at rank 49 already has 48 entities at least as close to it
  -- and can never reach the encoder's 32 -- whatever `remembered` holds. 256
  -- entity records is ~17 KB of JSON per step, the largest single block left,
  -- and eight times what anything reads.
  --
  -- `entity_sweep_limit` stays at v1's candidate budget on purpose. The cap
  -- bounds what is serialised; the sweep limit bounds what the engine hands us
  -- to sort. Lowering the sweep limit too would mean sorting 49 arbitrary
  -- entities and calling the result "the nearest 48", which is the very bug the
  -- sort was added to fix. Keeping it at 257 means v2 chooses its 48 out of
  -- exactly the pool v1 would have shipped.
  --
  -- Known consequence, recorded rather than hidden: `observations.snapshot`
  -- feeds the *capped* entity list to `memory.update`, and `memory.update`
  -- deletes any in-region entry the sweep did not report -- "the agent looked,
  -- and it was not there". Under a 48 cap, an entity inside the radius but at
  -- rank 49 is now forgotten even though it exists, so `remembered` in a later,
  -- sparser step can be thinner than v1 would have produced. It cannot change
  -- the encoder's input at the dense step (all 32 rows come from the visible
  -- set there), only at a later sparse one. The same defect already existed at
  -- 256 for any scene with more than 256 entities in radius; fixing it properly
  -- means handing memory the full candidate list, which is a change to
  -- `observations.lua` and `memory.lua`, not to this cap.
  ["local-v2"] = {
    name = "local-v2",
    version = 2,
    radius = 32,
    entity_cap = 48,
    entity_sweep_limit = 257,
    resource_cap = 512,
    resource_detail_radius = 12,
    terrain_detail = "mask",
    memory = true,
    slim = true,
    keys = {
      "episode_id", "tick", "absolute_tick", "profiles", "character",
      "inventory", "sensor", "terrain", "resources", "entities",
      "remembered", "task", "inflight", "goal",
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
    actions = action_catalog(false),
    -- No terrain memory, and that is a cost decision as much as a scope one:
    -- folding explored terrain costs two engine queries and ~49 block rewrites
    -- per observation, and a primitive run can never navigate, so it would be
    -- paying per step for a store nothing reads. Every Phase 3 and Phase 4
    -- measurement therefore still describes the same observation path it did
    -- before navigation existed.
    terrain_memory = false,
  },
  -- Available from Phase 5.1. The capability list is deliberately narrow:
  -- known-terrain navigation is implemented, bounded batches (PLAN.md 5.2) are
  -- not, and PLAN.md 6.4 forbids advertising deferred functionality as
  -- available -- so `assistance` names what this profile actually does today
  -- and gains "+bounded-batches" when 5.2 lands.
  ["assisted-v1"] = {
    name = "assisted-v1",
    version = 1,
    assistance = "navigation",
    available = true,
    drivers = {
      mining = "native",
      crafting = "native",
      placement = "assembled",
      -- Spelled out because PLAN.md section 2 requires any approximation to be
      -- documented in the action profile, and because the alternative -- the
      -- engine pathfinder, or a teleport -- is exactly what an assistance
      -- profile is most likely to be quietly implemented with.
      navigation = "own A* over explored terrain, walked with walking_state",
    },
    placement_requires_unlocked_recipe = true,
    actions = action_catalog(true),
    terrain_memory = true,
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

--- May this action profile send `action_name`?
---
--- `actions.dispatch` asks before every action, so an assisted action sent to
--- a primitive worker is rejected as unknown rather than executed. That is the
--- mechanism behind "primitive and assisted capabilities are distinguishable"
--- (PLAN.md 2.5) and behind every Phase 3/4 result staying comparable: a
--- primitive run cannot reach `navigate` even if a client asks for it.
function profiles.permits(action_profile, action_name)
  if not action_profile or not action_profile.actions then return true end
  for _, declared in ipairs(action_profile.actions) do
    if declared == action_name then return true end
  end
  return false
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
