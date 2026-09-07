-- Scene construction and reset (PLAN.md 2.x, and the seam Phase 3 builds on).
--
-- Two scenarios today. "reference" is byte-identical to the Phase 0/1 scene, so
-- the frozen protocol fixtures and gate_phase0 keep working unchanged.
-- "phase2-gate" adds the ore, fuel, machinery and locked recipe the scripted
-- embodied agent needs -- the world otherwise contains no ore at all, because
-- map generation disables every resource.
--
-- Scenario initialisation is evaluator-only (PLAN.md section 2), which is why
-- it lives here and not behind a typed action.

local handles = require("handles")

local world = {}

local SCENE_RADIUS = 16
local SURFACE_NAME = "nauvis"
local START_IRON_PLATES = 50

local function surface()
  return game.surfaces[SURFACE_NAME]
end

--- Destroy everything the episode may have created.
-- Player-force entities are swept across the whole surface, not just a box:
-- the old +-16 clear left anything built further out to survive a reset, which
-- would silently poison a long training run. Neutral entities (resources,
-- trees, rocks) are cleared inside the scenario box only.
local function clear_scene(srf, radius)
  local destroyed = 0
  for _, entity in pairs(srf.find_entities_filtered({ force = "player" })) do
    if entity.valid and entity.type ~= "character" then
      entity.destroy({ raise_destroy = true })
      destroyed = destroyed + 1
    end
  end
  local area = { { -radius, -radius }, { radius, radius } }
  for _, entity in pairs(srf.find_entities_filtered({ area = area, force = "neutral" })) do
    if entity.valid and entity.type ~= "character" then
      entity.destroy({ raise_destroy = true })
      destroyed = destroyed + 1
    end
  end
  return destroyed
end

-- ---------------------------------------------------------------- scenarios

local SCENARIOS = {}

--- The Phase 0/1 reference scene. Unchanged on purpose.
SCENARIOS.reference = {
  radius = SCENE_RADIUS,
  build = function(srf)
    srf.create_entity({ name = "wooden-chest", position = { 3, 0 }, force = "player" })
    srf.create_entity({ name = "wooden-chest", position = { -3, 0 }, force = "player" })
    for x = -3, 3 do
      srf.create_entity({ name = "stone-wall", position = { x, -6 }, force = "player" })
    end
    local src = srf.find_entities_filtered({
      area = { { 2.5, -0.5 }, { 3.5, 0.5 } }, name = "wooden-chest",
    })[1]
    local dst = srf.find_entities_filtered({
      area = { { -3.5, -0.5 }, { -2.5, 0.5 } }, name = "wooden-chest",
    })[1]
    if src then src.insert({ name = "iron-plate", count = START_IRON_PLATES }) end
    return { src = src, dst = dst }
  end,
}

--- Everything the Phase 2 exit gate's scripted agent needs.
SCENARIOS["phase2-gate"] = {
  radius = 48,
  build = function(srf)
    local aliases = {}
    -- Ore sits beyond the initial 32-tile sensor radius on purpose, so the gate
    -- exercises partial observability and the visibility transition rather than
    -- starting with everything in view.
    for dx = 0, 4 do
      for dy = -2, 2 do
        local p = { 40 + dx, dy }
        if srf.can_place_entity({ name = "iron-ore", position = p }) then
          srf.create_entity({ name = "iron-ore", position = p, amount = 2000 })
        end
      end
    end
    for dx = 0, 2 do
      for dy = 0, 2 do
        local p = { 40 + dx, 8 + dy }
        if srf.can_place_entity({ name = "coal", position = p }) then
          srf.create_entity({ name = "coal", position = p, amount = 2000 })
        end
      end
    end
    for dx = 0, 2 do
      for dy = 0, 2 do
        local p = { 36 + dx, -8 - dy }
        if srf.can_place_entity({ name = "stone", position = p }) then
          srf.create_entity({ name = "stone", position = p, amount = 2000 })
        end
      end
    end
    aliases.assembler = srf.create_entity({
      name = "assembling-machine-1", position = { -6, 0 }, force = "player",
    })
    aliases.belt = srf.create_entity({
      name = "transport-belt", position = { -8, 0 }, force = "player",
    })
    aliases.src = srf.create_entity({
      name = "wooden-chest", position = { 3, 0 }, force = "player",
    })
    if aliases.src then aliases.src.insert({ name = "iron-plate", count = 10 }) end
    return aliases
  end,
  after = function(force)
    -- The recipe-selection and placement technology cases need something that
    -- is genuinely locked. Recorded in the gate report as evaluator setup.
    force.recipes["electronic-circuit"].enabled = false
  end,
}

function world.scenarios()
  local names = {}
  for name in pairs(SCENARIOS) do names[#names + 1] = name end
  table.sort(names)
  return names
end

--- Build a named scenario, replacing whatever was there.
function world.build_scenario(name)
  local scenario = SCENARIOS[name] or SCENARIOS.reference
  local srf = surface()
  local destroyed = clear_scene(srf, scenario.radius)
  world.ensure_character()
  local aliases = scenario.build(srf) or {}
  storage.frrl_scene = { name = SCENARIOS[name] and name or "reference", aliases = aliases }
  if scenario.after then
    scenario.after(game.forces["player"])
  end
  return { scenario = storage.frrl_scene.name, destroyed = destroyed }
end

function world.build_reference_scene()
  return world.build_scenario(storage.frrl_scenario_name or "reference")
end

function world.reset_scene()
  return world.build_scenario(storage.frrl_scenario_name or "reference")
end

function world.scene()
  return storage.frrl_scene
end

--- Resolve a scenario alias ("src", "dst", "assembler", ...) to a handle.
-- Aliases are scenario names, not agent-visible identity: they exist so the
-- frozen Phase 0/1 fixtures keep addressing the same chests.
function world.alias_handle(name)
  local scene = storage.frrl_scene
  if not scene or not scene.aliases then return nil end
  local entity = scene.aliases[name]
  if not entity or not entity.valid then return nil end
  return handles.mint(entity)
end

function world.alias_entity(name)
  local scene = storage.frrl_scene
  if not scene or not scene.aliases then return nil end
  local entity = scene.aliases[name]
  if entity and entity.valid then return entity end
  return nil
end

-- Factorio 2.0 removed game.create_player; zero-player headless servers have
-- no LuaPlayer, so the agent body is a standalone character entity.
function world.ensure_character()
  local existing = storage.frrl_character
  if existing and existing.valid then return existing end
  local ch = surface().create_entity({
    name = "character",
    position = { 0, 0 },
    force = "player",
  })
  storage.frrl_character = ch
  return ch
end

return world
