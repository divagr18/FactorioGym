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


-- Return the force's technologies and recipes to their prototype state.
--
-- Factorio 2.0 has trigger-based technologies: several early ones are
-- researched by mining or crafting a particular item rather than by consuming
-- science. So an agent doing ordinary task work silently researches things,
-- and nothing here un-researched them -- `world.digest` reported technologies
-- but reset never restored them, so the leak was invisible until a reference
-- solution did enough mining to trip one. The 504-reset run then caught it as
-- `tech|steam-power` present on a long-running worker and absent on a fresh
-- one.
--
-- `reset_technologies` is the wrong tool despite the name: it reloads
-- prototype definitions while explicitly *preserving* research state. Nor is
-- `force.reset`, which additionally unchartes the map that `on_init`
-- deliberately charts. Un-researching directly and reapplying effects is the
-- surgical version.
function world.reset_force()
  local force = game.forces["player"]
  for _, tech in pairs(force.technologies) do
    if tech.researched then tech.researched = false end
  end
  force.research_queue = {}
  force.reset_recipes()
  force.reset_technology_effects()
end


-- Replace the character with a newly created one.
--
-- Facing was the one piece of character state reset never restored, so an
-- episode began pointing wherever the previous episode's last move left it --
-- and that reaches the policy directly, since `encoders.encode` puts
-- direction/16 into the self vector. Assigning `ch.direction` does not fix it:
-- the engine applies the write on the following tick, and the world is paused
-- between decisions, so the reset observation would still report the stale
-- value. A character created this tick reads direction 0 immediately.
--
-- Safe because reset invalidates every outstanding handle anyway, and
-- `clear_scene` deliberately skips characters, so nothing else is holding this
-- entity across the boundary.
function world.recreate_character()
  local existing = storage.frrl_character
  if existing and existing.valid then
    existing.destroy()
  end
  storage.frrl_character = nil
  return world.ensure_character()
end


-- ---------------------------------------------------------------- blueprints

--- Blueprints are installed once and referenced by hash thereafter, so a
--- steady-state reset carries a hash rather than the whole scene.
local BLUEPRINT_LIMIT = 64

local DIRECTIONS = {
  north = defines.direction.north,
  east = defines.direction.east,
  south = defines.direction.south,
  west = defines.direction.west,
}

function world.define_blueprint(hash, blueprint)
  storage.frrl_blueprints = storage.frrl_blueprints or { order = {}, by_hash = {} }
  local store = storage.frrl_blueprints
  if not store.by_hash[hash] then
    store.order[#store.order + 1] = hash
    while #store.order > BLUEPRINT_LIMIT do
      local oldest = table.remove(store.order, 1)
      store.by_hash[oldest] = nil
    end
  end
  store.by_hash[hash] = blueprint
  return hash
end

function world.has_blueprint(hash)
  local store = storage.frrl_blueprints
  return store and store.by_hash[hash] ~= nil
end

--- Build a stored blueprint, replacing whatever is there.
function world.build_blueprint(hash)
  local store = storage.frrl_blueprints
  local blueprint = store and store.by_hash[hash]
  if not blueprint then return nil end
  local srf = surface()
  local radius = blueprint.radius or 64
  local destroyed = clear_scene(srf, radius)
  world.ensure_character()
  -- Deterministic lighting: solar output must not depend on what time of day
  -- the map happens to be at when an episode starts.
  srf.always_day = true

  local aliases = {}
  for _, spec in pairs(blueprint.resources or {}) do
    local position = { spec.position[1], spec.position[2] }
    if srf.can_place_entity({ name = spec.name, position = position }) then
      srf.create_entity({
        name = spec.name, position = position, amount = spec.amount or 1000,
      })
    end
  end
  for _, spec in pairs(blueprint.entities or {}) do
    local created = srf.create_entity({
      name = spec.name,
      position = { spec.position[1], spec.position[2] },
      direction = DIRECTIONS[spec.direction or "north"],
      force = spec.force or "player",
    })
    if created then
      if spec.contents then
        for item, count in pairs(spec.contents) do
          created.insert({ name = item, count = count })
        end
      end
      if spec.recipe then pcall(function() created.set_recipe(spec.recipe) end) end
      if spec.marker then aliases[spec.marker] = created end
    end
  end

  local ch = world.ensure_character()
  local character = blueprint.character or {}
  if character.position then
    ch.teleport({ character.position[1], character.position[2] })
  end
  if character.inventory then
    local inv = ch.get_inventory(defines.inventory.character_main)
    for item, count in pairs(character.inventory) do
      inv.insert({ name = item, count = count })
    end
  end

  -- Declared unlocks. Evaluator setup, recorded in the blueprint rather than
  -- applied invisibly: a task that gives the agent an item must also make that
  -- item placeable, or the profile's technology guard refuses it.
  local force = game.forces["player"]
  for _, recipe_name in pairs(blueprint.unlock_recipes or {}) do
    local recipe = force.recipes[recipe_name]
    if recipe then recipe.enabled = true end
  end

  storage.frrl_scene = {
    name = "blueprint:" .. hash,
    aliases = aliases,
    markers = blueprint.markers or {},
    -- Which markers `observations` may publish. Every other marker stays
    -- evaluator-only: `deliver` scores `dst` and also carries `decoy_0` and
    -- `decoy_1`, and showing those would hand the agent the whole scene
    -- instead of the objective.
    public_markers = blueprint.public_markers or {},
    -- Items to count in `truth.produced` beyond the default list. A task whose
    -- objective names something else read zero from its own success predicate,
    -- with no error anywhere.
    extra_tracked_items = blueprint.extra_tracked_items or {},
    radius = radius,
  }
  return { scenario = storage.frrl_scene.name, destroyed = destroyed }
end

--- Positions of the markers the task's objective names (PLAN.md 5.x).
---
--- This is deliberately a *subset* of `truth().markers`. The separation
--- between observation and evaluator state stays structural: an unpublished
--- marker is unreachable from `observe`, and the allowlist is set by the task
--- at scene install rather than decided here.
function world.public_markers()
  local scene = storage.frrl_scene
  if not scene then return nil end
  local allowed = scene.public_markers or {}
  if #allowed == 0 then return nil end
  local out = {}
  for _, name in pairs(allowed) do
    local position = (scene.markers or {})[name]
    local entity = (scene.aliases or {})[name]
    if entity and entity.valid then
      out[name] = { entity.position.x, entity.position.y }
    elseif position then
      out[name] = position
    end
  end
  if next(out) == nil then return nil end
  return out
end

-- ---------------------------------------------------------------- truth

--- Evaluator-only ground truth (PLAN.md section 2: evaluator information must
--- not enter policy inputs). This is a separate request from `observe`, so the
--- separation is structural: a policy reading observations cannot reach it.
function world.truth()
  local scene = storage.frrl_scene or {}
  local force = game.forces["player"]
  local stats = force.get_item_production_statistics(surface())

  local containers = {}
  local working = {}
  for marker, entity in pairs(scene.aliases or {}) do
    if entity and entity.valid then
      local contents = {}
      for _, which in ipairs({
        defines.inventory.chest,
        defines.inventory.furnace_result,
        defines.inventory.assembling_machine_output,
        defines.inventory.furnace_source,
      }) do
        local inv = entity.get_inventory(which)
        if inv then
          for _, stack in pairs(inv.get_contents()) do
            contents[stack.name] = (contents[stack.name] or 0) + stack.count
          end
        end
      end
      containers[marker] = contents
      local ok, status = pcall(function() return entity.status end)
      working[marker] = ok and status == defines.entity_status.working or false
    end
  end

  local tracked = {
    "iron-plate", "copper-plate", "stone-furnace", "iron-gear-wheel",
    "iron-ore", "copper-ore", "coal", "stone",
  }
  for _, item in ipairs(scene.extra_tracked_items or {}) do
    tracked[#tracked + 1] = item
  end
  local produced = {}
  for _, item in ipairs(tracked) do
    local count = stats.get_input_count(item)
    if count and count > 0 then produced[item] = count end
  end

  local markers = {}
  for name, position in pairs(scene.markers or {}) do
    markers[name] = position
  end
  for marker, entity in pairs(scene.aliases or {}) do
    if entity and entity.valid then
      markers[marker] = { entity.position.x, entity.position.y }
    end
  end

  -- Prototype-keyed, because `containers` and `working` above are keyed on
  -- `scene.aliases` -- bound when the scene is *installed*. An entity the agent
  -- builds has no alias, so no predicate could name it, which made "the chain
  -- must be placed by the agent" unmeasurable. These aggregate by prototype
  -- name instead, so a machine counts whoever placed it.
  local working_counts = {}
  local placed_counts = {}
  local srf = surface()
  if srf then
    for _, entity in ipairs(srf.find_entities_filtered({ force = force })) do
      if entity.valid and entity.name then
        placed_counts[entity.name] = (placed_counts[entity.name] or 0) + 1
        local ok, status = pcall(function() return entity.status end)
        if ok and status == defines.entity_status.working then
          working_counts[entity.name] = (working_counts[entity.name] or 0) + 1
        end
      end
    end
  end

  -- What the *agent* built, as distinct from what the scene installed.
  --
  -- Not `get_entity_build_count_statistics`: that was the first attempt and it
  -- is empty here, because `surface.create_entity` -- which is how the `place`
  -- action builds -- does not feed the engine's build statistics. Measured: a
  -- correctly built and running line reported `placed_counts` for both
  -- machines, 49 plates produced, and `built = {}`.
  --
  -- Counted in the `place` handler instead, which is a better fit for what the
  -- predicate is asking anyway: "newly constructed" means *the agent placed
  -- it through the action interface*, and a scene's install path does not go
  -- through that handler, so a preplaced machine can never count.
  local built = {}
  for name, count in pairs(storage.frrl_built or {}) do
    if count and count > 0 then built[name] = count end
  end

  return {
    markers = markers,
    containers = containers,
    working = working,
    produced = produced,
    -- Additive: every existing predicate reads only the four above.
    working_counts = working_counts,
    placed_counts = placed_counts,
    built = built,
    scenario = scene.name,
  }
end

--- Clear cumulative statistics. Production statistics are exactly where
--- cross-episode leakage hides, and nothing cleared them before.
function world.clear_statistics()
  -- Placements the agent made, counted in the `place` handler. Cleared here
  -- with the rest, or a `BUILT` predicate would be satisfied by the previous
  -- episode's construction.
  storage.frrl_built = {}
  local force = game.forces["player"]
  for _, getter in ipairs({
    "get_item_production_statistics",
    "get_fluid_production_statistics",
    "get_entity_build_count_statistics",
    "get_kill_count_statistics",
  }) do
    local ok, stats = pcall(function() return force[getter](surface()) end)
    if ok and stats then pcall(function() stats.clear() end) end
  end
end


-- ---------------------------------------------------------------- digest

--- A canonical description of everything a reset must restore.
-- Returned as sorted lines rather than a hash, deliberately: a bare hash
-- mismatch tells you that something leaked but not what, and PLAN.md 3.3 wants
-- item, technology, timer and reward leakage distinguished. Python hashes it
-- for the fast comparison and diffs the lines when they disagree.
function world.digest()
  local srf = surface()
  local lines = {}

  local entities = {}
  for _, entity in pairs(srf.find_entities_filtered({ force = "player" })) do
    if entity.valid and entity.type ~= "character" then
      local parts = {
        entity.name,
        string.format("%.2f", entity.position.x),
        string.format("%.2f", entity.position.y),
        tostring(entity.direction),
      }
      local contents = {}
      for _, which in ipairs({
        defines.inventory.chest,
        defines.inventory.fuel,
        defines.inventory.furnace_source,
        defines.inventory.furnace_result,
      }) do
        local inv = entity.get_inventory(which)
        if inv then
          for _, stack in pairs(inv.get_contents()) do
            contents[#contents + 1] = stack.name .. "=" .. stack.count
          end
        end
      end
      table.sort(contents)
      parts[#parts + 1] = table.concat(contents, ",")
      entities[#entities + 1] = "entity|" .. table.concat(parts, "|")
    end
  end
  table.sort(entities)
  for _, line in ipairs(entities) do lines[#lines + 1] = line end

  local resources = {}
  for _, entity in pairs(srf.find_entities_filtered({ type = "resource" })) do
    if entity.valid then
      resources[#resources + 1] = string.format(
        "resource|%s|%.1f|%.1f|%d", entity.name, entity.position.x, entity.position.y,
        entity.amount or 0)
    end
  end
  table.sort(resources)
  for _, line in ipairs(resources) do lines[#lines + 1] = line end

  local ch = storage.frrl_character
  if ch and ch.valid then
    local inv = ch.get_inventory(defines.inventory.character_main)
    local items = {}
    if inv then
      for _, stack in pairs(inv.get_contents()) do
        items[#items + 1] = stack.name .. "=" .. stack.count
      end
    end
    table.sort(items)
    -- Direction is in the digest because it was not, and a leak hid there:
    -- reset restored position and inventory but left facing untouched, and the
    -- 500-reset equality run compared digests that could not see it.
    lines[#lines + 1] = string.format(
      "character|%.2f|%.2f|%d|%d|%s|queue=%d",
      ch.position.x, ch.position.y, ch.direction or 0, ch.health or 0,
      table.concat(items, ","),
      ch.crafting_queue and #ch.crafting_queue or 0)
  else
    lines[#lines + 1] = "character|absent"
  end

  local force = game.forces["player"]
  local techs = {}
  for name, tech in pairs(force.technologies) do
    if tech.researched then techs[#techs + 1] = name end
  end
  table.sort(techs)
  lines[#lines + 1] = "tech|" .. table.concat(techs, ",")
  lines[#lines + 1] = "research|" ..
    tostring(force.current_research and force.current_research.name or "none")

  -- Cumulative statistics: exactly where cross-episode leakage hides.
  local stats = force.get_item_production_statistics(srf)
  local produced = {}
  for _, item in ipairs({
    "iron-plate", "copper-plate", "iron-ore", "coal", "stone", "stone-furnace",
  }) do
    local count = stats.get_input_count(item)
    if count and count > 0 then produced[#produced + 1] = item .. "=" .. count end
  end
  table.sort(produced)
  lines[#lines + 1] = "produced|" .. table.concat(produced, ",")

  local task = storage.frrl_task or {}
  lines[#lines + 1] = string.format(
    "task|transfers=%d|items=%d", task.transfers or 0, task.items_moved or 0)

  return lines
end

return world
