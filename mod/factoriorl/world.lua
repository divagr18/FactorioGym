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

-- A declared inventory is a JSON object, so `pairs` over it follows the Lua
-- table's hash layout -- stable inside Factorio and specified nowhere else. It
-- decides which slot each stack lands in, which a simulator has to reproduce
-- and could not: a scene declaring coal, iron ore and a furnace was installed
-- furnace, coal, ore. Items go in by name instead.
local function sorted_pairs(map)
  local keys = {}
  for key in pairs(map) do keys[#keys + 1] = key end
  table.sort(keys)
  local index = 0
  return function()
    index = index + 1
    local key = keys[index]
    if key ~= nil then return key, map[key] end
  end
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
--
-- It was not enough, and `force.reset` is now the first step. A trigger
-- technology counts its items in a per-force counter the API does not expose
-- (`saved_progress` reads 0) and that neither un-researching, toggling
-- `researched`, disabling the technology nor clearing production statistics
-- touches. `steam-power` is "craft 50 iron plates", so a line that smelted 27
-- plates in one episode researched it 23 plates into the next -- mid-episode,
-- on a worker with history, and never on a fresh one. The M2 parity recorder
-- caught it as a recording and its replay disagreeing at one decision. Measured
-- on 2.0.60: after `force.reset` the same 27 + 27 plates research nothing.
-- It also unchartes the map, which only the open world uses, and the open world
-- does not come through here.
function world.reset_force()
  local force = game.forces["player"]
  force.reset()
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


-- ---------------------------------------------------------------- open world

-- Initialise an ordinary generated map without destroying it.
--
-- Every other route into a scene goes through `clear_scene`, which sweeps
-- **neutral**-force entities inside the scene box -- and natural resources,
-- trees and rocks are neutral. That is correct for a painted benchmark scene and
-- catastrophic for a generated one: it would delete the ore the agent is
-- supposed to go and find.
--
-- So this sweeps the **player** force only. The reference scene that
-- `runtime.on_init` paints into every freshly created save -- two chests, a wall
-- row, fifty iron plates -- is player force, so it goes; the map does not. That
-- also makes this safe to run on a save that already had a scene installed,
-- which gating `on_init` would not have been.
--
-- Three things it deliberately does not do:
--
--   * `world.reset_force()`, which un-researches every technology. A generated
--     save already has prototype-default research, and on a resumed world this
--     would silently throw away everything the agent had researched.
--   * `always_day`. A benchmark scene freezes the clock so lighting cannot vary
--     between episodes; an open world should have its ordinary day and night.
--   * grant anything beyond the declared inventory. The caller passes freeplay's
--     own `created_items`, read from the installed game.
--- Every resource patch inside the charted area, aggregated.
--
-- The world charts a large square at creation and published none of it, while
-- the agent's own sensor reaches 32 tiles. On seed 20260910 -- coal at 79.7
-- tiles, trees at 81.0, stone at 104.8 -- the prompt never mentioned stone at
-- all, and three paid runs spent most of their decisions walking in expanding
-- squares through territory the force had already mapped. "Why does it not go
-- and find stone" had a flat answer: it was never told stone exists.
--
-- Map-view information, which is what a player reads before deciding where to
-- walk. Not a build plan: it says what is out there and roughly where, and
-- nothing about what to do with it.
--
-- Aggregated on a coarse grid rather than clustered properly. A patch is
-- hundreds of tiles and the useful fact is "coal, about 80 tiles west, big",
-- so tiles are bucketed by `SURVEY_CELL` and merged; real clustering would be
-- more code for a distinction the agent cannot act on.
local SURVEY_CELL = 32

function world.survey(radius)
  local srf = surface()
  local buckets = {}
  for _, entity in pairs(srf.find_entities_filtered({
    area = { { -radius, -radius }, { radius, radius } },
    type = "resource",
  })) do
    if entity.valid then
      local cx = math.floor(entity.position.x / SURVEY_CELL)
      local cy = math.floor(entity.position.y / SURVEY_CELL)
      local key = entity.name .. ":" .. cx .. ":" .. cy
      local bucket = buckets[key]
      if not bucket then
        bucket = { name = entity.name, tiles = 0, amount = 0, sx = 0, sy = 0 }
        buckets[key] = bucket
      end
      bucket.tiles = bucket.tiles + 1
      bucket.amount = bucket.amount + (entity.amount or 0)
      bucket.sx = bucket.sx + entity.position.x
      bucket.sy = bucket.sy + entity.position.y
    end
  end
  local patches = {}
  for _, bucket in pairs(buckets) do
    patches[#patches + 1] = {
      name = bucket.name,
      tiles = bucket.tiles,
      amount = bucket.amount,
      -- Centre of mass, which is where "walk to the coal" should aim.
      position = { bucket.sx / bucket.tiles, bucket.sy / bucket.tiles },
    }
  end
  table.sort(patches, function(a, b)
    local da = a.position[1] ^ 2 + a.position[2] ^ 2
    local db = b.position[1] ^ 2 + b.position[2] ^ 2
    return da < db
  end)
  return patches
end

function world.open_world(options)
  options = options or {}
  -- `fresh` decides whether this is a *start* or a *resume*, and it gates the
  -- two destructive things below. The sweep in particular: on a freshly created
  -- save the player-force entities are the reference scene `on_init` paints --
  -- two chests, a wall row -- and removing them is the whole point. On a mid-run
  -- checkpoint they are the factory the agent spent thirty minutes building, and
  -- removing them would destroy exactly what a resume exists to recover.
  --
  -- This was ungated when open_world was written, because nothing could produce
  -- a save to resume from and so nothing could hit it.
  local fresh = options.fresh ~= false
  local srf = surface()
  local removed = 0
  if fresh then
    for _, entity in pairs(srf.find_entities_filtered({ force = "player" })) do
      if entity.valid and entity.type ~= "character" then
        entity.destroy({ raise_destroy = true })
        removed = removed + 1
      end
    end
  end

  local ch = world.ensure_character()
  -- Only moved on a fresh start. A resume leaves the character wherever the save
  -- holds it: teleporting it home would undo whatever walking the parent segment
  -- did, and the agent's memory of where it was is already gone.
  if fresh or options.position then
    local spawn = options.position or { 0, 0 }
    ch.teleport({ spawn[1], spawn[2] })
  end

  -- Requested against delivered, per item. `insert` returns what it actually
  -- took, and a partial insert would otherwise look like a successful start
  -- with a quietly smaller inventory.
  local inv = ch.get_inventory(defines.inventory.character_main)
  local delivered, undelivered = {}, {}
  for item, count in pairs(options.inventory or {}) do
    local taken = inv.insert({ name = item, count = count })
    delivered[item] = taken
    if taken < count then
      undelivered[#undelivered + 1] = { item = item, requested = count, delivered = taken }
    end
  end

  -- Chart the starting area so the first observation is not of a blank map.
  -- Freeplay charts too; an uncharted spawn would make the agent's first
  -- several decisions be about revealing terrain rather than about building.
  local radius = options.chart_radius or 96
  game.forces["player"].chart(srf, { { -radius, -radius }, { radius, radius } })
  -- Charting *reveals* chunks; it does not generate them. Surveying straight
  -- after a chart returned zero patches, because `find_entities_filtered` was
  -- scanning ground that did not exist yet. Requesting generation and forcing
  -- the queue makes the area real before it is measured.
  --
  -- This is the one place in the run where a synchronous chunk generation is
  -- affordable: it happens once, before the clock starts, and a world whose
  -- resources the agent will be told about has to have those resources
  -- generated.
  srf.request_to_generate_chunks({ 0, 0 }, math.ceil(radius / 32) + 1)
  srf.force_generate_chunk_requests()
  local survey = world.survey(radius)

  storage.frrl_scene = {
    name = "open_world",
    aliases = {},
    markers = {},
    -- No objective markers at all: there is no declared goal geometry to
    -- withhold, and nothing for `public_markers` to filter.
    public_markers = {},
    extra_tracked_items = options.extra_tracked_items or {},
    radius = radius,
    open_world = true,
    fresh = fresh,
  }

  return {
    -- Surveyed once at creation and carried here. Resource patches do not
    -- move, so this is world state that is constant for the run, which is
    -- exactly what belongs in a cached prefix rather than in every turn.
    survey = survey,
    survey_radius = radius,
    scenario = storage.frrl_scene.name,
    -- Zero on a resume, by construction. A non-zero count there means the sweep
    -- ran when it should not have, which is directly assertable.
    destroyed = removed,
    delivered = delivered,
    undelivered = undelivered,
    position = { ch.position.x, ch.position.y },
    fresh = fresh,
  }
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
  for _, spec in ipairs(blueprint.resources or {}) do
    local position = { spec.position[1], spec.position[2] }
    if srf.can_place_entity({ name = spec.name, position = position }) then
      srf.create_entity({
        name = spec.name, position = position, amount = spec.amount or 1000,
      })
    end
  end
  -- Contents a scene declared and the engine would not take, per entity. A
  -- scene that cannot install its own declaration is a defect in the task, and
  -- this used to be silent: `LuaEntity.insert` refuses iron plates into a
  -- stone furnace -- the result slot is not reachable that way -- so
  -- `diagnose_line` declared a hundred-plate output jam in every scene, the
  -- engine took none of them, and the fault the family was built around simply
  -- did not exist. Four solvability runs and a blueprint-level test that
  -- asserted the *declaration* all failed to see it. So the shortfall is
  -- reported now, and `install` carries it back to Python.
  local undelivered = {}
  for _, spec in ipairs(blueprint.entities or {}) do
    local created = srf.create_entity({
      name = spec.name,
      position = { spec.position[1], spec.position[2] },
      direction = DIRECTIONS[spec.direction or "north"],
      force = spec.force or "player",
    })
    if created then
      if spec.contents then
        for item, count in sorted_pairs(spec.contents) do
          local placed = created.insert({ name = item, count = count })
          if placed < count then
            -- The result slot, explicitly. `insert` picks an inventory by what
            -- the item is *for*, and a furnace's product is for neither its
            -- source nor its fuel slot, so a declared output has to name the
            -- inventory it belongs in.
            local output = created.get_output_inventory()
            if output then
              placed = placed + output.insert({ name = item, count = count - placed })
            end
          end
          if placed < count then
            undelivered[#undelivered + 1] =
              spec.name .. "|" .. item .. "|" .. placed .. "/" .. count
          end
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
    for item, count in sorted_pairs(character.inventory) do
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
  -- `undelivered` is empty on every well-formed scene, and non-empty is a
  -- task defect rather than a warning: the scene the evaluator installed is
  -- not the scene the task declared.
  return {
    scenario = storage.frrl_scene.name,
    destroyed = destroyed,
    undelivered = undelivered,
  }
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
--- Apply one declared disruption to the running scene.
--
-- The only mutation the evaluator can make to an installed scene, and it is
-- typed rather than arbitrary Lua on purpose. Every disruption in this repo
-- before now was a driver-side `bridge.run` string fired at a hard-coded point
-- in a five-phase script: nothing declared it, nothing recorded it in a task
-- spec, and a run's trace could not say what had been done to the world. A
-- declared kind with declared targets is comparable across runs; a Lua string
-- is not.
--
-- `empty_fuel` clears the *fuel inventory* and deliberately does not touch
-- `entity.energy`. That is not an oversight -- it is the disruption R4.2
-- measured. A burner drill keeps running for about 1,170 ticks on the item
-- already in its burner (2,999,833 J against 150 kW), so a task that wants a
-- real outage has to wait for the decay rather than assume the clear stopped
-- anything. `docs/evidence/r4-burner-decay.json` is that measurement.
--
-- Targets are marker aliases first, then prototype names, so a task can name
-- either the entity it declared or a class of machine.
--
-- Announces nothing. No marker is published and no event is appended, because
-- R4.3's persistent-operation track asks for a disruption detectable only by
-- monitoring -- and status, fuel and output are already observable.
function world.disrupt(kind, targets)
  local surface = game.surfaces[1]
  if not surface then return nil, "no surface" end
  local wanted = {}
  for _, name in pairs(targets or {}) do wanted[name] = true end

  local entities = {}
  local seen = {}
  for name in pairs(wanted) do
    -- Marker alias first: `world.alias_entity` is how every other evaluator
    -- read resolves a scene's declared names, and reaching into
    -- `storage.frrl_scene` directly would be a second way to do the same thing.
    local aliased = world.alias_entity(name)
    if aliased then
      seen[aliased.unit_number] = true
      entities[#entities + 1] = aliased
    end
    -- Only if the name is a real prototype. A target may be a *marker alias*
    -- -- `keep_line_running` declares `drill` and `furnace` -- and
    -- `find_entities_filtered` raises "Unknown entity name: drill" on anything
    -- that is not a prototype, which took the whole request down with it.
    -- Caught by `gate_r4`, whose disruption clause reported
    -- `touched: []`, `ok: false` beside an observation still showing 47 coal.
    local ok, found = pcall(function()
      return surface.find_entities_filtered({ name = name })
    end)
    if ok then
      for _, entity in pairs(found) do
        if not seen[entity.unit_number] then
          seen[entity.unit_number] = true
          entities[#entities + 1] = entity
        end
      end
    end
  end
  if #entities == 0 then
    return nil, "no entity matched " .. table.concat(targets or {}, ",")
  end

  local touched = {}
  if kind == "empty_fuel" then
    for _, entity in pairs(entities) do
      local inventory = entity.get_fuel_inventory()
      if inventory then
        local taken = 0
        for _, stack in pairs(inventory.get_contents()) do taken = taken + stack.count end
        inventory.clear()
        touched[#touched + 1] = entity.name .. "=" .. taken
      end
    end
  elseif kind == "fill_output" then
    -- The other measured way a line stops: a full result slot. Coal cannot fix
    -- it, which is what makes it a different fault rather than a louder one.
    for _, entity in pairs(entities) do
      local inventory = entity.get_output_inventory()
      if inventory then
        -- One stack, which is what a stone furnace's result slot holds and so
        -- what stops it. `insert` returns how many it actually took, and that
        -- number goes in the reply rather than the number asked for.
        local added = inventory.insert({ name = "iron-plate", count = 100 })
        touched[#touched + 1] = entity.name .. "+" .. added
      end
    end
  else
    return nil, "unknown disruption kind: " .. tostring(kind)
  end

  return {
    kind = kind,
    tick = game.tick,
    targets = targets,
    touched = touched,
    -- What the clear did *not* do, stated in the reply so a trace carries it.
    note = "fuel inventories only; entity.energy and burner.remaining_burning_fuel untouched",
  }
end


-- ------------------------------------------------------- by-hand tallies
--
-- Roadmap A4.2 requires cumulative *machine* production to be kept separate
-- from handcrafting and from inventory transfers. The engine will not do this
-- for us: `get_item_production_statistics():get_input_count(item)` counts every
-- item that enters the force's inventory space, so a plate a furnace made and a
-- plate a character hand-crafted are the same number, and mined ore is in there
-- too.
--
-- There is no counter for "machine output" to read. So it is derived --
--
--     machine_produced = produced - handcrafted - mined
--
-- and the three components are published separately, so a reader can check the
-- subtraction instead of trusting it. Anything the derivation cannot see (items
-- from the crash site, research rewards) lands in the machine column, which is
-- why `world.truth` labels it a derivation.

local function tally(bucket, name, count)
  if not name or not count or count <= 0 then return end
  storage.frrl_tally = storage.frrl_tally or {}
  local into = storage.frrl_tally[bucket] or {}
  into[name] = (into[name] or 0) + count
  storage.frrl_tally[bucket] = into
end

world.tally = tally

--- Items a *player* finished crafting by hand.
function world.on_player_crafted_item(event)
  local stack = event and event.item_stack
  if stack and stack.valid_for_read then tally("handcrafted", stack.name, stack.count) end
end

--- Items a *player* got by mining.
function world.on_player_mined_item(event)
  local stack = event and event.item_stack
  if stack and stack.valid_for_read then tally("mined", stack.name, stack.count) end
end

--- What the tallies hold, as plain tables.
function world.tallies()
  local held = storage.frrl_tally or {}
  return {
    handcrafted = held.handcrafted or {},
    mined = held.mined or {},
    -- Counted by the mod's own action handlers rather than by an engine event.
    -- Whether `on_player_crafted_item` and `on_player_mined_item` fire at all
    -- for a controlled character depends on whether that character has a
    -- LuaPlayer behind it, which is a fact about this build and this mod, not
    -- something to assume -- so both are recorded and `world.truth` reports
    -- which one actually moved.
    handcrafted_by_action = held.handcrafted_by_action or {},
    mined_by_action = held.mined_by_action or {},
  }
end


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

  -- The eight names below were what a painted benchmark scene needed. An open
  -- world produces whatever the agent decides to produce, so "outputs by item
  -- type" (roadmap A4.2) cannot come from a fixed list. The engine's own
  -- `input_counts` is the live set; the fixed names stay as a floor so a scene
  -- predicate keeps seeing a key it expects even at zero.
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
  -- Whatever else moved. Guarded because the accessor's spelling is a fact
  -- about this build: if it is not there, the fixed list above is still the
  -- answer and the report says the live set was unavailable.
  local live_set = true
  local ok, counts = pcall(function() return stats.input_counts end)
  if ok and type(counts) == "table" then
    for name, count in pairs(counts) do
      if count and count > 0 then produced[name] = count end
    end
  else
    live_set = false
  end

  -- A4.2's separation. Derived, because the engine has no counter for it.
  local held = world.tallies()
  -- Prefer whichever by-hand source actually moved. Player events do not fire
  -- for a character with no LuaPlayer behind it, and which of those two worlds
  -- this mod is in is a measured fact, reported below rather than assumed.
  local function total(bucket)
    local sum = 0
    for _, count in pairs(bucket) do sum = sum + count end
    return sum
  end
  local by_event = total(held.handcrafted) + total(held.mined)
  local handcrafted = by_event > 0 and held.handcrafted or held.handcrafted_by_action
  local mined = by_event > 0 and held.mined or held.mined_by_action
  local machine_produced = {}
  for name, count in pairs(produced) do
    local rest = count - (handcrafted[name] or 0) - (mined[name] or 0)
    if rest > 0 then machine_produced[name] = rest end
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
  -- Stored energy, which decides how long a machine keeps running after its
  -- fuel *inventory* is emptied. `get_fuel_inventory():clear()` does not touch
  -- it, so the Phase 5 disruption left the drill at `status = working` with
  -- `fuel = 0` -- and nothing anywhere could say for how long, because neither
  -- the observation, the truth channel nor the digest carried this.
  --
  -- Evaluator-side only, deliberately. R4.2 needs it to *account* for stored
  -- energy when defining an outage; putting it in the observation would change
  -- the policy contract and the profile version for a quantity a player reads
  -- off a fuel bar rather than a number.
  local stored_energy = {}
  local burning = {}
  local srf = surface()
  if srf then
    for _, entity in ipairs(srf.find_entities_filtered({ force = force })) do
      if entity.valid and entity.name then
        placed_counts[entity.name] = (placed_counts[entity.name] or 0) + 1
        local ok, status = pcall(function() return entity.status end)
        if ok and status == defines.entity_status.working then
          working_counts[entity.name] = (working_counts[entity.name] or 0) + 1
        end
        local ok_energy, energy = pcall(function() return entity.energy end)
        if ok_energy and energy and energy > 0 then
          stored_energy[entity.name] = (stored_energy[entity.name] or 0) + energy
        end
        local ok_burner, remaining = pcall(function()
          return entity.burner and entity.burner.remaining_burning_fuel or nil
        end)
        if ok_burner and remaining and remaining > 0 then
          burning[entity.name] = (burning[entity.name] or 0) + remaining
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
    -- The game tick this reading was taken at. A4.2 measures production in
    -- 60-second *game*-time windows, and a sample with no tick cannot be put
    -- in one -- which is what the wall-clock sampler needs it for. Every
    -- existing caller reads the four keys below and ignores this.
    tick = game.tick,
    markers = markers,
    containers = containers,
    working = working,
    produced = produced,
    -- Additive: every existing predicate reads only the four above.
    working_counts = working_counts,
    placed_counts = placed_counts,
    -- Joules, summed per prototype. See the comment where they are gathered.
    stored_energy = stored_energy,
    remaining_burning_fuel = burning,
    built = built,
    -- Roadmap A4.2. `machine_produced` is a **derivation**, not a reading:
    -- the engine counts every item entering the force's inventory space and
    -- has no separate counter for machine output. All three components are
    -- returned so a reader can check the subtraction rather than trust it.
    --
    -- Known biases, both of which inflate the machine column:
    --   * items granted outside production (crash-site loot, research rewards)
    --   * intermediates the crafting queue builds on the way to a requested
    --     recipe, which `handcrafted_by_action` does not see
    machine_produced = machine_produced,
    handcrafted = handcrafted,
    mined_by_hand = mined,
    by_hand_source = by_event > 0 and "player_events" or "mod_actions",
    tracked_item_set = live_set and "live" or "fixed_list",
    production_note = (
      "machine_produced is produced minus handcrafted minus mined_by_hand; "
      .. "it is derived, not measured"
    ),
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
  -- Cleared with the statistics they are subtracted from. Leaving them would
  -- make the next episode's derived machine output negative.
  storage.frrl_tally = {}
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
      -- Stored energy and the fuel item mid-burn. The digest is "everything a
      -- reset must restore", and it could not see either -- so two worlds with
      -- equal digests could hold different amounts of energy in a burner,
      -- which is precisely the quantity R4.2's three arms are comparing.
      -- Rounded to whole joules: an exact float would make the digest depend
      -- on formatting rather than on state.
      local ok_energy, energy = pcall(function() return entity.energy end)
      parts[#parts + 1] = "energy=" ..
        string.format("%d", (ok_energy and energy) and math.floor(energy + 0.5) or 0)
      local ok_burner, remaining = pcall(function()
        return entity.burner and entity.burner.remaining_burning_fuel or nil
      end)
      parts[#parts + 1] = "burning=" ..
        string.format("%d", (ok_burner and remaining) and math.floor(remaining + 0.5) or 0)
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


-- ---------------------------------------------------------------- hidden state

-- Exact floats as strings. A progress or energy value is state a simulator
-- must load and reproduce, and a JSON number's formatting is the encoder's
-- choice; "%.17g" round-trips an IEEE double exactly.
local function exact(value)
  if value == nil then return nil end
  return string.format("%.17g", value)
end

-- Positions in 256ths of a tile: the engine stores them fixed-point, so this is
-- an integer and loses nothing.
local function fixed(position)
  return { math.floor(position.x * 256 + 0.5), math.floor(position.y * 256 + 0.5) }
end

local STATUS_NAMES = nil
local function status_name(entity)
  if STATUS_NAMES == nil then
    STATUS_NAMES = {}
    for name, value in pairs(defines.entity_status) do STATUS_NAMES[value] = name end
  end
  local ok, code = pcall(function() return entity.status end)
  return ok and code and STATUS_NAMES[code] or nil
end

local function try(fn)
  local ok, value = pcall(fn)
  if ok then return value end
  return nil
end

-- Slot by slot, not `get_contents()`: which slot a stack sits in decides whether
-- the next insert fits, and a total cannot say.
local function slots(inv)
  if not inv then return nil end
  local out = {}
  for index = 1, #inv do
    local stack = inv[index]
    if stack.valid_for_read then
      out[#out + 1] = { index, stack.name, stack.count }
    end
  end
  return { size = #inv, stacks = out }
end

-- 2.0 reports the item mid-burn as a prototype pair; older builds as a
-- prototype. Either way only the name is state.
local function burning_name(burner)
  local current = try(function() return burner.currently_burning end)
  if not current then return nil end
  local name = try(function() return current.name end)
  if type(name) == "table" or type(name) == "userdata" then
    name = try(function() return name.name end)
  end
  return type(name) == "string" and name or nil
end

local function by_position(a, b)
  if a.position[2] ~= b.position[2] then return a.position[2] < b.position[2] end
  if a.position[1] ~= b.position[1] then return a.position[1] < b.position[1] end
  return a.name < b.name
end

--- Everything a one-step sync needs that neither the observation nor the
--- digest carries exactly: machine progress, stored energy, the item mid-burn,
--- slot layouts, the character's mining and walking state, ground items and
--- resource amounts. Evaluator-only, like the digest, and ordered by world
--- position so two engines -- or an engine and a simulator -- list it the same.
function world.hidden_state()
  local srf = surface()
  local out = { tick = game.tick }

  -- Player entities anywhere, and neutral ones inside the scene's box: a scene's
  -- walls are neutral, and a simulator that lost them would walk through them.
  -- Only inside the box, because outside it is map generation the reset never
  -- touches -- rocks, and fish that swim, so two recordings of one scene
  -- disagreed about a fish 60 tiles away. Resources and ground items are listed
  -- separately below.
  local radius = (storage.frrl_scene and storage.frrl_scene.radius) or 64
  local candidates = srf.find_entities_filtered({ force = "player" })
  for _, entity in pairs(srf.find_entities_filtered({
    force = "neutral", area = { { -radius, -radius }, { radius, radius } },
  })) do
    candidates[#candidates + 1] = entity
  end
  local entities = {}
  for _, entity in pairs(candidates) do
    if entity.valid and entity.type ~= "character" and entity.type ~= "resource"
        and entity.type ~= "item-entity" then
      local record = {
        name = entity.name,
        force = entity.force.name,
        position = fixed(entity.position),
        direction = entity.direction,
        status = status_name(entity),
        energy = exact(try(function() return entity.energy end)),
        mining_progress = exact(try(function() return entity.mining_progress end)),
        bonus_mining_progress = exact(try(function() return entity.bonus_mining_progress end)),
        crafting_progress = exact(try(function() return entity.crafting_progress end)),
        bonus_progress = exact(try(function() return entity.bonus_progress end)),
        products_finished = try(function() return entity.products_finished end),
      }
      local burner = try(function() return entity.burner end)
      if burner then
        record.remaining_burning_fuel = exact(burner.remaining_burning_fuel)
        record.currently_burning = burning_name(burner)
      end
      -- By entity type, not by index: `defines.inventory.chest` and `fuel` are
      -- both 1, so asking every entity for both listed a furnace's fuel twice.
      local inventories = {
        fuel = slots(try(function() return entity.get_fuel_inventory() end)),
        burnt_result = slots(try(function() return entity.get_burnt_result_inventory() end)),
      }
      if entity.type == "container" or entity.type == "logistic-container" then
        inventories.chest = slots(entity.get_inventory(defines.inventory.chest))
      elseif entity.type == "furnace" then
        inventories.furnace_source = slots(entity.get_inventory(defines.inventory.furnace_source))
        inventories.furnace_result = slots(entity.get_inventory(defines.inventory.furnace_result))
      end
      record.inventories = inventories
      entities[#entities + 1] = record
    end
  end
  table.sort(entities, by_position)
  out.entities = entities

  local ground = {}
  for _, entity in pairs(srf.find_entities_filtered({ type = "item-entity" })) do
    if entity.valid and entity.stack and entity.stack.valid_for_read then
      ground[#ground + 1] = {
        name = entity.stack.name,
        count = entity.stack.count,
        position = fixed(entity.position),
      }
    end
  end
  table.sort(ground, by_position)
  out.ground_items = ground

  local resources = {}
  for _, entity in pairs(srf.find_entities_filtered({ type = "resource" })) do
    if entity.valid then
      resources[#resources + 1] = {
        name = entity.name,
        position = fixed(entity.position),
        amount = entity.amount or 0,
      }
    end
  end
  table.sort(resources, by_position)
  out.resources = resources

  local ch = storage.frrl_character
  if ch and ch.valid then
    local mining = try(function() return ch.mining_state end) or {}
    local walking = try(function() return ch.walking_state end) or {}
    local selected = try(function() return ch.selected end)
    out.character = {
      position = fixed(ch.position),
      direction = ch.direction,
      mining = mining.mining or false,
      mining_position = mining.position and fixed(mining.position) or nil,
      mining_progress = exact(try(function() return ch.character_mining_progress end)),
      walking = walking.walking or false,
      walking_direction = walking.direction,
      selected = selected and selected.valid and {
        name = selected.name, position = fixed(selected.position),
      } or nil,
      crafting_queue_size = try(function() return ch.crafting_queue_size end) or 0,
      main = slots(ch.get_inventory(defines.inventory.character_main)),
    }
  end
  return out
end

return world
