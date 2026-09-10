-- Static game data, read straight out of the loaded prototypes.
--
-- The mod has always had `prototypes.*` in reach and has never returned any of
-- it. Three call sites use it, all of them to *check* something --
-- `actions.lua` reads `prototypes.recipe[...].products[1]` to know what a craft
-- will produce, `prototypes.item[...].place_result` to know whether an item
-- places anything -- and every one of those facts is thrown away as soon as the
-- action answers. So a client that wants to plan has no way to learn what iron
-- gear wheels are made of except by trying to craft them, and
-- `tasks/spec.py` ended up carrying a hardcoded `ENTITY_TILE_SIZES` table of
-- four prototypes because the footprint of a stone furnace was not askable.
--
-- Read-only, and deliberately outside `MUTATING`: this reads prototypes and
-- force state and writes nothing, so deduplicating it would make a second
-- request return a stored answer instead of a current one.
--
-- What is *not* static
-- --------------------
-- Recipe `enabled`, technology `researched` and the placement gate all come off
-- `game.forces["player"]`, not off a prototype: they are the force's state at
-- the moment of the call, and research moves them. The reply says which tick it
-- was taken at so a caller caching this on disk can tell how old the mutable
-- half is. Everything else -- ingredients, products, energy, tile sizes,
-- prerequisites, unlock effects -- is fixed for a given engine build and mod
-- set.
--
-- Ordering is settled here rather than left to the caller. Every collection
-- below is built by walking a Lua hash (`force.recipes`, `prototypes.item`,
-- `force.technologies`), and `pairs` order is not stable across runs, so an
-- unsorted reply would serialise differently every time it was fetched. The
-- Python side puts this in a cached prompt prefix, where a reordered line is
-- indistinguishable from a changed one.

local knowledge = {}

local function player_force()
  return game.forces["player"]
end

local function sorted_keys(collection)
  local names = {}
  for name in pairs(collection) do names[#names + 1] = name end
  table.sort(names)
  return names
end

--- `{ name = ..., amount = ... }` for a recipe's ingredient or product list.
--
-- Kept in prototype order, which is an array and therefore already stable --
-- sorting it would scramble the reading order a recipe is written in.
--
-- Two shapes need flattening. A fluid ingredient is the same table with
-- `type = "fluid"`, and that distinction matters to an agent because a fluid
-- cannot be carried or hand-crafted; it is reported only when it is not
-- "item", so the common case costs nothing. A probabilistic product carries
-- `amount_min`/`amount_max`/`probability` instead of `amount` -- uranium
-- processing and every ore-crushing recipe do -- and reporting `nil` there
-- would render as a missing number rather than as a range.
local function stack_list(entries)
  local out = {}
  for index, entry in ipairs(entries or {}) do
    local stack = { name = entry.name, amount = entry.amount }
    if stack.amount == nil then
      stack.amount = entry.amount_max or entry.amount_min or 0
      stack.amount_min = entry.amount_min
      stack.amount_max = entry.amount_max
    end
    if entry.type and entry.type ~= "item" then stack.type = entry.type end
    -- Only when it is actually a chance. `probability = 1` is the default and
    -- would otherwise appear on every product of every recipe.
    if entry.probability and entry.probability < 1 then
      stack.probability = entry.probability
    end
    out[index] = stack
  end
  return out
end

--- Every recipe the player force knows about, enabled or not.
--
-- Read off `force.recipes` rather than `prototypes.recipe` because the two
-- carry different halves of the answer: the prototype has the ingredients and
-- the timing, the force has whether it is unlocked, and a planner needs both in
-- the same row. `force.recipes` is keyed by every recipe prototype, so nothing
-- is lost by going through it.
--
-- Recipes with no products are dropped. That is not a taste judgement: on
-- 2.0.60 `parameter-0` through `parameter-9` are enabled, unhidden, and have
-- zero ingredients and zero products -- placeholders for parametrised
-- blueprints. `observations.lua` already filters them on the same property,
-- where they had been occupying 10 of the 22 recipe slots a policy could see.
-- The count that was dropped is reported, so the omission is visible rather
-- than inferred from a short list.
function knowledge.recipes()
  local force = player_force()
  local out, dropped = {}, 0
  if not force then return out, dropped end
  for _, name in ipairs(sorted_keys(force.recipes)) do
    local recipe = force.recipes[name]
    if not recipe.products or #recipe.products == 0 then
      dropped = dropped + 1
    else
      out[#out + 1] = {
        name = name,
        ingredients = stack_list(recipe.ingredients),
        products = stack_list(recipe.products),
        -- Seconds at crafting speed 1, which is what the recipe tooltip shows
        -- and what `actions.lua` already returns from a started craft.
        energy = recipe.energy,
        category = recipe.category,
        -- Force state, not prototype state. See the header.
        enabled = recipe.enabled,
        -- Reported only when true, so the common row stays short. A hidden
        -- recipe still runs -- it is only kept out of the crafting menu -- so
        -- dropping these would hide real production routes.
        hidden = recipe.hidden or nil,
      }
    end
  end
  return out, dropped
end

--- Items that place an entity, with the footprint that placement needs.
--
-- Keyed by the **item** name, because that is what the `place` action takes and
-- what the agent holds in its inventory. The entity name is carried alongside
-- and is usually but not always the same string.
--
-- `tile_width`/`tile_height` are the numbers `tasks/spec.py` hardcodes for four
-- prototypes and guesses as 1x1 for everything else. A guessed footprint is
-- worse than no footprint: it decides where a task believes an entity fits.
--
-- The recipe gate is reported because `actions.lua` applies a stricter rule
-- than the base game does. Base Factorio gates technology at the recipe layer
-- only, so holding an item is normally enough to place it; the action profile
-- refuses placement when a recipe of the same name exists and is disabled, so
-- PLAN 2.1's "placement cannot bypass technology restrictions" is directly
-- testable. An agent cannot predict that refusal without knowing which item
-- names have a recipe behind them.
--- The tiles a placement covers, as offsets from the tile the agent names.
--
-- "Centred on the placement position" is what this table used to say, and it is
-- false for every even-sized machine. Measured against the engine
-- (`tools/probe_footprints.py`, `docs/evidence/footprints.json`): a 2x2 stone
-- furnace asked for at [10.5, 10.5] covers x 10..11, y 10..11 -- the named tile
-- is its north-WEST corner -- while a 3x3 lab asked for at the same place
-- covers x 9..11 and genuinely is centred. Two rules, by parity, plus each
-- item's size: an agent cannot infer that, and run 7 did not. It put a burner
-- drill where one of the four tiles under it held ore and got
-- `no_minable_resources` in seconds.
--
-- So the conclusion is published instead of the ingredients, per item, and it
-- is derived from `collision_box` rather than from `tile_width`. They disagree:
-- an offshore pump reports 1x1 and occupies 3x2, and a rule built on tile size
-- would have shipped a second false statement to fix the first.
local function footprint(entity)
  -- Where the engine puts the entity when tile (0, 0) is named: an odd-sized
  -- machine lands on the tile centre, an even-sized one on its south-east
  -- corner. This reproduces all 22 measured placements, offshore pump included.
  local ox = (entity.tile_width % 2 == 1) and 0.5 or 1.0
  local oy = (entity.tile_height % 2 == 1) and 0.5 or 1.0
  local ok, box = pcall(function() return entity.collision_box end)
  if not ok or not box then return nil, ox, oy end
  return {
    math.floor(ox + box.left_top.x), math.floor(oy + box.left_top.y),
    math.ceil(ox + box.right_bottom.x) - 1, math.ceil(oy + box.right_bottom.y) - 1,
  }, ox, oy
end

--- What a machine needs before it will do anything, and what it makes.
--
-- A run placed a lab, and the lab sat there. The observation said `no power`,
-- which is the engine's own diagnosis and is true, and nothing anywhere said
-- what "power" is or where it comes from. A burner gets `FUEL SLOT EMPTY --
-- give it some`, which names the remedy; an electric machine got a symptom.
--
-- Worse, nothing said a lab consumes science packs, so even a powered one would
-- have stood idle for a second unstated reason. Both facts are on the
-- prototype and both are what the game shows a player in the entity tooltip:
-- the electricity bar, and the pack icons in the lab's input slots.
local function needs(entity)
  local row = {}
  local ok_electric, electric = pcall(function()
    return entity.electric_energy_source_prototype
  end)
  local ok_burner, burner = pcall(function() return entity.burner_prototype end)
  if ok_electric and electric then
    row.power = "electric"
  elseif ok_burner and burner then
    row.power = "burner"
  end
  -- A generator is the answer to `no power`, and it is only findable if it is
  -- labelled: a steam engine produces 15000, a solar panel 1000, and everything
  -- else in the early game produces nothing.
  local ok_out, produced = pcall(function() return entity.get_max_energy_production() end)
  if ok_out and produced and produced > 0 then row.generates = produced end
  local ok_lab, inputs = pcall(function() return entity.lab_inputs end)
  if ok_lab and inputs and #inputs > 0 then row.consumes = inputs end
  return row
end

function knowledge.placeable()
  local force = player_force()
  local out = {}
  for _, name in ipairs(sorted_keys(prototypes.item)) do
    local item = prototypes.item[name]
    local entity = item.place_result
    if entity then
      local gate = force and force.recipes[name] or nil
      local covers, ox, oy = footprint(entity)
      local mines = nil
      local ok_radius, radius = pcall(function() return entity.mining_drill_radius end)
      if ok_radius and radius and radius > 0 then
        mines = {
          math.floor(ox - radius), math.floor(oy - radius),
          math.ceil(ox + radius) - 1, math.ceil(oy + radius) - 1,
        }
      end
      local wants = needs(entity)
      out[#out + 1] = {
        name = name,
        entity = entity.name,
        width = entity.tile_width,
        height = entity.tile_height,
        covers = covers,
        power = wants.power,
        generates = wants.generates,
        consumes = wants.consumes,
        -- What a mining drill actually harvests. A burner drill searches
        -- exactly its own four tiles (radius 0.99) and an electric one a 5x5
        -- square (radius 2.49), so "stand it on ore" is true for neither
        -- without knowing which tiles are under it.
        mines = mines,
        -- Named rather than boolean: an agent that is refused a placement needs
        -- to know which recipe to research, and the item name is not always the
        -- recipe name for other prototypes.
        gated_by = gate and gate.name or nil,
        -- Force state, like `recipes[].enabled`. Absent when nothing gates it.
        unlocked = gate and gate.enabled or nil,
      }
    end
  end
  return out
end

--- The technology tree: cost, prerequisites, and what each unlock buys.
--
-- `unlocks` is the unlock-recipe effects only. The others (bonuses, inserter
-- capacity, character reach) do not name a recipe, and the question this table
-- exists to answer is "what do I research to be allowed to craft X".
--
-- `trigger` exists because Factorio 2.0 roots the tree in trigger technologies:
-- 7 of 196 complete by crafting or mining an item and cannot be queued at all,
-- which is why `protocol.py` carries `tech_not_selectable` as a code distinct
-- from `tech_locked`. Without this field an agent told "not selectable" has no
-- way to learn what would actually finish it, and would wait for prerequisites
-- that never arrive.
--
-- `research_unit_count` is behind a pcall: for an infinite technology it is
-- derived from a formula and a level rather than stored, and raising here would
-- take the whole request down over one late-game entry.
function knowledge.technologies()
  local force = player_force()
  local out = {}
  if not force then return out end
  for _, name in ipairs(sorted_keys(force.technologies)) do
    local tech = force.technologies[name]
    local prerequisites = sorted_keys(tech.prerequisites or {})

    local unlocks = {}
    for _, effect in ipairs(tech.prototype.effects or {}) do
      if effect.type == "unlock-recipe" then unlocks[#unlocks + 1] = effect.recipe end
    end
    table.sort(unlocks)

    local ok, count = pcall(function() return tech.research_unit_count end)
    local trigger = tech.prototype.research_trigger

    out[#out + 1] = {
      name = name,
      prerequisites = prerequisites,
      unit_count = ok and count or nil,
      unit_ingredients = stack_list(tech.research_unit_ingredients),
      unlocks = unlocks,
      trigger = trigger and trigger.type or nil,
      -- *What* satisfies the trigger, not merely that one exists. The type
      -- alone rendered as `trigger:craft-item`, which tells an agent it cannot
      -- research the thing and nothing about how to get it. Removing trigger
      -- technologies from the selectable frontier stopped the invalid calls; it
      -- did not tell anyone what action unlocks them.
      trigger_item = trigger and trigger.item and trigger.item.name or nil,
      trigger_count = trigger and trigger.count or nil,
      trigger_entity = trigger and trigger.entity or nil,
      -- Force state, like `recipes[].enabled`. See the header.
      researched = tech.researched,
    }
  end
  return out
end

--- Everything above, in one reply.
--
-- Counts are returned beside the tables so a caller can check that a truncated
-- or partially serialised answer is not silently accepted as a complete one --
-- this reply is the largest the protocol produces, and it is cached to disk and
-- then reused for the rest of a run.
function knowledge.snapshot()
  local recipes, dropped = knowledge.recipes()
  local placeable = knowledge.placeable()
  local technologies = knowledge.technologies()
  return {
    recipes = recipes,
    placeable = placeable,
    technologies = technologies,
    counts = {
      recipes = #recipes,
      placeable = #placeable,
      technologies = #technologies,
      -- Product-less placeholder recipes. See `knowledge.recipes`.
      recipes_without_products = dropped,
    },
    -- The mutable half -- `enabled`, `unlocked`, `researched` -- is the force's
    -- state at this tick and nothing else. Stated in the reply because the
    -- Python side caches this keyed on engine build and mod digest, neither of
    -- which moves when a technology completes.
    force_state_tick = game.tick,
    force = "player",
  }
end

return knowledge
