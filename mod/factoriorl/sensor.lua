-- The local sensor region (PLAN.md 2.3).
--
-- Three engine queries per observation, never per-entity lookups. Measured in
-- the Phase 2.0 spike at reference-scene scale: a radius-32 entity query costs
-- 0.011 ms and a full 65x65 water-tile query 0.018 ms, against a 16.7 ms round
-- trip. Re-measure at Phase 7 factory scale, where entity counts are two orders
-- of magnitude larger.
--
-- Terrain is an obstacle-only query. `find_tiles_filtered` with a collision
-- mask returns just the blocking tiles -- zero of them on a field of grass --
-- so terrain must never be built by iterating `get_tile` over 4225 tiles.

local handles = require("handles")

local sensor = {}

--- Entity types worth reporting. Keeping this closed bounds the payload and
--- keeps decorative entities out of the policy's input.
local ENTITY_TYPES = {
  "container", "logistic-container", "furnace", "assembling-machine",
  "mining-drill", "transport-belt", "underground-belt", "splitter",
  "inserter", "electric-pole", "boiler", "generator", "offshore-pump",
  "pipe", "pipe-to-ground", "lab", "lamp", "wall", "solar-panel",
  "accumulator", "roboport", "radar", "storage-tank", "pump", "character",
}

local INVENTORY_BY_TYPE = {
  ["container"] = defines.inventory.chest,
  ["logistic-container"] = defines.inventory.chest,
  ["furnace"] = defines.inventory.furnace_source,
  ["assembling-machine"] = defines.inventory.assembling_machine_input,
}

-- The slots a *stopped* machine's cause lives in. Without these, "the drill
-- has no fuel" and "the furnace has no ore" were the same absent field, so no
-- agent could tell which to fix. R2.3 asks for legitimately observable state,
-- and these are what a player reads off the machine.
local FUEL_BY_TYPE = {
  ["furnace"] = defines.inventory.fuel,
  ["mining-drill"] = defines.inventory.fuel,
  ["boiler"] = defines.inventory.fuel,
  ["inserter"] = defines.inventory.fuel,
}

local OUTPUT_BY_TYPE = {
  ["furnace"] = defines.inventory.furnace_result,
  ["assembling-machine"] = defines.inventory.assembling_machine_output,
}

--- Readable name for an entity status code.
--
-- The wire carried a raw `defines.entity_status` integer, which the encoder
-- collapsed to one bit and the language-model prompt rendered as "status 3".
-- Neither client could act on it.
local STATUS_NAMES = nil
local function status_name(code)
  if STATUS_NAMES == nil then
    STATUS_NAMES = {}
    for name, value in pairs(defines.entity_status) do
      STATUS_NAMES[value] = name
    end
  end
  return STATUS_NAMES[code]
end

local function inventory_contents(entity, which)
  if not which then return nil end
  local inv = entity.get_inventory(which)
  if not inv then return nil end
  local out, any = {}, false
  for _, stack in pairs(inv.get_contents()) do
    out[stack.name] = (out[stack.name] or 0) + stack.count
    any = true
  end
  if not any then return nil end
  return out
end

local function contents_of(entity)
  return inventory_contents(entity, INVENTORY_BY_TYPE[entity.type])
end

--- Compact record for one visible entity. Short keys only where they repeat
--- per entity; scalars keep readable names.
function sensor.entity_record(entity)
  local record = {
    h = handles.mint(entity),
    name = entity.name,
    type = entity.type,
    p = { entity.position.x, entity.position.y },
  }
  if entity.supports_direction then record.d = entity.direction end
  local contents = contents_of(entity)
  if contents then record.contents = contents end
  if entity.type == "assembling-machine" then
    local ok, recipe = pcall(function() return entity.get_recipe() end)
    if ok and recipe then record.recipe = recipe.name end
  end
  local ok_status, status = pcall(function() return entity.status end)
  -- `working` was omitted to keep the payload down, which made "working" and
  -- "this type has no status" indistinguishable. Both are now stated.
  if ok_status and status then
    record.status = status
    record.st = status_name(status)
    record.working = status == defines.entity_status.working
  end
  local fuel = inventory_contents(entity, FUEL_BY_TYPE[entity.type])
  if fuel then record.fuel = fuel end
  local output = inventory_contents(entity, OUTPUT_BY_TYPE[entity.type])
  if output then record.output = output end
  -- 2.0 moved this: max_health is on the entity, and get_health_ratio does
  -- the division. Only damaged entities carry the field, to keep the payload
  -- down.
  if entity.is_entity_with_health then
    local fraction = entity.get_health_ratio()
    if fraction and fraction < 1 then record.health = fraction end
  end
  return record
end

--- A tile-by-tile picture of the immediate surroundings.
--
-- The observation could already say "stone-furnace 26.6 tiles northwest" and
-- "coal, nearest 0.5 tiles". What it could not say is what occupies which
-- tile -- so an agent could not tell that its drill and its furnace were five
-- tiles apart rather than touching, and a burner drill drops its ore on one
-- adjacent tile. Measured on the first paid run: the agent built both, never
-- connected them, and spent the rest of the run looking for fuel for a loop
-- that could not have run anyway.
--
-- This is observation, not instruction. It describes the world the way a
-- player sees it on screen; it prescribes no placement and no ordering.
--
-- Bounded on purpose. One `find_entities_filtered` over the box, one
-- `get_tile` per cell: at radius 8 that is 289 cells, and it runs once per
-- decision rather than once per tick.
function sensor.grid(surface, origin, radius)
  local cx = math.floor(origin.x)
  local cy = math.floor(origin.y)
  local cells = {}
  local legend = {}
  for row = 0, 2 * radius do
    local line = {}
    for column = 0, 2 * radius do
      line[column + 1] = "."
      local tile = surface.get_tile(cx - radius + column, cy - radius + row)
      if tile and tile.valid then
        local name = tile.name or ""
        if string.find(name, "water", 1, true) then line[column + 1] = "~" end
      end
    end
    cells[row + 1] = line
  end

  local function put(x, y, glyph)
    local column = math.floor(x) - (cx - radius) + 1
    local row = math.floor(y) - (cy - radius) + 1
    if row >= 1 and row <= 2 * radius + 1 and column >= 1 and column <= 2 * radius + 1 then
      cells[row][column] = glyph
    end
  end

  -- Resources first, so a machine standing on ore hides the ore rather than
  -- the other way round: what the agent needs to know is what it would
  -- collide with.
  local RESOURCE_GLYPH = {
    ["iron-ore"] = "i", ["copper-ore"] = "c", ["coal"] = "k",
    ["stone"] = "s", ["uranium-ore"] = "u", ["crude-oil"] = "o",
  }
  local seen_resources = {}
  for _, entity in pairs(surface.find_entities_filtered({
    area = { { cx - radius, cy - radius }, { cx + radius + 1, cy + radius + 1 } },
    type = "resource",
  })) do
    if entity.valid then
      local glyph = RESOURCE_GLYPH[entity.name] or "?"
      put(entity.position.x, entity.position.y, glyph)
      if not seen_resources[entity.name] then
        seen_resources[entity.name] = true
        legend[#legend + 1] = { glyph = glyph, name = entity.name, kind = "resource" }
      end
    end
  end

  -- Then everything solid, over the top, marking every tile of its footprint
  -- so a 2x2 machine reads as 2x2.
  -- Deliberately excludes any letter whose lowercase is already taken by a
  -- resource or by scenery -- i k c s u o t r x. That frees *case* to carry
  -- meaning: an uppercase machine is running, a lowercase one is stopped, and
  -- a stopped machine is otherwise indistinguishable from a working one on a
  -- map made of single characters. The run before this built a drill, an
  -- inserter and a furnace, fuelled none of them, and read the map as a
  -- finished factory.
  local debris = {}
  local GLYPHS = "ABDEFGHJLMNPQVWYZ"

  -- Statuses that mean "this is built and doing nothing".
  local STOPPED = {
    [defines.entity_status.no_fuel] = true,
    [defines.entity_status.no_power] = true,
    [defines.entity_status.no_ingredients] = true,
    [defines.entity_status.no_minable_resources] = true,
    [defines.entity_status.waiting_for_source_items] = true,
    [defines.entity_status.item_ingredient_shortage] = true,
    [defines.entity_status.missing_required_fluid] = true,
  }
  local next_glyph = 1
  for _, entity in pairs(surface.find_entities_filtered({
    area = { { cx - radius, cy - radius }, { cx + radius + 1, cy + radius + 1 } },
  })) do
    if entity.valid and entity.type ~= "resource" and entity ~= storage.frrl_character then
      local glyph
      if entity.type == "tree" then
        glyph = "t"
      elseif entity.type == "simple-entity" then
        glyph = "r"
      elseif string.sub(entity.name, 1, 11) == "crash-site-" then
        -- The spawn wreckage is nine entities spread over a dozen tiles. Giving
        -- each its own letter consumed most of the alphabet and pushed the
        -- machines the agent actually built down to glyphs it had to hunt for.
        -- One glyph, one legend line, and the handles of the pieces that hold
        -- something worth taking.
        glyph = "x"
        debris[#debris + 1] = {
          handle = handles.mint(entity),
          name = entity.name,
          contents = contents_of(entity),
        }
      elseif next_glyph <= #GLYPHS then
        glyph = string.sub(GLYPHS, next_glyph, next_glyph)
        next_glyph = next_glyph + 1
        local entry = {
          glyph = glyph,
          name = entity.name,
          handle = handles.mint(entity),
          direction = entity.supports_direction and entity.direction or nil,
          position = { entity.position.x, entity.position.y },
        }
        -- Where this machine puts what it makes, and where it takes from.
        --
        -- "facing south" is a word, not a location: it does not tell an agent
        -- which tile a burner drill's ore lands on, and a drill whose output
        -- tile is not the furnace produces nothing however well fuelled it is.
        -- The engine renders exactly this as an arrow on the entity, so any
        -- player watching sees it -- it is world state, not the reference
        -- solver's measured geometry, which is what `FORBIDDEN_IN_PROMPT`
        -- withholds from the *static* instructions of a benchmark task.
        local ok_drop, drop = pcall(function() return entity.drop_position end)
        if ok_drop and drop then entry.drop = { drop.x, drop.y } end
        local ok_pick, pick = pcall(function() return entity.pickup_position end)
        if ok_pick and pick then entry.pickup = { pick.x, pick.y } end
        -- Fuel and status, on the machine itself. Both were in the ENTITIES
        -- block and neither was on the map, so a machine could sit there
        -- looking built while being empty.
        local ok_status, status = pcall(function() return entity.status end)
        if ok_status and status then entry.status = status end
        local ok_fuel, fuel = pcall(function()
          local inv = entity.get_fuel_inventory()
          if not inv then return nil end
          local total = 0
          for _, stack in pairs(inv.get_contents()) do total = total + stack.count end
          return total
        end)
        if ok_fuel and fuel ~= nil then entry.fuel = fuel end
        -- Lowercase means stopped. The letter still identifies the machine, so
        -- the legend needs no second entry and the map needs no second layer.
        if entry.status and STOPPED[entry.status] then
          glyph = string.lower(glyph)
          entry.glyph = glyph
          entry.stopped = true
        end
        legend[#legend + 1] = entry
      else
        glyph = "+"
      end
      local box = entity.bounding_box
      for x = math.floor(box.left_top.x), math.ceil(box.right_bottom.x) - 1 do
        for y = math.floor(box.left_top.y), math.ceil(box.right_bottom.y) - 1 do
          put(x, y, glyph)
        end
      end
    end
  end

  -- Output arrows last, so they sit on top of open ground but never hide a
  -- machine: a cell that already holds something solid keeps it.
  for _, entry in ipairs(legend) do
    if entry.drop then
      local column = math.floor(entry.drop[1]) - (cx - radius) + 1
      local row = math.floor(entry.drop[2]) - (cy - radius) + 1
      if row >= 1 and row <= 2 * radius + 1 and column >= 1 and column <= 2 * radius + 1 then
        local cell = cells[row][column]
        if cell == "." or string.match(cell, "%l") then
          local dx = entry.drop[1] - entry.position[1]
          local dy = entry.drop[2] - entry.position[2]
          local arrow = "v"
          if math.abs(dx) > math.abs(dy) then
            arrow = dx > 0 and ">" or "<"
          else
            arrow = dy > 0 and "v" or "^"
          end
          cells[row][column] = arrow
        end
      end
    end
  end

  put(origin.x, origin.y, "@")

  local rows = {}
  for index, line in ipairs(cells) do rows[index] = table.concat(line) end
  return {
    radius = radius,
    -- The world coordinate of the top-left cell, so a reader can turn a cell
    -- into a position without counting from the character.
    origin = { cx - radius, cy - radius },
    rows = rows,
    legend = legend,
    debris = debris,
  }
end

--- Sweep the region around `origin`.
-- Returns entities, per-tile resources, aggregated resource patches, blocked
-- tiles, and whether any cap truncated the result.
function sensor.sweep(surface, origin, profile)
  local radius = profile.radius
  local truncated = false

  -- The engine returns swept entities in its own internal order, not by
  -- distance, so the cap must be applied to a distance-ordered list. Applied to
  -- the raw sweep it kept whichever entities the engine happened to list first
  -- and dropped near ones in favour of far ones -- which is why the cap could
  -- not simply be lowered before. `encoders.encode` keeps only the 32 nearest
  -- of `entities` + `remembered`, so an unsorted cap decided the policy's input
  -- by accident; sorting first is what makes a small cap information-preserving.
  --
  -- The candidate limit is deliberately a separate number from the record cap:
  -- sorting 49 arbitrarily chosen entities does not find the nearest 48, so a
  -- profile that lowers `entity_cap` sets `entity_sweep_limit` to keep the
  -- candidate pool the size it was. A profile that does not set one keeps the
  -- old `entity_cap + 1`, which is what `local-v1` relies on.
  local sweep_limit = profile.entity_sweep_limit or (profile.entity_cap + 1)
  local candidates = {}
  local candidate_count = 0
  for _, entity in pairs(surface.find_entities_filtered({
    position = origin,
    radius = radius,
    type = ENTITY_TYPES,
    limit = sweep_limit,
  })) do
    if entity.valid and entity ~= storage.frrl_character then
      local dx = entity.position.x - origin.x
      local dy = entity.position.y - origin.y
      candidate_count = candidate_count + 1
      -- Squared distance: it orders identically to the distance and saves a
      -- `math.sqrt` per entity on a path that runs every step.
      candidates[candidate_count] = {
        entity = entity,
        d2 = dx * dx + dy * dy,
        i = candidate_count,
      }
    end
  end
  -- `table.sort` is not stable, so equal distances fall back to sweep order.
  -- Without that tie-break two observations of an unchanged scene could order
  -- co-located entities differently, and `test_protocol_fixtures` asserts the
  -- `entities` block of two consecutive observations is equal.
  table.sort(candidates, function(a, b)
    if a.d2 ~= b.d2 then return a.d2 < b.d2 end
    return a.i < b.i
  end)

  -- Same meaning as before: the cap clipped the sweep, so the observation
  -- admits it rather than silently dropping entities.
  local kept = candidate_count
  if kept > profile.entity_cap then
    kept = profile.entity_cap
    truncated = true
  end

  local entities = {}
  for index = 1, kept do
    -- Records are built only for entities that survive the cap. `entity_record`
    -- reads an inventory, calls `get_recipe`/`status` through `pcall` and mints
    -- a handle, so building 256 records to serialise 48 of them would pay the
    -- expensive half of the cost the cap exists to remove -- and would churn
    -- the 4096-entry handle registry for entities no consumer can address.
    entities[index] = sensor.entity_record(candidates[index].entity)
  end

  -- An ore patch is one resource entity per tile, so per-tile detail is emitted
  -- only close in and everything further out is aggregated. An observation that
  -- silently dropped entities would be worse than one that admits it.
  local resource_tiles = {}
  local patches = {}
  for _, resource in pairs(surface.find_entities_filtered({
    position = origin,
    radius = radius,
    type = "resource",
    limit = profile.resource_cap + 1,
  })) do
    if resource.valid then
      local dx = resource.position.x - origin.x
      local dy = resource.position.y - origin.y
      local d = math.sqrt(dx * dx + dy * dy)
      local patch = patches[resource.name]
      if not patch then
        patch = {
          name = resource.name, count = 0, total = 0,
          nearest = { resource.position.x, resource.position.y }, nearest_d = d,
        }
        patches[resource.name] = patch
      end
      patch.count = patch.count + 1
      patch.total = patch.total + (resource.amount or 0)
      if d < patch.nearest_d then
        patch.nearest_d = d
        patch.nearest = { resource.position.x, resource.position.y }
      end
      if d <= profile.resource_detail_radius then
        if #resource_tiles >= profile.resource_cap then
          truncated = true
        else
          resource_tiles[#resource_tiles + 1] = {
            h = handles.mint(resource),
            name = resource.name,
            p = { resource.position.x, resource.position.y },
            amount = resource.amount,
          }
        end
      end
    end
  end

  local blocked = {}
  for _, tile in pairs(surface.find_tiles_filtered({
    area = {
      { origin.x - radius, origin.y - radius },
      { origin.x + radius, origin.y + radius },
    },
    collision_mask = "water_tile",
  })) do
    blocked[#blocked + 1] = { tile.position.x, tile.position.y }
  end

  local patch_list = {}
  for _, patch in pairs(patches) do
    patch_list[#patch_list + 1] = {
      name = patch.name, count = patch.count,
      total = patch.total, nearest = patch.nearest,
    }
  end

  return {
    entities = entities,
    resource_tiles = resource_tiles,
    patches = patch_list,
    blocked = blocked,
    truncated = truncated,
  }
end

return sensor
