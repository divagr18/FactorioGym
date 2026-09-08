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
