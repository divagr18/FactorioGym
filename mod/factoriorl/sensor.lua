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

local function contents_of(entity)
  local which = INVENTORY_BY_TYPE[entity.type]
  if not which then return nil end
  local inv = entity.get_inventory(which)
  if not inv then return nil end
  -- One call, not a loop over 80 slots.
  local out = {}
  local any = false
  for _, stack in pairs(inv.get_contents()) do
    out[stack.name] = (out[stack.name] or 0) + stack.count
    any = true
  end
  if not any then return nil end
  return out
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
  -- `working` is the boring case; omitting it keeps the payload down.
  if ok_status and status and status ~= defines.entity_status.working then
    record.status = status
  end
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

  local entities = {}
  for _, entity in pairs(surface.find_entities_filtered({
    position = origin,
    radius = radius,
    type = ENTITY_TYPES,
    limit = profile.entity_cap + 1,
  })) do
    if entity.valid and entity ~= storage.frrl_character then
      if #entities >= profile.entity_cap then
        truncated = true
        break
      end
      entities[#entities + 1] = sensor.entity_record(entity)
    end
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
