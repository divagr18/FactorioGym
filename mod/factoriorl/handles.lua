-- Episode-scoped entity identity (DESIGN.md 2.4).
--
-- Handles are short opaque strings ("h1", "h2", ...) because they appear once
-- per entity in every observation, so their length is a payload cost.
--
-- Two descriptor forms, because the Phase 2.0 spike confirmed resources carry
-- no unit_number:
--
--   unit  keyed on unit_number, which the engine documents as unique for the
--         lifetime of a save and never reused. That is what makes "rebuilt
--         entities do not inherit stale identity" true by construction rather
--         than by a check: a rebuilt chest at the same tile gets a fresh
--         unit_number and therefore a fresh handle.
--   tile  keyed on (surface, tx, ty) plus a generation counter, for resources,
--         trees and rocks. Re-resolved by position; the generation bumps when
--         the tile empties, so a stale tile handle fails rather than silently
--         addressing whatever replaced it.
--
-- Destruction is detected with script.register_on_object_destroyed per observed
-- unit-form entity: O(1), and it fires regardless of raise_destroy, which
-- matters because world.lua destroys with raise_destroy = false. Registering
-- on_entity_died/on_built_entity instead would fire for every entity in the
-- world and cost per-tick throughput at Phase 7 factory scale.

local handles = {}

--- Bounded so mining a large patch (one resource entity per tile) cannot grow
--- the registry without limit.
local HANDLE_LIMIT = 4096

local function state()
  return storage.frrl_handles
end

function handles.reset()
  storage.frrl_handles = {
    next_id = 1,
    by_handle = {},
    by_unit = {},
    by_tile = {},
    order = {},
    generation = (storage.frrl_handles and storage.frrl_handles.generation or 0) + 1,
  }
end

local function tile_key(surface_index, tx, ty)
  return surface_index .. ":" .. tx .. ":" .. ty
end

local function evict_if_needed(s)
  while #s.order > HANDLE_LIMIT do
    local oldest = table.remove(s.order, 1)
    local entry = s.by_handle[oldest]
    if entry then
      if entry.kind == "unit" and entry.unit_number then
        s.by_unit[entry.unit_number] = nil
      elseif entry.kind == "tile" then
        s.by_tile[tile_key(entry.surface_index, entry.tx, entry.ty)] = nil
      end
      s.by_handle[oldest] = nil
    end
  end
end

--- Return the existing handle for `entity`, or mint one.
function handles.mint(entity)
  if not entity or not entity.valid then return nil end
  local s = state()
  if entity.unit_number then
    local existing = s.by_unit[entity.unit_number]
    if existing then return existing end
  else
    local tx, ty = math.floor(entity.position.x), math.floor(entity.position.y)
    local key = tile_key(entity.surface.index, tx, ty)
    local existing = s.by_tile[key]
    if existing then return existing end
  end

  local handle = "h" .. s.next_id
  s.next_id = s.next_id + 1
  local entry
  if entity.unit_number then
    entry = {
      kind = "unit",
      unit_number = entity.unit_number,
      entity = entity,
      name = entity.name,
      first_seen = game.tick,
    }
    s.by_unit[entity.unit_number] = handle
    -- Cheap, per-entity destruction detection.
    entry.registration = script.register_on_object_destroyed(entity)
  else
    local tx, ty = math.floor(entity.position.x), math.floor(entity.position.y)
    entry = {
      kind = "tile",
      surface_index = entity.surface.index,
      tx = tx,
      ty = ty,
      type = entity.type,
      name = entity.name,
      first_seen = game.tick,
    }
    s.by_tile[tile_key(entity.surface.index, tx, ty)] = handle
  end
  s.by_handle[handle] = entry
  s.order[#s.order + 1] = handle
  evict_if_needed(s)
  return handle
end

--- Resolve a handle to a live entity.
-- Returns (entity, nil) or (nil, reason) where reason distinguishes
-- "never minted in this episode" from "minted, now gone" -- different failures
-- for an agent, and the reset boundary makes every prior handle the former.
function handles.resolve(handle)
  if type(handle) ~= "string" then return nil, "unknown_handle" end
  local s = state()
  local entry = s.by_handle[handle]
  if not entry then return nil, "unknown_handle" end
  if entry.destroyed_tick then return nil, "target_missing" end

  if entry.kind == "unit" then
    if entry.entity and entry.entity.valid then return entry.entity, nil end
    entry.destroyed_tick = game.tick
    s.by_unit[entry.unit_number] = nil
    return nil, "target_missing"
  end

  local surface = game.surfaces[entry.surface_index]
  if not surface then return nil, "target_missing" end
  local found = surface.find_entities_filtered({
    position = { entry.tx + 0.5, entry.ty + 0.5 },
    radius = 0.5,
    type = entry.type,
    limit = 1,
  })[1]
  if not found or not found.valid or found.name ~= entry.name then
    entry.destroyed_tick = game.tick
    s.by_tile[tile_key(entry.surface_index, entry.tx, entry.ty)] = nil
    return nil, "target_missing"
  end
  return found, nil
end

--- Mark a handle destroyed from an on_object_destroyed event.
function handles.on_object_destroyed(registration_number, useful_id)
  local s = state()
  if not s then return end
  local handle = useful_id and s.by_unit[useful_id]
  if not handle then return end
  local entry = s.by_handle[handle]
  if entry then
    entry.destroyed_tick = game.tick
    entry.registration = entry.registration or registration_number
  end
  s.by_unit[useful_id] = nil
end

function handles.count()
  local s = state()
  local n = 0
  for _ in pairs(s.by_handle) do n = n + 1 end
  return n
end

--- The registry in mint order, for a simulator to load in sync mode.
-- A unit is named by prototype and fixed-point position rather than by
-- `unit_number`, which is the engine's own counter and means nothing elsewhere.
function handles.export()
  local s = state()
  local order = {}
  for _, handle in ipairs(s.order) do
    local entry = s.by_handle[handle]
    if entry then
      local record = {
        handle = handle,
        kind = entry.kind,
        name = entry.name,
        first_seen = entry.first_seen,
        destroyed_tick = entry.destroyed_tick,
      }
      if entry.kind == "unit" then
        if entry.entity and entry.entity.valid then
          record.position = {
            math.floor(entry.entity.position.x * 256 + 0.5),
            math.floor(entry.entity.position.y * 256 + 0.5),
          }
        end
      else
        record.tile = { entry.tx, entry.ty }
      end
      order[#order + 1] = record
    end
  end
  return { next_id = s.next_id, order = order }
end

function handles.generation()
  return state().generation
end

return handles
