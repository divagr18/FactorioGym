-- Observation snapshots (Phase 0 scope).
-- Full local structured observations arrive in Phase 2; this shape already
-- carries tick, episode, character state, inventory, scene entities, and task
-- counters, which is what the Phase 0 reset-repeatability proof compares.

local actions

local observations = {}

local function character_state()
  local ch = storage.frrl_character
  if not ch or not ch.valid then
    return { present = false }
  end
  return {
    present = true,
    position = { ch.position.x, ch.position.y },
    walking = ch.walking_state.walking,
    direction = ch.walking_state.direction,
  }
end

local function inventory_contents(inv)
  local out = {}
  if not inv then return out end
  for i = 1, #inv do
    local stack = inv[i]
    if stack.valid_for_read then
      out[stack.name] = (out[stack.name] or 0) + stack.count
    end
  end
  return out
end

local function chest_summary(entity)
  if not entity or not entity.valid then
    return { valid = false }
  end
  local inv = entity.get_inventory(defines.inventory.chest)
  return {
    valid = true,
    name = entity.name,
    position = { entity.position.x, entity.position.y },
    contents = inventory_contents(inv),
  }
end

function observations.snapshot(state)
  local scene = storage.frrl_scene or {}
  local ch = storage.frrl_character
  local char_inv = ch and ch.valid and ch.get_inventory(defines.inventory.character_main)
  local task = storage.frrl_task or { transfers = 0, items_moved = 0 }
  return {
    episode_id = state.episode_id,
    tick = game.tick - state.episode_start_tick,
    absolute_tick = game.tick,
    character = character_state(),
    inventory = inventory_contents(char_inv),
    entities = {
      src = chest_summary(scene.src),
      dst = chest_summary(scene.dst),
    },
    task = { transfers = task.transfers, items_moved = task.items_moved },
  }
end

return observations
