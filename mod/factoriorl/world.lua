-- Reference scene: deterministic world layout for the Phase 0 proofs.
--
-- Layout (fixed):
--   character spawn      (0, 0)
--   "src" wooden-chest   (3, 0)   stocked with 50 iron plates
--   "dst" wooden-chest   (-3, 0)  empty
--   stone walls          y = -6, x in [-3, 3]   (blocks walking north)
--
-- Rebuilt identically on every reset; no randomness.

local world = {}

local SCENE_RADIUS = 16
local SURFACE_NAME = "nauvis"
local START_IRON_PLATES = 50

local function surface()
  return game.surfaces[SURFACE_NAME]
end

local function clear_scene(srf)
  local area = {
    { -SCENE_RADIUS, -SCENE_RADIUS },
    { SCENE_RADIUS, SCENE_RADIUS },
  }
  for _, entity in pairs(srf.find_entities(area)) do
    if entity.valid and entity.type ~= "character" then
      entity.destroy({ raise_destroy = false })
    end
  end
end

local function build_entities(srf)
  srf.create_entity({ name = "wooden-chest", position = { 3, 0 }, force = "player" })
  srf.create_entity({ name = "wooden-chest", position = { -3, 0 }, force = "player" })
  for x = -3, 3 do
    srf.create_entity({ name = "stone-wall", position = { x, -6 }, force = "player" })
  end
end

local function stock_scene_entities(srf)
  local src = srf.find_entities_filtered({
    area = { { 2.5, -0.5 }, { 3.5, 0.5 } },
    name = "wooden-chest",
  })[1]
  local dst = srf.find_entities_filtered({
    area = { { -3.5, -0.5 }, { -2.5, 0.5 } },
    name = "wooden-chest",
  })[1]
  if src then src.insert({ name = "iron-plate", count = START_IRON_PLATES }) end
  return src, dst
end

function world.build_reference_scene()
  local srf = surface()
  clear_scene(srf)
  build_entities(srf)
  local src, dst = stock_scene_entities(srf)
  storage.frrl_scene = { src = src, dst = dst }
end

function world.reset_scene()
  world.build_reference_scene()
end

function world.scene()
  return storage.frrl_scene
end

-- Factorio 2.0 removed game.create_player; zero-player headless servers have
-- no LuaPlayer at all. The agent body is a standalone "character" entity held
-- in storage (walking_state and character_main inventory work without a
-- player). Evaluator-only setup provisions it; the typed protocol only moves it.
function world.ensure_character()
  local existing = storage.frrl_character
  if existing and existing.valid then
    return existing
  end
  local ch = surface().create_entity({
    name = "character",
    position = { 0, 0 },
    force = "player",
  })
  storage.frrl_character = ch
  return ch
end

return world
