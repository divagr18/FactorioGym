-- Mod entry point: wire engine events to the runtime and register the bridge.

local runtime = require("runtime")
local bridge = require("bridge")

script.on_init(runtime.on_init)
script.on_load(runtime.on_load)
script.on_configuration_changed(runtime.on_configuration_changed)
script.on_event(defines.events.on_tick, runtime.on_tick)

-- Per-entity destruction detection for episode-scoped handles (PLAN.md 2.4).
-- Registered per observed entity via script.register_on_object_destroyed, so
-- this fires only for entities the agent has actually seen -- and it fires
-- regardless of raise_destroy, which matters because scene teardown destroys
-- in bulk.
script.on_event(defines.events.on_object_destroyed, runtime.on_object_destroyed)

-- By-hand tallies (roadmap A4.2). The engine's production statistics cannot
-- separate a plate a furnace made from a plate a character hand-crafted, so
-- what the character does by hand is counted here and subtracted there.
--
-- These fire only for an entity the engine considers a *player*. Whether the
-- controlled character is one is a property of this mod's setup, not something
-- to assume, so `world.tallies` also carries counters written by the mod's own
-- action handlers and `world.truth` reports which of the two actually moved.
local world = require("world")
script.on_event(defines.events.on_player_crafted_item, world.on_player_crafted_item)
script.on_event(defines.events.on_player_mined_item, world.on_player_mined_item)

-- A human who joins to watch must not become part of the world.
--
-- Every worker is a dedicated server (`worker.py` runs the graphical binary
-- with `--start-server`), so a second Factorio can join and render the world
-- the agent is driving -- which is what `tools/watch_agent.py` is for. But a
-- joining player is not free: the engine creates a `LuaPlayer`, gives it a
-- character, and starts the freeplay intro cutscene. Measured on the first
-- watched run: the join stalled the server, the next `step` raised an
-- infrastructure failure, and the episode was truncated at decision 3 and
-- excluded from metrics. The player was left on "press TAB to skip the
-- cutscene".
--
-- So a joiner is made a spectator at once: no character, no cutscene, nothing
-- it can walk into or build. The agent's body is a standalone character
-- entity (`world.ensure_character`) and is unaffected -- 2.0 removed
-- `game.create_player`, so the two were never the same thing.
--
-- This changes nothing for a measured run, which has no players at all and
-- never fires this event.
local function make_spectator(event)
  local player = game.get_player(event.player_index)
  if not player then return end
  -- `exit_cutscene` first: setting a controller mid-cutscene leaves the camera
  -- attached to a character that is about to stop existing.
  if player.controller_type == defines.controllers.cutscene then
    pcall(function() player.exit_cutscene() end)
  end
  if player.controller_type == defines.controllers.spectator then return end
  local character = player.character
  pcall(function()
    player.set_controller({ type = defines.controllers.spectator })
  end)
  -- The character the engine handed the player is destroyed rather than left
  -- standing: `set_controller` detaches it, and an abandoned body is an entity
  -- in the scene the agent can walk into and the digest can see.
  --
  -- Never the agent's own body, though. `world.ensure_character` keeps it in
  -- `storage.frrl_character`, and 2.0 removed `game.create_player` precisely
  -- because the agent's character is a standalone entity rather than a
  -- player's -- but a scenario script that assigned it to a joining player
  -- would make this handler delete the thing the whole run is driving. The
  -- identity check costs nothing and the failure it prevents is total.
  local agent = storage.frrl_character
  local is_agent = agent and agent.valid and character == agent
  if character and character.valid and not is_agent then
    pcall(function() character.destroy() end)
  end
end

-- Both events, because one is not enough. Registering only
-- `on_player_created` left the watcher with a character anyway: the save is
-- freeplay-derived, and the freeplay scenario's own handler creates the
-- character and starts the intro cutscene, so whichever of the two runs last
-- wins. `on_player_joined_game` fires after all of that has settled, and the
-- function is idempotent -- it returns early if the player is already a
-- spectator.
script.on_event(defines.events.on_player_created, make_spectator)
script.on_event(defines.events.on_player_joined_game, make_spectator)

bridge.register(runtime)
