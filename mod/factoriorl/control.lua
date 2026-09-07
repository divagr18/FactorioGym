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

bridge.register(runtime)
