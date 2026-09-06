-- FactorioRL runtime entry point.
-- Owns: bridge registration, request dispatch, episode bookkeeping.

local runtime = require("runtime")
local bridge = require("bridge")

script.on_init(runtime.on_init)
script.on_load(runtime.on_load)
script.on_event(defines.events.on_tick, runtime.on_tick)

bridge.register(runtime)
