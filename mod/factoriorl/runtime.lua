-- Request dispatch, episode bookkeeping, dedup ledger, exact stepping.
--
-- Protocol v2 invariants:
--   * the dispatch table is closed: unknown types/actions are rejected, never
--     executed;
--   * every handler runs inside a pcall, so a Lua error becomes a structured
--     `error` response recorded in the ledger rather than an engine failure;
--   * mutating requests are recorded per episode under their request_id --
--     rejections included -- and resends return the stored response without
--     re-applying (code "duplicate");
--   * requests naming a foreign episode are rejected as stale;
--   * ongoing operations live in one in-flight registry and settle exactly
--     once, through ledger_settle;
--   * `step` fuses act + advance into one request, and `collect` retrieves the
--     settled observation. Two round trips is the floor: RCON commands execute
--     inside a tick, so Lua cannot block while the world advances.

local protocol = require("protocol")
local matrix = require("matrix")
local actions = require("actions")
local observations = require("observations")
local world = require("world")
local handles = require("handles")
local memory = require("memory")
local inflight = require("inflight")
local profiles = require("profiles")

local CODE, STATUS, ERR = protocol.CODE, protocol.STATUS, protocol.ERR

local LEDGER_LIMIT = 4096
local EVENT_LIMIT = 256
local MAX_ADVANCE_TICKS = 36000

local runtime = {}
local state

local function new_state()
  return {
    episode_counter = 0,
    episode_id = nil,
    episode_start_tick = 0,
    ledger = {},
    ledger_order = {},
    events = {},
    event_seq = 0,
    observation_profile = profiles.DEFAULT_OBSERVATION,
    action_profile = profiles.DEFAULT_ACTION,
    requests_handled = 0,
    duplicates_rejected = 0,
    stale_rejected = 0,
    errors = 0,
  }
end

-- ---------------------------------------------------------------- episodes

local function begin_episode()
  state.episode_counter = state.episode_counter + 1
  state.episode_id = string.format("ep-%d", state.episode_counter)
  state.episode_start_tick = game.tick
  state.ledger = {}
  state.ledger_order = {}
  state.events = {}
  state.event_seq = 0
  handles.reset()
  memory.reset()
  inflight.reset()
  -- Re-pause explicitly. Clearing the bookkeeping is not enough: if a reset
  -- lands while an advance is still running, the world would keep ticking with
  -- nothing left to stop it, and the next observation would report a tick the
  -- caller never asked for. PLAN.md section 2: the world pauses between
  -- decisions.
  game.tick_paused = true
  return state.episode_id
end

local function episode_tick()
  return game.tick - state.episode_start_tick
end

-- ---------------------------------------------------------------- ledger

local function ledger_state(stored)
  if stored.code ~= CODE.OK then return "rejected" end
  local status = stored.result and stored.result.status
  if status == STATUS.RUNNING then return "running" end
  if status == STATUS.FAILED then return "failed" end
  if status == STATUS.CANCELLED then return "cancelled" end
  return "completed"
end

local function ledger_store(request_id, request_type, response)
  if not state.ledger[request_id] then
    table.insert(state.ledger_order, request_id)
    while #state.ledger_order > LEDGER_LIMIT do
      local oldest = table.remove(state.ledger_order, 1)
      state.ledger[oldest] = nil
    end
  end
  state.ledger[request_id] = { request_type = request_type, response = response }
end

local function record_event(request_id, action, status)
  state.event_seq = state.event_seq + 1
  state.events[#state.events + 1] = {
    seq = state.event_seq,
    request_id = request_id,
    action = action,
    status = status,
    tick = episode_tick(),
  }
  while #state.events > EVENT_LIMIT do table.remove(state.events, 1) end
end

--- The one place a terminal status is written.
-- Refusing to settle an entry that is already terminal is what makes
-- "completion is reported exactly once" an invariant rather than a convention.
local function ledger_settle(request_id, status, result, error_code, error_message)
  local entry = state.ledger[request_id]
  if not entry then return false end
  local stored = entry.response
  if stored.settled_tick then return false end
  stored.settled_tick = episode_tick()
  stored.tick = episode_tick()
  local merged = stored.result or {}
  for key, value in pairs(result or {}) do merged[key] = value end
  merged.status = status
  stored.result = merged
  if error_code then
    stored.error = { code = error_code, message = error_message or status }
  end
  record_event(request_id, merged.action, status)
  return true
end

-- ---------------------------------------------------------------- responses

local function err(code, message, details)
  return { code = code, message = message, details = details }
end

local function respond(request, code, result, error_body)
  local body = {
    protocol = protocol.VERSION,
    request_id = request.request_id or "",
    episode_id = state.episode_id or "",
    code = code,
    tick = episode_tick(),
  }
  if result then body.result = result end
  if error_body then body.error = error_body end
  return body
end

-- ---------------------------------------------------------------- handlers

local function start_advance(request, ticks)
  if inflight.occupant("advance") then
    return nil, err(ERR.BUSY, "an advance is already running")
  end
  inflight.start(request.request_id, "advance", {
    deadline_tick = game.tick + ticks,
  })
  game.tick_paused = false
  return true, nil
end

local function handle_advance(request)
  local ticks = request.payload and request.payload.ticks
  if ticks == nil then
    return respond(request, CODE.REJECTED, nil,
      err(ERR.MISSING_FIELD, "payload.ticks is required"))
  end
  if type(ticks) ~= "number" or ticks ~= math.floor(ticks)
    or ticks < 1 or ticks > MAX_ADVANCE_TICKS then
    return respond(request, CODE.REJECTED, nil,
      err(ERR.BAD_TYPE, "payload.ticks must be an integer in [1, " .. MAX_ADVANCE_TICKS .. "]",
        { got = tostring(ticks) }))
  end
  local ok, problem = start_advance(request, ticks)
  if not ok then
    return respond(request, CODE.REJECTED, nil, problem)
  end
  return respond(request, CODE.OK, { status = STATUS.RUNNING, target_ticks = ticks })
end

--- Fused act + advance (protocol v2).
-- The action is applied now; the interval runs afterwards, and the settled
-- response carries the action result, the ticks advanced, and the observation.
-- A rejected action still consumes its interval: an invalid action must cost
-- the agent its turn rather than stalling time, or a masked policy that emits
-- an illegal action gets a free move.
local function handle_step(request)
  local ticks = (request.payload and request.payload.ticks) or 30
  if type(ticks) ~= "number" or ticks ~= math.floor(ticks)
    or ticks < 1 or ticks > MAX_ADVANCE_TICKS then
    return respond(request, CODE.REJECTED, nil,
      err(ERR.BAD_TYPE, "payload.ticks must be a positive integer",
        { got = tostring(ticks) }))
  end
  local action_payload = request.payload and request.payload.action
  if type(action_payload) ~= "table" then
    return respond(request, CODE.REJECTED, nil,
      err(ERR.MISSING_FIELD, "payload.action must be an object"))
  end
  if inflight.occupant("advance") then
    return respond(request, CODE.REJECTED, nil, err(ERR.BUSY, "an advance is already running"))
  end

  -- The inner action gets its own request id so its own in-flight entry (a
  -- move, a mine) is addressable and cancellable independently of the step.
  local inner = {
    request_id = request.request_id .. ":act",
    episode_id = request.episode_id,
    payload = action_payload,
  }
  local action_response = actions.dispatch(state, inner, respond, err)
  ledger_store(inner.request_id, "act", action_response)
  local ok, problem = start_advance(request, ticks)
  if not ok then
    return respond(request, CODE.REJECTED, nil, problem)
  end
  return respond(request, CODE.OK, {
    status = STATUS.RUNNING,
    target_ticks = ticks,
    action = action_response.result or {
      status = STATUS.REJECTED,
      error = action_response.error,
    },
  })
end

local function handle_collect(request)
  local target = request.payload and request.payload.request_id
  if not target then
    return respond(request, CODE.REJECTED, nil,
      err(ERR.MISSING_FIELD, "payload.request_id is required"))
  end
  local entry = state.ledger[target]
  if not entry then
    return respond(request, CODE.OK, {
      observed = false, settled = false, state = "unknown",
    })
  end
  local stored = entry.response
  local resolution = ledger_state(stored)
  return respond(request, CODE.OK, {
    observed = true,
    settled = resolution ~= "running",
    state = resolution,
    result = stored.result,
    error = stored.error,
  })
end

local function handle_request_status(request)
  local target = request.payload and request.payload.request_id
  if not target then
    return respond(request, CODE.REJECTED, nil,
      err(ERR.MISSING_FIELD, "payload.request_id is required"))
  end
  local entry = state.ledger[target]
  if not entry then
    return respond(request, CODE.OK, {
      stored = nil, observed = false, applied = false,
      settled = false, state = "unknown",
    })
  end
  local stored = entry.response
  local resolution = ledger_state(stored)
  return respond(request, CODE.OK, {
    stored = stored,
    observed = true,
    applied = resolution ~= "rejected",
    settled = resolution ~= "running",
    state = resolution,
  })
end

local function handle_status(request)
  return respond(request, CODE.OK, {
    status = "ready",
    absolute_tick = game.tick,
    episode_id = state.episode_id,
    tick_paused = game.tick_paused,
    speed = game.speed,
    advancing = inflight.occupant("advance") ~= nil,
    profiles = profiles.metadata(state.observation_profile, state.action_profile),
    scenario = storage.frrl_scene and storage.frrl_scene.name or nil,
    requests_handled = state.requests_handled,
    duplicates_rejected = state.duplicates_rejected,
    stale_rejected = state.stale_rejected,
    errors = state.errors,
    handles = handles.count(),
    remembered = memory.count(),
  })
end

local function handle_observe(request)
  return respond(request, CODE.OK, observations.snapshot(state))
end

--- Read-only self-description: the action matrix, profiles and versions the
--- worker is actually running, so a client can pin what it validated against.
local function handle_describe(request)
  return respond(request, CODE.OK, {
    protocol = protocol.VERSION,
    mod_version = script.active_mods["factoriorl"],
    actions = matrix.describe(),
    profiles = profiles.metadata(state.observation_profile, state.action_profile),
    scenarios = world.scenarios(),
    request_types = {
      "status", "observe", "reset", "advance", "act",
      "request_status", "step", "collect", "describe", "configure",
    },
  })
end

--- Evaluator-facing knobs that do not change simulation outcomes.
local function handle_configure(request)
  local payload = request.payload or {}
  local applied = {}
  if payload.speed ~= nil then
    if type(payload.speed) ~= "number" or payload.speed <= 0 then
      return respond(request, CODE.REJECTED, nil,
        err(ERR.BAD_TYPE, "speed must be a positive number"))
    end
    -- Pacing only. Factorio is tick-based, so the simulation is identical at
    -- any speed; `factoriorl bench speed` proves it by comparing episode
    -- records field by field.
    game.speed = payload.speed
    applied.speed = game.speed
  end
  return respond(request, CODE.OK, { status = STATUS.COMPLETED, applied = applied })
end

local function handle_reset(request)
  local payload = request.payload or {}
  if payload.observation_profile then
    if not profiles.observation(payload.observation_profile) then
      return respond(request, CODE.REJECTED, nil,
        err(ERR.INVALID_TARGET, "unknown observation profile"))
    end
    state.observation_profile = payload.observation_profile
  end
  if payload.action_profile then
    local action_profile = profiles.action(payload.action_profile)
    if not action_profile or not action_profile.available then
      return respond(request, CODE.REJECTED, nil,
        err(ERR.INVALID_TARGET, "unknown or unavailable action profile"))
    end
    state.action_profile = payload.action_profile
  end
  if payload.scenario then
    storage.frrl_scenario_name = payload.scenario
  end
  local scene = world.reset_scene()
  actions.reset_state()
  local new_episode = begin_episode()
  return respond(request, CODE.OK, {
    episode_id = new_episode,
    scenario = scene.scenario,
    destroyed = scene.destroyed,
    profiles = profiles.metadata(state.observation_profile, state.action_profile),
  })
end

local HANDLERS = {
  status = handle_status,
  observe = handle_observe,
  reset = handle_reset,
  advance = handle_advance,
  act = function(request)
    return actions.dispatch(state, request, respond, err)
  end,
  request_status = handle_request_status,
  step = handle_step,
  collect = handle_collect,
  describe = handle_describe,
  configure = handle_configure,
}

local MUTATING = { advance = true, act = true, reset = true, step = true }

function runtime.handle_json(json_string)
  local ok, request = pcall(helpers.json_to_table, json_string)
  if not ok or type(request) ~= "table" then
    return {
      protocol = protocol.VERSION,
      request_id = "",
      episode_id = state and state.episode_id or "",
      code = CODE.ERROR,
      error = { code = ERR.BAD_PROTOCOL, message = "unparseable request" },
    }
  end
  state.requests_handled = state.requests_handled + 1

  if request.protocol ~= protocol.VERSION then
    return respond(request, CODE.UNSUPPORTED, nil,
      err(ERR.BAD_PROTOCOL, "protocol version must be " .. protocol.VERSION,
        { got = tostring(request.protocol) }))
  end

  local handler = HANDLERS[request.type]
  if not handler then
    return respond(request, CODE.UNSUPPORTED, nil,
      err(ERR.UNKNOWN_REQUEST, "unknown request type: " .. tostring(request.type)))
  end

  if MUTATING[request.type] then
    if request.episode_id ~= state.episode_id then
      state.stale_rejected = state.stale_rejected + 1
      return respond(request, CODE.STALE_EPISODE, nil,
        err(ERR.BAD_EPISODE, "request targets episode " .. tostring(request.episode_id)
          .. " but the current episode is " .. tostring(state.episode_id)))
    end
    if not request.request_id or request.request_id == "" then
      return respond(request, CODE.REJECTED, nil, err(ERR.MISSING_FIELD, "request_id is required"))
    end
    local prior = state.ledger[request.request_id]
    if prior then
      state.duplicates_rejected = state.duplicates_rejected + 1
      local stored = prior.response
      return {
        protocol = protocol.VERSION,
        request_id = request.request_id,
        episode_id = state.episode_id,
        code = CODE.DUPLICATE,
        tick = episode_tick(),
        result = stored.result,
        error = stored.error,
      }
    end
  end

  -- Crash boundary. Ten action handlers is enough surface that an unguarded
  -- Lua error would eventually escape to the engine instead of becoming a
  -- structured response a client can resolve through request_status.
  local handled, response = pcall(handler, request)
  if not handled then
    state.errors = state.errors + 1
    response = respond(request, CODE.ERROR, { status = STATUS.FAILED },
      err(ERR.ENGINE, "handler error: " .. tostring(response)))
  end

  if MUTATING[request.type] and request.request_id and request.request_id ~= "" then
    ledger_store(request.request_id, request.type, response)
  end
  return response
end

-- ---------------------------------------------------------------- lifecycle

function runtime.on_init()
  state = new_state()
  storage.frrl_state = state
  handles.reset()
  memory.reset()
  inflight.reset()
  world.ensure_character()
  world.build_reference_scene()
  actions.on_init()
  begin_episode()
end

function runtime.on_load()
  state = storage.frrl_state
end

function runtime.on_configuration_changed(_data)
  -- A mod version change can invalidate the shape of anything in storage, so
  -- end the episode explicitly rather than resuming with a half-migrated state.
  if not state then state = new_state() end
  storage.frrl_state = state
  begin_episode()
end

function runtime.on_object_destroyed(event)
  handles.on_object_destroyed(event.registration_number, event.useful_id)
end

function runtime.on_tick(_)
  local settled = inflight.on_tick()
  for _, item in ipairs(settled) do
    local entry = item.entry
    if entry.action == "advance" then
      -- The step's settled response carries the observation, which is what
      -- makes the fused request worth having: one collect returns everything.
      local result = { ticks_advanced = item.result and item.result.ticks_advanced or 0 }
      local stored = state.ledger[entry.request_id]
      if stored and stored.request_type == "step" then
        result.observation = observations.snapshot(state)
      end
      ledger_settle(entry.request_id, item.status, result, item.error_code)
      game.tick_paused = true
    else
      ledger_settle(entry.request_id, item.status, item.result, item.error_code)
    end
  end

  -- Superseded operations settle as cancelled, exactly once each.
  local superseded = storage.frrl_superseded
  if superseded then
    for _, item in ipairs(superseded) do
      ledger_settle(item.request_id, STATUS.CANCELLED, item.result)
    end
    storage.frrl_superseded = nil
  end
  local cancelled = storage.frrl_cancelled
  if cancelled then
    ledger_settle(cancelled.request_id, STATUS.CANCELLED, cancelled.result)
    storage.frrl_cancelled = nil
  end
end

return runtime
