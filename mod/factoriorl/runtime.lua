-- Request dispatch, episode bookkeeping, dedup ledger, exact stepping.
--
-- Protocol v1 invariants:
--   * dispatch table is closed: unknown types/actions are rejected, never executed.
--   * mutating requests are recorded per episode under their request_id --
--     rejections included -- and resends return the stored response without
--     re-applying (code "duplicate").
--   * requests naming a foreign episode are rejected as stale.
--   * advance unpauses, runs exactly N ticks, re-pauses; the stored response
--     settles to completed in on_tick. Read-only request_status resolves it.

local actions = require("actions")
local observations = require("observations")
local world = require("world")

local PROTOCOL_VERSION = 1
local LEDGER_LIMIT = 4096

local runtime = {}
local state

local function new_state()
  return {
    episode_counter = 0,
    episode_id = nil,
    episode_start_tick = 0,
    -- request_id -> { request_type, response }
    ledger = {},
    ledger_order = {},
    advance_until = nil,
    pending_advance_request_id = nil,
    pending_advance_start_tick = nil,
    requests_handled = 0,
    duplicates_rejected = 0,
    stale_rejected = 0,
  }
end

-- ---------------------------------------------------------------- episodes

local function begin_episode()
  state.episode_counter = state.episode_counter + 1
  state.episode_id = string.format("ep-%d", state.episode_counter)
  state.episode_start_tick = game.tick
  state.ledger = {}
  state.ledger_order = {}
  state.advance_until = nil
  state.pending_advance_request_id = nil
  state.pending_advance_start_tick = nil
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

-- Resolution states for a recorded request. "rejected" means the worker saw
-- the request and refused it: no mutation applied, and no point retrying the
-- same id. Distinguishing it from "unknown" is what lets a client resolve a
-- transport timeout without guessing (PLAN.md 1.2, sec. 2 error handling).
local function ledger_state(stored)
  if stored.code ~= "ok" then return "rejected" end
  if stored.result ~= nil and stored.result.status == "running" then return "running" end
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

-- ---------------------------------------------------------------- responses

local function err(code, message, details)
  return { code = code, message = message, details = details }
end

local function respond(request, code, result, error_body)
  local body = {
    protocol = PROTOCOL_VERSION,
    request_id = request.request_id or "",
    episode_id = state.episode_id or "",
    code = code,
    tick = episode_tick(),
  }
  if result then body.result = result end
  if error_body then body.error = error_body end
  return body
end

-- ---------------------------------------------------------------- advance

local function handle_advance(request)
  local ticks = request.payload and request.payload.ticks
  if ticks == nil then
    return respond(request, "rejected", nil, err("missing_field", "payload.ticks is required"))
  end
  if type(ticks) ~= "number" or ticks ~= math.floor(ticks) or ticks < 1 then
    return respond(request, "rejected", nil,
      err("bad_type", "payload.ticks must be a positive integer", { got = tostring(ticks) }))
  end
  if state.advance_until then
    return respond(request, "rejected", nil,
      err("precondition", "an advance is already running until tick " .. state.advance_until))
  end
  state.advance_until = game.tick + ticks
  state.pending_advance_request_id = request.request_id
  state.pending_advance_start_tick = game.tick
  game.tick_paused = false
  return respond(request, "ok", { status = "running", target_ticks = ticks })
end

local function settle_advance()
  local request_id = state.pending_advance_request_id
  local start_tick = state.pending_advance_start_tick
  local entry = request_id and state.ledger[request_id]
  state.advance_until = nil
  state.pending_advance_request_id = nil
  state.pending_advance_start_tick = nil
  game.tick_paused = true
  if entry and start_tick then
    entry.response.tick = episode_tick()
    entry.response.result = {
      status = "completed",
      ticks_advanced = game.tick - start_tick,
    }
  end
end

-- ---------------------------------------------------------------- handlers

local function handle_request_status(request)
  local target = request.payload and request.payload.request_id
  if not target then
    return respond(request, "rejected", nil, err("missing_field", "payload.request_id is required"))
  end
  local entry = state.ledger[target]
  if not entry then
    -- Explicit resolution of uncertain transport outcomes: never seen here,
    -- so nothing was applied and the request is safe to send.
    return respond(request, "ok", {
      stored = nil,
      observed = false,
      applied = false,
      settled = false,
      state = "unknown",
    })
  end
  local stored = entry.response
  local resolution = ledger_state(stored)
  return respond(request, "ok", {
    stored = stored,
    observed = true,
    applied = resolution ~= "rejected",
    settled = resolution ~= "running",
    state = resolution,
  })
end

local function handle_status(request)
  return respond(request, "ok", {
    status = "ready",
    absolute_tick = game.tick,
    episode_id = state.episode_id,
    tick_paused = game.tick_paused,
    advancing = state.advance_until ~= nil,
    requests_handled = state.requests_handled,
    duplicates_rejected = state.duplicates_rejected,
    stale_rejected = state.stale_rejected,
  })
end

local function handle_observe(request)
  return respond(request, "ok", observations.snapshot(state))
end

local function handle_reset(request)
  world.reset_scene()
  actions.reset_state()
  local new_episode = begin_episode()
  return respond(request, "ok", { episode_id = new_episode })
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
}

local MUTATING = { advance = true, act = true, reset = true }

function runtime.handle_json(json_string)
  local ok, request = pcall(helpers.json_to_table, json_string)
  if not ok or type(request) ~= "table" then
    return {
      protocol = PROTOCOL_VERSION,
      request_id = "",
      episode_id = state and state.episode_id or "",
      code = "error",
      error = { code = "bad_protocol", message = "unparseable request: " .. tostring(request) },
    }
  end
  state.requests_handled = state.requests_handled + 1

  if request.protocol ~= PROTOCOL_VERSION then
    return respond(request, "unsupported", nil,
      err("bad_protocol", "protocol version must be " .. PROTOCOL_VERSION,
        { got = tostring(request.protocol) }))
  end

  local handler = HANDLERS[request.type]
  if not handler then
    return respond(request, "unsupported", nil,
      err("unknown_request", "unknown request type: " .. tostring(request.type)))
  end

  if MUTATING[request.type] then
    if request.episode_id ~= state.episode_id then
      state.stale_rejected = state.stale_rejected + 1
      return respond(request, "stale_episode", nil,
        err("bad_episode", "request targets episode " .. tostring(request.episode_id)
          .. " but the current episode is " .. tostring(state.episode_id)))
    end
    if not request.request_id or request.request_id == "" then
      return respond(request, "rejected", nil, err("missing_field", "request_id is required"))
    end
    local prior = state.ledger[request.request_id]
    if prior then
      state.duplicates_rejected = state.duplicates_rejected + 1
      local stored = prior.response
      return {
        protocol = PROTOCOL_VERSION,
        request_id = request.request_id,
        episode_id = state.episode_id,
        code = "duplicate",
        tick = episode_tick(),
        result = stored.result,
        error = stored.error,
      }
    end
  end

  local response = handler(request)
  -- Record every outcome of a mutating request, not only successes: a resend
  -- after a transport timeout must return the stored rejection rather than
  -- executing the action a second time.
  if MUTATING[request.type] and request.request_id and request.request_id ~= "" then
    ledger_store(request.request_id, request.type, response)
  end
  return response
end

-- ---------------------------------------------------------------- lifecycle

function runtime.on_init()
  state = new_state()
  storage.frrl_state = state
  world.ensure_character()
  world.build_reference_scene()
  actions.on_init()
  begin_episode()
  -- World waits for explicit advances between decisions.
  game.tick_paused = true
end

function runtime.on_load()
  state = storage.frrl_state
end

function runtime.on_tick(_)
  if state.advance_until and game.tick >= state.advance_until then
    settle_advance()
  end
  actions.on_tick(state)
end

return runtime
