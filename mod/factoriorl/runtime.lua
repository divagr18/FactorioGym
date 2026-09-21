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
local knowledge = require("knowledge")

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
    flat_events = {},
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
  state.flat_events = {}
  state.event_seq = 0
  handles.reset()
  memory.reset()
  inflight.reset()
  -- Re-pause explicitly. Clearing the bookkeeping is not enough: if a reset
  -- lands while an advance is still running, the world would keep ticking with
  -- nothing left to stop it, and the next observation would report a tick the
  -- caller never asked for. DESIGN.md section 2: the world pauses between
  -- decisions.
  --
  -- Unless the caller declared `free_running` -- see `handle_configure`, which
  -- also records why that is no longer needed to watch a run. A free-running
  -- reset still installs the scene correctly; what it gives up is that the
  -- first observation is taken at a tick nobody chose.
  game.tick_paused = not state.free_running
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

local function record_event(request_id, action, status, error_code)
  state.event_seq = state.event_seq + 1
  state.events[#state.events + 1] = {
    seq = state.event_seq,
    request_id = request_id,
    action = action,
    status = status,
    tick = episode_tick(),
  }
  while #state.events > EVENT_LIMIT do table.remove(state.events, 1) end

  -- The same event, flattened and copied now. A step's event holds the inner
  -- action's result table *by reference*, and that table is merged into again
  -- when the action settles later -- so the "same" event read differently on
  -- later frames, and a copy of the observation had to be taken at the right
  -- moment to mean anything. A flat record is fixed once written, carries only
  -- names and codes, and is what a profile with `flat_events` publishes.
  local flat = {
    seq = state.event_seq,
    request_id = request_id,
    tick = episode_tick(),
    status = status,
    error = error_code,
  }
  if type(action) == "table" then
    flat.action = action.action
    flat.result = action.status
    flat.error = flat.error or (type(action.error) == "table" and action.error.code or nil)
  else
    flat.action = action
  end
  state.flat_events = state.flat_events or {}
  state.flat_events[#state.flat_events + 1] = flat
  while #state.flat_events > EVENT_LIMIT do table.remove(state.flat_events, 1) end
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
  record_event(request_id, merged.action, status, error_code)
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
    -- Declared, so a status read says whether this worker is stepping
    -- exactly or merely quickly.
    free_running = state.free_running or false,
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
    -- Every type `HANDLERS` dispatches. A client that validates what it may
    -- send by reading `describe` is told this list and nothing else, so an
    -- omission here is a request type that exists and is invisible. This list
    -- silently missed `open_world` when it was added; a contract test now
    -- asserts it equals the handler table.
    request_types = {
      "status", "observe", "reset", "advance", "act",
      "request_status", "step", "collect", "describe", "configure",
      "scenario_define", "truth", "world_digest", "disrupt", "open_world",
      "save", "knowledge",
    },
  })
end

--- Evaluator-facing pacing knobs.
--
-- `speed` does not change simulation outcomes -- Factorio is tick-based, so the
-- simulation is identical at any speed, and `factoriorl bench speed` proves it
-- by comparing episode records field by field.
--
-- `free_running` **does**, and is the one knob here that has to be declared
-- rather than assumed harmless. With it set, the world is not re-paused between
-- decisions: it keeps ticking while the caller decides, so an action lands at
-- whatever tick it happens to arrive at instead of exactly `decision_ticks`
-- after the last one. Every measured run leaves it off.
--
-- It was added on the belief that exact stepping and *watching* are
-- incompatible: a server with a client connected and its tick loop paused
-- appeared to stop answering RCON altogether. That was a reply-ordering bug on
-- the Python side, not an engine property, and with it fixed exact stepping was
-- measured working with a client in the game -- six steps of exactly 30 ticks,
-- `tick_paused` true between every one, 0 ticks of drift across three idle
-- seconds. So nothing requires this knob any more. It stays because a
-- continuously moving world is easier to watch than a stuttering one, and
-- because a run that used it must still say so.
local function handle_configure(request)
  local payload = request.payload or {}
  local applied = {}
  if payload.free_running ~= nil then
    if type(payload.free_running) ~= "boolean" then
      return respond(request, CODE.REJECTED, nil,
        err(ERR.BAD_TYPE, "free_running must be a boolean"))
    end
    state.free_running = payload.free_running
    -- Symmetric, and it was not. Turning `free_running` **on** unpaused
    -- immediately; turning it off only changed the flag, and the world kept
    -- running until the next step re-applied `tick_paused = not free_running`.
    -- With no next step -- which is exactly the situation at the end of a run,
    -- when the world is being stopped so a final save can be taken -- the world
    -- never paused at all. Measured: 59 ticks drifted across `game.server_save`,
    -- so the save held a different world from the last observation the agent
    -- saw. Roadmap A4.3 asks for the world to be paused *for* the snapshot.
    game.tick_paused = not state.free_running
    applied.free_running = state.free_running
    applied.tick_paused = game.tick_paused
  end
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

local function handle_scenario_define(request)
  local blueprint = request.payload and request.payload.blueprint
  local hash = request.payload and request.payload.hash
  if type(blueprint) ~= "table" or type(hash) ~= "string" then
    return respond(request, CODE.REJECTED, nil,
      err(ERR.MISSING_FIELD, "payload.hash and payload.blueprint are required"))
  end
  world.define_blueprint(hash, blueprint)
  return respond(request, CODE.OK, {
    status = STATUS.COMPLETED, hash = hash,
    entities = #(blueprint.entities or {}),
    resources = #(blueprint.resources or {}),
  })
end

--- Evaluator-only ground truth. A separate request from `observe`, so the
--- separation between what a policy may see and what the evaluator knows is
--- structural rather than a naming convention (DESIGN.md section 2).
--- Evaluator-only leakage probe: sorted lines describing everything a reset
--- must restore, plus growth proxies that catch unbounded state.
local function handle_world_digest(request)
  local result = {
    lines = world.digest(),
    growth = {
      handles = handles.count(),
      remembered = memory.count(),
      ledger = #state.ledger_order,
      events = #state.events,
      inflight = #inflight.summary(),
    },
  }
  -- Opt-in, so the Phase 3 reset loop does not pay for it. What the parity
  -- recorder loads into a simulator for one-step sync: see `world.hidden_state`.
  if (request.payload or {}).hidden then
    local hidden = world.hidden_state()
    hidden.handles = handles.export()
    hidden.inflight = inflight.export()
    hidden.event_seq = state.event_seq
    result.hidden = hidden
  end
  return respond(request, CODE.OK, result)
end

--- Apply one declared disruption. Evaluator-only, like `truth` and
--- `world_digest`: it changes the world, so it is a mutating request, but it is
--- never reachable from an action the policy can name.
local function handle_disrupt(request)
  local payload = request.payload or {}
  if type(payload.kind) ~= "string" then
    return err(request, CODE.BAD_REQUEST, "disrupt needs a string `kind`")
  end
  if type(payload.targets) ~= "table" or #payload.targets == 0 then
    return err(request, CODE.BAD_REQUEST, "disrupt needs a non-empty `targets` list")
  end
  local applied, problem = world.disrupt(payload.kind, payload.targets)
  if not applied then
    return err(request, CODE.BAD_REQUEST, problem or "disruption did not apply")
  end
  return respond(request, CODE.OK, applied)
end

local function handle_truth(request)
  return respond(request, CODE.OK, world.truth())
end

local function handle_reset(request)
  local payload = request.payload or {}
  -- A reset that omits a profile restores the default rather than keeping the
  -- last episode's. Profiles are a capability boundary, so carrying one over
  -- silently is how a run measured "without assistance" ends up having had it:
  -- one episode asks for `assisted-v1`, every later episode inherits it, and
  -- nothing in the manifest or the observation says so. An episode boundary
  -- resets episode state, and this is episode state.
  if payload.observation_profile then
    if not profiles.observation(payload.observation_profile) then
      return respond(request, CODE.REJECTED, nil,
        err(ERR.INVALID_TARGET, "unknown observation profile"))
    end
    state.observation_profile = payload.observation_profile
  else
    state.observation_profile = profiles.DEFAULT_OBSERVATION
  end
  if payload.action_profile then
    local action_profile = profiles.action(payload.action_profile)
    if not action_profile or not action_profile.available then
      return respond(request, CODE.REJECTED, nil,
        err(ERR.INVALID_TARGET, "unknown or unavailable action profile"))
    end
    state.action_profile = payload.action_profile
  else
    state.action_profile = profiles.DEFAULT_ACTION
  end
  if payload.scenario then
    storage.frrl_scenario_name = payload.scenario
  end
  -- Character state is reset *before* the scene is built, not after. Building
  -- a blueprint sets the character's position and starting inventory, and
  -- reset_state clears both -- so running it afterwards silently discarded
  -- every item a task meant to hand the agent. That is why repair_belt could
  -- never place a belt it was explicitly given.
  actions.reset_state()
  world.recreate_character()
  -- Before the scene is built: the blueprint re-enables whatever recipes the
  -- task hands the agent, and that must land after the wipe rather than be
  -- undone by it.
  world.reset_force()

  local scene
  if payload.blueprint_hash then
    if not world.has_blueprint(payload.blueprint_hash) then
      return respond(request, CODE.REJECTED, nil,
        err(ERR.INVALID_TARGET, "unknown blueprint: " .. payload.blueprint_hash,
          { hash = payload.blueprint_hash }))
    end
    scene = world.build_blueprint(payload.blueprint_hash)
  else
    scene = world.reset_scene()
  end
  -- Cumulative statistics are exactly where cross-episode leakage hides, and
  -- nothing cleared them before.
  world.clear_statistics()
  local new_episode = begin_episode()
  return respond(request, CODE.OK, {
    episode_id = new_episode,
    scenario = scene.scenario,
    destroyed = scene.destroyed,
    profiles = profiles.metadata(state.observation_profile, state.action_profile),
  })
end

-- Initialise a generated map, as opposed to installing a painted scene.
--
-- A separate request rather than a flag on `reset`, because the two differ in
-- almost everything they do: `reset` clears the surface, rebuilds a declared
-- blueprint and un-researches the force, and every one of those is wrong for a
-- world the agent is meant to explore and keep. Sharing a handler would mean one
-- more branch in the most load-bearing function in this file.
--
-- `fresh = false` leaves the character and its inventory alone, which is what a
-- resumed world needs: the save already holds what the agent had.
local function handle_open_world(request)
  local payload = request.payload or {}
  if payload.observation_profile then
    if not profiles.observation(payload.observation_profile) then
      return respond(request, CODE.REJECTED, nil,
        err(ERR.INVALID_TARGET, "unknown observation profile"))
    end
    state.observation_profile = payload.observation_profile
  else
    state.observation_profile = profiles.DEFAULT_OBSERVATION
  end
  if payload.action_profile then
    local action_profile = profiles.action(payload.action_profile)
    if not action_profile or not action_profile.available then
      return respond(request, CODE.REJECTED, nil,
        err(ERR.INVALID_TARGET, "unknown or unavailable action profile"))
    end
    state.action_profile = payload.action_profile
  else
    state.action_profile = profiles.DEFAULT_ACTION
  end

  local fresh = payload.fresh ~= false
  if fresh then
    -- Clears the character's inventory and any in-flight operation, before the
    -- starting items are inserted -- the same ordering `reset` needs, and for
    -- the same reason: doing it afterwards discards what was just handed over.
    actions.reset_state()
  end

  local built = world.open_world({
    fresh = fresh,
    inventory = fresh and payload.inventory or nil,
    position = payload.position,
    chart_radius = payload.chart_radius,
  })
  -- Deliberately no `world.reset_force()` and no `world.clear_statistics()`:
  -- both are episode-boundary operations for a benchmark scene, and both would
  -- destroy state an open world is supposed to accumulate.
  local new_episode = begin_episode()
  return respond(request, CODE.OK, {
    episode_id = new_episode,
    scenario = built.scenario,
    destroyed = built.destroyed,
    delivered = built.delivered,
    undelivered = built.undelivered,
    position = built.position,
    -- Every resource patch inside the charted area. This handler names each
    -- field it passes through, so a new one on `built` is invisible until it is
    -- named here -- which is exactly what happened: `world.open_world` returned
    -- the survey, the reply dropped it, and the Python side read `None`.
    survey = built.survey,
    survey_radius = built.survey_radius,
    fresh = fresh,
    profiles = profiles.metadata(state.observation_profile, state.action_profile),
  })
end

-- Ask the server to write a save, and say only that the request was issued.
--
-- `game.server_save` is deferred: the engine executes it at the end of the
-- current tick, and there is no completion event to subscribe to. Lua also has
-- no `io` and no `game.file_exists`, so this side **cannot observe whether the
-- file appeared**. Reporting `completed` here would be the mod asserting a fact
-- it has no way to check, which is the defect class this repository keeps
-- paying for. Python owns verification.
--
-- `paused` is returned because it decides whether the save can happen at all.
-- Deferred means end-of-tick, and between decisions the world is paused, so a
-- save issued into a paused world sits there forever. The caller advances a tick
-- when this says true.
local function handle_save(request)
  local payload = request.payload or {}
  local name = payload.name
  if not name or name == "" then
    return respond(request, CODE.REJECTED, nil,
      err(ERR.MISSING_FIELD, "a save name is required"))
  end
  -- Only in multiplayer, which every worker is: `worker.py` starts the engine
  -- with `--start-server`. Stated because a single-player call silently does
  -- nothing, which would look exactly like a save that never finished.
  game.server_save(name)
  return respond(request, CODE.OK, {
    issued = true,
    name = name,
    tick = game.tick,
    paused = game.tick_paused,
    verified_by = "the caller; this side cannot read the filesystem",
  })
end

-- Static game data: recipes, placeable footprints and the technology tree.
--
-- Read-only, and the only request in the protocol that answers a question about
-- the *game* rather than about this world. Everything it returns was already
-- reachable from Lua and was never returned: `actions.lua` looks up a recipe's
-- first product and an item's `place_result` on every craft and place, and
-- discards both. A client wanting to plan therefore had to learn the recipe
-- graph by attempting crafts, and `tasks/spec.py` carries a hardcoded table of
-- four entity footprints because the size of a stone furnace was not askable.
--
-- Deliberately **not** in `MUTATING`. That table confers deduplication, and a
-- deduplicated read would answer a later call with an earlier snapshot -- which
-- is exactly wrong here, because the enabled/researched fields are force state
-- that research moves.
--
-- Takes no payload. A section filter was the obvious parameter and is not worth
-- it: the whole reply is fetched once per world and cached on disk by the
-- caller, so the cost of the parts nobody reads is paid once.
local function handle_knowledge(request)
  return respond(request, CODE.OK, knowledge.snapshot())
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
  scenario_define = handle_scenario_define,
  truth = handle_truth,
  world_digest = handle_world_digest,
  disrupt = handle_disrupt,
  open_world = handle_open_world,
  save = handle_save,
  knowledge = handle_knowledge,
}

-- `disrupt` mutates the scene, so it belongs here: whatever gating a
-- mutating request receives, a disruption must receive too.
-- `open_world` destroys player-force entities and moves the character, so it
-- is as mutating as `reset` and receives the same gating.
local MUTATING = {
  advance = true,
  act = true,
  reset = true,
  step = true,
  disrupt = true,
  open_world = true,
  -- Not a world mutation, and here anyway. What this table confers is
  -- deduplication and an episode check, and a save needs both: a resend after an
  -- ambiguous transport timeout must return the stored reply rather than write a
  -- second file, and a save belonging to a finished episode should be refused
  -- rather than applied to the current one.
  save = true,
}

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
  local snapshot = nil
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
        snapshot = result.observation
        -- Evaluator truth rides along: fetching it separately cost a third
        -- round trip on every single RL step.
        result.truth = world.truth()
      end
      ledger_settle(entry.request_id, item.status, result, item.error_code)
      game.tick_paused = not state.free_running
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

  -- The snapshot is taken before this tick's settles are recorded, including
  -- the step's own. A full-buffer profile still shows them, because it
  -- publishes `state.events` itself and the table is serialised later; a
  -- windowed profile publishes a copy, so it is refreshed here, once every
  -- event of the tick is in. Without this the two profiles disagreed about
  -- the last settled outcome on the first frame of every episode.
  if snapshot then observations.refresh_events(snapshot, state) end
end

return runtime
