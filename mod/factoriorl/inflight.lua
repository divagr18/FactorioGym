-- Ongoing actions (DESIGN.md 2.2): one registry for every operation that spans
-- decision intervals.
--
-- This replaces two single-purpose hacks: the advance's dedicated
-- (advance_until, pending_advance_request_id, pending_advance_start_tick)
-- fields, and `storage.frrl_move_deadline`, a bare global scalar with no
-- request association, no running status, no completion report and no
-- cancellation -- a second move silently overwrote it.
--
-- Storage rule: nothing here may hold a function, closure, metatable, or a
-- table keyed by LuaEntity. `runtime.on_load` restores this wholesale from
-- `storage`, and autosave is on, so a stored closure would break the worker at
-- the first save/load. `kind` is therefore a *string* dispatched through the
-- module-level POLLS table below.

local protocol = require("protocol")
local handles = require("handles")
local navigation = require("navigation")
-- `world` does not require this module, so the edge is one-way.
local world = require("world")

local STATUS = protocol.STATUS
local ERR = protocol.ERR

local inflight = {}

--- One operation per slot; crafting is a genuine engine queue, so it is exempt.
---
--- `navigate` shares the `move` slot rather than getting its own, and that is
--- the whole of its supersede story. Both drive `walking_state`, so two of them
--- running at once would fight for the same body every tick and the observable
--- result would depend on poller iteration order -- a nondeterminism that
--- would leak into every trace. Sharing the slot means a `move` supersedes a
--- running `navigate` for free (the agent taking manual control back), a new
--- `navigate` supersedes the old one (a new destination replaces the old
--- plan), and neither needs a special case in the other's handler.
local SLOT_OF = {
  move = "move",
  navigate = "move",
  mine = "mine",
  advance = "advance",
}

local function state()
  return storage.frrl_inflight
end

function inflight.reset()
  storage.frrl_inflight = { entries = {}, slots = {}, next_seq = 0 }
end

function inflight.get(request_id)
  return state().entries[request_id]
end

--- Everything currently running, for observations and `describe`.
function inflight.summary(ordered)
  local out = {}
  local seqs = {}
  for request_id, entry in pairs(state().entries) do
    if not entry.terminal then
      out[#out + 1] = {
        request_id = request_id,
        action = entry.action,
        started_tick = entry.started_tick,
        progress = inflight.progress(entry),
        target = entry.target_handle,
      }
      seqs[request_id] = entry.seq or 0
    end
  end
  if ordered then
    -- Start order, for a profile with `deterministic_order`. A character that
    -- walks while it crafts has two entries, so this is not a rare tie, and
    -- `pairs()` order is specified nowhere outside the engine. `seq` is kept out
    -- of the published record so the shape stays what it was.
    table.sort(out, function(a, b)
      if seqs[a.request_id] ~= seqs[b.request_id] then
        return seqs[a.request_id] < seqs[b.request_id]
      end
      return a.request_id < b.request_id
    end)
  end
  return out
end

--- Running operations with the fields their pollers read, in start order.
-- Evaluator-only: a simulator in sync mode loads these, because a mine's
-- completion depends on `baseline`, which no observation carries.
function inflight.export()
  local s = state()
  local out = {}
  for _, entry in pairs(s.entries) do
    if not entry.terminal then
      out[#out + 1] = {
        request_id = entry.request_id,
        seq = entry.seq or 0,
        action = entry.action,
        kind = entry.kind,
        started_tick = entry.started_tick,
        deadline_tick = entry.deadline_tick,
        target = entry.target_handle,
        goal = entry.goal,
        baseline = entry.baseline,
        queued = entry.queued,
        recipe = entry.recipe,
      }
    end
  end
  table.sort(out, function(a, b)
    if a.seq ~= b.seq then return a.seq < b.seq end
    return a.request_id < b.request_id
  end)
  return { next_seq = s.next_seq or 0, entries = out }
end

-- ---------------------------------------------------------------- mine baseline
--
-- A mine is complete when the character holds `goal.count` more of the mined
-- item than at `baseline`. That is a count of the inventory, not of mining, so
-- anything else that moved the item in or out was read as mining: a
-- `take_from` of iron ore during a hand-mine completed the mine and tallied
-- the taken ore as `mined_by_action`, and a `give_to` did the reverse. The
-- tally feeds `machine_produced`, which caps `construct_smelting_line`'s
-- verified plates, so a working line could be scored nothing. The M2 parity
-- trace `transfer_clamping` recorded exactly that.
--
-- The count stays; the baseline moves with every change that is not mining.
-- Two sources exist: actions (measured around each one in
-- `actions.dispatch`) and the engine's crafting queue delivering a product
-- (measured each tick, before the mine is polled).

local function held(item)
  local ch = storage.frrl_character
  if not (ch and ch.valid and item) then return 0 end
  local inv = ch.get_inventory(defines.inventory.character_main)
  return inv and inv.get_item_count(item) or 0
end

--- How many of `item` the crafting queue will still deliver.
function inflight.queued(item)
  local ch = storage.frrl_character
  if not (ch and ch.valid and item) then return 0 end
  local total = 0
  for _, entry in pairs(ch.crafting_queue or {}) do
    local recipe = prototypes.recipe[entry.recipe]
    for _, product in pairs(recipe and recipe.products or {}) do
      if product.name == item then
        total = total + entry.count * (product.amount or 1)
      end
    end
  end
  return total
end

--- The running mine, or nil.
function inflight.running_mine()
  local request_id = inflight.occupant("mine")
  return request_id and state().entries[request_id] or nil
end

--- Snapshot before something that is not mining touches the inventory.
function inflight.mine_guard()
  local entry = inflight.running_mine()
  if not entry or not entry.goal then return nil end
  return { entry = entry, held = held(entry.goal.item) }
end

--- Move the baseline by whatever that something did to the mined item.
function inflight.mine_unguard(guard)
  if not guard then return end
  local entry = guard.entry
  if entry.terminal then return end
  entry.baseline = (entry.baseline or 0) + held(entry.goal.item) - guard.held
  -- An action can also change the queue (crafting, cancelling a craft), and
  -- that is not a delivery.
  entry.queued = inflight.queued(entry.goal.item)
end

--- Credit items the crafting queue delivered since the last tick to the
--- baseline, not to mining.
local function account_crafting(entry)
  local now = inflight.queued(entry.goal.item)
  local before = entry.queued or now
  if now < before then
    entry.baseline = (entry.baseline or 0) + (before - now)
  end
  entry.queued = now
end

--- Is `slot` occupied, and by which request?
function inflight.occupant(action)
  local slot = SLOT_OF[action]
  if not slot then return nil end
  local request_id = state().slots[slot]
  if not request_id then return nil end
  local entry = state().entries[request_id]
  if not entry or entry.terminal then
    state().slots[slot] = nil
    return nil
  end
  return request_id
end

--- Register a new ongoing operation.
function inflight.start(request_id, action, fields)
  local s = state()
  -- Start order, recorded so polling can follow it. `next_seq` is absent in a
  -- state restored from a save that predates it, which is why it defaults.
  s.next_seq = (s.next_seq or 0) + 1
  local entry = {
    request_id = request_id,
    seq = s.next_seq,
    action = action,
    kind = fields.kind or action,
    started_tick = game.tick,
    deadline_tick = fields.deadline_tick,
    target_handle = fields.target_handle,
    goal = fields.goal,
    baseline = fields.baseline,
    recipe = fields.recipe,
    -- Free-form per-operation state, for operations that carry more than a
    -- deadline and a goal count -- currently only `navigate`, which holds its
    -- route, waypoint index and replan budget here. Subject to the storage
    -- rule at the top of this file: plain serialisable tables only, no
    -- closures and no LuaEntity keys, because `on_load` restores this whole
    -- structure from `storage` and autosave is on.
    data = fields.data,
    terminal = false,
  }
  if action == "mine" and entry.goal then
    -- What the crafting queue still owes, so a delivery during the mine is
    -- credited to the baseline rather than to mining. See `mine_guard`.
    entry.queued = inflight.queued(entry.goal.item)
  end
  s.entries[request_id] = entry
  local slot = SLOT_OF[action]
  if slot then s.slots[slot] = request_id end
  return entry
end

-- ---------------------------------------------------------------- pollers

local function character()
  local ch = storage.frrl_character
  if ch and ch.valid then return ch end
  return nil
end

--- Each poller returns (terminal_status, result_fields, error_code) or nil to
--- keep running. None of these may be stored; they are looked up by `kind`.
local POLLS = {}

POLLS.move = function(entry)
  local ch = character()
  if not ch then
    return STATUS.FAILED, { reason = "character missing" }, ERR.TARGET_MISSING
  end
  if game.tick >= (entry.deadline_tick or 0) then
    ch.walking_state = { walking = false }
    return STATUS.COMPLETED, {
      action = "move",
      ticks_walked = game.tick - entry.started_tick,
      position = { ch.position.x, ch.position.y },
    }, nil
  end
  return nil
end

POLLS.mine = function(entry)
  local ch = character()
  if not ch then
    return STATUS.FAILED, { reason = "character missing" }, ERR.TARGET_MISSING
  end
  account_crafting(entry)
  local inv = ch.get_inventory(defines.inventory.character_main)
  local produced = inv and inv.get_item_count(entry.goal.item) or 0
  local gained = produced - (entry.baseline or 0)

  local target, reason = handles.resolve(entry.target_handle)
  if not target then
    -- The patch tile emptied. Anything already mined still counts, but the
    -- operation cannot continue: a target becoming unavailable is a recorded
    -- failure (DESIGN.md 2.2), not a silent stop.
    ch.mining_state = { mining = false }
    -- Counted here rather than from `on_player_mined_item`, because whether
    -- that event fires for the controlled character depends on whether the
    -- engine considers it a player. Both are recorded; `world.truth` reports
    -- which moved. Roadmap A4.2.
    world.tally("mined_by_action", entry.goal.item, gained)
    if gained >= entry.goal.count then
      return STATUS.COMPLETED, { action = "mine", mined = gained }, nil
    end
    return STATUS.FAILED,
      { action = "mine", mined = gained, requested = entry.goal.count },
      reason == "unknown_handle" and ERR.UNKNOWN_HANDLE or ERR.TARGET_MISSING
  end

  if gained >= entry.goal.count then
    ch.mining_state = { mining = false }
    world.tally("mined_by_action", entry.goal.item, gained)
    return STATUS.COMPLETED, { action = "mine", mined = gained }, nil
  end
  -- Keep the engine mining: selection can lapse when a resource entity is
  -- consumed and the next one takes its place.
  ch.update_selected_entity(target.position)
  ch.mining_state = { mining = true, position = target.position }
  return nil
end

--- Route following lives in `navigation.lua`; this is only the registry hook.
--- It stays a thin forward so the poller table keeps its property of holding
--- nothing but module-level functions looked up by a stored `kind` string.
POLLS.navigate = function(entry)
  return navigation.poll(entry)
end

POLLS.craft = function(entry)
  local ch = character()
  if not ch then
    return STATUS.FAILED, { reason = "character missing" }, ERR.TARGET_MISSING
  end
  local queue = ch.crafting_queue
  local still_queued = false
  if queue then
    for _, item in pairs(queue) do
      if item.recipe == entry.recipe then still_queued = true break end
    end
  end
  if still_queued then return nil end
  local inv = ch.get_inventory(defines.inventory.character_main)
  local produced = inv and inv.get_item_count(entry.goal.item) or 0
  -- Only the requested recipe's own output. The crafting queue may build
  -- intermediates on the way, and those are *not* counted here -- which means
  -- they fall into the derived machine column. Named as a known bias rather
  -- than papered over: see `world.truth`.
  world.tally("handcrafted_by_action", entry.goal.item, produced - (entry.baseline or 0))
  return STATUS.COMPLETED, {
    action = "craft",
    recipe = entry.recipe,
    produced = produced - (entry.baseline or 0),
  }, nil
end

POLLS.advance = function(entry)
  if game.tick >= (entry.deadline_tick or 0) then
    return STATUS.COMPLETED, {
      ticks_advanced = game.tick - entry.started_tick,
    }, nil
  end
  return nil
end

--- Progress in [0, 1] where the engine exposes it, else nil.
function inflight.progress(entry)
  local ch = character()
  if entry.action == "mine" and ch then
    return ch.character_mining_progress
  elseif entry.action == "craft" and ch then
    return ch.crafting_queue_progress
  elseif entry.action == "navigate" and ch and entry.data and entry.data.goal then
    -- Fraction of the original distance closed, not fraction of the time
    -- budget spent. `navigate` carries a deadline only as a safety stop, so
    -- deadline-based progress would report a route that is nearly finished as
    -- barely started, and would climb steadily while the character stood
    -- against a wall.
    local total = entry.data.start_distance or 0
    if total > 0 then
      local left = navigation.goal_distance(entry.data.goal, ch.position.x, ch.position.y)
      return math.max(0.0, math.min(1.0, (total - left) / total))
    end
    return nil
  elseif entry.deadline_tick then
    local span = entry.deadline_tick - entry.started_tick
    if span > 0 then
      return math.min(1.0, (game.tick - entry.started_tick) / span)
    end
  end
  return nil
end

--- Advance every running operation one tick. Returns the entries that settled,
--- each as { entry, status, result, error_code }, for the runtime to record.
function inflight.on_tick()
  local s = state()
  local settled = {}
  -- Poll in the order operations started, not in `pairs()` order. The order
  -- matters beyond the event log: polls act on the world -- a mine re-asserts
  -- `mining_state` and counts an inventory change that a craft settling the
  -- same tick can alter -- and `pairs()` over request-id keys is stable in
  -- Factorio but specified nowhere, so nothing outside the engine could
  -- reproduce which of two same-tick completions came first.
  local running = {}
  for _, entry in pairs(s.entries) do
    running[#running + 1] = entry
  end
  table.sort(running, function(a, b)
    if (a.seq or 0) ~= (b.seq or 0) then return (a.seq or 0) < (b.seq or 0) end
    return a.request_id < b.request_id
  end)
  for _, entry in ipairs(running) do
    if not entry.terminal then
      local poll = POLLS[entry.kind]
      if poll then
        local ok, status, result, error_code = pcall(poll, entry)
        if not ok then
          -- One bad entry must not kill the game loop for the whole worker.
          settled[#settled + 1] = {
            entry = entry,
            status = STATUS.FAILED,
            result = { action = entry.action, error = tostring(status) },
            error_code = ERR.ENGINE,
          }
        elseif status then
          settled[#settled + 1] = {
            entry = entry,
            status = status,
            result = result,
            error_code = error_code,
          }
        end
      end
    end
  end
  for _, item in ipairs(settled) do
    inflight.finish(item.entry)
  end
  return settled
end

--- Mark an entry terminal and free its slot. Idempotent: this is the single
--- place a terminal status is written, which is what makes "completion is
--- reported exactly once" an invariant rather than a convention.
function inflight.finish(entry)
  if entry.terminal then return false end
  entry.terminal = true
  local slot = SLOT_OF[entry.action]
  if slot and state().slots[slot] == entry.request_id then
    state().slots[slot] = nil
  end
  return true
end

--- Stop an operation at the current (paused) tick.
-- The boundary is exact and observable precisely because the world is paused
-- while an RCON request is handled.
function inflight.cancel(entry)
  local ch = character()
  local result = { action = entry.action, cancelled_at_tick = game.tick }
  if entry.action == "move" and ch then
    ch.walking_state = { walking = false }
    result.position = { ch.position.x, ch.position.y }
  elseif entry.action == "navigate" and ch then
    -- Stopping the body is the whole cancellation: a route is a plan, not a
    -- commitment the world has to be walked back out of. The partial walk
    -- stands, which is why the report says where the character actually is and
    -- how much of the route was left.
    navigation.stop(ch)
    result.position = { ch.position.x, ch.position.y }
    local d = entry.data
    if d then
      result.arrived = false
      result.destination = { d.goal.x, d.goal.y }
      result.waypoints_remaining = math.max(0, #d.route - d.index + 1)
      result.remaining = navigation.goal_distance(d.goal, ch.position.x, ch.position.y)
      result.replans = d.replans
    end
  elseif entry.action == "mine" and ch then
    -- Native behaviour: partial mining progress is discarded.
    result.progress_lost = ch.character_mining_progress
    ch.mining_state = { mining = false }
    local inv = ch.get_inventory(defines.inventory.character_main)
    result.mined = (inv and inv.get_item_count(entry.goal.item) or 0) - (entry.baseline or 0)
  elseif entry.action == "craft" and ch then
    local queue = ch.crafting_queue
    if queue then
      for index = #queue, 1, -1 do
        if queue[index].recipe == entry.recipe then
          ch.cancel_crafting({ index = index, count = queue[index].count })
        end
      end
    end
    result.ingredients_refunded = true
  end
  result.ticks_spent = game.tick - entry.started_tick
  inflight.finish(entry)
  return result
end

return inflight
