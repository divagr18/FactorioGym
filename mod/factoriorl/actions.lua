-- Typed embodied actions (Phase 0 scope: move, transfer, wait).
--
-- Semantics:
--   move: character walks in a cardinal direction until the decision interval
--     deadline; collision stops it earlier. The action itself completes when
--     the walking state is applied; displacement is observable after advances.
--   transfer: all-or-nothing item move between character and a scene chest
--     within reach. Fails cleanly on out-of-reach or insufficient items.
--   wait: no state change; advances still consume game time.

local observations = require("observations")

local actions = {}

local DECISION_INTERVAL = 30
local REACH_DISTANCE = 6

local DIRECTIONS = {
  north = defines.direction.north,
  east = defines.direction.east,
  south = defines.direction.south,
  west = defines.direction.west,
}

local function character()
  return storage.frrl_character
end

local function distance(a, b)
  local dx = a.x - b.x
  local dy = a.y - b.y
  return math.sqrt(dx * dx + dy * dy)
end

-- ---------------------------------------------------------------- move

local function act_move(state, request, respond, err)
  local direction = request.payload and request.payload.direction
  if not direction then
    return respond(request, "rejected", nil, err("missing_field", "payload.direction is required"))
  end
  local dir = DIRECTIONS[direction]
  if not dir then
    return respond(request, "rejected", nil,
      err("bad_type", "direction must be north/east/south/west", { got = tostring(direction) }))
  end
  local ch = character()
  if not ch or not ch.valid then
    return respond(request, "rejected", nil, err("precondition", "no character present"))
  end
  ch.walking_state = { walking = true, direction = dir }
  storage.frrl_move_deadline = game.tick + (request.payload.ticks or DECISION_INTERVAL)
  return respond(request, "ok", {
    status = "completed",
    action = "move",
    direction = direction,
    position = { ch.position.x, ch.position.y },
  })
end

-- ---------------------------------------------------------------- transfer

local function scene_target(name)
  local scene = storage.frrl_scene
  if not scene then return nil end
  if name == "src" then return scene.src end
  if name == "dst" then return scene.dst end
  return nil
end

local function inventory_of(who)
  if who == "character" then
    local ch = character()
    if not ch or not ch.valid then return nil end
    return ch.get_inventory(defines.inventory.character_main)
  end
  local entity = scene_target(who)
  if not entity or not entity.valid then return nil end
  return entity.get_inventory(defines.inventory.chest)
end

local function act_transfer(state, request, respond, err)
  local payload = request.payload or {}
  local from, to, item, count = payload.from, payload.to, payload.item, payload.count
  if not from or not to or not item or not count then
    return respond(request, "rejected", nil,
      err("missing_field", "payload requires from, to, item, count"))
  end
  if type(count) ~= "number" or count ~= math.floor(count) or count < 1 then
    return respond(request, "rejected", nil,
      err("bad_type", "count must be a positive integer", { got = tostring(count) }))
  end
  if from == to then
    return respond(request, "rejected", nil,
      err("precondition", "from and to must differ"))
  end

  -- Reach check: character must be within reach of every non-character endpoint.
  local ch = character()
  if not ch or not ch.valid then
    return respond(request, "rejected", nil, err("precondition", "no character present"))
  end
  for _, ref in pairs({ from, to }) do
    if ref ~= "character" then
      local entity = scene_target(ref)
      if not entity or not entity.valid then
        return respond(request, "rejected", nil,
          err("precondition", "unknown scene reference: " .. tostring(ref)))
      end
      if distance(ch.position, entity.position) > REACH_DISTANCE then
        return respond(request, "rejected", nil,
          err("out_of_reach", tostring(ref) .. " is out of reach", {
            distance = distance(ch.position, entity.position),
            reach = REACH_DISTANCE,
          }))
      end
    end
  end

  local from_inv = inventory_of(from)
  local to_inv = inventory_of(to)
  if not from_inv or not to_inv then
    return respond(request, "rejected", nil, err("precondition", "inventory unavailable"))
  end

  local available = from_inv.get_item_count(item)
  if available < count then
    return respond(request, "rejected", nil,
      err("no_items", "source holds " .. available .. " of " .. item, {
        item = item, requested = count, available = available,
      }))
  end

  -- All-or-nothing: verify destination space before touching either inventory.
  local inserted_probe = to_inv.can_insert({ name = item, count = count })
  if not inserted_probe then
    return respond(request, "rejected", nil,
      err("precondition", "destination cannot hold " .. count .. " " .. item))
  end

  local removed = from_inv.remove({ name = item, count = count })
  if removed ~= count then
    -- Cannot happen after get_item_count, but never leave a half-applied transfer.
    if removed > 0 then from_inv.insert({ name = item, count = removed }) end
    return respond(request, "rejected", nil,
      err("engine", "removal returned " .. tostring(removed) .. " items"))
  end
  local inserted = to_inv.insert({ name = item, count = count })
  if inserted ~= count then
    from_inv.insert({ name = item, count = count - inserted })
    to_inv.remove({ name = item, count = inserted })
    return respond(request, "rejected", nil,
      err("engine", "insertion returned " .. tostring(inserted) .. " items"))
  end

  -- Task counters (Phase 0 task state): cleared on reset.
  storage.frrl_task = storage.frrl_task or { transfers = 0, items_moved = 0 }
  storage.frrl_task.transfers = storage.frrl_task.transfers + 1
  storage.frrl_task.items_moved = storage.frrl_task.items_moved + count

  return respond(request, "ok", {
    status = "completed",
    action = "transfer",
    item = item,
    count = count,
    from = from,
    to = to,
    source_remaining = from_inv.get_item_count(item),
    destination_total = to_inv.get_item_count(item),
  })
end

-- ---------------------------------------------------------------- wait

local function act_wait(state, request, respond, err)
  return respond(request, "ok", { status = "completed", action = "wait" })
end

local ACTION_DISPATCH = {
  move = act_move,
  transfer = act_transfer,
  wait = act_wait,
}

function actions.dispatch(state, request, respond, err)
  local action = request.payload and request.payload.action
  if not action then
    return respond(request, "rejected", nil, err("missing_field", "payload.action is required"))
  end
  local handler = ACTION_DISPATCH[action]
  if not handler then
    return respond(request, "rejected", nil,
      err("unknown_action", "unknown action: " .. tostring(action)))
  end
  return handler(state, request, respond, err)
end

-- ---------------------------------------------------------------- ticking

function actions.on_tick(state)
  local deadline = storage.frrl_move_deadline
  if deadline and game.tick >= deadline then
    local ch = character()
    if ch and ch.valid and ch.walking_state.walking then
      ch.walking_state = { walking = false }
    end
    storage.frrl_move_deadline = nil
  end
end

function actions.on_init()
  storage.frrl_task = { transfers = 0, items_moved = 0 }
  storage.frrl_move_deadline = nil
end

function actions.reset_state()
  storage.frrl_task = { transfers = 0, items_moved = 0 }
  storage.frrl_move_deadline = nil
  local ch = character()
  if ch and ch.valid then
    ch.teleport({ 0, 0 })
    ch.walking_state = { walking = false }
    local inv = ch.get_inventory(defines.inventory.character_main)
    if inv then inv.clear() end
  end
end

function actions.task_state()
  return storage.frrl_task or { transfers = 0, items_moved = 0 }
end

return actions
