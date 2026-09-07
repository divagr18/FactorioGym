-- The ten typed embodied actions (PLAN.md 2.1).
--
-- Dispatch is matrix-driven: payload presence, types, ranges and defaults are
-- validated once against `matrix.ACTIONS[name].payload` before any handler
-- runs, so no handler re-implements missing_field/bad_type checks and the
-- declared matrix cannot drift from the enforced behaviour.
--
-- Reach comes from the character's prototype, not a mod constant. There are
-- three distinct distances -- build 10, reach 10, resource 2.7 -- and
-- `can_reach_entity` applies the engine's own bounding-box-aware rule rather
-- than centre-to-centre distance.
--
-- Failed actions leave no partial changes: transfer and place both probe
-- before mutating and roll back on a short insert.

local protocol = require("protocol")
local matrix = require("matrix")
local handles = require("handles")
local inflight = require("inflight")
local world = require("world")

local CODE, STATUS, ERR = protocol.CODE, protocol.STATUS, protocol.ERR

local actions = {}

local DIRECTIONS = {
  north = defines.direction.north,
  east = defines.direction.east,
  south = defines.direction.south,
  west = defines.direction.west,
}

local function character()
  local ch = storage.frrl_character
  if ch and ch.valid then return ch end
  return nil
end

local function distance(a, b)
  local dx, dy = a.x - b.x, a.y - b.y
  return math.sqrt(dx * dx + dy * dy)
end

--- Resolve a transfer endpoint: the character, or an entity handle.
local function endpoint(name)
  if name == "character" then
    local ch = character()
    if not ch then return nil, ERR.PRECONDITION, "no character present" end
    return { character = true, entity = ch }, nil
  end
  local entity, reason = handles.resolve(name)
  if not entity then
    -- Scenario aliases ("src", "dst", ...) are evaluator-declared names, not
    -- agent-visible identity. They exist so the frozen Phase 0/1 fixtures keep
    -- addressing the same chests after handles were introduced.
    local aliased = world.alias_entity(name)
    if aliased then
      handles.mint(aliased)
      return { character = false, entity = aliased }, nil
    end
    return nil,
      reason == "unknown_handle" and ERR.UNKNOWN_HANDLE or ERR.TARGET_MISSING,
      "handle " .. tostring(name) .. ": " .. tostring(reason)
  end
  return { character = false, entity = entity }, nil
end

--- Pick the inventory an item actually belongs in.
-- A furnace has both a fuel and a source inventory, so "the first inventory
-- that exists" would send coal to the ore slot and quietly fail. Choosing by
-- what the inventory will accept keeps the action honest for machines with
-- several input slots.
local CANDIDATE_INVENTORIES = {
  defines.inventory.chest,
  defines.inventory.fuel,
  defines.inventory.furnace_source,
  defines.inventory.furnace_result,
  defines.inventory.assembling_machine_input,
  defines.inventory.assembling_machine_output,
}

local function inventory_of(target, item, for_removal)
  if target.character then
    return target.entity.get_inventory(defines.inventory.character_main)
  end
  local entity = target.entity
  local first = nil
  for _, which in ipairs(CANDIDATE_INVENTORIES) do
    local inv = entity.get_inventory(which)
    if inv then
      first = first or inv
      if item then
        if for_removal then
          if inv.get_item_count(item) > 0 then return inv end
        elseif inv.can_insert({ name = item, count = 1 }) then
          return inv
        end
      end
    end
  end
  return first
end

--- Reach check for an entity target, by the action's declared reach kind.
local function in_reach(ch, entity, reach_kind)
  if reach_kind == matrix.REACH.NONE then return true, nil end
  if reach_kind == matrix.REACH.RESOURCE then
    local d = distance(ch.position, entity.position)
    if d > ch.resource_reach_distance then
      return false, { distance = d, reach = ch.resource_reach_distance }
    end
    return true, nil
  end
  if ch.can_reach_entity(entity) then return true, nil end
  return false, {
    distance = distance(ch.position, entity.position),
    reach = ch.reach_distance,
  }
end

-- ---------------------------------------------------------------- handlers
-- Each handler is (state, request, payload, respond, err) -> response.

local H = {}

H.move = function(_, request, payload, respond, err)
  local ch = character()
  if not ch then
    return respond(request, CODE.REJECTED, nil, err(ERR.PRECONDITION, "no character present"))
  end
  -- A new move supersedes a running one: a policy emitting a direction every
  -- step must not have to spend an action cancelling first. The superseded
  -- entry settles as `cancelled` exactly once.
  --
  -- Moving also interrupts mining, because the two are physically exclusive
  -- for a character -- walking away from a rock stops mining it. Without this
  -- the mine poller keeps re-asserting mining_state every tick and pins the
  -- character in place, so a move would be accepted and then silently do
  -- nothing.
  local superseded = {}
  for _, slot in ipairs({ "move", "mine" }) do
    local occupant = inflight.occupant(slot)
    if occupant then
      local prior = inflight.get(occupant)
      superseded[#superseded + 1] = {
        request_id = occupant,
        action = prior.action,
        result = inflight.cancel(prior),
      }
    end
  end
  if #superseded > 0 then
    storage.frrl_superseded = superseded
  end
  ch.walking_state = { walking = true, direction = DIRECTIONS[payload.direction] }
  inflight.start(request.request_id, "move", {
    deadline_tick = game.tick + payload.ticks,
  })
  return respond(request, CODE.OK, {
    status = STATUS.RUNNING,
    action = "move",
    direction = payload.direction,
    ticks = payload.ticks,
    superseded = (function()
      local ids = {}
      for _, item in ipairs(superseded) do ids[#ids + 1] = item.request_id end
      return ids
    end)(),
  })
end

H.mine = function(_, request, payload, respond, err)
  local ch = character()
  if not ch then
    return respond(request, CODE.REJECTED, nil, err(ERR.PRECONDITION, "no character present"))
  end
  if inflight.occupant("mine") then
    return respond(request, CODE.REJECTED, nil,
      err(ERR.BUSY, "a mining operation is already running"))
  end
  local target, reason = handles.resolve(payload.handle)
  if not target then
    return respond(request, CODE.REJECTED, nil,
      err(reason == "unknown_handle" and ERR.UNKNOWN_HANDLE or ERR.TARGET_MISSING,
        "handle " .. payload.handle .. ": " .. reason))
  end
  local mineable = target.prototype.mineable_properties
  if not mineable or not mineable.minable then
    return respond(request, CODE.REJECTED, nil,
      err(ERR.NOT_MINEABLE, target.name .. " cannot be mined"))
  end
  local reach_kind = target.type == "resource"
    and matrix.REACH.RESOURCE or matrix.REACH.ENTITY
  local ok, details = in_reach(ch, target, reach_kind)
  if not ok then
    return respond(request, CODE.REJECTED, nil,
      err(ERR.OUT_OF_REACH, "target is out of " .. reach_kind .. " reach", details))
  end
  local product = mineable.products and mineable.products[1]
  if not product then
    return respond(request, CODE.REJECTED, nil,
      err(ERR.NOT_MINEABLE, target.name .. " yields nothing"))
  end
  local inv = ch.get_inventory(defines.inventory.character_main)
  if not inv or inv.count_empty_stacks() == 0 then
    return respond(request, CODE.REJECTED, nil, err(ERR.NO_SPACE, "character inventory is full"))
  end

  ch.update_selected_entity(target.position)
  ch.mining_state = { mining = true, position = target.position }
  inflight.start(request.request_id, "mine", {
    target_handle = payload.handle,
    goal = { item = product.name, count = payload.count },
    baseline = inv.get_item_count(product.name),
  })
  return respond(request, CODE.OK, {
    status = STATUS.RUNNING,
    action = "mine",
    target = payload.handle,
    item = product.name,
    count = payload.count,
    mining_time = mineable.mining_time,
  })
end

H.craft = function(_, request, payload, respond, err)
  local ch = character()
  if not ch then
    return respond(request, CODE.REJECTED, nil, err(ERR.PRECONDITION, "no character present"))
  end
  local force = ch.force
  local recipe = force.recipes[payload.recipe]
  if not recipe then
    return respond(request, CODE.REJECTED, nil,
      err(ERR.RECIPE_UNAVAILABLE, "no such recipe: " .. payload.recipe))
  end
  if not recipe.enabled then
    return respond(request, CODE.REJECTED, nil,
      err(ERR.TECH_LOCKED, payload.recipe .. " is not unlocked for this force"))
  end
  local craftable = ch.get_craftable_count(payload.recipe)
  if craftable < payload.count then
    return respond(request, CODE.REJECTED, nil,
      err(ERR.NO_ITEMS, "not enough ingredients",
        { requested = payload.count, craftable = craftable }))
  end
  local product = prototypes.recipe[payload.recipe].products[1]
  local inv = ch.get_inventory(defines.inventory.character_main)
  local baseline = product and inv and inv.get_item_count(product.name) or 0
  local started = ch.begin_crafting({ recipe = payload.recipe, count = payload.count })
  if not started or started == 0 then
    return respond(request, CODE.REJECTED, nil,
      err(ERR.PRECONDITION, "the engine refused to start crafting"))
  end
  inflight.start(request.request_id, "craft", {
    recipe = payload.recipe,
    goal = { item = product and product.name, count = started },
    baseline = baseline,
  })
  return respond(request, CODE.OK, {
    status = STATUS.RUNNING,
    action = "craft",
    recipe = payload.recipe,
    started = started,
    energy = prototypes.recipe[payload.recipe].energy,
  })
end

H.place = function(_, request, payload, respond, err)
  local ch = character()
  if not ch then
    return respond(request, CODE.REJECTED, nil, err(ERR.PRECONDITION, "no character present"))
  end
  local inv = ch.get_inventory(defines.inventory.character_main)
  if not inv or inv.get_item_count(payload.item) < 1 then
    return respond(request, CODE.REJECTED, nil,
      err(ERR.NO_ITEMS, "not holding " .. payload.item))
  end
  local item_proto = prototypes.item[payload.item]
  if not item_proto or not item_proto.place_result then
    return respond(request, CODE.REJECTED, nil,
      err(ERR.INVALID_TARGET, payload.item .. " does not place an entity"))
  end
  -- Base Factorio gates technology at the recipe layer, not at placement, so
  -- holding an item is normally enough. The action profile declares a stricter
  -- rule so PLAN 2.1's "placement cannot bypass technology restrictions" is
  -- directly testable rather than argued transitively.
  local recipe = ch.force.recipes[payload.item]
  if recipe and not recipe.enabled then
    return respond(request, CODE.REJECTED, nil,
      err(ERR.TECH_LOCKED, payload.item .. " is not unlocked for this force"))
  end
  local position = { payload.position[1], payload.position[2] }
  if distance(ch.position, { x = position[1], y = position[2] }) > ch.build_distance then
    return respond(request, CODE.REJECTED, nil,
      err(ERR.OUT_OF_REACH, "position is beyond build distance",
        { reach = ch.build_distance }))
  end
  local surface = ch.surface
  local spec = {
    name = item_proto.place_result.name,
    position = position,
    direction = DIRECTIONS[payload.direction],
    force = ch.force,
    build_check_type = defines.build_check_type.manual,
  }
  if not surface.can_place_entity(spec) then
    return respond(request, CODE.REJECTED, nil,
      err(ERR.COLLISION, "cannot place " .. payload.item .. " there",
        { position = position }))
  end
  local created = surface.create_entity({
    name = item_proto.place_result.name,
    position = position,
    direction = DIRECTIONS[payload.direction],
    force = ch.force,
  })
  if not created then
    return respond(request, CODE.REJECTED, nil,
      err(ERR.ENGINE, "create_entity returned nothing"))
  end
  local removed = inv.remove({ name = payload.item, count = 1 })
  if removed < 1 then
    -- Never leave a built entity that was not paid for.
    created.destroy({ raise_destroy = true })
    return respond(request, CODE.REJECTED, nil,
      err(ERR.NO_ITEMS, "item vanished before it could be debited"))
  end
  return respond(request, CODE.OK, {
    status = STATUS.COMPLETED,
    action = "place",
    entity = created.name,
    handle = handles.mint(created),
    position = { created.position.x, created.position.y },
  })
end

H.rotate = function(_, request, payload, respond, err)
  local ch = character()
  if not ch then
    return respond(request, CODE.REJECTED, nil, err(ERR.PRECONDITION, "no character present"))
  end
  local target, reason = handles.resolve(payload.handle)
  if not target then
    return respond(request, CODE.REJECTED, nil,
      err(reason == "unknown_handle" and ERR.UNKNOWN_HANDLE or ERR.TARGET_MISSING,
        "handle " .. payload.handle .. ": " .. reason))
  end
  local ok, details = in_reach(ch, target, matrix.REACH.ENTITY)
  if not ok then
    return respond(request, CODE.REJECTED, nil,
      err(ERR.OUT_OF_REACH, "target is out of reach", details))
  end
  local before = target.direction
  local rotated = target.rotate({ reverse = payload.reverse })
  if not rotated then
    return respond(request, CODE.REJECTED, nil,
      err(ERR.INVALID_TARGET, target.name .. " cannot be rotated"))
  end
  return respond(request, CODE.OK, {
    status = STATUS.COMPLETED,
    action = "rotate",
    from = before,
    to = target.direction,
  })
end

H.transfer = function(state, request, payload, respond, err)
  local ch = character()
  if not ch then
    return respond(request, CODE.REJECTED, nil, err(ERR.PRECONDITION, "no character present"))
  end
  if payload.from == payload.to then
    return respond(request, CODE.REJECTED, nil,
      err(ERR.PRECONDITION, "from and to must differ"))
  end
  local endpoints = {}
  for _, which in ipairs({ "from", "to" }) do
    local target, code, message = endpoint(payload[which])
    if not target then
      return respond(request, CODE.REJECTED, nil, err(code, message))
    end
    if not target.character then
      local ok, details = in_reach(ch, target.entity, matrix.REACH.ENTITY)
      if not ok then
        return respond(request, CODE.REJECTED, nil,
          err(ERR.OUT_OF_REACH, payload[which] .. " is out of reach", details))
      end
    end
    endpoints[which] = target
  end

  local from_inv = inventory_of(endpoints.from, payload.item, true)
  local to_inv = inventory_of(endpoints.to, payload.item, false)
  if not from_inv or not to_inv then
    return respond(request, CODE.REJECTED, nil,
      err(ERR.PRECONDITION, "an endpoint has no accessible inventory"))
  end
  local available = from_inv.get_item_count(payload.item)
  if available < payload.count then
    return respond(request, CODE.REJECTED, nil,
      err(ERR.NO_ITEMS, "source holds " .. available .. " of " .. payload.item,
        { item = payload.item, requested = payload.count, available = available }))
  end
  if not to_inv.can_insert({ name = payload.item, count = payload.count }) then
    return respond(request, CODE.REJECTED, nil,
      err(ERR.NO_SPACE, "destination cannot accept " .. payload.count))
  end
  -- All or nothing: probe, remove, insert, and put back what did not fit.
  local removed = from_inv.remove({ name = payload.item, count = payload.count })
  if removed < payload.count then
    from_inv.insert({ name = payload.item, count = removed })
    return respond(request, CODE.REJECTED, nil,
      err(ERR.NO_ITEMS, "source changed during transfer"))
  end
  local inserted = to_inv.insert({ name = payload.item, count = removed })
  if inserted < removed then
    from_inv.insert({ name = payload.item, count = removed - inserted })
    return respond(request, CODE.REJECTED, nil,
      err(ERR.NO_SPACE, "destination accepted only " .. inserted))
  end
  local task = storage.frrl_task
  task.transfers = task.transfers + 1
  task.items_moved = task.items_moved + inserted
  return respond(request, CODE.OK, {
    status = STATUS.COMPLETED,
    action = "transfer",
    item = payload.item,
    count = inserted,
    from = payload.from,
    to = payload.to,
  })
end

H.set_recipe = function(_, request, payload, respond, err)
  local ch = character()
  if not ch then
    return respond(request, CODE.REJECTED, nil, err(ERR.PRECONDITION, "no character present"))
  end
  local target, reason = handles.resolve(payload.handle)
  if not target then
    return respond(request, CODE.REJECTED, nil,
      err(reason == "unknown_handle" and ERR.UNKNOWN_HANDLE or ERR.TARGET_MISSING,
        "handle " .. payload.handle .. ": " .. reason))
  end
  local ok, details = in_reach(ch, target, matrix.REACH.ENTITY)
  if not ok then
    return respond(request, CODE.REJECTED, nil,
      err(ERR.OUT_OF_REACH, "target is out of reach", details))
  end
  if payload.recipe then
    local recipe = ch.force.recipes[payload.recipe]
    if not recipe then
      return respond(request, CODE.REJECTED, nil,
        err(ERR.RECIPE_UNAVAILABLE, "no such recipe: " .. payload.recipe))
    end
    if not recipe.enabled then
      return respond(request, CODE.REJECTED, nil,
        err(ERR.TECH_LOCKED, payload.recipe .. " is not unlocked for this force"))
    end
  end
  local set_ok, returned = pcall(function()
    return target.set_recipe(payload.recipe)
  end)
  if not set_ok then
    return respond(request, CODE.REJECTED, nil,
      err(ERR.INVALID_TARGET, target.name .. " does not take a recipe"))
  end
  local credited = {}
  if returned then
    local inv = ch.get_inventory(defines.inventory.character_main)
    for _, stack in pairs(returned) do
      local moved = inv and inv.insert(stack) or 0
      credited[#credited + 1] = { name = stack.name, count = moved }
    end
  end
  return respond(request, CODE.OK, {
    status = STATUS.COMPLETED,
    action = "set_recipe",
    recipe = payload.recipe,
    returned_items = credited,
  })
end

H.research = function(_, request, payload, respond, err)
  local ch = character()
  local force = ch and ch.force or game.forces["player"]
  if payload.cancel then
    force.cancel_current_research()
    return respond(request, CODE.OK, {
      status = STATUS.COMPLETED, action = "research", cancelled = true,
    })
  end
  if not payload.technology then
    return respond(request, CODE.REJECTED, nil,
      err(ERR.MISSING_FIELD, "payload.technology is required unless cancel is true"))
  end
  local tech = force.technologies[payload.technology]
  if not tech then
    return respond(request, CODE.REJECTED, nil,
      err(ERR.INVALID_TARGET, "no such technology: " .. payload.technology))
  end
  if tech.researched then
    return respond(request, CODE.REJECTED, nil,
      err(ERR.INVALID_TARGET, payload.technology .. " is already researched"))
  end
  -- Factorio 2.0 gates the root of the tech tree behind triggers rather than
  -- selection: 7 of 196 technologies complete by crafting or mining and cannot
  -- be queued at all. Reporting that as tech_locked would tell an agent to wait
  -- for prerequisites that will never arrive.
  if tech.prototype.research_trigger then
    local trigger = tech.prototype.research_trigger
    return respond(request, CODE.REJECTED, nil,
      err(ERR.TECH_NOT_SELECTABLE,
        payload.technology .. " completes from a trigger, not from selection",
        { trigger_type = trigger.type }))
  end
  if not force.add_research(payload.technology) then
    local unmet = {}
    for name, prereq in pairs(tech.prerequisites) do
      if not prereq.researched then unmet[#unmet + 1] = name end
    end
    return respond(request, CODE.REJECTED, nil,
      err(ERR.TECH_LOCKED, "cannot research " .. payload.technology,
        { unmet_prerequisites = unmet }))
  end
  return respond(request, CODE.OK, {
    status = STATUS.COMPLETED,
    action = "research",
    technology = payload.technology,
    current_research = force.current_research and force.current_research.name or nil,
  })
end

H.wait = function(_, request, _payload, respond, _err)
  return respond(request, CODE.OK, { status = STATUS.COMPLETED, action = "wait" })
end

H.cancel = function(_, request, payload, respond, err)
  local entry = inflight.get(payload.target_request_id)
  if not entry then
    return respond(request, CODE.REJECTED, nil,
      err(ERR.UNKNOWN_REQUEST_ID, "no such in-flight request: " .. payload.target_request_id))
  end
  if entry.terminal then
    return respond(request, CODE.REJECTED, nil,
      err(ERR.NOT_CANCELLABLE, "request already settled"))
  end
  local spec = matrix.ACTIONS[entry.action]
  if entry.action == "advance" or not (spec and spec.cancellable) then
    return respond(request, CODE.REJECTED, nil,
      err(ERR.NOT_CANCELLABLE, entry.action .. " cannot be cancelled"))
  end
  local result = inflight.cancel(entry)
  storage.frrl_cancelled = { request_id = entry.request_id, result = result }
  return respond(request, CODE.OK, {
    status = STATUS.COMPLETED,
    action = "cancel",
    target_request_id = payload.target_request_id,
    cancelled_action = entry.action,
    result = result,
  })
end

-- ---------------------------------------------------------------- dispatch

function actions.dispatch(state, request, respond, err)
  local name = request.payload and request.payload.action
  if not name then
    return respond(request, CODE.REJECTED, nil,
      err(ERR.MISSING_FIELD, "payload.action is required"))
  end
  local handler = H[name]
  if not handler or not matrix.ACTIONS[name] then
    return respond(request, CODE.REJECTED, nil,
      err(ERR.UNKNOWN_ACTION, "unknown action: " .. tostring(name)))
  end
  local code, message, details = matrix.validate(name, request.payload)
  if code then
    return respond(request, CODE.REJECTED, nil, err(code, message, details))
  end
  local payload = matrix.with_defaults(name, request.payload)
  return handler(state, request, payload, respond, err)
end

function actions.on_init()
  storage.frrl_task = { transfers = 0, items_moved = 0 }
end

function actions.reset_state()
  storage.frrl_task = { transfers = 0, items_moved = 0 }
  storage.frrl_superseded = nil
  storage.frrl_cancelled = nil
  local ch = character()
  if ch then
    ch.teleport({ 0, 0 })
    ch.walking_state = { walking = false }
    ch.mining_state = { mining = false }
    local queue = ch.crafting_queue
    if queue then
      for index = #queue, 1, -1 do
        ch.cancel_crafting({ index = index, count = queue[index].count })
      end
    end
    for _, which in ipairs({
      defines.inventory.character_main,
      defines.inventory.character_trash,
      defines.inventory.character_guns,
      defines.inventory.character_ammo,
      defines.inventory.character_armor,
    }) do
      local inv = ch.get_inventory(which)
      if inv then inv.clear() end
    end
    if ch.cursor_stack and ch.cursor_stack.valid_for_read then
      ch.cursor_stack.clear()
    end
    ch.health = ch.max_health
  end
end

function actions.task_state()
  return storage.frrl_task or { transfers = 0, items_moved = 0 }
end

return actions
