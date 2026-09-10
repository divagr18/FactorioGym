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
local navigation = require("navigation")
local profiles = require("profiles")
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

--- Take exclusive control of the character's body.
---
-- A new move supersedes a running one: a policy emitting a direction every
-- step must not have to spend an action cancelling first. The superseded
-- entry settles as `cancelled` exactly once.
--
-- Moving also interrupts mining, because the two are physically exclusive
-- for a character -- walking away from a rock stops mining it. Without this
-- the mine poller keeps re-asserting mining_state every tick and pins the
-- character in place, so a move would be accepted and then silently do
-- nothing.
--
-- `navigate` shares the "move" slot (see inflight.lua), so this one helper
-- gives both actions the same rule: whoever last asked to use the body has it,
-- and the loser settles as cancelled with its partial result intact.
local function take_over_body()
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
  local ids = {}
  for _, item in ipairs(superseded) do ids[#ids + 1] = item.request_id end
  return ids
end

H.move = function(_, request, payload, respond, err)
  local ch = character()
  if not ch then
    return respond(request, CODE.REJECTED, nil, err(ERR.PRECONDITION, "no character present"))
  end
  local superseded = take_over_body()
  ch.walking_state = { walking = true, direction = DIRECTIONS[payload.direction] }
  inflight.start(request.request_id, "move", {
    deadline_tick = game.tick + payload.ticks,
  })
  return respond(request, CODE.OK, {
    status = STATUS.RUNNING,
    action = "move",
    direction = payload.direction,
    ticks = payload.ticks,
    superseded = superseded,
  })
end

--- Known-terrain navigation (PLAN.md 5.1).
---
-- The route is planned here, at the paused tick the request is handled, so a
-- rejection carries no partial change: nothing has been touched and the body
-- has not been taken over. Everything after the plan succeeds is the ordinary
-- ongoing-action path -- one in-flight entry, settled exactly once by the
-- poller in `navigation.lua`.
--
-- What this handler does *not* do is the point of it. It does not teleport. It
-- does not ask the engine for a path (that would consult the true map and
-- route the agent around obstacles it has never seen). And when the target is
-- an entity it does not open, take from, rotate or mine it: it walks to a
-- position from which the agent could choose to, and stops.
H.navigate = function(_, request, payload, respond, err)
  local ch = character()
  if not ch then
    return respond(request, CODE.REJECTED, nil, err(ERR.PRECONDITION, "no character present"))
  end
  -- `matrix.validate` checks field types; "exactly one of these two" is a
  -- cross-field rule it cannot express, so it lives here.
  if (payload.position == nil) == (payload.handle == nil) then
    return respond(request, CODE.REJECTED, nil,
      err(ERR.MISSING_FIELD,
        "exactly one of payload.position or payload.handle is required"))
  end

  local goal, code, message, details = navigation.goal_for(ch, payload)
  if not goal then
    return respond(request, CODE.REJECTED, nil, err(code, message, details))
  end

  -- Look before planning. The store is only ever written from an observation,
  -- so this is the step that makes "routes use only known terrain" mean
  -- "terrain this character has stood close enough to see".
  navigation.observe(ch)

  if navigation.arrived(ch, goal) then
    -- Already within tolerance. Zero ticks is the correct amount of game time
    -- for zero distance; inventing an in-flight entry that settles next tick
    -- would bill the agent for movement that did not happen.
    return respond(request, CODE.OK, {
      status = STATUS.COMPLETED,
      action = "navigate",
      arrived = true,
      ticks_walked = 0,
      route_length = 0,
      position = { ch.position.x, ch.position.y },
      destination = { goal.x, goal.y },
      target = payload.handle,
      tolerance = goal.radius,
    })
  end

  local route, reason, plan_details = navigation.plan(ch.position.x, ch.position.y, goal)
  if not route then
    -- Naming the obstacle is only possible where the agent has looked;
    -- `blocker_at` refuses an unexplored tile, so an unreachable destination
    -- beyond the sensor region reports "no_route" without a name rather than
    -- reading the true map to produce one.
    local blocker = navigation.blocker_at(ch, math.floor(goal.x), math.floor(goal.y))
    return respond(request, CODE.REJECTED, nil,
      err(reason == "search_budget" and ERR.PRECONDITION or ERR.COLLISION,
        "no route over known terrain (" .. reason .. ")",
        {
          reason = reason,
          blocked_by = blocker,
          destination = { goal.x, goal.y },
          search = plan_details,
        }))
  end

  local superseded = take_over_body()
  local data = navigation.begin(ch, goal, route, payload)
  inflight.start(request.request_id, "navigate", {
    deadline_tick = data.deadline_tick,
    target_handle = payload.handle,
    data = data,
  })
  -- Command the body now rather than on the first poll, so the opening tick of
  -- the interval is spent walking instead of deciding to walk.
  navigation.launch(ch, data)

  local legs, legs_truncated = navigation.report_legs(route)
  return respond(request, CODE.OK, {
    status = STATUS.RUNNING,
    action = "navigate",
    destination = { goal.x, goal.y },
    target = payload.handle,
    tolerance = goal.radius,
    route_length = #route,
    legs = legs,
    legs_truncated = legs_truncated,
    max_ticks = payload.max_ticks,
    replan_limit = payload.replan_limit,
    superseded = superseded,
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
  -- A running navigation drives `walking_state` every tick and the mine poller
  -- re-asserts `mining_state` every tick; run together they pin the character
  -- in place and the navigator reports the stall as an obstacle that is not
  -- there. Rejecting is the honest outcome. This is reachable only under an
  -- assistance profile -- `navigate` cannot exist otherwise -- so `mine`
  -- behaves bit-for-bit as before under `primitive-v1`, and `busy` was already
  -- one of its declared failures.
  local body = inflight.occupant("move")
  if body then
    local occupant = inflight.get(body)
    if occupant and occupant.action == "navigate" then
      return respond(request, CODE.REJECTED, nil,
        err(ERR.BUSY, "a navigation is running; cancel it or issue a move to take control"))
    end
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
  -- Count what the agent constructed. The engine's own build statistics are
  -- not fed by `create_entity`, so this is the only channel a `BUILT`
  -- predicate can read -- and it is the right one: it counts placements made
  -- through this action, so a scene's install path can never contribute.
  -- Counted only after the item is debited, so a rolled-back placement does
  -- not count as construction.
  storage.frrl_built = storage.frrl_built or {}
  storage.frrl_built[created.name] = (storage.frrl_built[created.name] or 0) + 1
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
  if available <= 0 then
    return respond(request, CODE.REJECTED, nil,
      err(ERR.NO_ITEMS, "source holds no " .. payload.item,
        { item = payload.item, requested = payload.count, available = available }))
  end
  -- Take what is there, up to what was asked for.
  --
  -- This used to refuse outright whenever `available < count`, and that rule and
  -- the argument domain are each defensible on their own and broken together:
  -- `TRANSFER_AMOUNTS` offers 1, 5 and 20, so the exact number of plates in a
  -- furnace is usually not a number the agent is allowed to say. A run watched
  -- three plates accumulate, asked for 3, was told 3 is not a legal value, asked
  -- for 5, was told the source holds 3, and settled for 1 -- three decisions and
  -- two model calls to collect what one ctrl-click gives a player, who gets
  -- exactly this clamping behaviour from the real game.
  --
  -- Atomicity, which is what the "all or nothing" below was really protecting,
  -- is untouched: the removal is still put back if the destination will not take
  -- it, so no item is ever destroyed by a partial move.
  local wanted = math.min(payload.count, available)
  if not to_inv.can_insert({ name = payload.item, count = wanted }) then
    return respond(request, CODE.REJECTED, nil,
      err(ERR.NO_SPACE, "destination cannot accept " .. wanted))
  end
  local removed = from_inv.remove({ name = payload.item, count = wanted })
  if removed <= 0 then
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
    -- Published whenever the move was smaller than the ask, so "I asked for 20
    -- and got 3" is a fact in the outcome rather than something the agent has
    -- to infer from its inventory two turns later.
    requested = payload.count ~= inserted and payload.count or nil,
    available = payload.count ~= inserted and available or nil,
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

--- PLAN 5.2: a bounded sequence of typed interactions.
--
-- Execution stops at the first failure and the response names both halves: the
-- operations that completed and the index that did not. An agent that only
-- learns "the batch failed" has to re-derive the world state by observing,
-- which costs it the round trip the batch was meant to save.
--
-- Sub-operations run through the ordinary handler with a capturing `respond`,
-- so validation, reach and every failure code are exactly the ones a
-- standalone request would produce. Nothing about an operation's rules is
-- restated here, which is what keeps a batch from drifting away from the
-- actions it is made of.
H.batch = function(state, request, payload, respond, err)
  local operations = payload.operations or {}
  local limit = math.min(payload.max_operations or matrix.BATCH_LIMIT, matrix.BATCH_LIMIT)
  local completed = {}

  for index, operation in ipairs(operations) do
    if index > limit then
      return respond(request, CODE.OK, {
        action = "batch",
        status = "completed",
        requested = #operations,
        executed = #completed,
        stopped_at = index,
        stopped_reason = "max_operations",
        operations = completed,
      })
    end

    local name = operation.action
    local spec = matrix.ACTIONS[name]
    local function fail(code, message, details)
      return respond(request, CODE.REJECTED, {
        action = "batch",
        status = "rejected",
        requested = #operations,
        executed = #completed,
        failed_at = index,
        failed_action = name,
        operations = completed,
      }, err(code, message, details))
    end

    if not spec or not H[name] then
      return fail(ERR.UNKNOWN_ACTION, "unknown action in batch: " .. tostring(name),
        { index = index })
    end
    if not profiles.permits(profiles.action(state and state.action_profile), name) then
      return fail(ERR.UNKNOWN_ACTION, "action not in this profile: " .. tostring(name),
        { index = index })
    end
    if spec.ongoing then
      return fail(ERR.PRECONDITION,
        tostring(name) .. " is an ongoing action and cannot be batched",
        { index = index, action = name })
    end
    if name == "batch" then
      return fail(ERR.PRECONDITION, "a batch cannot contain a batch", { index = index })
    end

    local code, message, details = matrix.validate(name, operation)
    if code then
      return fail(code, message, details)
    end

    local captured = nil
    local function capture(_request, response_code, result, error)
      captured = { code = response_code, result = result, error = error }
    end
    H[name](state, request, matrix.with_defaults(name, operation), capture, err)

    if not captured then
      return fail(ERR.ENGINE, tostring(name) .. " produced no response inside a batch",
        { index = index })
    end
    if captured.code ~= CODE.OK then
      local error = captured.error or {}
      return respond(request, CODE.REJECTED, {
        action = "batch",
        status = "rejected",
        requested = #operations,
        executed = #completed,
        failed_at = index,
        failed_action = name,
        operations = completed,
      }, err(error.code or ERR.PRECONDITION, error.message or "operation failed",
        { index = index, action = name, details = error.details }))
    end
    completed[#completed + 1] = {
      index = index,
      action = name,
      status = "completed",
      result = captured.result,
    }
  end

  return respond(request, CODE.OK, {
    action = "batch",
    status = "completed",
    requested = #operations,
    executed = #completed,
    operations = completed,
  })
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
  -- The action profile is a capability boundary, not a label. An assisted
  -- action sent to a worker running `primitive-v1` is rejected as unknown --
  -- the same answer as a made-up action name, because from that profile's
  -- point of view it is one. Without this check the assisted catalog would be
  -- reachable from any run, and no Phase 3 or Phase 4 number could claim to
  -- have been measured without navigation assistance.
  local action_profile = profiles.action(state and state.action_profile)
  if not profiles.permits(action_profile, name) then
    return respond(request, CODE.REJECTED, nil,
      err(ERR.UNKNOWN_ACTION,
        "unknown action: " .. tostring(name),
        { action_profile = action_profile and action_profile.name or nil }))
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
