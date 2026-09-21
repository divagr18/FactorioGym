-- Known-terrain navigation (DESIGN.md 5.1).
--
-- Three decisions shape this file, and each of them is a decision *not* to
-- take an easier route that would have been wrong.
--
-- 1. The pathfinder is ours, over `memory`'s explored-terrain store, and the
--    engine's is not used at all. `surface.request_path` consults the true
--    map: it would happily route the agent around a wall it has never seen,
--    which fails DESIGN 5.1's first criterion silently and in exactly the way
--    that is hardest to notice from a successful-looking trace. (It is also
--    asynchronous -- it returns a handle and fires an event later -- so it
--    would not fit the in-flight registry without a second settle path.)
--
-- 2. Unexplored tiles are optimistically passable. The agent may plan straight
--    through ground it has never looked at, walk into whatever is there, and
--    replan. That is not a weakness of the planner; it is the behaviour PLAN
--    5.1's second criterion asks for, and the pessimistic alternative would
--    pin the agent inside its starting sensor region forever.
--
-- 3. Movement is real. The route is walked with `walking_state`, one cardinal
--    command per tick, settling over ticks through the same in-flight registry
--    as `move`. There is no `teleport` here and there must never be one. The
--    closest comparable project implements agent movement as a teleport plus a
--    synthetic tick counter; the throughput that buys is precisely the fidelity
--    this project exists to keep.
--
-- Navigation ends *next to* the target and does nothing else. It never opens a
-- chest, never transfers, never mines. "Cannot complete the target interaction
-- implicitly" is a property of this file having no such call in it.

local protocol = require("protocol")
local profiles = require("profiles")
local memory = require("memory")
local handles = require("handles")

local STATUS, ERR = protocol.STATUS, protocol.ERR

local navigation = {}

local DIRECTION = {
  north = defines.direction.north,
  east = defines.direction.east,
  south = defines.direction.south,
  west = defines.direction.west,
}

--- The route is 4-connected. Factorio 2.0 exposes sixteen directions and a
--- character will happily walk a diagonal, but an 8-connected grid needs a
--- corner-cutting rule to stay honest about collision, and the diagonal speed
--- behaviour is one more thing to be wrong about. Cardinal steps compose
--- exactly with the `move` action's own direction enum; 8-connected routing is
--- a later optimisation, not a correctness question.
local NEIGHBOURS = { { 1, 0 }, { -1, 0 }, { 0, 1 }, { 0, -1 } }

--- Movement granularity is the constraint every number below answers to. A
--- character runs 0.1484 tiles/tick, so:
---
---   * the walker steers every tick, which makes 0.1484 tiles the finest
---     stride available and bounds the terminal positioning error. The
---     oscillation recorded in `src/factoriorl/catalog.py` came from a
---     *30-tick* stride overshooting a tolerance smaller than itself; per-tick
---     control removes that failure mode rather than tuning around it.
---   * arriving "at" a tile means arriving within a tolerance, never exactly.
---     The minimum tolerance is 1.0 tile, derived below, and `navigate` does
---     not offer sub-tile positioning at all -- the primitive `move` nudge
---     exists for that and is the honest tool for it.
local APPROACH_MARGIN = 0.25

--- Why the minimum tolerance is 1.0 and not something tighter: the route ends
--- on a tile *centre*, and an arbitrary requested point can be a tile corner,
--- 0.7072 from the nearest centre. Add the 0.25 residual the per-tick walker
--- leaves and the smallest tolerance a centre-terminated route can honestly
--- promise is 0.958. Declaring 1.0 keeps the arithmetic visible.
navigation.MIN_TOLERANCE = 1.0

--- Intermediate waypoints are one tile apart, so the walker only needs to be
--- inside the right tile before switching to the next leg. This is larger than
--- one tick of travel, which is what stops the index from thrashing.
local WAYPOINT_TOLERANCE = 0.4

--- Ticks of commanded-but-motionless walking before the route is declared
--- blocked. Eight ticks is 0.13 s of game time: long enough that a single
--- oddly-ordered tick cannot fake a stall, short enough that the agent is not
--- billed a second of game time for an obstacle it has already hit.
local STALL_TICKS = 8

--- Distance below which a tick counts as "did not move".
local MOVED_EPSILON = 0.02

--- How often the walker re-observes terrain while under way. A 30-tick
--- decision interval covers ~4.45 tiles, so observing only when the runtime
--- takes an observation would let the character cross four tiles of unseen
--- ground between looks.
local REOBSERVE_TICKS = 15

--- Search bounds. Both exist because the plan runs to completion inside one
--- paused tick: an unbounded A* would stall the engine for as long as it took,
--- and a replanning walker would do it up to `replan_limit` times.
---
--- Measured off-engine (Lua 5.5 via lupa, this machine, not Factorio's 5.2):
--- an expansion costs about 9 us with the explored store fully populated, so
--- 12000 expansions is roughly 0.11 s of worst-case pause per plan. An open
--- 120-tile route costs 2.7 ms and a route through a fourteen-wall comb 68 ms.
--- Re-measure on the engine before raising either number.
---
--- The margin bounds the search box to the start-goal corridor plus 48 tiles,
--- which is about the same size as the expansion budget allows anyway. Running
--- out of either is reported as `search_budget`, which is a useful failure: it
--- tells the agent to pick a nearer waypoint rather than that the world is
--- impassable.
local MAX_EXPANSIONS = 12000
local SEARCH_MARGIN = 48

--- Legs reported on the wire. The full tile route can be hundreds of entries
--- and every one of them would be paid for in the settled response.
local MAX_REPORTED_LEGS = 32

local DEFAULT_RADIUS = 32

-- ------------------------------------------------------------- observation

--- The sensor radius currently in force, so terrain observed while walking
--- covers exactly the region an `observe` would have reported and not one tile
--- more. Falling back to 32 keeps navigation working if the runtime state is
--- not readable for any reason; it never widens the region beyond a declared
--- profile's radius.
function navigation.sensor_radius()
  local runtime_state = storage.frrl_state
  local profile = profiles.observation(runtime_state and runtime_state.observation_profile)
  return (profile and profile.radius) or DEFAULT_RADIUS
end

--- Take one terrain observation from where the character is standing.
function navigation.observe(ch)
  if not ch or not ch.valid then return false end
  return memory.observe_terrain(ch.surface, ch.position, navigation.sensor_radius())
end

-- ------------------------------------------------------------------ goals

local function point_to_box(px, py, box)
  local dx = math.max(box[1] - px, 0, px - box[3])
  local dy = math.max(box[2] - py, 0, py - box[4])
  return math.sqrt(dx * dx + dy * dy)
end

--- Distance from a point to the goal: to the entity's bounding box for a
--- handle target, to the declared point otherwise. Using the box rather than
--- the entity centre is what makes "reachable interaction position" mean the
--- same thing for a 1x1 chest and a 3x3 assembler.
local function goal_distance(goal, px, py)
  if goal.box then return point_to_box(px, py, goal.box) end
  local dx, dy = px - goal.x, py - goal.y
  return math.sqrt(dx * dx + dy * dy)
end

navigation.goal_distance = goal_distance

--- Build the goal descriptor for a validated payload.
--- Returns (goal, nil) or (nil, code, message, details).
function navigation.goal_for(ch, payload)
  local tolerance = payload.tolerance
  if payload.handle then
    local target, reason = handles.resolve(payload.handle)
    if not target then
      return nil,
        reason == "unknown_handle" and ERR.UNKNOWN_HANDLE or ERR.TARGET_MISSING,
        "handle " .. tostring(payload.handle) .. ": " .. tostring(reason)
    end
    local box = target.bounding_box
    if not box or not box.left_top or not box.right_bottom then
      return nil, ERR.INVALID_TARGET, target.name .. " has no bounding box to stand beside"
    end
    -- Default: comfortably inside the character's own reach, with two tiles of
    -- margin. The margin is not decoration -- `can_reach_entity` applies the
    -- engine's bounding-box rule and this planner applies an approximation of
    -- it, so stopping at the very edge of reach would produce routes that end
    -- one engine-side rounding away from being able to interact.
    local radius = tolerance or math.max(1.0, ch.reach_distance - 2.0)
    return {
      x = target.position.x,
      y = target.position.y,
      box = {
        box.left_top.x, box.left_top.y,
        box.right_bottom.x, box.right_bottom.y,
      },
      radius = radius,
      plan_radius = math.max(0.1, radius - APPROACH_MARGIN),
      handle = payload.handle,
      target_name = target.name,
    }, nil
  end

  local radius = tolerance or navigation.MIN_TOLERANCE
  return {
    x = payload.position[1],
    y = payload.position[2],
    radius = radius,
    plan_radius = math.max(0.1, radius - APPROACH_MARGIN),
  }, nil
end

--- Is the character close enough to be finished?
function navigation.arrived(ch, goal)
  return goal_distance(goal, ch.position.x, ch.position.y) <= goal.radius
end

-- -------------------------------------------------------------- the search

--- Numeric keys for the same reason `memory.lua` uses them: the closed set and
--- the cost map are touched four times per expansion, and building a string for
--- each was measurably the largest term in a long plan. Same offset and stride,
--- so the bound is the same: any tile on a Factorio map, exact in a double.
local KEY_OFFSET = 2097152
local KEY_STRIDE = 4194304

local function tile_key(x, y)
  return (x + KEY_OFFSET) * KEY_STRIDE + (y + KEY_OFFSET)
end

local function heap_push(heap, node)
  heap[#heap + 1] = node
  local index = #heap
  while index > 1 do
    local parent = math.floor(index / 2)
    if heap[parent].f <= heap[index].f then break end
    heap[parent], heap[index] = heap[index], heap[parent]
    index = parent
  end
end

local function heap_pop(heap)
  local size = #heap
  if size == 0 then return nil end
  local top = heap[1]
  heap[1] = heap[size]
  heap[size] = nil
  size = size - 1
  local index = 1
  while true do
    local left, right = index * 2, index * 2 + 1
    local smallest = index
    if left <= size and heap[left].f < heap[smallest].f then smallest = left end
    if right <= size and heap[right].f < heap[smallest].f then smallest = right end
    if smallest == index then break end
    heap[index], heap[smallest] = heap[smallest], heap[index]
    index = smallest
  end
  return top
end

--- A* over the explored-occupancy grid.
---
--- Returns (route, nil, nil) where route is an array of {x, y} tile centres
--- ending at the goal tile, or (nil, reason, details) with reason one of
--- "destination_blocked", "no_route", "search_budget".
function navigation.plan(start_x, start_y, goal)
  local sx, sy = math.floor(start_x), math.floor(start_y)

  local start_key = tile_key(sx, sy)
  local goal_tx, goal_ty = math.floor(goal.x), math.floor(goal.y)
  local lo_x = math.min(sx, goal_tx) - SEARCH_MARGIN
  local hi_x = math.max(sx, goal_tx) + SEARCH_MARGIN
  local lo_y = math.min(sy, goal_ty) - SEARCH_MARGIN
  local hi_y = math.max(sy, goal_ty) + SEARCH_MARGIN

  -- Manhattan is exact for unit-cost 4-connected movement; subtracting the
  -- goal radius keeps it admissible when the goal is a region rather than a
  -- point, so A* still returns a shortest route rather than a plausible one.
  local slack = goal.plan_radius * 1.5 + 1.0
  local function heuristic(x, y)
    local d = math.abs(x + 0.5 - goal.x) + math.abs(y + 0.5 - goal.y)
    return math.max(0, d - slack)
  end

  local function is_goal(x, y)
    return goal_distance(goal, x + 0.5, y + 0.5) <= goal.plan_radius
  end

  if is_goal(sx, sy) then
    return { { sx + 0.5, sy + 0.5 } }, nil, nil
  end

  local open = {}
  local cost_to = { [start_key] = 0 }
  local parent = {}
  local closed = {}
  -- The start goes in without a passability check, because the character's own
  -- tile is passable by definition: it is standing there. The store is one
  -- notch conservative about footprints, so a character pressed up against a
  -- machine can read as blocked, and checking here would make every route from
  -- that spot fail with "no route" when the way out is plainly open.
  heap_push(open, { x = sx, y = sy, f = heuristic(sx, sy) })

  local expansions = 0
  local best = nil
  local nearest_x, nearest_y, nearest_h = sx, sy, heuristic(sx, sy)

  while true do
    local node = heap_pop(open)
    if not node then break end
    local key = tile_key(node.x, node.y)
    if not closed[key] then
      closed[key] = true
      expansions = expansions + 1
      if is_goal(node.x, node.y) then
        best = node
        break
      end
      if expansions >= MAX_EXPANSIONS then
        return nil, "search_budget", { expansions = expansions }
      end
      local next_cost = cost_to[key] + 1
      for _, step in ipairs(NEIGHBOURS) do
        local nx, ny = node.x + step[1], node.y + step[2]
        if nx >= lo_x and nx <= hi_x and ny >= lo_y and ny <= hi_y then
          local nkey = tile_key(nx, ny)
          if not closed[nkey] and not memory.terrain_blocked(nx, ny) then
            local prior = cost_to[nkey]
            if prior == nil or next_cost < prior then
              cost_to[nkey] = next_cost
              parent[nkey] = { node.x, node.y }
              local h = heuristic(nx, ny)
              if h < nearest_h then
                nearest_h, nearest_x, nearest_y = h, nx, ny
              end
              heap_push(open, { x = nx, y = ny, f = next_cost + h })
            end
          end
        end
      end
    end
  end

  if not best then
    -- Two different things to tell an agent. "The tile you named has a wall on
    -- it" is a fixable mistake; "I cannot get there from here" is a fact about
    -- the world. Only a declared *position* can be blocked in the first sense:
    -- for an entity target, standing on the entity was never the plan, so its
    -- own tile being occupied says nothing and the answer is always no_route.
    local blocked_destination = goal.box == nil
      and memory.terrain_blocked(goal_tx, goal_ty)
    return nil, blocked_destination and "destination_blocked" or "no_route", {
      expansions = expansions,
      closest_tile = { nearest_x, nearest_y },
    }
  end

  local reversed = {}
  local cx, cy = best.x, best.y
  local guard = 0
  while true do
    reversed[#reversed + 1] = { cx + 0.5, cy + 0.5 }
    if cx == sx and cy == sy then break end
    local link = parent[tile_key(cx, cy)]
    if not link then break end
    cx, cy = link[1], link[2]
    guard = guard + 1
    if guard > MAX_EXPANSIONS then break end
  end

  local route = {}
  for index = #reversed, 1, -1 do
    route[#route + 1] = reversed[index]
  end
  -- Drop the tile the character is already standing in: walking back to its
  -- centre first would be motion in the wrong direction. Kept when it is the
  -- only element, because then it *is* the destination.
  if #route > 1 then table.remove(route, 1) end
  return route, nil, nil
end

--- The route collapsed to its corners, for the wire. Purely a report: the
--- walker steers on the full per-tile route so that replanning can test every
--- tile it is about to cross, not only the corners.
local function report_legs(route)
  local legs = {}
  for index = 1, #route do
    local keep = true
    if index > 1 and index < #route then
      local ax = route[index][1] - route[index - 1][1]
      local ay = route[index][2] - route[index - 1][2]
      local bx = route[index + 1][1] - route[index][1]
      local by = route[index + 1][2] - route[index][2]
      if math.abs(ax - bx) < 0.001 and math.abs(ay - by) < 0.001 then keep = false end
    end
    if keep then
      if #legs >= MAX_REPORTED_LEGS then return legs, true end
      legs[#legs + 1] = { route[index][1], route[index][2] }
    end
  end
  return legs, false
end

navigation.report_legs = report_legs

-- ------------------------------------------------------------- diagnostics

--- What is standing on this tile? Used only to name a blocker the character
--- has already walked into, so the lookup is one tile wide and next to the
--- body -- not a scan of the map.
---
--- Refuses an unexplored tile, and that refusal is the load-bearing part. A
--- failure message is agent-visible output, so naming what sits on a tile the
--- agent has never observed would smuggle the true map into the observation
--- through the error channel -- the same leak the engine pathfinder was
--- rejected for, arriving by a quieter door. An unreachable destination beyond
--- the sensor region therefore reports that there is no route and declines to
--- say why.
function navigation.blocker_at(ch, tx, ty)
  if not ch or not ch.valid then return nil end
  if not memory.terrain_explored(tx, ty) then return nil end
  -- The second half of the guard, and the exact one. `terrain_explored` answers
  -- at block granularity, so on its own it would permit naming something up to
  -- seven tiles outside anything the agent has looked at. This says: only what
  -- the character could see from where it is standing right now.
  local dx = (tx + 0.5) - ch.position.x
  local dy = (ty + 0.5) - ch.position.y
  if math.sqrt(dx * dx + dy * dy) > navigation.sensor_radius() then return nil end
  local surface = ch.surface
  local ok, found = pcall(surface.find_entities_filtered, {
    area = { { tx - 0.2, ty - 0.2 }, { tx + 1.2, ty + 1.2 } },
    limit = 16,
  })
  if ok and found then
    for _, entity in pairs(found) do
      if memory.blocks_character(ch, entity) then
        return {
          name = entity.name,
          type = entity.type,
          handle = handles.mint(entity),
          at = { entity.position.x, entity.position.y },
          tile = { tx, ty },
        }
      end
    end
  end
  local ok_tile, tile = pcall(surface.get_tile, tx, ty)
  if ok_tile and tile and memory.blocks_character_tile(ch, tile) then
    return { name = tile.name, type = "tile", at = { tx, ty }, tile = { tx, ty } }
  end
  return nil
end

-- ------------------------------------------------------------- the walker

local function heading_toward(ch, waypoint)
  local dx = waypoint[1] - ch.position.x
  local dy = waypoint[2] - ch.position.y
  if math.abs(dx) >= math.abs(dy) then
    return dx >= 0 and "east" or "west"
  end
  return dy >= 0 and "south" or "north"
end

--- Command the body. Written only when it differs from what the character is
--- already doing: `move` proves a single write persists across ticks, so
--- rewriting an unchanged state every tick would be work for nothing.
local function steer(ch, name)
  local walking = ch.walking_state
  if not walking.walking or walking.direction ~= DIRECTION[name] then
    ch.walking_state = { walking = true, direction = DIRECTION[name] }
  end
end

function navigation.stop(ch)
  if ch and ch.valid then
    ch.walking_state = { walking = false }
  end
end

--- The serialisable state one navigate carries. No functions, no LuaEntity
--- keys: this lives in `storage` and has to survive a save/load (inflight.lua).
function navigation.begin(ch, goal, route, payload)
  return {
    goal = goal,
    -- Taken from the goal, which `goal_for` already resolved, so the walker's
    -- notion of the target cannot disagree with the one the route was planned
    -- against.
    handle = goal.handle or payload.handle,
    route = route,
    index = 1,
    tiles_planned = #route,
    -- The matrix fills both of these in through `with_defaults`, so the
    -- fallbacks only matter to a direct caller; they are here so `begin` can
    -- never produce an entry with a nil deadline, which the poller reads as
    -- "already expired".
    deadline_tick = game.tick + (payload.max_ticks or 3600),
    replans = 0,
    replan_limit = payload.replan_limit or 8,
    observed_tick = game.tick,
    last_x = ch.position.x,
    last_y = ch.position.y,
    start_x = ch.position.x,
    start_y = ch.position.y,
    start_distance = goal_distance(goal, ch.position.x, ch.position.y),
    stall = 0,
    blocked_by = nil,
  }
end

--- Point the body at the first waypoint, so the opening tick already moves.
function navigation.launch(ch, d)
  local waypoint = d.route[d.index]
  if not waypoint then return end
  local name = heading_toward(ch, waypoint)
  steer(ch, name)
  d.heading = name
end

local function report(entry, ch, extra)
  local d = entry.data
  local out = {
    action = "navigate",
    destination = { d.goal.x, d.goal.y },
    ticks_walked = game.tick - entry.started_tick,
    tiles_planned = d.tiles_planned,
    waypoints_remaining = math.max(0, #d.route - d.index + 1),
    replans = d.replans,
  }
  if ch and ch.valid then
    out.position = { ch.position.x, ch.position.y }
    out.remaining = goal_distance(d.goal, ch.position.x, ch.position.y)
  end
  if d.handle then out.target = d.handle end
  if d.blocked_by then out.blocked_by = d.blocked_by end
  for key, value in pairs(extra or {}) do out[key] = value end
  return out
end

navigation.report = report

--- Re-observe and replan from where the character is now.
--- Returns (true, nil, nil) or (false, reason, details).
local function replan(ch, d)
  navigation.observe(ch)
  d.observed_tick = game.tick
  if d.replans >= d.replan_limit then
    return false, "replan_limit", { replans = d.replans }
  end
  d.replans = d.replans + 1
  local route, reason, details = navigation.plan(ch.position.x, ch.position.y, d.goal)
  if not route or #route == 0 then
    return false, reason or "no_route", details
  end
  d.route = route
  d.index = 1
  d.stall = 0
  d.tiles_planned = d.tiles_planned + #route
  return true, nil, nil
end

--- Does any tile the route still has to cross look blocked now?
local function route_blocked(d)
  for index = d.index, #d.route do
    local waypoint = d.route[index]
    local tx = math.floor(waypoint[1])
    local ty = math.floor(waypoint[2])
    if memory.terrain_blocked(tx, ty) then return tx, ty end
  end
  return nil
end

--- One tick of navigation. Returns (status, result, error_code) to settle, or
--- nil to keep running -- the same contract as every other poller.
function navigation.poll(entry)
  local ch = storage.frrl_character
  if not ch or not ch.valid then
    return STATUS.FAILED, { action = "navigate", reason = "character missing" },
      ERR.TARGET_MISSING
  end
  local d = entry.data
  if not d or not d.goal or not d.route then
    return STATUS.FAILED,
      { action = "navigate", reason = "navigation state missing" }, ERR.ENGINE
  end

  -- Arrival first, so a navigate that is already finished cannot spend another
  -- tick walking past its destination.
  if navigation.arrived(ch, d.goal) then
    navigation.stop(ch)
    local settled = report(entry, ch, { arrived = true })
    if d.handle then
      local target = handles.resolve(d.handle)
      -- The engine's own answer, reported rather than assumed: this planner
      -- approximates `can_reach_entity`, and the agent is entitled to know
      -- when the approximation and the engine disagree. Nothing is opened,
      -- taken or touched -- navigation ends beside the target and stops.
      settled.reachable = target ~= nil and ch.can_reach_entity(target) or false
    end
    return STATUS.COMPLETED, settled, nil
  end

  if game.tick >= (d.deadline_tick or 0) then
    navigation.stop(ch)
    return STATUS.FAILED,
      report(entry, ch, { arrived = false, reason = "time_budget" }),
      ERR.PRECONDITION
  end

  -- Did the last tick actually move the body?
  local moved = math.abs(ch.position.x - d.last_x) + math.abs(ch.position.y - d.last_y)
  if moved < MOVED_EPSILON then
    d.stall = d.stall + 1
  else
    d.stall = 0
  end
  d.last_x, d.last_y = ch.position.x, ch.position.y

  -- Periodic look-ahead: discover obstacles before walking into them where we
  -- can, which is what turns "replanning or a useful failure" into replanning
  -- most of the time.
  if game.tick - (d.observed_tick or 0) >= REOBSERVE_TICKS then
    navigation.observe(ch)
    d.observed_tick = game.tick
    local tx, ty = route_blocked(d)
    if tx then
      d.blocked_by = navigation.blocker_at(ch, tx, ty) or { name = "unknown", tile = { tx, ty } }
      local ok, reason, details = replan(ch, d)
      if not ok then
        navigation.stop(ch)
        return STATUS.FAILED,
          report(entry, ch, { arrived = false, reason = reason, details = details }),
          ERR.COLLISION
      end
    end
  end

  if d.handle then
    local target, why = handles.resolve(d.handle)
    if not target then
      -- The thing we were walking to is gone. DESIGN 2.2: a target becoming
      -- unavailable is a recorded failure, not a silent stop.
      navigation.stop(ch)
      return STATUS.FAILED,
        report(entry, ch, { arrived = false, reason = "target_gone" }),
        why == "unknown_handle" and ERR.UNKNOWN_HANDLE or ERR.TARGET_MISSING
    end
  end

  -- Advance past every waypoint already reached.
  while d.index < #d.route do
    local waypoint = d.route[d.index]
    local dx = math.abs(waypoint[1] - ch.position.x)
    local dy = math.abs(waypoint[2] - ch.position.y)
    if math.max(dx, dy) <= WAYPOINT_TOLERANCE then
      d.index = d.index + 1
    else
      break
    end
  end

  local waypoint = d.route[d.index]
  if not waypoint then
    -- The route ran out without the arrival test passing. Replanning from here
    -- is the honest response; the alternative is reporting an arrival that did
    -- not happen.
    local ok, reason, details = replan(ch, d)
    if not ok then
      navigation.stop(ch)
      return STATUS.FAILED,
        report(entry, ch, { arrived = false, reason = reason, details = details }),
        ERR.COLLISION
    end
    waypoint = d.route[d.index]
  end

  local name = heading_toward(ch, waypoint)

  if d.stall >= STALL_TICKS then
    -- Commanded to walk and not moving: something is there that the store did
    -- not know about. Look, name it, and try to route around it.
    navigation.observe(ch)
    d.observed_tick = game.tick
    local tx = math.floor(waypoint[1])
    local ty = math.floor(waypoint[2])
    d.blocked_by = navigation.blocker_at(ch, tx, ty)
      or navigation.blocker_at(ch,
        math.floor(ch.position.x + (name == "east" and 0.8 or name == "west" and -0.8 or 0)),
        math.floor(ch.position.y + (name == "south" and 0.8 or name == "north" and -0.8 or 0)))
      or { name = "unknown", type = "unknown", tile = { tx, ty } }
    local ok, reason, details = replan(ch, d)
    if not ok then
      navigation.stop(ch)
      -- The useful failure: what stopped us, and where. "failed" alone would
      -- leave an agent with no next action to take; a named entity and a tile
      -- can be mined, deconstructed, or routed around by hand.
      return STATUS.FAILED,
        report(entry, ch, { arrived = false, reason = "blocked", details = details }),
        ERR.COLLISION
    end
    waypoint = d.route[d.index]
    name = heading_toward(ch, waypoint)
    d.stall = 0
  end

  steer(ch, name)
  d.heading = name
  return nil
end

return navigation
