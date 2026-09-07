-- Remembered observations with age (PLAN.md 2.3), and the explored-terrain
-- store that known-terrain navigation plans over (PLAN.md 5.1, and the first
-- piece of 5.4).
--
-- ------------------------------------------------------------ remembered
--
-- One update rule does all the work, and it is the whole of "moving into and
-- out of visibility updates records correctly":
--
--   * every entity in the region this sweep is written with last_seen = tick;
--   * entities in memory but *outside* the region are reported under
--     `remembered`, carrying their age;
--   * entities in memory that are *inside* the region but absent from the
--     sweep are deleted -- the agent looked, and it was not there.
--
-- Remembered records carry their last-seen contents only inside the
-- `remembered` block, never in `entities`. PLAN.md section 2 forbids
-- presenting distant machine state as current, and putting the two in
-- different arrays enforces that structurally rather than by a caveat.
--
-- --------------------------------------------------------------- terrain
--
-- The terrain store answers exactly one question for the navigator: "can the
-- character's centre sit on this tile, and did the agent ever look?" It is
-- deliberately a *separate* store from the entity memory above, because 5.4
-- will grow it (action outcomes, plans) and because navigation must never be
-- able to read the entity memory's positions as though they were occupancy.
--
-- Three properties are load-bearing:
--
--   1. It is only ever written from a sensor-region observation. Nothing here
--      queries a tile the agent was not standing close enough to see, which is
--      what makes PLAN 5.1's "routes use only known terrain" true by
--      construction rather than by inspection of the planner.
--   2. "Blocked" is decided by intersecting the *character's own* collision
--      mask with each candidate's, read off the prototypes at runtime. There
--      is no hand-written list of blocking entity types, which is the mistake
--      that previously recorded transport belts as obstacles: belts do not
--      collide with the player layer and a character walks straight over them.
--      Getting that wrong makes a perfectly good route unplannable.
--   3. Storage is bounded (see TERRAIN_BLOCK_LIMIT) and forgetting always
--      moves a tile from "known blocked" to "unknown", never the other way.
--      Unknown is optimistically passable, so the worst case of eviction is
--      that the agent walks into a rediscovered obstacle and replans -- the
--      mechanism PLAN 5.1's second criterion already requires -- rather than
--      believing a route is clear when the store says otherwise.
--
-- Occupancy is stored per 8x8 block, not per tile. A radius-32 observation
-- covers 4225 tiles; writing one persistent table entry per tile per step
-- would be the single most expensive thing in the observation path. Only
-- blocked tiles are stored, so a block holds a handful of entries on open
-- ground and the clearing pass walks those rather than the block's 64 tiles.
-- Clearing is nevertheless decided per *tile*, against the observed square:
-- clearing whole blocks instead would either wipe obstacles in an unobserved
-- half or, if restricted to fully covered blocks, record nothing at all
-- whenever the sensor radius is under about twelve tiles.

local profiles = require("profiles")

local memory = {}

local MEMORY_LIMIT = 2048

--- 8x8 tiles per block: small enough that a radius-32 observation touches 81
--- of them, and large enough that the per-observation bookkeeping is 81 block
--- touches instead of 4225 tile writes. Blocks are a storage grouping only --
--- clearing is per tile -- so this number is a cost knob, not a semantics one.
local BLOCK = 8

--- 2048 blocks is 131072 tiles, a 362x362 region -- far more than one episode
--- explores. See the header for why exceeding it is safe.
local TERRAIN_BLOCK_LIMIT = 2048

local function state()
  return storage.frrl_memory
end

--- Lazily constructed as well as reset, so a save written by an earlier mod
--- version that has not yet reached `begin_episode` cannot nil-index here and
--- take the worker down on its first observation.
local function terrain()
  local t = storage.frrl_terrain
  if not t then
    t = { blocks = {}, order = {}, count = 0 }
    storage.frrl_terrain = t
  end
  return t
end

function memory.reset()
  storage.frrl_memory = { entries = {}, count = 0 }
  storage.frrl_terrain = { blocks = {}, order = {}, count = 0 }
end

local function evict_if_needed(s)
  if s.count <= MEMORY_LIMIT then return end
  local oldest_handle, oldest_tick = nil, nil
  for handle, entry in pairs(s.entries) do
    if not oldest_tick or entry.last_seen < oldest_tick then
      oldest_handle, oldest_tick = handle, entry.last_seen
    end
  end
  if oldest_handle then
    s.entries[oldest_handle] = nil
    s.count = s.count - 1
  end
end

-- ------------------------------------------------------------- collision

--- The collision layers the character itself collides on, read from its own
--- prototype. Hard-coding "player" here would be a second declaration of
--- something the engine already owns, and would silently stop matching if the
--- character prototype ever changed.
function memory.character_collision_layers(ch)
  if not ch or not ch.valid then return nil end
  local prototype = ch.prototype
  local mask = prototype and prototype.collision_mask
  return mask and mask.layers or nil
end

--- Half the character's collision box, so an obstacle's footprint can be
--- inflated by it: the question a route asks is not "does this tile contain an
--- obstacle" but "can the character's *centre* stand at this tile's centre".
--- Read from the prototype (nominally 0.2) rather than written down, for the
--- same reason as the mask above.
local function character_half_extent(ch)
  local prototype = ch.prototype
  local box = prototype and prototype.collision_box
  if not box or not box.left_top or not box.right_bottom then return 0.2 end
  return math.max(
    math.abs(box.left_top.x), math.abs(box.left_top.y),
    math.abs(box.right_bottom.x), math.abs(box.right_bottom.y)
  )
end

--- Does `entity` block this character? Pure mask intersection, which is the
--- engine's own rule.
function memory.blocks_character(ch, entity)
  if not entity or not entity.valid then return false end
  -- A character is a body, not terrain. The agent's own character is returned
  -- by any area query that covers it, and routing around itself would make
  -- every plan fail at step one.
  if entity.type == "character" then return false end
  local layers = memory.character_collision_layers(ch)
  if not layers then return false end
  local prototype = entity.prototype
  local mask = prototype and prototype.collision_mask
  if not mask or not mask.layers then return false end
  for layer in pairs(mask.layers) do
    if layers[layer] then return true end
  end
  return false
end

--- Does `tile` block this character? Same rule, applied to a tile prototype.
function memory.blocks_character_tile(ch, tile)
  if not tile then return false end
  local layers = memory.character_collision_layers(ch)
  if not layers then return false end
  local ok, mask = pcall(function() return tile.prototype.collision_mask end)
  if not ok or not mask or not mask.layers then return false end
  for layer in pairs(mask.layers) do
    if layers[layer] then return true end
  end
  return false
end

-- --------------------------------------------------------------- terrain

--- Block keys are numbers, not "bx:by" strings, and that is a measured choice
--- rather than a stylistic one: `terrain_blocked` is called four times per A*
--- expansion, and string concatenation plus string hashing on that path was
--- the single largest term in a long plan (a fourteen-wall detour went from
--- 157 ms to 68 ms off-engine).
---
--- The offset shifts negative block coordinates into range and the stride is
--- wide enough that they cannot collide anywhere on a Factorio map: it covers
--- block coordinates +/-2097152, or +/-16.7 million tiles against a map edge
--- at one million. The largest key is about 1.8e13, exact in a double, so
--- Lua 5.2's lack of an integer type costs nothing here.
local KEY_OFFSET = 2097152
local KEY_STRIDE = 4194304

local function block_key(bx, by)
  return (bx + KEY_OFFSET) * KEY_STRIDE + (by + KEY_OFFSET)
end

--- Fold one sensor-region observation into the terrain store.
---
--- Two engine queries, both bounded to the observed square, both filtered by
--- the character's own collision mask so the engine does the selecting. The
--- entity results are re-checked in Lua with the same intersection rule: if
--- the mask filter is ever more permissive than assumed the recheck removes
--- the surplus, and if it is ever less permissive the store is merely
--- optimistic, which degrades to "walk into it, discover it, replan" rather
--- than to a false claim that a tile is clear.
---
--- Returns true when the store was updated.
function memory.observe_terrain(surface, origin, radius)
  local ch = storage.frrl_character
  if not surface or not ch or not ch.valid then return false end
  local layers = memory.character_collision_layers(ch)
  if not layers then return false end
  local half = character_half_extent(ch)

  -- Tiles fully inside the observed square. Tile (tx, ty) covers
  -- [tx, tx+1] x [ty, ty+1], so it is fully inside iff tx >= ceil(min) and
  -- tx <= floor(max) - 1.
  local minx = math.ceil(origin.x - radius)
  local maxx = math.floor(origin.x + radius) - 1
  local miny = math.ceil(origin.y - radius)
  local maxy = math.floor(origin.y + radius) - 1
  if maxx < minx or maxy < miny then return false end

  -- Every block the observed square *touches*, not only the ones it covers
  -- entirely. Requiring full coverage looks tidier and is a trap: it makes the
  -- store silently record nothing whenever the sensor radius is smaller than
  -- about a block and a half, and the failure mode is a navigator that
  -- replans into the same obstacle until its budget runs out with no idea why.
  -- The tidiness is recovered by clearing per tile below instead of per block.
  local bx0 = math.floor(minx / BLOCK)
  local bx1 = math.floor(maxx / BLOCK)
  local by0 = math.floor(miny / BLOCK)
  local by1 = math.floor(maxy / BLOCK)

  local t = terrain()
  local fresh = {}
  for bx = bx0, bx1 do
    for by = by0, by1 do
      local key = block_key(bx, by)
      local block = t.blocks[key]
      if not block then
        block = { bx = bx, by = by, blocked = {} }
        t.blocks[key] = block
        t.count = t.count + 1
        t.order[#t.order + 1] = key
      end
      block.seen = game.tick
      block.blocked = block.blocked or {}
      -- Clear only the tiles of this block that the agent just looked at, so a
      -- mined rock or a deconstructed chest stops being an obstacle while an
      -- obstacle in the unobserved half of the same block is preserved. The
      -- loop walks the recorded obstacles, which are sparse, rather than the
      -- 64 tiles of the block, which are not.
      local stale = nil
      for index in pairs(block.blocked) do
        local tx = bx * BLOCK + (index % BLOCK)
        local ty = by * BLOCK + math.floor(index / BLOCK)
        if tx >= minx and tx <= maxx and ty >= miny and ty <= maxy then
          stale = stale or {}
          stale[#stale + 1] = index
        end
      end
      if stale then
        for _, index in ipairs(stale) do block.blocked[index] = nil end
      end
      fresh[key] = block
    end
  end

  local function mark(tx, ty)
    local bx = math.floor(tx / BLOCK)
    local by = math.floor(ty / BLOCK)
    local block = fresh[block_key(bx, by)]
    if block then
      block.blocked[(ty - by * BLOCK) * BLOCK + (tx - bx * BLOCK)] = true
    end
  end

  local area = {
    { origin.x - radius, origin.y - radius },
    { origin.x + radius, origin.y + radius },
  }

  -- Blocking tiles. `sensor.lua` proves the collision_mask form of this query
  -- works on this build with a single layer name; passing the character's
  -- whole layer dictionary is the documented dictionary form of the same
  -- filter. The fallback is the exact query the sensor already ships, so a
  -- filter-shape surprise costs water detection only, not the whole store.
  local ok_tiles, tiles = pcall(surface.find_tiles_filtered, {
    area = area, collision_mask = layers,
  })
  if not ok_tiles or not tiles then
    ok_tiles, tiles = pcall(surface.find_tiles_filtered, {
      area = area, collision_mask = "water_tile",
    })
  end
  if ok_tiles and tiles then
    for _, tile in pairs(tiles) do
      -- LuaTile.position is the tile's integer position, not its centre.
      mark(tile.position.x, tile.position.y)
    end
  end

  -- Blocking entities. The footprint is the entity's own bounding box inflated
  -- by the character's half extent, and the test is on *tile centres*, because
  -- the route walks centre to centre.
  local ok_entities, found = pcall(surface.find_entities_filtered, {
    area = area, collision_mask = layers,
  })
  if not ok_entities or not found then
    ok_entities, found = pcall(surface.find_entities_filtered, { area = area })
  end
  if ok_entities and found then
    for _, entity in pairs(found) do
      if memory.blocks_character(ch, entity) then
        local box = entity.bounding_box
        if box and box.left_top and box.right_bottom then
          local x1 = box.left_top.x - half
          local y1 = box.left_top.y - half
          local x2 = box.right_bottom.x + half
          local y2 = box.right_bottom.y + half
          -- Tile centre tx+0.5 lies in [x1, x2] iff tx in
          -- [ceil(x1-0.5), floor(x2-0.5)]. Touching counts as blocked, which
          -- is one notch conservative: the engine does not collide on exact
          -- contact, but a route that grazes a machine is a route the walker
          -- snags on.
          for tx = math.ceil(x1 - 0.5), math.floor(x2 - 0.5) do
            for ty = math.ceil(y1 - 0.5), math.floor(y2 - 0.5) do
              mark(tx, ty)
            end
          end
        end
      end
    end
  end

  -- Bounded, oldest-first. A block refreshed by *this* observation is never
  -- evicted: dropping the ground under the character's feet would make the
  -- current route unplannable for no reason.
  local attempts = 0
  while t.count > TERRAIN_BLOCK_LIMIT and attempts < 4096 do
    attempts = attempts + 1
    local oldest = table.remove(t.order, 1)
    if not oldest then break end
    if fresh[oldest] then
      t.order[#t.order + 1] = oldest
    elseif t.blocks[oldest] ~= nil then
      t.blocks[oldest] = nil
      t.count = t.count - 1
    end
    -- A key whose block is already gone was a duplicate `order` entry left by
    -- the requeue above; dropping it is the whole fix.
  end
  return true
end

--- Is this tile known to be blocked?
---
--- An unexplored tile answers `false`. That is deliberate and it is the whole
--- of PLAN 5.1's second criterion: routes are allowed to run through the
--- unknown, discover an obstacle by walking into it, and replan. The
--- alternative -- refusing to route through unexplored ground -- would make
--- the agent unable to leave the region it starts in, and refusing to route
--- through it *because the true map says it is blocked* would be exactly the
--- privileged-information failure that rules out the engine pathfinder.
function memory.terrain_blocked(tx, ty)
  local t = terrain()
  local bx = math.floor(tx / BLOCK)
  local by = math.floor(ty / BLOCK)
  local block = t.blocks[block_key(bx, by)]
  if not block or not block.blocked then return false end
  return block.blocked[(ty - by * BLOCK) * BLOCK + (tx - bx * BLOCK)] == true
end

--- Has the agent observed any part of the block containing this tile?
---
--- Deliberately coarse, and every caller has to treat it that way: it is a
--- cheap "was the agent ever near here" filter, not a per-tile record of what
--- was looked at. `navigation.blocker_at` pairs it with a live distance check
--- against the sensor radius, because a coarse yes is not a licence to read
--- the map.
function memory.terrain_explored(tx, ty)
  local t = terrain()
  local block = t.blocks[block_key(math.floor(tx / BLOCK), math.floor(ty / BLOCK))]
  return block ~= nil
end

--- The tick the block containing this tile was last observed, or nil. Stale
--- terrain stays distinguishable from current terrain (PLAN 5.4).
function memory.terrain_seen_tick(tx, ty)
  local t = terrain()
  local block = t.blocks[block_key(math.floor(tx / BLOCK), math.floor(ty / BLOCK))]
  return block and block.seen or nil
end

--- Growth proxy: stored blocks, not tiles.
function memory.terrain_count()
  return terrain().count
end

-- ------------------------------------------------------------ remembered

--- Is terrain memory switched on for the profile currently in force?
---
--- Terrain folding costs two engine queries and ~49 block rewrites on every
--- observation. Only the assisted profile can navigate, so only the assisted
--- profile pays: a `primitive-v1` training run does exactly the work it did
--- before this file changed, which is what keeps the Phase 3 and Phase 4
--- measurements comparable with the ones already recorded.
local function terrain_enabled()
  local runtime_state = storage.frrl_state
  local action_profile = profiles.action(runtime_state and runtime_state.action_profile)
  return action_profile ~= nil and action_profile.terrain_memory == true
end

--- Fold this sweep's visible entities into memory and return what is only
--- remembered.
function memory.update(visible, origin, radius)
  local s = state()

  -- Terrain is folded from the same observation the entity records came from.
  -- Hanging it here rather than in `observations.lua` means the store is
  -- refreshed on exactly the ticks the agent looked, with no second place that
  -- has to remember to call it.
  if terrain_enabled() then
    local ch = storage.frrl_character
    if ch and ch.valid then
      memory.observe_terrain(ch.surface, origin, radius)
    end
  end

  local seen = {}
  for _, record in ipairs(visible) do
    if record.h then
      seen[record.h] = true
      if not s.entries[record.h] then s.count = s.count + 1 end
      s.entries[record.h] = {
        name = record.name,
        type = record.type,
        p = record.p,
        d = record.d,
        contents = record.contents,
        recipe = record.recipe,
        last_seen = game.tick,
      }
      evict_if_needed(s)
    end
  end

  local remembered = {}
  local stale = {}
  for handle, entry in pairs(s.entries) do
    if not seen[handle] then
      local dx = entry.p[1] - origin.x
      local dy = entry.p[2] - origin.y
      if math.sqrt(dx * dx + dy * dy) <= radius then
        -- Inside the sensor region and not in the sweep: it is gone.
        stale[#stale + 1] = handle
      else
        remembered[#remembered + 1] = {
          h = handle,
          name = entry.name,
          type = entry.type,
          p = entry.p,
          d = entry.d,
          contents = entry.contents,
          recipe = entry.recipe,
          age = game.tick - entry.last_seen,
          last_tick = entry.last_seen,
        }
      end
    end
  end
  for _, handle in ipairs(stale) do
    s.entries[handle] = nil
    s.count = s.count - 1
  end
  return remembered
end

function memory.count()
  return state().count
end

return memory
