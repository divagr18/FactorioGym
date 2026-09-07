-- Remembered observations with age (PLAN.md 2.3).
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

local memory = {}

local MEMORY_LIMIT = 2048

local function state()
  return storage.frrl_memory
end

function memory.reset()
  storage.frrl_memory = { entries = {}, count = 0 }
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

--- Fold this sweep's visible entities into memory and return what is only
--- remembered.
function memory.update(visible, origin, radius)
  local s = state()
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
