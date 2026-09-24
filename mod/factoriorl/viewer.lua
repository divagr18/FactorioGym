-- An on-screen account of what the agent is doing, for a watched run.
--
-- A spectator sees the character walk and things appear, and nothing about
-- *why*: a refused placement looks like standing still, and a transfer into a
-- furnace looks like nothing at all. This draws each action the agent executes
-- as floating text above the character, keeps a small panel with the last few
-- actions and their outcomes, and keeps the camera on the character every tick
-- rather than every thirty.
--
-- Off unless a watch tool turns it on (`remote.call("frrl_viewer", "enable",
-- {...})`, sent by `factoriorl.agent.viewer.enable_overlay`). A measured run
-- never makes that call, so for one this module is a single `nil` check at the
-- dispatch point and nothing else: no tick handler is registered, nothing is
-- drawn, and `storage.frrl_viewer` does not exist.
--
-- It reads and never writes the world. Rendering objects and GUI elements are
-- not simulation: nothing an action, an observation or the evaluator reads can
-- see them. It also never resolves a handle through `handles.resolve`, which
-- marks a vanished entity destroyed as a side effect -- the registry is only
-- peeked at, so an overlay cannot change what the agent is told next.
--
-- All of its state is in `storage` and the tick handler is re-registered from
-- `on_load`, because every peer in a multiplayer game runs this script and a
-- client that joins mid-run must draw exactly what the server draws.

local protocol = require("protocol")

local CODE = protocol.CODE

local viewer = {}

--- Floating lines kept above the character at once. Older lines are pushed up
--- and dimmed; a fourth pushes the oldest off.
local FLOAT_LINES = 3
--- How long a floating line lasts. Ninety ticks is a second and a half at game
--- speed 1: long enough to read a short line, short enough not to pile up.
local FLOAT_TICKS = 90
--- Height above the character's position where the newest line sits, in tiles.
local FLOAT_BASE = 2.4
--- Rows in the panel's history.
local PANEL_ROWS = 10
--- How often the panel header (tick, decisions) refreshes without an action.
local HEADER_INTERVAL = 30

local COLOR = {
  ok = { 1.0, 1.0, 1.0, 1.0 },
  refused = { 1.0, 0.32, 0.28, 1.0 },
  quiet = { 0.62, 0.62, 0.62, 1.0 },
  line = { 0.35, 0.8, 1.0, 0.6 },
  line_refused = { 1.0, 0.32, 0.28, 0.6 },
}

--- Colours for panel rows. A label takes a colour table too, but naming them
--- keeps the history in storage small and readable.
local ROW_COLOR = {
  ok = { 0.72, 1.0, 0.62 },
  refused = { 1.0, 0.45, 0.4 },
  quiet = { 0.62, 0.62, 0.62 },
}

local PANEL_NAME = "frrl_viewer_panel"

local function agent()
  local ch = storage.frrl_character
  if ch and ch.valid then return ch end
  return nil
end

-- ------------------------------------------------------------------ strings

local function num(n)
  if type(n) ~= "number" then return "?" end
  if n == math.floor(n) then return tostring(math.floor(n)) end
  return string.format("%.1f", n)
end

local function pos_text(p)
  if not p then return nil end
  local x = p.x or p[1]
  local y = p.y or p[2]
  if x == nil or y == nil then return nil end
  return "(" .. num(x) .. "," .. num(y) .. ")"
end

local function as_position(p)
  if not p then return nil end
  local x = p.x or p[1]
  local y = p.y or p[2]
  if type(x) ~= "number" or type(y) ~= "number" then return nil end
  return { x = x, y = y }
end

--- What a handle names, without touching the registry.
--
-- `handles.resolve` would do, except that it records a destroyed entity as
-- destroyed, which is a write the agent could later observe.
local function peek(handle)
  if handle == "character" then return "character", nil end
  local registry = storage.frrl_handles
  local entry = registry and registry.by_handle and registry.by_handle[handle]
  if not entry then return tostring(handle), nil end
  local position
  if entry.kind == "unit" then
    if entry.entity and entry.entity.valid then position = entry.entity.position end
  elseif entry.tx then
    position = { x = entry.tx + 0.5, y = entry.ty + 0.5 }
  end
  return entry.name or tostring(handle), position
end

local function shorten(text, limit)
  text = tostring(text or "")
  if #text <= limit then return text end
  return string.sub(text, 1, limit - 3) .. "..."
end

--- One action as a short line, and where on the map it points, if anywhere.
--
-- Returns (text, target, quiet). `quiet` actions -- waiting, cancelling -- go
-- in the panel but never float: a program that waits for a furnace issues
-- dozens of them and each would cover the one line worth reading.
local function describe(name, payload, result)
  payload = payload or {}
  result = result or {}
  if name == "move" then
    return "move " .. tostring(payload.direction), nil, false
  elseif name == "navigate" then
    if payload.handle then
      local what, where = peek(payload.handle)
      return "walk to " .. what .. (where and (" @ " .. pos_text(where)) or ""), where, false
    end
    return "walk to " .. (pos_text(payload.position) or "?"), as_position(payload.position), false
  elseif name == "mine" then
    local what, where = peek(payload.handle)
    return "mine " .. what .. " x" .. num(payload.count or 1), where, false
  elseif name == "craft" then
    return "craft " .. tostring(payload.recipe) .. " x" .. num(payload.count or 1), nil, false
  elseif name == "place" then
    local where = as_position(result.position) or as_position(payload.position)
    return "place " .. tostring(payload.item) .. " (" .. tostring(payload.direction or "north")
      .. ") @ " .. (pos_text(where) or "?"), where, false
  elseif name == "rotate" then
    local what, where = peek(payload.handle)
    return "rotate " .. what .. (where and (" @ " .. pos_text(where)) or ""), where, false
  elseif name == "transfer" then
    local count = result.count or payload.count
    local item = tostring(payload.item) .. " x" .. num(count)
    if payload.from == "character" then
      local what, where = peek(payload.to)
      return "put " .. item .. " -> " .. what, where, false
    elseif payload.to == "character" then
      local what, where = peek(payload.from)
      return "take " .. item .. " <- " .. what, where, false
    end
    local source = peek(payload.from)
    local what, where = peek(payload.to)
    return "move " .. item .. " " .. source .. " -> " .. what, where, false
  elseif name == "set_recipe" then
    local what, where = peek(payload.handle)
    return "recipe " .. tostring(payload.recipe or "none") .. " on " .. what, where, false
  elseif name == "research" then
    if payload.cancel then return "cancel research", nil, false end
    return "research " .. tostring(payload.technology), nil, false
  elseif name == "wait" then
    return "wait", nil, true
  elseif name == "cancel" then
    return "cancel", nil, true
  elseif name == "batch" then
    return "batch x" .. num(#(payload.operations or {})), nil, false
  end
  return tostring(name), nil, false
end

local function refusal(response)
  local problem = response and response.error or {}
  local reason = problem.code or (response and response.code) or "refused"
  if problem.message and problem.message ~= "" then
    reason = reason .. ": " .. problem.message
  end
  return shorten(reason, 48)
end

-- ---------------------------------------------------------------- rendering

local function drop_floats(v)
  for _, object in ipairs(v.floats or {}) do
    if object and object.valid then object.destroy() end
  end
  v.floats = {}
end

--- Text size, as a `draw_text` scale, and the height of one line in tiles.
--
-- The text is drawn in world units (`scale_with_zoom`) and scaled up by the
-- inverse of the camera zoom the overlay set, so it reads at the same size on
-- screen whatever the zoom -- and so line spacing is a fixed number of tiles
-- rather than something that depends on each client's camera. A client and an
-- in-game screenshot then agree on what the overlay looks like.
local FONT = "default-large-bold"
local FONT_PIXELS = 18

local function float_metrics(v)
  local zoom = v.options.zoom or 1
  if zoom <= 0 then zoom = 1 end
  local scale = (v.options.text_scale or 1.6) / zoom
  return scale, scale * FONT_PIXELS / 32 * 1.35
end

local function float_text(v, ch, text, color)
  local kept = {}
  for _, object in ipairs(v.floats or {}) do
    if object and object.valid then kept[#kept + 1] = object end
  end
  while #kept >= FLOAT_LINES do
    table.remove(kept, 1).destroy()
  end
  local scale, spacing = float_metrics(v)
  -- Push the survivors up one line each and dim them, oldest highest.
  for index, object in ipairs(kept) do
    local height = FLOAT_BASE + spacing * (#kept - index + 1)
    object.target = { entity = ch, offset = { 0, -height } }
    local c = object.color
    object.color = { c.r * 0.6, c.g * 0.6, c.b * 0.6, 0.6 }
  end
  kept[#kept + 1] = rendering.draw_text({
    text = text,
    surface = ch.surface,
    target = { entity = ch, offset = { 0, -FLOAT_BASE } },
    color = color,
    font = FONT,
    scale = scale,
    scale_with_zoom = true,
    time_to_live = FLOAT_TICKS,
    alignment = "center",
    vertical_alignment = "bottom",
    use_rich_text = false,
  })
  v.floats = kept
end

local function point_at(ch, target, refused)
  if not target then return end
  local color = refused and COLOR.line_refused or COLOR.line
  rendering.draw_line({
    color = color,
    width = 2,
    from = ch,
    to = target,
    surface = ch.surface,
    time_to_live = 60,
    dash_length = 0.4,
    gap_length = 0.25,
  })
  rendering.draw_circle({
    color = color,
    radius = 0.55,
    width = 3,
    filled = false,
    target = target,
    surface = ch.surface,
    time_to_live = 60,
  })
end

-- -------------------------------------------------------------------- panel

local function header_text(v)
  return "tick " .. num(v.tick or 0) .. "   decisions " .. num(v.decisions or 0)
end

local function build_panel(v, player)
  local screen = player.gui.screen
  if screen[PANEL_NAME] then screen[PANEL_NAME].destroy() end
  local frame = screen.add({ type = "frame", name = PANEL_NAME, direction = "vertical" })
  frame.location = { x = v.options.panel_x or 16, y = v.options.panel_y or 16 }
  frame.style.minimal_width = 420
  frame.ignored_by_interaction = true
  local title = frame.add({ type = "label", name = "title", caption = v.options.title or "agent" })
  title.style.font = "heading-1"
  local header = frame.add({ type = "label", name = "header", caption = header_text(v) })
  header.style.font = "default-large-semibold"
  header.style.font_color = { 0.9, 0.8, 0.5 }
  frame.add({ type = "line", direction = "horizontal" })
  local rows = frame.add({ type = "flow", name = "rows", direction = "vertical" })
  rows.style.vertical_spacing = 0
  for index = 1, PANEL_ROWS do
    local row = rows.add({ type = "label", name = "r" .. index, caption = "" })
    row.style.font = "default-large"
  end
  return frame
end

local function refresh_header(v)
  for _, player in pairs(game.connected_players) do
    local frame = player.gui.screen[PANEL_NAME]
    if frame and frame.valid then frame.header.caption = header_text(v) end
  end
end

local function refresh_panel(v)
  for _, player in pairs(game.connected_players) do
    local frame = player.gui.screen[PANEL_NAME]
    if frame and frame.valid then
      frame.header.caption = header_text(v)
      -- Newest first, so the eye finds the latest action in the same place.
      local rows = frame.rows
      for index = 1, PANEL_ROWS do
        local row = v.rows[#v.rows - index + 1]
        local label = rows["r" .. index]
        if row then
          local caption = row.text
          if row.times > 1 then caption = caption .. "  (x" .. row.times .. ")" end
          label.caption = caption
          label.style.font_color = ROW_COLOR[row.kind] or ROW_COLOR.ok
          label.visible = true
        else
          -- Hidden rather than blank, so the panel is only as tall as its
          -- history and does not cover the map with an empty box.
          label.caption = ""
          label.visible = false
        end
      end
    end
  end
end

--- Add a history row, folding a repeat of the last one into a count, so that a
--- run of forty waits is one row that says so rather than the whole panel.
local function push_row(v, text, kind)
  local last = v.rows[#v.rows]
  if last and last.text == shorten(text, 80) and last.kind == kind then
    last.times = last.times + 1
    return
  end
  v.rows[#v.rows + 1] = { text = shorten(text, 80), kind = kind, times = 1 }
  while #v.rows > PANEL_ROWS do table.remove(v.rows, 1) end
end

-- --------------------------------------------------------------------- hook

--- Called from `actions.dispatch` after every action, with what it returned.
--
-- Wrapped in `pcall` by the caller's contract: an overlay that errors must not
-- turn into an action that errors.
local function record(state, request, response)
  local v = storage.frrl_viewer
  local payload = request and request.payload or {}
  local name = payload.action
  if not name then return end

  local episode = state and state.episode_id
  if episode ~= v.episode then
    v.episode = episode
    v.decisions = 0
  end
  v.decisions = (v.decisions or 0) + 1
  v.tick = response and response.tick or v.tick

  local refused = not response or response.code ~= CODE.OK
  local result = response and response.result or {}
  local text, target, quiet = describe(name, payload, result)
  local kind = refused and "refused" or (quiet and "quiet" or "ok")
  local shown = refused and ("refused " .. text .. " -- " .. refusal(response)) or text

  if name == "batch" then
    -- One row per operation, so the panel shows what the batch did rather
    -- than that there was one.
    local done = result.operations or {}
    for index, operation in ipairs(payload.operations or {}) do
      -- Operations after the one that failed never ran, so they get no row.
      if index > #done + 1 then break end
      local ok = done[index] and done[index].status == "completed"
      local op_text = describe(operation.action, operation, done[index] and done[index].result)
      push_row(v, "  " .. op_text, ok and "ok" or "refused")
    end
  end
  push_row(v, shown, kind)

  local ch = agent()
  if ch and not quiet and v.options.floating ~= false then
    -- Above the character a refusal carries only its code: the full message
    -- is in the panel, and a floating line has to be read in a second and a half.
    local floated = shown
    if refused then
      local problem = response and response.error or {}
      floated = "refused " .. text .. ": " .. tostring(problem.code or "refused")
    end
    float_text(v, ch, shorten(floated, 64), refused and COLOR.refused or COLOR.ok)
    if v.options.lines ~= false then point_at(ch, target, refused) end
  end
  if v.options.panel ~= false then refresh_panel(v) end
end

function viewer.on_action(state, request, response)
  if not storage.frrl_viewer then return end
  pcall(record, state, request, response)
end

-- --------------------------------------------------------------------- tick

--- The episode tick, supplied by `runtime` through `viewer.register` so this
--- module does not reach into its state.
local state_tick = nil

--- Every tick while enabled: keep spectators on the character, and set up any
--- player that has joined since the last one.
--
-- Every tick rather than control.lua's every thirty, because a camera that
-- jumps twice a second is fine for keeping an eye on a run and poor in a
-- video. Teleporting a spectator moves a camera and nothing else.
local function on_tick(event)
  local v = storage.frrl_viewer
  if not v then return end
  local ch = agent()
  v.configured = v.configured or {}
  for _, player in pairs(game.connected_players) do
    if player.controller_type == defines.controllers.spectator then
      if ch then pcall(function() player.teleport(ch.position, ch.surface) end) end
      if not v.configured[player.index] then
        v.configured[player.index] = true
        if v.options.zoom then pcall(function() player.zoom = v.options.zoom end) end
        if v.options.panel ~= false then
          pcall(build_panel, v, player)
          pcall(refresh_panel, v)
        end
      end
    end
  end
  if event.tick % HEADER_INTERVAL == 0 and v.options.panel ~= false and state_tick then
    v.tick = state_tick()
    pcall(refresh_header, v)
  end
end

local function listen()
  script.on_nth_tick(1, storage.frrl_viewer and on_tick or nil)
end

-- ------------------------------------------------------------------- remote

local function enable(options)
  options = type(options) == "table" and options or {}
  local previous = storage.frrl_viewer
  if previous then drop_floats(previous) end
  storage.frrl_viewer = {
    options = {
      title = options.title and tostring(options.title) or nil,
      zoom = type(options.zoom) == "number" and options.zoom or nil,
      text_scale = type(options.text_scale) == "number" and options.text_scale or 1.6,
      floating = options.floating ~= false,
      panel = options.panel ~= false,
      lines = options.lines ~= false,
      panel_x = type(options.panel_x) == "number" and options.panel_x or nil,
      panel_y = type(options.panel_y) == "number" and options.panel_y or nil,
    },
    floats = {},
    rows = previous and previous.rows or {},
    decisions = previous and previous.decisions or 0,
    episode = previous and previous.episode or nil,
    tick = previous and previous.tick or 0,
    -- Everyone is set up again, so a changed zoom or title reaches players
    -- who were already watching.
    configured = {},
  }
  listen()
  return { enabled = true }
end

local function disable()
  local v = storage.frrl_viewer
  if v then
    drop_floats(v)
    for _, player in pairs(game.players) do
      local frame = player.gui.screen[PANEL_NAME]
      if frame and frame.valid then frame.destroy() end
    end
  end
  storage.frrl_viewer = nil
  listen()
  return { enabled = false }
end

function viewer.register(tick_source)
  state_tick = tick_source
  remote.add_interface("frrl_viewer", {
    enable = enable,
    disable = disable,
    enabled = function() return storage.frrl_viewer ~= nil end,
  })
end

--- Re-register the tick handler for a save that had the overlay on. Reads
--- storage and writes nothing, as `on_load` must.
function viewer.on_load()
  if storage.frrl_viewer then script.on_nth_tick(1, on_tick) end
end

return viewer
