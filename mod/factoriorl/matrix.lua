-- The action matrix (DESIGN.md 2.1), as data rather than prose.
--
-- This table is the authority the dispatcher reads, not documentation about
-- the dispatcher. Three things are generated from it, so none of them can
-- drift from the others:
--   * generic payload validation, replacing per-handler missing_field/bad_type
--     boilerplate;
--   * docs/ACTION_MATRIX.md, via `factoriorl action-matrix`;
--   * the Python mirror in src/factoriorl/action_matrix.py, compared field for
--     field by an engine-free contract test.
--
-- Reach kinds map to the character's own prototype values, measured in the
-- Phase 2.0 spike: build 10, entity (reach) 10, resource 2.7, pickup 1. There
-- is no single reach distance, and `can_reach_entity` applies the engine's
-- bounding-box-aware rule rather than centre-to-centre distance.
--
-- There are two ordered catalogs, not one:
--
--   * `ORDER` is the primitive catalog. It is frozen: it is what
--     `primitive-v1` offers, what `src/factoriorl/action_matrix.py` mirrors
--     field for field, and what `docs/ACTION_MATRIX.md` is generated from.
--     Every Phase 3 and Phase 4 result was measured against exactly these ten
--     actions, so adding to it would silently redefine what those numbers mean.
--   * `ASSISTED_ORDER` holds actions that exist only under an assistance
--     profile. `profiles.lua` composes the two into each profile's catalog and
--     `actions.dispatch` refuses anything outside it, which is what makes
--     "navigation is absent from primitive-v1" a property of the dispatcher
--     rather than a claim in a document.
--
-- The Python mirror and the generated documentation currently cover `ORDER`
-- only. Mirroring `navigate` into `action_matrix.py` (and regenerating
-- `docs/ACTION_MATRIX.md` from it) is a Python-side follow-up; it is left
-- undone here rather than done by hand, because a hand-edited
-- `docs/ACTION_MATRIX.md` fails `factoriorl action-matrix --check` and a
-- doc that fails its own freshness check is worse than one that is behind.

local protocol = require("protocol")
local ERR = protocol.ERR

local matrix = {}

--- Reach kinds. "none" means the action is not a physical interaction.
matrix.REACH = {
  NONE = "none",
  ENTITY = "entity",
  BUILD = "build",
  RESOURCE = "resource",
}

local DIRECTIONS = { "north", "east", "south", "west" }

--- Ordered so generated docs and tests have a stable order.
matrix.ORDER = {
  "move",
  "mine",
  "craft",
  "place",
  "rotate",
  "transfer",
  "set_recipe",
  "research",
  "wait",
  "cancel",
}

--- Assistance-profile actions (DESIGN.md 5.1 onward). Not part of `ORDER`; see
--- the header for why the primitive catalog is frozen.
matrix.ASSISTED_ORDER = {
  "batch",
  "navigate",
}

--- Hard ceiling on operations in one batch, whatever the caller asks for.
--- A batch is executed inside a single request, so an unbounded one is an
--- unbounded amount of engine work between two ticks.
matrix.BATCH_LIMIT = 16

matrix.ACTIONS = {
  move = {
    ongoing = true,
    -- A policy emitting a direction every step must not have to spend an
    -- action cancelling the previous one first, so a new move supersedes the
    -- running one and the superseded entry settles as `cancelled` exactly once.
    -- Moving also interrupts mining: the two are physically exclusive for a
    -- character, and walking away from a rock stops mining it.
    supersedes = true,
    supersedes_actions = { "move", "mine" },
    cancellable = true,
    reach = matrix.REACH.NONE,
    mutates_inventory = false,
    time = "walks until the tick deadline, or until collision stops it",
    cancel_boundary = "walking stops at the paused tick the cancel is processed",
    failure_codes = { ERR.MISSING_FIELD, ERR.BAD_TYPE, ERR.PRECONDITION },
    payload = {
      direction = { kind = "enum", values = DIRECTIONS, required = true },
      ticks = { kind = "int", min = 1, max = 3600, required = false, default = 30 },
    },
  },

  mine = {
    ongoing = true,
    supersedes = false,
    cancellable = true,
    reach = matrix.REACH.RESOURCE,
    mutates_inventory = true,
    time = "native mining_state; duration from mineable_properties.mining_time "
      .. "divided by character mining speed",
    cancel_boundary = "mining stops at the paused tick; partial progress is "
      .. "discarded, which is native engine behaviour",
    failure_codes = {
      ERR.UNKNOWN_HANDLE,
      ERR.TARGET_MISSING,
      ERR.OUT_OF_REACH,
      ERR.NOT_MINEABLE,
      ERR.NO_SPACE,
      ERR.BUSY,
    },
    payload = {
      handle = { kind = "string", required = true },
      count = { kind = "int", min = 1, max = 1000, required = false, default = 1 },
    },
  },

  craft = {
    ongoing = true,
    supersedes = false,
    cancellable = true,
    reach = matrix.REACH.NONE,
    mutates_inventory = true,
    time = "native begin_crafting; ingredients are debited at start and "
      .. "products delivered on completion",
    cancel_boundary = "cancel_crafting refunds ingredients per engine rules; "
      .. "elapsed craft time is lost",
    failure_codes = {
      ERR.RECIPE_UNAVAILABLE,
      ERR.TECH_LOCKED,
      ERR.NO_ITEMS,
      ERR.BAD_TYPE,
    },
    payload = {
      recipe = { kind = "string", required = true },
      count = { kind = "int", min = 1, max = 100, required = false, default = 1 },
    },
  },

  place = {
    ongoing = false,
    supersedes = false,
    cancellable = false,
    reach = matrix.REACH.BUILD,
    mutates_inventory = true,
    -- There is no native character build path: build_from_cursor is declared
    -- on LuaPlayer, not LuaControl. Assembled from can_place_entity +
    -- create_entity + inventory debit, which excludes fast-replace, ghost
    -- revival, tile building, undo, and on_built_entity.
    time = "instantaneous",
    cancel_boundary = "n/a",
    failure_codes = {
      ERR.NO_ITEMS,
      ERR.COLLISION,
      ERR.OUT_OF_REACH,
      ERR.TECH_LOCKED,
      ERR.INVALID_TARGET,
    },
    payload = {
      item = { kind = "string", required = true },
      position = { kind = "position", required = true },
      direction = { kind = "enum", values = DIRECTIONS, required = false, default = "north" },
    },
  },

  rotate = {
    ongoing = false,
    supersedes = false,
    cancellable = false,
    reach = matrix.REACH.ENTITY,
    mutates_inventory = false,
    time = "instantaneous",
    cancel_boundary = "n/a",
    failure_codes = {
      ERR.UNKNOWN_HANDLE,
      ERR.TARGET_MISSING,
      ERR.OUT_OF_REACH,
      ERR.INVALID_TARGET,
    },
    payload = {
      handle = { kind = "string", required = true },
      reverse = { kind = "bool", required = false, default = false },
    },
  },

  transfer = {
    ongoing = false,
    supersedes = false,
    cancellable = false,
    reach = matrix.REACH.ENTITY,
    mutates_inventory = true,
    time = "instantaneous",
    cancel_boundary = "n/a",
    failure_codes = {
      ERR.NO_ITEMS,
      ERR.NO_SPACE,
      ERR.OUT_OF_REACH,
      ERR.UNKNOWN_HANDLE,
      ERR.TARGET_MISSING,
      ERR.PRECONDITION,
    },
    payload = {
      from = { kind = "string", required = true },
      to = { kind = "string", required = true },
      item = { kind = "string", required = true },
      count = { kind = "int", min = 1, max = 10000, required = true },
    },
  },

  set_recipe = {
    ongoing = false,
    supersedes = false,
    cancellable = false,
    reach = matrix.REACH.ENTITY,
    mutates_inventory = true,
    time = "instantaneous; items the machine returns are credited",
    cancel_boundary = "n/a",
    failure_codes = {
      ERR.RECIPE_UNAVAILABLE,
      ERR.TECH_LOCKED,
      ERR.INVALID_TARGET,
      ERR.OUT_OF_REACH,
      ERR.UNKNOWN_HANDLE,
      ERR.TARGET_MISSING,
    },
    payload = {
      handle = { kind = "string", required = true },
      recipe = { kind = "string", required = false },
    },
  },

  research = {
    ongoing = false,
    supersedes = false,
    cancellable = false,
    -- Force-level: no physical interaction, and that is stated rather than
    -- implied so the action profile discloses it.
    reach = matrix.REACH.NONE,
    mutates_inventory = false,
    time = "instantaneous selection; progress requires labs and science packs",
    cancel_boundary = "n/a; pass cancel=true to stop current research",
    failure_codes = { ERR.TECH_LOCKED, ERR.TECH_NOT_SELECTABLE, ERR.INVALID_TARGET },
    payload = {
      technology = { kind = "string", required = false },
      cancel = { kind = "bool", required = false, default = false },
    },
  },

  wait = {
    ongoing = false,
    supersedes = false,
    cancellable = false,
    reach = matrix.REACH.NONE,
    mutates_inventory = false,
    time = "instantaneous; game time is consumed by the surrounding advance/step",
    cancel_boundary = "n/a",
    failure_codes = {},
    payload = {},
  },

  cancel = {
    ongoing = false,
    supersedes = false,
    cancellable = false,
    reach = matrix.REACH.NONE,
    mutates_inventory = false,
    time = "instantaneous",
    cancel_boundary = "n/a",
    failure_codes = { ERR.UNKNOWN_REQUEST_ID, ERR.NOT_CANCELLABLE },
    payload = {
      target_request_id = { kind = "string", required = true },
    },
  },

  -- ------------------------------------------------------------- assisted

  --- A sequence of typed interactions executed under an explicit ceiling.
  --
  -- Only *instantaneous* operations are accepted. An ongoing action -- move,
  -- navigate, mine, craft -- does not finish inside the request that starts it,
  -- so a batch containing one could not say whether the operations after it ran
  -- before or after it landed, and "the completed prefix" would stop meaning
  -- anything. They are refused by name rather than silently reordered.
  --
  -- Reach is not re-implemented here. Each operation goes through the same
  -- validation and the same handler a standalone request would, so
  -- `REACH.ENTITY` is enforced per operation by the code that already enforces
  -- it, and a batch cannot reach further than the character can.
  batch = {
    assistance_only = true,
    ongoing = false,
    supersedes = false,
    cancellable = false,
    reach = matrix.REACH.NONE,
    mutates_inventory = true,
    time = "instantaneous",
    cancel_boundary = "n/a",
    failure_codes = {
      ERR.BAD_TYPE,
      ERR.MISSING_FIELD,
      ERR.UNKNOWN_ACTION,
      ERR.PRECONDITION,
    },
    payload = {
      operations = { kind = "list", required = true, max = matrix.BATCH_LIMIT },
      -- The caller's own ceiling, at or below the protocol's. Present so an
      -- agent can bound a batch more tightly than the server would.
      max_operations = { kind = "int", min = 1, max = matrix.BATCH_LIMIT,
        default = matrix.BATCH_LIMIT },
    },
  },

  navigate = {
    -- Assistance, not a new physical capability: it plans a route over the
    -- agent's own explored-terrain memory and then walks it with the same
    -- `walking_state` a `move` uses. No teleport, no engine pathfinder, no
    -- privileged look at unexplored ground.
    assistance_only = true,
    ongoing = true,
    -- Same reasoning as `move`, and it shares `move`'s in-flight slot: two
    -- operations driving `walking_state` would fight for one body. A new
    -- navigate replaces the old plan, a `move` takes manual control back, and
    -- both interrupt mining, which walking away from a rock does anyway.
    supersedes = true,
    supersedes_actions = { "move", "mine" },
    cancellable = true,
    -- Navigation itself is not a reach-limited interaction. It ends *beside* a
    -- target and touches nothing: it never opens, takes from, mines or
    -- otherwise completes the interaction the agent navigated there to perform.
    reach = matrix.REACH.NONE,
    mutates_inventory = false,
    time = "walks the planned route with walking_state, one cardinal command "
      .. "per tick at the character's own running speed (0.1484 tiles/tick); "
      .. "the route consumes real game time and cannot be skipped",
    cancel_boundary = "walking stops at the paused tick the cancel is "
      .. "processed; the partial walk stands and the report carries the "
      .. "position reached and the waypoints left",
    failure_codes = {
      ERR.MISSING_FIELD,
      ERR.BAD_TYPE,
      ERR.PRECONDITION,
      ERR.UNKNOWN_HANDLE,
      ERR.TARGET_MISSING,
      ERR.INVALID_TARGET,
      -- `collision` carries every "the route is blocked" outcome, discriminated
      -- by `result.reason`: destination_blocked, no_route, blocked,
      -- replan_limit. `precondition` carries the two budget stops,
      -- time_budget and search_budget. A dedicated `route_not_found` code
      -- would read better and needs a matching value in Python's ErrorCode,
      -- which is outside this change; the reason field is the discriminator
      -- until then.
      ERR.COLLISION,
    },
    payload = {
      position = { kind = "position", required = false },
      handle = { kind = "string", required = false },
      -- The minimum is 1.0 tile and that is a derivation, not a preference: a
      -- route ends on a tile centre, an arbitrary requested point can be a
      -- tile corner 0.7072 away from the nearest centre, and the per-tick
      -- walker leaves ~0.25 of residual error. Anything tighter would be a
      -- promise the walker cannot keep, and demanding sub-tile precision from
      -- a strided walk is exactly what left earlier solvers oscillating on a
      -- lattice (see src/factoriorl/catalog.py). Sub-tile positioning is the
      -- primitive `move` nudge's job, not navigation's.
      tolerance = { kind = "number", min = 1.0, max = 8.0, required = false },
      max_ticks = { kind = "int", min = 1, max = 36000, required = false, default = 3600 },
      replan_limit = { kind = "int", min = 0, max = 32, required = false, default = 8 },
    },
  },
}

--- Validate a request payload against an action's declared schema.
-- Returns nil on success, or (code, message, details) describing the first
-- problem. Centralising this is what removes per-handler validation drift.
function matrix.validate(action_name, payload)
  local spec = matrix.ACTIONS[action_name]
  if not spec then
    return ERR.UNKNOWN_ACTION, "unknown action: " .. tostring(action_name), nil
  end
  payload = payload or {}
  for field, rule in pairs(spec.payload) do
    local value = payload[field]
    if value == nil then
      if rule.required then
        return ERR.MISSING_FIELD, "payload." .. field .. " is required", { field = field }
      end
    elseif rule.kind == "int" then
      if type(value) ~= "number" or value ~= math.floor(value) then
        return ERR.BAD_TYPE, "payload." .. field .. " must be an integer",
          { field = field, got = tostring(value) }
      end
      if rule.min and value < rule.min then
        return ERR.BAD_TYPE, "payload." .. field .. " must be >= " .. rule.min,
          { field = field, got = value }
      end
      if rule.max and value > rule.max then
        return ERR.BAD_TYPE, "payload." .. field .. " must be <= " .. rule.max,
          { field = field, got = value }
      end
    elseif rule.kind == "number" then
      -- Distinct from "int": a navigation tolerance is a distance in tiles and
      -- rounding it to an integer would quietly change what the agent asked
      -- for. JSON numbers arrive as Lua numbers either way.
      if type(value) ~= "number" then
        return ERR.BAD_TYPE, "payload." .. field .. " must be a number",
          { field = field, got = tostring(value) }
      end
      if rule.min and value < rule.min then
        return ERR.BAD_TYPE, "payload." .. field .. " must be >= " .. rule.min,
          { field = field, got = value }
      end
      if rule.max and value > rule.max then
        return ERR.BAD_TYPE, "payload." .. field .. " must be <= " .. rule.max,
          { field = field, got = value }
      end
    elseif rule.kind == "string" then
      if type(value) ~= "string" or value == "" then
        return ERR.BAD_TYPE, "payload." .. field .. " must be a non-empty string",
          { field = field, got = tostring(value) }
      end
    elseif rule.kind == "bool" then
      if type(value) ~= "boolean" then
        return ERR.BAD_TYPE, "payload." .. field .. " must be a boolean",
          { field = field, got = tostring(value) }
      end
    elseif rule.kind == "enum" then
      local allowed = false
      for _, candidate in ipairs(rule.values) do
        if value == candidate then allowed = true break end
      end
      if not allowed then
        return ERR.BAD_TYPE,
          "payload." .. field .. " must be one of " .. table.concat(rule.values, "/"),
          { field = field, got = tostring(value) }
      end
    elseif rule.kind == "list" then
      if type(value) ~= "table" then
        return ERR.BAD_TYPE, "payload." .. field .. " must be a list",
          { field = field, got = type(value) }
      end
      local count = 0
      for index, entry in ipairs(value) do
        count = index
        if type(entry) ~= "table" or type(entry.action) ~= "string" then
          return ERR.BAD_TYPE,
            "payload." .. field .. "[" .. index .. "] must be an object with an action",
            { field = field, index = index }
        end
      end
      if count == 0 then
        return ERR.BAD_TYPE, "payload." .. field .. " must not be empty", { field = field }
      end
      if rule.max and count > rule.max then
        return ERR.BAD_TYPE,
          "payload." .. field .. " holds " .. count .. ", the limit is " .. rule.max,
          { field = field, got = count, limit = rule.max }
      end
    elseif rule.kind == "position" then
      local ok = type(value) == "table"
        and type(value[1]) == "number"
        and type(value[2]) == "number"
      if not ok then
        return ERR.BAD_TYPE, "payload." .. field .. " must be [x, y]",
          { field = field, got = tostring(value) }
      end
    end
  end
  return nil
end

--- Fill declared defaults so handlers never re-implement them.
function matrix.with_defaults(action_name, payload)
  local spec = matrix.ACTIONS[action_name]
  local out = {}
  for key, value in pairs(payload or {}) do out[key] = value end
  if spec then
    for field, rule in pairs(spec.payload) do
      if out[field] == nil and rule.default ~= nil then out[field] = rule.default end
    end
  end
  return out
end

--- Every action name a profile could offer, primitive catalog first.
function matrix.all_names()
  local names = {}
  for _, name in ipairs(matrix.ORDER) do names[#names + 1] = name end
  for _, name in ipairs(matrix.ASSISTED_ORDER) do names[#names + 1] = name end
  return names
end

--- Serialisable form, for the `describe` request and doc generation.
--- Assistance actions are included and flagged, so a client that pins the
--- matrix can see which of them its profile will actually be allowed to send.
function matrix.describe()
  local out = {}
  for _, name in ipairs(matrix.all_names()) do
    local spec = matrix.ACTIONS[name]
    local fields = {}
    for field, rule in pairs(spec.payload) do
      fields[field] = {
        kind = rule.kind,
        required = rule.required or false,
        default = rule.default,
        min = rule.min,
        max = rule.max,
        values = rule.values,
      }
    end
    out[#out + 1] = {
      name = name,
      ongoing = spec.ongoing,
      supersedes = spec.supersedes,
      cancellable = spec.cancellable,
      assistance_only = spec.assistance_only or false,
      reach = spec.reach,
      mutates_inventory = spec.mutates_inventory,
      time = spec.time,
      cancel_boundary = spec.cancel_boundary,
      failure_codes = spec.failure_codes,
      payload = fields,
    }
  end
  return out
end

return matrix
