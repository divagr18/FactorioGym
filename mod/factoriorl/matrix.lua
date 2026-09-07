-- The action matrix (PLAN.md 2.1), as data rather than prose.
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

--- Serialisable form, for the `describe` request and doc generation.
function matrix.describe()
  local out = {}
  for _, name in ipairs(matrix.ORDER) do
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
