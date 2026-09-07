-- Protocol v2 vocabulary: the single Lua declaration of the wire contract.
--
-- runtime.lua used to declare PROTOCOL_VERSION itself, so the Python literal
-- and the Lua literal could drift with only a live worker to notice. Both
-- sides now have exactly one declaration each, and an engine-free contract
-- test asserts they are equal.
--
-- Response codes and error codes are spelled out here rather than as free
-- string literals scattered through handlers, so tests/contract can compare
-- the two vocabularies mechanically.

local protocol = {}

protocol.VERSION = 2

--- Response-level status.
protocol.CODE = {
  OK = "ok",
  REJECTED = "rejected",
  DUPLICATE = "duplicate",
  STALE_EPISODE = "stale_episode",
  UNSUPPORTED = "unsupported",
  ERROR = "error",
}

--- Lifecycle of a (possibly ongoing) action.
protocol.STATUS = {
  COMPLETED = "completed",
  RUNNING = "running",
  FAILED = "failed",
  CANCELLED = "cancelled",
  REJECTED = "rejected",
}

--- Structured reasons. Every one of these must exist in Python's ErrorCode.
protocol.ERR = {
  MISSING_FIELD = "missing_field",
  BAD_TYPE = "bad_type",
  UNKNOWN_ACTION = "unknown_action",
  UNKNOWN_REQUEST = "unknown_request",
  PRECONDITION = "precondition",
  OUT_OF_REACH = "out_of_reach",
  COLLISION = "collision",
  NO_ITEMS = "no_items",
  NO_SPACE = "no_space",
  ENGINE = "engine",
  BAD_PROTOCOL = "bad_protocol",
  BAD_EPISODE = "bad_episode",
  BUSY = "busy",
  UNKNOWN_HANDLE = "unknown_handle",
  TARGET_MISSING = "target_missing",
  NOT_CANCELLABLE = "not_cancellable",
  NOT_MINEABLE = "not_mineable",
  INVALID_TARGET = "invalid_target",
  RECIPE_UNAVAILABLE = "recipe_unavailable",
  TECH_LOCKED = "tech_locked",
  -- Distinct from TECH_LOCKED on purpose. A trigger-based technology is not a
  -- blocked plan: 2.0 gates the root of the tech tree behind crafting rather
  -- than selection (7 of 196 technologies), so "research this" is simply not
  -- the way to obtain it. Collapsing the two would tell an agent to wait for
  -- prerequisites that will never come.
  TECH_NOT_SELECTABLE = "tech_not_selectable",
  UNKNOWN_REQUEST_ID = "unknown_request_id",
}

return protocol
