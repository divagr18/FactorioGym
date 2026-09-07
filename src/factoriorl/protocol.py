"""Protocol v1: typed JSON messages between Python and the Lua mod.

Every request carries a protocol version, a request id (dedup key), and the
episode id it belongs to. Responses echo request and episode ids and carry a
status code plus either a result body or a structured error.

Versioned here; a change requires bumping the version on both sides, adding a
new ``tests/fixtures/protocol_v<n>/`` directory, keeping the previous one, and
adding the refusal test that proves the old version is now rejected
(CONTRIBUTING.md).

v2 adds the ten-action matrix, fused stepping, entity handles, observation and
action profiles, and the error vocabulary those need.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

PROTOCOL_VERSION = 2


class ResultCode(StrEnum):
    """Response-level status."""

    OK = "ok"
    REJECTED = "rejected"  # invalid action/request: structured reason, no mutation
    DUPLICATE = "duplicate"  # request id already applied; stored result returned
    STALE_EPISODE = "stale_episode"  # request targets an ended episode
    UNSUPPORTED = "unsupported"  # unknown type or protocol version
    ERROR = "error"  # internal failure


class ActionStatus(StrEnum):
    """Lifecycle of a (possibly ongoing) action."""

    COMPLETED = "completed"
    RUNNING = "running"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class RequestType(StrEnum):
    ADVANCE = "advance"  # exact stepping
    ACT = "act"  # typed embodied action
    OBSERVE = "observe"
    RESET = "reset"
    STATUS = "status"  # also the liveness probe
    REQUEST_STATUS = "request_status"  # resolve uncertain transport outcomes
    # Fused act + advance: one request per RL transition. Two round trips is
    # the floor -- RCON commands execute inside a tick, so Lua cannot block
    # while the world advances -- and `collect` retrieves the settled result.
    STEP = "step"
    COLLECT = "collect"
    DESCRIBE = "describe"  # action matrix, profiles and versions actually running
    CONFIGURE = "configure"  # evaluator pacing knobs; never changes outcomes


class ActionType(StrEnum):
    """The ten actions of PLAN.md 2.1, mirroring ``mod/factoriorl/matrix.lua``."""

    MOVE = "move"
    MINE = "mine"
    CRAFT = "craft"
    PLACE = "place"
    ROTATE = "rotate"
    TRANSFER = "transfer"
    SET_RECIPE = "set_recipe"
    RESEARCH = "research"
    WAIT = "wait"
    CANCEL = "cancel"


class ErrorCode(StrEnum):
    """Structured reasons; the wire vocabulary for rejected/error responses."""

    MISSING_FIELD = "missing_field"
    BAD_TYPE = "bad_type"
    UNKNOWN_ACTION = "unknown_action"
    UNKNOWN_REQUEST = "unknown_request"
    PRECONDITION = "precondition"
    OUT_OF_REACH = "out_of_reach"
    COLLISION = "collision"
    NO_ITEMS = "no_items"
    NO_SPACE = "no_space"
    ENGINE = "engine"
    BAD_PROTOCOL = "bad_protocol"
    BAD_EPISODE = "bad_episode"
    DUPLICATE_REQUEST = "duplicate_request"
    BUSY = "busy"
    UNKNOWN_HANDLE = "unknown_handle"
    TARGET_MISSING = "target_missing"
    NOT_CANCELLABLE = "not_cancellable"
    NOT_MINEABLE = "not_mineable"
    INVALID_TARGET = "invalid_target"
    RECIPE_UNAVAILABLE = "recipe_unavailable"
    TECH_LOCKED = "tech_locked"
    # Deliberately distinct from TECH_LOCKED. Factorio 2.0 gates the root of
    # the tech tree behind triggers rather than selection -- 7 of 196
    # technologies complete by crafting or mining and cannot be queued at all.
    # Reporting that as "locked" would tell an agent to wait for prerequisites
    # that will never arrive; it needs to craft instead.
    TECH_NOT_SELECTABLE = "tech_not_selectable"
    UNKNOWN_REQUEST_ID = "unknown_request_id"


@dataclass(frozen=True)
class ErrorBody:
    code: ErrorCode
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        body: dict[str, Any] = {"code": self.code.value, "message": self.message}
        if self.details:
            body["details"] = self.details
        return body


@dataclass(frozen=True)
class ActionResult:
    """Outcome of an ``act`` request."""

    status: ActionStatus
    reason: ErrorBody | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        body: dict[str, Any] = {"status": self.status.value}
        if self.reason is not None:
            body["reason"] = self.reason.to_dict()
        if self.data:
            body["data"] = self.data
        return body

    @property
    def ok(self) -> bool:
        return self.status is ActionStatus.COMPLETED


@dataclass(frozen=True)
class Request:
    request_id: str
    episode_id: str
    type: RequestType
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": PROTOCOL_VERSION,
            "request_id": self.request_id,
            "episode_id": self.episode_id,
            "type": self.type.value,
            "payload": self.payload,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))


@dataclass(frozen=True)
class Response:
    request_id: str
    episode_id: str
    code: ResultCode
    result: dict[str, Any] = field(default_factory=dict)
    error: ErrorBody | None = None
    tick: int | None = None

    @property
    def ok(self) -> bool:
        return self.code is ResultCode.OK

    def to_dict(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "protocol": PROTOCOL_VERSION,
            "request_id": self.request_id,
            "episode_id": self.episode_id,
            "code": self.code.value,
        }
        if self.tick is not None:
            body["tick"] = self.tick
        if self.result:
            body["result"] = self.result
        if self.error is not None:
            body["error"] = self.error.to_dict()
        return body

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Response:
        error = None
        if "error" in raw:
            err = raw["error"]
            error = ErrorBody(
                code=ErrorCode(err.get("code", ErrorCode.ENGINE.value)),
                message=err.get("message", ""),
                details=err.get("details", {}),
            )
        return cls(
            request_id=raw.get("request_id", ""),
            episode_id=raw.get("episode_id", ""),
            code=ResultCode(raw.get("code", ResultCode.ERROR.value)),
            result=raw.get("result", {}),
            error=error,
            tick=raw.get("tick"),
        )
