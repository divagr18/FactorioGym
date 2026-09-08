"""Typed session over one worker: the transport contract Phase 0/1 use.

Every call is a protocol v1 request with a fresh request id; responses are
parsed into :class:`factoriorl.protocol.Response`. ``request_status`` resolves
uncertain transport outcomes without guessing whether a mutation applied.
"""

from __future__ import annotations

import itertools
import time
import uuid
from dataclasses import dataclass

from factoriorl.errors import ProtocolError, StaleEpisodeError, UnsupportedProtocolError
from factoriorl.protocol import (
    PROTOCOL_VERSION,
    ActionResult,
    ActionStatus,
    ErrorBody,
    ErrorCode,
    Request,
    RequestType,
    Response,
)
from factoriorl.rcon import LuaError, RCONClient, RCONError, lua_string
from factoriorl.worker import WorkerHandle

_DISPATCH_LUA = 'return remote.call("frrl_bridge", "dispatch", {json})'

#: Default ceiling for waiting out an in-flight advance; overridden by
#: SupervisionPolicy.settle_deadline_seconds when a supervisor owns the session.
DEFAULT_SETTLE_SECONDS = 30.0


@dataclass
class TimedResponse:
    response: Response
    round_trip_ms: float
    #: Type of the request that produced this response, when known.
    request_type: str | None = None

    @property
    def action(self) -> ActionResult | None:
        """The typed execution status of an ``act`` response.

        ``None`` for every other request type, so callers can tell "not an
        action" from "action failed". The type gate matters: ``status``
        answers carry ``result.status == "ready"`` (a worker state) and
        ``advance`` carries running/completed for the stepping request -- and
        neither is an action outcome.
        """
        if self.request_type != RequestType.ACT.value:
            return None
        result = self.response.result or {}
        raw_status = result.get("status")
        if raw_status is None:
            return None
        try:
            status = ActionStatus(raw_status)
        except ValueError as exc:
            raise ProtocolError(f"unknown action status {raw_status!r}") from exc
        reason = self.response.error
        # The mod writes a refused action's cause under `error`; only older
        # shapes used `reason`. Reading `reason` alone meant this property
        # reported a rejected action with no cause at all, while
        # `env.step_payload` -- which reads the raw dict -- saw it.
        raw_reason = result.get("error")
        if not isinstance(raw_reason, dict):
            raw_reason = result.get("reason")
        if reason is None and isinstance(raw_reason, dict):
            raw = raw_reason
            try:
                reason_code = ErrorCode(raw.get("code", ""))
            except ValueError as exc:
                raise ProtocolError(f"unknown error code {raw.get('code')!r}") from exc
            reason = ErrorBody(
                code=reason_code,
                message=raw.get("message", ""),
                details=raw.get("details", {}),
            )
        data = {k: v for k, v in result.items() if k not in ("status", "reason", "error")}
        return ActionResult(status=status, reason=reason, data=data)


class WorkerSession:
    """One logical connection to a worker's typed protocol."""

    def __init__(
        self,
        handle: WorkerHandle,
        timeout: float = 15.0,
        settle_timeout: float = DEFAULT_SETTLE_SECONDS,
    ) -> None:
        self.handle = handle
        self._client = RCONClient(handle.spec.rcon_endpoint, timeout=timeout)
        self._client.connect()
        self._counter = itertools.count(1)
        # Request ids must be unique across every session that ever talks to
        # this worker, not just within one. A reconnecting client restarting
        # its counter at 1 would otherwise collide with ids the worker already
        # recorded, and the dedup ledger would answer a genuinely new mutation
        # with an old stored result -- silently dropping it (PLAN.md 1.2).
        self._nonce = uuid.uuid4().hex[:8]
        self.settle_timeout = settle_timeout
        #: Worker pacing multiplier, kept in sync so `step` can predict how long
        #: an interval takes instead of polling blindly.
        self.speed = 1.0
        self.episode_id: str | None = None

    # ------------------------------------------------------------ transport

    def _request(self, request: Request) -> TimedResponse:
        return self.dispatch_raw(request.to_dict(), stale_raises=True)

    def dispatch_raw(self, request_dict: dict, stale_raises: bool = True) -> TimedResponse:
        """Send an already-built request body; used for negative fixtures.

        Normal requests adopt the worker's episode from every response, so a
        reset observed through the wire never leaves a stale cached episode.
        """
        import json as _json

        body = _json.dumps(request_dict, ensure_ascii=False)
        code = _DISPATCH_LUA.replace("{json}", lua_string(body))
        start = time.perf_counter()
        try:
            raw = self._client.lua(code)
        except (RCONError, LuaError, TimeoutError, OSError) as exc:
            # A socket timeout is a transport failure like any other: it must
            # arrive as a typed ProtocolError so callers can resolve it with
            # request_status instead of catching bare OSErrors.
            raise ProtocolError(f"transport failure: {exc}") from exc
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if not isinstance(raw, dict):
            raise ProtocolError(f"non-object dispatch response: {raw!r}")
        protocol = raw.get("protocol")
        if protocol != PROTOCOL_VERSION:
            raise UnsupportedProtocolError(f"worker protocol {protocol!r} != {PROTOCOL_VERSION}")
        response = Response.from_dict(raw)
        self._adopt_episode(response)
        if stale_raises and response.code.value == "stale_episode":
            message = response.error.message if response.error else ""
            raise StaleEpisodeError(f"episode {request_dict.get('episode_id')} ended: {message}")
        return TimedResponse(
            response=response,
            round_trip_ms=elapsed_ms,
            request_type=request_dict.get("type"),
        )

    def _next_request_id(self, prefix: str) -> str:
        return f"{prefix}-{self._nonce}-{next(self._counter)}"

    # ------------------------------------------------------------ API

    def status(self) -> TimedResponse:
        request = Request(
            request_id=self._next_request_id("status"),
            episode_id=self.episode_id or "",
            type=RequestType.STATUS,
        )
        timed = self._request(request)
        self._adopt_episode(timed.response)
        reported = (timed.response.result or {}).get("speed")
        if isinstance(reported, int | float) and reported > 0:
            self.speed = float(reported)
        return timed

    def observe(self) -> TimedResponse:
        request = Request(
            request_id=self._next_request_id("observe"),
            episode_id=self.episode_id or "",
            type=RequestType.OBSERVE,
        )
        timed = self._request(request)
        self._adopt_episode(timed.response)
        return timed

    def advance(self, ticks: int) -> TimedResponse:
        """Request exact advancement of ``ticks`` game ticks.

        Returns once the advance is *settled*: the poll loop below waits for
        the worker to re-pause, so the caller observes the final tick state
        (PLAN.md 0.3: observations report only after the interval completes).
        """
        request = Request(
            request_id=self._next_request_id("advance"),
            episode_id=self._require_episode(),
            type=RequestType.ADVANCE,
            payload={"ticks": ticks},
        )
        timed = self._request(request)
        return self._settle(request, timed)

    def _settle(
        self, request: Request, timed: TimedResponse, poll_interval: float = 0.02
    ) -> TimedResponse:
        """Poll request_status until the stored response stops running."""
        result = timed.response.result or {}
        if result.get("status") != "running":
            return timed
        # Wall clock from the moment the advance was accepted. Reporting only
        # the first dispatch plus the last probe -- as this did -- hid every
        # intermediate probe and sleep, and understated a 30-tick advance by
        # roughly 300 ms. Profiling built on that number would be fiction.
        settle_started = time.perf_counter()
        deadline = time.monotonic() + self.settle_timeout
        while time.monotonic() < deadline:
            probe = self.request_status(request.request_id)
            resolution = probe.response.result or {}
            stored = resolution.get("stored")
            if resolution.get("state") == "rejected":
                raise ProtocolError(
                    f"advance {request.request_id} was rejected by the worker: {stored}"
                )
            if resolution.get("settled") and stored:
                return TimedResponse(
                    response=Response(
                        request_id=stored.get("request_id", request.request_id),
                        episode_id=stored.get("episode_id", self.episode_id or ""),
                        code=probe.response.code,
                        result=stored.get("result", {}),
                        tick=stored.get("tick"),
                    ),
                    round_trip_ms=(
                        timed.round_trip_ms + (time.perf_counter() - settle_started) * 1000.0
                    ),
                    request_type=timed.request_type,
                )
            time.sleep(poll_interval)
        raise ProtocolError(
            f"advance {request.request_id} did not settle within {self.settle_timeout}s"
        )

    def act(self, action: str, **payload) -> TimedResponse:
        request = Request(
            request_id=self._next_request_id("act"),
            episode_id=self._require_episode(),
            type=RequestType.ACT,
            payload={"action": action, **payload},
        )
        return self._request(request)

    def configure(self, speed: float) -> TimedResponse:
        """Set the worker's pacing multiplier.

        Pacing only: Factorio is tick-based and the simulation is identical at
        any speed, which `factoriorl bench speed` proves by comparing episode
        records field by field. At the default 60 UPS a 30-tick interval costs
        a hard 500 ms of wall clock, so this is the largest throughput lever.
        """
        request = Request(
            request_id=self._next_request_id("configure"),
            episode_id=self.episode_id or "",
            type=RequestType.CONFIGURE,
            payload={"speed": speed},
        )
        timed = self._request(request)
        if timed.response.ok:
            self.speed = float((timed.response.result or {}).get("applied", {}).get("speed", speed))
        return timed

    def describe(self) -> TimedResponse:
        """The action matrix, profiles and versions this worker is running."""
        request = Request(
            request_id=self._next_request_id("describe"),
            episode_id=self.episode_id or "",
            type=RequestType.DESCRIBE,
        )
        return self._request(request)

    def define_scenario(self, blueprint: dict, digest: str) -> TimedResponse:
        """Install a scene blueprint, referenced by hash thereafter.

        Installing once and resetting by hash keeps a steady-state reset at a
        ~120-byte payload instead of resending the whole scene every episode.
        """
        request = Request(
            request_id=self._next_request_id("scenario"),
            episode_id=self.episode_id or "",
            type=RequestType.SCENARIO_DEFINE,
            payload={"hash": digest, "blueprint": blueprint},
        )
        return self._request(request)

    def truth(self) -> TimedResponse:
        """Evaluator-only ground truth. Never handed to a policy."""
        request = Request(
            request_id=self._next_request_id("truth"),
            episode_id=self.episode_id or "",
            type=RequestType.TRUTH,
        )
        return self._request(request)

    def world_digest(self) -> TimedResponse:
        """Evaluator-only: everything a reset must restore, as sorted lines."""
        request = Request(
            request_id=self._next_request_id("digest"),
            episode_id=self.episode_id or "",
            type=RequestType.WORLD_DIGEST,
        )
        return self._request(request)

    def step(self, action: dict, ticks: int = 30) -> TimedResponse:
        """One RL transition: apply an action, run the interval, observe.

        Two round trips, which is the floor for exact stepping over RCON: a
        command executes inside a tick, so Lua cannot block while the world
        advances and `rcon.print` cannot be deferred to a later tick. The wait
        between them is *predicted* from the interval and the current speed
        rather than polled blindly, so the first collect normally finds it
        settled.
        """
        request = Request(
            request_id=self._next_request_id("step"),
            episode_id=self._require_episode(),
            type=RequestType.STEP,
            payload={"action": action, "ticks": ticks},
        )
        timed = self._request(request)
        if not timed.response.ok:
            return timed
        return self._collect(request.request_id, timed, ticks)

    def collect(self, request_id: str) -> TimedResponse:
        request = Request(
            request_id=self._next_request_id("collect"),
            episode_id=self.episode_id or "",
            type=RequestType.COLLECT,
            payload={"request_id": request_id},
        )
        return self._request(request)

    def _collect(
        self, request_id: str, timed: TimedResponse, ticks: int, poll_interval: float = 0.0005
    ) -> TimedResponse:
        started = time.perf_counter()
        # Predicted from the configured speed. An attempt to bound this by a
        # measured tick rate made throughput five times worse: the only rate
        # observable here is `ticks / settle time`, which includes the sleep
        # being predicted, so a long sleep produced a low estimate that
        # lengthened the next sleep. The engine's true tick rate is not visible
        # from this side of the socket, and a fine poll interval handles a
        # wrong prediction better than a feedback loop does.
        expected = ticks / (60.0 * max(self.speed, 0.01))
        if expected > 0.001:
            time.sleep(expected)
        deadline = time.monotonic() + self.settle_timeout
        while time.monotonic() < deadline:
            probe = self.collect(request_id)
            resolution = probe.response.result or {}
            if resolution.get("settled"):
                return TimedResponse(
                    response=Response(
                        request_id=request_id,
                        episode_id=self.episode_id or "",
                        code=probe.response.code,
                        result=resolution.get("result", {}),
                        error=probe.response.error,
                        tick=probe.response.tick,
                    ),
                    round_trip_ms=(timed.round_trip_ms + (time.perf_counter() - started) * 1000.0),
                    request_type=RequestType.STEP.value,
                )
            time.sleep(poll_interval)
        raise ProtocolError(f"step {request_id} did not settle within {self.settle_timeout}s")

    def reset(
        self,
        scenario: str | None = None,
        observation_profile: str | None = None,
        action_profile: str | None = None,
        blueprint_hash: str | None = None,
    ) -> TimedResponse:
        payload: dict = {}
        if scenario is not None:
            payload["scenario"] = scenario
        if blueprint_hash is not None:
            payload["blueprint_hash"] = blueprint_hash
        if observation_profile is not None:
            payload["observation_profile"] = observation_profile
        if action_profile is not None:
            payload["action_profile"] = action_profile
        request = Request(
            request_id=self._next_request_id("reset"),
            episode_id=self.episode_id or "",
            type=RequestType.RESET,
            payload=payload,
        )
        timed = self._request(request)
        self._adopt_episode(timed.response)
        return timed

    def request_status(self, request_id: str) -> TimedResponse:
        request = Request(
            request_id=self._next_request_id("reqstat"),
            episode_id=self.episode_id or "",
            type=RequestType.REQUEST_STATUS,
            payload={"request_id": request_id},
        )
        return self._request(request)

    # ------------------------------------------------------------ episodes

    def _require_episode(self) -> str:
        if not self.episode_id:
            self.status()
        if not self.episode_id:
            raise ProtocolError("worker reported no active episode")
        return self.episode_id

    def _adopt_episode(self, response: Response) -> None:
        if response.episode_id:
            self.episode_id = response.episode_id

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> WorkerSession:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
