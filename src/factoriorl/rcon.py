"""Source RCON client (stdlib only), verified against Factorio 2.0.60.

Wire format (empirically confirmed):

    packet = int32 length | int32 id | int32 type | body

``length`` covers id + type + body, and the body is terminated by **two** NUL
bytes (Factorio 2.0 rejects single-NUL packets as invalid message types).

- Auth: type 3, body = password. Success answers id = request id (echo),
  empty body; failure answers id = -1.
- Command: type 2. The console executes the body as a server command, so
  Lua must be prefixed with ``/c `` (or ``/silent-command ``). The reply is one
  type-0 packet carrying the request id -- **this build does not split long
  replies**: a ``rcon.print`` of 4 096, 65 536, 200 000, 1 000 000 and
  4 000 000 bytes each came back as exactly one packet of size+1. Source RCON
  permits a 4096-byte split and Factorio never uses it, so the reassembly loop
  below is for the protocol rather than for any reply this engine sends.

Lua runs through the mod bridge: ``remote.call("frrl_bridge", "run", code)``
returns ``{result = ...}`` or ``{error = ...}``; Lua errors never kill the
server session. Values are serialized with ``helpers.table_to_json`` —
in Factorio 2.0 the JSON helpers live on ``helpers``, not ``game``.

Lua 5.2 note: JSON ``\\u`` escapes are not Lua string escapes, so every string
embedded in Lua is serialized with ``ensure_ascii=False``.
"""

from __future__ import annotations

import json
import os
import socket
import struct
from dataclasses import dataclass

_HEADER = struct.Struct("<iii")  # length, id, type

TYPE_AUTH = 3
TYPE_COMMAND = 2
TYPE_RESPONSE_VALUE = 0

AUTH_REQUEST_ID = 3

#: First id handed out by the per-call counter. Above AUTH_REQUEST_ID so an
#: auth reply can never be mistaken for a command reply.
FIRST_REQUEST_ID = 100
#: Ids stay well inside int32 and never reach the -1 that marks auth failure.
MAX_REQUEST_ID = 1 << 30

#: A console command that runs successfully and prints nothing, so the server
#: answers it with a single empty-bodied packet. Following every real command
#: with one of these turns "have I received the whole response?" from a timeout
#: guess into a fact -- see :meth:`RCONClient.command` for why the sentinel's
#: arrival is necessary but not sufficient.
#:
#: Always silent, in both modes. It is framing, not a command any run meant to
#: issue, so echoing it doubles the console log and puts a `local _ = 1` in a
#: watching player's chat between every pair of real commands.
SENTINEL_BODY = "/silent-command local _ = 1"

#: `/c` is echoed to every connected player -- the command text lands in their
#: chat, and every command this repo sends is a page of `pcall`/`remote.call`
#: wrapper. Watching a run through a real client, that is the whole screen:
#: `local ok, result = pcall(function() return remote.call("frrl_bridge", ...`
#: printed verbatim over the world, several times a second.
#:
#: `/silent-command` runs identically and prints nothing. Measured on this
#: build: a `/sc rcon.print('sc-ok')` answered normally with a 6-byte reply and
#: appeared in neither the console log nor a connected player's chat, while the
#: five `/c` probes sent alongside it were all logged.
#:
#: Silent is the default, which costs something worth naming: `--console-log`
#: no longer records the commands a run issued, because the engine is the thing
#: that stops writing them. Set `FACTORIO_RL_ECHO_COMMANDS=1` (or pass
#: `silent=False`) to get that record back when a run needs to be audited from
#: the engine's own log rather than from Python's side of the wire.
SILENT_PREFIX = "/silent-command "
COMMAND_PREFIX = "/c "

#: Opt back in to `/c`, so the engine logs and broadcasts every command.
ECHO_ENV_VAR = "FACTORIO_RL_ECHO_COMMANDS"


def echo_commands_requested() -> bool:
    """True when the environment asks for `/c` instead of `/silent-command`."""
    return os.environ.get(ECHO_ENV_VAR, "").strip().lower() in ("1", "true", "yes", "on")


_LUA_BRIDGE = (
    'local ok, result = pcall(function() return remote.call("frrl_bridge", "run", {code}) end) '
    "if not ok then rcon.print(helpers.table_to_json({ error = tostring(result) })) "
    "else rcon.print(helpers.table_to_json(result)) end"
)


class RCONError(Exception):
    """Transport-level RCON failure."""


class LuaError(RCONError):
    """The wrapped Lua command raised an error."""


@dataclass(frozen=True)
class RCONEndpoint:
    host: str
    port: int
    password: str


def lua_string(text: str) -> str:
    """Quote ``text`` as a Lua double-quoted string literal.

    JSON string escaping (with raw unicode) is a subset of Lua 5.2 escaping:
    ``\\"``, ``\\\\``, ``\\n``, ``\\t`` are all valid there.
    """
    return json.dumps(text, ensure_ascii=False)


def wrap_lua(code: str) -> str:
    """Embed a Lua snippet in the bridge call."""
    return _LUA_BRIDGE.replace("{code}", lua_string(code))


def encode_packet(packet_id: int, packet_type: int, body: str) -> bytes:
    payload = body.encode("utf-8") + b"\x00\x00"
    return _HEADER.pack(len(payload) + 8, packet_id, packet_type) + payload


class RCONClient:
    """Blocking client; one request/response per call."""

    def __init__(
        self, endpoint: RCONEndpoint, timeout: float = 10.0, silent: bool | None = None
    ) -> None:
        self._next_request_id = FIRST_REQUEST_ID
        self.endpoint = endpoint
        self.timeout = timeout
        #: Issue Lua through `/silent-command` rather than `/c`, so the command
        #: text is not echoed into a watching player's chat or the console log.
        #: ``None`` defers to :data:`ECHO_ENV_VAR`, which is how a run gets the
        #: engine-side command record back without editing code.
        self.silent = not echo_commands_requested() if silent is None else silent
        #: Calls whose sentinel arrived before the command's own reply. Zero on
        #: every player-free run, so a non-zero count is a positive signal that
        #: something is connected to the worker -- worth reporting rather than
        #: absorbing silently, since that inversion is what used to look like a
        #: dead transport.
        self.reordered_replies = 0
        self._sock: socket.socket | None = None

    def connect(self) -> None:
        self._sock = socket.create_connection(
            (self.endpoint.host, self.endpoint.port), timeout=self.timeout
        )
        # Every request is one small write, and the sentinel makes it two back
        # to back. Nagle would hold the second waiting for an ACK of the first --
        # the classic ~40 ms delayed-ACK stall.
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._authenticate()

    def _authenticate(self) -> None:
        self._send(AUTH_REQUEST_ID, TYPE_AUTH, self.endpoint.password)
        packet_id, body = self._read_packet()
        # Success: empty body, id echoed. Failure: id == -1.
        if packet_id == AUTH_REQUEST_ID and body == b"":
            return
        raise RCONError(f"rcon authentication failed (id={packet_id}, body={body[:64]!r})")

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def __enter__(self) -> RCONClient:
        self.connect()
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def _send(self, packet_id: int, packet_type: int, body: str) -> None:
        self._sock.sendall(encode_packet(packet_id, packet_type, body))

    def _read_exact(self, count: int) -> bytes:
        chunks: list[bytes] = []
        received = 0
        while received < count:
            chunk = self._sock.recv(count - received)
            if not chunk:
                raise RCONError("connection closed by server")
            chunks.append(chunk)
            received += len(chunk)
        return b"".join(chunks)

    def _read_packet(self) -> tuple[int, bytes]:
        header = self._read_exact(_HEADER.size)
        length, packet_id, packet_type = _HEADER.unpack(header)
        body = self._read_exact(length - 8)
        del packet_type
        return packet_id, body.rstrip(b"\x00")

    def _allocate_ids(self) -> tuple[int, int]:
        """A fresh (command, sentinel) id pair for one call."""
        request_id = self._next_request_id
        sentinel_id = request_id + 1
        self._next_request_id += 2
        if self._next_request_id >= MAX_REQUEST_ID:
            self._next_request_id = FIRST_REQUEST_ID
        return request_id, sentinel_id

    def command(self, body: str, expect_response: bool = True) -> str:
        """Send a console command and reassemble its complete response.

        Completion is determined by a **sentinel**, not by waiting: the real
        command is followed immediately by a command known to print nothing.
        The sentinel's reply proves the real response is complete, and it costs
        one extra packet instead of a 250 ms timeout.

        The sentinel is necessary but **not sufficient**, because "answers in
        order on one connection" is only true of a server with no players.
        Attach a Factorio client to the worker and the order inverts for
        anything longer than a trivial command -- the console log during the
        failure shows the sentinel executing before the command it was meant to
        follow::

            [JOIN] Keno joined the game
            [COMMAND] <server> (command): rcon.print('p')
            [COMMAND] <server> (command): local _ = 1
            [COMMAND] <server> (command): local _ = 1     <- sentinel, ahead of
            [COMMAND] <server> (command): local ok, result = pcall(...)   its own
                                                                       command

        Treating the sentinel alone as completion then returned an **empty
        body** for every call: the real reply arrived during the *next* call,
        where its lower id marked it stale and it was drained. One lost reply
        per call, indefinitely, and `lua` reported it as
        `non-json rcon response: ''`. That is what made a watched run
        impossible, and it was read as a property of the RCON transport rather
        than of this loop. Ordering was the only thing wrong: with a client
        attached the same server answered a 16 KB command and a 64 KB reply
        without complaint.

        So completion is now "both replies seen", in either order. Every
        command gets at least one packet bearing its id -- an empty-bodied one
        when it printed nothing, which is exactly how the sentinel itself
        answers -- so this is a fact about the connection and not another
        assumption about timing.

        Per-call ids also make a timed-out call survivable. Every call used to
        reuse a single id, so one timeout desynchronized the connection
        permanently -- the late reply arrived during the *next* call and every
        call after it raised "unexpected rcon packet id". Replies bearing an id
        older than the current call are now drained as the stale packets they
        are, which is what PLAN.md section 2 means by resolving an uncertain
        transport outcome rather than poisoning the transport.

        ``expect_response=False`` is for commands that end the session (``/quit``),
        where waiting for a sentinel the server will never answer would hang.
        """
        request_id, sentinel_id = self._allocate_ids()
        self._send(request_id, TYPE_COMMAND, body)
        if not expect_response:
            return ""
        self._send(sentinel_id, TYPE_COMMAND, SENTINEL_BODY)

        chunks: list[bytes] = []
        seen_request = False
        seen_sentinel = False
        while not (seen_request and seen_sentinel):
            packet_id, chunk = self._read_packet()
            if packet_id == sentinel_id:
                seen_sentinel = True
                if not seen_request:
                    self.reordered_replies += 1
                continue
            if packet_id == request_id:
                seen_request = True
                if chunk:
                    chunks.append(chunk)
                continue
            if packet_id < request_id:
                # A reply to a call that already gave up. Drain it; the ids
                # prove it is not ours.
                continue
            raise RCONError(
                f"unexpected rcon packet id {packet_id} (awaiting {request_id}/{sentinel_id})"
            )
        return b"".join(chunks).decode("utf-8").rstrip("\n")

    def lua(self, code: str) -> object:
        """Run Lua through the bridge and decode the bridge payload."""
        prefix = SILENT_PREFIX if self.silent else COMMAND_PREFIX
        raw = self.command(prefix + wrap_lua(code))
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RCONError(f"non-json rcon response: {raw[:200]!r}") from exc
        if not isinstance(payload, dict):
            raise RCONError(f"unexpected bridge payload: {payload!r}")
        if "error" in payload:
            raise LuaError(str(payload["error"]))
        if "result" in payload:
            return payload["result"]
        raise RCONError(f"bridge payload missing result/error: {payload!r}")
