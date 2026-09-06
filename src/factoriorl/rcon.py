"""Source RCON client (stdlib only), verified against Factorio 2.0.60.

Wire format (empirically confirmed):

    packet = int32 length | int32 id | int32 type | body

``length`` covers id + type + body, and the body is terminated by **two** NUL
bytes (Factorio 2.0 rejects single-NUL packets as invalid message types).

- Auth: type 3, body = password. Success answers id = request id (echo),
  empty body; failure answers id = -1.
- Command: type 2. The console executes the body as a server command, so
  Lua must be prefixed with ``/c ``. Responses arrive as type-0 chunks
  carrying the request id; a final empty-body chunk terminates multi-packet
  responses (a short read timeout guards against servers that skip it).

Lua runs through the mod bridge: ``remote.call("frrl_bridge", "run", code)``
returns ``{result = ...}`` or ``{error = ...}``; Lua errors never kill the
server session. Values are serialized with ``helpers.table_to_json`` —
in Factorio 2.0 the JSON helpers live on ``helpers``, not ``game``.

Lua 5.2 note: JSON ``\\u`` escapes are not Lua string escapes, so every string
embedded in Lua is serialized with ``ensure_ascii=False``.
"""

from __future__ import annotations

import json
import socket
import struct
from dataclasses import dataclass

_HEADER = struct.Struct("<iii")  # length, id, type

TYPE_AUTH = 3
TYPE_COMMAND = 2
TYPE_RESPONSE_VALUE = 0

REQUEST_ID = 10
AUTH_REQUEST_ID = 3
CONTINUATION_TIMEOUT = 0.25

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

    def __init__(self, endpoint: RCONEndpoint, timeout: float = 10.0) -> None:
        self.endpoint = endpoint
        self.timeout = timeout
        self._sock: socket.socket | None = None

    def connect(self) -> None:
        self._sock = socket.create_connection(
            (self.endpoint.host, self.endpoint.port), timeout=self.timeout
        )
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

    def _read_packet_or_none(self) -> tuple[int, bytes] | None:
        try:
            return self._read_packet()
        except TimeoutError:
            return None

    def command(self, body: str) -> str:
        """Send a console command and reassemble the chunked response."""
        self._send(REQUEST_ID, TYPE_COMMAND, body)
        packet = self._read_packet()
        packet_id, body_bytes = packet
        if packet_id != REQUEST_ID:
            raise RCONError(f"unexpected rcon packet id {packet_id}")
        chunks: list[bytes] = []
        if body_bytes:
            chunks.append(body_bytes)
            # Multi-packet responses end with an empty chunk; a short
            # continuation timeout handles servers that omit it.
            self._sock.settimeout(CONTINUATION_TIMEOUT)
            try:
                while True:
                    packet = self._read_packet_or_none()
                    if packet is None:
                        break
                    chunk_id, chunk_body = packet
                    if chunk_id != REQUEST_ID:
                        raise RCONError(f"unexpected rcon packet id {chunk_id}")
                    if not chunk_body:
                        break
                    chunks.append(chunk_body)
            finally:
                self._sock.settimeout(self.timeout)
        return b"".join(chunks).decode("utf-8").rstrip("\n")

    def lua(self, code: str) -> object:
        """Run Lua through the bridge and decode the bridge payload."""
        raw = self.command("/c " + wrap_lua(code))
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
