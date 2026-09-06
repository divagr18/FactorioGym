"""RCON framing against a fake server (engine-free).

The real bug this guards was invisible to every engine test: Factorio never
sends the empty terminator chunk that ends a multi-packet Source RCON response,
so the client fell back to a 250 ms timeout on *every* call. Measured, that was
250.03 ms of a 266.69 ms request.

A fake server lets the awkward cases be reproduced deliberately -- a response
with no terminator, a response split across packets, and a late reply from a
call that already timed out -- none of which a live Factorio can be asked for
on demand.
"""

from __future__ import annotations

import socket
import struct
import threading

import pytest

from factoriorl.rcon import (
    SENTINEL_BODY,
    TYPE_AUTH,
    TYPE_COMMAND,
    TYPE_RESPONSE_VALUE,
    RCONClient,
    RCONEndpoint,
    encode_packet,
)

_HEADER = struct.Struct("<iii")
PASSWORD = "test-password"


class FakeRCONServer:
    """A single-connection Source RCON server with scripted replies.

    Mirrors the real server's two relevant behaviours: it answers in order on
    one connection, and it never sends a terminator chunk.
    """

    def __init__(
        self,
        replies: list[list[str]],
        extra_packets: list[tuple[int, str]] | None = None,
    ):
        #: One entry per non-sentinel command; each entry is the list of body
        #: chunks that command answers with.
        self.replies = replies
        #: (id, body) packets injected before the next reply, simulating a late
        #: response from a call the client already abandoned.
        self.extra_packets = extra_packets or []
        self.command_bodies: list[str] = []
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(1)
        self.port = self._sock.getsockname()[1]
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _read_packet(self, conn) -> tuple[int, int, str] | None:
        header = self._recv_exact(conn, _HEADER.size)
        if header is None:
            return None
        length, packet_id, packet_type = _HEADER.unpack(header)
        body = self._recv_exact(conn, length - 8)
        if body is None:
            return None
        return packet_id, packet_type, body.rstrip(b"\x00").decode("utf-8")

    @staticmethod
    def _recv_exact(conn, count: int) -> bytes | None:
        chunks = []
        received = 0
        while received < count:
            chunk = conn.recv(count - received)
            if not chunk:
                return None
            chunks.append(chunk)
            received += len(chunk)
        return b"".join(chunks)

    def _serve(self) -> None:
        conn, _ = self._sock.accept()
        try:
            reply_index = 0
            while True:
                packet = self._read_packet(conn)
                if packet is None:
                    return
                packet_id, packet_type, body = packet
                if packet_type == TYPE_AUTH:
                    conn.sendall(encode_packet(packet_id, TYPE_RESPONSE_VALUE, ""))
                    continue
                if packet_type != TYPE_COMMAND:
                    continue
                if body == SENTINEL_BODY:
                    # The sentinel: one empty-bodied packet, exactly as Factorio.
                    conn.sendall(encode_packet(packet_id, TYPE_RESPONSE_VALUE, ""))
                    continue
                self.command_bodies.append(body)
                for stale_id, stale_body in self.extra_packets:
                    conn.sendall(encode_packet(stale_id, TYPE_RESPONSE_VALUE, stale_body))
                self.extra_packets = []
                chunks = self.replies[reply_index] if reply_index < len(self.replies) else [""]
                reply_index += 1
                for chunk in chunks:
                    conn.sendall(encode_packet(packet_id, TYPE_RESPONSE_VALUE, chunk))
                # Deliberately no terminator chunk -- this is the whole point.
        except OSError:
            return
        finally:
            conn.close()

    def close(self) -> None:
        self._sock.close()


@pytest.fixture
def server_factory():
    made: list[FakeRCONServer] = []

    def make(replies, extra_packets=None) -> FakeRCONServer:
        server = FakeRCONServer(replies, extra_packets)
        made.append(server)
        return server

    yield make
    for server in made:
        server.close()


def _client(server: FakeRCONServer) -> RCONClient:
    client = RCONClient(
        RCONEndpoint(host="127.0.0.1", port=server.port, password=PASSWORD), timeout=5.0
    )
    client.connect()
    return client


def test_single_packet_response_without_terminator(server_factory):
    """The case that used to cost 250 ms: one chunk and no terminator."""
    server = server_factory([['{"result": 1}']])
    client = _client(server)
    try:
        assert client.command("/c rcon.print(1)") == '{"result": 1}'
    finally:
        client.close()


def test_multi_packet_response_is_reassembled(server_factory):
    """A 32-tile observation will exceed the 4096-byte split, so this is normal."""
    server = server_factory([['{"a":', '"' + "x" * 50 + '",', '"b":2}']])
    client = _client(server)
    try:
        assert client.command("/c big()") == '{"a":"' + "x" * 50 + '","b":2}'
    finally:
        client.close()


def test_sentinel_is_sent_after_every_command(server_factory):
    server = server_factory([["ok"]])
    client = _client(server)
    try:
        client.command("/c thing()")
    finally:
        client.close()
    assert server.command_bodies == ["/c thing()"]


def test_stale_packet_from_an_abandoned_call_is_drained(server_factory):
    """A timeout used to poison the connection permanently.

    Every call reused one id, so a late reply arrived during the *next* call
    and raised "unexpected rcon packet id" from then on. Per-call ids make the
    late packet identifiable as stale, so it is dropped rather than fatal.
    """
    server = server_factory([["fresh"]], extra_packets=[(2, "reply to a long-dead call")])
    client = _client(server)
    try:
        assert client.command("/c thing()") == "fresh"
    finally:
        client.close()


def test_request_ids_advance_per_call(server_factory):
    server = server_factory([["a"], ["b"]])
    client = _client(server)
    try:
        first = client._allocate_ids()
        second = client._allocate_ids()
        assert second[0] > first[1], "command and sentinel ids must not collide"
        assert first[1] == first[0] + 1
    finally:
        client.close()


def test_expect_response_false_returns_without_waiting(server_factory):
    """`/quit` kills the server, so no sentinel reply is ever coming."""
    server = server_factory([])
    client = _client(server)
    try:
        assert client.command("/quit", expect_response=False) == ""
    finally:
        client.close()
