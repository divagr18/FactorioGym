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
        reorder: bool = False,
    ):
        #: One entry per non-sentinel command; each entry is the list of body
        #: chunks that command answers with.
        self.replies = replies
        #: (id, body) packets injected before the next reply, simulating a late
        #: response from a call the client already abandoned.
        self.extra_packets = extra_packets or []
        #: Answer the sentinel *before* the command it followed, which is what
        #: the real server does once a player is connected: a command too long
        #: to execute inline is deferred, and the trivial sentinel behind it
        #: runs first. Reproducing that is the only way to test the case
        #: engine-free, since a live Factorio cannot be asked to invert on
        #: demand -- it does it only when someone is watching.
        self.reorder = reorder
        self._held: list[tuple[int, list[str]]] = []
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
                    # Then whatever the preceding command was holding, which is
                    # the inversion the client has to survive.
                    for held_id, held_chunks in self._held:
                        for chunk in held_chunks:
                            conn.sendall(encode_packet(held_id, TYPE_RESPONSE_VALUE, chunk))
                    self._held = []
                    continue
                self.command_bodies.append(body)
                for stale_id, stale_body in self.extra_packets:
                    conn.sendall(encode_packet(stale_id, TYPE_RESPONSE_VALUE, stale_body))
                self.extra_packets = []
                chunks = self.replies[reply_index] if reply_index < len(self.replies) else [""]
                reply_index += 1
                if self.reorder:
                    self._held.append((packet_id, chunks))
                    continue
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

    def make(replies, extra_packets=None, reorder=False) -> FakeRCONServer:
        server = FakeRCONServer(replies, extra_packets, reorder=reorder)
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


# ---------------------------------------------------------------- reordering
#
# The bug that made watching a run in a real client look impossible, and the
# reason it looked like a property of the transport rather than of this loop.
#
# `command` sent a sentinel after every command and treated its reply as proof
# the real reply was complete, on the grounds that a Source RCON server answers
# in order on one connection. That holds only with **no players**. Attach a
# client and the order inverts for anything longer than a trivial command: the
# sentinel came back first, was read as an empty body, and the real reply
# arrived during the *next* call where its lower id marked it stale and it was
# drained. One lost reply per call, indefinitely, surfacing as
# `non-json rcon response: ''`.
#
# Measured on the live engine: 89 inversions across one watched run, 0 on every
# player-free run.


def test_a_sentinel_that_overtakes_its_command_still_yields_the_body(server_factory):
    """The regression. Before the fix this returned '' and lost the reply."""
    server = server_factory([['{"result": 1}']], reorder=True)
    client = _client(server)
    try:
        assert client.command("/silent-command return 1") == '{"result": 1}'
        assert client.reordered_replies == 1
    finally:
        client.close()


def test_every_call_keeps_its_own_body_when_every_reply_is_inverted(server_factory):
    """The failure mode was cross-call: a dropped reply reappeared in the next
    call as a stale packet. Three inverted calls in a row must each return
    their own answer, which is what proves nothing is being carried over."""
    server = server_factory([["first"], ["second"], ["third"]], reorder=True)
    client = _client(server)
    try:
        assert client.command("/silent-command a()") == "first"
        assert client.command("/silent-command b()") == "second"
        assert client.command("/silent-command c()") == "third"
        assert client.reordered_replies == 3
    finally:
        client.close()


def test_a_split_reply_survives_the_inversion_too(server_factory):
    """With the sentinel already seen it no longer bounds the read, so the loop
    would exit on the first chunk and truncate.

    This build never splits a reply -- 4 KB to 4 MB each came back as one
    packet -- so the drain finds nothing in practice. It exists because that is
    a fact about one build, and a truncated observation does not announce
    itself."""
    server = server_factory([['{"a":', '"' + "y" * 40 + '",', '"b":2}']], reorder=True)
    client = _client(server)
    try:
        assert client.command("/silent-command big()") == '{"a":"' + "y" * 40 + '","b":2}'
        assert client.reordered_replies == 1
    finally:
        client.close()


def test_an_in_order_server_reports_no_inversions(server_factory):
    """The counter is a signal, so it must stay silent when nothing is wrong:
    a non-zero count means something is connected to the worker."""
    server = server_factory([["ok"], ["ok"]])
    client = _client(server)
    try:
        client.command("/silent-command a()")
        client.command("/silent-command b()")
        assert client.reordered_replies == 0
    finally:
        client.close()


# ------------------------------------------------ one owner (roadmap A4.3)


def test_two_threads_sharing_a_client_each_get_their_own_answer(server_factory):
    """A4.3: "serialize game access through one owner."

    Without the lock this does not race, it *loses*: each call allocates its id
    pair from a plain counter and then reads until it has seen both of its own
    ids, draining any *lower* id as a stale packet from an abandoned call. The
    second thread's perfectly valid reply, arriving inside the first thread's
    read loop, was discarded as garbage and the second thread blocked until its
    timeout. A higher id raised instead. Neither outcome named the real cause.

    Sixteen calls across four threads, on the connection they share.
    """
    calls = 16
    server = server_factory([[f"answer-{index}"] for index in range(calls)])
    client = _client(server)
    answers: dict[int, str] = {}
    errors: list[BaseException] = []
    start = threading.Barrier(4)

    def caller(slots: range) -> None:
        try:
            start.wait(timeout=5)
            for slot in slots:
                answers[slot] = client.command(f"ask-{slot}")
        except BaseException as failure:  # noqa: BLE001 - reported, not raised here
            errors.append(failure)

    threads = [
        threading.Thread(target=caller, args=(range(offset, calls, 4),)) for offset in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20)
        assert not thread.is_alive(), "a caller never returned; a reply was lost"

    assert not errors, errors
    # Every call got *an* answer, and the answers are exactly the set the
    # server sent -- no duplicates, none dropped.
    assert len(answers) == calls
    assert sorted(answers.values()) == sorted(f"answer-{index}" for index in range(calls))
    client.close()


def test_the_connection_is_held_for_a_whole_exchange_not_just_the_send(server_factory):
    """The lock has to span the read loop, because the read loop is what
    consumes ids allocated by the send. Guarding allocation alone would leave
    exactly the defect it was meant to remove."""
    import inspect

    source = inspect.getsource(RCONClient.command)
    assert "with self._lock:" in source
    exchange = inspect.getsource(RCONClient._exchange)
    # Allocation and the read loop live together, inside the held section.
    assert "_allocate_ids()" in exchange
    assert "_read_packet()" in exchange
