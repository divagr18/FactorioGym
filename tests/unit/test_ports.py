"""Port allocation and reservation (DESIGN.md 1.3: ports are not accidentally
shared across workers).

These run without Factorio on purpose. Port correctness used to be provable
only by launching engines, which made it the slowest and least-run part of the
suite -- and it was the one lifecycle test that failed the Phase 1 gate.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from factoriorl.errors import StartupFailureKind
from factoriorl.worker import (
    PORT_BASE,
    PORT_PAIRS,
    WorkerManager,
    _pid_alive,
    _reserve_port,
    allocate_ports,
    release_ports,
    reservation_path,
)


@pytest.fixture
def released():
    """Track allocations so a failing assertion cannot leak the pool."""
    taken = []
    yield taken
    for ports in taken:
        release_ports(ports)


def _dead_pid() -> int:
    """A pid that has certainly exited."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait(timeout=30)
    return proc.pid


def test_allocation_returns_adjacent_pair_in_pool(released):
    ports = allocate_ports()
    released.append(ports)
    assert ports.rcon == ports.game + 1
    assert PORT_BASE <= ports.game < PORT_BASE + PORT_PAIRS * 2
    assert ports.game % 2 == PORT_BASE % 2


def test_allocations_never_repeat_within_a_process(released):
    pairs = []
    for _ in range(5):
        ports = allocate_ports()
        released.append(ports)
        pairs.append((ports.game, ports.rcon))
    assert len(set(pairs)) == len(pairs)


def test_allocation_writes_and_releases_reservations():
    ports = allocate_ports()
    game_claim = reservation_path(ports.game)
    rcon_claim = reservation_path(ports.rcon)
    try:
        assert game_claim.is_file() and rcon_claim.is_file()
        assert json.loads(game_claim.read_text(encoding="utf-8"))["pid"] == os.getpid()
    finally:
        release_ports(ports)
    assert not game_claim.exists()
    assert not rcon_claim.exists()


def test_release_is_idempotent(released):
    ports = allocate_ports()
    release_ports(ports)
    release_ports(ports)  # must not raise
    assert not reservation_path(ports.game).exists()


def test_live_claim_blocks_that_port(released):
    """A port claimed by a live process is skipped, not raced for.

    This is the cross-process case the old module-global counter missed: a
    second Python process restarting at the same base would hand out ports
    this process is already using.
    """
    victim = allocate_ports()
    released.append(victim)
    # The claim is held by us, and we are alive, so re-reserving must fail.
    assert _reserve_port(victim.game) is False
    for _ in range(10):
        other = allocate_ports()
        released.append(other)
        assert other.game != victim.game
        assert other.rcon != victim.rcon


def test_stale_claim_from_dead_owner_is_reaped(released):
    ports = allocate_ports()
    released.append(ports)
    claim = reservation_path(ports.game)
    claim.write_text(json.dumps({"pid": _dead_pid(), "claimed_at": 0}), encoding="utf-8")
    # The owner is gone, so the claim must be reclaimable rather than leaked.
    assert _reserve_port(ports.game) is True
    assert json.loads(claim.read_text(encoding="utf-8"))["pid"] == os.getpid()


def test_malformed_claim_is_treated_as_stale(released):
    ports = allocate_ports()
    released.append(ports)
    claim = reservation_path(ports.game)
    claim.write_text("{ this is not json", encoding="utf-8")
    # Written by a process that died mid-write; keeping it leaks the port.
    assert _reserve_port(ports.game) is True


def test_pid_liveness_probe():
    assert _pid_alive(os.getpid()) is True
    assert _pid_alive(_dead_pid()) is False
    assert _pid_alive(0) is False
    assert _pid_alive(-1) is False


def test_bind_failure_is_classified_as_port_unavailable():
    """A bind failure must name the port problem, not a generic engine error."""
    tail = (
        "0.123 Info ServerMultiplayerManager.cpp:100: Quitting multiplayer connection.\n"
        "0.124 Error ServerSocket.cpp:39: Host address is already in use: 127.0.0.1:23400\n"
    )
    assert WorkerManager._classify_engine_exit(tail) is StartupFailureKind.PORT_UNAVAILABLE


def test_unrelated_engine_failure_is_not_classified_as_port_unavailable():
    tail = "0.9 Error Util.cpp:97: The mod FactorioRL Runtime caused a non-recoverable error.\n"
    assert WorkerManager._classify_engine_exit(tail) is StartupFailureKind.ENGINE_ERROR
