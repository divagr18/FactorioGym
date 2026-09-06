"""Engine lifecycle tests: stepping, actions, reset, supervision, multi-worker.

Requires a real Factorio worker (engine marker).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time

import pytest

from factoriorl.errors import InfrastructureFailure, StartupFailure
from factoriorl.paths import workspace_root
from factoriorl.supervisor import WorkerSupervisor
from factoriorl.worker import _pid_alive, reservation_path

pytestmark = pytest.mark.engine


def wid(base: str) -> str:
    """A worker id unique to this run.

    Worker directories outlive the run that made them, and an orphaned engine
    holds its write-data lock until something kills it. Reusing a fixed id
    would make a fresh worker fail on a stale lock and report the wrong cause.
    """
    return f"{base}-{os.getpid()}"


def test_exact_stepping_advances_and_no_drift(module_session):
    base = module_session.observe().response.result["absolute_tick"]
    for ticks in (1, 30, 120):
        timed = module_session.advance(ticks)
        assert timed.response.result["ticks_advanced"] == ticks
    # Mixed repeated sequence.
    sequence = (30, 1, 120, 30, 30) * 2
    for ticks in sequence:
        module_session.advance(ticks)
    final = module_session.observe().response.result["absolute_tick"]
    assert final == base + 1 + 30 + 120 + sum(sequence)


def test_idle_between_requests_does_not_advance(module_session):
    a = module_session.observe().response.result["absolute_tick"]
    time.sleep(0.5)
    b = module_session.observe().response.result["absolute_tick"]
    assert a == b


def test_observations_report_after_interval_completes(module_session):
    """advance() must block until the requested interval finished."""
    timed = module_session.advance(120)
    assert timed.response.result["status"] == "completed"
    obs = module_session.observe().response.result
    # Session tick must reflect the full advance, not a mid-advance sample.
    assert timed.response.tick is not None
    assert obs["absolute_tick"] >= 120


def test_collision_blocks_movement(module_session):
    module_session.reset()
    module_session.act("move", direction="north", ticks=600)
    module_session.advance(600)
    pos = module_session.observe().response.result["character"]["position"]
    # Wall row at y = -6 stops the character around y = -5.
    assert pos[1] > -5.6
    assert pos[0] == 0


def test_out_of_reach_transfer_fails(module_session):
    module_session.reset()
    # Move far away first (south), then try to reach the chest.
    module_session.act("move", direction="south", ticks=120)
    module_session.advance(120)
    resp = module_session.act(
        "transfer", **{"from": "src", "to": "character", "item": "iron-plate", "count": 1}
    )
    assert resp.response.code.value == "rejected"
    assert resp.response.error.code.value == "out_of_reach"


def test_transfer_conserves_items(module_session):
    module_session.reset()
    resp = module_session.act(
        "transfer", **{"from": "src", "to": "character", "item": "iron-plate", "count": 10}
    )
    assert resp.response.ok
    obs = module_session.observe().response.result
    total = (
        sum(obs["entities"]["src"]["contents"].values())
        + sum(obs["inventory"].values())
        + sum(obs["entities"]["dst"]["contents"].values())
    )
    assert total == 50


def test_reset_repeatability_ten_cycles(module_session):
    baseline = None
    for i in range(10):
        module_session.reset()
        module_session.act("move", direction="south", ticks=30)
        module_session.advance(30)
        module_session.act(
            "transfer", **{"from": "src", "to": "character", "item": "iron-plate", "count": 10}
        )
        obs = module_session.observe().response.result
        record = {
            "tick": obs["tick"],
            "pos": obs["character"]["position"],
            "inventory": obs["inventory"],
            "src": obs["entities"]["src"]["contents"],
            "dst": obs["entities"]["dst"]["contents"],
            "task": obs["task"],
        }
        if baseline is None:
            baseline = record
        else:
            assert record == baseline, f"cycle {i} diverged"


def test_kill_is_classified_infrastructure_failure(worker_manager):
    supervisor = WorkerSupervisor(manager=worker_manager)
    worker = supervisor.start(wid("t-kill-classify"))
    try:
        worker.handle.process.kill()
        worker.handle.process.wait(timeout=10)
        with pytest.raises(InfrastructureFailure):
            worker.assert_alive()
        evidence = supervisor.record_failure_evidence(worker, "forced kill test")
        assert evidence.is_file()
    finally:
        supervisor.shutdown(worker, preserve_evidence=True)


def test_restart_does_not_overwrite_failed_evidence(worker_manager):
    supervisor = WorkerSupervisor(manager=worker_manager)
    failed = supervisor.start(wid("t-restart-evidence"))
    failed_dir = failed.spec.directory
    supervisor.record_failure_evidence(failed, "simulated crash")
    failed.handle.process.kill()
    failed.handle.process.wait(timeout=10)
    replacement = supervisor.restart(failed)
    try:
        assert replacement.spec.directory != failed_dir
        assert (failed_dir / "failure-evidence.json").is_file()
        assert replacement.check_liveness()
    finally:
        supervisor.shutdown(replacement, preserve_evidence=False)
        worker_manager.cleanup(failed.handle)


def test_two_workers_independent(worker_manager):
    """Two pooled workers keep separate ports, directories, and episodes."""
    from factoriorl.pool import WorkerPool

    with WorkerPool(manager=worker_manager) as pool:
        w1 = pool.start(wid("t-multi-a"))
        w2 = pool.start(wid("t-multi-b"))
        assert w1.spec.ports != w2.spec.ports
        assert w1.spec.directory != w2.spec.directory
        assert set(pool.worker_ids) == {wid("t-multi-a"), wid("t-multi-b")}
        assert pool.heartbeat(force=True) == {wid("t-multi-a"): True, wid("t-multi-b"): True}
        # Divergent state: w1 moves, w2 transfers.
        w1.session.act("move", direction="east", ticks=30)
        w1.session.advance(30)
        w2.session.act(
            "transfer", **{"from": "src", "to": "character", "item": "iron-plate", "count": 25}
        )
        obs1 = w1.session.observe().response.result
        obs2 = w2.session.observe().response.result
        assert obs1["character"]["position"][0] > 2.0
        assert obs1["inventory"] == {}
        assert obs2["inventory"].get("iron-plate") == 25
        assert obs2["character"]["position"] == [0, 0]
        # Resetting one does not affect the other.
        w1.session.reset()
        obs1b = w1.session.observe().response.result
        obs2b = w2.session.observe().response.result
        assert obs1b["inventory"] == {}
        assert obs2b["inventory"].get("iron-plate") == 25
    # Leaving the pool closes both; their ports return to the pool.
    assert len(pool) == 0
    for worker in (w1, w2):
        assert not worker.handle.alive()
        assert not reservation_path(worker.spec.ports.game).exists()
        shutil.rmtree(worker.spec.directory, ignore_errors=True)


def test_workers_close_when_the_parent_run_fails(worker_manager, tmp_path):
    """PLAN.md 1.4: workers close cleanly after a *failed* parent run.

    The child starts a pooled worker and then exits through ``os._exit``, so
    no ``finally``, no ``atexit``, and no shutdown call of ours runs -- the
    same situation as a crashed trainer. Only the kill-on-close job object can
    reap the engine, which is exactly what this asserts.
    """
    if sys.platform != "win32":
        pytest.skip("kill-on-close job objects are Windows-only for now")

    handoff = tmp_path / "child.json"
    script = tmp_path / "crashing_parent.py"
    script.write_text(
        "import json, os, sys\n"
        "from factoriorl.pool import WorkerPool\n"
        "pool = WorkerPool()\n"
        "worker = pool.start(sys.argv[2])\n"
        "open(sys.argv[1], 'w').write(json.dumps({\n"
        "    'engine_pid': worker.handle.pid,\n"
        "    'directory': str(worker.spec.directory),\n"
        "}))\n"
        "os._exit(1)\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [sys.executable, str(script), str(handoff), wid("t-parent-crash")],
        cwd=str(workspace_root()),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert handoff.is_file(), f"child never started a worker: {completed.stderr[-2000:]}"
    child = json.loads(handoff.read_text(encoding="utf-8"))
    engine_pid = child["engine_pid"]
    try:
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline and _pid_alive(engine_pid):
            time.sleep(0.5)
        assert not _pid_alive(engine_pid), (
            f"engine {engine_pid} outlived its failed parent run; it would hold "
            "write-data/.lock and its ports forever"
        )
    finally:
        shutil.rmtree(child["directory"], ignore_errors=True)


def test_pool_restart_resumes_from_the_failed_workers_save(worker_manager):
    """A replacement starts from the failed worker's own save, not a new map."""
    from factoriorl.pool import WorkerPool

    with WorkerPool(manager=worker_manager) as pool:
        failed = pool.start(wid("t-pool-restart"))
        failed_dir = failed.spec.directory
        original_save = (failed_dir / "save.zip").read_bytes()
        failed.handle.process.kill()
        failed.handle.process.wait(timeout=30)

        replacement = pool.restart(wid("t-pool-restart"), note="killed for restart test")
        assert replacement.spec.directory != failed_dir
        assert replacement.spec.worker_id == wid("t-pool-restart") + "-restart1"
        # Byte-identical save: the known state was carried over, not regenerated.
        assert (replacement.spec.directory / "save.zip").read_bytes() == original_save
        # Evidence of the interrupted run survives the restart.
        assert (failed_dir / "failure-evidence.json").is_file()
        assert replacement.check_liveness()
        assert pool.worker_ids == (wid("t-pool-restart") + "-restart1",)
        # The failed worker's ports went back to the pool, so a run that
        # restarts repeatedly cannot exhaust the range.
        assert not reservation_path(failed.spec.ports.game).exists()
        assert not reservation_path(failed.spec.ports.rcon).exists()
    shutil.rmtree(failed_dir, ignore_errors=True)


def test_startup_failure_classification_missing_executable():
    from pathlib import Path

    from factoriorl.engine_config import EXPECTED_BUILD, EXPECTED_VERSION, EngineConfig
    from factoriorl.worker import WorkerManager

    bogus = EngineConfig(
        executable=Path("D:/FactorioRL/runtime/nonexistent/factorio.exe"),
        version=EXPECTED_VERSION,
        build=EXPECTED_BUILD,
    )
    manager = WorkerManager(engine=bogus)
    with pytest.raises(StartupFailure) as excinfo:
        manager.launch(wid("t-missing-exe"))
    assert excinfo.value.kind.value == "missing_executable"


def test_startup_failure_classification_invalid_save(worker_manager):
    """Save creation with invalid map-gen settings is classified invalid_save."""
    import shutil

    from factoriorl.errors import StartupFailureKind
    from factoriorl.worker import allocate_ports, create_reference_save, release_ports
    from factoriorl.worker_config import WorkerSpec

    ports = allocate_ports()
    spec = WorkerSpec(worker_id=wid("t-bad-save"), ports=ports)
    spec.write_config_files()
    # Break the map-gen settings: the engine refuses unparseable settings at
    # save creation and exits non-zero.
    spec.map_gen_settings.write_text("{invalid json", encoding="utf-8")
    try:
        with pytest.raises(StartupFailure) as excinfo:
            create_reference_save(worker_manager.engine, spec)
        assert excinfo.value.kind is StartupFailureKind.INVALID_SAVE
    finally:
        release_ports(ports)
        shutil.rmtree(spec.directory, ignore_errors=True)


def test_port_unavailable_detection(worker_manager, monkeypatch):
    """A held game port is reported as a port problem, not a mystery exit.

    The blocker takes a pair that ``allocate_ports`` actually issued, so the
    test can never collide with a stray worker from an earlier run -- the
    failure that sank this test in the first Phase 1 gate attempt, when it
    hardcoded port 23510.
    """
    import socket as socket_mod

    from factoriorl.errors import StartupFailureKind
    from factoriorl.paths import worker_dir
    from factoriorl.supervisor import WorkerSupervisor
    from factoriorl.worker import allocate_ports, release_ports

    worker_id = wid("t-port-busy")
    ports = allocate_ports()
    # Factorio's game port is UDP. No SO_REUSEADDR: the block must be real.
    blocker = socket_mod.socket(socket_mod.AF_INET, socket_mod.SOCK_DGRAM)
    try:
        blocker.bind(("127.0.0.1", ports.game))
        monkeypatch.setattr("factoriorl.worker.allocate_ports", lambda base=None: ports)
        supervisor = WorkerSupervisor(manager=worker_manager)
        with pytest.raises(StartupFailure) as excinfo:
            supervisor.start(worker_id)
        assert excinfo.value.kind is StartupFailureKind.PORT_UNAVAILABLE
        assert str(ports.game) in excinfo.value.message or "in use" in excinfo.value.message
    finally:
        blocker.close()
        release_ports(ports)
        shutil.rmtree(worker_dir(worker_id), ignore_errors=True)


def test_engine_version_pin_enforced():
    from pathlib import Path

    from factoriorl.engine_config import EngineConfig, StartupFailure

    old = EngineConfig(
        executable=Path("D:/Factorio/bin/x64/factorio.exe"),
        version="1.1.0",
        build=50000,
    )
    with pytest.raises(StartupFailure):
        old.assert_compatible()
