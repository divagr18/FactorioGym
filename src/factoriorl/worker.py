"""Worker process lifecycle (PLAN.md 0.2).

One Factorio headless server per worker: isolated write-data, mod directory,
save, ports, and logs. Startup failures are classified; close releases the
process and connection; cleanup touches only this worker's files/processes.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from factoriorl.engine_config import EngineConfig, resolve_engine_config
from factoriorl.errors import StartupFailure, StartupFailureKind
from factoriorl.modpack import package_mod
from factoriorl.paths import ports_dir
from factoriorl.protocol import Request, RequestType
from factoriorl.rcon import LuaError, RCONClient, RCONError, lua_string
from factoriorl.worker_config import WorkerPorts, WorkerSpec

LAUNCH_TIMEOUT_SECONDS = 120.0
SAVE_TIMEOUT_SECONDS = 90.0
RCON_POLL_INTERVAL = 0.25

#: Port pool: each worker takes an adjacent (game=even, rcon=game+1) pair.
#: Below the Windows dynamic range (49152+), so ephemeral sockets never land here.
PORT_BASE = 23400
PORT_PAIRS = 200

#: Engine log fragments meaning "the address we asked for was already taken".
PORT_BUSY_MARKERS = (
    "address is already in use",
    "is already in use",
    "bind: address in use",
    "failed to bind",
)

_next_port_offset: int | None = None


_job_handle: int | None = None


def _kill_on_close_job() -> int | None:
    """A Windows job object whose closure terminates every assigned process.

    PLAN.md 1.4 requires workers to close cleanly "after a failed parent run".
    Normal shutdown paths and ``atexit`` cover an orderly exit, but neither
    runs when the parent is killed outright -- and an orphaned engine keeps
    its ``write-data/.lock``, blocking the next worker in that directory.

    Windows closes every handle a dying process owns, whatever killed it. A
    job created with ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`` therefore reaps
    its assigned engines as soon as this process is gone. The handle is
    deliberately leaked for the lifetime of the process: closing it early
    would kill the workers it protects.

    Returns ``None`` off Windows (no supported equivalent yet); callers then
    fall back to explicit shutdown, which is the documented limitation.
    """
    global _job_handle
    if sys.platform != "win32":
        return None
    if _job_handle is not None:
        return _job_handle

    import ctypes
    from ctypes import wintypes

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
            ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.POINTER(ctypes.c_ulong)),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    job_object_limit_kill_on_job_close = 0x2000
    job_object_extended_limit_information = 9

    kernel32 = ctypes.windll.kernel32
    handle = kernel32.CreateJobObjectW(None, None)
    if not handle:
        return None
    limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    limits.BasicLimitInformation.LimitFlags = job_object_limit_kill_on_job_close
    if not kernel32.SetInformationJobObject(
        handle,
        job_object_extended_limit_information,
        ctypes.byref(limits),
        ctypes.sizeof(limits),
    ):
        kernel32.CloseHandle(handle)
        return None
    _job_handle = handle
    return _job_handle


def _assign_to_job(process: subprocess.Popen) -> bool:
    """Put a spawned engine under the kill-on-close job. Best effort."""
    job = _kill_on_close_job()
    if job is None:
        return False
    import ctypes

    process_set_quota = 0x0100
    process_terminate = 0x0001
    kernel32 = ctypes.windll.kernel32
    target = kernel32.OpenProcess(process_set_quota | process_terminate, False, process.pid)
    if not target:
        return False
    try:
        return bool(kernel32.AssignProcessToJobObject(job, target))
    finally:
        kernel32.CloseHandle(target)


def allocate_ports(base: int = PORT_BASE) -> WorkerPorts:
    """Reserve a free game+rcon port pair on localhost.

    Three separate races can hand one pair to two engines; all three are
    closed here (PLAN.md 1.3: ports are not accidentally shared across
    workers).

    * **Within a process** the offset advances monotonically. A freshly closed
      worker leaves sockets in TIME_WAIT and no probe distinguishes that from
      a free socket, so a pair is never reused within one run.
    * **Across processes** every port is claimed by an exclusive reservation
      file under ``runtime/ports/``. A second process -- another pytest run, a
      CLI worker, a second trainer -- skips ports this one holds instead of
      racing it to ``bind``. Claims whose owning process is gone are reaped,
      so a crash cannot leak the pool.
    * **Between probe and launch** the claim is taken *before* the bind probe
      and held until :func:`release_ports`, closing the window in which a
      concurrent allocator could take the port we just tested.

    The starting offset is randomized per process so two allocators do not
    walk the pool in lockstep, contending on every entry.
    """
    global _next_port_offset
    if _next_port_offset is None:
        _next_port_offset = random.randrange(PORT_PAIRS)
    for attempt in range(PORT_PAIRS):
        offset = (_next_port_offset + attempt) % PORT_PAIRS
        game_port = base + offset * 2
        rcon_port = game_port + 1
        if not _reserve_port(game_port):
            continue
        if not _reserve_port(rcon_port):
            _release_port(game_port)
            continue
        if _port_free(game_port) and _port_free(rcon_port):
            _next_port_offset = (offset + 1) % PORT_PAIRS
            return WorkerPorts(game=game_port, rcon=rcon_port)
        _release_port(game_port)
        _release_port(rcon_port)
    raise StartupFailure(
        StartupFailureKind.PORT_UNAVAILABLE,
        f"no free port pair in {base}-{base + PORT_PAIRS * 2 - 1} "
        f"after probing {PORT_PAIRS} pairs (reservations: {ports_dir()})",
    )


def release_ports(ports: WorkerPorts) -> None:
    """Return a pair to the pool. Idempotent, and only drops our own claims."""
    _release_port(ports.game)
    _release_port(ports.rcon)


def reservation_path(port: int) -> Path:
    return ports_dir() / f"{port}.port"


def _reserve_port(port: int) -> bool:
    """Claim ``port`` for this process, reaping a dead owner's claim first."""
    path = reservation_path(port)
    path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if not _reservation_is_stale(path):
                return False
            try:
                path.unlink()  # owner is gone; drop the claim and retry once
            except OSError:
                return False
            continue
        except OSError:
            return False
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid(), "claimed_at": time.time()}, handle)
        return True
    return False


def _release_port(port: int) -> None:
    path = reservation_path(port)
    try:
        if _reservation_owner(path) == os.getpid():
            path.unlink()
    except OSError:
        pass


def _reservation_owner(path: Path) -> int | None:
    try:
        return int(json.loads(path.read_text(encoding="utf-8"))["pid"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _reservation_is_stale(path: Path) -> bool:
    """A claim is stale once its owning process is gone.

    An unreadable or malformed claim counts as stale: it was written by a
    process that died mid-write, and keeping it would leak the port forever.
    """
    owner = _reservation_owner(path)
    if owner is None:
        return True
    return not _pid_alive(owner)


def _pid_alive(pid: int) -> bool:
    """True if ``pid`` names a live process.

    Errs toward "alive" whenever the probe is ambiguous, so an access-denied
    answer never steals a port from someone else's running worker.
    """
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        error_invalid_parameter = 87
        process_query_limited_information = 0x1000
        still_active = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
        if not handle:
            # Only ERROR_INVALID_PARAMETER reliably means "no such process";
            # access-denied means a live process we are not allowed to inspect.
            return kernel32.GetLastError() != error_invalid_parameter
        try:
            exit_code = ctypes.c_ulong()
            if kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return exit_code.value == still_active
            return True
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _port_free(port: int) -> bool:
    """Both TCP (RCON) and UDP (game) must bind. No SO_REUSEADDR: a TIME_WAIT
    or held socket is not free to Factorio."""
    for kind in (socket.SOCK_STREAM, socket.SOCK_DGRAM):
        with socket.socket(socket.AF_INET, kind) as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                return False
    return True


def create_reference_save(engine: EngineConfig, spec: WorkerSpec) -> None:
    """``factorio --create`` with deterministic map-gen settings."""
    cmd = [
        str(engine.executable),
        "--config",
        str(spec.config_ini),
        "--mod-directory",
        str(spec.mod_directory),
        "--create",
        str(spec.save_path),
        "--map-gen-settings",
        str(spec.map_gen_settings),
        "--map-settings",
        str(spec.map_settings),
        "--map-gen-seed",
        str(spec.map_seed),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=SAVE_TIMEOUT_SECONDS)
    except FileNotFoundError as exc:
        raise StartupFailure(StartupFailureKind.MISSING_EXECUTABLE, str(engine.executable)) from exc
    except subprocess.TimeoutExpired as exc:
        raise StartupFailure(StartupFailureKind.TIMEOUT, str(exc)) from exc
    if proc.returncode != 0 or not spec.save_path.is_file():
        output = (proc.stdout + proc.stderr)[-2000:]
        raise StartupFailure(StartupFailureKind.INVALID_SAVE, f"save creation failed: {output}")


@dataclass
class WorkerHandle:
    """A running worker: process, transport, and identity."""

    spec: WorkerSpec
    process: subprocess.Popen
    engine: EngineConfig
    started_at: float = field(default_factory=time.monotonic)

    @property
    def pid(self) -> int:
        return self.process.pid

    def alive(self) -> bool:
        return self.process.poll() is None


class WorkerManager:
    """Launch, connect, and tear down isolated workers."""

    def __init__(self, engine: EngineConfig | None = None) -> None:
        self.engine = engine or resolve_engine_config()
        self.engine.assert_compatible()

    def launch(
        self,
        worker_id: str,
        map_seed: int = 424242,
        create_save: bool = True,
        save_source: Path | None = None,
    ) -> WorkerHandle:
        """Bring up one isolated worker.

        ``save_source`` starts the worker from an existing save instead of
        generating a fresh world -- how :meth:`WorkerSupervisor.restart`
        resumes a failed run from its known state (PLAN.md 1.3). The save is
        copied into the new worker's own directory, so the failed run's
        directory stays untouched as evidence.
        """
        if not self.engine.executable.is_file():
            raise StartupFailure(StartupFailureKind.MISSING_EXECUTABLE, str(self.engine.executable))
        ports = allocate_ports()
        try:
            spec = WorkerSpec(worker_id=worker_id, ports=ports, map_seed=map_seed)
            spec.write_config_files()
            package_mod(spec.mod_directory)
            if save_source is not None:
                if not save_source.is_file():
                    raise StartupFailure(
                        StartupFailureKind.INVALID_SAVE, f"save source missing: {save_source}"
                    )
                shutil.copy2(save_source, spec.save_path)
            elif create_save:
                create_reference_save(self.engine, spec)
            if not spec.save_path.is_file():
                raise StartupFailure(
                    StartupFailureKind.INVALID_SAVE, f"no save to start: {spec.save_path}"
                )
            process = self._spawn(spec)
            handle = WorkerHandle(spec=spec, process=process, engine=self.engine)
            try:
                self._wait_for_rcon(handle)
            except Exception:
                self._kill_process(process)
                raise
            return handle
        except Exception:
            # A worker that never came up must not keep its ports claimed.
            release_ports(ports)
            raise

    def _spawn(self, spec: WorkerSpec) -> subprocess.Popen:
        cmd = [
            str(self.engine.executable),
            "--config",
            str(spec.config_ini),
            "--mod-directory",
            str(spec.mod_directory),
            "--start-server",
            str(spec.save_path),
            "--bind",
            f"127.0.0.1:{spec.ports.game}",
            "--rcon-bind",
            f"127.0.0.1:{spec.ports.rcon}",
            "--rcon-password",
            spec.rcon_password,
            "--server-settings",
            str(spec.server_settings),
            "--server-id",
            str(spec.server_id_file),
            "--console-log",
            str(spec.console_log),
            "--disable-audio",
        ]
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=str(spec.directory),
            )
        except OSError as exc:
            raise StartupFailure(StartupFailureKind.MISSING_EXECUTABLE, str(exc)) from exc
        _assign_to_job(process)
        spec.pid_file.write_text(
            json.dumps({"engine_pid": process.pid, "parent_pid": os.getpid()}, indent=2),
            encoding="utf-8",
        )
        return process

    def _wait_for_rcon(self, handle: WorkerHandle) -> None:
        deadline = time.monotonic() + LAUNCH_TIMEOUT_SECONDS
        last_error = "no connection attempt"
        while time.monotonic() < deadline:
            if not handle.alive():
                tail = self._log_tail(handle.spec)
                raise StartupFailure(
                    self._classify_engine_exit(tail),
                    f"engine exited during startup (rc={handle.process.returncode}); "
                    f"log tail: {tail}",
                )
            try:
                client = RCONClient(handle.spec.rcon_endpoint, timeout=2.0)
                client.connect()
                # Protocol handshake: status request must parse and carry our episode.
                raw = client.lua(
                    'return remote.call("frrl_bridge", "dispatch", '
                    + lua_string(_status_request_json())
                    + ")"
                )
                client.close()
                if not isinstance(raw, dict):
                    raise LuaError(f"handshake returned {raw!r}")
                if raw.get("code") != "ok":
                    raise StartupFailure(
                        StartupFailureKind.ENGINE_ERROR,
                        f"mod handshake failed: {raw}",
                    )
                return
            except (OSError, RCONError) as exc:
                last_error = str(exc)
                time.sleep(RCON_POLL_INTERVAL)
        raise StartupFailure(
            StartupFailureKind.TIMEOUT,
            f"rcon not reachable after {LAUNCH_TIMEOUT_SECONDS}s: {last_error}; "
            f"log tail: {self._log_tail(handle.spec)}",
        )

    @staticmethod
    def _classify_engine_exit(log_tail: str) -> StartupFailureKind:
        """Separate "the port was taken" from a generic engine error.

        PLAN.md 0.2 requires startup failures to name the unavailable port
        rather than collapsing every early exit into one opaque kind.
        """
        lowered = log_tail.lower()
        if any(marker in lowered for marker in PORT_BUSY_MARKERS):
            return StartupFailureKind.PORT_UNAVAILABLE
        return StartupFailureKind.ENGINE_ERROR

    @staticmethod
    def _log_tail(spec: WorkerSpec, chars: int = 1200) -> str:
        # --console-log copies the server log; the engine also always writes
        # write-data/factorio-current.log. Read whichever exists.
        for candidate in (spec.console_log, spec.write_data / "factorio-current.log"):
            try:
                text = candidate.read_text(encoding="utf-8", errors="replace")
                if text.strip():
                    return text[-chars:]
            except OSError:
                continue
        return "(no engine log)"

    @staticmethod
    def _kill_process(process: subprocess.Popen) -> None:
        if process.poll() is None:
            process.kill()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass

    def shutdown(self, handle: WorkerHandle) -> None:
        """Terminate the worker process and release its connection and ports."""
        if handle.process.poll() is None:
            # Ask the engine to exit via the console; fall back to kill.
            try:
                with RCONClient(handle.spec.rcon_endpoint, timeout=2.0) as client:
                    client.command("/quit")
            except (RCONError, OSError):
                self._kill_process(handle.process)
            try:
                handle.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self._kill_process(handle.process)
        # Only once the process is gone: while it lives it owns the sockets.
        release_ports(handle.spec.ports)

    def cleanup(self, handle: WorkerHandle) -> None:
        """Remove this worker's directory. Never touches other workers."""
        self.shutdown(handle)
        shutil.rmtree(handle.spec.directory, ignore_errors=True)


def _status_request_json() -> str:
    return Request(
        request_id="handshake-status",
        episode_id="",
        type=RequestType.STATUS,
    ).to_json()
