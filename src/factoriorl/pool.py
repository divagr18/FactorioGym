"""Multi-worker orchestration (DESIGN.md 1.4).

A :class:`WorkerPool` owns a set of supervised workers and guarantees they are
released -- ports, processes, and directories -- however the parent run ends.

Three layers cover "both close cleanly after a failed parent run", strongest
first:

1. **Kill-on-close job object** (Windows): every engine is assigned to a job
   owned by this process, so the OS terminates them even on a hard kill,
   where no Python code of ours gets to run. See
   :func:`factoriorl.worker._kill_on_close_job`.
2. **``atexit``**: covers an unhandled exception or ``sys.exit`` -- the
   interpreter still unwinds, so workers shut down in an orderly way and
   their evidence is preserved.
3. **Orphan reaping**: :func:`reap_orphans` finds worker directories whose
   parent process is gone but whose engine still runs, and terminates them.
   This is the recovery path for anything that escaped 1 and 2 (a previous
   run on a machine where the job object could not be created).

Isolation is a property of the workers themselves -- separate directories,
saves, ports, and episodes -- so the pool adds bookkeeping, not coupling:
resetting one worker cannot affect another.
"""

from __future__ import annotations

import atexit
import json
import subprocess
import weakref
from collections.abc import Iterator
from pathlib import Path

from factoriorl.paths import workers_dir
from factoriorl.supervisor import SupervisedWorker, SupervisionPolicy, WorkerSupervisor
from factoriorl.worker import WorkerManager, _pid_alive

#: Live pools, so the interpreter can close them if the run ends badly.
_live_pools: weakref.WeakSet[WorkerPool] = weakref.WeakSet()


def _close_live_pools() -> None:
    for pool in list(_live_pools):
        try:
            pool.close(preserve_evidence=True)
        except Exception:  # noqa: BLE001 - shutdown must never raise at exit
            pass


atexit.register(_close_live_pools)


class WorkerPool:
    """A registry of supervised workers with guaranteed teardown.

    Use as a context manager; the pool closes every worker it started on the
    way out, including when the body raised.
    """

    def __init__(
        self,
        supervisor: WorkerSupervisor | None = None,
        manager: WorkerManager | None = None,
        policy: SupervisionPolicy | None = None,
    ) -> None:
        self.supervisor = supervisor or WorkerSupervisor(manager=manager, policy=policy)
        self._workers: dict[str, SupervisedWorker] = {}
        _live_pools.add(self)

    # ------------------------------------------------------------ membership

    def start(self, worker_id: str, map_seed: int = 424242) -> SupervisedWorker:
        if worker_id in self._workers:
            raise ValueError(f"worker id already in this pool: {worker_id}")
        worker = self.supervisor.start(worker_id, map_seed=map_seed)
        self._workers[worker.spec.worker_id] = worker
        return worker

    def get(self, worker_id: str) -> SupervisedWorker:
        return self._workers[worker_id]

    @property
    def worker_ids(self) -> tuple[str, ...]:
        return tuple(self._workers)

    def __len__(self) -> int:
        return len(self._workers)

    def __iter__(self) -> Iterator[SupervisedWorker]:
        return iter(tuple(self._workers.values()))

    # ------------------------------------------------------------ health

    def heartbeat(self, force: bool = False) -> dict[str, bool]:
        """Liveness of every worker, honoring the policy's probe interval."""
        return {wid: worker.ensure_alive(force=force) for wid, worker in self._workers.items()}

    def restart(self, worker_id: str, note: str = "pool restart") -> SupervisedWorker:
        """Replace one worker from its known state, keeping the pool's shape.

        The replacement is registered under its own id; the failed worker
        leaves the pool but its directory and evidence stay on disk.
        """
        failed = self._workers.pop(worker_id)
        replacement = self.supervisor.restart(failed, note=note)
        self._workers[replacement.spec.worker_id] = replacement
        return replacement

    # ------------------------------------------------------------ teardown

    def shutdown(self, worker_id: str, preserve_evidence: bool = True) -> None:
        worker = self._workers.pop(worker_id, None)
        if worker is not None:
            self.supervisor.shutdown(worker, preserve_evidence=preserve_evidence)

    def close(self, preserve_evidence: bool = True) -> None:
        """Close every worker. Safe to call twice; never raises for one bad worker."""
        for worker_id in tuple(self._workers):
            try:
                self.shutdown(worker_id, preserve_evidence=preserve_evidence)
            except Exception:  # noqa: BLE001 - one failure must not strand the rest
                self._workers.pop(worker_id, None)
        _live_pools.discard(self)

    def __enter__(self) -> WorkerPool:
        return self

    def __exit__(self, *_exc) -> None:
        self.close(preserve_evidence=True)


def reap_orphans(kill: bool = True) -> list[dict]:
    """Find (and by default terminate) engines whose parent run is gone.

    A worker directory records both pids at launch. If the parent is dead and
    the engine is not, nothing will ever shut that engine down, and it keeps
    holding its ``write-data/.lock`` and its ports. Returns one record per
    orphan found, so a caller can report what it cleaned up.
    """
    orphans: list[dict] = []
    root = workers_dir()
    if not root.is_dir():
        return orphans
    for pid_file in sorted(root.glob("*/worker-pids.json")):
        try:
            pids = json.loads(pid_file.read_text(encoding="utf-8"))
            engine_pid = int(pids["engine_pid"])
            parent_pid = int(pids["parent_pid"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if _pid_alive(parent_pid) or not _pid_alive(engine_pid):
            continue
        record = {
            "worker_dir": str(pid_file.parent),
            "engine_pid": engine_pid,
            "parent_pid": parent_pid,
            "killed": False,
        }
        if kill:
            record["killed"] = _terminate(engine_pid)
        orphans.append(record)
    return orphans


def _terminate(pid: int) -> bool:
    """Terminate one process by pid. Only ever called for a verified orphan."""
    try:
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return not _pid_alive(pid)


def orphan_report() -> list[dict]:
    """Dry-run :func:`reap_orphans`: report without terminating anything."""
    return reap_orphans(kill=False)


def stale_worker_dirs() -> list[Path]:
    """Worker directories left behind by a run that is no longer alive."""
    root = workers_dir()
    if not root.is_dir():
        return []
    stale: list[Path] = []
    for pid_file in sorted(root.glob("*/worker-pids.json")):
        try:
            pids = json.loads(pid_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        parent = pids.get("parent_pid")
        if isinstance(parent, int) and not _pid_alive(parent):
            stale.append(pid_file.parent)
    return stale
