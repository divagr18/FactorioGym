"""Worker supervision (DESIGN.md 1.3).

Responsibilities:

- configurable deadlines and liveness checks per worker;
- captured engine output (the worker's console log) attached to failures;
- forced-kill detection classified as an infrastructure failure;
- restart from a known state -- the failed worker's own save, copied into a
  fresh worker directory so the failed run's evidence is never overwritten.

Every knob in :class:`SupervisionPolicy` is read by something here; a policy
field that no code consults is a documentation defect, not a feature.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from factoriorl.errors import InfrastructureFailure, ProtocolError
from factoriorl.session import WorkerSession
from factoriorl.worker import WorkerHandle, WorkerManager, WorkerSpec


@dataclass(frozen=True)
class SupervisionPolicy:
    """Deadlines and retry behavior, all configurable and all consulted."""

    #: Socket deadline for one request; reaches RCONClient via WorkerSession.
    request_deadline_seconds: float = 15.0
    #: Ceiling for waiting out an in-flight advance.
    settle_deadline_seconds: float = 30.0
    #: How stale a liveness answer may be before ``ensure_alive`` re-probes.
    liveness_interval_seconds: float = 5.0
    #: Consecutive failed probes tolerated before a worker is declared dead.
    liveness_attempts: int = 3
    #: Pause between liveness attempts.
    liveness_backoff_seconds: float = 0.5
    #: Worker-id suffix for replacements, keeping the failed id intact.
    restart_prefix: str = "restart"


class SupervisedWorker:
    """A worker plus its session and supervision bookkeeping."""

    def __init__(
        self,
        manager: WorkerManager,
        handle: WorkerHandle,
        session: WorkerSession,
        policy: SupervisionPolicy,
    ) -> None:
        self.manager = manager
        self.handle = handle
        self.session = session
        self.policy = policy
        self.last_liveness: float = time.monotonic()

    @property
    def spec(self) -> WorkerSpec:
        return self.handle.spec

    def alive(self) -> bool:
        return self.handle.alive()

    def check_liveness(self) -> bool:
        """Probe with read-only status requests; False once attempts run out.

        Retries ``liveness_attempts`` times because a single timeout is a
        transport hiccup, not proof of death -- but a dead process is
        recognised immediately, without burning the retry budget.
        """
        for attempt in range(self.policy.liveness_attempts):
            if not self.handle.alive():
                return False
            try:
                self.session.status()
            except (ProtocolError, OSError):
                if attempt + 1 < self.policy.liveness_attempts:
                    time.sleep(self.policy.liveness_backoff_seconds)
                continue
            self.last_liveness = time.monotonic()
            return True
        return False

    def ensure_alive(self, force: bool = False) -> bool:
        """Re-probe only when the last answer is older than the interval.

        This is what makes ``liveness_interval_seconds`` meaningful: callers
        can invoke it per step without paying for a round trip every time.
        """
        age = time.monotonic() - self.last_liveness
        if not force and age < self.policy.liveness_interval_seconds:
            return True
        return self.check_liveness()

    def assert_alive(self) -> None:
        """Raise a classified infrastructure failure if the worker died."""
        if not self.handle.alive():
            tail = WorkerManager._log_tail(self.spec)
            raise InfrastructureFailure(
                f"worker {self.spec.worker_id} died "
                f"(rc={self.handle.process.returncode}); engine log tail:\n{tail}"
            )


class WorkerSupervisor:
    """Launch, monitor, and recover supervised workers."""

    def __init__(
        self,
        manager: WorkerManager | None = None,
        policy: SupervisionPolicy | None = None,
    ) -> None:
        self.manager = manager or WorkerManager()
        self.policy = policy or SupervisionPolicy()
        self._restart_counts: dict[str, int] = {}

    def start(
        self,
        worker_id: str,
        map_seed: int = 424242,
        save_source: Path | None = None,
    ) -> SupervisedWorker:
        handle = self.manager.launch(worker_id, map_seed=map_seed, save_source=save_source)
        session = WorkerSession(
            handle,
            timeout=self.policy.request_deadline_seconds,
            settle_timeout=self.policy.settle_deadline_seconds,
        )
        return SupervisedWorker(self.manager, handle, session, self.policy)

    def restart(
        self, failed: SupervisedWorker, note: str = "automatic restart"
    ) -> SupervisedWorker:
        """Restart a failed worker from its known state.

        The replacement is started from the failed worker's own save, so the
        world it resumes is the one that was interrupted -- not a fresh map
        that merely shares a seed. It gets a distinct worker id, so the failed
        run's directory (save, logs, evidence) is never overwritten, and its
        evidence is recorded before the replacement starts.
        """
        self.record_failure_evidence(failed, note)
        # Release the failed worker's process handle, connection, and ports
        # before claiming new ones. Its directory and evidence stay on disk;
        # only the live resources go back to the pool, so a run that restarts
        # repeatedly cannot exhaust the port range.
        self.shutdown(failed, preserve_evidence=True)
        base = failed.spec.worker_id
        count = self._restart_counts.get(base, 0) + 1
        self._restart_counts[base] = count
        replacement_id = f"{base}-{self.policy.restart_prefix}{count}"
        save = failed.spec.save_path if failed.spec.save_path.is_file() else None
        return self.start(replacement_id, map_seed=failed.spec.map_seed, save_source=save)

    def shutdown(self, worker: SupervisedWorker, preserve_evidence: bool = True) -> None:
        try:
            worker.session.close()
        except OSError:
            pass
        if preserve_evidence:
            self.manager.shutdown(worker.handle)
        else:
            self.manager.cleanup(worker.handle)

    def record_failure_evidence(self, worker: SupervisedWorker, note: str) -> Path:
        """Write a small failure record into the worker's own directory."""
        evidence_path = worker.spec.directory / "failure-evidence.json"
        evidence_path.write_text(
            json.dumps(
                {
                    "worker_id": worker.spec.worker_id,
                    "recorded_at": time.time(),
                    "note": note,
                    "process_alive": worker.alive(),
                    "return_code": worker.handle.process.returncode,
                    "ports": {
                        "game": worker.spec.ports.game,
                        "rcon": worker.spec.ports.rcon,
                    },
                    "engine_log_tail": WorkerManager._log_tail(worker.spec, 4000),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return evidence_path
