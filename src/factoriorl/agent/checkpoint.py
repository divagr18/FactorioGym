"""Saves that are verified before they are reported.

Roadmap A1.3: save before the first action, every five wall-clock minutes, and
at termination; preserve the final save before stopping the worker; if saving
fails, record it and retain the latest verified checkpoint; **never report a
nonexistent final save.**

That last clause is why this module exists rather than a two-line call to
`session.save`. Three engine facts make a save something you have to check
rather than something you can assume:

**`game.server_save` is deferred and has no completion event.** The engine
executes it at the end of the tick. Nothing fires when the file is written, and
`mod/factoriorl/` cannot look: Lua has no `io` and no `game.file_exists`. So the
mod reports only that the request was *issued*, and the verifying happens here.

**Deferred means end-of-tick, and the world is often paused.** Under exact
stepping the world is paused between decisions, so a save issued there sits
forever. The session response carries `paused`; when it is true a single tick is
advanced so the deferred save can run.

**The worker directory is deleted at teardown.** `WorkerManager.cleanup` does
`shutil.rmtree` on it, and the engine writes saves under `write-data/saves/`. A
verified checkpoint is therefore copied into the *run* directory, which is what
"preserve the final save before stopping the worker" actually requires.

The stale-file trap
-------------------
`tools/capture.py` learned this for screenshots and left the reasoning in place:
a stale file under the expected name "would be indistinguishable from a fresh
capture". Same hazard here, worse consequence -- a checkpoint that is silently
the previous one would resume the wrong world. So every save gets a unique name,
and the file is required to *appear*, not merely to exist.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Roadmap A1.3's cadence.
DEFAULT_INTERVAL_SECONDS = 300.0

#: How long to wait for a deferred save to materialise. A large world takes
#: seconds, not minutes; beyond this something is wrong and saying so beats
#: blocking a run that has a wall-clock deadline.
DEFAULT_TIMEOUT_SECONDS = 120.0

#: Poll spacing, and how long the size must hold still before the file counts as
#: finished. The engine writes the zip incrementally, so "exists" is not "done".
_POLL_SECONDS = 0.25
_STABLE_READINGS = 3

#: Factorio save names become filenames. Anything else is refused rather than
#: sanitised: a silently renamed save is one a resume cannot find.
_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]{1,96}$")


class CheckpointError(RuntimeError):
    """A save that could not be verified. Never raised past `save()`."""


@dataclass
class Checkpointer:
    """Issues saves, verifies them, and keeps the last one that was real."""

    session: Any
    #: The worker's `write-data`; the engine writes saves to `saves/` under it.
    write_data: Path
    #: Where verified checkpoints are copied, so they outlive the worker.
    run_dir: Path
    interval_seconds: float = DEFAULT_INTERVAL_SECONDS
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS

    records: list[dict] = field(default_factory=list)
    #: The most recent checkpoint that was verified on disk. A later failure
    #: does not clear it -- A1.3: "retain the latest verified checkpoint".
    latest: dict | None = None
    _last_attempt: float | None = field(default=None, repr=False)

    @property
    def saves_dir(self) -> Path:
        return self.write_data / "saves"

    @property
    def destination(self) -> Path:
        return self.run_dir / "saves"

    def maybe_save(self, label: str = "periodic") -> dict | None:
        """Save if the interval has elapsed. Called once per decision."""
        now = time.monotonic()
        if self._last_attempt is not None and now - self._last_attempt < self.interval_seconds:
            return None
        return self.save(label)

    def save(self, label: str) -> dict:
        """Issue, wait, verify, and copy. Returns a record either way.

        Does not raise: a run that cannot save should carry on and say so, not
        die at minute twenty-nine holding a world it was about to lose anyway.
        """
        self._last_attempt = time.monotonic()
        started = time.perf_counter()
        name = self._name(label)
        record: dict = {"label": label, "name": name, "verified": False}
        try:
            record.update(self._save(name, record))
        except Exception as failure:  # noqa: BLE001 - recorded, never propagated
            record["error"] = f"{type(failure).__name__}: {failure}"
        record["wall_seconds"] = round(time.perf_counter() - started, 3)
        self.records.append(record)
        if record["verified"]:
            self.latest = record
        return record

    # ------------------------------------------------------------- internals

    def _name(self, label: str) -> str:
        # Unique per save. A fixed name would make a stale file from a previous
        # attempt indistinguishable from a fresh one, and resuming the wrong
        # world is the failure that matters here.
        stamp = time.strftime("%Y%m%dT%H%M%S")
        safe = re.sub(r"[^A-Za-z0-9._-]", "-", label)[:32] or "checkpoint"
        name = f"frrl-{safe}-{stamp}"
        if not _SAFE_NAME.match(name):
            raise CheckpointError(f"unsafe save name {name!r}")
        return name

    def _save(self, name: str, record: dict) -> dict:
        target = self.saves_dir / f"{name}.zip"
        # A file already under this name would be verified as though it were
        # ours. Unique names make this near-impossible; removing it makes it
        # impossible.
        if target.exists():
            target.unlink()

        issued = self.session.save(name)
        result = (issued.response.result or {}) if hasattr(issued, "response") else {}
        out: dict = {"issued_tick": result.get("tick"), "world_was_paused": result.get("paused")}
        if result.get("paused"):
            # `server_save` runs at end of tick, and a paused world never ends
            # one. Without this the save is issued and silently never happens.
            self.session.advance(1)
            out["advanced_a_tick"] = True

        path = self._await_file(target)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.destination.mkdir(parents=True, exist_ok=True)
        kept = self.destination / path.name
        # Copied out of the worker directory, which `WorkerManager.cleanup`
        # deletes wholesale at teardown.
        shutil.copy2(path, kept)
        out.update(
            {
                "verified": True,
                "path": str(kept),
                "engine_path": str(path),
                "bytes": kept.stat().st_size,
                "sha256": digest,
            }
        )
        return out

    def _await_file(self, target: Path) -> Path:
        """Wait for the save to appear *and* stop growing."""
        deadline = time.perf_counter() + self.timeout_seconds
        stable, previous = 0, -1
        while time.perf_counter() < deadline:
            if target.is_file():
                size = target.stat().st_size
                # "Exists" is not "finished": the engine writes the zip
                # incrementally, so a size read once could catch it mid-write and
                # copy out a truncated archive.
                stable = stable + 1 if size == previous and size > 0 else 0
                previous = size
                if stable >= _STABLE_READINGS:
                    return target
            time.sleep(_POLL_SECONDS)
        raise CheckpointError(
            f"no verified save at {target} after {self.timeout_seconds:.0f}s "
            f"(last size {previous} bytes). The request was issued; the file "
            f"never settled, so nothing is reported as saved"
        )

    def to_dict(self) -> dict:
        return {
            "interval_seconds": self.interval_seconds,
            "saves": list(self.records),
            "verified": sum(1 for record in self.records if record.get("verified")),
            "failed": sum(1 for record in self.records if not record.get("verified")),
            "latest_verified": self.latest,
            "accounting": (
                "a checkpoint is reported only after its file appeared, stopped "
                "growing and was copied out of the worker directory"
            ),
        }


def newest_checkpoint(run_dir: Path) -> Path | None:
    """The most recent verified save copied into `run_dir`, if any.

    By modification time, not by name. Names are `frrl-<label>-<stamp>.zip`, so
    sorting them lexicographically orders by *label* first -- `frrl-final-...`
    sorts before `frrl-initial-...` -- and a resume would silently pick up the
    world as it was before the run instead of after it.
    """
    saves = sorted((run_dir / "saves").glob("*.zip"), key=lambda path: path.stat().st_mtime)
    return saves[-1] if saves else None
