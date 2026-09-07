"""Run manifests (PLAN.md section 2).

    Every run records the game build, mod version, protocol version, task
    version, observation profile, assistance profile, reward configuration,
    model configuration, and seeds.

Before this the entire "run artifact" system was ``WorkerSpec.manifest()``'s
four keys plus a report nothing ever read back. A manifest that nothing
validates is decoration, so this ships with ``factoriorl runs show`` and
``runs verify``, and the Phase 3 gate reads manifests rather than only writing
them.

The manifest is written **before the first step**, so an interrupted run still
has one.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from factoriorl.modpack import mod_version
from factoriorl.paths import mod_source_dir, runtime_dir
from factoriorl.protocol import PROTOCOL_VERSION

MANIFEST_VERSION = 1

#: Fields PLAN.md section 2 requires. The schema test asserts each is present
#: and non-null, so "we record everything" is checkable rather than asserted.
REQUIRED_FIELDS = (
    "engine",
    "mod",
    "protocol",
    "task",
    "profiles",
    "reward",
    "seeds",
    "host",
)


def runs_dir() -> Path:
    return runtime_dir() / "runs"


def mod_source_digest() -> str:
    """Hash of the Lua sources, so a manifest pins the mod that produced it."""
    digest = hashlib.sha256()
    for path in sorted(mod_source_dir().rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(mod_source_dir()).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def config_digest(payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _git() -> dict:
    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], capture_output=True, text=True, timeout=10, check=False
            )
            return out.stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            return None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {"commit": commit, "dirty": bool(status)}


def host_info() -> dict:
    """Measured, never hardcoded.

    PLAN.md names a Ryzen 5 5600 desktop as the training machine; the actual
    host is this laptop. A manifest that claimed the desktop's figures would be
    a false provenance record, so the host is read at runtime.
    """
    return {
        "node": platform.node(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
    }


@dataclass
class RunManifest:
    run_id: str
    engine: dict
    task: dict
    profiles: dict
    reward: dict
    seeds: dict
    budgets: dict = field(default_factory=dict)
    workers: list[dict] = field(default_factory=list)
    model: dict | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "manifest_version": MANIFEST_VERSION,
            "run_id": self.run_id,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "host": host_info(),
            "git": _git(),
            "engine": self.engine,
            "mod": {"version": mod_version(), "source_digest": mod_source_digest()},
            "protocol": {"version": PROTOCOL_VERSION},
            "task": self.task,
            "profiles": self.profiles,
            "reward": self.reward,
            # Present-but-null until Phase 4 fills it, so its absence is never
            # mistaken for "this run had no model".
            "model": self.model,
            "seeds": self.seeds,
            "budgets": self.budgets,
            "workers": self.workers,
            **self.extra,
        }

    def write(self) -> Path:
        directory = runs_dir() / self.run_id
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "manifest.json"
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


def new_run_id(prefix: str = "run") -> str:
    stamp = time.strftime("%Y%m%dT%H%M%S")
    salt = hashlib.blake2b(str(time.time_ns()).encode(), digest_size=4).hexdigest()
    return f"{stamp}-{salt}" if prefix == "run" else f"{prefix}-{stamp}-{salt}"


def load(run_id: str) -> dict:
    return json.loads((runs_dir() / run_id / "manifest.json").read_text(encoding="utf-8"))


def list_runs(limit: int = 25) -> list[dict]:
    root = runs_dir()
    if not root.is_dir():
        return []
    rows = []
    for directory in sorted(root.iterdir(), reverse=True)[:limit]:
        manifest = directory / "manifest.json"
        if not manifest.is_file():
            continue
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except ValueError:
            continue
        rows.append(
            {
                "run_id": data.get("run_id", directory.name),
                "created_at": data.get("created_at"),
                "task": (data.get("task") or {}).get("id"),
                "task_version": (data.get("task") or {}).get("version"),
            }
        )
    return rows


def verify(run_id: str) -> dict:
    """Re-derive what can be re-derived and report every mismatch.

    This is what stops a manifest becoming decoration: the mod digest and
    protocol version are recomputed from the working tree, so a manifest that
    no longer describes the code it claims to says so.
    """
    data = load(run_id)
    problems: list[str] = []
    for key in REQUIRED_FIELDS:
        if data.get(key) in (None, {}, ""):
            problems.append(f"missing required field: {key}")

    recorded_mod = (data.get("mod") or {}).get("source_digest")
    current_mod = mod_source_digest()
    if recorded_mod and recorded_mod != current_mod:
        problems.append(f"mod source changed: recorded {recorded_mod}, current {current_mod}")

    recorded_protocol = (data.get("protocol") or {}).get("version")
    if recorded_protocol != PROTOCOL_VERSION:
        problems.append(
            f"protocol changed: recorded {recorded_protocol}, current {PROTOCOL_VERSION}"
        )
    return {"run_id": run_id, "ok": not problems, "problems": problems}
