"""Workspace path layout.

All worker runtime state (saves, generated configs, engine logs, script output)
lives under ``runtime/``, which is excluded from source control.
"""

from __future__ import annotations

from pathlib import Path


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


def runtime_dir() -> Path:
    return workspace_root() / "runtime"


def workers_dir() -> Path:
    return runtime_dir() / "workers"


def worker_dir(worker_id: str) -> Path:
    if not worker_id or any(sep in worker_id for sep in ("/", "\\", "..")):
        raise ValueError(f"invalid worker id: {worker_id!r}")
    return workers_dir() / worker_id


def ports_dir() -> Path:
    """Cross-process port reservations, one file per reserved port."""
    return runtime_dir() / "ports"


def mod_source_dir() -> Path:
    """Lua mod source tree, packaged into a zip for each worker's mod folder."""
    return workspace_root() / "mod"


def evidence_dir() -> Path:
    """Small, committed evidence artifacts (summaries, measurements)."""
    return workspace_root() / "docs" / "evidence"
