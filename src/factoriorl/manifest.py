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
import os
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
    state = {"commit": commit, "dirty": bool(status)}
    if status:
        # A commit plus a boolean does not identify the code that ran. Several
        # results in this repository were produced on dirty trees, and the
        # working copy that produced them cannot be recovered from `dirty:
        # true`. The digest is over the tracked diff *and* the porcelain
        # status, so an untracked-but-loaded file changes it too, and two runs
        # on the same dirty tree share an identifier while two different dirty
        # trees do not.
        diff = run("diff", "HEAD") or ""
        payload = "\n--\n".join([status, diff]).encode()
        state["diff_digest"] = hashlib.sha256(payload).hexdigest()[:16]
        # Porcelain v1 is two status characters then the path, but a rename
        # reads `R  old -> new` and an unmerged entry pads differently, so the
        # status field is stripped rather than sliced at a fixed offset. A
        # fixed `line[3:]` silently ate the first character of every path.
        state["dirty_paths"] = sorted(
            line[2:].strip() for line in status.splitlines() if len(line) > 2
        )[:64]
    return state


def host_info() -> dict:
    """Measured, never hardcoded.

    PLAN.md names a Ryzen 5 5600 desktop as the training machine; the actual
    host is this laptop. A manifest that claimed the desktop's figures would be
    a false provenance record, so the host is read at runtime.
    """
    info = {
        "node": platform.node(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    # Recorded, never assumed. Every document in this repo claimed an RTX 4060
    # until torch was first asked: the machine actually has an RTX 3050 Laptop
    # with 4.3 GB. A manifest that inherits a hardware claim from prose is a
    # false provenance record.
    # Broad on purpose. Importing torch pulls in multi-GB CUDA DLLs, and on a
    # machine already near its commit limit that raises OSError
    # ("the paging file is too small"), not ImportError. Writing the run
    # manifest must never fail because the GPU could not be described -- losing
    # the record of a run is far worse than recording an unknown GPU.
    try:
        import torch

        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            info["gpu"] = {
                "name": properties.name,
                "vram_gb": round(properties.total_memory / 1e9, 1),
                "capability": f"sm_{properties.major}{properties.minor}",
                "count": torch.cuda.device_count(),
                "torch": torch.__version__,
            }
        else:
            info["gpu"] = {"name": None, "reason": "cuda unavailable"}
    except ImportError:
        info["gpu"] = {"name": None, "reason": "torch not installed"}
    except Exception as exc:  # noqa: BLE001 - see above
        info["gpu"] = {"name": None, "reason": f"{type(exc).__name__}: {exc}"}
    return info


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


def amend(run_id: str, patch: dict) -> Path | None:
    """Merge facts into an already-written manifest.

    The manifest is written before the first step so an interrupted run still
    has one, which means anything only knowable afterwards -- the checkpoint's
    hash, the seed streams each evaluated row actually used, how many episodes
    were excluded -- had nowhere to go and was simply absent. Those are exactly
    the fields R0.2 requires for two comparisons to be shown to have scored the
    same scenes.

    Merging one level deep, so `{"seeds": {...}}` extends the seeds block
    rather than replacing it. A missing manifest is not an error: a smoke run
    may never have written one, and losing provenance must not lose the run.
    """
    path = runs_dir() / run_id / "manifest.json"
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(document.get(key), dict):
            document[key] = {**document[key], **value}
        else:
            document[key] = value
    try:
        path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    except OSError:
        return None
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

    # The action space a checkpoint was trained against. Nothing compared this,
    # so a catalog edit that kept the template count produced a checkpoint that
    # loaded clean, ran, and meant something different -- the same silent
    # failure the extractor version exists to catch on the observation side.
    problems.extend(_catalog_problems(data))
    problems.extend(_model_problems(data))
    return {"run_id": run_id, "ok": not problems, "problems": problems}


def _model_problems(data: dict) -> list[str]:
    """A run that trained a model must say which architecture it trained.

    Conditional rather than in `REQUIRED_FIELDS`, and the condition is the
    point: `model` is deliberately present-but-null for a run with no policy --
    every language-model agent run is one -- so requiring it outright would
    fail manifests that are correct. Requiring it *when a model exists* closes
    the gap without inventing a defect in the runs that never had one.

    The gap was real. `extractor_version` was optional as far as this function
    was concerned, and `architecture_signature` is what actually decides
    whether a checkpoint loads: `GRID_FEATURES` went 64 to 128 in one commit
    and every checkpoint written before it is unloadable, which nothing
    recorded and nothing could detect from the manifest alone.
    """
    model = data.get("model")
    if not model:
        return []
    problems = []
    if model.get("extractor_version") is None:
        problems.append("model recorded without an extractor_version")
    if not model.get("architecture_signature"):
        # Not fatal for runs written before the field existed, so it is
        # reported as a distinct, quieter problem than a wrong one.
        problems.append(
            "model recorded without an architecture_signature (predates the field; "
            "loadability cannot be checked from this manifest)"
        )
    return problems


def _catalog_problems(data: dict) -> list[str]:
    recorded = (data.get("profiles") or {}).get("catalog_digest")
    task_id = (data.get("task") or {}).get("id")
    if not recorded or not task_id:
        return []
    try:
        from factoriorl import catalog as catalog_module
        from factoriorl.tasks import get as get_task

        spec = get_task(task_id).spec
        current = catalog_module.resolve(spec.catalog, spec.catalog_subset).digest()
    except Exception as exc:  # noqa: BLE001 - a renamed task must not crash verify
        return [f"catalog digest could not be re-derived for {task_id}: {exc}"]
    if recorded != current:
        return [f"action catalog changed for {task_id}: recorded {recorded}, current {current}"]
    return []
