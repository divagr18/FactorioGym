"""On-disk cache for random-action baselines.

The random floor depends on the task, its version, the split, the master seed,
the episode budget and the resolved action catalog -- and on nothing else. It
does not depend on the model, yet every run and every sweep arm recomputed it,
which is half of all evaluation episodes. ``tools/feasibility_sweep.py`` and
``tools/shaping_comparison.py`` run many arms over the same handful of tasks and
paid for that floor once per arm.

Nothing here may import the training stack: ``tests/unit`` asserts the
environment side of the package stays free of torch and stable-baselines3.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from factoriorl import catalog
from factoriorl.paths import runtime_dir

#: Bump when the *meaning* of a stored baseline changes -- a different episode
#: termination rule, say -- so entries written by older code are not reused.
CACHE_VERSION = 1


def baselines_dir() -> Path:
    return runtime_dir() / "baselines"


def cache_key(task: Any, split: str, master_seed: int, episodes: int) -> str:
    """Everything that can move the floor, and nothing that cannot.

    Correctness is the whole point of this key. A baseline is the number every
    success rate is read against, so a stale one does not fail loudly -- it
    silently flatters or damns every comparison drawn against it, in a sweep
    whose arms all look internally consistent. The task version and the
    resolved catalog digest are therefore both in the key: editing a task's
    layouts, budgets or rewards moves its version, and adding or removing an
    action changes the catalog digest, and either one changes what a random
    agent achieves.
    """
    spec = task.spec
    resolved = catalog.resolve(spec.catalog, spec.catalog_subset)
    payload = json.dumps(
        {
            "cache_version": CACHE_VERSION,
            "task_id": spec.id,
            "task_version": spec.version,
            "split": split,
            "master_seed": master_seed,
            "episodes": episodes,
            "catalog_digest": resolved.digest(),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read(path: Path) -> dict | None:
    # A cache is an optimisation, never a dependency: a truncated or
    # half-written file must cost one recomputation, not the whole run.
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    baseline = data.get("baseline")
    return baseline if isinstance(baseline, dict) else None


def _write(path: Path, key: dict, baseline: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Written whole, then moved into place, so a crash mid-write leaves the
        # previous entry rather than a truncated one the next run must repair.
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps({"key": key, "baseline": baseline}, indent=2), encoding="utf-8"
        )
        temporary.replace(path)
    except OSError:
        # Failing to cache is not failing to evaluate.
        pass


def cached_random_baseline(
    task: Any,
    split: str,
    master_seed: int,
    episodes: int,
    compute,
) -> dict:
    """Return the cached floor for this key, or compute and store it.

    ``compute`` is a zero-argument callable returning the baseline dict; it is
    only invoked on a miss, which is the point -- on a hit no episodes are run
    at all. The returned dict carries ``cached`` so a result file says whether
    its floor was measured in that run or recalled.
    """
    digest = cache_key(task, split, master_seed, episodes)
    path = baselines_dir() / f"{digest}.json"
    hit = _read(path)
    if hit is not None:
        return {**hit, "cached": True}

    baseline = compute()
    _write(
        path,
        {
            "task_id": task.spec.id,
            "task_version": task.spec.version,
            "split": split,
            "master_seed": master_seed,
            "episodes": episodes,
            "catalog_digest": catalog.resolve(task.spec.catalog, task.spec.catalog_subset).digest(),
        },
        baseline,
    )
    return {**baseline, "cached": False}
