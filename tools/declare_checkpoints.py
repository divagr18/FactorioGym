"""Declare which checkpoints Phase 4's exit gate is about.

PLAN.md:710 — *"another run can reproduce the learning procedure and evaluate
**the provided checkpoints** without manual intervention"*. Until this file
existed, "the provided checkpoints" named nothing: `gate_phase4` scanned
gitignored `runtime/runs/`, sorted alphabetically and took the last three. On
this machine 78 directories hold a `model.zip`, so that window drifted as runs
accumulated — by 2026-09-09 it had moved off both runs the 2026-09-07 gate
certified and onto three `extractor_version: 1` runs that cannot load at all.
The gate's own smoke run writes a `gate…` prefix that sorts before every other,
so it never checked the checkpoint it had just produced. And `release/` is not
the source either: its single run records `checkpoint_bytes: null`.

What a declaration records, and why each field
----------------------------------------------
`architecture_signature` is the load-bearing one. It hashes every parameter's
name and shape, which is exactly what `MaskablePPO.load` compares, and unlike
`extractor_version` it is not orthogonal to loadability: of six version bumps
only one changed a width the loader cannot absorb (`GRID_FEATURES` 64 to 128),
four were semantics-only, and a `GOAL_ENCODING_VERSION` change bumped nothing
at all. A checkpoint recording version 3 loads cleanly against today's 7.

`mod_source_digest` and `catalog_digest` are recorded because `manifest.verify`
compares them against the working tree, so a declaration is only valid until
the mod or a catalog changes. That is deliberate — a checkpoint whose
environment moved underneath it should stop being described as verified — but
it means a declaration states *when* it was true, hence `declared_at_commit`.

Measured at the time of writing: **no run on this machine could be declared.**
All 78 fail `manifest.verify` on the current tree — the R4 mod edits moved
`mod_source_digest`, and every task's action catalog moved with R2.3 — and none
records an `architecture_signature` because the field did not exist. So the
first valid declaration has to come from a run made after this commit, which is
what the Phase 4 work does next.

Run: uv run python tools/declare_checkpoints.py <run_id> [<run_id> ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from factoriorl import manifest as manifest_module  # noqa: E402
from factoriorl.gate_phase4 import DECLARATION  # noqa: E402
from factoriorl.learn.policy import checkpoint_signature  # noqa: E402
from factoriorl.paths import evidence_dir  # noqa: E402


def describe(run_id: str) -> dict:
    """One declaration entry, plus every reason it might not deserve to be one."""
    directory = manifest_module.runs_dir() / run_id
    checkpoint = directory / "model.zip"
    entry: dict = {"run_id": run_id, "problems": []}
    if not checkpoint.is_file():
        entry["problems"].append("no model.zip")
        return entry

    data = manifest_module.load(run_id)
    task = data.get("task") or {}
    model = data.get("model") or {}
    entry.update(
        {
            "task": task.get("id"),
            "task_version": task.get("version"),
            "extractor_version": model.get("extractor_version"),
            "recorded_architecture_signature": model.get("architecture_signature"),
            "mod_source_digest": (data.get("mod") or {}).get("source_digest"),
            "catalog_digest": (data.get("profiles") or {}).get("catalog_digest"),
            "checkpoint_sha256": model.get("checkpoint_sha256")
            or (data.get("extra") or {}).get("checkpoint_sha256"),
        }
    )
    try:
        entry["architecture_signature"] = checkpoint_signature(checkpoint)
    except Exception as exc:  # noqa: BLE001
        entry["problems"].append(f"architecture unreadable: {exc}")

    # A declared checkpoint has to load, or the exit clause it exists to
    # satisfy is false by inspection.
    try:
        from sb3_contrib import MaskablePPO

        MaskablePPO.load(directory / "model", device="cpu")
    except Exception as exc:  # noqa: BLE001
        entry["problems"].append(f"does not load: {str(exc).strip().splitlines()[-1][:160]}")

    verification = manifest_module.verify(run_id)
    if not verification["ok"]:
        entry["problems"].extend(verification["problems"])

    recorded = entry.get("recorded_architecture_signature")
    actual = entry.get("architecture_signature")
    if recorded and actual and recorded != actual:
        entry["problems"].append(
            f"manifest records architecture {recorded} but the file carries {actual}"
        )
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", help="run ids under runtime/runs/")
    parser.add_argument(
        "--force",
        action="store_true",
        help="declare a run that has problems anyway. The problems stay in the "
        "file either way -- this only decides whether writing it is refused",
    )
    args = parser.parse_args()

    entries = [describe(run_id) for run_id in args.runs]
    for entry in entries:
        state = "ok" if not entry["problems"] else f"{len(entry['problems'])} problem(s)"
        print(f"{entry['run_id']}: {state}", flush=True)
        for problem in entry["problems"]:
            print(f"    - {problem}", flush=True)

    broken = [entry for entry in entries if entry["problems"]]
    if broken and not args.force:
        print(
            f"\nrefusing to declare {len(broken)} run(s) with problems. A declared "
            "checkpoint that does not load makes PLAN.md:710's exit clause false "
            "by inspection. Pass --force to record them anyway.",
            flush=True,
        )
        return 1

    body = {
        "declares": "the checkpoints PLAN.md:710's exit gate evaluates",
        "why": (
            "Before this file, `gate_phase4` scanned gitignored runtime/runs/, "
            "sorted alphabetically and took the last three, so its subject "
            "drifted as runs accumulated and it never checked its own smoke run."
        ),
        "declared_at_commit": manifest_module._git().get("commit"),
        "checkpoints": entries,
        "note": (
            "`architecture_signature` is what `MaskablePPO.load` compares; "
            "`extractor_version` is a declaration of intent and is orthogonal to "
            "loadability in both directions. `mod_source_digest` and "
            "`catalog_digest` are recorded because `manifest.verify` compares "
            "them against the working tree, so this declaration states when it "
            "was true."
        ),
    }
    path = evidence_dir() / DECLARATION
    path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {path} declaring {len(entries)} checkpoint(s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
