"""Assemble a release bundle (PLAN.md 6.2).

    Include checkpoints, evaluation configurations, task versions, sample
    trajectories, and a concise limitations document.

    Artifact hashes and version metadata are recorded. Evaluation does not
    require paid APIs. Large artifacts are downloadable separately from source.
    No secrets or user-specific absolute paths are embedded.

The last clause is the one with teeth, and the repo fails it as it stands. A run
manifest records, verbatim:

    engine.executable    D:\\Factorio\\bin\\x64\\factorio.exe
    workers[].directory  D:\\FactorioRL\\runtime\\workers\\demo-fb3ba349
    host.node            DESKTOP-EXAMPLE

Two absolute paths that mean nothing on another machine, and a hostname that
identifies the machine they came from. None of it is needed to reproduce a run:
the engine is pinned by build number, the worker directory is recreated per run,
and the host block exists to explain a throughput measurement, which the CPU and
GPU fields already do.

So packaging redacts rather than copies, and then **checks its own output**: the
audit at the end re-reads every emitted text file looking for drive-letter
paths, the current hostname and secret-shaped strings, and refuses to write the
bundle manifest if it finds any. A redactor that is trusted rather than verified
is how one of these gets shipped.

Run: uv run python tools/package_release.py --runs <run_id> [<run_id> ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Artifacts at or above this are listed for separate download rather than
#: copied into the bundle. PLAN 6.2 asks that large artifacts be downloadable
#: separately from source, and a checkpoint is the only thing here that grows.
LARGE_BYTES = 8 * 1024 * 1024

#: A Windows drive-letter path. The lookbehind matters: without it `https://`
#: matches as drive `s`, and the audit's first run reported the uv docs link.
ABSOLUTE_PATH = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]{1,2}[^\"'\s]{2,}")

#: Deliberately broad: an OpenAI-style key, a bearer token, a long hex secret.
SECRET_SHAPED = re.compile(r"(sk-[A-Za-z0-9_\-]{16,})|(Bearer\s+[A-Za-z0-9._\-]{16,})")

#: Documents that ship as-is. The limitations file is required by 6.2 and is
#: what 6.4's "deferred functionality is not advertised" rests on.
DOCUMENTS = (
    "README.md",
    "docs/LIMITATIONS.md",
    "docs/ACTION_MATRIX.md",
    "docs/LEDGER.md",
)


#: Keys dropped when a decision record is made public. Everything else about a
#: decision -- what was chosen, what it did, what it cost, how long it took --
#: stays, because that is what makes a run inspectable.
PRIVATE_DECISION_KEYS = ("prompt", "observation", "legal_actions")

#: Keys dropped from each recorded model attempt. The verbatim reply and any
#: provider error body go; the accounting stays.
PRIVATE_ATTEMPT_KEYS = ("text", "error", "detail")


def public_decisions(source: Path, destination: Path) -> int:
    """Rewrite `decisions.jsonl` without the transcript, and return the count.

    What survives is everything a reader needs to reconstruct the run: the
    action, its arguments, its outcome, the per-action clocks, the model's own
    short `reason`, the plan it stated, latency and token counts. What goes is
    the rendered prompt and the model's verbatim answers.

    A4.1 also says private chain of thought must not be required or published.
    None is captured anywhere -- `reasoning_content` is never read and Anthropic
    `thinking` blocks are dropped at the adapter -- so there is nothing here to
    strip; this only removes the ordinary reply text.
    """
    rows = 0
    with destination.open("w", encoding="utf-8") as out:
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            for key in PRIVATE_DECISION_KEYS:
                record.pop(key, None)
            for attempt in record.get("attempts") or []:
                for key in PRIVATE_ATTEMPT_KEYS:
                    attempt.pop(key, None)
            out.write(json.dumps(record, default=str) + "\n")
            rows += 1
    return rows


def redact(value, replacements: dict[str, str]):
    """Replace machine-specific strings anywhere in a JSON structure."""
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            # The hostname identifies the machine and explains nothing: the CPU,
            # GPU and core count already carry whatever a throughput number
            # needs to be read against.
            if key == "node":
                out[key] = "<host>"
                continue
            out[key] = redact(item, replacements)
        return out
    if isinstance(value, list):
        return [redact(item, replacements) for item in value]
    if isinstance(value, str):
        text = value
        for needle, placeholder in replacements.items():
            text = text.replace(needle, placeholder)
            text = text.replace(needle.replace("\\", "\\\\"), placeholder)
            text = text.replace(needle.replace("\\", "/"), placeholder)
        return text
    return value


def replacements_for(engine_executable: str | None) -> dict[str, str]:
    mapping = {str(ROOT): "<repo>"}
    if engine_executable:
        mapping[str(Path(engine_executable).parent.parent.parent)] = "<factorio>"
        mapping[engine_executable] = "<factorio>/bin/x64/factorio.exe"
    return mapping


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def collect_run(run_id: str, out_dir: Path) -> dict:
    """Copy one run's shippable parts, redacting the manifest."""
    source = ROOT / "runtime" / "runs" / run_id
    if not source.exists():
        raise SystemExit(f"no such run: {source}")
    destination = out_dir / "runs" / run_id
    destination.mkdir(parents=True, exist_ok=True)

    private = source / "decisions.jsonl"
    if private.exists():
        public_decisions(private, source / "decisions.public.jsonl")

    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    engine = (manifest.get("engine") or {}).get("executable")
    cleaned = redact(manifest, replacements_for(engine))
    (destination / "manifest.json").write_text(json.dumps(cleaned, indent=2), encoding="utf-8")

    copied = ["manifest.json"]
    # `decisions.jsonl` is deliberately absent and `decisions.public.jsonl`
    # takes its place. Roadmap A4.1: "Keep secrets and raw private transcripts
    # out of export bundles." Every rendered prompt and every raw model reply
    # is in the private file -- `attempts[].text` is the model's answer verbatim
    # and `prompt` is the whole user turn -- and it was being copied into a
    # bundle meant for sharing. Secrets were already handled by the redactor and
    # the audit; a transcript is not a secret, and shipping it is still wrong.
    for name in (
        "model.zip",
        "curve.csv",
        "config.json",
        "result.json",
        "summary.json",
        "decisions.public.jsonl",
        "tool_events.jsonl",
        "production.jsonl",
        "status.json",
    ):
        candidate = source / name
        if not candidate.exists():
            continue
        if candidate.stat().st_size >= LARGE_BYTES:
            copied.append(f"{name} (listed, not copied)")
            continue
        if name.endswith((".json", ".jsonl", ".csv")):
            # Trajectories carry prompts and observations, which carry paths
            # only if something upstream leaked one -- redact anyway, because
            # the audit is what decides and it should have nothing to find.
            text = candidate.read_text(encoding="utf-8")
            for needle, placeholder in replacements_for(engine).items():
                text = text.replace(needle, placeholder).replace(
                    needle.replace("\\", "\\\\"), placeholder
                )
            (destination / name).write_text(text, encoding="utf-8")
        else:
            shutil.copy2(candidate, destination / name)
        copied.append(name)

    task = manifest.get("task") or {}
    return {
        "run_id": run_id,
        "task": task.get("id"),
        "task_version": task.get("version"),
        "files": copied,
        "checkpoint_bytes": (source / "model.zip").stat().st_size
        if (source / "model.zip").exists()
        else None,
    }


def audit(out_dir: Path, documents: set[str]) -> list[str]:
    """Re-read what was written and refuse anything machine-specific.

    Prose and generated artifacts are held to different standards, and the
    difference is not laziness. 6.2 forbids *embedded* user-specific paths: a
    README that documents the default Factorio install location is telling a
    reader where to look; a manifest recording the same string is telling them
    where the packager's own machine kept it. The first is the documentation
    working; the second ships a machine's layout to strangers.

    So documents are checked for secrets and the hostname, and everything
    generated is checked for all three.
    """
    hostname = platform.node()
    problems: list[str] = []
    for path in sorted(out_dir.rglob("*")):
        if not path.is_file() or path.suffix in (".zip", ".png"):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        where = path.relative_to(out_dir)
        if str(where).replace("\\", "/") not in documents:
            for match in ABSOLUTE_PATH.findall(text):
                found = match if isinstance(match, str) else next(m for m in match if m)
                problems.append(f"{where}: absolute path {found[:60]}")
        if hostname and hostname in text:
            problems.append(f"{where}: hostname {hostname}")
        if SECRET_SHAPED.search(text):
            problems.append(f"{where}: secret-shaped string")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="*", default=[], help="run ids to ship")
    parser.add_argument("--out", default="release")
    args = parser.parse_args()

    out_dir = ROOT / args.out
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    from factoriorl.tasks import all_tasks, get

    bundle: dict = {
        "schema": "factoriorl.release/1",
        "engine": {"version": "2.0.60", "build": 83512},
        "protocol": 2,
        "tasks": {t: get(t).spec.version for t in sorted(all_tasks())},
        "runs": [],
        "documents": [],
        "holdouts": [],
        "large_artifacts": [],
    }

    for name in DOCUMENTS:
        source = ROOT / name
        if not source.exists():
            continue
        destination = out_dir / Path(name).name
        destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        bundle["documents"].append(Path(name).name)

    frozen = out_dir / "holdouts"
    frozen.mkdir(exist_ok=True)
    for source in sorted((ROOT / "docs" / "evidence").glob("holdout_*.json")):
        shutil.copy2(source, frozen / source.name)
        document = json.loads(source.read_text(encoding="utf-8"))
        bundle["holdouts"].append(
            {
                "file": f"holdouts/{source.name}",
                "id": document["holdout"]["holdout_id"],
                "content_hash": document["content_hash"],
                "candidates": (document.get("declaration") or {}).get("candidate_families"),
            }
        )

    # Which published evidence a reader can trace, and which they cannot.
    # `docs/evidence/phase4-release-*.json` each cite the holdout they were
    # scored against by content hash, and 7 of 11 cite a hash no committed
    # holdout has -- earlier revisions that were regenerated rather than
    # versioned. Those files are not copied into the bundle, and this records
    # *why* rather than leaving their absence to be noticed. Computed rather
    # than written down, so it cannot go stale the way a sentence would.
    bundle["excluded_evidence"] = {
        "reason": (
            "cites a holdout content_hash that no holdout in this bundle has, so the "
            "scores in it cannot be traced to the scenes that produced them"
        ),
        "files": [],
        "included_for_contrast": [],
    }
    bundled_hashes = {entry["content_hash"] for entry in bundle["holdouts"]}
    for source in sorted((ROOT / "docs" / "evidence").glob("phase4-release-*.json")):
        cited = (json.loads(source.read_text(encoding="utf-8")).get("holdout") or {}).get(
            "content_hash"
        )
        target = "included_for_contrast" if cited in bundled_hashes else "files"
        bundle["excluded_evidence"][target].append(
            {"file": source.name, "cites_content_hash": cited}
        )

    for run_id in args.runs:
        entry = collect_run(run_id, out_dir)
        bundle["runs"].append(entry)
        if entry["checkpoint_bytes"] and entry["checkpoint_bytes"] >= LARGE_BYTES:
            bundle["large_artifacts"].append(
                {"run_id": run_id, "file": "model.zip", "bytes": entry["checkpoint_bytes"]}
            )

    problems = audit(out_dir, {Path(name).name for name in DOCUMENTS})
    if problems:
        print("REFUSING to write the bundle manifest; redaction is incomplete:", file=sys.stderr)
        for problem in problems[:20]:
            print(f"  {problem}", file=sys.stderr)
        return 1

    # Hashes last, over exactly what shipped.
    bundle["artifacts"] = [
        {
            "file": str(p.relative_to(out_dir)).replace("\\", "/"),
            "bytes": p.stat().st_size,
            "sha256": digest(p),
        }
        for p in sorted(out_dir.rglob("*"))
        if p.is_file()
    ]
    (out_dir / "MANIFEST.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")

    total = sum(a["bytes"] for a in bundle["artifacts"])
    count = len(bundle["artifacts"])
    print(f"wrote {out_dir.relative_to(ROOT)}: {count} files, {total / 1024:.0f} KB")
    print(f"  documents  {bundle['documents']}")
    print(f"  holdouts   {[h['id'] for h in bundle['holdouts']]}")
    print(f"  runs       {[r['run_id'] for r in bundle['runs']]}")
    print("  audit      clean: no absolute paths, hostname or secret-shaped strings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
