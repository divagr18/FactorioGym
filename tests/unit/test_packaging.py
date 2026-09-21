"""Release packaging (DESIGN.md 6.2), checked without an engine."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import package_release as pkg  # noqa: E402

WINDOWS_PATH = "D:" + "\\" + "FactorioRL" + "\\" + "runtime" + "\\" + "workers"
ENGINE_PATH = "D:" + "\\" + "Factorio" + "\\" + "bin" + "\\" + "x64" + "\\" + "factorio.exe"


def test_a_manifest_loses_the_machine_it_came_from():
    """Two absolute paths and a hostname, none of which reproduce anything.

    The engine is pinned by build number, the worker directory is recreated per
    run, and the host block exists to explain a throughput measurement -- which
    the CPU and GPU fields already carry.
    """
    manifest = {
        "engine": {"executable": ENGINE_PATH, "build": 83512},
        "workers": [{"directory": str(ROOT / "runtime" / "workers" / "w1")}],
        "host": {"node": "DESKTOP-EXAMPLE", "cpu_count": 16},
    }
    cleaned = pkg.redact(manifest, pkg.replacements_for(ENGINE_PATH))
    text = json.dumps(cleaned)
    assert "DESKTOP-EXAMPLE" not in text
    assert "Factorio" + "\\" + "bin" not in text
    assert str(ROOT) not in text
    # And keeps what a reader actually needs.
    assert cleaned["engine"]["build"] == 83512
    assert cleaned["host"]["cpu_count"] == 16


def test_a_windows_path_is_matched_and_a_url_is_not():
    """Two ways to get this wrong, and both happened.

    `[\\/]` is an escaped forward slash, so the first pattern matched only `/`
    and never saw a Windows path at all -- it reported a clean audit for the
    wrong reason. And without the lookbehind, `https://` matches as drive `s`,
    which made the audit's first real run flag the uv documentation link.
    """
    assert pkg.ABSOLUTE_PATH.search(WINDOWS_PATH)
    assert pkg.ABSOLUTE_PATH.search("D:/FactorioGym/runtime")
    assert not pkg.ABSOLUTE_PATH.search("see https://docs.astral.sh/uv/ for uv")
    assert not pkg.ABSOLUTE_PATH.search('"base_url": "https://api.openai.com/v1"')


def test_the_audit_refuses_a_generated_artifact_that_kept_a_path(tmp_path):
    bundle = tmp_path / "release"
    (bundle / "runs").mkdir(parents=True)
    (bundle / "runs" / "manifest.json").write_text(
        json.dumps({"directory": WINDOWS_PATH}), encoding="utf-8"
    )
    problems = pkg.audit(bundle, documents=set())
    assert problems and "absolute path" in problems[0]


def test_the_audit_allows_a_document_to_document_a_path(tmp_path):
    """6.2 forbids *embedded* user-specific paths. A README naming the default
    Factorio install location is the documentation working; a manifest naming
    the same string ships a machine's layout to strangers."""
    bundle = tmp_path / "release"
    bundle.mkdir(parents=True)
    (bundle / "README.md").write_text(f"default engine: {ENGINE_PATH}", encoding="utf-8")
    assert pkg.audit(bundle, documents={"README.md"}) == []
    assert pkg.audit(bundle, documents=set()) != []


def test_the_audit_refuses_a_hostname_or_a_secret_even_in_a_document(tmp_path):
    bundle = tmp_path / "release"
    bundle.mkdir(parents=True)
    (bundle / "README.md").write_text(f"built on {platform.node()}", encoding="utf-8")
    assert any("hostname" in p for p in pkg.audit(bundle, documents={"README.md"}))

    (bundle / "README.md").write_text(
        "authorization: Bearer abcdefghijklmnopqrstuvwxyz012345", encoding="utf-8"
    )
    assert any("secret" in p for p in pkg.audit(bundle, documents={"README.md"}))


def test_the_limitations_document_is_part_of_a_release():
    """6.2 lists it explicitly, and 6.4's honesty clause rests on it."""
    assert "docs/LIMITATIONS.md" in pkg.DOCUMENTS


# ---------------------------------------- A4.1: transcripts stay out of bundles


def _decision_line() -> str:
    import json

    return json.dumps(
        {
            "episode": 0,
            "step": 3,
            "action_key": "place_at",
            "action_index": 7,
            "arguments": {"position": [1.5, 2.5]},
            "reason": "put the furnace by the ore",
            "plan": "smelt iron",
            "result": {"action_status": "completed"},
            "actions": [{"position": 0, "key": "place_at", "status": "completed"}],
            "observation_block": "TASK open_factory ... the whole rendered user turn",
            "observation": {"entities": [{"h": "h1"}]},
            "legal_actions": [{"index": 7, "key": "place_at"}],
            "attempts": [
                {
                    "attempt": 1,
                    "latency_ms": 812.0,
                    "usage": {"total_tokens": 4211},
                    "text": '{"action": 7, "reason": "put the furnace by the ore"}',
                    "error": None,
                }
            ],
        }
    )


def test_the_public_decision_record_drops_the_transcript(tmp_path):
    """A4.1: keep raw private transcripts out of export bundles.

    `decisions.jsonl` carries every rendered prompt and every raw model reply,
    and it was copied into the bundle verbatim. Secrets were already handled by
    the redactor and the audit -- a transcript is not a secret, and shipping it
    is still wrong.
    """
    import json

    source = tmp_path / "decisions.jsonl"
    source.write_text(_decision_line() + "\n", encoding="utf-8")
    destination = tmp_path / "decisions.public.jsonl"

    assert pkg.public_decisions(source, destination) == 1
    record = json.loads(destination.read_text(encoding="utf-8").strip())

    # Gone: the prompt, the observation, and the model's verbatim answer.
    assert "observation_block" not in record
    assert "observation" not in record
    assert "legal_actions" not in record
    assert "text" not in record["attempts"][0]
    assert "the whole rendered user turn" not in destination.read_text(encoding="utf-8")

    # Kept: everything that makes the run inspectable.
    assert record["action_key"] == "place_at"
    assert record["arguments"] == {"position": [1.5, 2.5]}
    assert record["reason"] == "put the furnace by the ore"
    assert record["plan"] == "smelt iron"
    assert record["actions"][0]["status"] == "completed"
    assert record["attempts"][0]["usage"]["total_tokens"] == 4211
    assert record["attempts"][0]["latency_ms"] == 812.0


def test_the_bundle_allowlist_names_the_public_file_and_not_the_private_one():
    """The allowlist is the whole mechanism -- there is no denylist, so a file
    ships if and only if it is named here."""
    import inspect

    source = inspect.getsource(pkg)
    allowlist = source[source.index('copied = ["manifest.json"]') :]
    allowlist = allowlist[: allowlist.index("candidate = source / name")]

    assert '"decisions.public.jsonl"' in allowlist
    assert '"decisions.jsonl"' not in allowlist
    # And the A4.1 artifacts do ship, because a bundle without them cannot be
    # reconciled against the claims made from it.
    for name in ("config.json", "summary.json", "tool_events.jsonl", "production.jsonl"):
        assert f'"{name}"' in allowlist, name
