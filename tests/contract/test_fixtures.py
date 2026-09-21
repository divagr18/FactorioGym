"""Protocol fixtures are internally consistent (engine-free).

Two corpora live side by side:

* ``protocol_v1/`` is **frozen history**. It is not migrated and not deleted;
  it is validated only against the shape it was written with, and the engine
  suite sends one of its requests verbatim to prove a v1 request is now
  refused. That turns a dead directory into live evidence for DESIGN.md 1.1's
  "unknown protocol versions fail clearly" -- against a real historical
  version rather than an invented one.
* ``protocol_v2/`` is the live corpus, validated against the current Python
  types.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from factoriorl.protocol import (
    PROTOCOL_VERSION,
    ActionType,
    ErrorCode,
    Request,
    RequestType,
    ResultCode,
)

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures"
V1_DIR = FIXTURE_ROOT / "protocol_v1"
V2_DIR = FIXTURE_ROOT / "protocol_v2"


def _fixtures(directory: Path):
    return sorted(directory.glob("*.json"), key=lambda p: p.name)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _known_request_types() -> set[str]:
    return {t.value for t in RequestType}


def test_live_corpus_is_the_current_version():
    # v2 added the ten-action matrix, fused stepping, entity handles and
    # profiles. Bumping this is the deliberate act CONTRIBUTING.md describes:
    # both sides move together, the previous fixture corpus is kept, and a
    # refusal test proves the old version is now rejected.
    assert PROTOCOL_VERSION == 2
    assert V2_DIR.is_dir(), "the live fixture corpus must exist"


def test_frozen_v1_corpus_is_retained():
    """Deleting it would lose the only evidence that v1 is genuinely refused."""
    assert V1_DIR.is_dir()
    assert _fixtures(V1_DIR), "v1 fixtures must stay checked in"
    for path in _fixtures(V1_DIR):
        assert _load(path)["protocol"] == 1


def test_v2_covers_the_required_cases():
    names = {p.stem for p in _fixtures(V2_DIR)}
    # DESIGN.md 1.1: success, invalid action, ongoing operation, reset,
    # observation -- plus what v2 adds.
    assert {
        "advance_success",
        "step_success",
        "act_transfer_success",
        "act_unknown_action",
        "observe_snapshot",
        "reset_success",
        "describe_matrix",
        "stale_episode",
    } <= names, f"missing required fixtures: {sorted(names)}"


@pytest.mark.parametrize("path", _fixtures(V2_DIR), ids=lambda p: p.stem)
def test_v2_fixture_structure(path):
    fixture = _load(path)
    assert fixture["protocol"] == PROTOCOL_VERSION
    assert fixture.get("description"), "every fixture explains what it pins"
    request = fixture["request"]
    response = fixture["response"]

    # The bad-protocol fixture deliberately sends a foreign version.
    if response.get("error", {}).get("code") != "bad_protocol":
        assert request["protocol"] == PROTOCOL_VERSION
    assert response["protocol"] == PROTOCOL_VERSION

    if request["type"] in _known_request_types():
        Request(
            request_id="x",
            episode_id="ep-1",
            type=RequestType(request["type"]),
            payload=request.get("payload", {}),
        )
    else:
        # Unknown types appear only as negative fixtures: the worker must
        # answer unsupported, never execute.
        assert response["code"] == "unsupported", path.name

    assert response["code"] in {c.value for c in ResultCode}
    if "error" in response:
        assert response["error"]["code"] in {c.value for c in ErrorCode}


@pytest.mark.parametrize("path", _fixtures(V2_DIR), ids=lambda p: p.stem)
def test_v2_action_fixtures_name_real_actions(path):
    fixture = _load(path)
    payload = fixture["request"].get("payload") or {}
    if fixture["request"]["type"] != "act":
        return
    action = payload.get("action")
    known = {a.value for a in ActionType}
    if fixture["response"].get("error", {}).get("code") == "unknown_action":
        assert action not in known, "the negative fixture must name a non-action"
    else:
        assert action in known, f"{path.name} names an action that does not exist"


def test_ongoing_fixtures_declare_their_settled_half():
    """An ongoing operation is only pinned if both halves are."""
    for name in ("advance_success", "step_success"):
        fixture = _load(V2_DIR / f"{name}.json")
        assert fixture["response"]["result"]["status"] == "running"
        assert fixture["settled"]["result"]["status"] == "completed", (
            f"{name} must pin the settled response, not just the first answer"
        )
