"""Contract tests: protocol v1 fixtures are internally consistent.

Engine-free: validates the frozen fixtures against the Python protocol types
so Python and Lua can only disagree at the wire, which the engine suite
covers. Runs without Factorio.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from factoriorl.protocol import (
    PROTOCOL_VERSION,
    ErrorCode,
    Request,
    RequestType,
    ResultCode,
)

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "protocol_v1"


def _fixtures():
    return sorted(FIXTURE_DIR.glob("*.json"), key=lambda p: p.name)


def test_fixture_directory_is_populated():
    names = {p.stem for p in _fixtures()}
    # PLAN.md 1.1: success, invalid action, ongoing action, reset, observation.
    assert {
        "advance_success",
        "act_transfer_success",
        "act_unknown_action",
        "observe_snapshot",
        "reset_success",
    } <= names


def _known_request_types():
    return {t.value for t in RequestType}


@pytest.mark.parametrize("path", _fixtures(), ids=lambda p: p.stem)
def test_fixture_structure(path):
    fixture = json.loads(path.read_text(encoding="utf-8"))
    assert fixture["protocol"] == PROTOCOL_VERSION
    request = fixture["request"]
    response = fixture["response"]
    # The bad-protocol fixture deliberately sends a foreign protocol version;
    # every other request speaks the frozen one.
    if response.get("error", {}).get("code") != "bad_protocol":
        assert request["protocol"] == PROTOCOL_VERSION
    assert response["protocol"] == PROTOCOL_VERSION
    if request["type"] in _known_request_types():
        # Request parses into the Python type.
        Request(
            request_id="x",
            episode_id="ep-1",
            type=RequestType(request["type"]),
            payload=request.get("payload", {}),
        )
    else:
        # Unknown types only appear as negative fixtures: the worker must
        # answer unsupported, never execute.
        assert response["code"] == "unsupported", path.name
    # Response code is in the frozen vocabulary.
    assert response["code"] in {c.value for c in ResultCode}
    if "error" in response:
        assert response["error"]["code"] in {c.value for c in ErrorCode}


@pytest.mark.parametrize("path", _fixtures(), ids=lambda p: p.stem)
def test_fixture_request_serializes_roundtrip(path):
    fixture = json.loads(path.read_text(encoding="utf-8"))
    request_body = dict(fixture["request"])
    if request_body["type"] not in _known_request_types():
        pytest.skip("negative fixture: unknown request type by design")
    request = Request(
        request_id="r-1",
        episode_id="ep-1",
        type=RequestType(request_body["type"]),
        payload=request_body.get("payload", {}),
    )
    raw = json.loads(request.to_json())
    assert raw["type"] == request_body["type"]
    assert raw["payload"] == request_body.get("payload", {})
    assert raw["protocol"] == PROTOCOL_VERSION


def test_negative_fixture_uses_unknown_type():
    """The unknown-type fixture exercises the closed dispatch table."""
    path = FIXTURE_DIR / "error_unknown_request_type.json"
    fixture = json.loads(path.read_text(encoding="utf-8"))
    assert fixture["request"]["type"] not in _known_request_types()
    assert fixture["response"]["code"] == "unsupported"
