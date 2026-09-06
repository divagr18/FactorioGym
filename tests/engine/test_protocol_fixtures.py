"""Contract tests: Python and Lua agree on the frozen protocol fixtures.

Requires a real Factorio worker (engine marker). Each fixture is a
request/response pair; we send the request through the typed session and
assert the worker's response matches the frozen expectation.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pytest

from factoriorl.errors import StaleEpisodeError
from factoriorl.protocol import ActionStatus, Request, RequestType
from factoriorl.session import WorkerSession

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "protocol_v1"

pytestmark = pytest.mark.engine


def _load_fixtures():
    return sorted(FIXTURE_DIR.glob("*.json"), key=lambda p: p.name)


def _send_raw(session: WorkerSession, fixture: dict):
    """Send a fixture request as a raw dict (negative fixtures need types and
    protocol versions the Python enums deliberately refuse to build)."""
    request_body = dict(fixture["request"])
    if request_body.get("request_id") == "<auto>":
        request_body["request_id"] = session._next_request_id("fixture")
    if request_body.get("episode_id") in ("<current>", "<current-or-empty>"):
        session._require_episode()
        request_body["episode_id"] = session.episode_id
    stale_raises = fixture.get("id") == "stale_episode"
    return session.dispatch_raw(request_body, stale_raises=stale_raises).response


@pytest.mark.parametrize("fixture_path", _load_fixtures(), ids=lambda p: p.stem)
def test_fixture_agreement(module_session, fixture_path):
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert fixture["protocol"] == 1

    for step in fixture.get("setup", []):
        if step == "reset":
            module_session.reset()

    if fixture["id"] == "stale_episode":
        # The session raises on stale episodes by design.
        with pytest.raises(StaleEpisodeError):
            _send_raw(module_session, fixture)
        return

    response = _send_raw(module_session, fixture)
    expected = fixture["response"]
    assert response.code.value == expected["code"], response

    if "error" in expected:
        assert response.error is not None
        assert response.error.code.value == expected["error"]["code"]

    if "result" in expected:
        for key, value in expected["result"].items():
            actual = response.result.get(key)
            if key == "character":
                assert actual.get("present") == value.get("present")
            elif key == "entities":
                for name, ent in value.items():
                    actual_ent = response.result["entities"][name]
                    for field_name, field_value in ent.items():
                        assert actual_ent.get(field_name) == field_value, (
                            name,
                            field_name,
                        )
            else:
                assert actual == value, key

    if "result_episode_id_pattern" in fixture:
        assert re.match(
            fixture["result_episode_id_pattern"],
            response.result.get("episode_id", ""),
        )


def test_unknown_action_cannot_reach_arbitrary_execution(module_session):
    """Unsupported action/request types must not execute game mutations."""
    module_session.reset()
    before = module_session.observe().response.result
    # Attempt code-execution style request types; all must be rejected.
    for rtype, payload in (
        ("execute_lua", {"code": "game.tick_paused = false"}),
        ("admin", {"cmd": "spawn items"}),
    ):
        resp = _send_raw(
            module_session,
            {
                "request": {
                    "protocol": 1,
                    "request_id": "<auto>",
                    "episode_id": "<current>",
                    "type": rtype,
                    "payload": payload,
                }
            },
        )
        assert resp.code.value == "unsupported", (rtype, resp)
    after = module_session.observe().response.result
    assert before["tick"] == after["tick"]
    assert before["entities"] == after["entities"]


def test_duplicate_transfer_applies_once(module_session):
    module_session.reset()
    request = Request(
        request_id="dup-test-1",
        episode_id=module_session.episode_id,
        type=RequestType.ACT,
        payload={
            "action": "transfer",
            "from": "src",
            "to": "character",
            "item": "iron-plate",
            "count": 10,
        },
    )
    first = module_session._request(request)
    second = module_session._request(request)
    assert first.response.code.value == "ok"
    assert second.response.code.value == "duplicate"
    obs = module_session.observe().response.result
    assert obs["entities"]["src"]["contents"]["iron-plate"] == 40
    assert obs["task"]["transfers"] == 1


def test_request_status_resolves_uncertain_advance(module_session):
    module_session.reset()
    timed = module_session.advance(30)
    # Re-query the settled advance through request_status: applied + settled.
    probe = module_session.request_status(timed.response.request_id)
    resolution = probe.response.result
    stored = resolution["stored"]
    assert resolution["applied"] is True
    assert resolution["observed"] is True
    assert resolution["settled"] is True
    assert resolution["state"] == "completed"
    assert stored["result"]["status"] == "completed"
    # Never-applied id resolves explicitly to applied=False.
    missing = module_session.request_status("never-sent-id").response.result
    assert missing["applied"] is False
    assert missing["observed"] is False
    assert missing["state"] == "unknown"


def test_advance_settles_from_running_to_completed(module_session):
    """The ongoing-request lifecycle the advance fixture describes.

    ``dispatch_raw`` returns the first answer (``running``); the settled
    ``completed`` response only exists in the worker's ledger. Asserting both
    halves is what makes the fixture's ``settled`` block real evidence rather
    than documentation (PLAN.md 1.1: fixtures cover ongoing operations).
    """
    module_session.reset()
    fixture = json.loads((FIXTURE_DIR / "advance_success.json").read_text(encoding="utf-8"))
    first = _send_raw(module_session, fixture)
    assert first.code.value == fixture["response"]["code"]
    assert first.result["status"] == "running"
    assert first.result["target_ticks"] == fixture["request"]["payload"]["ticks"]

    deadline = time.monotonic() + 30.0
    resolution = {}
    while time.monotonic() < deadline:
        resolution = module_session.request_status(first.request_id).response.result
        if resolution.get("settled"):
            break
        time.sleep(0.02)
    settled = fixture["settled"]
    assert resolution["state"] == "completed"
    assert resolution["stored"]["code"] == settled["code"]
    assert resolution["stored"]["result"] == settled["result"]


def test_reconnected_session_ids_do_not_collide(module_worker, module_session):
    """A reconnecting client must not have its mutation swallowed as a duplicate.

    Request ids used to be a bare per-session counter, so a new session on a
    live worker restarted at ``act-1`` and collided with ids the worker had
    already recorded. The worker would answer ``duplicate`` with the *old*
    stored result and never apply the new transfer -- a silent lost mutation
    in exactly the reconnect path request_status exists to make safe.
    """
    module_session.reset()
    first = module_session.act(
        "transfer", **{"from": "src", "to": "character", "item": "iron-plate", "count": 10}
    )
    assert first.response.code.value == "ok"

    # Reconnect: a brand-new session over the same live worker and episode.
    reconnected = WorkerSession(module_worker)
    try:
        reconnected.status()
        assert reconnected.episode_id == module_session.episode_id
        second = reconnected.act(
            "transfer", **{"from": "src", "to": "character", "item": "iron-plate", "count": 5}
        )
        assert second.response.code.value == "ok", "reconnect collided with a recorded id"
        assert second.response.request_id != first.response.request_id
        obs = reconnected.observe().response.result
        assert obs["inventory"]["iron-plate"] == 15
        assert obs["task"]["transfers"] == 2
    finally:
        reconnected.close()


def test_rejected_mutation_is_recorded_and_not_reexecuted(module_session):
    """A resent request id returns the stored rejection instead of re-running.

    Only successful mutations used to enter the ledger, so a retry after a
    timeout on a rejected action executed it a second time, and request_status
    could not tell "never arrived" from "arrived and was refused".
    """
    module_session.reset()
    request = Request(
        request_id="reject-once-1",
        episode_id=module_session.episode_id,
        type=RequestType.ACT,
        payload={
            "action": "transfer",
            "from": "src",
            "to": "character",
            "item": "iron-plate",
            "count": 9999,
        },
    )
    first = module_session._request(request)
    assert first.response.code.value == "rejected"
    assert first.response.error.code.value == "no_items"

    resolution = module_session.request_status("reject-once-1").response.result
    assert resolution["observed"] is True, "the worker saw the request"
    assert resolution["applied"] is False, "but nothing was mutated"
    assert resolution["state"] == "rejected"

    resend = module_session._request(request)
    assert resend.response.code.value == "duplicate"
    assert resend.response.error.code.value == "no_items"
    obs = module_session.observe().response.result
    assert obs["entities"]["src"]["contents"]["iron-plate"] == 50
    assert obs["task"]["transfers"] == 0


def test_act_response_carries_a_typed_action_result(module_session):
    """`act` answers parse into ActionResult; other request types do not."""
    module_session.reset()
    ok = module_session.act(
        "transfer", **{"from": "src", "to": "character", "item": "iron-plate", "count": 3}
    )
    result = ok.action
    assert result is not None
    assert result.status is ActionStatus.COMPLETED
    assert result.ok is True
    assert result.data["count"] == 3

    # A worker status answer carries result.status == "ready", which is not an
    # action outcome and must not be mistaken for one.
    assert module_session.status().action is None
    assert module_session.observe().action is None
