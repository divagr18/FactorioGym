"""Bounded batches (PLAN.md 5.2), against a real engine.

A batch exists to save round trips, and the way it earns that is by failing
informatively. An agent told only "the batch failed" has to re-observe the world
to find out what happened, which costs it exactly the round trip the batch
saved -- so the response names the completed prefix and the operation that
stopped it, and these tests assert both halves rather than the status alone.

Reach and validation are not re-implemented for batched operations: each one
runs through the handler a standalone request would use. That is asserted here
by batching an out-of-reach transfer and expecting the ordinary out-of-reach
refusal, at the right index, with the operations before it applied.
"""

from __future__ import annotations

import pytest

from factoriorl.rcon import RCONClient

pytestmark = pytest.mark.engine

SPEED = 90.0
NEAR = (1.5, 0.5)
FAR = (25.5, 0.5)


@pytest.fixture()
def batching(module_session, module_worker):
    session, worker = module_session, module_worker
    with RCONClient(worker.spec.rcon_endpoint, timeout=30.0) as client:
        client.lua(f"game.speed = {SPEED} return 1")
    session.reset(action_profile="assisted-v1")
    with RCONClient(worker.spec.rcon_endpoint, timeout=30.0) as client:
        client.lua(
            "local s=game.surfaces[1] "
            f"local a=s.create_entity{{name='wooden-chest', position={{{NEAR[0]}, {NEAR[1]}}},"
            " force='player'} "
            "if a then a.insert{name='iron-plate', count=40} end "
            f"local f=s.create_entity{{name='wooden-chest', position={{{FAR[0]}, {FAR[1]}}},"
            " force='player'} "
            "return 1"
        )
    session.step({"action": "wait"}, ticks=30)
    observation = session.observe().response.result
    chests = [e for e in (observation.get("entities") or []) if e.get("name") == "wooden-chest"]
    stocked = [c for c in chests if (c.get("contents") or {}).get("iron-plate")]
    distant = [c for c in chests if abs((c.get("p") or [0, 0])[0]) > 20]
    assert stocked and distant, "fixture scene is not what the tests assume"
    return session, stocked[0]["h"], distant[0]["h"]


def _batch(session, operations, **extra):
    response = session.step({"action": "batch", "operations": operations, **extra}, ticks=30)
    return (response.response.result or {}).get("action") or {}


def _plates(session) -> int:
    inventory = session.observe().response.result.get("inventory") or {}
    return int(inventory.get("iron-plate") or 0)


def test_every_underlying_action_is_recorded(batching):
    session, source, _ = batching
    before = _plates(session)
    action = _batch(
        session,
        [
            {
                "action": "transfer",
                "from": source,
                "to": "character",
                "item": "iron-plate",
                "count": 5,
            },
            {
                "action": "transfer",
                "from": "character",
                "to": source,
                "item": "iron-plate",
                "count": 2,
            },
        ],
    )
    assert action["status"] == "completed", action
    assert action["requested"] == 2 and action["executed"] == 2
    recorded = action["operations"]
    assert [op["index"] for op in recorded] == [1, 2]
    assert [op["action"] for op in recorded] == ["transfer", "transfer"]
    # Each carries the result the standalone action would have returned, not a
    # bare ok: a batch that summarised its operations away would be unauditable.
    assert recorded[0]["result"]["count"] == 5
    assert recorded[1]["result"]["to"] == source
    assert _plates(session) == before + 3


def test_a_failure_names_the_operation_and_the_completed_prefix(batching):
    """Reach is enforced per operation by the ordinary handler, so an
    out-of-reach transfer inside a batch fails the way it would alone."""
    session, source, distant = batching
    before = _plates(session)
    action = _batch(
        session,
        [
            {
                "action": "transfer",
                "from": source,
                "to": "character",
                "item": "iron-plate",
                "count": 3,
            },
            {
                "action": "transfer",
                "from": distant,
                "to": "character",
                "item": "iron-plate",
                "count": 1,
            },
            {
                "action": "transfer",
                "from": source,
                "to": "character",
                "item": "iron-plate",
                "count": 4,
            },
        ],
    )
    assert action["status"] == "rejected", action
    assert action["failed_at"] == 2
    assert action["failed_action"] == "transfer"
    assert action["executed"] == 1
    assert [op["index"] for op in action["operations"]] == [1]
    # The prefix really happened and the rest really did not.
    assert _plates(session) == before + 3


def test_an_ongoing_action_cannot_be_batched(batching):
    """A batch runs inside one request. An ongoing action does not finish
    inside one, so 'the completed prefix' would stop meaning anything if it
    could be mixed in -- it is refused by name rather than reordered."""
    session, _, _ = batching
    action = _batch(session, [{"action": "move", "direction": "north", "ticks": 30}])
    assert action["status"] == "rejected", action
    assert action["failed_at"] == 1
    assert action["failed_action"] == "move"
    assert action["executed"] == 0


def test_the_protocol_ceiling_is_enforced_before_anything_runs(batching):
    session, _, _ = batching
    before = _plates(session)
    action = _batch(session, [{"action": "wait"} for _ in range(20)])
    assert action["status"] == "rejected", action
    error = action.get("error") or {}
    assert error.get("code") == "bad_type"
    assert "limit is 16" in error.get("message", ""), error
    assert _plates(session) == before


def test_the_caller_may_impose_a_lower_ceiling(batching):
    """'Explicit execution limits' has to include the caller's own, or the only
    bound is the server's and an agent cannot ask for a smaller one."""
    session, _, _ = batching
    action = _batch(
        session,
        [{"action": "wait"}, {"action": "wait"}, {"action": "wait"}],
        max_operations=2,
    )
    assert action["status"] == "completed", action
    assert action["executed"] == 2
    assert action["stopped_at"] == 3
    assert action["stopped_reason"] == "max_operations"


def test_an_empty_batch_and_a_nested_batch_are_both_refused(batching):
    session, _, _ = batching
    empty = _batch(session, [])
    assert empty["status"] == "rejected"
    assert "empty" in ((empty.get("error") or {}).get("message") or "")

    nested = _batch(session, [{"action": "batch", "operations": [{"action": "wait"}]}])
    assert nested["status"] == "rejected"
    assert nested["failed_action"] == "batch"


def test_batch_is_absent_from_the_primitive_profile(module_session):
    """The action profile is a capability boundary. A batch reachable from
    `primitive-v1` would mean no Phase 3 or Phase 4 number could claim to have
    been measured without assistance."""
    module_session.reset()
    response = module_session.step(
        {"action": "batch", "operations": [{"action": "wait"}]}, ticks=30
    )
    action = (response.response.result or {}).get("action") or {}
    assert action.get("status") == "rejected"
    error = action.get("error") or {}
    assert error.get("code") == "unknown_action"
    assert (error.get("details") or {}).get("action_profile") == "primitive-v1"
