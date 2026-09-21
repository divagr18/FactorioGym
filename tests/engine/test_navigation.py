"""Known-terrain navigation (DESIGN.md 5.1), against a real engine.

The assistance profile is the part of this system most likely to be quietly
implemented with a teleport or with the engine's own pathfinder, and neither
would fail any test that only checked where the character ended up. So every
clause here is asserted on something a shortcut would get wrong: the ticks
spent, the intermediate positions, the route's length relative to the straight
line, and what the world looks like afterwards.

``assisted-v1`` states the implementation it is claiming -- "own A* over
explored terrain, walked with walking_state" -- and these tests are what make
that a checkable claim rather than a comment.
"""

from __future__ import annotations

import pytest

from factoriorl.rcon import RCONClient

pytestmark = pytest.mark.engine

#: Fast enough that a route costs a fraction of a second of wall clock, and the
#: same value training uses, so tick accounting matches what runs elsewhere.
SPEED = 90.0


@pytest.fixture()
def navigator(module_session, module_worker):
    """A session on the assisted profile, with a raw client for world edits."""
    with RCONClient(module_worker.spec.rcon_endpoint, timeout=30.0) as client:
        client.lua(f"game.speed = {SPEED} return 1")
    module_session.reset(action_profile="assisted-v1")
    return module_session, module_worker


def _lua(worker, code: str):
    with RCONClient(worker.spec.rcon_endpoint, timeout=30.0) as client:
        return client.lua(code)


def _walk(session, payload, limit: int = 80) -> dict:
    """Issue one navigate and poll to completion, recording the path taken."""
    before = session.observe().response.result
    start = (before.get("character") or {}).get("position")
    tick0 = before.get("absolute_tick") or 0
    response = session.step({"action": "navigate", **payload}, ticks=30)
    action = (response.response.result or {}).get("action") or {}
    path = [tuple(start)] if start else []
    for _ in range(limit):
        snapshot = session.observe().response.result
        position = (snapshot.get("character") or {}).get("position")
        if position:
            path.append(tuple(position))
        if not (snapshot.get("inflight") or []):
            break
        session.step({"action": "wait"}, ticks=30)
    after = session.observe().response.result
    return {
        "code": response.response.code,
        "action": action,
        "start": start,
        "end": (after.get("character") or {}).get("position"),
        "ticks": (after.get("absolute_tick") or 0) - tick0,
        "path": path,
        "observation": after,
    }


def _near(position, target, tolerance: float = 2.0) -> bool:
    return bool(
        position
        and abs(position[0] - target[0]) <= tolerance
        and abs(position[1] - target[1]) <= tolerance
    )


def test_movement_consumes_game_time_and_is_not_a_teleport(navigator):
    """A teleport would satisfy 'ended up at the destination' perfectly.

    What it cannot fake is spending ticks and passing through the tiles in
    between, so those are what is asserted.
    """
    session, _ = navigator
    walk = _walk(session, {"position": [16.0, 9.0]})

    assert walk["code"] == "ok"
    assert _near(walk["end"], (16.0, 9.0)), walk["end"]
    assert walk["ticks"] > 30, f"arrived in {walk['ticks']} ticks, which is not walking"
    # Distinct intermediate positions: a teleport produces two, start and end.
    assert len({tuple(round(c, 1) for c in p) for p in walk["path"]}) >= 4, walk["path"]


def test_a_route_goes_around_a_known_obstacle(navigator):
    """Routing over known terrain means the route is longer than the straight
    line when something is in the way -- and that the character still arrives."""
    session, worker = navigator
    built = _lua(
        worker,
        "local s=game.surfaces[1] local n=0 "
        "for dy=-4,4 do "
        "  local e=s.create_entity{name='stone-wall', position={8, dy}, force='neutral'} "
        "  if e then n=n+1 end end return n",
    )
    assert int(built) > 0, "the obstacle was not built, so the test proves nothing"

    walk = _walk(session, {"position": [20.0, 0.0]})
    assert _near(walk["end"], (20.0, 0.0)), walk["end"]
    # Straight-line distance is 20; a route that ignored the wall would match it.
    assert walk["action"].get("route_length", 0) > 20, walk["action"]


def test_an_obstacle_discovered_mid_walk_is_replanned_around(navigator):
    """The clause is 'newly discovered obstacles trigger replanning or a useful
    failure'. A planner that committed to its first route would walk into the
    wall and stall silently, which is neither."""
    session, worker = navigator
    session.step({"action": "navigate", "position": [26.0, 0.0]}, ticks=30)
    session.step({"action": "wait"}, ticks=30)

    moving = session.observe().response.result
    launched = (moving.get("character") or {}).get("position")
    assert launched and launched[0] > 2.0, "the walk had not started; nothing to replan"

    built = _lua(
        worker,
        "local s=game.surfaces[1] local n=0 "
        "for dy=-6,6 do "
        "  local e=s.create_entity{name='stone-wall', position={16, dy}, force='neutral'} "
        "  if e then n=n+1 end end return n",
    )
    assert int(built) > 0

    for _ in range(80):
        snapshot = session.observe().response.result
        if not (snapshot.get("inflight") or []):
            break
        session.step({"action": "wait"}, ticks=30)

    end = (session.observe().response.result.get("character") or {}).get("position")
    # Either it got there around the new wall, or it stopped short of it. What
    # it must not do is end up past x=16 without going around, or hang.
    assert end is not None
    assert _near(end, (26.0, 0.0)) or end[0] < 16.0, end


def test_a_destination_outside_known_terrain_is_refused_by_name(navigator):
    """'Routes use only known terrain' is only meaningful if asking for a route
    over unknown terrain fails, and says so."""
    session, _ = navigator
    walk = _walk(session, {"position": [400.0, 400.0]}, limit=6)

    action = walk["action"]
    assert action.get("status") == "rejected", action
    error = action.get("error") or {}
    assert error.get("code") == "precondition", error
    assert "known terrain" in (error.get("message") or ""), error
    # And it must not have wandered off in the general direction anyway.
    assert _near(walk["end"], tuple(walk["start"]), tolerance=1.0), walk["end"]


def test_navigating_to_an_entity_does_not_perform_the_interaction(navigator):
    """Navigation may deliver the character to a chest. It may not open it.

    An assistance layer that quietly completed the interaction would make every
    transfer task easier than its action catalog says it is, and the success
    predicate would never notice.
    """
    session, worker = navigator
    _lua(
        worker,
        "local s=game.surfaces[1] "
        "local c=s.create_entity{name='wooden-chest', position={18.5, 6.5}, force='player'} "
        "if c then c.insert{name='iron-plate', count=20} end return 1",
    )
    session.step({"action": "wait"}, ticks=30)

    before = session.observe().response.result
    character = (before.get("character") or {}).get("position") or [0.0, 0.0]

    def distance(entity):
        point = entity.get("p") or [0.0, 0.0]
        return abs(point[0] - character[0]) + abs(point[1] - character[1])

    chests = sorted(
        (e for e in (before.get("entities") or []) if e.get("name") == "wooden-chest"),
        key=distance,
        reverse=True,
    )
    assert chests, "no chest in the observation to navigate to"
    target = chests[0]
    assert distance(target) > 4.0, "the chest was already adjacent; the walk proves nothing"

    inventory_before = dict(before.get("inventory") or {})
    contents_before = dict(target.get("contents") or {})

    walk = _walk(session, {"handle": target.get("h") or target.get("handle")})
    after = walk["observation"]

    assert dict(after.get("inventory") or {}) == inventory_before, (
        "navigation moved items into the character's inventory"
    )
    matching = [
        e
        for e in (after.get("entities") or [])
        if (e.get("h") or e.get("handle")) == (target.get("h") or target.get("handle"))
    ]
    if matching:
        assert dict(matching[0].get("contents") or {}) == contents_before, (
            "navigation changed the target's contents"
        )
