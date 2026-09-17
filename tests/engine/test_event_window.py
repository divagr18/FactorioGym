"""`local-v2` v7 publishes a short, fixed event window.

Requires a real Factorio worker (engine marker).

Every frame used to resend up to 256 events, 62% of a `local-v2` frame, and a
step's event held its action's result table by reference, so an event already
published read differently once the action settled. v7 publishes the last
eight events as flat records fixed when they were written, plus the settled
and refused counts over the whole buffer.
"""

from __future__ import annotations

import math

import pytest

from factoriorl import tasks as task_registry
from factoriorl.env import FactorioEnv
from factoriorl.seeding import Branch, SeedPlan

pytestmark = pytest.mark.engine

WINDOW = 8


@pytest.fixture(scope="module")
def paced_session(module_session):
    module_session.configure(speed=60.0)
    return module_session


def test_events_are_a_short_flat_window_with_buffer_counts(paced_session):
    env = FactorioEnv(
        task_registry.get("construct_smelting_line"),
        paced_session,
        SeedPlan(master=5, run_id="event-window"),
        branch=Branch.TRAIN,
        split="train",
    )
    env.reset(options={"scene_index": 0})
    assert env._observation["profiles"]["observation_version"] == 7

    # Walk onto the patch so a mine is in reach.
    patch = env._truth["markers"]["patch"]
    keys = env.catalog.keys()
    for _ in range(30):
        here = env._observation["character"]["position"]
        dx, dy = patch[0] - here[0], patch[1] - here[1]
        if max(abs(dx), abs(dy)) < 1.5:
            break
        axis = (
            ("east" if dx > 0 else "west")
            if abs(dx) >= abs(dy)
            else ("south" if dy > 0 else "north")
        )
        stride = "move" if max(abs(dx), abs(dy)) > 5 else "step"
        env.step(keys.index(f"{stride}_{axis}"))

    here = env._observation["character"]["position"]
    tiles = env._observation["resources"]["tiles"]
    ore = str(min(tiles, key=lambda t: math.dist(t["p"], here))["h"])
    env.step_arguments(keys.index("mine_at"), {"handle": ore})
    # The step's own event appears on the next frame, with the mine still running.
    env.step(keys.index("wait"))
    mine_step = next(e for e in env._observation["events"] if e.get("action") == "mine")
    assert mine_step["result"] == "running"
    frozen = dict(mine_step)

    # Four more decisions settle the 121-tick mine. The record of the step that
    # started it is still in the window, and still reads as it did.
    for _ in range(4):
        env.step(keys.index("wait"))
    assert not env._observation.get("inflight"), "the mine should have settled"
    again = [e for e in env._observation["events"] if e["request_id"] == frozen["request_id"]]
    assert again == [frozen], "a published event changed after it was written"

    for _ in range(12):
        env.step(keys.index("wait"))
    events = env._observation["events"]
    assert len(events) == WINDOW
    assert all(not isinstance(e.get("action"), dict) for e in events)
    counts = env._observation["event_counts"]
    assert counts["settled"] > WINDOW
    assert counts["refused"] == 0
