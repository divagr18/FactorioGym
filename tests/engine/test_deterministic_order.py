"""Every order in a `local-v2` observation is a function of the world alone.

Requires a real Factorio worker (engine marker).

A fast simulator has to reproduce these observations exactly -- their order
included, because the encoder keeps the 32 nearest entity rows with a stable
sort, and the parameterized target dimension indexes entities and resource
tiles by position in the list. Before `local-v2` v6 three orders came from the
engine rather than from the world:

* resource tiles, in `find_entities_filtered` order, from a sweep capped at
  `resource_cap + 1` over the whole 32-tile radius -- so the engine's order
  decided their handle numbers and, with a large patch nearby, which near tiles
  were kept at all;
* entities at equal distance, by sweep index;
* `remembered`, by `pairs()` over the memory store.

This measures the rule on real frames rather than trusting it: each list the
worker returns must equal itself re-sorted in Python by the declared key.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from factoriorl import tasks as task_registry
from factoriorl.env import FactorioEnv
from factoriorl.parameterized import sample_masked, wrap_for_policy
from factoriorl.seeding import Branch, SeedPlan

pytestmark = pytest.mark.engine

#: Scenes with an ore patch and several machines, so equal distances occur.
TASKS = ("build_line", "construct_smelting_line", "mine_smelt", "plate_line")
STEPS = 25


@pytest.fixture(scope="module")
def paced_session(module_session):
    module_session.configure(speed=60.0)
    return module_session


def _origin(observation: dict) -> tuple[float, float]:
    origin = (observation.get("sensor") or {}).get("origin")
    if isinstance(origin, dict):
        return float(origin["x"]), float(origin["y"])
    return float(origin[0]), float(origin[1])


def _world_key(origin: tuple[float, float]):
    ox, oy = origin

    def key(record: dict):
        x, y = float(record["p"][0]), float(record["p"][1])
        return ((x - ox) ** 2 + (y - oy) ** 2, y, x, record.get("name") or "")

    return key


def _assert_ordered(observation: dict, where: str) -> dict:
    origin = _origin(observation)
    key = _world_key(origin)
    tiles = (observation.get("resources") or {}).get("tiles") or []
    assert tiles == sorted(tiles, key=key), f"{where}: resource tiles out of world order"

    entities = observation.get("entities") or []
    assert entities == sorted(entities, key=key), f"{where}: entity ties out of world order"

    remembered = observation.get("remembered") or []
    by_place = sorted(
        remembered,
        key=lambda r: (float(r["p"][1]), float(r["p"][0]), r.get("name") or "", r.get("h")),
    )
    assert remembered == by_place, f"{where}: remembered out of world order"

    # Distance to the tile centre, exactly as the sensor measures it.
    detail = 12
    assert all(math.dist(origin, t["p"]) <= detail for t in tiles), f"{where}: tile past detail"
    return {
        "tiles": len(tiles),
        "entities": len(entities),
        "remembered": len(remembered),
        "tied_entities": len(entities) - len({key(e)[0] for e in entities}),
    }


@pytest.mark.parametrize("task_id", TASKS)
def test_every_list_is_in_world_order(paced_session, task_id):
    task = task_registry.get(task_id)
    assert task.spec.observation_profile == "local-v2"
    env = FactorioEnv(
        task,
        paced_session,
        SeedPlan(master=4242, run_id="deterministic-order"),
        branch=Branch.TRAIN,
        split="train",
    )
    policy_env = wrap_for_policy(env)
    rng = np.random.default_rng(0)
    policy_env.reset()
    seen = [_assert_ordered(env._observation, f"{task_id} reset")]
    for step in range(STEPS):
        action = sample_masked(policy_env.action_space, policy_env.action_masks(), rng)
        _, _, terminated, truncated, _ = policy_env.step(action)
        seen.append(_assert_ordered(env._observation, f"{task_id} step {step}"))
        if terminated or truncated:
            break
    assert sum(frame["tiles"] for frame in seen) > 0, "no resource tiles were ever observed"
