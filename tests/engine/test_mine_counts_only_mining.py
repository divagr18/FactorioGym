"""A hand-mine counts the ore it mined, not ore moved by other actions.

Requires a real Factorio worker (engine marker).

The mine poller completes a mine when the character holds `count` more of the
item than when it started. Before the fix that was all it looked at: in the M2
parity trace `transfer_clamping`, a `take_from` of five iron ore during a mine
completed it at once as "mined 5", and those five went into `mined_by_action`,
which `machine_produced` subtracts. A `give_to` did the reverse and kept a mine
running past its ore.

The furnace here has no fuel, so ore given to it stays in its source slot.
"""

from __future__ import annotations

import math

import pytest

from factoriorl import tasks as task_registry
from factoriorl.env import FactorioEnv
from factoriorl.rcon import RCONClient
from factoriorl.seeding import Branch, SeedPlan

pytestmark = pytest.mark.engine

SETUP = """
local ch = storage.frrl_character
local px, py = %f, %f
ch.teleport({px + 0.5, py + 0.5})
local f = game.surfaces[1].create_entity({
  name = "stone-furnace", position = {px + 2, py + 1}, force = "player"})
f.get_inventory(defines.inventory.furnace_source).insert({name = "iron-ore", count = 10})
return true
"""


@pytest.fixture(scope="module")
def paced_session(module_session):
    module_session.configure(speed=60.0)
    return module_session


@pytest.fixture()
def env(paced_session, module_worker):
    env = FactorioEnv(
        task_registry.get("construct_smelting_line"),
        paced_session,
        SeedPlan(master=11, run_id="mine-counts"),
        branch=Branch.TRAIN,
        split="train",
    )
    env.reset(options={"scene_index": 0})
    px, py = env._truth["markers"]["patch"]
    with RCONClient(module_worker.spec.rcon_endpoint, timeout=30.0) as client:
        client.lua(SETUP % (math.floor(px), math.floor(py)))
    env.resync("test setup: teleport onto the patch and place a loaded furnace")
    return env


def _do(env: FactorioEnv, key: str, **arguments) -> dict:
    index = env.catalog.keys().index(key)
    step = env.step_arguments(index, arguments) if arguments else env.step(index)
    return step[4]


def _nearest_ore(env: FactorioEnv) -> str:
    here = env._observation["character"]["position"]
    tiles = env._observation["resources"]["tiles"]
    return str(min(tiles, key=lambda t: math.dist(t["p"], here))["h"])


def _furnace(env: FactorioEnv) -> str:
    return next(str(e["h"]) for e in env._observation["entities"] if e["type"] == "furnace")


def _mining(env: FactorioEnv) -> bool:
    return any(entry.get("action") == "mine" for entry in env._observation.get("inflight") or [])


def _mined(env: FactorioEnv) -> int:
    return int((env._truth.get("mined_by_hand") or {}).get("iron-ore", 0))


def _held(env: FactorioEnv) -> int:
    return int((env._observation.get("inventory") or {}).get("iron-ore", 0))


def test_a_plain_mine_takes_one_ore_and_121_ticks(env):
    _do(env, "mine_at", handle=_nearest_ore(env))
    for _ in range(3):
        _do(env, "wait")
        assert _mining(env), "a 121-tick mine finished early"
    _do(env, "wait")
    assert not _mining(env)
    assert _held(env) == 1 and _mined(env) == 1


def test_taking_ore_during_a_mine_is_not_mining(env):
    _do(env, "mine_at", handle=_nearest_ore(env))
    info = _do(env, "take_from", **{"from": _furnace(env)}, item="iron-ore", count=5)
    assert info["action_status"] == "completed"
    assert _held(env) == 5
    assert _mining(env), "taking five ore completed the mine"
    for _ in range(3):
        _do(env, "wait")
    assert not _mining(env)
    assert _held(env) == 6
    assert _mined(env) == 1, "the taken ore was tallied as mined"


def test_giving_ore_away_during_a_mine_does_not_extend_it(env):
    _do(env, "mine_at", handle=_nearest_ore(env))
    for _ in range(4):
        _do(env, "wait")
    assert _held(env) == 1 and _mined(env) == 1

    _do(env, "mine_at", handle=_nearest_ore(env))
    info = _do(env, "give_to", to=_furnace(env), item="iron-ore", count=1)
    assert info["action_status"] == "completed"
    assert _held(env) == 0
    for _ in range(3):
        _do(env, "wait")
    assert not _mining(env), "giving ore away kept the mine running"
    assert _held(env) == 1 and _mined(env) == 2
