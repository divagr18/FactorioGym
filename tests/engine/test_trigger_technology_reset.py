"""Reset clears trigger-technology progress, not only research state.

Requires a real Factorio worker (engine marker).

`steam-power` is researched by crafting 50 iron plates, and a furnace's output
counts. The engine keeps that count in a per-force counter no API reads, and
un-researching the technology on reset left it alone -- so plates smelted in
one episode researched `steam-power` partway through a later one, on a worker
with history and never on a fresh one. The M2 parity recorder found it as a
recording and its replay disagreeing about `tech|steam-power` at one decision.

Three episodes on one worker: 30 plates, then 30 more after a reset -- which
must research nothing -- then 60 in a single episode, which must, so the test
cannot pass merely because furnaces stopped counting.
"""

from __future__ import annotations

import pytest

from factoriorl import tasks as task_registry
from factoriorl.env import FactorioEnv
from factoriorl.rcon import RCONClient
from factoriorl.seeding import Branch, SeedPlan

pytestmark = pytest.mark.engine

#: Ten ore per furnace is ten plates: 194 ticks for the first, 192 after.
ORE_PER_FURNACE = 10
SMELT_TICKS = 2_100

SMELTERS = """
local s = game.surfaces[1]
for k = 1, %d do
  local f = s.create_entity({name = "stone-furnace", position = {20 + 2 * k, 20}, force = "player"})
  f.insert({name = "coal", count = 5})
  f.insert({name = "iron-ore", count = %d})
end
return true
"""

RESEARCHED = 'return game.forces.player.technologies["steam-power"].researched'


@pytest.fixture(scope="module")
def paced_session(module_session):
    module_session.configure(speed=60.0)
    return module_session


def _episode(env: FactorioEnv, client: RCONClient, furnaces: int) -> bool:
    env.reset(options={"scene_index": 0})
    assert client.lua(RESEARCHED) is False, "reset left steam-power researched"
    client.lua(SMELTERS % (furnaces, ORE_PER_FURNACE))
    env.advance(SMELT_TICKS)
    plates = int((env._truth.get("produced") or {}).get("iron-plate", 0))
    assert plates == furnaces * ORE_PER_FURNACE, f"expected every ore smelted, got {plates}"
    return bool(client.lua(RESEARCHED))


def test_plates_from_an_earlier_episode_do_not_research_anything(paced_session, module_worker):
    env = FactorioEnv(
        task_registry.get("construct_smelting_line"),
        paced_session,
        SeedPlan(master=7, run_id="trigger-reset"),
        branch=Branch.TRAIN,
        split="train",
    )
    with RCONClient(module_worker.spec.rcon_endpoint, timeout=30.0) as client:
        assert _episode(env, client, furnaces=3) is False
        assert _episode(env, client, furnaces=3) is False, (
            "30 + 30 plates across a reset researched steam-power: trigger progress leaked"
        )
        assert _episode(env, client, furnaces=6) is True, (
            "60 plates in one episode researched nothing, so this test proves nothing"
        )
