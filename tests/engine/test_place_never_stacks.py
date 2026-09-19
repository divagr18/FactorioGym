"""A placement never leaves two entities sharing a tile.

Requires a real Factorio worker (engine marker).

`create_entity` does not honour collisions. Measured against 2.0.60
(`tools/probe_duplicate_drill.py`): with a drill already on a tile,
`can_place_entity` refuses a second one and `create_entity` builds it anyway,
leaving two drills with two unit numbers on one tile.

The transfer replay is what found it -- the simulator reported one drill where
the engine reported two, handles h47 and h48 at the same position, right after
a placement. It matters beyond tidiness: `local-v2` addresses a target by its
row in the entity table, so one duplicated row moves every row after it and the
policy names a different entity than the one it chose.
"""

from __future__ import annotations

import math

import pytest

from factoriorl import tasks as task_registry
from factoriorl.env import FactorioEnv
from factoriorl.rcon import RCONClient
from factoriorl.seeding import Branch, SeedPlan

pytestmark = pytest.mark.engine

#: Stand on the patch holding drills, with one drill already on the tile below.
SETUP = """
local ch = storage.frrl_character
local px, py = %f, %f
ch.teleport({px + 3.5, py + 0.5})
local srf = game.surfaces[1]
for _, e in pairs(srf.find_entities_filtered({
  area = {{px - 6, py - 6}, {px + 6, py + 6}}, name = "burner-mining-drill" })) do
  e.destroy()
end
local first = srf.create_entity({
  name = "burner-mining-drill", position = {px, py},
  direction = defines.direction.south, force = "player" })
ch.get_main_inventory().insert({ name = "burner-mining-drill", count = 4 })
return helpers.table_to_json({ x = first.position.x, y = first.position.y })
"""

COUNT = """
local srf = game.surfaces[1]
local at = srf.find_entities_filtered({
  area = {{%f - 0.4, %f - 0.4}, {%f + 0.4, %f + 0.4}}, name = "burner-mining-drill" })
return tostring(#at)
"""


@pytest.fixture()
def env(module_session, module_worker):
    env = FactorioEnv(
        task_registry.get("construct_smelting_line"),
        module_session,
        SeedPlan(master=50, run_id="place-never-stacks"),
        branch=Branch.TRAIN,
        split="train",
    )
    env.reset(options={"scene_index": 0})
    return env


def test_placing_a_turned_drill_onto_one_already_there_does_not_stack(env, module_worker):
    """The exact request the guard caught in a live episode.

    Measured against 2.0.60 (`tools/probe_duplicate_drill.py`): with a
    south-facing drill at (px, py), asking to place a drill at
    (px - 0.5, py - 0.5) is refused by `can_place_entity` facing *south* and
    allowed facing north, east or west -- because a player doing that would
    fast-replace the drill, turning it. `create_entity` does not fast-replace;
    it builds a second drill on the same tile.

    The request comes from the domain because occupancy is one tile per entity
    -- `math.floor` of its position -- while a drill covers 2x2, so three of
    the four tiles it stands on are offered as legal placements.
    """
    patch = env._truth["markers"]["patch"]
    px, py = math.floor(patch[0]), math.floor(patch[1])
    target = [px - 0.5, py - 0.5]
    with RCONClient(module_worker.spec.rcon_endpoint, timeout=30.0) as client:
        client.lua(SETUP % (px, py))
        env.resync("test setup: one south-facing drill already on the tile")
        assert target in [list(p) for p in env.argument_domains()["placements"]], (
            "the placement domain no longer offers a tile inside the drill's "
            "footprint; if occupancy has learnt about entity size, rewrite this"
        )
        keys = [template.key for template in env.catalog.templates]
        env.step_arguments(
            keys.index("place_at"),
            {"position": target, "direction": "west", "item": "burner-mining-drill"},
        )
        standing = int(client.lua(COUNT % (px, py, px, py)))
    assert standing == 1, f"{standing} drills share one tile; a player cannot reach that state"

    seen = [
        (round(r["p"][0], 2), round(r["p"][1], 2), r.get("name"))
        for r in (env._observation.get("entities") or [])
        if r.get("type") != "item-entity"
    ]
    assert len(seen) == len(set(seen)), f"the same position is reported twice: {seen}"
