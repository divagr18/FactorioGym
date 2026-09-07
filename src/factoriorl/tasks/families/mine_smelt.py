"""Mine ore and produce plates (PLAN.md 3.2, family 3).

The production family, and the qualifying one for PLAN 4.5's "at least one must
involve production or repair".

Success is measured from **force production statistics**, not from an inventory
count, so plates that already existed at reset can never be counted as produced.
Hand-crafting is deliberately off the critical path: the furnace is pre-placed,
so the family does not depend on crafting-queue behaviour.
"""

from __future__ import annotations

import math

from factoriorl.tasks import RegisteredTask, register
from factoriorl.tasks.spec import (
    Blueprint,
    EntitySpec,
    LayoutFamily,
    Predicate,
    PredicateKind,
    ResourceSpec,
    RewardComponent,
    RewardKind,
    TaskSpec,
)

TARGET_COUNT = 3

FAMILIES = (
    LayoutFamily("near_patch", "train"),
    LayoutFamily("fuelled_furnace", "train"),
    LayoutFamily("split_patch", "val"),
    # Matched distance, different arrangement: the patch is screened rather
    # than farther away.
    LayoutFamily("screened_patch", "test"),
)

SPEC = TaskSpec(
    id="mine_smelt",
    version="1.0.0",
    description="Mine iron ore, feed a furnace, and produce iron plates.",
    layout_families=FAMILIES,
    success=(Predicate(PredicateKind.PRODUCED, item="iron-plate", at_least=TARGET_COUNT),),
    rewards=(
        RewardComponent("smelted", RewardKind.SPARSE_SUCCESS, weight=1.0, shaping=False),
        RewardComponent(
            "plates_produced",
            RewardKind.HIGH_WATER,
            weight=0.2,
            cap=0.6,
            predicate=Predicate(PredicateKind.PRODUCED, item="iron-plate"),
        ),
        RewardComponent(
            "ore_mined",
            RewardKind.HIGH_WATER,
            weight=0.05,
            cap=0.25,
            predicate=Predicate(PredicateKind.INVENTORY_HOLDS, item="iron-ore"),
        ),
        RewardComponent("step_cost", RewardKind.STEP_COST, weight=0.001),
    ),
    max_decision_steps=400,
    max_game_ticks=24000,
    landmarks=(
        Predicate(PredicateKind.INVENTORY_HOLDS, item="iron-ore", at_least=1),
        Predicate(PredicateKind.INVENTORY_HOLDS, item="iron-ore", at_least=5),
    ),
    catalog_subset=(
        "move_north",
        "move_east",
        "move_south",
        "move_west",
        "step_north",
        "step_east",
        "step_south",
        "step_west",
        "mine_nearest",
        "mine_nearest_5",
        "take_iron-ore_5",
        "give_iron-ore_5",
        "give_coal_5",
        "take_iron-plate_5",
        "wait",
    ),
)


#: Far enough from spawn that the character never starts inside the furnace,
#: close enough to carry ore back inside the episode budget.
FURNACE_RADIUS = 6.0


def generate(family: LayoutFamily, rng) -> Blueprint:
    angle = rng.uniform(0, 2 * math.pi)
    distance = rng.uniform(8.0, 14.0)
    centre = (round(math.cos(angle) * distance), round(math.sin(angle) * distance))

    resources = []
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            resources.append(
                ResourceSpec("iron-ore", (centre[0] + dx, centre[1] + dy), amount=1500)
            )
    if family.name == "split_patch":
        for dx in range(-1, 2):
            resources.append(
                ResourceSpec("iron-ore", (centre[0] + dx + 6, centre[1] + 5), amount=800)
            )

    if family.name == "screened_patch":
        for offset in (-2, -1, 1, 2):
            resources.append(
                ResourceSpec("iron-ore", (centre[0] + 3, centre[1] + offset), amount=800)
            )
    # Perpendicular to the patch bearing at a fixed radius, so walking to the
    # ore never runs into the furnace *and* the furnace can never land on the
    # spawn tile. Scaling the patch centre and adding a constant offset did
    # exactly that: a patch centred (-5, 9) put the furnace at (0, 0) and the
    # character spawned inside its 2x2 footprint, blocked on its first move.
    furnace_angle = angle + math.pi / 2
    furnace = (
        round(math.cos(furnace_angle) * FURNACE_RADIUS),
        round(math.sin(furnace_angle) * FURNACE_RADIUS),
    )
    contents = {"coal": 6} if family.name == "fuelled_furnace" else {}
    entities = [EntitySpec("stone-furnace", furnace, contents=contents, marker="furnace")]
    inventory = {"coal": 10} if family.name != "fuelled_furnace" else {"coal": 4}

    return Blueprint(
        entities=tuple(entities),
        resources=tuple(resources),
        character_position=(0.0, 0.0),
        character_inventory=inventory,
        markers={"patch": centre, "furnace": furnace},
        radius=64,
    )


register(RegisteredTask(spec=SPEC, generate=generate))
