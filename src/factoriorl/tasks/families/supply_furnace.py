"""Supply an existing furnace (PLAN.md 3.2, family 4).

The furnace is already built and fuelled; the task is to keep it fed from a
nearby ore chest so it keeps producing. Shorter causal chain than mine_smelt,
which is why PLAN 4.2 reaches for this one first among production families.
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
    RewardComponent,
    RewardKind,
    TaskSpec,
)

TARGET_COUNT = 5

FAMILIES = (
    LayoutFamily("linear_row", "train"),
    LayoutFamily("two_furnaces", "train"),
    LayoutFamily("far_ore", "val"),
    LayoutFamily("shared_input", "test"),
)

SPEC = TaskSpec(
    id="supply_furnace",
    version="1.0.0",
    description="Keep a fuelled furnace supplied with ore until N plates exist.",
    layout_families=FAMILIES,
    success=(Predicate(PredicateKind.PRODUCED, item="iron-plate", at_least=TARGET_COUNT),),
    rewards=(
        RewardComponent("supplied", RewardKind.SPARSE_SUCCESS, weight=1.0, shaping=False),
        RewardComponent(
            "plates_produced",
            RewardKind.HIGH_WATER,
            weight=0.15,
            predicate=Predicate(PredicateKind.PRODUCED, item="iron-plate"),
        ),
        RewardComponent(
            "ore_carried",
            RewardKind.HIGH_WATER,
            weight=0.02,
            predicate=Predicate(PredicateKind.INVENTORY_HOLDS, item="iron-ore"),
        ),
        RewardComponent("step_cost", RewardKind.STEP_COST, weight=0.001),
    ),
    max_decision_steps=300,
    max_game_ticks=18000,
    catalog_subset=(
        "move_north",
        "move_east",
        "move_south",
        "move_west",
        "step_north",
        "step_east",
        "step_south",
        "step_west",
        "take_iron-ore_5",
        "take_iron-ore_20",
        "give_iron-ore_5",
        "give_coal_5",
        "take_iron-plate_5",
        "wait",
    ),
)


def generate(family: LayoutFamily, rng) -> Blueprint:
    angle = rng.uniform(0, 2 * math.pi)
    # `far_ore` is the validation family and had no generator branch at all,
    # so validation was scored on a layout identical to training.
    ore_distance = rng.uniform(15.0, 21.0) if family.name == "far_ore" else rng.uniform(5.0, 10.0)
    ore_chest = (round(math.cos(angle) * ore_distance, 1), round(math.sin(angle) * ore_distance, 1))
    furnace_angle = angle + rng.uniform(1.5, 2.5)
    furnace = (round(math.cos(furnace_angle) * 8), round(math.sin(furnace_angle) * 8))

    entities = [
        EntitySpec("wooden-chest", ore_chest, contents={"iron-ore": 50}, marker="ore_chest"),
        EntitySpec("stone-furnace", furnace, contents={"coal": 10}, marker="furnace"),
    ]
    if family.name == "two_furnaces":
        entities.append(
            EntitySpec(
                "stone-furnace",
                (furnace[0] + 3, furnace[1]),
                contents={"coal": 10},
                marker="furnace_b",
            )
        )
    elif family.name == "shared_input":
        entities.append(
            EntitySpec(
                "wooden-chest",
                (ore_chest[0] + 2, ore_chest[1] + 2),
                contents={"coal": 20},
                marker="fuel_chest",
            )
        )

    return Blueprint(
        entities=tuple(entities),
        character_position=(0.0, 0.0),
        markers={"ore_chest": ore_chest, "furnace": furnace},
        radius=64,
    )


register(RegisteredTask(spec=SPEC, generate=generate))
