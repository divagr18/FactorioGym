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

#: Five plates is one carry and one deposit, which a uniform policy over the
#: skill catalog completes by accident: the measured random baseline on the
#: structural split was 0.88, against a 0.80 acceptance bar, with the trained
#: policy at 0.20 -- far below its own floor. A benchmark whose random floor
#: sits above its threshold measures nothing. Twenty plates needs repeated
#: supply runs, which is a sequence rather than a coincidence. The same
#: correction was applied to `mine_smelt`, whose floor fell 0.80 -> 0.24.
TARGET_COUNT = 20

FAMILIES = (
    LayoutFamily("linear_row", "train"),
    LayoutFamily("two_furnaces", "train"),
    LayoutFamily("far_ore", "val"),
    LayoutFamily("shared_input", "test"),
)

SPEC = TaskSpec(
    id="supply_furnace",
    # 1.1.0: plate target 5 -> 20, so the task outruns its own random floor.
    # The version is in the random-baseline cache key, which keeps a cached
    # 0.88 from ever being compared against the harder task.
    version="1.1.0",
    description="Keep a fuelled furnace supplied with ore until N plates exist.",
    layout_families=FAMILIES,
    success=(Predicate(PredicateKind.PRODUCED, item="iron-plate", at_least=TARGET_COUNT),),
    rewards=(
        RewardComponent("supplied", RewardKind.SPARSE_SUCCESS, weight=1.0, shaping=False),
        RewardComponent(
            "plates_produced",
            RewardKind.HIGH_WATER,
            weight=0.15,
            cap=0.6,
            predicate=Predicate(PredicateKind.PRODUCED, item="iron-plate"),
        ),
        RewardComponent(
            "ore_carried",
            RewardKind.HIGH_WATER,
            weight=0.02,
            cap=0.2,
            predicate=Predicate(PredicateKind.INVENTORY_HOLDS, item="iron-ore"),
        ),
        RewardComponent("step_cost", RewardKind.STEP_COST, weight=0.001),
    ),
    max_decision_steps=300,
    max_game_ticks=18000,
    # Success is a force production statistic and names no position, so the split
    # audit reported this task's difficulty parity as unmeasurable. The route is
    # spawn -> ore chest -> furnace, and `ore_chest` is the leg that decides how
    # hard a scene is: its distance is drawn per scene and is the one term any
    # family varies -- `far_ore` draws uniform(15, 21) against uniform(5, 10)
    # everywhere else. The furnace is pinned to a radius of 8, so its distance
    # spans about one tile of integer-rounding jitter (7.6 to 8.6 over 500 seeds);
    # measuring to `furnace` would compare that rounding noise between splits and
    # publish a per-family difficulty column in which `far_ore` -- the family that
    # exists precisely to be farther -- looked identical to training.
    #
    # The descriptor measures one spawn -> marker walk, not the full
    # spawn -> chest -> furnace -> chest shuttle, so it understates the absolute
    # cost of a scene. `ore_chest` is still the right leg to publish, because the
    # furnace leg is near-constant and contributes nothing a parity comparison
    # could read.
    #
    # No version bump: this declares where difficulty is measured, not what the
    # task is; no scene, predicate, reward or budget changes.
    difficulty_marker="ore_chest",
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
        "take_iron-ore_5",
        "take_iron-ore_20",
        "give_iron-ore_5",
        "give_iron-ore_20",
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
        # Stocked well past the target. The supply must not be what limits the
        # task: at 50 ore the reference solution ran the chest dry mid-episode
        # and failed with `no_items`, which makes the family unsolvable rather
        # than harder -- a different thing from the difficulty the raised
        # target is meant to add.
        EntitySpec("wooden-chest", ore_chest, contents={"iron-ore": 200}, marker="ore_chest"),
        EntitySpec("stone-furnace", furnace, contents={"coal": 50}, marker="furnace"),
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
