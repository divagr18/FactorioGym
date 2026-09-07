"""Retrieve and deliver items (PLAN.md 3.2, family 2).

Success is measured on the destination container, and the failure predicate
catches the degenerate case where the items stop existing at all.
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

TARGET_ITEM = "iron-plate"
TARGET_COUNT = 10

FAMILIES = (
    LayoutFamily("two_chest", "train"),
    LayoutFamily("decoy_chest", "train"),
    LayoutFamily("stacked_depot", "val"),
    # Structural holdout at matched distance: the destination sits behind a
    # wall segment, so the route differs while the distance does not.
    LayoutFamily("screened_depot", "test"),
)

SPEC = TaskSpec(
    id="deliver",
    version="1.0.0",
    description="Carry items from a source container to a destination container.",
    layout_families=FAMILIES,
    success=(
        Predicate(
            PredicateKind.CONTAINER_HOLDS,
            marker="dst",
            item=TARGET_ITEM,
            at_least=TARGET_COUNT,
        ),
    ),
    rewards=(
        RewardComponent("delivered", RewardKind.SPARSE_SUCCESS, weight=1.0, shaping=False),
        # High-water, so taking items back out of the destination and putting
        # them in again cannot pay twice.
        RewardComponent(
            "at_destination",
            RewardKind.HIGH_WATER,
            weight=0.05,
            predicate=Predicate(PredicateKind.CONTAINER_HOLDS, marker="dst", item=TARGET_ITEM),
        ),
        RewardComponent(
            "carried",
            RewardKind.HIGH_WATER,
            weight=0.02,
            predicate=Predicate(PredicateKind.INVENTORY_HOLDS, item=TARGET_ITEM),
        ),
        RewardComponent("step_cost", RewardKind.STEP_COST, weight=0.001),
    ),
    max_decision_steps=250,
    max_game_ticks=15000,
    catalog_subset=(
        "move_north",
        "move_east",
        "move_south",
        "move_west",
        "step_north",
        "step_east",
        "step_south",
        "step_west",
        "take_iron-plate_5",
        "take_iron-plate_20",
        "give_iron-plate_5",
        "give_iron-plate_20",
        "wait",
    ),
)


def generate(family: LayoutFamily, rng) -> Blueprint:
    src_angle = rng.uniform(0, 2 * math.pi)
    src_distance = rng.uniform(4.0, 9.0)
    src = (
        round(math.cos(src_angle) * src_distance, 1),
        round(math.sin(src_angle) * src_distance, 1),
    )

    # One distance range for every family. A holdout that is simply farther
    # away measures added difficulty, not transfer; `screened_depot` varies the
    # route instead, at matched distance.
    dst_distance = rng.uniform(14.0, 24.0)
    dst_angle = src_angle + rng.uniform(math.pi * 0.5, math.pi * 1.5)
    dst = (
        round(math.cos(dst_angle) * dst_distance, 1),
        round(math.sin(dst_angle) * dst_distance, 1),
    )

    entities = [
        EntitySpec("wooden-chest", src, contents={TARGET_ITEM: 40}, marker="src"),
        EntitySpec("wooden-chest", dst, marker="dst"),
    ]
    if family.name == "decoy_chest":
        # A chest holding the wrong item: success must depend on what is
        # delivered, not on visiting a container.
        decoy_angle = src_angle + math.pi
        decoy = (round(math.cos(decoy_angle) * 7, 1), round(math.sin(decoy_angle) * 7, 1))
        entities.append(EntitySpec("wooden-chest", decoy, contents={"coal": 30}, marker="decoy"))
    elif family.name == "stacked_depot":
        entities.append(EntitySpec("wooden-chest", (dst[0] + 2, dst[1]), marker="dst_neighbour"))
    elif family.name == "screened_depot":
        # A wall segment across the direct line to the depot: same distance,
        # different route. Laid perpendicular to the bearing so it screens the
        # approach rather than merely standing near it.
        normal = dst_angle + math.pi * 0.5
        centre = (
            math.cos(dst_angle) * dst_distance * 0.6,
            math.sin(dst_angle) * dst_distance * 0.6,
        )
        taken = {(round(dst[0]), round(dst[1])), (round(src[0]), round(src[1])), (0, 0)}
        for offset in (-2.0, -1.0, 0.0, 1.0, 2.0):
            tile = (
                round(centre[0] + math.cos(normal) * offset),
                round(centre[1] + math.sin(normal) * offset),
            )
            if tile in taken:
                continue
            taken.add(tile)
            entities.append(EntitySpec("stone-wall", (float(tile[0]), float(tile[1]))))

    return Blueprint(
        entities=tuple(entities),
        character_position=(0.0, 0.0),
        markers={"src": src, "dst": dst},
        radius=64,
    )


register(RegisteredTask(spec=SPEC, generate=generate))
