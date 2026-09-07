"""Repair a belt defect (PLAN.md 3.2, family 5).

A belt line runs from a loader chest to an unloading chest with one tile
missing. Because throughput before the repair is exactly zero, the unloading
chest's contents *are* the proof the repair worked -- no separate instrumentation
is needed, and the success predicate cannot be satisfied by the initial state.
"""

from __future__ import annotations

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

FAMILIES = (
    LayoutFamily("gap", "train"),
    LayoutFamily("gap_far", "train"),
    LayoutFamily("misrotation", "val"),
    LayoutFamily("double_gap", "test"),
)

SPEC = TaskSpec(
    id="repair_belt",
    version="1.0.0",
    description="Restore a broken belt line so items reach the unloading chest.",
    layout_families=FAMILIES,
    success=(
        Predicate(PredicateKind.CONTAINER_HOLDS, marker="sink", item="iron-plate", at_least=1),
    ),
    rewards=(
        RewardComponent("repaired", RewardKind.SPARSE_SUCCESS, weight=1.0, shaping=False),
        RewardComponent(
            "delivered",
            RewardKind.HIGH_WATER,
            weight=0.1,
            predicate=Predicate(PredicateKind.CONTAINER_HOLDS, marker="sink", item="iron-plate"),
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
        "place_transport_belt",
        "rotate_target",
        "give_iron-plate_5",
        "wait",
    ),
)

LINE_Y = 0.0


def generate(family: LayoutFamily, rng) -> Blueprint:
    length = rng.randint(6, 9)
    start_x = 4
    gap_index = rng.randint(2, length - 2)
    gaps = {gap_index}
    if family.name == "double_gap":
        other = rng.randint(2, length - 2)
        gaps.add(other)

    entities = [
        EntitySpec(
            "wooden-chest",
            (float(start_x - 2), LINE_Y),
            contents={"iron-plate": 40},
            marker="source",
        ),
        EntitySpec("wooden-chest", (float(start_x + length + 1), LINE_Y), marker="sink"),
    ]
    for index in range(length):
        if index in gaps:
            continue
        direction = "east"
        if family.name == "misrotation" and index == gap_index + 1 and index < length:
            direction = "north"
        entities.append(
            EntitySpec("transport-belt", (float(start_x + index), LINE_Y), direction=direction)
        )
    # Inserters move items on and off the line, so the belt is the only thing
    # standing between source and sink.
    entities.append(
        EntitySpec("inserter", (float(start_x - 1), LINE_Y), direction="east", marker="loader")
    )
    entities.append(
        EntitySpec(
            "inserter", (float(start_x + length), LINE_Y), direction="east", marker="unloader"
        )
    )

    return Blueprint(
        entities=tuple(entities),
        character_position=(0.0, 0.0),
        character_inventory={"transport-belt": 5},
        markers={"sink": (float(start_x + length + 1), LINE_Y)},
        radius=64,
    )


register(RegisteredTask(spec=SPEC, generate=generate))
