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
            cap=0.5,
            predicate=Predicate(PredicateKind.CONTAINER_HOLDS, marker="sink", item="iron-plate"),
        ),
        RewardComponent("step_cost", RewardKind.STEP_COST, weight=0.001),
    ),
    max_decision_steps=300,
    max_game_ticks=18000,
    landmarks=(
        # Falls from true to false as the spare is spent, which is what tells
        # the policy a placement actually happened.
        Predicate(PredicateKind.INVENTORY_HOLDS, item="transport-belt", at_least=1),
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
        "nudge_north",
        "nudge_east",
        "nudge_south",
        "nudge_west",
        "place_transport_belt_north",
        "place_transport_belt_east",
        "place_transport_belt_south",
        "place_transport_belt_west",
        "rotate_target",
        "wait",
    ),
)

LINE_Y = 0.0


def generate(family: LayoutFamily, rng) -> Blueprint:
    # `gap_far` used to be a byte-identical copy of `gap`: the name promised a
    # distance variation the generator never implemented, so two of the four
    # declared layout families were the same content.
    if family.name == "gap_far":
        length = rng.randint(11, 14)
        start_x = 4
        gap_index = rng.randint(length - 4, length - 2)
    else:
        length = rng.randint(6, 9)
        start_x = 4
        gap_index = rng.randint(2, length - 2)
    gaps = {gap_index}
    if family.name == "double_gap":
        # Distinct, or `double_gap` silently degenerates into `gap`.
        choices = [i for i in range(2, length - 1) if i != gap_index]
        if choices:
            gaps.add(choices[rng.randrange(len(choices))])

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
    # standing between source and sink. Burner inserters, because the scene has
    # no power network: electric ones would never move an item even after a
    # correct repair, making the task unsolvable for reasons unrelated to the
    # belt.
    #
    # Both face *west*, which reads backwards until you check the engine: an
    # inserter's `direction` points at its PICKUP tile and it drops 1.2 tiles
    # behind itself. Built facing east, the loader picked from the belt and
    # dropped into the source chest -- the line ran in reverse, both inserters
    # sat in `waiting_for_source_items`, and a perfectly repaired belt still
    # delivered nothing.
    entities.append(
        EntitySpec(
            "burner-inserter",
            (float(start_x - 1), LINE_Y),
            direction="west",
            marker="loader",
            contents={"coal": 5},
        )
    )
    entities.append(
        EntitySpec(
            "burner-inserter",
            (float(start_x + length), LINE_Y),
            direction="west",
            marker="unloader",
            contents={"coal": 5},
        )
    )

    return Blueprint(
        entities=tuple(entities),
        character_position=(0.0, 0.0),
        character_inventory={"transport-belt": 5},
        unlock_recipes=("transport-belt",),
        markers={"sink": (float(start_x + length + 1), LINE_Y)},
        radius=64,
    )


register(RegisteredTask(spec=SPEC, generate=generate))
