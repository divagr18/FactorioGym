"""Restore a power connection (PLAN.md 3.2, family 6).

A steam setup drives an electric mining drill through a pole chain with one pole
missing, so the drill is unpowered. Success is the drill reporting `working`,
which the evaluator reads from entity status rather than inferring.

Deliberately pole-only: an offshore-pump variant would depend on painting water
tiles, which is a Phase 3 scene-construction question this family does not need
to answer.
"""

from __future__ import annotations

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

FAMILIES = (
    LayoutFamily("pole_gap", "train"),
    LayoutFamily("pole_gap_far", "train"),
    LayoutFamily("two_gaps", "val"),
    # Same chain length as training, gap in a different place: structure, not
    # magnitude.
    LayoutFamily("gap_near_drill", "test"),
)

SPEC = TaskSpec(
    id="restore_power",
    version="1.0.0",
    description="Reconnect a power pole chain so the mining drill runs again.",
    layout_families=FAMILIES,
    success=(Predicate(PredicateKind.ENTITY_WORKING, marker="drill"),),
    rewards=(
        RewardComponent("restored", RewardKind.SPARSE_SUCCESS, weight=1.0, shaping=False),
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
        "place_small_electric_pole_north",
        "place_small_electric_pole_east",
        "place_small_electric_pole_south",
        "place_small_electric_pole_west",
        "wait",
    ),
)


def generate(family: LayoutFamily, rng) -> Blueprint:
    chain = 5
    gap = rng.randint(1, chain - 2)
    if family.name == "gap_near_drill":
        gap = chain - 1
    gaps = {gap}
    if family.name == "two_gaps":
        gaps.add(min(chain - 1, gap + 2))

    # Solar rather than a boiler: a boiler needs a water supply, and without
    # one the drill could never be powered no matter where the missing pole
    # went -- the task would be unsolvable for a reason unrelated to the fault
    # it is meant to test.
    entities = [
        EntitySpec("solar-panel", (-10.0, -8.0), marker="generator"),
        EntitySpec("solar-panel", (-7.0, -8.0)),
        EntitySpec("solar-panel", (-4.0, -8.0)),
        EntitySpec("small-electric-pole", (-1.0, -8.0)),
    ]
    for index in range(chain):
        if index in gaps:
            continue
        entities.append(EntitySpec("small-electric-pole", (float(2 + index * 4), -8.0)))
    drill_x = float(2 + chain * 4)
    entities.append(
        EntitySpec("electric-mining-drill", (drill_x, -8.0), direction="south", marker="drill")
    )
    resources = tuple(
        ResourceSpec("iron-ore", (drill_x + dx, float(dy) - 8.0), amount=1000)
        for dx in (-1.0, 0.0, 1.0)
        for dy in (-1, 0, 1)
    )

    return Blueprint(
        entities=tuple(entities),
        resources=resources,
        character_position=(0.0, 0.0),
        character_inventory={"small-electric-pole": 4},
        unlock_recipes=("small-electric-pole",),
        markers={"drill": (drill_x, -8.0)},
        radius=64,
    )


register(RegisteredTask(spec=SPEC, generate=generate))
