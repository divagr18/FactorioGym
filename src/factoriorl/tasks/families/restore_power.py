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
    LayoutFamily("long_chain", "test"),
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
        "place_small_electric_pole",
        "wait",
    ),
)


def generate(family: LayoutFamily, rng) -> Blueprint:
    chain = 5 if family.name != "long_chain" else 8
    gap = rng.randint(1, chain - 1)
    gaps = {gap}
    if family.name == "two_gaps":
        gaps.add(min(chain - 1, gap + 2))

    entities = [
        EntitySpec("boiler", (-6.0, 0.0), direction="east", marker="boiler", contents={"coal": 20}),
        EntitySpec("steam-engine", (-2.0, 0.0), direction="east", marker="engine"),
    ]
    for index in range(chain):
        if index in gaps:
            continue
        entities.append(EntitySpec("small-electric-pole", (float(2 + index * 4), 0.0)))
    drill_x = float(2 + chain * 4)
    entities.append(
        EntitySpec("electric-mining-drill", (drill_x, 0.0), direction="south", marker="drill")
    )
    resources = tuple(
        ResourceSpec("iron-ore", (drill_x + dx, float(dy)), amount=1000)
        for dx in (-1.0, 0.0, 1.0)
        for dy in (-1, 0, 1)
    )

    return Blueprint(
        entities=tuple(entities),
        resources=resources,
        character_position=(0.0, 0.0),
        character_inventory={"small-electric-pole": 4},
        markers={"drill": (drill_x, 0.0)},
        radius=64,
    )


register(RegisteredTask(spec=SPEC, generate=generate))
