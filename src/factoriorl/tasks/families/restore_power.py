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
    # 1.1.0: the line's row and starting column are sampled, so the holdout
    # admits a distribution of scenes rather than a single one.
    version="1.1.0",
    description="Reconnect a power pole chain so the mining drill runs again.",
    layout_families=FAMILIES,
    success=(Predicate(PredicateKind.ENTITY_WORKING, marker="drill"),),
    rewards=(
        RewardComponent("restored", RewardKind.SPARSE_SUCCESS, weight=1.0, shaping=False),
        RewardComponent("step_cost", RewardKind.STEP_COST, weight=0.001),
    ),
    max_decision_steps=250,
    max_game_ticks=15000,
    landmarks=(Predicate(PredicateKind.INVENTORY_HOLDS, item="small-electric-pole", at_least=1),),
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
        "place_small_electric_pole_north",
        "place_small_electric_pole_east",
        "place_small_electric_pole_south",
        "place_small_electric_pole_west",
        "wait",
    ),
)


def generate(family: LayoutFamily, rng) -> Blueprint:
    # `pole_gap_far` was a byte-identical copy of `pole_gap` for the same
    # reason `gap_far` was a copy of `gap`: a name implying distance, and a
    # generator with no distance parameter.
    # Every coordinate in this family used to be a constant and the only
    # randomised axis was the gap index, so `pole_gap` admitted three scenes and
    # the held-out `gap_near_drill` -- which overwrites the sampled gap with
    # `chain - 1` -- admitted exactly **one**. A hundred evaluation episodes
    # against a one-scene generator measure one scene a hundred times while the
    # Wilson interval is computed as though there were a hundred draws, so the
    # holdout could not have supported any claim at all.
    #
    # The line's row and its starting column are now sampled, which multiplies
    # the space without touching what any family *is*: the fault kind still
    # distinguishes them, and the gap remains at the far end for the holdout.
    row = float(rng.randint(-12, -6))
    # Bounded at 3, not 7, for two independent reasons. The pole feeding the
    # solar farm sits at x = -1 on the same row, and a small pole's wire reach
    # is 7.5 tiles, so a chain starting past x = 6 would never connect to its
    # own power source. And a first pole more than 4 tiles from x = -1 opens a
    # gap the solver would read as *the* fault, sending it to place a pole into
    # an occupied tile.
    start_x = rng.randint(1, 3)
    if family.name == "pole_gap_far":
        chain = rng.randint(7, 9)
    elif family.name == "gap_near_drill":
        # The holdout spans the *pooled* training range rather than one family's.
        # Training is `pole_gap` at 4-6 and `pole_gap_far` at 7-9, so a holdout
        # fixed at 4-6 is systematically shorter than the training pool and a
        # score against it measures an easier task rather than a transferred
        # one -- which is what difficulty parity is there to prevent.
        chain = rng.randint(4, 9)
    else:
        chain = rng.randint(4, 6)
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
    #
    # The two poles on the row above the farm are load-bearing, not decoration.
    # A small pole's supply area is 5x5, so the pole at x=-1 reaches only to
    # x=-3: it collected the nearest panel and left the other two reporting
    # `not_plugged_in_electric_network`. One connected panel is 60 kW against a
    # 90 kW drill, so the drill stayed unpowered however well the chain was
    # repaired. These two poles bring all three panels (180 kW) onto the
    # network and are within the 7.5-tile wire reach of each other and of the
    # chain head.
    entities = [
        EntitySpec("solar-panel", (-10.0, row), marker="generator"),
        EntitySpec("solar-panel", (-7.0, row)),
        EntitySpec("solar-panel", (-4.0, row)),
        EntitySpec("small-electric-pole", (-9.0, row - 3.0)),
        EntitySpec("small-electric-pole", (-5.0, row - 3.0)),
        EntitySpec("small-electric-pole", (-1.0, row)),
    ]
    for index in range(chain):
        if index in gaps:
            continue
        entities.append(EntitySpec("small-electric-pole", (float(start_x + index * 4), row)))
    # One tile closer than the pole spacing: a 3x3 drill centred 4 tiles from
    # the last pole has its near edge exactly on the boundary of that pole's
    # supply area, which does not count as covered.
    drill_x = float(start_x - 1 + chain * 4)
    entities.append(
        EntitySpec("electric-mining-drill", (drill_x, row), direction="south", marker="drill")
    )
    resources = tuple(
        ResourceSpec("iron-ore", (drill_x + dx, float(dy) + row), amount=1000)
        for dx in (-1.0, 0.0, 1.0)
        for dy in (-1, 0, 1)
    )

    return Blueprint(
        entities=tuple(entities),
        resources=resources,
        character_position=(0.0, 0.0),
        character_inventory={"small-electric-pole": 4},
        unlock_recipes=("small-electric-pole",),
        markers={"drill": (drill_x, row)},
        radius=64,
    )


register(RegisteredTask(spec=SPEC, generate=generate))
