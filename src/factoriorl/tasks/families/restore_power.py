"""Restore a power connection (DESIGN.md 3.2, family 6).

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
    # As `repair_belt`: the missing pole's tile is published, so this is a
    # fix rather than a search.
    track="repair",
    # 1.1.0: the line's row and starting column are sampled, so the holdout
    # admits a distribution of scenes rather than a single one.
    # 1.6.0: training samples the terminal gap position the holdout always
    # uses, and both missing poles are published on `two_gaps` instead of only
    # `min(gaps)`.
    version="1.6.0",
    description="Reconnect a power pole chain so the mining drill runs again.",
    layout_families=FAMILIES,
    success=(Predicate(PredicateKind.ENTITY_WORKING, marker="drill"),),
    rewards=(
        RewardComponent("restored", RewardKind.SPARSE_SUCCESS, weight=1.0, shaping=False),
        # This family had *no* shaping component whatsoever: a sparse success
        # and a step cost, so the reward was a constant negative drip for the
        # whole of training and undirected exploration could never find the
        # goal. See repair_belt for the measurement and the reasoning; the same
        # citation applies (S&B §17.4 p.386, Wiewiora 2003), and potential-based
        # shaping cannot change which policy is optimal.
        RewardComponent(
            "toward_gap",
            RewardKind.POTENTIAL,
            weight=0.5,
            predicate=Predicate(PredicateKind.CHARACTER_WITHIN, marker="gap"),
        ),
        RewardComponent("step_cost", RewardKind.STEP_COST, weight=0.001),
    ),
    # 1/(1-gamma) = 1000, comfortably past this family's episode, so the
    # potential's one-time approach gain exceeds its per-step drag instead
    # of being swamped by it. At the 0.99 default the horizon is 100 steps.
    gamma=0.999,
    max_decision_steps=250,
    max_game_ticks=15000,
    # The gap is where the agent must act; the success predicate names only
    # where the result is counted. Without this the `toward_gap` potential
    # paid for approaching a point the observation never contained.
    extra_public_markers=("gap", "gap2"),
    # Where the fault is, for the coverage check. Not in `to_dict()`:
    # measurement metadata, so it moves no digest.
    fault_markers=("gap", "gap2"),
    # Two faults, three goal-geometry slots: the environment points at the
    # nearest unrepaired one. Declared because it decides which fault to
    # attack on the policy's behalf, which the manifest must show.
    focus_policy="nearest_unrepaired",
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
    # The held-out family fixes the fault at the chain's terminal position while
    # training used to sample `randint(1, chain - 2)`, which never reaches it.
    # Measured over 600 scenes per family: the missing pole was flanked by poles
    # on both sides in 600/600 training scenes and 0/600 held-out scenes, where
    # it is instead always the last position with the drill beyond it and no
    # pole further along. Two policies then fit training equally well -- "fill
    # the hole between two poles", which scores exactly 0.00 on that holdout by
    # construction, and "walk to the published gap and place", which transfers.
    # Which one a run learned was a seed lottery, and that is the bimodal
    # 0.77 / 0.00 / 0.00 recorded in phase4-release-v3-restore_power*.json --
    # not transfer variance. See docs/evidence/holdout-distribution-audit.json.
    #
    # Training now includes the terminal position, so the holdout tests a
    # regime training has shown instances of. `gap_near_drill` keeps its
    # identity: it is still *always* terminal, which is what its name claims.
    gap = rng.randint(1, chain - 1)
    if family.name == "gap_near_drill":
        gap = chain - 1
    gaps = {gap}
    if family.name == "two_gaps":
        second = min(chain - 1, gap + 2)
        if second == gap:
            second = max(1, gap - 2)
        gaps.add(second)

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
        markers={
            "drill": (drill_x, row),
            # Published, like repair_belt's, and for the same reason: no
            # predicate names `gap`, but `extra_public_markers` above lists it
            # and `public_markers` unions both sources, so it has been in the
            # observation since v1.6.0. This comment claimed the opposite
            # until R4.6. The first missing pole in the chain is the place the
            # agent has to get to before anything can happen -- and the agent
            # is told where it is.
            # Tile centre, as in repair_belt: a 1x1 pole snaps to (x+.5, y+.5).
            # Every missing pole, not just the first. `two_gaps` published only
            # `min(gaps)`, so the second fault had no coordinate anywhere the
            # policy could read and that family scored 0.00 for the same reason
            # `repair_belt`'s two-hole holdout did.
            **{
                ("gap" if rank == 0 else f"gap{rank + 1}"): (
                    float(start_x + index * 4) + 0.5,
                    row + 0.5,
                )
                for rank, index in enumerate(sorted(gaps))
            },
        },
        radius=64,
    )


register(RegisteredTask(spec=SPEC, generate=generate))
