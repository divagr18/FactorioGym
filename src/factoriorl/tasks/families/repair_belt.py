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
    # 1.1.0: the line's row and its starting column are sampled and the line
    # length moves in steps of two, so every family admits a distribution of
    # scenes and the holdout spans the pooled training difficulty. The version
    # is part of the random-baseline cache key, and none of the cached
    # baselines were measured against these scenes.
    version="1.5.0",
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
        # The only thing that pays before the belt is repaired.
        #
        # Measured without it: 190 training episodes over 50,000 steps, zero
        # successes, mean episode reward -0.300 -- exactly the step cost, because
        # `delivered` fires on a plate in the sink and success needs one plate,
        # so the shaping could only ever pay for winning. The policy had nothing
        # to ascend and no budget fixes that.
        #
        # Sutton & Barto §17.4 (p.386) warns against answering that with
        # hand-designed subgoal rewards and recommends initialising the value
        # function instead; Wiewiora (2003) shows the two are equivalent, and
        # potential-based shaping is the form this codebase already implements
        # and tests. Being potential-based it cannot change which policy is
        # optimal, so pointing it at the gap rather than at the true subgoal
        # costs learning speed and not correctness.
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
    max_decision_steps=300,
    max_game_ticks=18000,
    # The gap is where the agent must act; the success predicate names only
    # where the result is counted. Without this the `toward_gap` potential
    # paid for approaching a point the observation never contained.
    extra_public_markers=("gap",),
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


def generate(family: LayoutFamily, rng) -> Blueprint:
    # The line's row and its starting column used to be the constants `LINE_Y =
    # 0.0` and `start_x = 4`, so the only randomised axes were the length and
    # the gap index and `gap` admitted 18 distinct scenes over 200 seeds. A
    # hundred evaluation episodes against an 18-scene generator are not a
    # hundred independent draws, and the Wilson interval published beside them
    # assumes they are.
    #
    # Both are sampled now. Neither changes what a family *is* -- the fault
    # kind still distinguishes them -- and neither disturbs the scripted
    # solver, which infers the belt row as the modal row of the belts it can
    # see and reads no coordinate out of the task declaration.
    #
    # Rows are negative, so the scene always sits north of the spawn. Two
    # reasons, both load-bearing: (0, 0) then cannot land inside an entity for
    # any sampled `start_x`, and the solver's approach tile -- one tile *south*
    # of the gap -- is always on the spawn's side of the line, so the walk never
    # has to cross the row it is repairing.
    row = float(rng.randint(-10, -3))
    # Bounded above so the far end of the longest line stays inside the 32-tile
    # sensor radius of a character standing at spawn: the sink of a 14-tile
    # line at start_x = 8 sits at (23, row), 25 tiles from (0, 0) at the
    # furthest row. That bound matters because `solve_repair_belt` infers the
    # gap from the entity list *before* it walks anywhere, and an inference from
    # a truncated list is what broke `restore_power`. Truncation cannot invent a
    # gap here -- the sensor region is a circle and the line is a single row, so
    # what it clips is a prefix or a suffix, never a hole -- but it can hide the
    # real one, and a hidden gap is an unsolvable episode.
    start_x = rng.randint(3, 8)
    # Lengths move in steps of two, which is a deliberate choice rather than a
    # tidy one. The only measured descriptor that can see a second gap is the
    # number of belts left standing, and over contiguous lengths "one gap in an
    # n-tile line" leaves exactly as many belts as "two gaps in an (n+1)-tile
    # line": the holdout would be structurally indistinguishable from a
    # training scene one tile shorter, which is the thing its name claims to
    # differ by. With even lengths a one-gap family always leaves an odd number
    # of belts and a two-gap family an even number, at every length, so the
    # fault *count* is visible in the content and not merely in the label.
    if family.name == "gap_far":
        # `gap_far` used to be a byte-identical copy of `gap`: the name promised
        # a distance variation the generator never implemented, so two of the
        # four declared layout families were the same content. It is the long
        # line, with the fault out at its far end.
        length = 2 * rng.randint(6, 7)
        gap_index = rng.randint(length - 4, length - 2)
    elif family.name == "double_gap":
        # The holdout spans the *pooled* training range rather than one
        # family's. Training pools `gap` (6-10 tiles) with `gap_far` (12-14), so
        # a holdout fixed at the short end sits systematically below the
        # training pool: a score against it measures an easier task rather than
        # a transferred one, which is precisely what difficulty parity exists to
        # catch.
        length = 2 * rng.randint(3, 7)
        gap_index = rng.randint(2, length - 2)
    else:
        length = 2 * rng.randint(3, 5)
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
            (float(start_x - 2), row),
            contents={"iron-plate": 40},
            marker="source",
        ),
        EntitySpec("wooden-chest", (float(start_x + length + 1), row), marker="sink"),
    ]
    for index in range(length):
        if index in gaps:
            continue
        direction = "east"
        if family.name == "misrotation" and index == gap_index + 1 and index < length:
            direction = "north"
        entities.append(
            EntitySpec("transport-belt", (float(start_x + index), row), direction=direction)
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
            (float(start_x - 1), row),
            direction="west",
            marker="loader",
            contents={"coal": 5},
        )
    )
    entities.append(
        EntitySpec(
            "burner-inserter",
            (float(start_x + length), row),
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
        markers={
            "sink": (float(start_x + length + 1), row),
            # Evaluator-only: `gap` is named by no success or failure predicate,
            # so `TaskSpec.public_markers` does not publish it and it never
            # reaches an observation. Shaping may read truth -- the evaluator
            # computes the reward -- but the policy input may not.
            # +0.5 on both axes: a 1x1 belt created at (x, y) is snapped by
            # the engine to the tile centre (x+0.5, y+0.5), so the pre-snap
            # coordinate names a point half a tile off the slot the belt
            # actually occupies. Small against a 64-tile normalisation, and
            # wrong.
            "gap": (float(start_x + min(gaps)) + 0.5, row + 0.5),
        },
        radius=64,
    )


register(RegisteredTask(spec=SPEC, generate=generate))
