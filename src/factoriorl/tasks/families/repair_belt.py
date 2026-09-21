"""Repair a belt defect (DESIGN.md 3.2, family 5).

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
    # The gap's coordinate is published through `extra_public_markers`, so
    # the fault is given and the work is the fix. `describe_assistance` names
    # it `fault-location:published`.
    track="repair",
    # 1.1.0: the line's row and its starting column are sampled and the line
    # length moves in steps of two, so every family admits a distribution of
    # scenes and the holdout spans the pooled training difficulty. The version
    # is part of the random-baseline cache key, and none of the cached
    # baselines were measured against these scenes.
    # 1.6.0: the evaluated splits stop testing regimes training never showed.
    # Both `gap2` and `gap` are published, so a two-fault scene has a
    # coordinate for each fault rather than only for `min(gaps)`; training
    # spans one and two holes and with-and-without a misrotation; and the inert
    # `delivered` component is gone.
    version="1.6.0",
    description="Restore a broken belt line so items reach the unloading chest.",
    layout_families=FAMILIES,
    success=(
        Predicate(PredicateKind.CONTAINER_HOLDS, marker="sink", item="iron-plate", at_least=1),
    ),
    rewards=(
        RewardComponent("repaired", RewardKind.SPARSE_SUCCESS, weight=1.0, shaping=False),
        # `delivered` was removed here in 1.6.0. It was HIGH_WATER on
        # CONTAINER_HOLDS(sink, iron-plate) with cap 0.5, while `success` is the
        # same predicate at `at_least=1` and the env terminates on success -- so
        # it could only ever pay on the terminal transition and could never
        # approach its cap. `docs/evidence/reward-audit.json` already exempted
        # it ("cap needs 5, at or past the 1 for success"). Declared shaping
        # with provably no shaping effect is worse than none: the manifest read
        # as though this family had two gradients when it had one, and the
        # "190 episodes, zero successes, -0.300" note that used to stand here
        # was itself an artifact of the vector-summing curve logger.
        #
        # The only thing that pays before the belt is repaired:
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
    extra_public_markers=("gap", "gap2"),
    # Where the fault is, for the coverage check. Not in `to_dict()`:
    # measurement metadata, so it moves no digest.
    fault_markers=("gap", "gap2"),
    # Two faults, three goal-geometry slots: the environment points at the
    # nearest unrepaired one. Declared because it decides which fault to
    # attack on the policy's behalf, which the manifest must show.
    focus_policy="nearest_unrepaired",
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
    # Fault *count* and fault *kind* both have to appear in training, or the
    # holdout measures an extrapolation instead of a transfer.
    #
    # This generator used to give every training family exactly one hole and no
    # misrotation, while `double_gap` (test) always had two holes and
    # `misrotation` (val) always had a hole plus a crooked belt. A policy that
    # had perfectly learned the training task therefore scored zero on both
    # evaluated splits by construction, and 50,000 steps duly produced 0.00
    # with 1 success in 198 episodes. The same defect on a different axis cost
    # `restore_power` two of three seeds; see
    # docs/evidence/holdout-distribution-audit.json.
    #
    # Training now spans both regimes: a third of `gap`/`gap_far` scenes carry
    # a second hole, and a third carry a misrotation. The evaluated families
    # keep their identities -- `double_gap` is still always two holes and
    # `misrotation` always a hole plus a twist -- so each still names a
    # structural regime, but it is a regime training has shown instances of.
    second_gap = family.name == "double_gap" or (family.split == "train" and rng.random() < 1 / 3)
    if second_gap:
        # Distinct, or `double_gap` silently degenerates into `gap`.
        choices = [i for i in range(2, length - 1) if i != gap_index]
        if choices:
            gaps.add(choices[rng.randrange(len(choices))])
    twisted = family.name == "misrotation" or (family.split == "train" and rng.random() < 1 / 3)

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
        if twisted and index == gap_index + 1 and index < length:
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
            # Published, not evaluator-only. This said the opposite until
            # R4.6: no success or failure predicate names `gap`, but
            # `extra_public_markers` above lists it, and
            # `TaskSpec.public_markers` takes *both* sources -- so `gap` and
            # `gap2` have been in the observation since v1.6.0. The comment
            # survived the change that falsified it, sitting 170 lines below
            # the declaration that falsifies it.
            #
            # It matters beyond tidiness: whether the fault coordinate is in
            # the policy input is the difference between "repair a line" and
            # "walk to a published tile and press place", which is the seam
            # R4.3's track split runs along. `describe_assistance` now names
            # it as `fault-location:published`.
            # +0.5 on both axes: a 1x1 belt created at (x, y) is snapped by
            # the engine to the tile centre (x+0.5, y+0.5), so the pre-snap
            # coordinate names a point half a tile off the slot the belt
            # actually occupies. Small against a 64-tile normalisation, and
            # wrong.
            **{
                ("gap" if rank == 0 else f"gap{rank + 1}"): (
                    float(start_x + index) + 0.5,
                    row + 0.5,
                )
                for rank, index in enumerate(sorted(gaps))
            },
        },
        radius=64,
    )


register(RegisteredTask(spec=SPEC, generate=generate))
