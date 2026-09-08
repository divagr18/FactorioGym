"""Construct a plate line from nothing, and keep it running (R3.1).

`plate_line` hands the agent two aligned machines and asks it to fuel them,
which establishes **commissioning**. This family establishes **construction**:
the scene contains ore and nothing else, the machines are in the agent's
inventory, and success requires that the agent placed both *and* that the line
is still producing at the end.

The geometry, measured rather than reasoned about
------------------------------------------------
Every number below came off a real engine (2.0.60), because the same geometry
was previously argued by hand in `plate_line`'s docstring and got the count of
valid answers wrong.

* `place_at` offers tile centres (`x.5`), and a 2x2 machine snaps to the
  integer centre one up: a drill requested at (0.5, 0.5) lands at (1, 1). So a
  requested tile centre `t + 0.5` becomes a snapped centre `t + 1`.
* A burner drill's `can_place_entity` is **false with no ore under it**, so the
  drill's position is not free -- it is chosen from the patch.
* A south-facing drill at (1, 1) drops at (1.5, 2.2969), i.e. into tile (1, 2).
* A 2x2 furnace at centre `(cx, cy)` occupies tiles `x in {cx-1, cx}`,
  `y in {cy-1, cy}`. The drop lands in its footprint when `cy = drill.y + 2`
  and `cx in {drill.x, drill.x + 1}` -- **two** legal centres, not one.
  `cy = drill.y + 1` also catches the drop but overlaps the drill, and
  `can_place_entity` refuses it. Measured: (1, 3) and (2, 3) both produce;
  (0, 3) produces nothing and the drill stalls with a full output slot.
* A correctly built line makes 14 plates in its first 3600 ticks and 15 per
  3600 thereafter -- the same rate `plate_line` commissions, which is what
  makes the two tasks a matched pair: commissioning's time-to-first-output is
  construction's floor.

Why success is a conjunction
----------------------------
`BUILT` alone would accept two machines dropped anywhere, and `PRODUCED` alone
would accept a single transient plate -- or, worse, a burst that stopped, since
it is a monotone counter with no timestamp. `SUSTAINED_OUTPUT` reads a windowed
history instead, so a line that ran once and died fails however large its
total. R3.1 asks for exactly this: "newly constructed required components and
sustained output, not just final inventory or one transient plate".

The window alone was not enough, and the gate caught it. A trailing window that
begins at tick 0 spans the construction phase, so acceptance at tick 3600 was
observed with a drill mined out 90 ticks earlier -- the window still held every
plate the line had ever made. `SETTLE_TICKS` moves the earliest acceptable
window to [3600, 7200], which is entirely post-construction.

Requiring the machines to be *currently* working would have been the obvious
alternative and it was measured and rejected: a healthy line reads status
`working` on 7.5% of sampled ticks, because one burner drill outpaces one stone
furnace and spends 87% of its time in `waiting_for_space_in_destination`.
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
    ResourceSpec,
    RewardComponent,
    RewardKind,
    TaskSpec,
)

#: Plates required inside the measurement window. Steady state is 15 per 3600
#: ticks and the first window yields 14, so 10 is reachable by any correctly
#: built line with margin for a slow builder -- and unreachable without one,
#: since iron plates cannot be hand-crafted at all. Picked from the measured
#: baseline, not by taste: the risk this task carries is that a badly chosen
#: window either fails every correct build or lets a burst pass.
WINDOW_PLATES = 10
WINDOW_TICKS = 3600

#: The criterion may not be satisfied before this tick, and the reason is
#: measured rather than stylistic. A trailing window that starts at tick 0
#: spans the construction phase, so at `WINDOW_TICKS` the window still holds
#: every plate the line ever made: the gate observed acceptance at tick 3600
#: with a drill that had been mined out 90 ticks earlier. Two windows means the
#: earliest acceptable window is [3600, 7200], entirely after construction, so
#: a line that stopped cannot pass however large its total.
SETTLE_TICKS = 2 * WINDOW_TICKS

#: Two of each machine, not one. A single misplacement is recoverable by
#: `mine_at`, but a family whose reference solver can wedge itself on one bad
#: guess measures the precision of that guess rather than construction. Coal is
#: sized off consumption: a burner drill is 150 kW and a stone furnace 90 kW
#: against coal's 4 MJ, so one coal lasts about 1600 and 2700 ticks
#: respectively -- under 20 apiece across the whole budget.
STARTING_INVENTORY = {"burner-mining-drill": 2, "stone-furnace": 2, "coal": 60}

#: Where the patch sits before a family shifts it.
PATCH_CENTRE = (0.0, 0.0)

FAMILIES = (
    LayoutFamily("square_patch", "train"),
    LayoutFamily("offset_patch", "val"),
    LayoutFamily("narrow_patch", "test"),
)

SPEC = TaskSpec(
    id="build_line",
    version="1.1.0",
    description=(
        "Build a plate line from nothing: place a burner mining drill on the ore, "
        "place a stone furnace where the drill drops, fuel both, and keep the line "
        "producing."
    ),
    layout_families=FAMILIES,
    success=(
        # Both machines, counted from placements made through the `place`
        # action, so a scene that preplaced them could not satisfy this even if
        # one did -- the install path does not go through that handler.
        Predicate(PredicateKind.BUILT, item="burner-mining-drill", at_least=1),
        Predicate(PredicateKind.BUILT, item="stone-furnace", at_least=1),
        Predicate(
            PredicateKind.SUSTAINED_OUTPUT,
            item="iron-plate",
            at_least=WINDOW_PLATES,
            over_ticks=WINDOW_TICKS,
            not_before_tick=SETTLE_TICKS,
        ),
    ),
    rewards=(
        RewardComponent("constructed", RewardKind.SPARSE_SUCCESS, weight=1.0, shaping=False),
        # Graded to the goal so its cap stays unreachable while the task is
        # unfinished, which is the shape `tools/reward_audit.py` requires. It
        # tracks cumulative production rather than the window: a component that
        # can *fall* is not a high-water mark, and rewarding the window's level
        # would pay repeatedly for holding still.
        RewardComponent(
            "plates_produced",
            RewardKind.HIGH_WATER,
            weight=0.15 / WINDOW_PLATES,
            cap=0.15,
            predicate=Predicate(PredicateKind.PRODUCED, item="iron-plate"),
        ),
        RewardComponent("step_cost", RewardKind.STEP_COST, weight=0.001),
    ),
    # Construction is fast and production is slow. At the default 30 ticks per
    # decision, 600 decisions is 18,000 ticks: roughly 600 to walk and build,
    # then the settling period, and the earliest acceptance is tick 7,200 or 240
    # decisions. Deliberately short of the 24,000 ticks it takes to fill the
    # furnace's 100-plate output slot, so the line cannot stall on a full output
    # and make the measured window read zero for a reason the agent did not
    # cause.
    max_decision_steps=600,
    max_game_ticks=18000,
    catalog="parameterized-v1",
    landmarks=(Predicate(PredicateKind.PRODUCED, item="iron-plate", at_least=1),),
    #: The patch is where all the work happens and it is not an entity, so
    #: nothing in the observation names it as a place.
    extra_public_markers=("patch",),
)


def _patch_tiles(family: LayoutFamily, rng) -> list[tuple[float, float]]:
    """Ore tiles. The patch's *shape* is the structural split.

    A drill needs ore under it, so the shape decides which drill positions and
    facings exist -- which makes it structure in the sense PLAN section 3
    means, rather than a reskin.
    """
    cx, cy = PATCH_CENTRE
    if family.name == "narrow_patch":
        # A two-tile-wide strip, in one of two orientations. The square-patch
        # layout does not transfer unexamined: the furnace has to go off the
        # ore, and half the drill facings have no ore to sit on.
        vertical = rng.random() < 0.5
        xs = range(-1, 1) if vertical else range(-5, 5)
        ys = range(-5, 5) if vertical else range(-1, 1)
        return [(cx + dx, cy + dy) for dx in xs for dy in ys]
    if family.name == "offset_patch":
        # The same square, shifted, so the split differs in where the work is
        # and not only in where the character starts.
        ox = rng.choice([-8, 8])
        oy = rng.choice([-8, 8])
        return [(cx + ox + dx, cy + oy + dy) for dx in range(-3, 4) for dy in range(-3, 4)]
    return [(cx + dx, cy + dy) for dx in range(-3, 4) for dy in range(-3, 4)]


def generate(family: LayoutFamily, rng) -> Blueprint:
    """One scene: an ore patch, an inventory, and nothing built."""
    tiles = _patch_tiles(family, rng)
    resources = tuple(
        ResourceSpec("iron-ore", (float(x), float(y)), amount=10000) for x, y in tiles
    )
    centre = (
        sum(x for x, _ in tiles) / len(tiles),
        sum(y for _, y in tiles) / len(tiles),
    )
    # Start off the patch, at a distance and bearing that vary, so the agent
    # cannot memorise a fixed approach. Far enough that walking is required,
    # near enough that `PLACEMENT_RADIUS` is reached in a few strides.
    angle = rng.uniform(0, 2 * math.pi)
    distance = rng.uniform(8.0, 13.0)
    start = (
        round(centre[0] + math.cos(angle) * distance, 1),
        round(centre[1] + math.sin(angle) * distance, 1),
    )

    entities: list[EntitySpec] = []
    if family.name == "narrow_patch":
        # A screen between the start and the patch, on integer tiles for the
        # banker's-rounding reason `plate_line` documents. Three tiles: long
        # enough to need rounding, short enough that the reference's sidestep
        # can round it, since a family its own reference cannot finish is a
        # defect and not a holdout.
        side = 1 if math.cos(angle) >= 0 else -1
        for offset in range(-1, 2):
            entities.append(
                EntitySpec(
                    "stone-wall",
                    (float(int(centre[0]) + side * 6), float(int(centre[1]) + offset)),
                    force="neutral",
                )
            )

    return Blueprint(
        entities=tuple(entities),
        resources=resources,
        character_position=start,
        character_inventory=dict(STARTING_INVENTORY),
        markers={"patch": (round(centre[0], 1), round(centre[1], 1))},
        # Both are placeable from the start in freeplay, but the action
        # profile's placement guard checks the recipe rather than the item, so
        # a task handing out a machine has to say the machine is unlocked.
        unlock_recipes=("burner-mining-drill", "stone-furnace"),
        radius=48,
    )


register(RegisteredTask(spec=SPEC, generate=generate, solve=None))
