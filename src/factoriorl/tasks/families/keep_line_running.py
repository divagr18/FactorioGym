"""Keep a running line running across a declared outage (R4.3).

R4.3's third track: *persistent operation*. The line arrives built, aligned and
fuelled, and it is producing before the agent does anything. Then, at a tick
the **task** declares, the evaluator empties both fuel inventories. Nothing
announces it. Acceptance is production *after* the outage, sustained.

Why the disruption is declared on the spec
------------------------------------------
Every disruption in this repo before now was a driver-side `bridge.run` string
fired at a hard-coded point in a five-phase script. The tick was a property of
the driver, the targets were a property of the string, and a run's trace could
not state either -- so two runs of "the same disruption" were not comparable.
`TaskSpec.disruptions` puts it in the task, in `to_dict()`, and so in the
config digest a run is compared by.

It also fixes the staleness that made the old demonstration's recovery claim
unreadable. The env applies a declared disruption from the one place every
observation refresh goes through and calls `resync` immediately, so the first
post-outage observation is fresh **by construction**. The driver that fired raw
Lua could not do that: its next prompt said "game tick 450" while the world was
at 4050, and reported a drill holding 48 coal at the moment it was empty.

The outage is not immediate, and the task is built around that
--------------------------------------------------------------
`empty_fuel` clears fuel *inventories* and leaves `entity.energy` and
`burner.remaining_burning_fuel` alone -- deliberately, because that is the
disruption R4.2 measured. A burner mining drill keeps producing for about
**1,170 ticks** on the item already in its burner: 2,999,833 J against 150 kW
is 1,199 ticks by arithmetic, and the measured curve agrees
(`docs/evidence/r4-burner-decay.json`).

So the accepted window has to begin *after* the run-on, or the task would pass
on production the outage had not yet stopped. That is exactly the error being
withdrawn from `phase5-demonstration.json`, which called a 90-tick refuel a
recovery. Here:

* the outage lands at tick 3,600;
* the run-on ends by about tick 4,800;
* `not_before_tick` is 8,400, so the earliest accepted window covers
  4,800..8,400 -- entirely after the run-on.

Which leaves the agent roughly 1,200 ticks, or 40 decisions, to notice and
refuel before it starts losing window. That grace period is the task.

Unannounced means unannounced
-----------------------------
Applying a disruption publishes no marker and appends no event. It is
detectable only by monitoring what is already observable, which since
`SUMMARY_ENCODING_VERSION` 3 includes each machine's status name, `working`
flag, fuel and output contents. Before that fix the prompt rendered `status 18`
and nothing else, and this task would have been unsolvable through the
language-model client for a reason that had nothing to do with the task.

Note what monitoring can and cannot see at the moment of the outage: both
burners keep reporting `working` while they run on stored energy, so the
*fuel* line is the tell, not the status. An agent that only watches status
learns about the outage 1,170 ticks late.

The splits
----------
Layout, not disruption: `disruptions` is declared per task, so every family
here receives the same outage and the holdout varies the scene. The axis is
`plate_line`'s, which the split audit already passes on -- and which is a
structural difference rather than a renamed distance, the mistake
`restore_power`'s `pole_gap_far` recorded.

* **train** -- `operation`: open ground.
* **val** -- `operation_scattered`: two decoy stone furnaces, unfuelled and
  unconnected, so "the nearest furnace" is not always the line's furnace.
* **test** -- `operation_walled`: a screen between the start and the line, so
  the agent must walk around it -- and must do so again to get back after the
  outage if it has wandered.
"""

from __future__ import annotations

import math

from factoriorl.tasks import RegisteredTask, register
from factoriorl.tasks.spec import (
    Blueprint,
    Disruption,
    EntitySpec,
    LayoutFamily,
    Predicate,
    PredicateKind,
    ResourceSpec,
    RewardComponent,
    RewardKind,
    TaskSpec,
)

#: The line, pinned as in `plate_line` and `diagnose_line`: a drill whose drop
#: tile moved would change what the measured window measures.
DRILL_POSITION = (1.0, 1.0)
FURNACE_POSITION = (1.0, 3.0)
ORE_RADIUS = 3

#: Fuel each machine starts with. Enough to be running well before the outage
#: and not so much that the cleared inventory leaves a second reserve.
FUELLED = 50

#: Coal the agent carries. Two stacks and a margin, as in `plate_line`:
#: `give_coal_20` fuels the *nearest* entity and the machines are two tiles
#: apart, so a misjudged approach spends a stack on the wrong machine.
STARTING_COAL = 120

#: When the outage lands, and what it touches.
OUTAGE_TICK = 3600

#: The measured run-on: production continues about this long after the fuel
#: inventories are emptied, on stored energy the clear does not touch.
RUN_ON_TICKS = 1200

WINDOW_TICKS = 3600
WINDOW_PLATES = 10

#: Earliest acceptance. The outage, plus the run-on, plus a full window -- so
#: the accepted window cannot contain a tick at which the line was still
#: running on energy it had before the outage.
SETTLE_TICKS = OUTAGE_TICK + RUN_ON_TICKS + WINDOW_TICKS

#: Start distance, in the regime `navigate` measures a 0.00 random floor at
#: (path length 24.7) and `diagnose_line` had to move into for the same reason:
#: every transfer in this catalog addresses the nearest entity, so with the
#: character already beside the machines a random policy is nearly a competent
#: one. Inside the 32-tile sensor radius, so the line is visible from the
#: first observation.
START_MIN_DISTANCE = 24.0
START_MAX_DISTANCE = 30.0

FAMILIES = (
    LayoutFamily("operation", "train"),
    LayoutFamily("operation_scattered", "val"),
    LayoutFamily("operation_walled", "test"),
)

SPEC = TaskSpec(
    id="keep_line_running",
    track="persistent_operation",
    version="1.0.0",
    description=(
        "This plate line is already running. Keep it producing iron plates for the "
        "rest of the episode, whatever happens to it."
    ),
    layout_families=FAMILIES,
    success=(
        Predicate(
            PredicateKind.SUSTAINED_OUTPUT,
            item="iron-plate",
            at_least=WINDOW_PLATES,
            over_ticks=WINDOW_TICKS,
            not_before_tick=SETTLE_TICKS,
        ),
    ),
    # Declared here, so it is in `to_dict()` and so in the config digest. The
    # env applies it through the refresh path and records what the engine
    # reported, including the case where it matched nothing.
    disruptions=(Disruption(kind="empty_fuel", at_tick=OUTAGE_TICK, targets=("drill", "furnace")),),
    rewards=(
        RewardComponent("kept_running", RewardKind.SPARSE_SUCCESS, weight=1.0, shaping=False),
        # Cumulative production, not the window's level: a component that can
        # fall is not a high-water mark. Graded to the goal so the cap stays
        # unreachable while the task is unfinished.
        RewardComponent(
            "plates_produced",
            RewardKind.HIGH_WATER,
            weight=0.15 / WINDOW_PLATES,
            cap=0.15,
            predicate=Predicate(PredicateKind.PRODUCED, item="iron-plate"),
        ),
        RewardComponent("step_cost", RewardKind.STEP_COST, weight=0.001),
    ),
    # SETTLE_TICKS is 8,400 and the window is judged on a trailing 3,600, so the
    # budget has to clear 8,400 with room for a slow start: 400 decisions is
    # 12,000 ticks. Short of the 24,000 that fill a furnace's output slot from
    # empty, so the line cannot stall on a full output for a reason the agent
    # did not cause.
    max_decision_steps=400,
    max_game_ticks=12000,
    difficulty_marker="line",
    landmarks=(Predicate(PredicateKind.PRODUCED, item="iron-plate", at_least=1),),
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
        "give_coal_5",
        "give_coal_20",
        "take_iron-plate_5",
        "wait",
    ),
)

#: Decoy furnaces on the val split, and how far out they sit. Unfuelled and on
#: no drill's drop tile, so coal put into one is coal wasted -- which is what
#: makes "the nearest furnace" worth checking.
DECOY_OFFSETS = ((6.0, 0.0), (-6.0, 2.0))

NUDGE_STEP = 1.0
NUDGE_ATTEMPTS = 8


def _clear_of(
    start: tuple[float, float], angle: float, blocked: set[tuple[int, int]]
) -> tuple[float, float]:
    """Push `start` outward along its bearing until its tile is free.

    Same helper and same reason as `plate_line`'s: `build_blueprint` teleports
    the character without a collision check, so a scene that draws a start
    independently of its obstacles can put the character inside one and nothing
    raises. The split audit found exactly that in 10 of 200 `plate_line` scenes.
    A no-op when the start is already clear.
    """
    position = start
    for _ in range(NUDGE_ATTEMPTS):
        if (math.floor(position[0]), math.floor(position[1])) not in blocked:
            return position
        position = (
            round(position[0] + math.cos(angle) * NUDGE_STEP, 1),
            round(position[1] + math.sin(angle) * NUDGE_STEP, 1),
        )
    return position


def generate(family: LayoutFamily, rng) -> Blueprint:
    """One running line, plus whatever the family puts between it and the agent."""
    angle = rng.uniform(0, 2 * math.pi)
    distance = rng.uniform(START_MIN_DISTANCE, START_MAX_DISTANCE)
    start = (
        round(DRILL_POSITION[0] + math.cos(angle) * distance, 1),
        round(DRILL_POSITION[1] + math.sin(angle) * distance, 1),
    )

    resources = [
        ResourceSpec("iron-ore", (DRILL_POSITION[0] + dx, DRILL_POSITION[1] + dy), amount=5000)
        for dx in range(-ORE_RADIUS, ORE_RADIUS + 1)
        for dy in range(-ORE_RADIUS, ORE_RADIUS + 1)
    ]

    entities = [
        EntitySpec(
            "burner-mining-drill",
            DRILL_POSITION,
            direction="south",
            contents={"coal": FUELLED},
            marker="drill",
        ),
        EntitySpec(
            "stone-furnace",
            FURNACE_POSITION,
            contents={"coal": FUELLED},
            marker="furnace",
        ),
    ]

    if family.name == "operation_scattered":
        for dx, dy in DECOY_OFFSETS:
            entities.append(
                EntitySpec("stone-furnace", (DRILL_POSITION[0] + dx, DRILL_POSITION[1] + dy))
            )

    if family.name == "operation_walled":
        # A screen the agent walks around. Three tiles, not nine: the reference
        # walks axis-first with no pathfinding, so a screen long enough to need
        # a real detour makes the *solver* fail rather than the task hard, and a
        # family its own reference cannot finish is a defect. `plate_line`
        # records the same bound for the same reason.
        side = 1 if math.cos(angle) >= 0 else -1
        blocked = {
            (int(DRILL_POSITION[0]) + side * 5, int(DRILL_POSITION[1]) + offset)
            for offset in range(-1, 2)
        }
        for x, y in sorted(blocked):
            entities.append(EntitySpec("stone-wall", (float(x), float(y)), force="neutral"))
        start = _clear_of(start, angle, blocked)

    return Blueprint(
        entities=tuple(entities),
        resources=tuple(resources),
        character_position=start,
        character_inventory={"coal": STARTING_COAL},
        # `line` only, and it is not published: the success predicate names no
        # marker and `extra_public_markers` is empty, so
        # `TaskSpec.public_markers` is empty. It is the pinned geometry the
        # split audit measures routes against.
        markers={"line": DRILL_POSITION},
        radius=48,
    )


register(RegisteredTask(spec=SPEC, generate=generate, solve=None))
