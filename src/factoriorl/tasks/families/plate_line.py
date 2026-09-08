"""An automated plate line, for the Phase 5 demonstration (PLAN.md 5.7).

    Construct an automated plate line, inject a documented disruption, and
    restore sustained production.

The line is a burner mining drill sitting on iron ore with a stone furnace on
its drop tile. Measured on a real engine: it produces about thirty plates per
7,200 ticks with the character standing still, until the furnace's output slot
fills at a hundred. That is what makes a measurement window meaningful -- during
it nobody replenishes anything, and production either continues or it does not.

What the agent does, and what it does not
-----------------------------------------
Both machines are placed and aligned by the scene, and **unfuelled**. The agent
commissions the line: it must reach each machine and put coal in it. Then the
line runs on its own, and the disruption -- a fuel outage, injected by the
demonstration driver and recorded -- stops it until the agent refuels.

The agent is deliberately not asked to *place* the machines, and the reason is a
property of the action catalog rather than a shortcut:

* `place_<item>_<direction>` builds at the character's position plus a one-tile
  offset, so a placement's position is quantised to where the character happens
  to stand.
* A burner mining drill and a stone furnace are both 2x2 and snap to integer
  centres. A drill at (1, 1) drops its ore at (1.5, 2.30).
* The only furnace centre that catches that drop is (1, 3): a furnace at (1, 2)
  overlaps the drill and `can_place_entity` refuses it.
* A furnace centred at (1, 3) covers tiles y in {2, 3} -- including the tile the
  character must stand on to reach (1, 3) with a one-tile offset. So the
  placement that would build the line collides with the builder.

Aligning a furnace to a drill's drop tile is therefore not reliably expressible
in this catalog, which is worth knowing before Phase 7 asks for persistent
factories built the same way. `restore_power`'s solvability run already shows
the milder version of it: `place_small_electric_pole_south: collision`.
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

#: Plates that count as the line having run. One furnace makes roughly one plate
#: every 240 ticks, so this is about an eighth of the tick budget spent
#: producing -- long enough that a single hand-fed smelt cannot reach it.
TARGET_COUNT = 30

#: Coal handed to the agent. A burner drill's fuel slot holds a full stack of
#: fifty, and `give_coal_20` moves coal to the *nearest* entity, so an agent
#: that fuels the drill first can put its whole supply into that one machine
#: before it ever reaches the furnace. The first demonstration run did exactly
#: that -- 47 coal into the drill, 3 left, and the furnace never lit -- which
#: measured the scene's provisioning rather than the agent. Two full stacks
#: plus a margin for the disruption's refuelling removes the trap without
#: removing the task: the agent still has to reach both machines.
STARTING_COAL = 120

#: Where the drill sits relative to the scene origin, and which way it faces.
#: Fixed rather than drawn: the demonstration is a scripted scenario, and a
#: drill whose drop tile moved between runs would make two runs incomparable.
DRILL_POSITION = (1.0, 1.0)
FURNACE_POSITION = (1.0, 3.0)
ORE_RADIUS = 3

FAMILIES = (
    LayoutFamily("commissioning", "train"),
    LayoutFamily("commissioning_far", "val"),
    LayoutFamily("commissioning_walled", "test"),
)

SPEC = TaskSpec(
    id="plate_line",
    version="1.1.0",
    description=(
        "Commission an automated plate line: fuel the mining drill and the furnace "
        "so the line produces iron plates without further help."
    ),
    layout_families=FAMILIES,
    success=(Predicate(PredicateKind.PRODUCED, item="iron-plate", at_least=TARGET_COUNT),),
    rewards=(
        RewardComponent("commissioned", RewardKind.SPARSE_SUCCESS, weight=1.0, shaping=False),
        # Graded to the goal, so its cap is unreachable while the task is
        # unfinished -- the shape `tools/reward_audit.py` requires.
        RewardComponent(
            "plates_produced",
            RewardKind.HIGH_WATER,
            weight=0.15 / TARGET_COUNT,
            cap=0.15,
            predicate=Predicate(PredicateKind.PRODUCED, item="iron-plate"),
        ),
        RewardComponent("step_cost", RewardKind.STEP_COST, weight=0.001),
    ),
    # Smelting is slow, so the budget is dominated by waiting rather than acting:
    # 30 plates is roughly 7,200 ticks of production once the line runs.
    max_decision_steps=400,
    max_game_ticks=24000,
    difficulty_marker="furnace",
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


def generate(family: LayoutFamily, rng) -> Blueprint:
    """One scene: an ore patch, an unfuelled drill, an unfuelled furnace.

    Only the character's start varies. The machines are pinned because the
    demonstration compares two measurement windows within one run, and a line
    whose geometry moved would change what the second window is measuring.
    """
    angle = rng.uniform(0, 2 * math.pi)
    distance = (
        rng.uniform(12.0, 16.0) if family.name == "commissioning_far" else rng.uniform(5.0, 9.0)
    )
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
            marker="drill",
        ),
        EntitySpec("stone-furnace", FURNACE_POSITION, marker="furnace"),
    ]

    if family.name == "commissioning_walled":
        # The structural holdout: a screen the agent has to walk around to reach
        # the machines, so the test split differs in layout rather than only in
        # where the character starts.
        side = 1 if math.cos(angle) >= 0 else -1
        # Three tiles, not nine. The reference solution walks axis-first with no
        # pathfinding -- PLAN defers that to 5.1 -- so a screen long enough to
        # need a real detour makes the *solver* fail rather than the task hard,
        # and a family its own reference cannot finish is a defect, not a
        # holdout. Three has to be rounded and can be.
        for offset in range(-1, 2):
            entities.append(
                EntitySpec(
                    # Integer tiles. `footprint_conflicts` keys on `round()`,
                    # which is banker's rounding, so a column spaced on .5
                    # boundaries collides with itself in pairs: round(-2.5) and
                    # round(-1.5) are both -2.
                    "stone-wall",
                    (
                        float(int(DRILL_POSITION[0]) + side * 5),
                        float(int(DRILL_POSITION[1]) + offset),
                    ),
                    force="neutral",
                )
            )

    return Blueprint(
        entities=tuple(entities),
        resources=tuple(resources),
        character_position=start,
        character_inventory={"coal": STARTING_COAL},
        markers={"line": DRILL_POSITION},
        radius=48,
    )


register(RegisteredTask(spec=SPEC, generate=generate, solve=None))
