"""Navigate to a work site (DESIGN.md 3.2, family 1).

The simplest family, and deliberately so: it is the vertical slice every other
family copies, and the one that answers "does the whole pipeline work" in the
Phase 4 pilot before any harder task is attempted.
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
    RewardComponent,
    RewardKind,
    TaskSpec,
)

FAMILIES = (
    LayoutFamily("open", "train"),
    LayoutFamily("wall_corridor", "train"),
    LayoutFamily("pillar_field", "val"),
    # Held out structurally at *matched* difficulty: the same goal-distance
    # range as training, a different obstacle topology. A holdout that is
    # simply farther away measures difficulty, not transfer.
    LayoutFamily("wall_arc", "test"),
)

SPEC = TaskSpec(
    id="navigate",
    # Reach a published position. No construction, no fault.
    track="movement",
    version="1.2.0",
    description="Walk to a marked work site within the step budget.",
    layout_families=FAMILIES,
    success=(Predicate(PredicateKind.CHARACTER_WITHIN, marker="goal", within=2.0),),
    rewards=(
        RewardComponent("arrived", RewardKind.SPARSE_SUCCESS, weight=1.0, shaping=False),
        RewardComponent(
            "approach",
            RewardKind.POTENTIAL,
            weight=1.0,
            predicate=Predicate(PredicateKind.CHARACTER_WITHIN, marker="goal"),
        ),
        RewardComponent("step_cost", RewardKind.STEP_COST, weight=0.001),
    ),
    # 1/(1-gamma) = 333, past this family's 150-step budget. At the 0.99
    # default the horizon is 100 and the potential's per-step drag outweighs
    # its approach gain by 1.5x. navigate got away with it because its
    # episodes end in a handful of decisions, so the drag never accumulates
    # -- but that is a property of the task being easy, not of the shaping
    # being right, and the same setting on a 300-step family measured a
    # reward of -2.5.
    gamma=0.997,
    max_decision_steps=150,
    max_game_ticks=9000,
    catalog="primitive-v1",
    catalog_subset=(
        "move_north",
        "move_east",
        "move_south",
        "move_west",
        "step_north",
        "step_east",
        "step_south",
        "step_west",
        "wait",
    ),
)


def generate(family: LayoutFamily, rng) -> Blueprint:
    bearing = rng.uniform(0, 2 * math.pi)
    distance = rng.uniform(12.0, 30.0)
    goal = (round(math.cos(bearing) * distance, 1), round(math.sin(bearing) * distance, 1))

    entities: list[EntitySpec] = [
        # A chest marks the site: it is visible to the sensor, so the goal is
        # something the policy can actually see rather than an invisible point.
        EntitySpec("wooden-chest", goal, marker="goal"),
    ]

    if family.name == "wall_corridor":
        mid_x = goal[0] / 2
        for offset in range(-4, 5):
            if abs(offset) > 1:
                entities.append(EntitySpec("stone-wall", (round(mid_x, 1), float(offset))))
    elif family.name == "pillar_field":
        # Distinct tiles only. Sampling positions independently let two pillars
        # land on the same tile roughly once in 130 seeds, which the engine
        # would refuse to build -- a generator defect, not a task variation.
        taken: set[tuple[int, int]] = {(0, 0), (round(goal[0]), round(goal[1]))}
        attempts = 0
        while len(taken) < 8 and attempts < 100:
            attempts += 1
            candidate = (rng.randint(-20, 20), rng.randint(-20, 20))
            if candidate in taken:
                continue
            taken.add(candidate)
            entities.append(EntitySpec("stone-wall", (float(candidate[0]), float(candidate[1]))))
    elif family.name == "wall_arc":
        # An arc across the direct line, open at both ends: a different
        # topology to walk around, not a harder one to reach.
        taken = set()
        for step in range(-60, 61, 15):
            angle = bearing + math.radians(step)
            radius = distance * 0.55
            tile = (round(math.cos(angle) * radius), round(math.sin(angle) * radius))
            if tile in taken or math.dist(tile, goal) < 3 or math.hypot(*tile) < 3:
                continue
            taken.add(tile)
            entities.append(EntitySpec("stone-wall", (float(tile[0]), float(tile[1]))))

    return Blueprint(
        entities=tuple(entities),
        character_position=(0.0, 0.0),
        markers={"goal": goal},
        radius=64,
    )


register(RegisteredTask(spec=SPEC, generate=generate))
