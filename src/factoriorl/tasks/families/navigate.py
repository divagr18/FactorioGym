"""Navigate to a work site (PLAN.md 3.2, family 1).

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
    # Held out structurally, not merely by seed.
    LayoutFamily("wall_ring", "test"),
)

SPEC = TaskSpec(
    id="navigate",
    version="1.0.0",
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
    max_decision_steps=150,
    max_game_ticks=9000,
    catalog="primitive-v1",
    catalog_subset=("move_north", "move_east", "move_south", "move_west", "wait"),
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
    elif family.name == "wall_ring":
        for step in range(0, 360, 30):
            angle = math.radians(step)
            radius = distance * 0.6
            position = (round(math.cos(angle) * radius, 1), round(math.sin(angle) * radius, 1))
            if math.dist(position, goal) > 4 and math.hypot(*position) > 4:
                entities.append(EntitySpec("stone-wall", position))

    return Blueprint(
        entities=tuple(entities),
        character_position=(0.0, 0.0),
        markers={"goal": goal},
        radius=64,
    )


register(RegisteredTask(spec=SPEC, generate=generate))
