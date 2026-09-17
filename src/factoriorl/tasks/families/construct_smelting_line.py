"""T1's verifiable, agent-built iron-smelting task.

Unlike the older ``build_line`` family, this task has a hard episode boundary:
when construction ends, the evaluator advances 3,600 ticks without dispatching an
action and scores only the machine-production delta in that window.
Construction ends when a driver calls ``run_verification`` (the agentic bridge's
``finish``), or automatically when the episode's decision or construction-tick
budget runs out, which is how an RL episode reaches it. The initial scene
contains ore and starting equipment, but no machine, plate, or preloaded
inventory.

**1.1.0** caps the counted plates by the iron ore machines mined during the same
window. 1.0.0 paid for a furnace fed by hand: ``machine_produced`` subtracts
hand-mined *ore* but not the plates smelted from it, so a furnace, some coal and
ten hand-mined ore scored 1.0 with no drill. With actions locked during the
window, ore mined in it came from a drill, so a drill-less line now scores 0.
The measured drill-to-furnace line is drill-limited at about fifteen plates a
minute, so a working line still clears the target.

**1.1.1** changes no rule, only a measurement under it. The mod's hand-mine
counted any change in held iron ore as mining, so taking ore out of a furnace
during a mine was tallied as mined, and giving ore away was the reverse. The
tally is subtracted from ``machine_produced``, which the cap reads, so a
working line could score 0 after such a transfer. The mod now counts only what
a mine mined (``tests/engine/test_mine_counts_only_mining.py``).
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
    VerificationSpec,
)

TARGET_PLATES = 10
VERIFICATION_TICKS = 3600
CONSTRUCTION_TICKS = 18000

FAMILIES = (
    LayoutFamily("open_patch", "train"),
    LayoutFamily("offset_patch", "train"),
    LayoutFamily("obstructed_patch", "test"),
)

SPEC = TaskSpec(
    id="construct_smelting_line",
    version="1.1.1",
    description=(
        "Build and fuel an iron-smelting line, then finish. Success is at least "
        "ten iron plates made by machines during an action-locked verification minute."
    ),
    layout_families=FAMILIES,
    success=(
        Predicate(PredicateKind.VERIFIED_MACHINE_OUTPUT, item="iron-plate", at_least=TARGET_PLATES),
    ),
    rewards=(
        RewardComponent("verified_output", RewardKind.SPARSE_SUCCESS, weight=1.0, shaping=False),
    ),
    track="production",
    max_decision_steps=600,
    max_game_ticks=CONSTRUCTION_TICKS + VERIFICATION_TICKS,
    verification=VerificationSpec(
        "iron-plate", TARGET_PLATES, VERIFICATION_TICKS, source="iron-ore"
    ),
    catalog="parameterized-v1",
    extra_public_markers=("patch",),
    difficulty_marker="patch",
)


def _ore_tiles(family: LayoutFamily, rng) -> list[tuple[int, int]]:
    ox, oy = (0, 0)
    if family.name == "offset_patch":
        ox, oy = rng.choice((-9, 9)), rng.choice((-9, 9))
    if family.name == "obstructed_patch":
        # Structural, held-out change: a narrow patch gives fewer legal drill
        # centres and the short wall requires an approach rather than a straight
        # walk. It never overlaps ore or the player start.
        return [(ox + x, oy + y) for x in range(-1, 2) for y in range(-5, 6)]
    return [(ox + x, oy + y) for x in range(-3, 4) for y in range(-3, 4)]


def generate(family: LayoutFamily, rng) -> Blueprint:
    tiles = _ore_tiles(family, rng)
    cx = sum(x for x, _ in tiles) / len(tiles)
    cy = sum(y for _, y in tiles) / len(tiles)
    angle = rng.uniform(0, 2 * math.pi)
    start = (
        round(cx + math.cos(angle) * rng.uniform(9, 13), 1),
        round(cy + math.sin(angle) * rng.uniform(9, 13), 1),
    )
    entities: list[EntitySpec] = []
    if family.name == "obstructed_patch":
        # Deliberately off the patch; a legitimate route exists around either end.
        for y in range(int(cy) - 2, int(cy) + 3):
            entities.append(EntitySpec("stone-wall", (cx + 5, float(y)), force="neutral"))
    return Blueprint(
        entities=tuple(entities),
        resources=tuple(
            ResourceSpec("iron-ore", (float(x), float(y)), amount=10000) for x, y in tiles
        ),
        character_position=start,
        character_inventory={"burner-mining-drill": 2, "stone-furnace": 2, "coal": 60},
        markers={"patch": (cx, cy)},
        unlock_recipes=("burner-mining-drill", "stone-furnace"),
        radius=48,
    )


def solve(driver) -> None:
    """Evaluator-only solvability witness; it never runs on evaluated episodes."""
    # Reuse the measured placement geometry from the existing construction
    # reference rather than inventing a second, subtly different one.
    from factoriorl.tasks.reference import _build_at

    for x, y in sorted(driver.resource_tiles()):
        if _build_at(driver, (x + 1, y + 1)):
            outcome = driver.env.run_verification()
            driver.success = bool(outcome["success"])
            return
        if driver.terminated or driver.truncated:
            return
    driver.trace.stuck_reason = "no resource tile accepted the reference build"


register(RegisteredTask(spec=SPEC, generate=generate, solve=solve))
