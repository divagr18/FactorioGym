"""A stopped plate line, with no marker saying why (R4.3, diagnosis track).

R4.3 asks for three declared tracks and names this one: *find the fault before
fixing it*. The two repair families are the same scene with the fault's tile
handed over -- `repair_belt` and `restore_power` both publish `gap` and `gap2`
through `extra_public_markers`, which `describe_assistance` now declares as
`fault-location:published`. With the coordinate given, "repair a broken line"
is "walk to a published tile and place one item".

This family withholds it. The line is built and aligned, one or both machines
are dry, and **nothing in the goal geometry says which**:

* the success predicate is `SUSTAINED_OUTPUT`, whose `marker` is `None`;
* `extra_public_markers` is empty;
* `focus_policy` is `static`.

So `TaskSpec.public_markers` is empty, `world.public_markers()` returns nil, and
the goal geometry slots stay zero. The fault is in truth, where the evaluator
reads it, and `validate_all` refuses this track if a marker leaks into
publication or if a shaping component points at a withheld one -- because
`rewards._measure`'s `CHARACTER_WITHIN` branch reads *truth*, so a reward
pointing at an unpublished fault would hand back through the reward exactly
what the observation withheld.

What this does and does not test
--------------------------------
The fault is *diagnosable from legitimately observable state*, which is what
R2.3 asks for and not the same thing as published. `sensor.entity_record`
carries each machine's status name, `working` flag and fuel contents, so an
agent that looks at the two machines can read which one is dry -- a drill with
no coal reports `no_fuel`. What it cannot do is be told the answer as a
coordinate before it looks.

Stated plainly, because it bounds the claim: the line is inside the 32-tile
sensor radius from every start in this family, so this is not a blind *search*
for the machines. It is a walk to a line the agent can see, and then a choice
of which fixes to apply -- read the two machines, decide which is starved, and
clear the output as well.

That is the reason the fault *kinds* are the split axis rather than
distance. `restore_power` learned the other lesson the hard way:
`pole_gap_far` was a byte-identical copy of `pole_gap`, so a name implying
distance held out nothing.

Every scene needs at least two fixes, and that is measured
--------------------------------------------------------
The first version of this family gave each scene a single fault. A solvability
run on a real engine refused it, and the numbers say why: on the training
families a **random** policy scored 0.40 against a 0.10 ceiling, and on the
test family random scored **0.60 against the reference's 0.40**. Both readings
have the same cause. The line is otherwise built, aligned and fuelled, so any
one lucky action restores it, and a 500-decision budget offers a great many
chances to be lucky. A benchmark whose floor is 0.40 measures the scene's own
provisioning.

So every family here also starts with the furnace's output slot full. A stone
furnace's result slot holds one stack of a hundred and the furnace stops when
it is full, so clearing it with `take_iron-plate_5` is necessary in every
scene, and it is a different *kind* of fix from inserting coal. Restoring the
line therefore takes at least two correct actions of two kinds, and a random
policy has to do both.

What the measured floors are, and which one to quote against
-----------------------------------------------------------
At the settled configuration -- 24..30 tiles, a blocked output in every scene,
200 decisions -- ten episodes per split give:

| split | reference | random, primitives | random, skills |
|---|---|---|---|
| train | 1.00 | 0.10 | 0.80 |
| test  | 1.00 | 0.00 | 0.80 |

The primitive floor is at or under the 0.10 ceiling `tools/solvability.py`
declares, and the family is reported `solvable`, not `defective`.

The **skills** floor is 0.80, and that is not a defect peculiar to this family:
that tool's own docstring records `mine_smelt` at 0.80, `supply_furnace` at
0.88 and `deliver` at 0.80 in the same action space. A random walk over
`approach_entity_k` is a far better agent than a random walk over
`move_north`, and for a task whose difficulty is *which* fix to apply beside
*which* machine, a macro that goes and stands beside a machine removes most of
it.

So this is a **primitive-space** benchmark, under the discipline the repo
already applies to those three families: an 80% skills-space result on it would
demonstrate nothing, because the floor is 0.80. Cutting the budget moved that
number from 1.00 to 0.80, which is worth having and is not enough to make it a
skills-space benchmark. Reported rather than tuned away -- a fourth iteration
chasing a floor is how a task ends up shaped around its own measurement.

The splits
----------
* **train** -- `drill_dry` and `furnace_dry`: the blocked output plus exactly
  one unfuelled machine. Each fuel fault appears on its own, so both are
  trained.
* **val** -- `jam_only`: both machines fuelled, output blocked. The scene that
  isolates the shared fix, for model selection.
* **test** -- `both_dry`: the blocked output plus *both* machines unfuelled.
  Three fixes, and the held-out structure is the **conjunction** -- each fuel
  fault is seen singly in training and never together. That is the
  one-fault-training / two-fault-test shape R4.3's gate lists as the desirable
  form, and it makes the holdout harder than training rather than merely
  differently named.

The test split being harder than its training split is deliberate and is the
point of a compositional holdout. It costs nothing in the difficulty-parity
check, which measures geometry: the line is pinned, so goal distance, path
length, detour factor and budget slack are identical across all four families.

The catalog is `plate_line`'s: movement, the two coal transfers, one plate
withdrawal, and `wait`. No placement action, because nothing here needs
building -- and `plate_line`'s docstring records at length why aligning a
furnace to a drill's drop tile is not reliably expressible in `primitive-v1`.
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

#: The line, pinned exactly as `plate_line` pins it and for the same reason: a
#: drill whose drop tile moved between scenes would change what the measured
#: window measures. A drill at (1, 1) facing south drops at (1.5, 2.3), and the
#: only furnace centre that catches that drop without overlapping the drill is
#: (1, 3).
DRILL_POSITION = (1.0, 1.0)
FURNACE_POSITION = (1.0, 3.0)
ORE_RADIUS = 3

#: Coal handed to the agent. Two full stacks and a margin, matching
#: `plate_line`: `give_coal_20` fuels the *nearest* entity and the two machines
#: are two tiles apart, so an agent that misjudges which one it is beside can
#: spend a stack on the wrong machine and still have enough to recover. Less
#: than this measures the scene's provisioning rather than the agent -- measured
#: on `plate_line`, where 47 coal went into the drill and the furnace never lit.
STARTING_COAL = 120

#: Coal already in the machine that is *not* the fault. Enough to run well past
#: the acceptance window, so the fault the scene declares is the only reason the
#: line is stopped.
FUELLED = 50

#: Plates in the jammed furnace's output. A stone furnace's result slot holds
#: one stack of a hundred and the furnace stops when it is full, so this is the
#: fault: not a missing input, a blocked output.
JAM_PLATES = 100

#: The window the line has to hold, and how long it may take to reach it.
#:
#: `SUSTAINED_OUTPUT` rather than `PRODUCED`, for the reason `build_line`
#: records: a cumulative counter cannot distinguish "the line is running" from
#: "someone hand-fed it once", and this family's whole point is that the line
#: runs afterwards without help. The floor is measured, not chosen -- a fuelled
#: drill and furnace make about 15 plates per 3,600 ticks -- and set below it so
#: one slow window does not fail a working line.
WINDOW_TICKS = 3600
WINDOW_PLATES = 10

#: How far the character starts from the line.
#:
#: Measured, after two solvability runs refused shorter distances. At 5..9
#: tiles -- `plate_line`'s annulus -- a **random** policy scored 0.40 on the
#: training families and 0.80 on the held-out one, against a 0.10 ceiling, and
#: making every scene need two fixes of two kinds barely moved it. The reason
#: is the catalog: `give_coal_N` and `take_iron-plate_5` both address the
#: *nearest* entity, so with two machines a few tiles away there is no target
#: to get wrong, and a policy sampling sixteen actions for hundreds of
#: decisions presses the right one beside the right machine soon enough. No
#: tightening of the budget fixes that -- the fix is cheap and the sampling is
#: dense.
#:
#: Distance is the one lever this catalog leaves, and the regime is not
#: guessed: `navigate`'s goals sit about 25 tiles out (measured path length
#: 24.7) and its random floor is 0.00. So the start moves into that regime.
#: Still inside the 32-tile sensor radius, so the machines and their status are
#: visible from the first observation and the task stays a diagnosis rather
#: than becoming a search.
#:
#: The cost is honest and small: a walk of about seven long strides, against a
#: budget dominated by smelting.
START_MIN_DISTANCE = 24.0
START_MAX_DISTANCE = 30.0

#: No acceptance before this tick. Without it the window is satisfiable by the
#: scene's own head start rather than by anything the agent did: the jammed
#: furnace already holds a hundred plates, and `PRODUCED` counts what the force
#: has smelted, so an unrepaired scene must not be able to pass on arithmetic
#: from before the repair.
SETTLE_TICKS = 3600

FAMILIES = (
    LayoutFamily("drill_dry", "train"),
    LayoutFamily("furnace_dry", "train"),
    LayoutFamily("jam_only", "val"),
    LayoutFamily("both_dry", "test"),
)

#: Which machines each family starves. The blocked output is in every family
#: and so is not listed here. Declared rather than inferred, so the generator
#: cannot disagree with the spec about what is broken.
STARVED: dict[str, tuple[str, ...]] = {
    "drill_dry": ("drill",),
    "furnace_dry": ("furnace",),
    "jam_only": (),
    "both_dry": ("drill", "furnace"),
}

SPEC = TaskSpec(
    id="diagnose_line",
    # Find the fault, then fix it. The fault's location is withheld, which is
    # the whole difference from the two `repair` families.
    track="diagnosis",
    version="1.0.0",
    description=(
        "This plate line has stopped. Find out why and get it producing iron plates "
        "again, without being told where the problem is."
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
    rewards=(
        RewardComponent("line_running", RewardKind.SPARSE_SUCCESS, weight=1.0, shaping=False),
        # Cumulative production, not the window's level: a component that can
        # *fall* is not a high-water mark, and paying for the window's level
        # would pay repeatedly for standing still. Graded to the goal so the
        # cap stays unreachable while the task is unfinished, which is what
        # `tools/reward_audit.py` requires.
        #
        # There is deliberately **no** shaping component pointing at the fault.
        # `rewards._measure`'s CHARACTER_WITHIN branch reads truth, so a
        # `toward_fault` potential would restore the hint the observation
        # withholds -- and would do it invisibly, through the reward rather
        # than the prompt. `validate_all` refuses this track if one is added.
        RewardComponent(
            "plates_produced",
            RewardKind.HIGH_WATER,
            weight=0.15 / WINDOW_PLATES,
            cap=0.15,
            predicate=Predicate(PredicateKind.PRODUCED, item="iron-plate"),
        ),
        RewardComponent("step_cost", RewardKind.STEP_COST, weight=0.001),
    ),
    # 200 decisions is 6,000 ticks, and that number is the third thing this
    # family had to have measured for it.
    #
    # The window is *trailing*, so with a long budget acceptance can be reached
    # from a fix made at almost any point in the episode -- and a policy with
    # macro-skills that eventually stumbles into the right one passes. Measured:
    # at 500 decisions a skill-equipped random policy solved this **1.00** of
    # the time, while `plate_line`'s floor in the same action space is 0.00.
    # The difference is not the scene, it is the deadline: `plate_line` asks for
    # 30 cumulative plates, which only a line running for most of the episode
    # produces.
    #
    # So the budget is cut to just past the earliest possible acceptance. The
    # reference reaches it at decision 120 -- about 23 decisions of walking and
    # fixing, then the window closing at tick 3,600 -- which leaves 80
    # decisions of slack for a slower route and none for a policy that has to
    # find the fix by trial. Still short of the 24,000 ticks that fill a
    # furnace's output slot from empty, so a repaired line cannot re-jam for a
    # reason the agent did not cause.
    max_decision_steps=200,
    max_game_ticks=6000,
    # Measurement metadata, absent from `to_dict()`: the machines are pinned, so
    # the only thing a route can be measured to is the line itself.
    difficulty_marker="line",
    # What the evaluator treats as the fault. Never published -- that is the
    # track -- and `validate_all` checks it stays that way.
    fault_markers=("drill", "furnace"),
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
        # The only fix for `furnace_jammed`, and useless on every training
        # scene. That asymmetry is the held-out structure.
        "take_iron-plate_5",
        "wait",
    ),
)


def generate(family: LayoutFamily, rng) -> Blueprint:
    """One stopped line. The fault kind is the family; the start varies.

    Nothing about the fault reaches `markers`, so nothing about it reaches the
    observation. `drill` and `furnace` are entity aliases -- `Blueprint`
    markers at runtime, because `world.build_blueprint` registers an entity's
    `marker` as an alias -- which is how the evaluator resolves them without
    publishing them.
    """
    starved = STARVED[family.name]
    angle = rng.uniform(0, 2 * math.pi)
    distance = rng.uniform(START_MIN_DISTANCE, START_MAX_DISTANCE)
    start = (
        round(DRILL_POSITION[0] + math.cos(angle) * distance, 1),
        round(DRILL_POSITION[1] + math.sin(angle) * distance, 1),
    )

    resources = tuple(
        ResourceSpec("iron-ore", (DRILL_POSITION[0] + dx, DRILL_POSITION[1] + dy), amount=5000)
        for dx in range(-ORE_RADIUS, ORE_RADIUS + 1)
        for dy in range(-ORE_RADIUS, ORE_RADIUS + 1)
    )

    drill_contents = {} if "drill" in starved else {"coal": FUELLED}
    # The blocked output is in every family, so `take_iron-plate_5` is
    # necessary everywhere and the fuel fault is what the splits vary.
    furnace_contents: dict[str, int] = {"iron-plate": JAM_PLATES}
    if "furnace" not in starved:
        furnace_contents["coal"] = FUELLED

    entities = (
        EntitySpec(
            "burner-mining-drill",
            DRILL_POSITION,
            direction="south",
            contents=drill_contents,
            marker="drill",
        ),
        EntitySpec(
            "stone-furnace",
            FURNACE_POSITION,
            contents=furnace_contents,
            marker="furnace",
        ),
    )

    return Blueprint(
        entities=entities,
        resources=resources,
        character_position=start,
        character_inventory={"coal": STARTING_COAL},
        # `line` only, and it is not a fault marker: it is the pinned geometry
        # the split audit measures routes against. The success predicate names
        # no marker and `extra_public_markers` is empty, so
        # `TaskSpec.public_markers` is empty and this never reaches an
        # observation.
        markers={"line": DRILL_POSITION},
        radius=48,
    )


register(RegisteredTask(spec=SPEC, generate=generate, solve=None))
