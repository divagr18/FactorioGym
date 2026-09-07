"""Retrieve and deliver items (PLAN.md 3.2, family 2).

Success is measured on the destination container, by marker: not on the
character's inventory, and not on "a container somewhere holds the goods". That
distinction is the whole task, and the scene is built to make a policy prove it
-- every family puts four containers in reach, only one of which is the
destination.
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

TARGET_ITEM = "iron-plate"

#: Twenty plates, which is exactly one `take_iron-plate_20` followed by one
#: `give_iron-plate_20`.
#:
#: This is a *ceiling*, not a chosen difficulty, and the distinction matters for
#: reading everything below. `reference.py::solve_deliver` takes twenty once --
#: its loop breaks as soon as the inventory holds ten -- and gives twenty once,
#: so twenty is the largest count the scripted solution can deliver. The
#: sibling correction that fixed `mine_smelt` (3 -> 12 plates) and
#: `supply_furnace` (5 -> 20 plates) worked because those solvers loop over
#: whole carry cycles; this one does not, so raising the count past twenty would
#: not make the task harder, it would make it unsolvable through the catalog and
#: the solvability check would say so.
#:
#: Twenty rather than ten still buys something, because `transfer` in
#: actions.lua is all-or-nothing: a `give_iron-plate_20` while holding less than
#: twenty is rejected outright rather than moving what it can. At a target of
#: ten, two stray `give_iron-plate_5` calls finished the task; at twenty that
#: route needs four consecutive correct ones and effectively disappears, leaving
#: the single take/give pair as the only realistic path. It is a small gain, and
#: it is the reason the random floor is attacked structurally below instead.
TARGET_COUNT = 20

#: The source is stocked with exactly two loads. A policy that gives the first
#: load to the wrong container has one more chance and no margin after that,
#: which is the cost that makes the destination's identity worth learning
#: rather than worth guessing. It is a cost and not a dead end: the plates are
#: still sitting in whichever chest received them and can be taken back out.
SOURCE_STOCK = TARGET_COUNT * 2

#: Every scene carries four containers, one of which is the destination.
#:
#: This is the lever that actually moves the random floor, and four is not a
#: round number: `skills.ADDRESSABLE_ENTITIES` is 4, so a policy over the skill
#: catalog can name the nearest four entities and no more. Four containers is
#: therefore the largest field in which the destination is *guaranteed* to be
#: addressable from wherever the character stands. A fifth would reproduce the
#: defect recorded in `skills.py`: on the walled holdout the destination was
#: pushed past the addressable range, "the task was unrepresentable rather than
#: unlearned", and the arm scored 0/25. Making a benchmark harder by making it
#: unrepresentable is not making it harder.
CONTAINERS_PER_SCENE = 4

#: What a distractor holds when it is not empty. Anything but the target item:
#: a `take_iron-plate_*` against it is rejected for `no_items`, so a wrong
#: approach cannot accidentally load the shipment.
DISTRACTOR_ITEM = "coal"
DISTRACTOR_STOCK = 30

FAMILIES = (
    # The names are historical and are load-bearing elsewhere -- `screened_depot`
    # is pinned by docs/evidence/holdout_v1.json and the earlier gate and
    # ablation reports name the training families -- so they are kept even
    # though every family now carries decoy containers. What each family means
    # is documented at its branch in `generate`.
    LayoutFamily("two_chest", "train"),
    LayoutFamily("decoy_chest", "train"),
    LayoutFamily("stacked_depot", "val"),
    # Structural holdout at matched distance: the destination sits behind a
    # wall segment, so the route differs while the distance does not.
    LayoutFamily("screened_depot", "test"),
)

SPEC = TaskSpec(
    id="deliver",
    # 1.1.0: the task was not a benchmark against the skill action space. A
    # uniform-random policy over primitives scored 0.12, but over primitives
    # plus the six skills it scored 0.72-0.80 against an 0.80 acceptance bar,
    # and a trained policy's 0.44 on the same split was therefore *below* its
    # own floor while reading as partial success. Three changes answer that:
    # the delivered count rose 10 -> 20, every scene now holds four containers
    # instead of two, and the decision budget fell 250 -> 120. The version is
    # part of the random-baseline cache key, so bumping it is what stops a
    # cached 0.80 floor -- or any published score measured against it -- from
    # being served for the harder task under the same name.
    version="1.1.0",
    description="Carry items from a source container to a destination container.",
    layout_families=FAMILIES,
    success=(
        Predicate(
            PredicateKind.CONTAINER_HOLDS,
            marker="dst",
            item=TARGET_ITEM,
            at_least=TARGET_COUNT,
        ),
    ),
    rewards=(
        RewardComponent("delivered", RewardKind.SPARSE_SUCCESS, weight=1.0, shaping=False),
        # High-water, so taking items back out of the destination and putting
        # them in again cannot pay twice.
        RewardComponent(
            "at_destination",
            RewardKind.HIGH_WATER,
            weight=0.05,
            cap=0.2,
            predicate=Predicate(PredicateKind.CONTAINER_HOLDS, marker="dst", item=TARGET_ITEM),
        ),
        RewardComponent(
            "carried",
            RewardKind.HIGH_WATER,
            weight=0.02,
            cap=0.15,
            predicate=Predicate(PredicateKind.INVENTORY_HOLDS, item=TARGET_ITEM),
        ),
        RewardComponent("step_cost", RewardKind.STEP_COST, weight=0.001),
    ),
    # 250 decisions was fifteen times what the scripted solution spends:
    # docs/evidence/phase3-solvability.json records the reference solver
    # finishing in 15.8 decisions on the training split and 16.8 on the walled
    # holdout. That surplus is what a uniform-random policy converts into
    # attempts -- it wanders between containers issuing transfers until one pair
    # of them happens to be the right pair at the right chests, and the number of
    # such attempts is very nearly proportional to this budget. Halving it
    # roughly halves the floor.
    #
    # 120 still leaves seven times the reference cost, which is more headroom
    # than either sibling family allows (`mine_smelt` gives 400 against 141
    # measured decisions, `supply_furnace` 300 against 140), and comfortably
    # more than a competent skill policy needs: two approaches at roughly ten
    # primitive decisions each plus two transfers is about twenty-five, leaving
    # room for several wrong approaches on the way.
    max_decision_steps=120,
    max_game_ticks=15000,
    landmarks=(
        # The episode's pivot: everything before this is fetching, everything
        # after is delivering, and the success predicate cannot tell them apart.
        Predicate(PredicateKind.INVENTORY_HOLDS, item=TARGET_ITEM, at_least=1),
        Predicate(PredicateKind.INVENTORY_HOLDS, item=TARGET_ITEM, at_least=TARGET_COUNT),
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
        "take_iron-plate_5",
        "take_iron-plate_20",
        "give_iron-plate_5",
        "give_iron-plate_20",
        "wait",
    ),
)


#: The band the decoy containers are drawn from. It deliberately straddles the
#: destination's own band (14 to 24), so the destination is sometimes the
#: farthest container in the scene and sometimes not.
#:
#: A band that sat entirely inside the destination's would have handed every
#: family the same shortcut -- "approach the farthest chest" -- which a policy
#: can learn from rank alone without ever learning what a destination is, and
#: which would then transfer to the holdout and be read as generalisation. The
#: point of adding containers is to make the rank uninformative, and a rank that
#: is uninformative only on average is not uninformative.
DECOY_BAND = (11.0, 22.0)


def generate(family: LayoutFamily, rng) -> Blueprint:
    src_angle = rng.uniform(0, 2 * math.pi)
    src_distance = rng.uniform(4.0, 9.0)
    src = (
        round(math.cos(src_angle) * src_distance, 1),
        round(math.sin(src_angle) * src_distance, 1),
    )

    # One distance range for every family. A holdout that is simply farther
    # away measures added difficulty, not transfer; `screened_depot` varies the
    # route instead, at matched distance.
    dst_distance = rng.uniform(14.0, 24.0)
    dst_bearing_offset = rng.uniform(math.pi * 0.5, math.pi * 1.5)
    dst_angle = src_angle + dst_bearing_offset
    dst = (
        round(math.cos(dst_angle) * dst_distance, 1),
        round(math.sin(dst_angle) * dst_distance, 1),
    )

    entities = [
        EntitySpec("wooden-chest", src, contents={TARGET_ITEM: SOURCE_STOCK}, marker="src"),
        EntitySpec("wooden-chest", dst, marker="dst"),
    ]

    # The decoys sit on the two bearings that bisect the arcs between the source
    # and the destination. Bisecting is not decoration: it puts every decoy at
    # least 45 degrees off both the source and the destination bearing whatever
    # angle the scene drew, which over the declared radius bands keeps every
    # decoy at least 7.9 tiles from the source and 9.9 from the destination.
    #
    # That clearance is a solvability requirement, not tidiness. `take` and
    # `give` bind `$target` to the *nearest* entity, and the reference solver
    # stops within 1.4 tiles of a chest -- or, if an obstacle blocks it, may
    # settle up to 6 tiles away and transfer from there. A decoy closer than
    # that would silently steal the shipment from the solver and turn a
    # difficulty change into an unsolvable family.
    bisector = src_angle + dst_bearing_offset / 2
    decoy_bearings = (bisector, bisector + math.pi)
    # `stacked_depot` spends one of its four container slots on the neighbour
    # beside the destination, so it takes one decoy rather than two. Every
    # family ends with exactly CONTAINERS_PER_SCENE containers, which is what
    # keeps the addressable-rank field -- and so the difficulty of choosing
    # within it -- matched across the splits.
    decoy_slots = CONTAINERS_PER_SCENE - 3 if family.name == "stacked_depot" else 2
    # Empty decoys are the harder case and the default: an empty wooden chest is
    # indistinguishable from the destination by contents, so the policy has
    # nothing to go on but the marker the task is scored against.
    decoy_contents: dict[str, int] = {}
    if family.name == "decoy_chest":
        # The family that gives the original `decoy_chest` its name, kept as a
        # named variation rather than promoted to the norm: here the wrong
        # chests are *stocked* with the wrong item, so success must depend on
        # what is delivered and not merely on visiting a container. Stocking
        # them also blocks a wrong `take`, which the empty case does not.
        decoy_contents = {DISTRACTOR_ITEM: DISTRACTOR_STOCK}

    decoys: list[tuple[float, float]] = []
    for index, bearing in enumerate(decoy_bearings):
        # Drawn before the slot check so every family consumes the same RNG
        # values in the same order; a family that consumed fewer would shift
        # every later draw and make the split audit's shared-seed duplicate
        # probe compare scenes that were never meant to line up.
        radius = rng.uniform(*DECOY_BAND)
        if index >= decoy_slots:
            continue
        position = (round(math.cos(bearing) * radius, 1), round(math.sin(bearing) * radius, 1))
        decoys.append(position)
        # Markered so evaluator truth reports what ended up inside them. The
        # marker is deliberately absent from `Blueprint.markers`, which keeps
        # these counted as distractors by tools/generator_diagnostics.py and
        # keeps `dst` the marker difficulty is measured to.
        entities.append(
            EntitySpec(
                "wooden-chest", position, contents=dict(decoy_contents), marker=f"decoy_{index}"
            )
        )

    if family.name == "stacked_depot":
        entities.append(EntitySpec("wooden-chest", (dst[0] + 2, dst[1]), marker="dst_neighbour"))
    elif family.name == "screened_depot":
        # A wall segment across the direct line to the depot: same distance,
        # different route. Laid perpendicular to the bearing so it screens the
        # approach rather than merely standing near it.
        normal = dst_angle + math.pi * 0.5
        centre = (
            math.cos(dst_angle) * dst_distance * 0.6,
            math.sin(dst_angle) * dst_distance * 0.6,
        )
        taken = {(round(dst[0]), round(dst[1])), (round(src[0]), round(src[1])), (0, 0)}
        # The decoys are 45 degrees or more off the destination bearing, so the
        # screen cannot geometrically reach them; they are listed anyway because
        # a wall sharing a tile with a chest is a generator bug that
        # `Blueprint.footprint_conflicts` would report as a task defect, and the
        # cost of ruling it out here is one set insertion.
        taken.update((round(p[0]), round(p[1])) for p in decoys)
        for offset in (-2.0, -1.0, 0.0, 1.0, 2.0):
            tile = (
                round(centre[0] + math.cos(normal) * offset),
                round(centre[1] + math.sin(normal) * offset),
            )
            if tile in taken:
                continue
            taken.add(tile)
            entities.append(EntitySpec("stone-wall", (float(tile[0]), float(tile[1]))))

    return Blueprint(
        entities=tuple(entities),
        character_position=(0.0, 0.0),
        markers={"src": src, "dst": dst},
        radius=64,
    )


register(RegisteredTask(spec=SPEC, generate=generate))
