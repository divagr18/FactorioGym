"""Measure declared train/val/test splits as content (PLAN.md section 3, Generalization).

PLAN.md says a declared split "is a claim about content, so [it] must be measured
as content rather than asserted by name", and asks for three published numbers per
task family:

* the count of distinct generated scenes per layout family, with a declared
  minimum -- "a holdout whose generator admits one scene is evaluated once, no
  matter how many episodes are run against it";
* difficulty parity between the training and test splits on measured
  descriptors, so a holdout that is merely harder is not reported as transfer;
* structural separation on at least one descriptor that is *not* difficulty, so
  the split is doing the work its name claims.

Nothing computed any of that. The Phase 3 gate checks only that the held-out
family's *name* differs from the training families' names, which is satisfied by
a generator that ignores the family entirely. Three real defects passed that
check and were found by hand instead:

* ``repair_belt.gap_far`` was byte-identical to ``repair_belt.gap``;
* ``restore_power.pole_gap_far`` was byte-identical to ``restore_power.pole_gap``;
* ``supply_furnace.far_ore`` had no generator branch, so the validation split
  silently scored the training layout.

Difficulty is measured as a route, so it needs a position to route *to*. A task
states that position in ``TaskSpec.difficulty_marker``; where it does not, the
marker is taken from a positional success or failure predicate. A task offering
neither is reported ``INCOMPLETE`` rather than ``PASS`` -- see the verdict
comment at the end of ``analyse_task_object`` for why an unmeasured claim must
never wear a green verdict.

This tool is engine-free: blueprints are pure Python, so a split can be audited
without a Factorio worker, and the audit is cheap enough to run on every change
to a generator.

Run: uv run python tools/generator_diagnostics.py [--samples N] [--tasks a,b,c]

Thresholds
----------
``MIN_DISTINCT_SCENES``, ``PARITY_MIN_OVERLAP`` and ``SEPARATION_MAX_OVERLAP``
are **starting points, not principled values**. They are calibrated against the
sampling noise of this tool (see ``PARITY_MIN_OVERLAP``) and against how the
repo actually evaluates, but none of them is derived from a power calculation
and none should be quoted as if it were. They are written down here so that
changing them is a visible decision rather than a silent one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from factoriorl.seeding import Branch, SeedPlan  # noqa: E402
from factoriorl.tasks import all_tasks, get  # noqa: E402
from factoriorl.tasks.spec import Blueprint  # noqa: E402

# --------------------------------------------------------------- thresholds

#: A layout family must admit at least this many distinct scenes.
#:
#: STARTING POINT, not a principled value. The rationale is only that below
#: roughly this many scenes a "holdout" stops being a distribution and becomes a
#: fixed test case: PLAN 3's acceptance target evaluates 100 episodes per
#: training seed on the held-out split, and a generator admitting one scene
#: reports that one scene 100 times while presenting a Wilson interval computed
#: as if there were 100 independent draws. 32 is a round number below any
#: evaluation batch this repo runs; it is not the output of a power calculation
#: and must not be quoted as one.
MIN_DISTINCT_SCENES = 32

#: Train and test must overlap at least this much on every difficulty
#: descriptor, or the holdout is measuring added difficulty rather than transfer.
#:
#: STARTING POINT, but calibrated rather than guessed. The overlapping
#: coefficient of two finite samples drawn from the *same* distribution is below
#: 1.0 purely from binning noise: with n samples spread over b bins the expected
#: shortfall is roughly (1/2) * sum_i E|p_i - q_i|, which at n=200 and b=12
#: costs about 0.12. So identical generators score near 0.88, not 1.00, and a
#: threshold of 0.60 leaves comfortable headroom above that noise floor while
#: still rejecting a holdout whose range is shifted (a uniform range moved by
#: half its width scores near 0.5, and a disjoint range scores 0.0).
PARITY_MIN_OVERLAP = 0.60

#: At least one *structural* descriptor must overlap no more than this, or the
#: split is not structurally separated and its name is doing work the content
#: does not.
#:
#: STARTING POINT. Chosen as the mirror of the parity threshold: 0.40 is far
#: enough below the ~0.88 same-distribution noise floor that a descriptor
#: clearing it cannot be explained by sampling, while not demanding the total
#: disjointness (0.0) that would rule out holdouts which vary a structure
#: continuously rather than categorically.
SEPARATION_MAX_OVERLAP = 0.40

#: Histogram bins for the overlapping coefficient. Fewer bins means a higher
#: same-distribution noise floor is avoided but real differences are blurred;
#: 12 puts the noise floor near 0.88 at the default sample count.
BINS = 12

#: Seeds used for the duplicate-family probe. Deliberately small: exact digest
#: equality over a few dozen shared seeds is already conclusive, and a family
#: pair that agrees on 48 independent seeds and disagrees later is not a case
#: this repo has ever seen.
PROBE_SAMPLES = 48

#: Fixed base for the probe seeds so the probe is reproducible across runs and
#: across processes. It must not depend on the family name, because the whole
#: point is to hand every family the *same* RNG state.
PROBE_SEED_BASE = 90_210

DIFFICULTY_DESCRIPTORS = ("goal_distance", "path_length", "detour_factor", "budget_slack")
STRUCTURAL_DESCRIPTORS = (
    "obstacle_count",
    "occupancy_fraction",
    "entity_type_count",
    "distractor_count",
    # How far the goal sits from the nearest other declared entity.
    #
    # The descriptors around it measure scene *geometry*, which is the wrong
    # instrument for a repair task: `restore_power`'s holdout differs from its
    # training families only in where the fault sits in the chain, and moving a
    # missing pole from the middle to the end changes no count, no occupancy
    # and no footprint. The tool therefore reported "no structural descriptor
    # separates train from test" for a holdout whose whole point is a different
    # fault position, which PLAN.md section 3 names as a holdout axis in its own
    # right. Isolating the goal is exactly what a fault adjacent to it does.
    "goal_isolation",
    # The four descriptors above see entities only, and PLAN.md section 3 lists
    # "resource arrangements" as a structural holdout axis in its own right.
    # `mine_smelt.screened_patch` varies nothing but the ore layout -- and ore is
    # walkable, so it changes neither the obstacle count nor the occupancy -- so
    # without this descriptor the tool reported that holdout as structurally
    # identical to training when it is not.
    "content_tile_count",
)

#: Difficulty descriptors whose *smaller* value is the harder scene. Slack is the
#: budget left over after the walk, so more of it is easier -- reading a rise in
#: slack as "the holdout is harder" would report the direction backwards, and the
#: direction is the whole point of PLAN.md's parity requirement.
HARDER_WHEN_SMALLER = frozenset({"budget_slack"})

# ------------------------------------------------------------------ geometry

#: Tile footprints for the entity prototypes these six families place. Anything
#: absent is treated as 1x1, which is correct for every chest, wall, belt,
#: inserter and small pole in use. The two multi-tile entities are named
#: explicitly because getting them wrong silently changes reachability: the
#: mine_smelt generator carries a comment about a 2x2 furnace landing on the
#: spawn tile, and the restore_power generator about a 3x3 drill's near edge
#: sitting on a pole's supply boundary. A 1x1 approximation would have made both
#: of those scenes look walkable when they were not.
ENTITY_TILE_SIZES: dict[str, tuple[int, int]] = {
    "stone-furnace": (2, 2),
    "electric-mining-drill": (3, 3),
    "solar-panel": (3, 3),
}

#: Prototypes the character can walk over. Transport belts do not block the
#: player in Factorio -- they carry it. Treating a belt line as a wall would
#: report a fictitious detour around every repair_belt scene and would make the
#: sink chest look unreachable, which is a measurement defect rather than a
#: task defect.
WALKABLE_ENTITIES = frozenset({"transport-belt"})

#: A 30-tick `move_*` covers 0.1484 tiles/tick * 30 = 4.45 tiles (catalog.py
#: LONG_MOVE_TICKS carries the same arithmetic). Budget slack is expressed in
#: decisions, not tiles, because the episode budget is counted in decisions: a
#: path of L tiles costs at best ceil(L / 4.45) decisions of pure locomotion.
LONG_MOVE_TILES = 4.45

#: Padding around the scene's content bounding box, in tiles. The BFS needs room
#: to route around an obstacle that sits on the edge of the content; with zero
#: padding a wall touching the bounding box would look like a sealed border and
#: report an unreachable goal that the engine walks around trivially.
BOX_PADDING = 8


def entity_tiles(name: str, position: tuple[float, float]) -> list[tuple[int, int]]:
    """The integer tiles an entity occupies, from its prototype footprint."""
    width, height = ENTITY_TILE_SIZES.get(name, (1, 1))
    # floor(p - size/2 + 0.5) rather than round(): Python's round() is
    # banker's rounding, which sends a footprint whose left edge lands on .5 to
    # the wrong tile half the time. The same trap is called out in env.py's
    # placement-offset comment.
    x0 = math.floor(position[0] - width / 2 + 0.5)
    y0 = math.floor(position[1] - height / 2 + 0.5)
    return [(x0 + dx, y0 + dy) for dx in range(width) for dy in range(height)]


def blocked_tiles(blueprint: Blueprint) -> set[tuple[int, int]]:
    """Tiles the character cannot stand on. Resources are walkable; ore is not a wall."""
    blocked: set[tuple[int, int]] = set()
    for entity in blueprint.entities:
        if entity.name in WALKABLE_ENTITIES:
            continue
        blocked.update(entity_tiles(entity.name, entity.position))
    return blocked


def scene_box(blueprint: Blueprint) -> tuple[int, int, int, int]:
    """The padded bounding box of everything the scene declares, clamped to the radius."""
    xs = [blueprint.character_position[0]]
    ys = [blueprint.character_position[1]]
    for entity in blueprint.entities:
        for tile in entity_tiles(entity.name, entity.position):
            xs.append(tile[0])
            ys.append(tile[1])
    for resource in blueprint.resources:
        xs.append(resource.position[0])
        ys.append(resource.position[1])
    for position in blueprint.markers.values():
        xs.append(position[0])
        ys.append(position[1])
    radius = blueprint.radius
    return (
        max(-radius, math.floor(min(xs)) - BOX_PADDING),
        max(-radius, math.floor(min(ys)) - BOX_PADDING),
        min(radius, math.ceil(max(xs)) + BOX_PADDING),
        min(radius, math.ceil(max(ys)) + BOX_PADDING),
    )


def _bfs(
    blocked: set[tuple[int, int]],
    box: tuple[int, int, int, int],
    start: tuple[int, int],
    targets: set[tuple[int, int]],
) -> int | None:
    """Shortest 4-connected walk from ``start`` to any target tile, in tiles.

    4-connected, not 8, because every family's declared ``catalog_subset``
    contains only cardinal moves (`move_north` .. `nudge_west`). An 8-connected
    BFS would report a diagonal shortcut the policy has no action for, which
    would understate path length by up to 40%.
    """
    min_x, min_y, max_x, max_y = box
    if not (min_x <= start[0] <= max_x and min_y <= start[1] <= max_y):
        return None
    if start in blocked:
        return None
    if start in targets:
        return 0
    seen = {start}
    queue: deque[tuple[tuple[int, int], int]] = deque([(start, 0)])
    while queue:
        (x, y), distance = queue.popleft()
        for cell in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if cell in seen or cell in blocked:
                continue
            if not (min_x <= cell[0] <= max_x and min_y <= cell[1] <= max_y):
                continue
            if cell in targets:
                return distance + 1
            seen.add(cell)
            queue.append((cell, distance + 1))
    return None


def goal_marker(spec) -> str | None:
    """The marker difficulty is measured to, or None if the task declares none.

    Two sources, in order, and never a guess:

    1. ``spec.difficulty_marker``, the task's explicit statement of where the
       route it is scored on ends. A task whose success is a production count
       has a perfectly well-defined route -- spawn to the ore patch and back to
       the furnace -- that no predicate mentions, and this is how it says so.
    2. Otherwise the marker a success (or failure) predicate names. For
       `navigate` and `deliver` the predicate names a position and that position
       *is* the goal, so the fallback is correct rather than convenient.

    If neither exists the answer is None, and the caller reports the parity as
    unmeasurable. It must stay None rather than falling back to "whichever marker
    the blueprint happens to declare first": `mine_smelt` declares `furnace` at a
    fixed radius from spawn, so that guess would report the same distance for
    every scene in every family and produce a parity of 1.00 that checked
    nothing -- a green number standing in for an unmeasured claim, which is the
    single defect this tool exists to remove.
    """
    if getattr(spec, "difficulty_marker", None):
        return spec.difficulty_marker
    for predicate in (*spec.success, *spec.failure):
        if predicate.marker:
            return predicate.marker
    return None


def goal_marker_source(spec) -> str | None:
    """Whether the measured marker was declared or inferred from a predicate.

    Published alongside the marker so a reader of the report can tell a route the
    task author chose from one the tool derived, without re-deriving it. The two
    carry different warranties: a declared marker is a reviewable claim, an
    inferred one is only as good as the predicate it came from.
    """
    if getattr(spec, "difficulty_marker", None):
        return "declared"
    for predicate in (*spec.success, *spec.failure):
        if predicate.marker:
            return "success/failure predicate"
    return None


def goal_target_tiles(
    blueprint: Blueprint, marker: str, blocked: set[tuple[int, int]]
) -> set[tuple[int, int]]:
    """Free tiles from which the goal is reached: the goal footprint and its neighbours.

    The goal is usually an entity the character cannot stand inside -- a chest,
    or a 3x3 mining drill whose centre tile is the declared marker. Targeting the
    marker tile alone would report every such goal as unreachable.
    """
    position = blueprint.markers[marker]
    footprint = {(math.floor(position[0]), math.floor(position[1]))}
    for entity in blueprint.entities:
        if entity.marker == marker:
            footprint = set(entity_tiles(entity.name, entity.position))
            break
    targets: set[tuple[int, int]] = set()
    for x, y in footprint:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                cell = (x + dx, y + dy)
                if cell not in blocked:
                    targets.add(cell)
    return targets


# ------------------------------------------------------------ scene identity


def scene_digest(
    blueprint: Blueprint,
    public_markers: tuple[str, ...] = (),
    extra_tracked_items: tuple[str, ...] = (),
) -> str:
    """A canonical, process-stable digest of a blueprint's full content.

    Same algorithm as ``factoriorl.env.blueprint_digest`` -- canonical JSON,
    sha256, first 16 hex characters -- so "distinct scenes" here means exactly
    "distinct scenes the worker would install", and a test pins the two together
    so they cannot drift apart. Recomputed here rather than imported because
    ``factoriorl.env`` pulls in gymnasium and numpy, and a split audit must stay
    runnable without the `rl` extra.

    ``public_markers`` is part of the installed payload, so it is part of the
    identity: publishing a task's objective changes what an agent can see, and a
    holdout frozen before that change must not silently pass afterwards. So is
    ``extra_tracked_items``, for the same reason -- it changes what the truth
    channel counts. Both are empty for every task frozen so far, so no existing
    digest moves; a caller that omits them where the task declares one would
    name a payload no run installs.

    Built-in ``hash()`` is randomised per process (PYTHONHASHSEED) and this repo
    has already been bitten by that: seeding blueprint sampling with it made the
    task validation suite pass or fail at random. Nothing here may use it.
    """
    canonical = json.dumps(
        blueprint.to_dict(public_markers=public_markers, extra_tracked_items=extra_tracked_items),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------- descriptors


def describe(task, blueprint: Blueprint) -> dict[str, float | None]:
    """Difficulty and structural descriptors for one generated scene.

    Difficulty descriptors are the ones that must *match* across splits;
    structural descriptors are the ones a holdout is allowed -- and required --
    to differ on. A descriptor that cannot be computed is None, never a
    substituted number: an invented distance would be indistinguishable from a
    measured one in the report, which is the exact failure mode this tool exists
    to remove.
    """
    spec = task.spec
    blocked = blocked_tiles(blueprint)
    box = scene_box(blueprint)
    min_x, min_y, max_x, max_y = box
    area = (max_x - min_x + 1) * (max_y - min_y + 1)
    inside = sum(1 for (x, y) in blocked if min_x <= x <= max_x and min_y <= y <= max_y)

    # A distractor is an entity the task declaration never points at: it carries
    # no marker listed in the blueprint and is named by no success or failure
    # predicate. Walls, decoy chests, spare furnaces and belt segments all count;
    # the goal and the source do not. This is the descriptor that catches a
    # holdout differing only in scenery.
    named = set(blueprint.markers) | {p.marker for p in (*spec.success, *spec.failure) if p.marker}
    distractors = sum(1 for e in blueprint.entities if e.marker not in named)

    # Tiles covered by anything the scene declares, walkable or not. This is the
    # only descriptor that can see a resource-arrangement holdout.
    content_tiles = set(blocked)
    for entity in blueprint.entities:
        content_tiles.update(entity_tiles(entity.name, entity.position))
    content_tiles.update(
        (math.floor(r.position[0]), math.floor(r.position[1])) for r in blueprint.resources
    )

    values: dict[str, float | None] = {
        "obstacle_count": float(
            sum(1 for e in blueprint.entities if e.name not in WALKABLE_ENTITIES)
        ),
        "occupancy_fraction": round(inside / area, 6) if area else None,
        "entity_type_count": float(len({e.name for e in blueprint.entities})),
        "distractor_count": float(distractors),
        "content_tile_count": float(len(content_tiles)),
        "goal_isolation": None,
        "goal_distance": None,
        "path_length": None,
        "detour_factor": None,
        "budget_slack": None,
    }

    marker = goal_marker(spec)
    if marker is None or marker not in blueprint.markers:
        return values

    goal = blueprint.markers[marker]
    straight = blueprint.distance_from_character(goal)
    values["goal_distance"] = round(straight, 4)

    # Distance from the goal to the nearest *other* declared entity. A fault
    # sitting next to the goal -- a missing pole at the end of a chain rather
    # than in its middle -- shows up here and nowhere else, because it changes
    # no count and no footprint.
    others = [math.dist(goal, e.position) for e in blueprint.entities if e.marker != marker]
    if others:
        values["goal_isolation"] = round(min(others), 4)

    start = (
        math.floor(blueprint.character_position[0]),
        math.floor(blueprint.character_position[1]),
    )
    tiles = _bfs(blocked, box, start, goal_target_tiles(blueprint, marker, blocked))
    if tiles is None:
        # Left as None rather than as a large sentinel. An unreachable goal is a
        # generator defect and is reported as one; folding it into the
        # distribution as "very far" would hide it inside a parity number.
        return values

    values["path_length"] = float(tiles)
    # Path-to-a-tile-beside-the-goal over straight-line-to-the-goal-centre, so an
    # unobstructed scene sits near 1.0 and can dip slightly below it: the walk
    # stops one tile short of a goal the character cannot stand inside. A value
    # under 1.0 is not a measurement error; a value well above it is a detour.
    values["detour_factor"] = round(tiles / straight, 4) if straight > 0 else None
    # Decisions, not tiles: the budget is counted in decisions and the longest
    # catalog stride covers 4.45 tiles, so dividing the budget by tiles would
    # understate the slack by that factor uniformly.
    move_decisions = max(1, math.ceil(tiles / LONG_MOVE_TILES))
    values["budget_slack"] = round(spec.max_decision_steps / move_decisions, 4)
    return values


# ------------------------------------------------------------------- overlap


def overlap_coefficient(
    left: list[float | None], right: list[float | None], bins: int = BINS
) -> float | None:
    """Overlapping coefficient of two samples: the shared area of their histograms.

    1.0 when the two distributions coincide, 0.0 when they are disjoint. Computed
    from a common binning over the pooled range so the two histograms are
    directly comparable; each is normalised to sum to 1 first, so unequal sample
    sizes do not tilt the result.

    Returns None when either side has no measurable values -- an unmeasured
    descriptor must not be reported as perfect agreement.
    """
    a = [float(v) for v in left if v is not None]
    b = [float(v) for v in right if v is not None]
    if not a or not b:
        return None
    low = min(min(a), min(b))
    high = max(max(a), max(b))
    if high - low <= 1e-12:
        # Every observed value is the same number on both sides. That is total
        # agreement, and it is also what a constant generator looks like -- the
        # distinct-scene count is what separates those two cases, not this one.
        return 1.0
    width = (high - low) / bins
    left_hist = [0] * bins
    right_hist = [0] * bins
    for value in a:
        left_hist[min(bins - 1, int((value - low) / width))] += 1
    for value in b:
        right_hist[min(bins - 1, int((value - low) / width))] += 1
    return round(sum(min(left_hist[i] / len(a), right_hist[i] / len(b)) for i in range(bins)), 4)


# -------------------------------------------------------------------- sampling


def sample_family(task, family, samples: int, plan: SeedPlan) -> dict:
    """Generate `samples` scenes for one layout family the way the environment does.

    ``FactorioEnv.prepare_scene`` builds ``seed_plan.generator_rng(branch, index)``,
    draws ``rng.randrange(len(families_in_split))`` to choose the family, and then
    hands that *same, already-advanced* RNG to ``task.generate``. Sampling with a
    fresh ``random.Random(index)`` instead would explore RNG states the
    environment never produces, so the family-choice draw is replayed here even
    though its result is discarded.
    """
    split_size = max(1, len(task.spec.families(family.split)))
    marker = goal_marker(task.spec)
    digests: list[str] = []
    records: list[dict[str, float | None]] = []
    unreachable = 0
    missing_marker = 0
    for index in range(samples):
        rng = plan.generator_rng(Branch.TRAIN, index)
        rng.randrange(split_size)
        blueprint = task.generate(family, rng)
        digests.append(scene_digest(blueprint))
        values = describe(task, blueprint)
        if marker:
            # A marker the task declares but the generator never places would
            # otherwise sink without trace: every difficulty descriptor comes
            # back None and the verdict is `incomplete` with a message saying no
            # marker was declared, which is the opposite of what happened. A
            # typo in `difficulty_marker` must read as a typo.
            missing_marker += int(marker not in blueprint.markers)
            if values["goal_distance"] is not None:
                unreachable += int(values["path_length"] is None)
        records.append(values)
    return {
        "split": family.split,
        "samples": samples,
        "digests": digests,
        "records": records,
        "unreachable_goals": unreachable,
        "missing_marker": missing_marker,
    }


def probe_digests(task, family, count: int = PROBE_SAMPLES) -> tuple[str, ...]:
    """Digests over seeds that are identical for every family.

    Duplicate detection cannot use the environment-faithful sampling above: the
    family-choice draw depends on how many families share a split, so a train
    family and a test family are handed different RNG states even when their
    generators are the same code path. Feeding every family the same seeds makes
    "these two families generate identical content" an exact tuple comparison --
    which is precisely the `gap_far == gap` and `pole_gap_far == pole_gap`
    defect that a name comparison could never see.
    """
    return tuple(
        scene_digest(task.generate(family, random.Random(PROBE_SEED_BASE + index)))
        for index in range(count)
    )


# -------------------------------------------------------------------- analysis


def _summarise(records: list[dict[str, float | None]], key: str) -> dict:
    values = [r[key] for r in records if r[key] is not None]
    if not values:
        return {"available": 0}
    return {
        "available": len(values),
        "min": round(min(values), 4),
        "mean": round(sum(values) / len(values), 4),
        "max": round(max(values), 4),
    }


def analyse_task(task_id: str, samples: int, plan: SeedPlan) -> dict:
    """Everything PLAN.md section 3 asks to be published about one registered task."""
    return analyse_task_object(get(task_id), samples, plan)


def analyse_task_object(task, samples: int, plan: SeedPlan) -> dict:
    """The analysis, taking the task object rather than its id.

    Split out so the regression tests can drive it with a synthetic task whose
    families are known duplicates. Depending on the six registered families would
    make those tests report a maintainer's in-progress generator edit as a
    failure of this tool.
    """
    spec = task.spec
    families: dict[str, dict] = {}
    probes: dict[str, tuple[str, ...]] = {}
    for family in spec.layout_families:
        sampled = sample_family(task, family, samples, plan)
        probes[family.name] = probe_digests(task, family)
        families[family.name] = sampled

    entry: dict = {
        "version": spec.version,
        "goal_marker": goal_marker(spec),
        "goal_marker_source": goal_marker_source(spec),
        "layout_families": {},
        "duplicate_families": [],
        "difficulty_overlap": {},
        "structural_overlap": {},
        "failures": [],
    }

    total_digests: set[str] = set()
    for name, sampled in families.items():
        distinct = len(set(sampled["digests"]))
        total_digests.update(sampled["digests"])
        entry["layout_families"][name] = {
            "split": sampled["split"],
            "samples": sampled["samples"],
            "distinct_scenes": distinct,
            "distinct_fraction": round(distinct / sampled["samples"], 4),
            "unreachable_goals": sampled["unreachable_goals"],
            "missing_marker": sampled["missing_marker"],
            "descriptors": {
                key: _summarise(sampled["records"], key)
                for key in (*DIFFICULTY_DESCRIPTORS, *STRUCTURAL_DESCRIPTORS)
            },
        }
        if distinct < MIN_DISTINCT_SCENES:
            entry["failures"].append(
                f"{name} ({sampled['split']}) admits {distinct} distinct "
                f"scene{'' if distinct == 1 else 's'} over {sampled['samples']} seeds, "
                f"below the declared minimum of {MIN_DISTINCT_SCENES}"
            )
        if sampled["unreachable_goals"]:
            entry["failures"].append(
                f"{name} ({sampled['split']}) has {sampled['unreachable_goals']} scenes "
                "whose goal is not reachable by a cardinal walk"
            )
        if sampled["missing_marker"]:
            entry["failures"].append(
                f"{name} ({sampled['split']}) has {sampled['missing_marker']} scenes whose "
                f"blueprint declares no marker named '{entry['goal_marker']}', so difficulty "
                "was measured against nothing there"
            )
    entry["distinct_scenes_total"] = len(total_digests)

    # ---- duplicate layout families ------------------------------------
    names = sorted(probes)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            if probes[left] == probes[right]:
                entry["duplicate_families"].append([left, right])
                entry["failures"].append(
                    f"{left} and {right} generate identical content on every probe seed; "
                    "the split is a name, not a difference"
                )

    # ---- difficulty parity and structural separation -------------------
    train_records = [
        record for family in spec.families("train") for record in families[family.name]["records"]
    ]
    test_records = [
        record for family in spec.families("test") for record in families[family.name]["records"]
    ]
    if not train_records or not test_records:
        entry["failures"].append("task does not declare both a train and a test split")
        entry["verdict"] = "fail"
        return entry

    # The overlapping coefficient is symmetric, so on its own it cannot tell a
    # holdout that is *harder* than training -- the failure PLAN.md names -- from
    # one whose difficulty is a strict subset of the training range, which is
    # merely narrow. The two means are recorded alongside it so a reader can see
    # which of the two a low overlap means.
    entry["difficulty_means"] = {}
    for key in DIFFICULTY_DESCRIPTORS:
        entry["difficulty_overlap"][key] = overlap_coefficient(
            [r[key] for r in train_records], [r[key] for r in test_records]
        )
        entry["difficulty_means"][key] = {
            "train": _summarise(train_records, key).get("mean"),
            "test": _summarise(test_records, key).get("mean"),
        }
    for key in STRUCTURAL_DESCRIPTORS:
        entry["structural_overlap"][key] = overlap_coefficient(
            [r[key] for r in train_records], [r[key] for r in test_records]
        )

    measured_difficulty = {k: v for k, v in entry["difficulty_overlap"].items() if v is not None}
    entry["difficulty_parity"] = (
        "unavailable"
        if not measured_difficulty
        else "ok"
        if all(v >= PARITY_MIN_OVERLAP for v in measured_difficulty.values())
        else "violated"
    )
    for key, value in measured_difficulty.items():
        if value < PARITY_MIN_OVERLAP:
            means = entry["difficulty_means"][key]
            rose = (means["test"] or 0) > (means["train"] or 0)
            harder = rose != (key in HARDER_WHEN_SMALLER)
            direction = (
                "the holdout is harder"
                if harder
                else "the holdout is narrower or easier than training"
            )
            entry["failures"].append(
                f"difficulty parity: train/test overlap on {key} is {value:.2f}, below "
                f"{PARITY_MIN_OVERLAP:.2f} (train mean {means['train']}, test mean "
                f"{means['test']}); {direction}, so a held-out score is not a transfer score"
            )

    separating = sorted(
        k
        for k, v in entry["structural_overlap"].items()
        if v is not None and v <= SEPARATION_MAX_OVERLAP
    )
    entry["separating_descriptors"] = separating
    if not separating:
        entry["failures"].append(
            "structural separation: no structural descriptor separates train from test "
            f"at overlap <= {SEPARATION_MAX_OVERLAP:.2f}; the holdout is the same shape "
            "of scene under a different name"
        )

    if entry["failures"]:
        entry["verdict"] = "fail"
    elif entry["difficulty_parity"] == "unavailable":
        # Not a pass. PLAN.md requires difficulty parity to be *published*, and a
        # task that names no position -- neither a `difficulty_marker` nor a
        # positional success predicate -- cannot produce the number. Reporting
        # that as a pass would reproduce the failure this tool exists to remove:
        # a green result standing in for an unmeasured claim. Note that adding
        # `difficulty_marker` gave every task a way *out* of this verdict, which
        # makes it more important, not less, that the way out is an explicit
        # declaration: an undeclared task must still land here rather than
        # sliding to `pass` on a marker the tool picked for itself.
        entry["verdict"] = "incomplete"
        entry["failures"].append(
            "difficulty parity is unmeasurable: the task declares no difficulty_marker and "
            "no success or failure predicate names a marker, so there is no declared "
            "position to measure a route to"
        )
    else:
        entry["verdict"] = "pass"
    return entry


# ------------------------------------------------------------------ reporting


def _fmt(value: float | None) -> str:
    return " n/a" if value is None else f"{value:.2f}"


def _mean(summary: dict, width: int) -> str:
    """A descriptor with no measurable value prints `n/a`, never a number.

    Printing an unmeasured descriptor as 0.0 or nan is how an unavailable
    measurement gets read as a measured one, which is the reporting defect this
    whole tool exists to remove.
    """
    value = summary.get("mean")
    return f"{'n/a':>{width}}" if value is None else f"{value:>{width}.1f}"


def print_report(report: dict) -> None:
    for task_id, entry in sorted(report["tasks"].items()):
        # The source is printed next to the marker so the reader can see at a
        # glance whether the route was chosen by the task author or derived from
        # a predicate; a report that showed only the name would make the two
        # indistinguishable.
        source = f", {entry['goal_marker_source']}" if entry["goal_marker"] else ""
        print(
            f"\n{task_id}  (v{entry['version']}, "
            f"difficulty measured to: {entry['goal_marker'] or 'none'}{source})"
        )
        header = (
            f"{'layout family':<18}{'split':<7}{'distinct':>9}{'obst':>7}{'path':>8}{'slack':>8}"
        )
        print(f"  {header}")
        for name, family in entry["layout_families"].items():
            descriptors = family["descriptors"]
            print(
                f"  {name:<18}{family['split']:<7}{family['distinct_scenes']:>9}"
                f"{_mean(descriptors['obstacle_count'], 7)}"
                f"{_mean(descriptors['path_length'], 8)}"
                f"{_mean(descriptors['budget_slack'], 8)}"
            )
        difficulty = "  ".join(f"{k}={_fmt(v)}" for k, v in entry["difficulty_overlap"].items())
        structural = "  ".join(f"{k}={_fmt(v)}" for k, v in entry["structural_overlap"].items())
        print(f"  difficulty overlap (want >= {PARITY_MIN_OVERLAP:.2f}): {difficulty}")
        print(f"  structural overlap (want one <= {SEPARATION_MAX_OVERLAP:.2f}): {structural}")
        print(f"  VERDICT: {entry['verdict'].upper()}")
        for failure in entry["failures"]:
            print(f"    - {failure}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--tasks", default=",".join(sorted(all_tasks())))
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument(
        "--exit-zero",
        action="store_true",
        help="always exit 0; for exploring a generator mid-edit",
    )
    args = parser.parse_args()

    task_ids = [t.strip() for t in args.tasks.split(",") if t.strip()]
    plan = SeedPlan(master=args.seed, run_id="generator-diagnostics")
    started = time.perf_counter()

    report: dict = {
        "samples_per_family": args.samples,
        "seed": args.seed,
        "bins": BINS,
        "thresholds": {
            "min_distinct_scenes": MIN_DISTINCT_SCENES,
            "parity_min_overlap": PARITY_MIN_OVERLAP,
            "separation_max_overlap": SEPARATION_MAX_OVERLAP,
            "note": (
                "starting points, not principled values; calibrated against this tool's "
                "own sampling noise and the repo's evaluation batch sizes"
            ),
        },
        "tasks": {},
    }
    for task_id in task_ids:
        report["tasks"][task_id] = analyse_task(task_id, args.samples, plan)

    report["wall_seconds"] = round(time.perf_counter() - started, 1)
    report["passing"] = sorted(t for t, e in report["tasks"].items() if e["verdict"] == "pass")
    report["failing"] = sorted(t for t, e in report["tasks"].items() if e["verdict"] != "pass")

    out = ROOT / "docs" / "evidence" / "phase3-generator-diagnostics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    # The digests and per-sample records are working data, not evidence; the
    # published report keeps the aggregates a reader can check.
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print_report(report)
    print(f"\npassing: {report['passing']}")
    print(f"failing: {report['failing']}")
    print(f"wrote {out}")
    # Non-zero on any non-pass, following tools/solvability.py rather than
    # tools/feasibility_sweep.py: a feasibility sweep asks an open question and a
    # split audit asserts a property PLAN.md requires. The intended consumer is a
    # future Phase 3 gate step, which will read the exit code before it reads the
    # JSON, and a tool that exits 0 while reporting a defect teaches that gate to
    # ignore it. `--exit-zero` covers editing a generator interactively.
    return 0 if args.exit_zero or not report["failing"] else 1


if __name__ == "__main__":
    sys.exit(main())
