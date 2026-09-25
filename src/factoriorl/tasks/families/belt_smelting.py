"""Build a belt-fed smelting line between three sites too far apart to share.

The scene holds an iron ore patch, a coal patch and a pre-placed output chest,
each at least 21 tiles (edge to edge) from the other two, so no standing
position is within the character's 10-tile reach of two of them. The agent
starts with the parts of a line and nothing built:

    4 burner mining drills, 4 stone furnaces, 40 transport belts,
    8 burner inserters, 20 coal

and has to connect ore to the chest. The objective is **iron plates delivered
into the chest** during an action-locked 36,000-tick (ten-minute) verification
window that starts when construction ends: reward ``min(1, plates / 60)``,
success at 60. It is scored by the same machinery as
``construct_smelting_line`` -- ``VerificationSpec`` -- with ``container`` set,
so the count is the chest's increase over the window, and with
``source="iron-ore"``, so plates smelted from hand-mined ore are capped away
exactly as they are there. Nothing is in the chest at reset and the verifier
has not run, so success is false at reset by construction.

Coal is deliberately short
--------------------------
Twenty coal is 80 MJ, and a delivered plate costs about 1.02 MJ: a burner
drill spends 600 kJ an ore (150 kW at 0.25 ore/s), a stone furnace 288 kJ a
plate (90 kW, crafting speed 1, 3.2 s) and each of the plate's two burner
inserter swings 66.9 kJ (``docs/sim-logistics.md``). So 20 coal buys at most
about 78 plates, and only if none of it burns before the window: fuelling the
drills early, or over-filling one machine, loses the margin over 60. The
reference spends it in whole coal and delivers 66. The coal patch is there so a
line can be made to outlast that -- by a coal drill pair, a coal belt, or
hand-mining where the action profile has it -- and the reward pays for it only
through throughput. Nothing names a coal line in the objective.

Layout families
---------------
* ``open`` (train): three sites, nothing else.
* ``walled`` (train): one or two short stone-wall segments across the lines
  between sites, so walking and belt routes may have to go around.
* ``split_patch`` (train): the iron patch is two rectangles with a two-tile
  ore-free strip between them.
* ``obstructed`` (test): two or three longer segments plus scattered single
  walls, and both patches have their corners clipped.
* ``far_chest`` (test): the chest sits at the far end of the belt budget,
  beyond anything a training scene draws, and single walls are scattered
  everywhere except the corridor between the iron patch and the chest.

Both test families carry scattered single walls and no training family does,
which is what separates the splits structurally (`tools/generator_diagnostics.py`
reads it as ``obstacle_count``); walking difficulty stays at parity.

Every wall keeps ``CLEARANCE`` tiles from both patches and ``CHEST_CLEARANCE``
from the chest, so a site's own build area is never walled in; the scene lives
inside ``[-SCENE_HALF, SCENE_HALF)`` on both axes, well inside the simulator's
belt-delay table.

Generator draw order (for a draw-for-draw port)
-----------------------------------------------
Only ``rng.randint`` is used; every loop is either a fixed count or a
rejection loop that redraws the same values in the same order.

1. Iron patch. ``split_patch``: ``axis = randint(0, 1)``,
   ``long = randint(10, 12)``, ``short = randint(4, 6)``,
   ``cut = randint(4, long - 6)``. ``obstructed``: ``w = randint(6, 8)``,
   ``h = randint(6, 8)``, ``clip = randint(1, 2)``. Otherwise
   ``w = randint(4, 7)``, ``h = randint(4, 7)``. Then, for every family,
   ``x0 = randint(-8, 8) - w // 2``, ``y0 = randint(-8, 8) - h // 2``.
2. Coal patch size: ``w = randint(4, 6)``, ``h = randint(4, 6)``;
   ``obstructed`` clips its corners by 1 without a draw.
3. Chest, rejection loop: ``dx = randint(-40, 40)``, ``dy = randint(-40, 40)``
   from the iron patch's centre tile, until ``_chest_ok``.
4. Coal patch position, rejection loop: ``dx = randint(-40, 40)``,
   ``dy = randint(-40, 40)`` from the iron centre tile to the coal patch's
   top-left corner, until ``_coal_ok``.
5. Wall segments (``walled``: ``n = randint(1, 2)``; ``obstructed``:
   ``n = randint(2, 3)``; others none). Per segment, always four draws:
   ``pair = randint(0, 2)``, ``t = randint(35, 65)``, ``length`` (``randint(3, 6)``,
   ``obstructed`` ``randint(4, 8)``), ``offset = randint(-3, 3)``. A segment with
   any illegal tile is dropped whole; a tile another wall already holds is
   skipped.
6. Clutter (``obstructed``: ``k = randint(6, 10)``; ``far_chest``:
   ``k = randint(18, 26)``; others none), then per item ``x = randint(bx0, bx1)``,
   ``y = randint(by0, by1)`` over the three sites' bounding box; an illegal or
   taken tile is dropped, and so, in ``far_chest``, is a tile inside the
   iron/chest bounding box grown by ``CORRIDOR_MARGIN``.
7. Character start, rejection loop: ``x = randint(bx0, bx1)``,
   ``y = randint(by0, by1)`` until the tile holds no wall and no chest.

Every distance test is integer arithmetic on tile rectangles (see ``_gap_sq``
and ``_centre_dist_sq4``), so a port reproduces acceptance exactly.
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import dataclass, field

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

TARGET_PLATES = 60
VERIFICATION_TICKS = 36000
DECISION_BUDGET = 2500
DECISION_TICKS = 30
#: Construction may use every decision the budget allows.
CONSTRUCTION_TICKS = DECISION_BUDGET * DECISION_TICKS

STARTING_INVENTORY = {
    "burner-mining-drill": 4,
    "stone-furnace": 4,
    "transport-belt": 40,
    "burner-inserter": 8,
    "coal": 20,
}
BELTS = STARTING_INVENTORY["transport-belt"]

#: Every scene tile lies in [-SCENE_HALF, SCENE_HALF) on both axes.
SCENE_HALF = 60
#: Edge-to-edge gap between any two sites, squared. 21 rather than 20: two
#: points 10 tiles from sites 21 apart would be 20 apart, so with a gap of 21
#: no position is within reach (10) of both.
MIN_GAP_SQ = 21 * 21
#: Centre-to-centre distance between any two sites, in [20, 40] tiles. Compared
#: as the squared distance between *doubled* centres, which are integers.
MIN_CENTRE_SQ4 = (2 * 20) ** 2
MAX_CENTRE_SQ4 = (2 * 40) ** 2
#: Manhattan gap from the iron patch to the chest, per family. The line needs
#: about this many belts plus a small constant, so the ceiling is what keeps
#: every scene buildable with 40 belts; the tests check every sampled scene
#: against the planner rather than trusting the arithmetic.
CHEST_MANHATTAN = {
    "open": (21, 34),
    "walled": (21, 30),
    "split_patch": (21, 34),
    "obstructed": (21, 30),
    "far_chest": (35, 38),
}
#: Walls keep this Chebyshev gap from each patch and from the chest.
CLEARANCE = 6
CHEST_CLEARANCE = 4
#: `far_chest` clutter stays outside the iron/chest bounding box grown by this.
CORRIDOR_MARGIN = 3

FAMILIES = (
    LayoutFamily("open", "train"),
    LayoutFamily("walled", "train"),
    LayoutFamily("split_patch", "train"),
    LayoutFamily("obstructed", "test"),
    LayoutFamily("far_chest", "test"),
)

SPEC = TaskSpec(
    id="belt_smelting",
    version="1.0.0",
    description=(
        "Build a smelting line from an iron patch to a distant output chest with "
        "drills, furnaces, belts and burner inserters. Success is at least 60 iron "
        "plates delivered into the chest during an action-locked ten-minute window."
    ),
    layout_families=FAMILIES,
    success=(
        Predicate(PredicateKind.VERIFIED_MACHINE_OUTPUT, item="iron-plate", at_least=TARGET_PLATES),
    ),
    rewards=(
        RewardComponent("verified_output", RewardKind.SPARSE_SUCCESS, weight=1.0, shaping=False),
    ),
    track="production",
    max_decision_steps=DECISION_BUDGET,
    decision_ticks=DECISION_TICKS,
    max_game_ticks=CONSTRUCTION_TICKS + VERIFICATION_TICKS,
    verification=VerificationSpec(
        "iron-plate",
        TARGET_PLATES,
        VERIFICATION_TICKS,
        source="iron-ore",
        container="output",
    ),
    # The `v3` profile: a 15x15 placement window, one goal slot per public
    # marker (all three sites), and a 96-entity observation that holds a
    # whole built line.
    catalog="parameterized-v3",
    observation_profile="local-v3",
    extra_public_markers=("iron", "coal", "output"),
    difficulty_marker="output",
)


# ------------------------------------------------------------------ geometry

Rect = tuple[int, int, int, int]  # x0, y0, x1, y1, inclusive tile indices
Tile = tuple[int, int]


def _gap_sq(a: Rect, b: Rect) -> int:
    """Squared Euclidean gap between two tile rectangles, edge to edge."""
    gx = max(0, b[0] - a[2] - 1, a[0] - b[2] - 1)
    gy = max(0, b[1] - a[3] - 1, a[1] - b[3] - 1)
    return gx * gx + gy * gy


def _manhattan_gap(a: Rect, b: Rect) -> int:
    gx = max(0, b[0] - a[2] - 1, a[0] - b[2] - 1)
    gy = max(0, b[1] - a[3] - 1, a[1] - b[3] - 1)
    return gx + gy


def _chebyshev_gap(a: Rect, b: Rect) -> int:
    gx = max(0, b[0] - a[2] - 1, a[0] - b[2] - 1)
    gy = max(0, b[1] - a[3] - 1, a[1] - b[3] - 1)
    return max(gx, gy)


def _centre_dist_sq4(a: Rect, b: Rect) -> int:
    """Squared distance between the rectangles' centres, times four."""
    dx = (b[0] + b[2]) - (a[0] + a[2])
    dy = (b[1] + b[3]) - (a[1] + a[3])
    return dx * dx + dy * dy


def _rect_of(tile: Tile) -> Rect:
    return (tile[0], tile[1], tile[0], tile[1])


def _centre_tile(r: Rect) -> Tile:
    return (r[0] + (r[2] - r[0]) // 2, r[1] + (r[3] - r[1]) // 2)


def _in_scene(r: Rect) -> bool:
    return -SCENE_HALF <= r[0] and r[2] < SCENE_HALF and -SCENE_HALF <= r[1] and r[3] < SCENE_HALF


def _pair_ok(a: Rect, b: Rect) -> bool:
    return (
        _gap_sq(a, b) >= MIN_GAP_SQ and MIN_CENTRE_SQ4 <= _centre_dist_sq4(a, b) <= MAX_CENTRE_SQ4
    )


def _clipped(rect: Rect, clip: int) -> list[Tile]:
    """A rectangle's tiles minus corner triangles of `clip` tiles."""
    x0, y0, x1, y1 = rect
    tiles = []
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            if min(x - x0, x1 - x) + min(y - y0, y1 - y) >= clip:
                tiles.append((x, y))
    return tiles


@dataclass(frozen=True)
class Scene:
    """The generator's decisions, before they become a `Blueprint`."""

    iron: tuple[Tile, ...]
    iron_rect: Rect
    coal: tuple[Tile, ...]
    coal_rect: Rect
    chest: Tile
    walls: tuple[Tile, ...]
    start: Tile


def _iron_patch(family: str, rng) -> tuple[list[Tile], Rect]:
    if family == "split_patch":
        axis = rng.randint(0, 1)
        long = rng.randint(10, 12)
        short = rng.randint(4, 6)
        cut = rng.randint(4, long - 6)
        w, h = (long, short) if axis == 0 else (short, long)
    elif family == "obstructed":
        w = rng.randint(6, 8)
        h = rng.randint(6, 8)
        clip = rng.randint(1, 2)
    else:
        w = rng.randint(4, 7)
        h = rng.randint(4, 7)
    x0 = rng.randint(-8, 8) - w // 2
    y0 = rng.randint(-8, 8) - h // 2
    rect = (x0, y0, x0 + w - 1, y0 + h - 1)
    if family == "split_patch":
        # A two-tile strip without ore at `cut`, `cut + 1` along the long axis.
        tiles = [
            (x, y)
            for x in range(rect[0], rect[2] + 1)
            for y in range(rect[1], rect[3] + 1)
            if ((x - x0) if axis == 0 else (y - y0)) not in (cut, cut + 1)
        ]
        return tiles, rect
    if family == "obstructed":
        return _clipped(rect, clip), rect
    return _clipped(rect, 0), rect


def _chest_ok(family: str, iron: Rect, chest: Rect) -> bool:
    low, high = CHEST_MANHATTAN[family]
    return _in_scene(chest) and _pair_ok(iron, chest) and low <= _manhattan_gap(iron, chest) <= high


def _coal_ok(iron: Rect, chest: Rect, coal: Rect) -> bool:
    return _in_scene(coal) and _pair_ok(iron, coal) and _pair_ok(chest, coal)


def _wall_ok(tile: Tile, iron: Rect, coal: Rect, chest: Rect) -> bool:
    t = (tile[0], tile[1], tile[0], tile[1])
    return (
        _in_scene(t)
        and _chebyshev_gap(t, iron) >= CLEARANCE
        and _chebyshev_gap(t, coal) >= CLEARANCE
        and _chebyshev_gap(t, chest) >= CHEST_CLEARANCE
    )


def scene(family: LayoutFamily | str, rng) -> Scene:
    """Draw one scene. See the module docstring for the exact draw order."""
    name = family if isinstance(family, str) else family.name
    iron, iron_rect = _iron_patch(name, rng)
    coal_w = rng.randint(4, 6)
    coal_h = rng.randint(4, 6)

    ix, iy = _centre_tile(iron_rect)
    while True:
        cx = ix + rng.randint(-40, 40)
        cy = iy + rng.randint(-40, 40)
        chest_rect = (cx, cy, cx, cy)
        if _chest_ok(name, iron_rect, chest_rect):
            break
    while True:
        qx = ix + rng.randint(-40, 40)
        qy = iy + rng.randint(-40, 40)
        coal_rect = (qx, qy, qx + coal_w - 1, qy + coal_h - 1)
        if _coal_ok(iron_rect, chest_rect, coal_rect):
            break
    coal = _clipped(coal_rect, 1 if name == "obstructed" else 0)

    walls: list[Tile] = []
    taken: set[Tile] = set()
    segments = {"walled": (1, 2), "obstructed": (2, 3)}.get(name)
    if segments:
        rects = (iron_rect, coal_rect, chest_rect)
        pairs = ((0, 2), (0, 1), (1, 2))  # iron-chest, iron-coal, coal-chest
        for _ in range(rng.randint(*segments)):
            pair = rng.randint(0, 2)
            t = rng.randint(35, 65)
            length = rng.randint(4, 8) if name == "obstructed" else rng.randint(3, 6)
            offset = rng.randint(-3, 3)
            a = _centre_tile(rects[pairs[pair][0]])
            b = _centre_tile(rects[pairs[pair][1]])
            px = a[0] + (b[0] - a[0]) * t // 100
            py = a[1] + (b[1] - a[1]) * t // 100
            # Across the line between the two sites: a line running mostly
            # east-west gets a north-south wall, and vice versa.
            across_x = abs(b[0] - a[0]) < abs(b[1] - a[1])
            segment = [
                (px + offset - length // 2 + k, py)
                if across_x
                else (px, py + offset - length // 2 + k)
                for k in range(length)
            ]
            if not all(_wall_ok(tile, iron_rect, coal_rect, chest_rect) for tile in segment):
                continue
            for tile in segment:
                if tile not in taken:
                    taken.add(tile)
                    walls.append(tile)

    bx0 = min(iron_rect[0], coal_rect[0], chest_rect[0])
    by0 = min(iron_rect[1], coal_rect[1], chest_rect[1])
    bx1 = max(iron_rect[2], coal_rect[2], chest_rect[2])
    by1 = max(iron_rect[3], coal_rect[3], chest_rect[3])
    clutter = {"obstructed": (6, 10), "far_chest": (18, 26)}.get(name)
    if clutter:
        # `far_chest` keeps its clutter out of the iron-to-chest corridor: its
        # belt already runs at the end of the budget, and the family tests the
        # distance, not a detour on top of it.
        corridor = (
            min(iron_rect[0], chest_rect[0]) - CORRIDOR_MARGIN,
            min(iron_rect[1], chest_rect[1]) - CORRIDOR_MARGIN,
            max(iron_rect[2], chest_rect[2]) + CORRIDOR_MARGIN,
            max(iron_rect[3], chest_rect[3]) + CORRIDOR_MARGIN,
        )
        for _ in range(rng.randint(*clutter)):
            tile = (rng.randint(bx0, bx1), rng.randint(by0, by1))
            if tile in taken or not _wall_ok(tile, iron_rect, coal_rect, chest_rect):
                continue
            if name == "far_chest" and _chebyshev_gap(_rect_of(tile), corridor) == 0:
                continue
            taken.add(tile)
            walls.append(tile)

    while True:
        start = (rng.randint(bx0, bx1), rng.randint(by0, by1))
        if start not in taken and start != (chest_rect[0], chest_rect[1]):
            break

    return Scene(
        iron=tuple(iron),
        iron_rect=iron_rect,
        coal=tuple(coal),
        coal_rect=coal_rect,
        chest=(chest_rect[0], chest_rect[1]),
        walls=tuple(walls),
        start=start,
    )


def _patch_centre(tiles) -> tuple[float, float]:
    return (
        sum(x for x, _ in tiles) / len(tiles) + 0.5,
        sum(y for _, y in tiles) / len(tiles) + 0.5,
    )


def generate(family: LayoutFamily, rng) -> Blueprint:
    s = scene(family, rng)
    entities = [
        EntitySpec("wooden-chest", (s.chest[0] + 0.5, s.chest[1] + 0.5), marker="output"),
        *(EntitySpec("stone-wall", (x + 0.5, y + 0.5), force="neutral") for x, y in s.walls),
    ]
    resources = [
        *(ResourceSpec("iron-ore", (x + 0.5, y + 0.5), amount=10000) for x, y in s.iron),
        *(ResourceSpec("coal", (x + 0.5, y + 0.5), amount=10000) for x, y in s.coal),
    ]
    return Blueprint(
        entities=tuple(entities),
        resources=tuple(resources),
        character_position=(s.start[0] + 0.5, s.start[1] + 0.5),
        character_inventory=dict(STARTING_INVENTORY),
        # `output` is also the chest's alias; declaring it here as well keeps
        # every published marker in `Blueprint.markers`, where the registry
        # sweeps look for it.
        markers={
            "iron": _patch_centre(s.iron),
            "coal": _patch_centre(s.coal),
            "output": (s.chest[0] + 0.5, s.chest[1] + 0.5),
        },
        unlock_recipes=(
            "burner-mining-drill",
            "stone-furnace",
            "transport-belt",
            "burner-inserter",
        ),
        radius=64,
    )


# ------------------------------------------------------------------ planner
#
# Evaluator-side only: the reference solver's layout, computed from the scene
# declaration. Kept engine-free so the tests can check that every sampled scene
# has a line that fits in 40 belts, which is the property the generator's
# Manhattan ceilings exist to guarantee.

#: Compass facings as unit tile steps (y grows southward).
STEP = {"north": (0, -1), "east": (1, 0), "south": (0, 1), "west": (-1, 0)}
_CLOCKWISE = ("north", "east", "south", "west")


def _turn(direction: str, quarter_turns: int) -> str:
    return _CLOCKWISE[(_CLOCKWISE.index(direction) + quarter_turns) % 4]


def _rotate_tile(tile: Tile, quarter_turns: int) -> Tile:
    """Rotate a tile clockwise about the lattice origin, by its centre."""
    x, y = tile[0] + 0.5, tile[1] + 0.5
    for _ in range(quarter_turns % 4):
        x, y = -y, x
    return (math.floor(x), math.floor(y))


def _rotate_point(point: Tile, quarter_turns: int) -> Tile:
    x, y = point
    for _ in range(quarter_turns % 4):
        x, y = -y, x
    return (x, y)


def footprint(name: str, centre: tuple[float, float]) -> list[Tile]:
    if name in ("burner-mining-drill", "stone-furnace"):
        cx, cy = round(centre[0]), round(centre[1])
        return [(cx - 1, cy - 1), (cx, cy - 1), (cx - 1, cy), (cx, cy)]
    return [(math.floor(centre[0]), math.floor(centre[1]))]


@dataclass(frozen=True)
class Placement:
    item: str
    #: Where the entity ends up: a lattice point for a 2x2, a tile centre for 1x1.
    centre: tuple[float, float]
    direction: str

    @property
    def tiles(self) -> list[Tile]:
        return footprint(self.item, self.centre)

    @property
    def request_tile(self) -> Tile:
        """The tile to name in `place_at`: a 2x2 snaps from `t + 0.5` to `t + 1`."""
        if self.item in ("burner-mining-drill", "stone-furnace"):
            return (round(self.centre[0]) - 1, round(self.centre[1]) - 1)
        return (math.floor(self.centre[0]), math.floor(self.centre[1]))


#: The canonical cell, facing south: two drills side by side on the ore, each
#: dropping into its own furnace (a south drill at lattice point D drops into
#: tile (D.x, D.y + 1), measured), an inserter under each furnace picking from
#: it (a burner inserter's direction points at its pickup tile, measured), and a
#: three-tile collection belt under the inserters.
_CELL_MACHINES = (
    ("burner-mining-drill", (0, 0), "south"),
    ("burner-mining-drill", (2, 0), "south"),
    ("stone-furnace", (0, 2), None),
    ("stone-furnace", (2, 2), None),
)
_CELL_INSERTERS = (((-1, 3), "north"), ((1, 3), "north"))
_CELL_COLLECTION = ((-1, 4), (0, 4), (1, 4))


@dataclass
class LinePlan:
    """The reference layout for one scene."""

    machines: list[Placement] = field(default_factory=list)
    inserters: list[Placement] = field(default_factory=list)
    belts: list[Placement] = field(default_factory=list)
    chest_inserter: Placement | None = None

    @property
    def drills(self) -> list[Placement]:
        return [p for p in self.machines if p.item == "burner-mining-drill"]

    @property
    def furnaces(self) -> list[Placement]:
        return [p for p in self.machines if p.item == "stone-furnace"]


def _cell(anchor: Tile, turns: int, reverse: bool):
    machines = []
    for item, point, facing in _CELL_MACHINES:
        dx, dy = _rotate_point(point, turns)
        machines.append(
            Placement(
                item,
                (float(anchor[0] + dx), float(anchor[1] + dy)),
                _turn(facing, turns) if facing else "north",
            )
        )
    inserters = []
    for tile, facing in _CELL_INSERTERS:
        rx, ry = _rotate_tile(tile, turns)
        inserters.append(
            Placement(
                "burner-inserter",
                (anchor[0] + rx + 0.5, anchor[1] + ry + 0.5),
                _turn(facing, turns),
            )
        )
    collection = [
        (anchor[0] + _rotate_tile(t, turns)[0], anchor[1] + _rotate_tile(t, turns)[1])
        for t in _CELL_COLLECTION
    ]
    if reverse:
        collection.reverse()
    return machines, inserters, collection


def _direction_between(a: Tile, b: Tile) -> str:
    step = (b[0] - a[0], b[1] - a[1])
    for name, delta in STEP.items():
        if delta == step:
            return name
    raise ValueError(f"{a} and {b} are not adjacent")


def _bfs(start: Tile, blocked: set[Tile], goal: Tile | None = None) -> dict[Tile, Tile | None]:
    """Breadth-first parents over 4-connected in-scene tiles."""
    parents: dict[Tile, Tile | None] = {start: None}
    queue = deque([start])
    while queue:
        here = queue.popleft()
        if here == goal:
            break
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (here[0] + dx, here[1] + dy)
            if nxt in parents or nxt in blocked:
                continue
            if not (-SCENE_HALF <= nxt[0] < SCENE_HALF and -SCENE_HALF <= nxt[1] < SCENE_HALF):
                continue
            parents[nxt] = here
            queue.append(nxt)
    return parents


def _depths(start: Tile, blocked: set[Tile]) -> dict[Tile, int]:
    """Breadth-first distance from `start` to every reachable in-scene tile."""
    depth = {start: 0}
    queue = deque([start])
    while queue:
        here = queue.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (here[0] + dx, here[1] + dy)
            if nxt in depth or nxt in blocked:
                continue
            if not (-SCENE_HALF <= nxt[0] < SCENE_HALF and -SCENE_HALF <= nxt[1] < SCENE_HALF):
                continue
            depth[nxt] = depth[here] + 1
            queue.append(nxt)
    return depth


def _path(parents: dict[Tile, Tile | None], goal: Tile) -> list[Tile]:
    path = [goal]
    while parents[path[-1]] is not None:
        path.append(parents[path[-1]])
    return path[::-1]


def plan_line(s: Scene, belts: int = BELTS) -> LinePlan | None:
    """The shortest-belt reference layout for a scene, or None if none fits.

    Enumerates every cell position and facing whose two drills sit wholly on
    iron ore, and every side of the chest for the chest inserter; routes the
    belt from the collection's end to the tile behind the chest inserter by
    breadth-first search around walls, the cell and the chest; and keeps the
    layout using the fewest belts. The static distance map from each chest side
    is a lower bound on every cell's route, so exact searches stop as soon as
    no remaining candidate can beat the best one found.
    """
    ore = set(s.iron)
    static = set(s.walls) | {s.chest}
    candidates = []
    for side in _CLOCKWISE:
        sx, sy = STEP[side]
        ins_tile = (s.chest[0] + sx, s.chest[1] + sy)
        pick_tile = (s.chest[0] + 2 * sx, s.chest[1] + 2 * sy)
        if ins_tile in static or pick_tile in static:
            continue
        lower = _depths(pick_tile, static | {ins_tile})
        for ax in range(s.iron_rect[0] - 2, s.iron_rect[2] + 3):
            for ay in range(s.iron_rect[1] - 2, s.iron_rect[3] + 3):
                for turns in range(4):
                    for reverse in (False, True):
                        machines, inserters, collection = _cell((ax, ay), turns, reverse)
                        drill_tiles = [t for m in machines[:2] for t in m.tiles]
                        if not all(t in ore for t in drill_tiles):
                            continue
                        cell_tiles = {t for m in machines for t in m.tiles}
                        cell_tiles |= {t for i in inserters for t in i.tiles}
                        cell_tiles |= set(collection)
                        if cell_tiles & (static | {ins_tile, pick_tile}):
                            continue
                        if collection[-1] not in lower:
                            continue
                        bound = lower[collection[-1]] + len(collection)
                        candidates.append(
                            (bound, side, (ax, ay), turns, reverse, ins_tile, pick_tile)
                        )
    candidates.sort(key=lambda c: (c[0], _CLOCKWISE.index(c[1]), c[2], c[3], c[4]))

    best: tuple[int, LinePlan] | None = None
    for bound, side, anchor, turns, reverse, ins_tile, pick_tile in candidates:
        if bound > belts or (best is not None and bound >= best[0]):
            break
        machines, inserters, collection = _cell(anchor, turns, reverse)
        cell_tiles = {t for m in machines for t in m.tiles}
        cell_tiles |= {t for i in inserters for t in i.tiles}
        cell_tiles |= set(collection[:-1])
        blocked = static | cell_tiles | {ins_tile}
        parents = _bfs(collection[-1], blocked, goal=pick_tile)
        if pick_tile not in parents:
            continue
        route = collection[:-1] + _path(parents, pick_tile)
        if len(route) > belts or (best is not None and len(route) >= best[0]):
            continue
        plan = LinePlan(machines=machines, inserters=inserters)
        for here, nxt in zip(route, route[1:], strict=False):
            plan.belts.append(
                Placement(
                    "transport-belt", (here[0] + 0.5, here[1] + 0.5), _direction_between(here, nxt)
                )
            )
        plan.belts.append(
            Placement(
                "transport-belt",
                (pick_tile[0] + 0.5, pick_tile[1] + 0.5),
                _direction_between(pick_tile, ins_tile),
            )
        )
        plan.chest_inserter = Placement(
            "burner-inserter", (ins_tile[0] + 0.5, ins_tile[1] + 0.5), side
        )
        best = (len(route), plan)
    return best[1] if best else None


# ------------------------------------------------------------------- solver
#
# The reference builder. It reads the scene declaration (regenerated from the
# episode's own seed, which is evaluator knowledge a reference is allowed) and
# acts only through the task's catalog: strides to walk, `place_at` to build,
# `mine_at` to hand-mine coal, `give_to` to fuel. The layout is `plan_line`'s.

#: Coal the builder hand-mines before it builds. **Zero**: the reference runs on
#: the 20 starting coal alone, so it does not depend on hand-mining being in
#: the action profile. `mine_coal` stays for a caller that sets this.
EXTRA_COAL = 0
#: Coal per machine, in whole coal, for 66 delivered plates from 20. The
#: drills are the bottleneck by design: 10 coal is 66 ore at 600 kJ an ore.
#: 3 per furnace smelts 41 plates each (288 kJ a plate) against the 33 each
#: gets. An inserter starts burning a quarter wood (500 kJ, about 7 swings) and
#: each coal adds 60 swings at 66.9 kJ, so 1 covers an output inserter's 33
#: and 2 the chest inserter's 66. The drills get what is left and are fuelled
#: last: nothing else burns anything while it has nothing to do, so every
#: joule they hold is spent inside the window. Measured on the engine with 2
#: coal per furnace: both furnaces ran dry holding 11 ore, 54 plates delivered.
FUEL_OUTPUT_INSERTER = 1
FUEL_FURNACE = 3
FUEL_CHEST_INSERTER = 2
#: How close each walk has to end to its tile centre. A placement binds to the
#: character's tile, so this keeps `floor(position)` on the chosen tile, and the
#: character's 0.4-tile body clear of the neighbouring tiles.
STAND_TOLERANCE = 0.25
#: Placement window half-width when the environment does not say
#: (`env.PLACEMENT_RADIUS`); a `v3` environment carries its own, 7.
REACH_TILES = 5
#: Standing tile centre to entity centre, in tiles. The character's build
#: distance is 10; this leaves room for where inside its tile it stops.
BUILD_REACH = 8.5
#: Walking over a placed belt is allowed but avoided, at up to this many tiles
#: of detour per belt tile: a belt carries whatever stands on it, so a stride
#: that ends on one is carried off -- measured, through a one-tile gap between
#: two walls, to where the only way back was against the belt's flow.
BELT_WALK_COST = 60
#: Re-plans a walk may make after being stopped or carried.
WALK_ATTEMPTS = 5
#: Standing this far (tiles, Euclidean) from a machine is inside its 10-tile
#: interaction reach with room to spare.
FUEL_REACH = 6.0


def _episode_scene(env) -> Scene:
    """The scene the environment installed, drawn again from its own seed."""
    families = env.spec_.families(env.split)
    rng = env.seed_plan.generator_rng(env.branch, env._episode_index)
    family = families[rng.randrange(len(families))]
    return scene(family, rng)


class _Builder:
    def __init__(self, driver, s: Scene, plan: LinePlan) -> None:
        self.driver = driver
        self.scene = s
        self.plan = plan
        self.walls = set(s.walls) | {s.chest}
        #: Non-belt footprints built so far, and belt tiles built so far.
        self.solid: set[Tile] = set()
        self.belts: set[Tile] = set()
        #: Every tile the plan will build on, until it is built.
        self.reserved: set[Tile] = {
            t
            for p in [*plan.machines, *plan.inserters, *plan.belts, plan.chest_inserter]
            for t in p.tiles
        }
        self.handles: dict[tuple[str, tuple[float, float]], str] = {}
        self._parent: dict[Tile, Tile | None] | None = None

    # ---- reading the world ------------------------------------------------

    def here(self) -> Tile:
        x, y = self.driver.position
        return (math.floor(x), math.floor(y))

    def fail(self, reason: str) -> bool:
        if self.driver.trace.stuck_reason is None:
            self.driver.trace.stuck_reason = reason
        return False

    def entity(self, placement: Placement) -> dict | None:
        best = None
        for record in self.driver.env._observation.get("entities") or []:
            if record.get("name") != placement.item or not record.get("p"):
                continue
            d = math.dist(record["p"], placement.centre)
            if d <= 0.6 and (best is None or d < best[0]):
                best = (d, record)
        return best[1] if best else None

    # ---- walking -----------------------------------------------------------

    def costs_from(self, origin: Tile) -> tuple[dict[Tile, int], dict[Tile, Tile | None]]:
        """Dijkstra over walkable tiles: walls and machines block, belts cost more."""
        blocked = self.walls | self.solid
        cost = {origin: 0}
        parent: dict[Tile, Tile | None] = {origin: None}
        heap = [(0, origin)]
        while heap:
            c, here = heapq.heappop(heap)
            if c > cost.get(here, 1 << 30):
                continue
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (here[0] + dx, here[1] + dy)
                if nxt in blocked:
                    continue
                if not (-SCENE_HALF <= nxt[0] < SCENE_HALF and -SCENE_HALF <= nxt[1] < SCENE_HALF):
                    continue
                step = BELT_WALK_COST if nxt in self.belts else 1
                if c + step < cost.get(nxt, 1 << 30):
                    cost[nxt] = c + step
                    parent[nxt] = here
                    heapq.heappush(heap, (c + step, nxt))
        return cost, parent

    def settled_on(self, goal: Tile) -> bool:
        x, y = self.driver.position
        return (
            abs(x - goal[0] - 0.5) <= STAND_TOLERANCE and abs(y - goal[1] - 0.5) <= STAND_TOLERANCE
        )

    def walk_tile(self, goal: Tile, parent: dict[Tile, Tile | None] | None = None) -> bool:
        for _attempt in range(WALK_ATTEMPTS):
            if self.settled_on(goal):
                return True
            start = self.here()
            if parent is None or start not in parent:
                _, parent = self.costs_from(start)
            if goal not in parent:
                return self.fail(f"no walkable route from {start} to {goal}")
            route = [goal]
            while parent[route[-1]] is not None and route[-1] != start:
                route.append(parent[route[-1]])
            route.reverse()
            # Keep only the corners: between two corners the route is straight
            # and free, so an axis-first walk between them cannot clip anything.
            corners = [route[0]]
            for a, b, c in zip(route, route[1:], route[2:], strict=False):
                if (b[0] - a[0], b[1] - a[1]) != (c[0] - b[0], c[1] - b[1]):
                    corners.append(b)
            corners.append(route[-1])
            ok = True
            for corner in corners[1:]:
                if not self.driver.walk_to(
                    (corner[0] + 0.5, corner[1] + 0.5), tolerance=STAND_TOLERANCE, budget=80
                ):
                    ok = False
                    break
            if ok and self.settled_on(goal):
                return True
            if self.driver.terminated or self.driver.truncated:
                return False
            # Stuck on something, or carried off by a belt: re-plan from
            # wherever the character ended up.
            self.driver.trace.stuck_reason = None
            parent = None
        return self.fail(f"could not walk to {goal}")

    def stand_for(self, placement: Placement) -> Tile | None:
        """The cheapest tile to build `placement` from."""
        request = placement.request_tile
        footprint_tiles = placement.tiles
        cost, parent = self.costs_from(self.here())
        reach = int(getattr(self.driver.env, "placement_radius", REACH_TILES))
        best = None
        for dx in range(-reach, reach + 1):
            for dy in range(-reach, reach + 1):
                tile = (request[0] + dx, request[1] + dy)
                if tile not in cost or tile in self.belts or tile in self.reserved:
                    continue
                if any(max(abs(tile[0] - f[0]), abs(tile[1] - f[1])) < 2 for f in footprint_tiles):
                    continue
                # The window is a square and build reach is a circle: a `v3`
                # corner tile can be offered and refused `out_of_reach` (a 2x2
                # snapped 10.06 tiles away was, measured).
                if math.dist((tile[0] + 0.5, tile[1] + 0.5), placement.centre) > BUILD_REACH:
                    continue
                key = (cost[tile], abs(dx) + abs(dy), tile)
                if best is None or key < best[0]:
                    best = (key, tile)
        if best is None:
            return None
        self._parent = parent
        return best[1]

    def stand_near(self, centre: tuple[float, float]) -> bool:
        cost, parent = self.costs_from(self.here())
        options = [
            (c, t)
            for t, c in cost.items()
            if t not in self.belts
            and t not in self.reserved
            and math.dist((t[0] + 0.5, t[1] + 0.5), centre) <= FUEL_REACH
        ]
        if not options:
            return self.fail(f"nowhere to stand within reach of {centre}")
        return self.walk_tile(min(options)[1], parent)

    # ---- acting ------------------------------------------------------------

    def act(self, key: str, **arguments) -> bool:
        self.driver.trace.last_error = None
        if not self.driver.do(key, **arguments):
            return False
        if self.driver.trace.last_error:
            return self.fail(f"{key} refused: {self.driver.trace.last_error}")
        return True

    def place(self, placement: Placement) -> bool:
        stand = self.stand_for(placement)
        if stand is None:
            return self.fail(f"nowhere to stand to build {placement.item} at {placement.centre}")
        if not self.walk_tile(stand, self._parent):
            return False
        position = self.driver.placement_for(placement.request_tile)
        if position is None:
            return self.fail(f"tile {placement.request_tile} is not an offered placement")
        if not self.act(
            "place_at", item=placement.item, position=position, direction=placement.direction
        ):
            return False
        record = self.entity(placement)
        if record is None:
            return self.fail(
                f"placed {placement.item} but none is observable at {placement.centre}"
            )
        self.handles[(placement.item, placement.centre)] = str(record["h"])
        for tile in placement.tiles:
            self.reserved.discard(tile)
            (self.belts if placement.item == "transport-belt" else self.solid).add(tile)
        return True

    def give_coal(self, placement: Placement, count: int) -> bool:
        handle = self.handles.get((placement.item, placement.centre))
        if handle is None:
            return self.fail(f"no handle for {placement.item} at {placement.centre}")
        if math.dist(self.driver.position, placement.centre) > FUEL_REACH:
            if not self.stand_near(placement.centre):
                return False
        for amount in (20, 5, 1):
            while count >= amount:
                if not self.act("give_to", to=handle, item="coal", count=amount):
                    return False
                count -= amount
        return True

    def mine_coal(self, count: int) -> bool:
        coal = set(self.scene.coal)
        cost, parent = self.costs_from(self.here())
        reachable = [(cost[t], t) for t in coal if t in cost]
        if not reachable:
            return self.fail("no coal tile is reachable")
        if not self.walk_tile(min(reachable)[1], parent):
            return False
        for _ in range(count):
            held = int(self.driver.inventory().get("coal", 0))
            tiles = [
                tile
                for tile in (self.driver.env._observation.get("resources") or {}).get("tiles") or []
                if tile.get("name") == "coal" and tile.get("h") and tile.get("p")
            ]
            if not tiles:
                return self.fail("no coal tile in the observation at the coal patch")
            nearest = min(tiles, key=lambda t: math.dist(t["p"], self.driver.position))
            if not self.act("mine_at", handle=str(nearest["h"])):
                return False
            # Hand-mining one coal takes 120 ticks, four decisions.
            for _ in range(12):
                if int(self.driver.inventory().get("coal", 0)) > held:
                    break
                if not self.act("wait"):
                    return False
            else:
                return self.fail("hand-mining produced no coal")
        return True


def solve(driver) -> None:
    """Evaluator-only solvability witness; it never runs on evaluated episodes.

    Build the planned cell on the iron patch, fuel its furnaces and output
    inserters, lay the belt to the chest, build and fuel the chest inserter,
    walk back and fuel the drills with the rest, then run the verification
    window. No coal line and no hand-mining: 20 coal, spent well, is the whole
    fuel supply (see `FUEL_*`).
    """
    s = _episode_scene(driver.env)
    plan = plan_line(s)
    if plan is None:
        driver.trace.stuck_reason = "no line fits this scene in 40 belts"
        return
    builder = _Builder(driver, s, plan)
    if EXTRA_COAL and not builder.mine_coal(EXTRA_COAL):
        return
    for placement in [*plan.machines, *plan.inserters]:
        if not builder.place(placement):
            return
    for placement in plan.inserters:
        if not builder.give_coal(placement, FUEL_OUTPUT_INSERTER):
            return
    for placement in plan.furnaces:
        if not builder.give_coal(placement, FUEL_FURNACE):
            return
    for placement in [*plan.belts, plan.chest_inserter]:
        if not builder.place(placement):
            return
    if not builder.give_coal(plan.chest_inserter, FUEL_CHEST_INSERTER):
        return
    remaining = int(driver.inventory().get("coal", 0))
    share = remaining // len(plan.drills)
    for placement in plan.drills:
        if not builder.give_coal(placement, share):
            return
    outcome = driver.env.run_verification()
    driver.verification = outcome
    driver.success = bool(outcome["success"])
    if not driver.success:
        driver.trace.stuck_reason = (
            f"line delivered {outcome['machine_output']:g} of {TARGET_PLATES} plates "
            f"(uncapped {outcome['uncapped_output']:g}, ore mined in window "
            f"{outcome['machine_source']})"
        )


register(RegisteredTask(spec=SPEC, generate=generate, solve=solve))
