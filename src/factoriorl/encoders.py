"""Observation encoding for compact policies (DESIGN.md 3.4).

The wire format is compact JSON; the tensor layout is a Python-side concern, so
the two can evolve independently. The space is a fixed-shape ``Dict`` per
versioned profile, and the encoder asserts its own shapes against the profile
declaration -- a silently reshaped observation would invalidate every checkpoint
trained on it.

Entity features carry the **is-remembered flag and observation age** rather than
dropping them, so a policy can tell "I can see this" from "I saw this 500 ticks
ago" -- which is the entire point of Phase 2.3's remembered observations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from gymnasium import spaces

#: Entity type vocabulary. Closed and ordered, so an index means one thing
#: forever; unknown types fold into the last slot rather than shifting others.
ENTITY_TYPES: tuple[str, ...] = (
    "container",
    "furnace",
    "assembling-machine",
    "mining-drill",
    "transport-belt",
    "inserter",
    "electric-pole",
    "boiler",
    "generator",
    "pipe",
    "wall",
    "other",
)

#: Item vocabulary for the inventory vector. Same rule.
ITEMS: tuple[str, ...] = (
    "iron-ore",
    "copper-ore",
    "coal",
    "stone",
    "iron-plate",
    "copper-plate",
    "stone-furnace",
    "iron-gear-wheel",
    "transport-belt",
    "wood",
    # Added because `restore_power` hands the agent poles and the policy could
    # not see them: the item was outside this vocabulary, so its inventory slot
    # was absent and the parameterized action space had no index for it. The
    # rest are what the seven families actually place or hold, so a
    # construction task's materials are visible before R3 needs them.
    "small-electric-pole",
    "wooden-chest",
    "burner-mining-drill",
    "burner-inserter",
)

#: The `v3` item vocabulary: `ITEMS` in its own order, then the Stage-2 items
#: (power and assembly), appended so every `ITEMS` index keeps its meaning.
ITEMS_V3: tuple[str, ...] = (
    *ITEMS,
    "assembling-machine-1",
    "boiler",
    "steam-engine",
    "offshore-pump",
)

RESOURCES: tuple[str, ...] = ("iron-ore", "copper-ore", "coal", "stone")

#: Factorio 2.0 uses 16 compass directions (0..15), not 8.
DIRECTION_COUNT = 16.0

#: Machine stop causes, in a declared order so an index means one thing across
#: runs. The wire carries a readable name (`st`) alongside the raw code; before
#: that, `no_fuel`, `no_power` and `waiting_for_source_items` were the same
#: single bit, so nothing could tell which fault to fix.
ENTITY_STATUS: tuple[str, ...] = (
    "working",
    "normal",
    "no_power",
    "low_power",
    "no_fuel",
    "no_minable_resources",
    "waiting_for_source_items",
    "waiting_for_space_in_destination",
    "full_output",
    "item_ingredient_shortage",
    "missing_required_fluid",
    "disabled",
    "marked_for_deconstruction",
)

#: Action statuses that mean the engine has finished with a request.
_SETTLED_STATUS = frozenset({"completed", "failed", "cancelled", "rejected"})
#: ...and those that mean it refused or failed it.
_REFUSED_STATUS = frozenset({"failed", "cancelled", "rejected"})

MAX_ENTITIES = 32


def entity_row_order(
    observation, origin, limit: int = MAX_ENTITIES
) -> list[tuple[float, dict, bool]]:
    """The entity table's rows: `(distance, record, remembered)`, nearest first.

    Visible entities and remembered ones together, sorted by distance from the
    character and capped at `limit` (`MAX_ENTITIES`, or the layout's own
    `max_entities` under `v3`). The sort is stable, so a visible
    entity precedes a remembered one at the same distance.

    It is a named function because `parameterized-v2` addresses a target *by
    row of this table*, and a policy trained against one ordering and evaluated
    against another would be reading a different entity than it chose. Both the
    tensor encoder and that action space call this one.
    """

    def rows(source, remembered: bool):
        for record in source:
            position = record.get("p", [0, 0])
            yield (
                float(np.hypot(position[0] - origin[0], position[1] - origin[1])),
                record,
                remembered,
            )

    return sorted(
        [
            *rows(observation.get("entities", []), False),
            *rows(observation.get("remembered", []), True),
        ],
        key=lambda row: row[0],
    )[:limit]


ENTITY_FEATURES = 16
SELF_FEATURES = 12
GOAL_FEATURES = 12

#: Bumped when the *meaning* of the goal vector changes without its shape
#: changing -- the sensor contract (`local-v1`) is untouched, but a policy
#: trained against version 1 read different semantics in the same slots.
#: Version 2 appends observable landmarks after the success predicates.
#: Version 3 reserves the last `GOAL_GEOMETRY_SLOTS` for where the objective
#: *is*: offset to the nearest published marker, and whether one exists. Before
#: it the vector said only whether the goal had been reached, never where it
#: was, so `deliver`'s destination was unlearnable except as a placement habit
#: of the generator.
#: Version 4: on a task declaring several faults, those slots hold the nearest
#: *unrepaired* one, chosen by the environment and retargeted as faults close --
#: not simply the nearest published marker. That is an assistance, so the task
#: declares a `focus_policy` and the manifest records it; see
#: `env.FactorioEnv._focus_target`. Version 3 was left in place after the
#: behaviour changed, so a v3 checkpoint and a v3 run could mean different
#: things in the same three slots.
GOAL_ENCODING_VERSION = 4

#: Tail slots of the goal vector holding (dx, dy, present) for the objective.
#: Predicates fill from the front and stop short of these.
GOAL_GEOMETRY_SLOTS = 3


@dataclass(frozen=True)
class TensorLayout:
    """How many rows, features, items and goal slots the tensors carry.

    Separate from `ObservationProfile`, which is about the *sensor*: a layout is
    the Python-side shape a policy is trained against. `LAYOUT_V1` is the
    layout every `parameterized-v1`/`v2` result was measured under and must not
    move; `LAYOUT_V3` is what a `parameterized-v3` task encodes with.
    """

    name: str
    max_entities: int
    entity_features: int
    items: tuple[str, ...]
    goal_features: int
    #: Named-marker triples after the 12-slot v1 goal vector (`v3` only).
    marker_slots: int = 0


LAYOUT_V1 = TensorLayout(
    name="v1",
    max_entities=MAX_ENTITIES,
    entity_features=ENTITY_FEATURES,
    items=ITEMS,
    goal_features=GOAL_FEATURES,
)

#: The `v3` layout, for belt-and-inserter logistics.
#:
#: * 96 rows instead of 32: a belt line is one entity per tile, so a smelting
#:   line fed by a twenty-tile belt already overflows 32 rows before the
#:   machines at either end are counted.
#: * 32 features per row: the 16 of `v1`, unchanged, then what a player reads
#:   off a belt, an inserter and a drill (see `_logistics_features`).
#: * The `ITEMS_V3` inventory.
#: * A goal vector of the 12 `v1` slots followed by `MARKER_SLOTS_V3` triples
#:   `(dx, dy, present)`, one per public marker in the task's
#:   `public_markers` order, scaled by `MARKER_SCALE_V3` rather than the sensor
#:   radius. The `v1` slots can point at one marker, clipped at 32 tiles; a task
#:   whose ore, furnace site and output chest are sixty tiles apart needs all
#:   of them, and further away than that.
MAX_ENTITIES_V3 = 96
ENTITY_FEATURES_V3 = 32
MARKER_SLOTS_V3 = 6
MARKER_SCALE_V3 = 128.0
GOAL_FEATURES_V3 = GOAL_FEATURES + 3 * MARKER_SLOTS_V3

#: Items one belt lane is reported against. A straight lane holds four; the
#: outer lane of a turn a little more. Eight is never reached, so the feature is
#: an exact count / 8 rather than a saturating one.
LANE_CAP = 8.0
#: Pickup and drop offsets are within ~1.3 tiles of the entity; /2 keeps them
#: exact in float32 and well inside [-1, 1].
OFFSET_SCALE = 2.0

LAYOUT_V3 = TensorLayout(
    name="v3",
    max_entities=MAX_ENTITIES_V3,
    entity_features=ENTITY_FEATURES_V3,
    items=ITEMS_V3,
    goal_features=GOAL_FEATURES_V3,
    marker_slots=MARKER_SLOTS_V3,
)

LAYOUTS: dict[str, TensorLayout] = {"v1": LAYOUT_V1, "v3": LAYOUT_V3}


@dataclass(frozen=True)
class ObservationProfile:
    name: str
    version: int
    radius: int = 32
    #: World tiles per grid cell. The wire carries a sparse tile list, so the
    #: raster resolution costs nothing on the wire and is a pure Python-side
    #: choice -- a coarser grid can be A/B'd against `local-v1` without
    #: touching the mod or the sensor contract.
    cell_size: int = 1

    @property
    def grid_size(self) -> int:
        return 2 * self.radius // self.cell_size + 1


LOCAL_V1 = ObservationProfile(name="local-v1", version=1, radius=32)

#: Same sensor radius, half the resolution (33x33 instead of 65x65). Shares the
#: `local-v1` name and version because the wire request is byte-identical; only
#: the tensor layout differs, and that difference is carried by
#: `policy.EXTRACTOR_VERSION`.
LOCAL_V1_COARSE = ObservationProfile(name="local-v1", version=1, radius=32, cell_size=2)


def observation_space(
    profile: ObservationProfile = LOCAL_V1, layout: TensorLayout = LAYOUT_V1
) -> spaces.Dict:
    rows, features = layout.max_entities, layout.entity_features
    return spaces.Dict(
        {
            "grid": spaces.Box(
                0.0, 1.0, (len(RESOURCES) + 2, profile.grid_size, profile.grid_size), np.float32
            ),
            "entities": spaces.Box(-1.0, 1.0, (rows, features), np.float32),
            "entity_mask": spaces.Box(0, 1, (rows,), np.int8),
            "self": spaces.Box(-1.0, 1.0, (SELF_FEATURES,), np.float32),
            "inventory": spaces.Box(0.0, 1.0, (len(layout.items),), np.float32),
            "goal": spaces.Box(-1.0, 1.0, (layout.goal_features,), np.float32),
        }
    )


def _norm(value: float, scale: float) -> float:
    return float(np.clip(value / scale, -1.0, 1.0))


def _log_count(count: float, cap: float = 200.0) -> float:
    return float(np.clip(np.log1p(max(count, 0.0)) / np.log1p(cap), 0.0, 1.0))


def _entity_type_index(entity_type: str) -> int:
    try:
        return ENTITY_TYPES.index(entity_type)
    except ValueError:
        return len(ENTITY_TYPES) - 1


def _item_slot(item, items: tuple[str, ...]) -> float:
    """`(index + 1) / len(items)`, or 0 for no item or one outside the vocabulary."""
    if item in items:
        return (items.index(item) + 1) / len(items)
    return 0.0


def dominant_item(contents: dict | None, items: tuple[str, ...]):
    """The item a record holds most of, among `items`; ties go to the lower index."""
    best, best_count = None, 0
    for item in items:
        count = (contents or {}).get(item, 0) or 0
        if count > best_count:
            best, best_count = item, count
    return best


def _logistics_features(record: dict, position, features: np.ndarray, items) -> None:
    """Slots 16-31 of a `v3` row: what a player reads off belts, inserters, drills.

    Every value comes from the record's own wire fields (`local-v3`'s
    `logistics_detail`); a record without them -- any remembered one, or any
    observation under an older profile -- leaves the slots zero, exactly as an
    empty belt or an empty hand would.

    16-17  items on lane 1 (left of travel) and lane 2, / `LANE_CAP`
    18-19  belt turns left, turns right (both 0: straight, or not a belt)
    20-21  the inserter's hand holds something; which item, `(i + 1) / n`
    22-24  pickup point less the entity's position, / `OFFSET_SCALE`; present
    25-27  drop point likewise (inserters and drills); present
    28     the item the record's `contents` holds most of, `(i + 1) / n`
    29     power satisfaction (Stage 2; nothing publishes it yet)
    30-31  reserved
    """
    lanes = record.get("lanes")
    if lanes:
        features[16] = min(float(lanes[0] or 0), LANE_CAP) / LANE_CAP
        features[17] = min(float(lanes[1] or 0), LANE_CAP) / LANE_CAP
    shape = record.get("shape")
    features[18] = 1.0 if shape == "left" else 0.0
    features[19] = 1.0 if shape == "right" else 0.0
    held = record.get("held")
    if held:
        features[20] = 1.0
        features[21] = _item_slot(held, items)
    for key, base in (("pickup", 22), ("drop", 25)):
        point = record.get(key)
        if point:
            features[base] = _norm(point[0] - position[0], OFFSET_SCALE)
            features[base + 1] = _norm(point[1] - position[1], OFFSET_SCALE)
            features[base + 2] = 1.0
    features[28] = _item_slot(dominant_item(record.get("contents"), items), items)
    power = record.get("power")
    if power is not None:
        features[29] = float(np.clip(float(power), 0.0, 1.0))


def encode(
    observation: dict,
    goal: np.ndarray | None = None,
    profile: ObservationProfile = LOCAL_V1,
    layout: TensorLayout = LAYOUT_V1,
) -> dict[str, np.ndarray]:
    """Turn one wire observation into fixed-shape tensors."""
    size = profile.grid_size
    grid = np.zeros((len(RESOURCES) + 2, size, size), dtype=np.float32)
    origin = (observation.get("sensor") or {}).get("origin", [0.0, 0.0])

    span = 2 * profile.radius

    def to_cell(position) -> tuple[int, int] | None:
        col = int(round(position[0] - origin[0])) + profile.radius
        row = int(round(position[1] - origin[1])) + profile.radius
        # Bounds are checked in tiles rather than in cells so that the covered
        # area is exactly the sensor radius at every `cell_size`. Out-of-range
        # positions are dropped, not clamped: a tile the sensor does not cover
        # must not be painted onto the grid edge, where the policy would read it
        # as an adjacent obstacle or ore patch.
        if 0 <= row <= span and 0 <= col <= span:
            return row // profile.cell_size, col // profile.cell_size
        return None

    # Resource planes, plus an amount plane and an obstacle plane.
    for tile in (observation.get("resources") or {}).get("tiles", []):
        cell = to_cell(tile.get("p") or [0.0, 0.0])
        if cell is None:
            continue
        row, col = cell
        # `.get`, like every other field here. An unnamed or position-less tile
        # is a decoder input to tolerate, not a crash; the mod always sends both.
        name = tile.get("name")
        if name in RESOURCES:
            grid[RESOURCES.index(name), row, col] = 1.0
        # Max, not last-write-wins: at `cell_size > 1` several tiles share a
        # cell, and assignment would make the amount plane depend on the order
        # the mod happened to serialise the tile list -- the same scene would
        # encode two different ways between steps. Max is order-independent and
        # identical to assignment at `cell_size == 1`.
        amount = _log_count(tile.get("amount", 0), 4000.0)
        grid[len(RESOURCES), row, col] = max(grid[len(RESOURCES), row, col], amount)
    for blocked in (observation.get("terrain") or {}).get("blocked", []):
        cell = to_cell(blocked)
        if cell is not None:
            grid[len(RESOURCES) + 1, cell[0], cell[1]] = 1.0

    entities = np.zeros((layout.max_entities, layout.entity_features), dtype=np.float32)
    mask = np.zeros((layout.max_entities,), dtype=np.int8)

    candidates = entity_row_order(observation, origin, layout.max_entities)

    for index, (distance, record, remembered) in enumerate(candidates):
        position = record.get("p", [0, 0])
        features = np.zeros(layout.entity_features, dtype=np.float32)
        features[0] = _norm(position[0] - origin[0], profile.radius)
        features[1] = _norm(position[1] - origin[1], profile.radius)
        features[2] = _norm(distance, profile.radius)
        type_index = _entity_type_index(record.get("type", "other"))
        features[3] = type_index / max(len(ENTITY_TYPES) - 1, 1)
        features[4] = (record.get("d", 0) or 0) / DIRECTION_COUNT
        contents = record.get("contents") or {}
        features[5] = _log_count(sum(contents.values()))
        features[6] = 1.0 if record.get("recipe") else 0.0
        features[7] = 1.0 if record.get("status") is not None else 0.0
        features[8] = float(record.get("health", 1.0))
        # The two features that make remembered observations usable rather than
        # merely present.
        features[9] = 1.0 if remembered else 0.0
        features[10] = _log_count(record.get("age", 0), 3600.0) if remembered else 0.0
        # Slots 11-15 were structurally zero in every observation of every
        # task. They now carry what a player reads off a stopped machine.
        status_name = record.get("st")
        if status_name in ENTITY_STATUS:
            features[11] = (ENTITY_STATUS.index(status_name) + 1) / len(ENTITY_STATUS)
        working = record.get("working")
        features[12] = 1.0 if working else 0.0
        # Distinguishes "not working" from "this type has no status at all",
        # which the absent-field encoding conflated.
        features[13] = 1.0 if working is not None else 0.0
        features[14] = _log_count(sum((record.get("fuel") or {}).values()))
        features[15] = _log_count(sum((record.get("output") or {}).values()))
        if layout.entity_features > ENTITY_FEATURES:
            _logistics_features(record, position, features, layout.items)
        entities[index] = features
        mask[index] = 1

    character = observation.get("character") or {}
    position = character.get("position", [0, 0])
    self_vector = np.zeros(SELF_FEATURES, dtype=np.float32)
    self_vector[0] = _norm(position[0], 128.0)
    self_vector[1] = _norm(position[1], 128.0)
    self_vector[2] = 1.0 if character.get("walking") else 0.0
    self_vector[3] = (character.get("direction") or 0) / DIRECTION_COUNT
    mining = character.get("mining") or {}
    self_vector[4] = 1.0 if mining.get("active") else 0.0
    self_vector[5] = float(mining.get("progress") or 0.0)
    crafting = character.get("crafting") or {}
    self_vector[6] = 1.0 if crafting.get("queue") else 0.0
    self_vector[7] = float(crafting.get("progress") or 0.0)
    self_vector[8] = 1.0 if observation.get("inflight") else 0.0
    # Whether the last settled action was refused, and how often refusals have
    # been happening. Derived from the observation's own `events` log, so this
    # stays a pure function of what the policy can see. Before `events` was
    # published there was no channel at all: a refused placement, an
    # out-of-reach transfer and an empty inventory were four distinct codes at
    # the engine and zero bits in the policy input.
    events = [e for e in (observation.get("events") or []) if isinstance(e, dict)]
    settled = [e for e in events if e.get("status") in _SETTLED_STATUS]
    if settled:
        last = settled[-1]
        self_vector[9] = 1.0 if last.get("status") in _REFUSED_STATUS else 0.0
        self_vector[10] = 1.0 if last.get("status") == "completed" else 0.0
        refused = sum(1 for e in settled if e.get("status") in _REFUSED_STATUS)
        self_vector[11] = refused / len(settled)
    # A profile that publishes only the last few events (`local-v2` v7,
    # `open-v1` v3) sends the counts over the full buffer alongside, so the
    # rate means what it meant when every event was sent.
    counts = observation.get("event_counts")
    if isinstance(counts, dict) and counts.get("settled"):
        self_vector[11] = float(counts.get("refused") or 0) / float(counts["settled"])

    inventory = np.zeros(len(layout.items), dtype=np.float32)
    for index, item in enumerate(layout.items):
        inventory[index] = _log_count((observation.get("inventory") or {}).get(item, 0))

    return {
        "grid": grid,
        "entities": entities,
        "entity_mask": mask,
        "self": self_vector,
        "inventory": inventory,
        "goal": (
            goal.astype(np.float32)
            if goal is not None
            else np.zeros(layout.goal_features, dtype=np.float32)
        ),
    }
