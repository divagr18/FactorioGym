"""Observation encoding for compact policies (PLAN.md 3.4).

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
)

RESOURCES: tuple[str, ...] = ("iron-ore", "copper-ore", "coal", "stone")

#: Factorio 2.0 uses 16 compass directions (0..15), not 8.
DIRECTION_COUNT = 16.0

MAX_ENTITIES = 32
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


def observation_space(profile: ObservationProfile = LOCAL_V1) -> spaces.Dict:
    return spaces.Dict(
        {
            "grid": spaces.Box(
                0.0, 1.0, (len(RESOURCES) + 2, profile.grid_size, profile.grid_size), np.float32
            ),
            "entities": spaces.Box(-1.0, 1.0, (MAX_ENTITIES, ENTITY_FEATURES), np.float32),
            "entity_mask": spaces.Box(0, 1, (MAX_ENTITIES,), np.int8),
            "self": spaces.Box(-1.0, 1.0, (SELF_FEATURES,), np.float32),
            "inventory": spaces.Box(0.0, 1.0, (len(ITEMS),), np.float32),
            "goal": spaces.Box(-1.0, 1.0, (GOAL_FEATURES,), np.float32),
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


def encode(
    observation: dict,
    goal: np.ndarray | None = None,
    profile: ObservationProfile = LOCAL_V1,
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
        cell = to_cell(tile["p"])
        if cell is None:
            continue
        row, col = cell
        if tile["name"] in RESOURCES:
            grid[RESOURCES.index(tile["name"]), row, col] = 1.0
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

    entities = np.zeros((MAX_ENTITIES, ENTITY_FEATURES), dtype=np.float32)
    mask = np.zeros((MAX_ENTITIES,), dtype=np.int8)

    def entity_rows(source, remembered: bool):
        for record in source:
            position = record.get("p", [0, 0])
            yield (
                float(np.hypot(position[0] - origin[0], position[1] - origin[1])),
                record,
                remembered,
            )

    candidates = sorted(
        [
            *entity_rows(observation.get("entities", []), False),
            *entity_rows(observation.get("remembered", []), True),
        ],
        key=lambda row: row[0],
    )[:MAX_ENTITIES]

    for index, (distance, record, remembered) in enumerate(candidates):
        position = record.get("p", [0, 0])
        features = np.zeros(ENTITY_FEATURES, dtype=np.float32)
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

    inventory = np.zeros(len(ITEMS), dtype=np.float32)
    for index, item in enumerate(ITEMS):
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
            else np.zeros(GOAL_FEATURES, dtype=np.float32)
        ),
    }
