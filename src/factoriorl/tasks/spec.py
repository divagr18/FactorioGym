"""Task specifications (PLAN.md 3.1).

A task declares its scene, budgets, success and failure predicates, reward
components and action catalog. Nothing here imports worker management -- an
import-direction test asserts that, so "a task can be added without editing
worker management" is provable rather than promised.

Predicates and reward components come from a small **declarative vocabulary**
rather than arbitrary callables, so they can be validated at load time and
enumerated by name for per-component logging. That is what lets
``factoriorl tasks validate`` fail before a worker is ever launched instead of
at step 40,000.

Scenes are **blueprints built programmatically**, not map generation. Map-gen
settings are written at worker launch and consumed by ``--create``, so a
map-gen-driven task would need an engine relaunch to switch or randomise --
which would put scenario selection squarely inside worker management. A
blueprint also resets in milliseconds, which is what PLAN 3.3's 500 consecutive
resets actually permits, and it is something a solvability check can reason
about: "a 4x4 iron patch 18 tiles out with a walkable route" is a question you
can ask a declaration and cannot ask Perlin noise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

# --------------------------------------------------------------------- scene


@dataclass(frozen=True)
class EntitySpec:
    name: str
    position: tuple[float, float]
    direction: str = "north"
    force: str = "player"
    contents: dict[str, int] = field(default_factory=dict)
    recipe: str | None = None
    marker: str | None = None

    def to_dict(self) -> dict:
        body: dict[str, Any] = {
            "name": self.name,
            "position": list(self.position),
            "direction": self.direction,
            "force": self.force,
        }
        if self.contents:
            body["contents"] = self.contents
        if self.recipe:
            body["recipe"] = self.recipe
        if self.marker:
            body["marker"] = self.marker
        return body


@dataclass(frozen=True)
class ResourceSpec:
    name: str
    position: tuple[float, float]
    amount: int = 1000

    def to_dict(self) -> dict:
        return {"name": self.name, "position": list(self.position), "amount": self.amount}


@dataclass(frozen=True)
class Blueprint:
    """A complete, deterministic description of a scene."""

    entities: tuple[EntitySpec, ...] = ()
    resources: tuple[ResourceSpec, ...] = ()
    character_position: tuple[float, float] = (0.0, 0.0)
    character_inventory: dict[str, int] = field(default_factory=dict)
    markers: dict[str, tuple[float, float]] = field(default_factory=dict)
    radius: int = 64

    def to_dict(self) -> dict:
        return {
            "entities": [e.to_dict() for e in self.entities],
            "resources": [r.to_dict() for r in self.resources],
            "character": {
                "position": list(self.character_position),
                "inventory": self.character_inventory,
            },
            "markers": {k: list(v) for k, v in self.markers.items()},
            "radius": self.radius,
        }

    # ---- engine-free structural validation (PLAN.md 3.2 solvability) -------

    def footprint_conflicts(self) -> list[str]:
        """Two entities declared on the same tile is a generator bug."""
        seen: dict[tuple[int, int], str] = {}
        problems = []
        for entity in self.entities:
            key = (round(entity.position[0]), round(entity.position[1]))
            if key in seen:
                problems.append(f"{entity.name} overlaps {seen[key]} at {key}")
            seen[key] = entity.name
        return problems

    def out_of_box(self) -> list[str]:
        problems = []
        for item in (*self.entities, *self.resources):
            x, y = item.position
            if abs(x) > self.radius or abs(y) > self.radius:
                problems.append(f"{item.name} at {item.position} is outside the scene box")
        return problems

    def distance_from_character(self, position) -> float:
        return math.dist(self.character_position, position)


# ----------------------------------------------------------------- predicates


class PredicateKind(StrEnum):
    """The closed vocabulary. Anything else fails validation at load time."""

    CHARACTER_WITHIN = "character_within"
    CONTAINER_HOLDS = "container_holds"
    INVENTORY_HOLDS = "inventory_holds"
    PRODUCED = "produced"
    ENTITY_WORKING = "entity_working"


@dataclass(frozen=True)
class Predicate:
    kind: PredicateKind
    #: Marker name for spatial/entity predicates.
    marker: str | None = None
    item: str | None = None
    at_least: float = 1.0
    within: float = 1.5

    def describe(self) -> str:
        return f"{self.kind.value}({self.marker or ''} {self.item or ''} >= {self.at_least})"

    def evaluate(self, observation: dict, truth: dict) -> bool:
        if self.kind is PredicateKind.CHARACTER_WITHIN:
            target = (truth.get("markers") or {}).get(self.marker)
            position = (observation.get("character") or {}).get("position")
            if not target or not position:
                return False
            return math.dist(position, target) <= self.within
        if self.kind is PredicateKind.CONTAINER_HOLDS:
            contents = (truth.get("containers") or {}).get(self.marker, {})
            return contents.get(self.item, 0) >= self.at_least
        if self.kind is PredicateKind.INVENTORY_HOLDS:
            return (observation.get("inventory") or {}).get(self.item, 0) >= self.at_least
        if self.kind is PredicateKind.PRODUCED:
            # Force production statistics, not an inventory delta: items that
            # already existed at reset can never count as produced.
            return (truth.get("produced") or {}).get(self.item, 0) >= self.at_least
        if self.kind is PredicateKind.ENTITY_WORKING:
            return bool((truth.get("working") or {}).get(self.marker))
        return False


# -------------------------------------------------------------------- rewards


class RewardKind(StrEnum):
    SPARSE_SUCCESS = "sparse_success"
    #: Monotone: reward the increase of a high-water mark, never the level.
    #: Moving items out of the goal and back cannot pay twice.
    HIGH_WATER = "high_water"
    #: Potential-based: gamma*phi(s') - phi(s). Any closed loop in state space
    #: sums to zero, which kills dismantle/rebuild farming.
    POTENTIAL = "potential"
    STEP_COST = "step_cost"


@dataclass(frozen=True)
class RewardComponent:
    name: str
    kind: RewardKind
    weight: float = 1.0
    shaping: bool = True
    #: For HIGH_WATER/POTENTIAL: which measurement drives it.
    predicate: Predicate | None = None
    scale: float = 1.0


# ---------------------------------------------------------------------- task


@dataclass(frozen=True)
class LayoutFamily:
    """Structural variation, and the unit of held-out generalisation.

    PLAN.md section 3 asks for "unfamiliar seeds and unfamiliar structures"
    reported separately, so the split lives on the structure, not the seed.
    """

    name: str
    split: str = "train"


@dataclass(frozen=True)
class TaskSpec:
    id: str
    version: str
    description: str
    layout_families: tuple[LayoutFamily, ...]
    success: tuple[Predicate, ...]
    rewards: tuple[RewardComponent, ...]
    max_decision_steps: int = 600
    max_game_ticks: int = 36000
    decision_ticks: int = 30
    observation_profile: str = "local-v1"
    action_profile: str = "primitive-v1"
    catalog: str = "primitive-v1"
    catalog_subset: tuple[str, ...] = ()
    failure: tuple[Predicate, ...] = ()

    def families(self, split: str) -> tuple[LayoutFamily, ...]:
        return tuple(f for f in self.layout_families if f.split == split)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "version": self.version,
            "description": self.description,
            "decision_ticks": self.decision_ticks,
            "budgets": {
                "max_decision_steps": self.max_decision_steps,
                "max_game_ticks": self.max_game_ticks,
            },
            "observation_profile": self.observation_profile,
            "action_profile": self.action_profile,
            "catalog": self.catalog,
            "catalog_subset": list(self.catalog_subset),
            "layout_families": [{"name": f.name, "split": f.split} for f in self.layout_families],
            "success": [p.describe() for p in self.success],
            "failure": [p.describe() for p in self.failure],
            "rewards": [
                {
                    "name": r.name,
                    "kind": r.kind.value,
                    "weight": r.weight,
                    "shaping": r.shaping,
                }
                for r in self.rewards
            ],
        }
