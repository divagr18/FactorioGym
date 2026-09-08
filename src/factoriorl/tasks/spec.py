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


#: Tile footprints, keyed by prototype. Read off a real engine's bounding
#: boxes, not guessed: `test_entity_footprints` pins the table, and a prototype
#: absent from it is treated as 1x1.
#:
#: This lived in `tools/generator_diagnostics.py`, where the reachability BFS
#: needed it, while `footprint_conflicts` compared one rounded centre tile per
#: entity -- so the validator that was supposed to catch overlapping machines
#: could not see a 2x2 at all.
ENTITY_TILE_SIZES: dict[str, tuple[int, int]] = {
    "burner-mining-drill": (2, 2),
    "electric-mining-drill": (3, 3),
    "solar-panel": (3, 3),
    "stone-furnace": (2, 2),
}


def entity_tiles(name: str, position: tuple[float, float]) -> list[tuple[int, int]]:
    """The integer tiles an entity occupies, from its prototype footprint.

    `floor(p - size/2 + 0.5)` rather than `round()`: Python's `round()` is
    banker's rounding, which sends a footprint whose left edge lands on .5 to
    the wrong tile half the time. `plate_line` documents the same trap for a
    column of walls spaced on .5 boundaries.
    """
    width, height = ENTITY_TILE_SIZES.get(name, (1, 1))
    x0 = math.floor(position[0] - width / 2 + 0.5)
    y0 = math.floor(position[1] - height / 2 + 0.5)
    return [(x0 + dx, y0 + dy) for dx in range(width) for dy in range(height)]


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
    #: Recipes this scenario unlocks. Declared, not implied: the action
    #: profile's placement guard refuses an item whose recipe is locked, so a
    #: task handing out such an item must say so.
    unlock_recipes: tuple[str, ...] = ()
    radius: int = 64

    def to_dict(
        self,
        public_markers: tuple[str, ...] = (),
        extra_tracked_items: tuple[str, ...] = (),
    ) -> dict:
        payload = {
            "public_markers": list(public_markers),
            "entities": [e.to_dict() for e in self.entities],
            "resources": [r.to_dict() for r in self.resources],
            "character": {
                "position": list(self.character_position),
                "inventory": self.character_inventory,
            },
            "markers": {k: list(v) for k, v in self.markers.items()},
            "unlock_recipes": list(self.unlock_recipes),
            "radius": self.radius,
        }
        # Emitted only when the task needs an item the mod does not already
        # count. Adding the key unconditionally would move every blueprint
        # digest and so invalidate every frozen holdout entry -- and with it
        # every result measured against one.
        if extra_tracked_items:
            payload["extra_tracked_items"] = sorted(extra_tracked_items)
        return payload

    # ---- engine-free structural validation (PLAN.md 3.2 solvability) -------

    def footprint_conflicts(self) -> list[str]:
        """Two entities sharing a tile is a generator bug.

        Over real footprints, not centre tiles. The previous version compared
        `round()` of one centre per entity, so two adjacent 2x2 machines whose
        footprints overlap passed validation -- the exact geometry
        `plate_line`'s docstring reasons about by hand, and the geometry
        `build_line` had to measure on an engine because nothing checked it.
        """
        seen: dict[tuple[int, int], str] = {}
        problems = []
        for entity in self.entities:
            for tile in entity_tiles(entity.name, entity.position):
                if tile in seen and seen[tile] != entity.name:
                    problems.append(f"{entity.name} overlaps {seen[tile]} at {tile}")
                elif tile in seen:
                    problems.append(f"two {entity.name} overlap at {tile}")
                seen[tile] = entity.name
        return problems

    def initial_state(self) -> tuple[dict, dict]:
        """A synthetic `(observation, truth)` for the scene at reset.

        Enough for `Predicate.evaluate` to decide the kinds a blueprint fully
        determines, so "the success predicate must be False at reset" can be
        checked without an engine. The cumulative channels are empty by
        construction rather than by assumption: `world.clear_statistics` clears
        production and the placement counter at every episode start, and only
        the `place` action increments the latter.
        """
        markers = {name: list(position) for name, position in self.markers.items()}
        containers: dict[str, dict] = {}
        placed: dict[str, int] = {}
        entities = []
        for entity in self.entities:
            placed[entity.name] = placed.get(entity.name, 0) + 1
            entities.append({"name": entity.name, "p": list(entity.position), "type": entity.name})
            if entity.marker:
                markers[entity.marker] = list(entity.position)
                containers[entity.marker] = dict(entity.contents)
        observation = {
            "tick": 0,
            "character": {"position": list(self.character_position)},
            "inventory": dict(self.character_inventory),
            "entities": entities,
            "resources": {
                "tiles": [{"name": r.name, "p": list(r.position)} for r in self.resources]
            },
            "terrain": {"blocked": []},
            "inflight": [],
        }
        truth = {
            "markers": markers,
            "containers": containers,
            "placed_counts": placed,
            "produced": {},
            "built": {},
            "window": [],
            # Undecidable without an engine; see `UNDECIDABLE_AT_RESET`. Left
            # empty so a predicate reading them is False, which is why a task
            # relying on one is *reported* rather than quietly passed.
            "working": {},
            "working_counts": {},
        }
        return observation, truth

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
    #: Counts entities of a prototype the *agent* built, from placements the
    #: `place` handler recorded. `ENTITY_WORKING` keys on a scene alias bound at
    #: install, so it cannot name a machine the agent placed; this can. Not the
    #: engine's build statistics: `create_entity` does not feed those, measured
    #: as `built = {}` beside a running line and 49 plates.
    BUILT = "built"
    #: At least `at_least` entities of a prototype currently working, whoever
    #: placed them. Prototype-keyed for the same reason.
    ANY_WORKING = "any_working"
    #: Production *within a window*, not since reset. `PRODUCED` is a monotone
    #: counter with no timestamp, so it cannot tell "30 plates, the last 15 in
    #: the final window" from "30 plates, line dead since tick 4000" -- which is
    #: exactly the distinction a sustained-operation objective is made of.
    SUSTAINED_OUTPUT = "sustained_output"


#: Items the mod counts in `truth["produced"]` unconditionally. Mirrors the
#: literal in `mod/factoriorl/world.lua`; `test_tracked_items` asserts they
#: agree, because a silent drift here means a task's output is invisible to its
#: own success predicate.
DEFAULT_TRACKED_ITEMS: frozenset[str] = frozenset(
    {
        "iron-plate",
        "copper-plate",
        "stone-furnace",
        "iron-gear-wheel",
        "iron-ore",
        "copper-ore",
        "coal",
        "stone",
    }
)


#: Predicate kinds a blueprint cannot decide. Whether a machine is *working*
#: depends on fuel, power networks and heat -- engine state, not scene
#: declaration -- so `Blueprint.initial_state` leaves those channels empty and
#: `validate_all` reports the gap instead of reading the empty channel as a
#: passing check.
UNDECIDABLE_AT_RESET: frozenset = frozenset(
    {PredicateKind.ENTITY_WORKING, PredicateKind.ANY_WORKING}
)


#: The predicate kinds whose measurement comes from `truth["produced"]`. Kept
#: beside the enum so a new production-shaped kind is added in one place.
PRODUCTION_KINDS: frozenset[PredicateKind] = frozenset(
    {PredicateKind.PRODUCED, PredicateKind.SUSTAINED_OUTPUT}
)


@dataclass(frozen=True)
class Predicate:
    kind: PredicateKind
    #: Marker name for spatial/entity predicates.
    marker: str | None = None
    item: str | None = None
    at_least: float = 1.0
    within: float = 1.5
    #: Window for `SUSTAINED_OUTPUT`, in game ticks. 3600 is one minute at 60
    #: UPS, and the measured commissioning baseline is 15 plates per 3600.
    over_ticks: int = 3600

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
        if self.kind is PredicateKind.BUILT:
            return (truth.get("built") or {}).get(self.item, 0) >= self.at_least
        if self.kind is PredicateKind.ANY_WORKING:
            return (truth.get("working_counts") or {}).get(self.item, 0) >= self.at_least
        if self.kind is PredicateKind.SUSTAINED_OUTPUT:
            # A full window must have elapsed. Without this the predicate
            # degenerates into `PRODUCED` for the first `over_ticks` of every
            # episode, because the baseline falls back to the earliest sample:
            # measured on a real engine, `build_line` passed at tick 2850 with
            # 10 plates *in total*, which a line that then died would also have
            # produced. That is exactly the case a sustained objective exists
            # to reject.
            if not self.window_elapsed(truth):
                return False
            return self.window_output(truth) >= self.at_least
        return False

    def window_elapsed(self, truth: dict) -> bool:
        """Whether the history spans at least `over_ticks`."""
        history = truth.get("window") or []
        if len(history) < 2:
            return False
        return (history[-1][0] - history[0][0]) >= self.over_ticks

    def window_output(self, truth: dict) -> float:
        """Production of `item` inside the last `over_ticks`.

        Reads `truth["window"]`, an ordered `[(tick, {item: cumulative})]`
        history the environment maintains. Kept as an argument rather than
        state on the predicate so `evaluate` stays a pure function of its
        inputs and is testable without an engine.

        The oldest sample at or before the window's start is the baseline. With
        no such sample the episode is younger than the window and the earliest
        sample is used, which makes this a *lower bound* on the rate rather
        than the rate -- so `evaluate` additionally refuses to answer at all
        until `window_elapsed`, or the predicate would mean "produced this
        much" for the first `over_ticks` of every episode.
        """
        history = truth.get("window") or []
        if not history:
            return 0.0
        latest_tick, latest = history[-1]
        cutoff = latest_tick - self.over_ticks
        baseline = history[0][1]
        for tick, counts in history:
            if tick <= cutoff:
                baseline = counts
            else:
                break
        return max(
            0.0,
            float(latest.get(self.item, 0)) - float(baseline.get(self.item, 0)),
        )


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
    #: Cumulative ceiling for this component over one episode. Required for
    #: HIGH_WATER, whose driving measurement is usually unbounded: mining ore
    #: pays per ore, so a policy that mines and never smelts could out-earn
    #: finishing the task several times over. Capping the total keeps the
    #: subgoal a hint rather than a substitute goal.
    cap: float | None = None


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
    #: Discount for this family, for both the learner and the shaping. Declared
    #: per task because the useful horizon is `1/(1-gamma)` and it has to cover
    #: the episode: at 0.99 that is 100 steps, and a 300-step family shaping
    #: toward a goal pays more drag for approaching than the approach is worth,
    #: whatever weight the component carries. `None` takes the trainer default.
    gamma: float | None = None
    #: Markers published beyond those a success predicate names.
    #:
    #: The default rule -- publish what the objective names -- covers a task
    #: whose goal *is* a place, like `deliver`'s destination. It does not cover
    #: one whose goal is a state and whose *work* happens somewhere else:
    #: `repair_belt` succeeds on a plate in the sink and the plate only moves if
    #: a belt is placed in the gap, so the gap is where the agent must act and
    #: the sink is merely where the result is counted.
    #:
    #: Measured consequence of leaving it out: the `toward_gap` potential paid
    #: for approaching (9, -4) while the only position in the observation was
    #: the sink at (12.5, -3.5), and the gap existed solely as a *missing*
    #: element of an unordered 32-entity list. `deliver` scored 0.35 in exactly
    #: that condition and 0.92 once its target was published as a coordinate.
    #:
    #: This cannot inflate a benchmark: a random policy does not read
    #: observations, so every measured floor is unchanged.
    extra_public_markers: tuple[str, ...] = ()
    #: How the goal vector's three geometry slots choose among several declared
    #: faults. `"static"` always points at `focus_marker`. `"nearest_unrepaired"`
    #: lets the environment pick the closest fault with nothing standing on it
    #: and retarget as faults close -- which resolves the fault-selection
    #: sub-problem on the agent's behalf, so it is an assistance and is recorded
    #: as one in the manifest. It reads occupancy from the observation's own
    #: entity list, never from truth.
    focus_policy: str = "static"
    decision_ticks: int = 30
    #: local-v2 by default: it omits the blocks nothing reads and caps the
    #: entity list at 48 after a distance sort, which is 2.6-5.6x less JSON per
    #: step. A reconstruction run over all six families confirms it decodes to
    #: byte-identical tensors, so the saving is free rather than a trade.
    observation_profile: str = "local-v2"
    action_profile: str = "primitive-v1"
    catalog: str = "primitive-v1"
    catalog_subset: tuple[str, ...] = ()
    failure: tuple[Predicate, ...] = ()
    #: Observable subgoals, in the order they are expected to become true.
    #: The goal vector currently spends most of its slots on success
    #: predicates that are all false until the moment the episode ends, which
    #: gives a long-horizon policy nothing to orient by. Landmarks are the
    #: intermediate conditions a solution passes through, and they must be
    #: computable from the declared observation alone -- a landmark that reads
    #: evaluator truth would put privileged information into the policy input,
    #: which section 2 forbids. `test_landmarks_do_not_read_truth` enforces it.
    landmarks: tuple[Predicate, ...] = ()
    #: What must hold before this task can be attempted, and what it leaves
    #: true when it succeeds. Nothing consumes these yet; they are the
    #: vocabulary a refinement layer needs in order to chain tasks, and
    #: declaring them alongside the task is far cheaper than reconstructing
    #: them later from generators and predicates.
    preconditions: tuple[Predicate, ...] = ()
    provides: tuple[Predicate, ...] = ()
    #: Names the deliberation layer that chooses actions for this task:
    #: `flat-v1` is a policy over primitive actions. A skill layer is a
    #: different profile and must be distinguishable in a published result.
    deliberation_profile: str = "flat-v1"
    #: The blueprint marker that difficulty is measured *to*, naming the far end
    #: of the route a solution must walk. Declared here rather than inferred,
    #: because the two are only accidentally the same thing.
    #:
    #: For `navigate` and `deliver` the success predicate names a position and
    #: that position *is* the goal, so reading the marker off the predicate is
    #: correct. For a task whose success is a production count -- `mine_smelt`
    #: and `supply_furnace` succeed on a force production statistic -- the route
    #: that determines difficulty is still perfectly well defined (spawn to the
    #: ore patch and back to the furnace; spawn to the ore chest and on to the
    #: furnace) but nothing in the predicate says so. Without this field
    #: `tools/generator_diagnostics.py` had no position to measure a route to and
    #: reported both families' difficulty parity as unmeasurable, which PLAN.md
    #: section 3 requires to be published before a held-out score may be called a
    #: transfer score.
    #:
    #: The alternative -- having the tool pick whichever marker looks
    #: goal-shaped -- is what this field exists to prevent. `mine_smelt` declares
    #: both `patch` and `furnace`, and the furnace is pinned to a fixed radius
    #: from spawn, its distance spanning about one tile of integer-rounding jitter
    #: against the patch's six. A guess that landed on it would hand back a parity
    #: near 1.00 that had compared rounding noise, not the task. Declaring the
    #: marker makes the choice reviewable instead of accidental.
    #:
    #: This is metadata about how the task is *measured*, not about what the task
    #: *is*: it changes no scene, no predicate and no budget, so setting it does
    #: not warrant a `version` bump. For the same reason it is deliberately
    #: absent from `to_dict()` -- see the note there.
    difficulty_marker: str | None = None

    def families(self, split: str) -> tuple[LayoutFamily, ...]:
        return tuple(f for f in self.layout_families if f.split == split)

    @property
    def extra_tracked_items(self) -> tuple[str, ...]:
        """Items this task measures that the mod does not count by default.

        `truth["produced"]` was a fixed list of eight, so a task whose
        objective named anything else would read zero from its own success
        predicate -- silently, with no error anywhere. Derived from the
        predicates rather than declared, so it cannot fall out of step with
        what the task actually scores.

        Only the kinds that actually read `produced` count. `BUILT` reads
        `truth["built"]` and `CONTAINER_HOLDS` reads `truth["containers"]`, so
        collecting their items would ask the mod to count production of a
        machine nobody smelts -- and, because this argument is part of the
        installed payload, would move that task's blueprint digest for no
        reason at all.
        """
        items: set[str] = set()
        predicates = list(self.success)
        predicates += [r.predicate for r in self.rewards if r.predicate is not None]
        for predicate in predicates:
            if predicate.item and predicate.kind in PRODUCTION_KINDS:
                items.add(predicate.item)
        return tuple(sorted(items - DEFAULT_TRACKED_ITEMS))

    @property
    def public_markers(self) -> tuple[str, ...]:
        """Markers the objective names, and which the agent may therefore see.

        A benchmark whose objective is unstated is not testing generalisation,
        it is testing guessing. `deliver` puts four identical containers in
        every scene and scores exactly one of them, and nothing in the
        observation said which: a language model spent whole episodes
        delivering plates and taking them back to find out, and a trained
        policy could only learn the generator's placement habits. `navigate`'s
        goal was unpublished too -- it was merely the one non-wall entity in
        the scene, which is why a single skill solved it 5/5.

        Two sources, and no others. A marker a success or failure predicate
        references, because that is what the agent is scored on; and a marker
        the task declares in `extra_public_markers`, because a goal that *is* a
        state has its work somewhere else -- `repair_belt` succeeds on a plate
        in the sink, and the plate only moves if a belt goes in the gap.

        Decoys stay hidden either way: `deliver`'s `decoy_0` and `decoy_1` are
        in neither source, so the task still asks the agent to reach the right
        container rather than handing it the scene.

        This raises no random floor -- a uniform policy cannot read an
        observation -- so every floor measured against the old contract stays
        comparable to one measured against this one.
        """
        names: list[str] = []
        for predicate in (*self.success, *self.failure):
            name = getattr(predicate, "marker", None)
            if name and name not in names:
                names.append(name)
        for name in self.extra_public_markers:
            if name not in names:
                names.append(name)
        return tuple(names)

    @property
    def focus_marker(self) -> str | None:
        """The marker the goal vector's geometry points at by default.

        Under `focus_policy = "nearest_unrepaired"` this is only the fallback:
        the environment selects among `extra_public_markers` at runtime. A
        manifest reporting this field alone therefore understates what the
        policy received, which is why `focus_policy` is recorded beside it.

        Where the agent must *act*, which is not always what it is scored on.
        A declared `extra_public_markers` entry exists precisely because those
        two came apart, so it wins; otherwise the objective's own marker.

        Chosen explicitly because the alternative was alphabetical: the goal
        vector used to take the first published marker by sort order, which gave
        `repair_belt` the gap by luck and `restore_power` the drill -- the place
        its success is *measured*, four poles away from the place it has to put
        one.
        """
        if self.extra_public_markers:
            return self.extra_public_markers[0]
        for predicate in (*self.success, *self.failure):
            name = getattr(predicate, "marker", None)
            if name:
                return name
        return None

    def to_dict(self) -> dict:
        # `difficulty_marker` is deliberately not published here. This dict is
        # what `learn/train.py` hands to `manifest.config_digest`, so every key
        # in it is part of the identity a run is compared by. Adding a key that
        # describes only how the split audit measures the task would change that
        # digest for `mine_smelt` and `supply_furnace` without changing the task,
        # and two runs of a byte-identical task would then read as runs of
        # different tasks. The marker is published where it is actually used, in
        # the generator-diagnostics report.
        return {
            "id": self.id,
            "version": self.version,
            "description": self.description,
            "decision_ticks": self.decision_ticks,
            "gamma": self.gamma,
            "budgets": {
                "max_decision_steps": self.max_decision_steps,
                "max_game_ticks": self.max_game_ticks,
            },
            # What the policy can *see* of the objective, and the marker the
            # goal vector's geometry points at. Both were absent from this
            # dict, so a run's manifest could not distinguish a task whose
            # target was published from one whose target was evaluator-only --
            # and "the fix was never trained" was argued from their absence,
            # off a field that was simply never recorded. They belong in the
            # digest: publishing a marker changes the observation, so two runs
            # differing in it are runs of different tasks.
            "extra_public_markers": list(self.extra_public_markers),
            "focus_marker": self.focus_marker,
            "focus_policy": self.focus_policy,
            "observation_profile": self.observation_profile,
            "action_profile": self.action_profile,
            "deliberation_profile": self.deliberation_profile,
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
