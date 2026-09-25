"""Task specifications (DESIGN.md 3.1).

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
blueprint also resets in milliseconds, which is what DESIGN 3.3's 500 consecutive
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


#: Disruption kinds the mod implements, mirroring `world.disrupt`.
#:
#: `empty_fuel` clears fuel *inventories* and deliberately leaves
#: `entity.energy` and `burner.remaining_burning_fuel` alone, because that is
#: the disruption R4.2 measured: a burner drill keeps producing for about 1,170
#: ticks on the item already in its burner. A task declaring it is declaring a
#: delayed outage, not an immediate one.
#:
#: `fill_output` fills a machine's result slot, which stops it for a reason no
#: amount of fuel fixes.
DISRUPTION_KINDS: frozenset[str] = frozenset({"empty_fuel", "fill_output"})


@dataclass(frozen=True)
class Disruption:
    """Something the evaluator does to a running scene, declared in advance.

    Declared on the task rather than fired by a driver, which is the whole
    point. `tools/demonstration.py` injected its outage with a raw Lua string
    at a fixed point in a five-phase script: the tick was a property of the
    driver, the targets were a property of the string, and a run's trace could
    not state either. Two runs of "the same disruption" were not comparable
    because nothing said what the disruption was.

    Unannounced by construction. Applying one publishes no marker and appends
    no event, so it is detectable only by monitoring what is already
    observable -- machine status, fuel and output contents. That is what R4.3's
    persistent-operation track asks for, and it is why `at_tick` is declared
    here rather than told to the agent.
    """

    kind: str
    #: Game tick at which it lands. Absolute, and compared against the
    #: observation's own tick, so a disruption cannot be missed by a step that
    #: advances past it.
    at_tick: int
    #: Marker aliases or prototype names. Resolved alias-first by the mod.
    targets: tuple[str, ...]

    def to_dict(self) -> dict:
        return {"kind": self.kind, "at_tick": self.at_tick, "targets": list(self.targets)}


#: Every track a task may declare. `validate_all` refuses anything else,
#: including the `"unclassified"` default, so a task that forgets to declare
#: one fails validation instead of being quietly filed under whichever name a
#: tool happened to guess.
#:
#: The first four are `tools/release_matrix.CATEGORY`'s names, kept exactly so
#: that moving the categories onto the specs changes where they are declared
#: and not what they mean. The last two are R4.3's new tracks:
#:
#: * `movement` -- get somewhere.
#: * `logistics` -- move items between places.
#: * `production` -- make something, including building or commissioning the
#:   line that makes it.
#: * `repair` -- fix a declared fault whose location is *published*.
#: * `diagnosis` -- the same repair with the fault's location withheld.
#:   Finding it is the task, so publishing the marker would remove the task
#:   rather than make it easier.
#: * `persistent_operation` -- keep a line running across a declared
#:   disruption.
TRACKS: frozenset[str] = frozenset(
    {
        "movement",
        "logistics",
        "production",
        "repair",
        "diagnosis",
        "persistent_operation",
    }
)


#: Tile footprints, keyed by prototype. Read off a real engine's bounding
#: boxes, not guessed: `test_entity_footprints` pins the table, and a prototype
#: absent from it is treated as 1x1.
#:
#: This lived in `tools/generator_diagnostics.py`, where the reachability BFS
#: needed it, while `footprint_conflicts` compared one rounded centre tile per
#: entity -- so the validator that was supposed to catch overlapping machines
#: could not see a 2x2 at all.
ENTITY_TILE_SIZES: dict[str, tuple[int, int]] = {
    # 1x1 is also the default; listed because `belt_smelting` places or hands
    # out all four, and an explicit entry is a measured fact rather than a
    # fallback (`MEASURED_FOOTPRINTS` pins each one).
    "burner-inserter": (1, 1),
    "burner-mining-drill": (2, 2),
    "electric-mining-drill": (3, 3),
    "solar-panel": (3, 3),
    "stone-furnace": (2, 2),
    "stone-wall": (1, 1),
    "transport-belt": (1, 1),
    "wooden-chest": (1, 1),
}


#: Prototypes the character can stand on. Factorio carries the character over a
#: transport belt rather than blocking it, so a belt under the start position is
#: not an obstruction. `tools/generator_diagnostics.py` excludes the same set
#: from its reachability BFS, for the same reason.
WALKABLE_PROTOTYPES: frozenset[str] = frozenset({"transport-belt"})


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

    # ---- engine-free structural validation (DESIGN.md 3.2 solvability) -------

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

    def character_obstructed(self) -> list[str]:
        """The character must not start inside a declared entity's footprint.

        `build_blueprint` teleports the character without a collision check, so
        a scene that draws a start position independently of its obstacles can
        put the character inside one and nothing raises. The split audit found
        exactly that in `plate_line`: 10 of 200 `commissioning_walled` scenes
        started the character on a wall tile, because the screen sits at x = 6
        and the start is drawn at radius 5..9 from the drill.

        Checked here rather than in that one generator, because the defect is a
        property of "draw a start, then place obstacles" and any family can have
        it. Belts are excluded: Factorio carries the character over a transport
        belt rather than blocking it, so a belt line under the start is not an
        obstruction -- `tools/generator_diagnostics.py` makes the same exclusion
        for the same reason.
        """
        start = (
            math.floor(self.character_position[0]),
            math.floor(self.character_position[1]),
        )
        problems = []
        for entity in self.entities:
            if entity.name in WALKABLE_PROTOTYPES:
                continue
            if start in set(entity_tiles(entity.name, entity.position)):
                problems.append(
                    f"the character starts at {self.character_position}, inside the "
                    f"{entity.name} at {entity.position} (tile {start})"
                )
        return problems

    def all_markers(self) -> dict[str, list[float]]:
        """Every name that resolves to a position at runtime.

        `world.truth()` merges `scene.markers` with the positions of the
        entities `scene.aliases` names, so a marker carried by an `EntitySpec`
        is a marker as far as any predicate is concerned. Anything reading only
        `self.markers` therefore sees a smaller set than the environment does:
        `plate_line` declares `difficulty_marker = "furnace"`, which is an
        entity alias, and the split audit reported all 600 of its scenes as
        declaring no such marker -- difficulty measured against nothing, in a
        task where the marker exists.
        """
        markers = {name: list(position) for name, position in self.markers.items()}
        for entity in self.entities:
            if entity.marker:
                markers[entity.marker] = list(entity.position)
        return markers

    def initial_state(self) -> tuple[dict, dict]:
        """A synthetic `(observation, truth)` for the scene at reset.

        Enough for `Predicate.evaluate` to decide the kinds a blueprint fully
        determines, so "the success predicate must be False at reset" can be
        checked without an engine. The cumulative channels are empty by
        construction rather than by assumption: `world.clear_statistics` clears
        production and the placement counter at every episode start, and only
        the `place` action increments the latter.
        """
        markers = self.all_markers()
        containers: dict[str, dict] = {}
        placed: dict[str, int] = {}
        entities = []
        for entity in self.entities:
            placed[entity.name] = placed.get(entity.name, 0) + 1
            entities.append({"name": entity.name, "p": list(entity.position), "type": entity.name})
            if entity.marker:
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
    #: Output measured by an evaluator-controlled, action-locked verification
    #: window. It is absent until that window has finished.
    VERIFIED_MACHINE_OUTPUT = "verified_machine_output"


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


#: Default for `Predicate.max_sample_gap`: the widest gap allowed between
#: consecutive samples covering a window.
#:
#: That gap is exactly how much production a window can misattribute: with no
#: sample between the baseline and the end, output produced just before the
#: window is indistinguishable from output produced inside it. One decision
#: interval is the natural bound -- every stepping path samples at least that
#: often, so a dense history always satisfies it and an externally-advanced
#: jump does not.
SAMPLE_TOLERANCE_TICKS = 30


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
    {PredicateKind.PRODUCED, PredicateKind.SUSTAINED_OUTPUT, PredicateKind.VERIFIED_MACHINE_OUTPUT}
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
    #: Earliest tick at which this predicate may be satisfied.
    #:
    #: A trailing window that begins at tick 0 spans the period before anything
    #: was built, so its output is construction plus a partial run rather than a
    #: rate. Measured on a real engine: `build_line` was accepted at tick 3600
    #: with a drill that had been mined out 90 ticks earlier, because the window
    #: [0, 3600] still held every plate the line had ever made. At these
    #: parameters the criterion was therefore doing almost nothing that
    #: `PRODUCED` does not.
    #:
    #: Setting this to twice `over_ticks` means the first window that can be
    #: accepted is [over_ticks, 2*over_ticks] -- entirely after construction --
    #: so a line that stopped cannot pass, whatever its cumulative total.
    #:
    #: The alternative was conjoining `ANY_WORKING`, and it was measured and
    #: rejected: a *healthy* line reads status `working` on 7.5% of sampled
    #: ticks, because one burner drill outpaces one stone furnace and sits in
    #: `waiting_for_space_in_destination` 87% of the time. That conjunct would
    #: have failed correct builds nine times in ten.
    not_before_tick: int = 0
    #: Widest gap allowed between consecutive samples covering the window.
    #:
    #: Declared per predicate rather than fixed, because it has to be at least
    #: the task's `decision_ticks` or every claim is refused: a task stepping
    #: every 60 ticks cannot satisfy a 30-tick bound. `validate_all` checks the
    #: relationship, so getting it wrong fails at load rather than silently
    #: rejecting every window.
    max_sample_gap: int = SAMPLE_TOLERANCE_TICKS

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
            # And it must have been *observed*: a window nothing sampled cannot
            # tell a rate from a jump. See `window_sampled`.
            if not self.window_sampled(truth):
                return False
            if self.not_before_tick and not self.settled(truth):
                return False
            return self.window_output(truth) >= self.at_least
        if self.kind is PredicateKind.VERIFIED_MACHINE_OUTPUT:
            return float((truth.get("verification") or {}).get(self.item, 0)) >= self.at_least
        return False

    def window_elapsed(self, truth: dict) -> bool:
        """Whether the history spans at least `over_ticks`."""
        history = truth.get("window") or []
        if len(history) < 2:
            return False
        return (history[-1][0] - history[0][0]) >= self.over_ticks

    def window_sampled(self, truth: dict) -> bool:
        """Whether the window was actually *observed*, not merely spanned.

        `window_elapsed` checks span, and span alone is not enough. Measured
        before this existed:

            history = [(450, 1 plate), (4080, 16 plates)]
            span 3630, samples 2 -> elapsed True, output 15.0, evaluate True

        Two samples 3,630 ticks apart satisfied a criterion built to reject a
        burst. The reason is not the baseline's position -- it sits a plausible
        30 ticks before the cutoff -- it is that **no sample lies inside the
        window at all**, so all 15 plates are attributed to it whether they
        were produced in it or in the 30 ticks before it.

        So what has to be bounded is the spacing between consecutive samples
        covering the window: that spacing is exactly how much production can be
        misattributed. `SAMPLE_TOLERANCE_TICKS` is the bound.

        Unreachable while every stepping path sampled every `decision_ticks`;
        `FactorioEnv.advance` makes external advancement possible, so it
        becomes reachable and is checked.
        """
        history = truth.get("window") or []
        if len(history) < 2:
            return False
        cutoff = history[-1][0] - self.over_ticks
        # Start at the sample `window_output` will use as its baseline: the
        # newest one at or before the cutoff, else the earliest.
        start = 0
        for index, (tick, _counts) in enumerate(history):
            if tick <= cutoff:
                start = index
            else:
                break
        ticks = [tick for tick, _counts in history[start:]]
        if len(ticks) < 2:
            return False
        widest = max(b - a for a, b in zip(ticks[:-1], ticks[1:], strict=True))
        return widest <= self.max_sample_gap

    def settled(self, truth: dict) -> bool:
        """Whether enough game time has passed for the window to mean a rate.

        See `not_before_tick`. False with no history at all, so a predicate that
        declares a settling period can never be satisfied before the
        environment has recorded anything.
        """
        history = truth.get("window") or []
        if not history:
            return False
        return history[-1][0] >= self.not_before_tick

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


@dataclass(frozen=True)
class VerificationSpec:
    """An evaluator-controlled output measurement at the end of an episode.

    During the window no policy action is dispatched. The count is a delta of
    ``truth[\"machine_produced\"]``, so output from before the window cannot
    satisfy it.

    That delta alone does not keep hand labour out, whatever this docstring used
    to claim. ``machine_produced`` is ``produced - handcrafted - mined``, so a
    plate smelted from hand-mined ore is machine output, and a furnace loaded by
    hand before the window scores in full with no drill anywhere.
    ``docs/evidence/a4-production.json`` records exactly that: five "machine"
    plates from thirteen hand-mined ore, with only a furnace built. ``source``
    closes it: when set, the counted output is capped by how much of that input
    machines produced inside the same window.

    ``container`` changes *what* is counted, not how it is capped. When set, the
    count is the increase of ``item`` inside the marked container over the
    window (``truth["containers"][container]``) rather than the machine
    production delta: a plate smelted but never delivered does not count, and
    neither does anything put in the container before the window, because the
    reading is a delta and no action runs during it. ``source`` still applies on
    top, so plates smelted from hand-mined ore are capped away exactly as for a
    production count.
    """

    item: str
    target: int
    ticks: int
    #: A machine-made input the counted output must be matched by, inside the
    #: window. ``None`` keeps the uncapped delta.
    source: str | None = None
    #: A marker naming a container whose contents are counted instead of
    #: machine production. ``None`` keeps the production delta.
    container: str | None = None

    def to_dict(self) -> dict:
        body: dict = {"item": self.item, "target": self.target, "ticks": self.ticks}
        # Omitted rather than written as null, so a spec without a source keeps
        # the digest it had before the field existed.
        if self.source is not None:
            body["source"] = self.source
        # Same rule, same reason: every existing task's digest is unchanged.
        if self.container is not None:
            body["container"] = self.container
        return body


# ---------------------------------------------------------------------- task


@dataclass(frozen=True)
class LayoutFamily:
    """Structural variation, and the unit of held-out generalisation.

    DESIGN.md section 3 asks for "unfamiliar seeds and unfamiliar structures"
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
    #: Optional action-locked evaluator window. Its ticks are reserved from
    #: ``max_game_ticks``; policy actions stop at the earlier construction cap.
    verification: VerificationSpec | None = None
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
    #: reported both families' difficulty parity as unmeasurable, which DESIGN.md
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
    #: Markers naming where this task's fault is, for coverage analysis.
    #:
    #: Declared rather than inferred from a `gap` naming convention, for the
    #: same reason `difficulty_marker` is declared: a convention that silently
    #: matches nothing reads as "this task has no faults" rather than as a typo.
    #:
    #: This is metadata about how the task is *measured*, so like
    #: `difficulty_marker` it is deliberately absent from `to_dict()`: it
    #: changes no scene, no predicate and no budget, and adding it must not
    #: bump a version or move a frozen holdout digest.
    fault_markers: tuple[str, ...] = ()

    #: Which of R4.3's three declared tracks this task belongs to.
    #:
    #: Declared on the task rather than looked up by id. It replaces
    #: `tools/release_matrix.CATEGORY`, a dict hard-coded in a tool whose own
    #: comment said *"the task specs carry no category field"* -- and which
    #: listed six of the eight tasks, so the release qualification filter
    #: (`QUALIFYING_CATEGORIES = {"production", "repair"}`) could not see
    #: `plate_line` or `build_line` at all. A task absent from a hand-written
    #: dict fails the filter silently, which is the failure mode a declared
    #: field removes.
    #:
    #: In `to_dict()`, unlike `fault_markers` and `difficulty_marker`: those
    #: describe how a task is *measured* and change nothing the agent meets,
    #: while different tracks are different benchmarks -- a production result
    #: and a diagnosis result are not comparable, and a manifest that cannot
    #: tell them apart invites exactly that comparison.
    #:
    #: The default is deliberately not a real track. A defaulted-but-valid
    #: value would file a new task under some existing track by accident, which
    #: is the `CATEGORY.get(task, "unknown")` failure with a different spelling;
    #: `"unclassified"` is refused by `validate_all`, so the omission surfaces
    #: as a validation error before a worker is launched.
    track: str = "unclassified"

    #: Disruptions the environment applies during an episode, in declared
    #: order. Empty for every task that does not want one.
    #:
    #: In `to_dict()`: a disruption changes what happens in an episode, so two
    #: runs differing in it are runs of different tasks. `validate_all` checks
    #: the kinds and that each lands inside the tick budget -- a disruption
    #: declared after `max_game_ticks` would never fire and nothing would say
    #: so.
    disruptions: tuple[Disruption, ...] = ()

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
                "verification": self.verification.to_dict() if self.verification else None,
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
            # Landmarks are policy input, not evaluator bookkeeping: each one
            # occupies a slot in the goal vector `_goal_vector` hands the
            # encoder. Two tasks that agree on everything else and differ here
            # present the policy with different observations, which is the same
            # argument that put `extra_public_markers` and `focus_marker` in
            # this dict -- and this field was left out when they went in.
            "landmarks": [p.describe() for p in self.landmarks],
            "track": self.track,
            "disruptions": [d.to_dict() for d in self.disruptions],
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
