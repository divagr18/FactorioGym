"""Bounded action catalogs (DESIGN.md 3.4).

A catalog is an **ordered list of fully-bound action templates**: file order is
the action index. Reproducibility rests on three things, not on hoping the order
never changes:

* the resolved catalog is content-hashed into the run manifest;
* a task's subset is re-indexed by *catalog* order, never by the order the task
  happened to list them in;
* a frozen golden index fixture fails the suite if any index moves -- the same
  discipline the protocol fixtures use.

Every catalog contains ``wait``, and the mask builder always leaves it legal, so
a valid no-op fallback exists in every state.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

WAIT = "wait"

#: Two verbs the environment answers itself rather than forwarding to a mod
#: action. Both consume the step they are given -- looking around and waiting
#: are things that take time -- but neither changes the world, so both ride on
#: the mod's `wait`, which is precisely a no-op that consumes the step.
#:
#: Named here rather than matched on a string in three places, because the
#: environment has to recognise them before it binds a payload.
INSPECT = "inspect"
WAIT_FOR = "wait_for"
ENV_HANDLED = (INSPECT, WAIT_FOR)

#: `parameterized-v3`'s `finish`: the agent declares construction done, the
#: task's verification window runs at once and the episode ends on its result
#: (user decision, 2026-09-25). Answered by the environment, like the two
#: above, and never sent to the mod.
FINISH = "finish"

#: `parameterized-v3`'s `take_fuel`: a transfer out of a burner's fuel slot.
#: The item is the one the slot holds, filled in from the observed record, so
#: the verb names a target and an amount only.
TAKE_FUEL = "take_fuel"

#: Payload values beginning with this are *arguments*: the caller supplies them.
#: Values beginning with "$" are bound from the observation-derived context
#: instead, which is the older, fully-bound form.
#:
#: Declaring an argument by writing it into the payload means the catalog digest
#: covers it with no extra field, so an action that gains or loses an argument
#: is a different action by construction.
ARGUMENT_PREFIX = "?"

#: Which observation-derived domain supplies an argument's legal values. Every
#: domain must be reconstructible from a policy-visible observation -- the same
#: discipline `skill_context` follows -- or a masked argument would be smuggling
#: evaluator state.
ARGUMENT_DOMAINS: dict[str, str] = {
    "handle": "targets",
    "from": "targets",
    "to": "targets",
    "position": "placements",
    "direction": "directions",
    "item": "items",
    "count": "amounts",
    "recipe": "recipes",
    "technology": "technologies",
    "target_request_id": "requests",
    "until": "conditions",
    "seconds": "durations",
    # Deliberately not `position`, which draws from `placements` -- tile centres
    # within `PLACEMENT_RADIUS` of the character, because a placement further
    # than build distance would only be refused. A destination is the opposite
    # problem: the whole point of walking is to reach something *out* of reach,
    # and on the generated map this exists for the nearest ore is 28 tiles away.
    # One argument name, one domain; sharing `position` would have made
    # `walk_to_position` unable to name anywhere worth walking to.
    "destination": "destinations",
    # `parameterized-v3`'s `mine_tile`: a resource tile named by where it is.
    "tile": "resource_tiles",
}

#: Argument domains that depend on which *action* is asking, keyed by catalog
#: key and then by argument name. Overrides `ARGUMENT_DOMAINS`.
#:
#: `item` is the case that forced this. Both `give_to` and `take_from` send an
#: `item` field, and both were validated against the character's inventory --
#: correct for giving and wrong for taking. Collecting the *first* iron plate
#: out of a furnace was therefore impossible through the advertised tool: the
#: agent holds no plates, so `iron-plate` is not in `items`, so the request is
#: refused before the game sees it. It also handed an arbitrary advantage to
#: whatever the character happened to still be carrying.
#: `set_recipe_at` is the other case. A stone furnace picks its recipe from
#: whatever you put in it and has no settable recipe at all -- `set_recipe`
#: raises on one, and the handler turns that into `invalid_target`. The domain
#: offered every handle in sight regardless, so a run with one furnace and no
#: assembling machine could see `set_recipe_at` in its legal actions for its
#: whole length, and spent a decision discovering it was never real. Sixth
#: instance of a domain advertising what the runtime always refuses.
ARGUMENT_DOMAINS_BY_KEY: dict[str, dict[str, str]] = {
    "take_from": {"item": "source_items"},
    "set_recipe_at": {"handle": "recipe_targets"},
}

#: What `wait_for` will wait on. Deliberately a short enumerated list rather
#: than an expression language: a domain the model can see is a domain it can
#: choose from, and an argument whose values are unguessable is one that gets
#: guessed wrong. Each is cheap to evaluate from the observation alone.
WAIT_CONDITIONS = ("world_changes", "inventory_grows", "nothing_in_flight")

#: Seconds `wait_for` may spend. The ceiling matters: a condition that never
#: becomes true would otherwise consume the whole run, and a stone furnace
#: takes 3.2 seconds per plate so the useful range is small.
WAIT_SECONDS = (1, 3, 10, 30)


@dataclass(frozen=True)
class ActionTemplate:
    """One index in the catalog: an action name plus its bound arguments."""

    key: str
    action: str
    payload: dict[str, Any] = field(default_factory=dict)
    #: Which observation feature the mask builder consults, if any.
    requires: str = ""

    @property
    def arguments(self) -> tuple[str, ...]:
        """Payload slots the caller must supply, in payload order."""
        return tuple(
            value[len(ARGUMENT_PREFIX) :]
            for value in self.payload.values()
            if isinstance(value, str) and value.startswith(ARGUMENT_PREFIX)
        )

    @property
    def parameterized(self) -> bool:
        return bool(self.arguments)

    def bind(self, context: dict, arguments: dict | None = None) -> dict:
        """Produce the wire payload.

        `$name` is filled from the observation-derived context, as before.
        `?name` is filled from `arguments`, and its absence is an error rather
        than a silent `None` -- a payload with a missing position is a
        different request, not the same one with a hole in it.
        """
        body = {"action": self.action, **self.payload}
        for key, value in list(body.items()):
            if not isinstance(value, str):
                continue
            if value.startswith(ARGUMENT_PREFIX):
                name = value[len(ARGUMENT_PREFIX) :]
                if not arguments or name not in arguments:
                    raise ValueError(f"{self.key} needs argument {name!r}")
                body[key] = arguments[name]
            elif value.startswith("$"):
                body[key] = context.get(value[1:])
        return body


DIRECTIONS = ("north", "east", "south", "west")

#: A 30-tick move covers about 4.45 tiles, so a catalog with only that stride
#: cannot approach anything within ~4 tiles: the agent overshoots and
#: oscillates forever. The reference solvers hit this on every family. A short
#: stride costs four action indices and makes positioning possible at all.
#: The character runs at 0.1484 tiles/tick, so a 7-tick stride covers 1.039
#: tiles. That is 2.6x the tolerance a solver needs to land on a specific tile,
#: which left the walker oscillating between two points on a ~1-tile lattice --
#: never blocked, just unable to stop anywhere useful. Placement binds to
#: floor(position), so *tile-level* precision is a hard requirement of the
#: action set, not a nicety. A 2-tick nudge covers 0.297 tiles and bounds the
#: final positioning error at ~0.15.
LONG_MOVE_TICKS = 30
SHORT_MOVE_TICKS = 7
NUDGE_MOVE_TICKS = 2


def _move_templates() -> list[ActionTemplate]:
    templates = [
        ActionTemplate(f"move_{d}", "move", {"direction": d, "ticks": LONG_MOVE_TICKS})
        for d in DIRECTIONS
    ]
    templates += [
        ActionTemplate(f"step_{d}", "move", {"direction": d, "ticks": SHORT_MOVE_TICKS})
        for d in DIRECTIONS
    ]
    templates += [
        ActionTemplate(f"nudge_{d}", "move", {"direction": d, "ticks": NUDGE_MOVE_TICKS})
        for d in DIRECTIONS
    ]
    return templates


#: Placement offsets, one per facing. Binding placement to a single fixed
#: offset (character + 2 east) meant a belt gap could only ever be filled by
#: standing exactly two tiles west of it -- an unreachable precondition for an
#: agent that also could not position precisely.
PLACE_OFFSETS = {"north": (0, -1), "east": (1, 0), "south": (0, 1), "west": (-1, 0)}


def _place_templates(items: tuple[str, ...]) -> list[ActionTemplate]:
    out = []
    for item in items:
        short = item.replace("-", "_")
        for direction in DIRECTIONS:
            out.append(
                ActionTemplate(
                    f"place_{short}_{direction}",
                    "place",
                    # Facing follows the placement direction: "place a belt to
                    # my east" means one that carries east.
                    {"item": item, "position": f"$at_{direction}", "direction": direction},
                    requires=f"at_{direction}",
                )
            )
    return out


def _transfer_templates(items: tuple[str, ...]) -> list[ActionTemplate]:
    out = []
    for item in items:
        for count in (5, 20):
            out.append(
                ActionTemplate(
                    f"take_{item}_{count}",
                    "transfer",
                    {"from": "$target", "to": "character", "item": item, "count": count},
                    requires="target",
                )
            )
            out.append(
                ActionTemplate(
                    f"give_{item}_{count}",
                    "transfer",
                    {"from": "character", "to": "$target", "item": item, "count": count},
                    requires="target",
                )
            )
    return out


#: The primitive catalog. Templates that need a runtime target use "$target",
#: resolved to the nearest reachable entity handle at bind time -- a bounded,
#: declared way to address entities without an unbounded action space.
PRIMITIVE_V1: tuple[ActionTemplate, ...] = tuple(
    [
        *_move_templates(),
        ActionTemplate(
            "mine_nearest", "mine", {"handle": "$resource", "count": 1}, requires="resource"
        ),
        ActionTemplate(
            "mine_nearest_5", "mine", {"handle": "$resource", "count": 5}, requires="resource"
        ),
        *_transfer_templates(("iron-ore", "coal", "iron-plate", "stone")),
        ActionTemplate("craft_stone_furnace", "craft", {"recipe": "stone-furnace", "count": 1}),
        ActionTemplate("craft_iron_gear", "craft", {"recipe": "iron-gear-wheel", "count": 1}),
        *_place_templates(("stone-furnace", "transport-belt", "small-electric-pole")),
        ActionTemplate("rotate_target", "rotate", {"handle": "$target"}, requires="target"),
        ActionTemplate(WAIT, "wait", {}),
    ]
)

#: The same verbs, taking arguments instead of being fully bound.
#:
#: `primitive-v1` welds *where* to *which facing*: `place_<item>_<direction>`
#: targets `floor(position) + PLACE_OFFSETS[direction]` and sets the facing to
#: the same direction, so filling a belt gap with an east-facing belt requires
#: standing west of it -- on the belt line. The reference solver works around
#: that with `place_transport_belt_north` then `rotate_target`, two ordered
#: actions at tile precision, and `repair_belt` earned 7 successes in 221
#: episodes trying to find them.
#:
#: The Lua `place` handler has always taken an arbitrary in-reach position and
#: an independent direction, so this exposes an existing capability rather than
#: adding one. Every verb here already has a handler that enforces reach,
#: collision, inventory and technology.
PARAMETERIZED_V1: tuple[ActionTemplate, ...] = tuple(
    [
        *_move_templates(),
        # Position and facing chosen independently -- the point of the profile.
        ActionTemplate(
            "place_at",
            "place",
            {"item": "?item", "position": "?position", "direction": "?direction"},
        ),
        # A chosen resource, not merely the nearest one. `count` is pinned at 1
        # and stays pinned: `build_line`'s published numbers were measured
        # against this catalog's digest, so the open world overrides it in
        # `OPEN_V1` rather than editing a frozen benchmark out from under its
        # own results.
        ActionTemplate("mine_at", "mine", {"handle": "?handle", "count": 1}),
        ActionTemplate("rotate_at", "rotate", {"handle": "?handle"}),
        ActionTemplate("rotate_at_reverse", "rotate", {"handle": "?handle", "reverse": True}),
        ActionTemplate(
            "give_to",
            "transfer",
            {"from": "character", "to": "?to", "item": "?item", "count": "?count"},
        ),
        ActionTemplate(
            "take_from",
            "transfer",
            {"from": "?from", "to": "character", "item": "?item", "count": "?count"},
        ),
        # Permitted by `primitive-v1` in the mod and untemplated until now.
        # Both stay masked until recipes are observable (R2.3): an argument
        # whose domain the policy cannot see is not selectable, and reporting
        # that honestly is better than guessing a vocabulary.
        ActionTemplate("set_recipe_at", "set_recipe", {"handle": "?handle", "recipe": "?recipe"}),
        ActionTemplate("craft_recipe", "craft", {"recipe": "?recipe", "count": "?count"}),
        ActionTemplate("cancel_request", "cancel", {"target_request_id": "?target_request_id"}),
        ActionTemplate(WAIT, "wait", {}),
    ]
)

#: How close `walk_to` has to get. The mod refuses anything under 1.0 tile and
#: the derivation is worth repeating: a route ends on a tile centre, an
#: arbitrary requested point can be a tile corner 0.7072 away from the nearest
#: centre, and the per-tick walker leaves ~0.25 of residual error. Demanding
#: tighter than a tile is asking the walker for a promise it cannot keep;
#: sub-tile positioning is what the `nudge_*` strides are for.
WALK_TOLERANCE = 1.5

#: Ceiling on one navigation, in ticks. At 0.1484 tiles/tick a 3,600-tick walk
#: covers ~534 tiles, far past anything in sensor range, so this bounds a
#: pathological route rather than shaping ordinary ones.
WALK_MAX_TICKS = 3600


#: The verbs an open world needs and the benchmark catalogs do not have.
#:
#: `parameterized-v1` is what `build_line` was measured against, and
#: `manifest.verify` re-derives its digest, so adding to it would change what an
#: existing result means. This is a separate catalog for the same reason
#: `open-v1` is a separate *observation* profile: a frozen contract is not
#: edited for the benefit of a mode no frozen task uses.
#:
#: Everything reused from `parameterized-v1` is reused *by reference*, so the
#: two cannot drift apart in the six verbs they share.
def _with_batched_mining(templates: tuple[ActionTemplate, ...]) -> list[ActionTemplate]:
    """`mine_at` with a chosen count, in the position the pinned one held.

    The primitive catalog has shipped `mine_nearest_5` since it was written, so
    a fixed count of 5 was always within what the mod does -- and yet the
    *parameterized* `mine_at` was pinned at 1, which made the richer profile the
    weaker one for the verb a run spends most of its early decisions on. Run 7
    mined coal one tile per reply: three decisions and three model calls for
    what a single `count` could have asked for.

    Batching is safe rather than hopeful. `inflight.lua`'s `POLLS.mine`
    completes with `mined` and `requested` when the tile empties underneath it,
    so asking for 20 from a tile holding 5 returns 5 and says so; it does not
    hang on an exhausted deposit.

    Substituted in place rather than appended because a catalog's order is its
    action indices, and it is substituted *here* rather than edited in
    `PARAMETERIZED_V1` because that catalog is frozen: `build_line`'s recorded
    `catalog_digest` is checked by `manifest.verify`, and moving it would
    invalidate every committed result for that task.
    """
    return [
        ActionTemplate("mine_at", "mine", {"handle": "?handle", "count": "?count"})
        if template.key == "mine_at"
        else template
        for template in templates
    ]


OPEN_V1: tuple[ActionTemplate, ...] = tuple(
    [
        *_with_batched_mining(PARAMETERIZED_V1),
        # Walking, plural, because the mod's `navigate` takes a position *or* a
        # handle and the template system fills every declared argument -- one
        # template with an optional slot would be a template that sometimes
        # sends a hole. Measured on the natural map this exists for: the nearest
        # iron ore sits 28 tiles from spawn, and a catalog of fixed-direction
        # strides reaches it only by guessing.
        ActionTemplate(
            "walk_to_position",
            "navigate",
            {
                "position": "?destination",
                "tolerance": WALK_TOLERANCE,
                "max_ticks": WALK_MAX_TICKS,
            },
        ),
        ActionTemplate(
            "walk_to_entity",
            "navigate",
            {"handle": "?handle", "tolerance": WALK_TOLERANCE, "max_ticks": WALK_MAX_TICKS},
        ),
        # Permitted by the mod since the action matrix was written and never
        # templated, because `env.argument_domains` returned an empty
        # `technologies` list and an argument whose domain the policy cannot see
        # is not selectable. The open world publishes researchable technologies,
        # which is what makes this reachable rather than decorative.
        ActionTemplate("research_technology", "research", {"technology": "?technology"}),
        # Look at one entity in detail. The observation shows the twelve nearest
        # (`summary.MAX_ENTITIES_SHOWN`), so on a map with anything built on it
        # most of what exists is not in the prompt.
        ActionTemplate(INSPECT, "wait", {"handle": "?handle"}),
        # A bounded wait on a stated condition. One plain `wait` buys the
        # step's 30 ticks -- half a second -- and a stone furnace takes 3.2
        # seconds to smelt one plate, so waiting for a plate the honest way
        # costs six decisions and six model calls.
        ActionTemplate(WAIT_FOR, "wait", {"until": "?until", "seconds": "?seconds"}),
    ]
)

#: The catalog a `v3` task declares, and what selects the `v3` profile.
#:
#: `parameterized-v1`'s verbs, in the same order and reused by reference so
#: the two cannot drift apart, then one more at the end, so no `v1` index
#: moves. Every verb a belt and inserter task needs already exists there --
#: `place_at` takes any item in any facing, `rotate_at` turns a belt or an
#: inserter, `take_from` empties a chest. What changes is what the argument
#: indices *mean*, which is `parameterized.ParameterizedEnv`'s `v3` profile: a
#: target is a row of the 96-row entity table and a placement a fixed tile of a
#: 15x15 window.
#:
#: The one addition is hand-mining a resource tile (user decision, 2026-09-25;
#: `docs/sim-logistics.md`). Resource tiles have no row in the entity table,
#: so under `v2` they could not be named at all; `mine_tile` names one by its
#: tile, through the placement dimension, which is where the grid planes that
#: show the ore already put it, with the count from the amount dimension. The
#: mod's `mine` action does the rest exactly as it does for `mine_at`.
#:
#: Stage 2's `set_recipe_at` with a recipe dimension lands here too, without
#: moving `parameterized-v1`'s frozen digest.
#:
#: Then two more (user decisions, 2026-09-25): `take_fuel`, a transfer out of a
#: burner's fuel slot -- `take_from` offers what an entity holds as contents or
#: output, never its fuel -- measured first by `tools/probe_inventory.py`
#: (`inventory-fuel`); and `finish`, which runs the task's verification now and
#: ends the episode on it.
PARAMETERIZED_V3: tuple[ActionTemplate, ...] = (
    *PARAMETERIZED_V1,
    ActionTemplate("mine_tile", "mine", {"handle": "?tile", "count": "?count"}),
    ActionTemplate(
        TAKE_FUEL,
        "transfer",
        {"from": "?from", "to": "character", "item": "$fuel_item", "count": "?count"},
    ),
    ActionTemplate(FINISH, "wait", {}),
)

#: Catalogs whose argument indices follow the `v3` profile.
V3_CATALOGS = frozenset({"parameterized-v3"})

CATALOGS: dict[str, tuple[ActionTemplate, ...]] = {
    "primitive-v1": PRIMITIVE_V1,
    "parameterized-v1": PARAMETERIZED_V1,
    "open-v1": OPEN_V1,
    "parameterized-v3": PARAMETERIZED_V3,
}


@dataclass(frozen=True)
class ResolvedCatalog:
    """A task's subset, indexed by catalog order."""

    name: str
    templates: tuple[ActionTemplate, ...]

    def __len__(self) -> int:
        return len(self.templates)

    @property
    def wait_index(self) -> int:
        for index, template in enumerate(self.templates):
            if template.action == WAIT:
                return index
        raise ValueError("catalog has no wait action; masks would have no legal fallback")

    def keys(self) -> tuple[str, ...]:
        return tuple(t.key for t in self.templates)

    def digest(self) -> str:
        # `requires` is in here because it is what the mask builder consults,
        # and a template whose availability rule changed is a different action
        # even when its wire payload is identical. Without it, editing mask
        # semantics left the digest -- and therefore every manifest citing it --
        # unchanged.
        rows: list = [[t.key, t.action, t.payload, t.requires] for t in self.templates]
        # A `v3` catalog shares `parameterized-v1`'s templates but not what its
        # argument indices mean, so its digest says so. Only for `v3` names:
        # every other digest is exactly what it was.
        if self.name in V3_CATALOGS:
            rows = [["profile", "v3"], *rows]
        payload = json.dumps(rows, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def resolve(name: str, subset: tuple[str, ...] = ()) -> ResolvedCatalog:
    if name not in CATALOGS:
        raise ValueError(f"unknown catalog: {name}")
    templates = CATALOGS[name]
    if subset:
        wanted = set(subset)
        # Re-indexed by catalog order, never by the order the task listed them.
        chosen = tuple(t for t in templates if t.key in wanted)
        missing = wanted - {t.key for t in chosen}
        if missing:
            raise ValueError(f"catalog {name} has no templates: {sorted(missing)}")
        if not any(t.action == WAIT for t in chosen):
            chosen = (*chosen, next(t for t in templates if t.action == WAIT))
        templates = chosen
    return ResolvedCatalog(name=name, templates=templates)
