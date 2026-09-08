"""Bounded action catalogs (PLAN.md 3.4).

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


@dataclass(frozen=True)
class ActionTemplate:
    """One index in the catalog: an action name plus its bound arguments."""

    key: str
    action: str
    payload: dict[str, Any] = field(default_factory=dict)
    #: Which observation feature the mask builder consults, if any.
    requires: str = ""

    def bind(self, context: dict) -> dict:
        """Produce the wire payload, filling any runtime reference."""
        body = {"action": self.action, **self.payload}
        for key, value in list(body.items()):
            if isinstance(value, str) and value.startswith("$"):
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

CATALOGS: dict[str, tuple[ActionTemplate, ...]] = {"primitive-v1": PRIMITIVE_V1}


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
        payload = json.dumps(
            [[t.key, t.action, t.payload, t.requires] for t in self.templates],
            sort_keys=True,
            separators=(",", ":"),
        )
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
