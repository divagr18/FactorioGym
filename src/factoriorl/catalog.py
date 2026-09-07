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


def _move_templates() -> list[ActionTemplate]:
    return [
        ActionTemplate(f"move_{d}", "move", {"direction": d, "ticks": 30})
        for d in ("north", "east", "south", "west")
    ]


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
        ActionTemplate(
            "place_stone_furnace",
            "place",
            {"item": "stone-furnace", "position": "$ahead"},
            requires="ahead",
        ),
        ActionTemplate(
            "place_transport_belt",
            "place",
            {"item": "transport-belt", "position": "$ahead"},
            requires="ahead",
        ),
        ActionTemplate(
            "place_small_electric_pole",
            "place",
            {"item": "small-electric-pole", "position": "$ahead"},
            requires="ahead",
        ),
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
        payload = json.dumps(
            [[t.key, t.action, t.payload] for t in self.templates],
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
