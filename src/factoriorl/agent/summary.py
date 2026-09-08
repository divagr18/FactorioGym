"""Rendering the canonical observation for a language model (PLAN.md 2, 5.3).

PLAN section 2 requires *one* canonical structured observation that serves both
"a spatial tensor and padded entity representation for compact policies" and
"typed objects and concise summaries for language-model agents". This module is
the second half of that sentence. It is deliberately a sibling of
``factoriorl.encoders`` rather than a layer above it: both read the same wire
observation dict, so a model and a policy cannot end up looking at different
worlds.

Three boundaries are structural here rather than merely intended:

* **No task truth and no evaluator state.** ``summarise`` accepts a wire
  observation, a declared :class:`TaskBrief` and the legal actions the
  environment published. It has no parameter through which ``FactorioEnv._truth``
  could arrive, so a future edit cannot leak evaluator state into a prompt
  without changing the signature. The same rule ``skills.py`` follows applies to
  the brief: success-predicate *marker names* are the vocabulary of the answer,
  so the brief carries the task's declared prose description and nothing derived
  from its predicates.
* **Legality is the environment's, not ours.** Legal actions come from the mask
  the environment publishes, so a model is never offered an action the
  environment would reject; recomputing availability here would be a second
  implementation of the mask that could disagree with the first.
* **Action descriptions are derived from the template payload**, never from a
  per-key lookup table. A catalog entry added tomorrow gets a description
  automatically, and nothing in this file can name an action the catalog does
  not contain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: How many entities a summary lists before it starts counting the rest. A
#: 32-tile sensor can return 48 records; pasting all of them into every prompt
#: buys little and costs tokens on every decision of a 600-step episode. The
#: nearest few are the ones an action can address, and the count of what was
#: dropped is printed so the model is never silently told the world is smaller
#: than it is.
MAX_ENTITIES_SHOWN = 12

#: Recent action outcomes carried into the prompt. The engine keeps a bounded
#: event log; without a few of them the model cannot tell "my last placement
#: failed" from "my last placement worked", and repeats the failing action for
#: the rest of the episode.
MAX_EVENTS_SHOWN = 6


@dataclass(frozen=True)
class TaskBrief:
    """What the agent is allowed to be told about its task.

    Built from a task's *declared* fields only. PLAN section 2 lists the task
    description as part of the observation profile, so it is policy-visible;
    success predicates are not, because their markers name the answer.
    """

    id: str
    version: str
    description: str
    max_decision_steps: int

    @classmethod
    def from_spec(cls, spec: Any) -> TaskBrief:
        return cls(
            id=spec.id,
            version=spec.version,
            description=spec.description,
            max_decision_steps=spec.max_decision_steps,
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "version": self.version,
            "description": self.description,
            "max_decision_steps": self.max_decision_steps,
        }


@dataclass(frozen=True)
class LegalAction:
    """One action index the environment will currently accept."""

    index: int
    key: str
    description: str

    def to_dict(self) -> dict:
        return {"index": self.index, "key": self.key, "description": self.description}


# ------------------------------------------------------------------ actions


def describe_template(template: Any) -> str:
    """A human sentence for one bound catalog template.

    Derived from the template's action and payload, so the vocabulary can only
    ever be the catalog's own. ``$name`` arguments are runtime references the
    environment binds from the observation at step time (nearest entity, nearest
    resource, a tile beside the character), and they are rendered as the phrase
    that binding means rather than as the raw sigil -- a model shown
    ``"from": "$target"`` has to guess what it addresses, and the obvious guess
    is that it may name an entity of its choosing, which the catalog does not
    allow.
    """
    payload = dict(getattr(template, "payload", {}) or {})
    action = getattr(template, "action", "")

    def reference(value: Any) -> str:
        if isinstance(value, str) and value.startswith("$"):
            name = value[1:]
            if name == "target":
                return "the nearest entity"
            if name == "resource":
                return "the nearest resource tile"
            if name.startswith("at_"):
                return f"the tile immediately {name[3:]} of you"
            return name
        if value == "character":
            # The catalog's transfer endpoints are literally "character"; read
            # back as prose that names the container instead of the actor, a
            # model reads "to character" as a third party it should walk to.
            return "your own inventory"
        return str(value)

    if action == "move":
        return f"walk {payload.get('direction', '?')} for {payload.get('ticks', '?')} ticks"
    if action == "mine":
        return f"mine {reference(payload.get('handle'))} ({payload.get('count', 1)}x)"
    if action == "craft":
        return f"craft {payload.get('count', 1)}x {payload.get('recipe', '?')}"
    if action == "place":
        return f"place {payload.get('item', '?')} on {reference(payload.get('position'))}"
    if action == "transfer":
        return (
            f"move {payload.get('count', '?')}x {payload.get('item', '?')} "
            f"from {reference(payload.get('from'))} to {reference(payload.get('to'))}"
        )
    if action == "rotate":
        return f"rotate {reference(payload.get('handle'))}"
    if action == "wait":
        return "do nothing this decision interval"
    # A catalog entry this function has never seen still gets a description,
    # because falling back to nothing would silently hide a new action from
    # every model agent until someone remembered to edit this file.
    rendered = ", ".join(f"{k}={reference(v)}" for k, v in sorted(payload.items()))
    return f"{action}({rendered})" if rendered else action


def action_vocabulary(env: Any) -> tuple[tuple[str, str], ...]:
    """``(key, description)`` for every index the environment's mask covers.

    The primitive catalog first, in catalog order, then any temporally extended
    actions a wrapper appended -- which is exactly how ``SkillEnv`` lays out its
    action space. This is the seam a richer action profile plugs into: an
    assisted vocabulary that *appends* indices is described here without this
    module knowing what those actions are, while one that renumbered the
    primitive indices would break the catalog contract the manifests, the
    checkpoints and the golden index fixture all rest on, so it is not a case
    worth guessing at.
    """
    entries = [(t.key, describe_template(t)) for t in env.catalog.templates]
    for skill in getattr(env, "skills", ()) or ():
        entries.append((skill.key, skill.description))
    return tuple(entries)


def legal_actions(vocabulary: tuple[tuple[str, str], ...], mask: Any) -> tuple[LegalAction, ...]:
    """Filter the vocabulary by the environment's published mask.

    Anything the mask does not cover is dropped rather than assumed legal: a
    mask shorter than the vocabulary means the caller paired an environment with
    the wrong wrapper, and offering the uncovered tail would hand the model
    action indices the environment cannot even look up.
    """
    return tuple(
        LegalAction(index=index, key=key, description=description)
        for index, (key, description) in enumerate(vocabulary)
        if index < len(mask) and bool(mask[index])
    )


# --------------------------------------------------------------- observation


def _compass(dx: float, dy: float) -> str:
    """Factorio's y axis grows southward, which is the opposite of intuition."""
    parts = []
    if dy < -0.5:
        parts.append("north")
    elif dy > 0.5:
        parts.append("south")
    if dx > 0.5:
        parts.append("east")
    elif dx < -0.5:
        parts.append("west")
    return "".join(parts) or "here"


def _distance(dx: float, dy: float) -> float:
    return float((dx * dx + dy * dy) ** 0.5)


def _entity_row(record: dict, origin: list[float], remembered: bool) -> dict:
    position = record.get("p") or [0.0, 0.0]
    dx = float(position[0]) - float(origin[0])
    dy = float(position[1]) - float(origin[1])
    row: dict[str, Any] = {
        "handle": record.get("h"),
        "name": record.get("name"),
        "type": record.get("type"),
        "offset": [round(dx, 1), round(dy, 1)],
        "distance": round(_distance(dx, dy), 1),
        "bearing": _compass(dx, dy),
        "remembered": remembered,
    }
    if record.get("contents"):
        row["contents"] = record["contents"]
    if record.get("recipe"):
        row["recipe"] = record["recipe"]
    if record.get("status") is not None:
        row["status"] = record["status"]
    if record.get("health") is not None:
        row["health"] = record["health"]
    if remembered:
        # The age is the whole point of remembered observations: PLAN section 2
        # forbids presenting distant machine state as current, and a summary
        # that dropped the age would do exactly that in prose.
        row["age_ticks"] = record.get("age", 0)
    return row


def _entity_rows(observation: dict, origin: list[float]) -> tuple[list[dict], int]:
    rows = [_entity_row(r, origin, False) for r in observation.get("entities") or []]
    rows += [_entity_row(r, origin, True) for r in observation.get("remembered") or []]
    rows.sort(key=lambda row: row["distance"])
    return rows[:MAX_ENTITIES_SHOWN], max(0, len(rows) - MAX_ENTITIES_SHOWN)


def _resource_rows(observation: dict, origin: list[float]) -> list[dict]:
    """Per-name aggregates, because a 32-tile ore patch is hundreds of tiles.

    Listing every tile would bury the rest of the summary and say nothing a
    model can act on: the catalog addresses the *nearest* resource, so the
    nearest tile per resource name is the actionable fact and the tile count is
    the context for it.
    """
    grouped: dict[str, dict] = {}
    for tile in (observation.get("resources") or {}).get("tiles") or []:
        position = tile.get("p") or [0.0, 0.0]
        dx = float(position[0]) - float(origin[0])
        dy = float(position[1]) - float(origin[1])
        distance = _distance(dx, dy)
        name = tile.get("name", "unknown")
        row = grouped.get(name)
        if row is None:
            row = {"name": name, "tiles": 0, "distance": float("inf")}
            grouped[name] = row
        row["tiles"] += 1
        if distance <= row["distance"]:
            row["distance"] = distance
            row["nearest_offset"] = [round(dx, 1), round(dy, 1)]
            row["bearing"] = _compass(dx, dy)
            row["amount"] = tile.get("amount")
    for row in grouped.values():
        row["distance"] = round(row["distance"], 1)
    return sorted(grouped.values(), key=lambda row: row["distance"])


@dataclass(frozen=True)
class ObservationSummary:
    """The typed-object half of PLAN section 2's language-model rendering."""

    task: dict
    step: int
    tick: int
    character: dict
    inventory: dict
    entities: list[dict]
    entities_omitted: int
    resources: list[dict]
    inflight: list[dict]
    events: list[dict]
    actions: tuple[LegalAction, ...] = ()
    counters: dict = field(default_factory=dict)
    #: Positions of the markers the objective names. `deliver` publishes
    #: `dst` and not its two decoys, so this says which container to fill
    #: without saying which are wrong.
    goal: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "step": self.step,
            "tick": self.tick,
            "character": self.character,
            "inventory": self.inventory,
            "entities": self.entities,
            "goal": self.goal,
            "entities_omitted": self.entities_omitted,
            "resources": self.resources,
            "inflight": self.inflight,
            "events": self.events,
            "counters": self.counters,
            "actions": [a.to_dict() for a in self.actions],
        }

    def render(self) -> str:
        """The concise text a model is actually shown."""
        task = self.task
        lines: list[str] = [
            f"TASK {task.get('id')} v{task.get('version')}",
            f"  {task.get('description')}",
            f"  decision {self.step} of {task.get('max_decision_steps')}; game tick {self.tick}",
            "",
            "CHARACTER",
        ]
        character = self.character
        if not character.get("present", True):
            lines.append("  no character present")
        else:
            position = character.get("position") or [0.0, 0.0]
            state = "walking" if character.get("walking") else "standing"
            lines.append(f"  at ({float(position[0]):.1f}, {float(position[1]):.1f}), {state}")
            mining = character.get("mining") or {}
            if mining.get("active"):
                lines.append(f"  mining, progress {float(mining.get('progress') or 0.0):.2f}")
            crafting = character.get("crafting") or {}
            if crafting.get("queue"):
                queued = ", ".join(
                    f"{item.get('count', 1)}x {item.get('recipe')}" for item in crafting["queue"]
                )
                lines.append(f"  crafting {queued}")
        for entry in self.inflight:
            progress = float(entry.get("progress") or 0.0)
            lines.append(f"  in flight: {entry.get('action')} (progress {progress:.2f})")

        lines.append("")
        if self.goal:
            origin = (
                (self.character.get("position") or [0.0, 0.0]) if self.character else [0.0, 0.0]
            )
            lines.append("OBJECTIVE")
            for name in sorted(self.goal):
                point = self.goal[name] or [0.0, 0.0]
                dx = float(point[0]) - float(origin[0])
                dy = float(point[1]) - float(origin[1])
                lines.append(
                    f"  {name} at offset ({dx:+.1f}, {dy:+.1f}), "
                    f"{_distance(dx, dy):.1f} tiles {_compass(dx, dy)}"
                )
        lines.append("")
        if self.inventory:
            lines.append("INVENTORY")
            lines += [f"  {item} x{count}" for item, count in sorted(self.inventory.items())]
        else:
            lines.append("INVENTORY (empty)")

        lines.append("")
        header = f"NEARBY ENTITIES ({len(self.entities)} shown"
        if self.entities_omitted:
            header += f", {self.entities_omitted} further away not listed)"
        else:
            header += ")"
        lines.append(header)
        if not self.entities:
            lines.append("  none in sensor range")
        for row in self.entities:
            offset = row["offset"]
            detail = (
                f"  {row['name']} [{row['handle']}] at offset "
                f"({offset[0]:+.1f}, {offset[1]:+.1f}), "
                f"{row['distance']} tiles {row['bearing']}"
            )
            if row.get("contents"):
                held = ", ".join(f"{k} x{v}" for k, v in sorted(row["contents"].items()))
                detail += f"; holds {held}"
            if row.get("recipe"):
                detail += f"; recipe {row['recipe']}"
            if row.get("status") is not None:
                detail += f"; status {row['status']}"
            if row.get("remembered"):
                detail += f"; REMEMBERED, last seen {row.get('age_ticks', 0)} ticks ago"
            lines.append(detail)

        lines.append("")
        lines.append("RESOURCES")
        if not self.resources:
            lines.append("  none in sensor range")
        for row in self.resources:
            offset = row.get("nearest_offset") or [0.0, 0.0]
            lines.append(
                f"  {row['name']}: {row['tiles']} tiles, nearest at offset "
                f"({offset[0]:+.1f}, {offset[1]:+.1f}), "
                f"{row['distance']} tiles {row.get('bearing')}"
            )

        if self.events:
            lines.append("")
            lines.append("RECENT ACTION OUTCOMES (oldest first)")
            for event in self.events:
                lines.append(
                    f"  tick {event.get('tick')}: {event.get('action')} -> {event.get('status')}"
                )

        lines.append("")
        lines.append(f"LEGAL ACTIONS ({len(self.actions)}); anything else will be rejected")
        for action in self.actions:
            lines.append(f"  {action.index}: {action.key} -- {action.description}")
        return "\n".join(lines)


def summarise(
    observation: dict,
    *,
    brief: TaskBrief,
    actions: tuple[LegalAction, ...],
    step: int,
) -> ObservationSummary:
    """Render one wire observation for a model.

    The signature is the boundary: an observation, a declared brief, the legal
    actions the environment published, and the decision counter. There is no
    parameter for evaluator truth, so the leak PLAN section 2 forbids is not
    expressible here rather than merely avoided.
    """
    character = observation.get("character") or {}
    origin = (
        (observation.get("sensor") or {}).get("origin") or character.get("position") or [0.0, 0.0]
    )
    entities, omitted = _entity_rows(observation, origin)
    return ObservationSummary(
        task=brief.to_dict(),
        goal=observation.get("goal") or {},
        step=step,
        tick=int(observation.get("tick") or 0),
        character=character,
        inventory=dict(observation.get("inventory") or {}),
        entities=entities,
        entities_omitted=omitted,
        resources=_resource_rows(observation, origin),
        inflight=list(observation.get("inflight") or []),
        events=list(observation.get("events") or [])[-MAX_EVENTS_SHOWN:],
        actions=actions,
        # The mod's own progress counters. They are part of the declared
        # observation every policy already sees, not evaluator bookkeeping.
        counters=dict(observation.get("task") or {}),
    )
