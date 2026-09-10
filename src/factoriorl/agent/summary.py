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

from factoriorl.catalog import ARGUMENT_PREFIX

#: Bumped when what the language-model client is *shown* changes, even if the
#: prompt text does not. `model.system_prompt_digest` pins the prompt and not
#: this file, so how many entities are listed, which of their fields are
#: surfaced, and whether every marker or one selected marker appears were all
#: unpinned -- while the RL client had `goal_encoding` and `extractor_version`.
#: Two clients whose encodings are not comparably pinned cannot be compared.
#: 2 added argument domains, for `parameterized-v1`. A model shown a
#: `place_at` it cannot supply a position for has no legal answer, so the
#: encoding version is part of what a run's numbers mean.
#: 3 surfaces the per-entity status *name*, `working`, `fuel` and `output`.
#: `local-v2` v4 was bumped to publish exactly those four -- "what a stopped
#: machine's cause actually is" -- and `_entity_row` never copied them, so the
#: prompt kept rendering `status 18` through every language-model run to date,
#: including R3.2's baseline and R4.2's first agent arm. Those numbers were
#: measured against a prompt that could not state why a machine had stopped.
SUMMARY_ENCODING_VERSION = 3

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
        if isinstance(value, str) and value.startswith(ARGUMENT_PREFIX):
            # A `parameterized-v1` argument: the *model* supplies this, unlike
            # a `$` reference the environment binds. Rendering it as the raw
            # sigil told a model nothing about whose job it was.
            return f"<{value[1:]} you choose>"
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
        return (
            f"craft {reference(payload.get('count', 1))}x {reference(payload.get('recipe', '?'))}"
        )
    if action == "place":
        item = reference(payload.get("item", "?"))
        where = reference(payload.get("position"))
        facing = payload.get("direction")
        rendered = f"place {item} at {where}"
        return f"{rendered} facing {reference(facing)}" if facing else rendered
    if action == "transfer":
        # Every field through `reference`, not just the endpoints: a
        # parameterized transfer's count and item are `?` arguments too, and
        # rendering them raw put "move ?countx ?item" in the prompt.
        return (
            f"move {reference(payload.get('count', '?'))}x "
            f"{reference(payload.get('item', '?'))} "
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


#: How many placement candidates to show as examples. The domain is a disc of
#: tile centres around the character -- 121 of them at `PLACEMENT_RADIUS` 5 --
#: and enumerating it would be most of the prompt for the least information.
#: The rule plus the entity list is what the environment itself derives the
#: domain from, so the model is shown the same basis; a value outside it comes
#: back as a named decision failure with the domain's size, not as a crash.
PLACEMENT_EXAMPLES = 8

#: Repeated in the static reference's wording of the placement rule. Mirrors
#: `env.PLACEMENT_RADIUS`, stated here because the prompt says it in prose.
PLACEMENT_RULE_TILES = 5

#: Handles listed in full before the list is truncated.
MAX_TARGETS_SHOWN = 20

#: Domains small enough to enumerate in full. `placements` is the exception
#: above; `recipes` is capped by the observation profile already.
ENUMERATED_DOMAINS = (
    "directions",
    "amounts",
    "items",
    "recipes",
    "targets",
    # What `research` may name. Empty until a profile publishes the researchable
    # frontier, which is why the verb was unreachable for so long.
    "technologies",
    # `wait_for`'s two arguments. Both are short fixed lists, and enumerating
    # them is the difference between a domain the model can choose from and one
    # it has to guess at.
    "conditions",
    "durations",
)


def argument_domains(env: Any) -> dict:
    """What values the model may supply, from the environment's own domains.

    Read straight off `env.argument_domains()`, which is derived from the
    observation, so this cannot widen what a policy could see -- the same
    property `test_useful_observations.py` asserts for the RL path.
    """
    if not any(getattr(t, "parameterized", False) for t in env.catalog.templates):
        return {}
    domains = env.argument_domains()
    placements = list(domains.get("placements") or [])
    rendered: dict = {
        "placements": {
            "rule": (
                "an ABSOLUTE world coordinate, not an offset -- any tile centre "
                "within 5 tiles of you with nothing standing on it, written "
                "[x.5, y.5]. Your own absolute position is under CHARACTER"
            ),
            "count": len(placements),
            "examples": [list(p) for p in placements[:PLACEMENT_EXAMPLES]],
        }
    }
    destinations = list(domains.get("destinations") or [])
    if destinations:
        # Listed in full rather than sampled like `placements`, because a
        # destination the prompt omits is somewhere the agent cannot go. The
        # domain is capped at the source (`env.DESTINATION_CAP`).
        rendered["destinations"] = {
            "rule": (
                "an ABSOLUTE world coordinate to walk to -- somewhere you have "
                "seen, which may be far outside arm's reach. Unlike placements "
                "these are not limited to 5 tiles"
            ),
            "count": len(destinations),
            "values": [list(p) for p in destinations],
        }
    for name in ENUMERATED_DOMAINS:
        values = list(domains.get(name) or [])
        if values:
            rendered[name] = values
    return rendered


def static_reference(env: Any) -> str:
    """The half of the action surface that never changes, rendered once.

    Every request carries the whole conversation, so a line repeated in each
    turn is a line paid for on every later turn too. Measured on a real
    `open_factory` prompt: of 2,731 characters, 1,440 were invariant -- verb
    descriptions, the coordinate rules, and the fixed enumerations. Over a
    300-turn run that is most of the accumulated input.

    Moving them here costs one cache miss and nothing afterwards, and it is the
    same argument the static game knowledge already makes. It also matters for
    reasons other than money: a history that grows at 722 tokens a turn reaches
    660k tokens over a 30-minute realtime run, and the context window is 1M.
    """
    vocabulary = action_vocabulary(env)
    requires = argument_requirements(env)
    lines = [
        "=== ACTION REFERENCE ===",
        "What each verb does. Which of them are legal *right now* changes every",
        "turn and is listed under LEGAL ACTIONS in the observation.",
        "",
    ]
    for index, (key, description) in enumerate(vocabulary):
        needed = requires.get(key) or ()
        suffix = f"  [needs: {', '.join(needed)}]" if needed else ""
        lines.append(f"  {index}: {key} -- {description}{suffix}")

    domains = env.argument_domains() if hasattr(env, "argument_domains") else {}
    lines += [
        "",
        "ARGUMENT RULES -- an action marked [needs: ...] must be answered with an",
        '"arguments" object supplying exactly those names.',
        "  position: an ABSOLUTE world coordinate, not an offset -- a tile centre",
        f"    within {PLACEMENT_RULE_TILES} tiles of you with nothing standing on it,",
        "    written [x.5, y.5]. Your own position is under CHARACTER.",
        "  destination: an ABSOLUTE world coordinate to walk to -- somewhere you",
        "    have seen. Unlike a placement it may be far outside arm's reach.",
        "  handle / from / to: an entity handle, listed under ENTITIES, or a",
        "    resource tile handle.",
    ]
    for name, label in (
        ("directions", "direction"),
        ("conditions", "until"),
        ("durations", "seconds"),
    ):
        values = list(domains.get(name) or [])
        if values:
            lines.append(f"  {label}: {', '.join(str(v) for v in values)}")
    return "\n".join(lines)


def argument_requirements(env: Any) -> dict[str, tuple[str, ...]]:
    """Which arguments each action key needs, so the prompt can say so."""
    return {
        template.key: tuple(template.arguments)
        for template in env.catalog.templates
        if getattr(template, "arguments", ())
    }


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


def targetable_actions(env: Any) -> frozenset[str]:
    """Catalog keys whose action binds to an entity.

    These are the templates carrying `$target`, which the catalog resolves to
    the *nearest* entity because a discrete index cannot carry an argument. They
    are exactly the actions worth letting a model address by name.
    """
    keys = set()
    for template in env.catalog.templates:
        if any(value == "$target" for value in template.payload.values()):
            keys.add(template.key)
    return frozenset(keys)


def visible_handles(observation: dict) -> frozenset[str]:
    """Handles the agent can currently see, and may therefore name."""
    return frozenset(
        str(entity.get("h") or entity.get("handle"))
        for entity in (observation.get("entities") or [])
        if entity.get("h") or entity.get("handle")
    )


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
    # The three fields the v4 profile bump was made for. `sensor.entity_record`
    # publishes all of them and `local-v2` v5 declares `entities`, so they were
    # arriving on the wire and being dropped here -- which left the prompt
    # rendering `status 18`, the exact string `sensor.lua`'s own comment says
    # "neither client could act on". A stopped machine's *cause* lives in
    # `fuel` and `output`; without them "the drill has no fuel" and "the furnace
    # has no ore" are the same unreadable integer.
    if record.get("st") is not None:
        row["st"] = record["st"]
    if record.get("working") is not None:
        row["working"] = bool(record["working"])
    if record.get("fuel"):
        row["fuel"] = record["fuel"]
    if record.get("output"):
        row["output"] = record["output"]
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
    resources = observation.get("resources") or {}
    grouped: dict[str, dict] = {}
    for tile in resources.get("tiles") or []:
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

    # Anything further away than `resource_detail_radius` -- 12 tiles, for both
    # observation profiles -- reaches the observation only as a per-name
    # aggregate under `resources.patches`. Until now nothing read it: the mod's
    # own comment says so ("`encoders.encode` and every task predicate read
    # `resources.tiles`; nothing reads `patches`"), and neither did this.
    #
    # On a benchmark scene that never mattered, because a declared scene places
    # its ore within a few tiles of the character. On a generated map it is
    # fatal: measured on a natural spawn, the nearest iron ore is 28 tiles away,
    # so the prompt said "RESOURCES: none in sensor range" while 29 resource
    # entities sat inside the sensor radius. An agent told there is no ore does
    # not go looking for ore.
    # Keyed by name on the wire, because the mod builds it as a Lua table keyed
    # by resource name. Both shapes are accepted so a future profile that emits
    # a list does not silently render nothing.
    raw_patches = resources.get("patches") or {}
    patch_rows = raw_patches.values() if isinstance(raw_patches, dict) else raw_patches
    for patch in patch_rows:
        if not isinstance(patch, dict):
            continue
        name = patch.get("name", "unknown")
        if name in grouped:
            # Detail wins: if some tiles are close enough to address directly,
            # the exact nearest tile is more useful than the patch average.
            grouped[name]["patch_tiles"] = patch.get("count")
            grouped[name]["patch_amount"] = patch.get("total")
            continue
        nearest = patch.get("nearest") or [0.0, 0.0]
        dx = float(nearest[0]) - float(origin[0])
        dy = float(nearest[1]) - float(origin[1])
        grouped[name] = {
            "name": name,
            "tiles": patch.get("count", 0),
            "distance": round(float(patch.get("nearest_d") or _distance(dx, dy)), 1),
            "nearest_offset": [round(dx, 1), round(dy, 1)],
            "bearing": _compass(dx, dy),
            "amount": patch.get("total"),
            # Marked, because the offset is to the nearest tile of the patch
            # rather than to a tile this observation can address by handle. A
            # model that tries to name a handle for it should be told why there
            # is not one.
            "aggregate": True,
        }
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
    #: Legal values for each argument a parameterized action needs. Empty for
    #: a discrete catalog, where the environment binds every reference itself.
    arguments: dict = field(default_factory=dict)
    #: Which arguments each action key requires.
    requires: dict = field(default_factory=dict)
    #: The most recent `inspect`, carried forward with the step it was taken at.
    #: The observation shows the twelve nearest entities, so on a built-up map
    #: most of what exists is not in the prompt, and this is how the agent reads
    #: the rest.
    inspected: dict | None = None

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
            "inspected": self.inspected,
            "inflight": self.inflight,
            "events": self.events,
            "counters": self.counters,
            "actions": [a.to_dict() for a in self.actions],
            "arguments": self.arguments,
            "requires": {k: list(v) for k, v in self.requires.items()},
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
            if row.get("st") or row.get("status") is not None:
                # Name first: `status 18` is not actionable and
                # `waiting_for_source_items` is. The integer stays only as the
                # fallback for a status the engine has no name for.
                detail += f"; status {row.get('st') or row['status']}"
            if row.get("fuel"):
                fuel = ", ".join(f"{k} x{v}" for k, v in sorted(row["fuel"].items()))
                detail += f"; fuel {fuel}"
            # No "fuel: empty" line when the field is absent, deliberately.
            # `sensor.inventory_contents` returns nil both for an empty fuel
            # inventory and for an entity that has none, so an absent field
            # cannot tell a drained drill from a transport belt. The engine's
            # own status name carries that diagnosis instead: a burner out of
            # fuel reports `no_fuel`.
            if row.get("output"):
                held = ", ".join(f"{k} x{v}" for k, v in sorted(row["output"].items()))
                detail += f"; output {held}"
            if row.get("remembered"):
                detail += f"; REMEMBERED, last seen {row.get('age_ticks', 0)} ticks ago"
            lines.append(detail)

        lines.append("")
        lines.append("RESOURCES")
        if not self.resources:
            lines.append("  none in sensor range")
        for row in self.resources:
            offset = row.get("nearest_offset") or [0.0, 0.0]
            # An aggregate row has no addressable handle: it is further out than
            # the detail radius, so the only way to act on it is to walk there
            # first. Saying so beats letting the model discover it by having an
            # action rejected.
            note = " (too far to address; walk closer)" if row.get("aggregate") else ""
            lines.append(
                f"  {row['name']}: {row['tiles']} tiles, nearest at offset "
                f"({offset[0]:+.1f}, {offset[1]:+.1f}), "
                f"{row['distance']} tiles {row.get('bearing')}{note}"
            )

        if self.inspected:
            lines.append("")
            age = self.step - int(self.inspected.get("at_step") or self.step)
            seen = self.inspected.get("seen")
            when = "this step" if age <= 0 else f"{age} steps ago"
            lines.append(f"INSPECTED ({when}, {seen})")
            for key, value in sorted(self.inspected.items()):
                if key in ("at_step", "seen"):
                    continue
                lines.append(f"  {key}: {value}")

        if self.events:
            lines.append("")
            lines.append("RECENT ACTION OUTCOMES (oldest first)")
            for event in self.events:
                lines.append(
                    f"  tick {event.get('tick')}: {event.get('action')} -> {event.get('status')}"
                )

        lines.append("")
        # Index and key only. What each verb *does*, and which arguments it
        # needs, is the same on every turn and lives in the static prefix --
        # measured at 769 characters of description repeated into a history that
        # is re-sent in full on every request. The legal *set* is what changes,
        # so the set is what is sent.
        lines.append(f"LEGAL ACTIONS ({len(self.actions)}); anything else will be rejected")
        lines.append("  " + ", ".join(f"{a.index}: {a.key}" for a in self.actions))
        if self.arguments:
            lines.append("")
            lines.append(
                "ARGUMENT VALUES -- an action marked [needs: ...] must be answered with "
                'an "arguments" object supplying exactly those names'
            )
            placements = self.arguments.get("placements") or {}
            if placements:
                examples = ", ".join(
                    f"[{p[0]:.1f}, {p[1]:.1f}]" for p in placements.get("examples") or ()
                )
                # The rule is in the static reference; only the count and a
                # few live examples change.
                lines.append(f"  position: {placements.get('count')} legal now, e.g. {examples}")
            destinations = self.arguments.get("destinations") or {}
            if destinations:
                # Listed in full rather than sampled like `placements`: a
                # destination the prompt omits is somewhere the agent cannot go,
                # and the domain is already capped at its source. On the map
                # this exists for, the only entry at spawn is the iron ore 28
                # tiles away -- omitting it would leave the agent nothing to
                # walk to at all.
                shown = ", ".join(f"[{p[0]:.1f}, {p[1]:.1f}]" for p in destinations.get("values"))
                lines.append(f"  destination: {shown}")
            # `directions`, `conditions` and `durations` are fixed lists and
            # live in the static reference; only what actually varies with the
            # world is repeated here.
            for name, label in (
                ("items", "item"),
                ("amounts", "count"),
                ("recipes", "recipe"),
                # Research the force can start now: prerequisites met, not yet
                # researched. Empty until a profile publishes the frontier,
                # which is what kept `research` unreachable.
                ("technologies", "technology"),
            ):
                values = self.arguments.get(name)
                if values:
                    shown = ", ".join(str(v) for v in values[:24])
                    more = "" if len(values) <= 24 else f" (+{len(values) - 24} more)"
                    lines.append(f"  {label}: {shown}{more}")
            targets = self.arguments.get("targets")
            if targets:
                # Enumerated, not described. "any handle in the ENTITIES list"
                # was wrong whenever the legal set included resource tiles,
                # which are listed under RESOURCES without handles -- so at the
                # first decision of `build_line` the model was told to name a
                # handle from an empty list while 31 were legal.
                shown = ", ".join(str(t) for t in targets[:MAX_TARGETS_SHOWN])
                more = (
                    ""
                    if len(targets) <= MAX_TARGETS_SHOWN
                    else f" (+{len(targets) - MAX_TARGETS_SHOWN} more)"
                )
                lines.append(f"  handle / from / to: {shown}{more}")
        return "\n".join(lines)


def summarise(
    observation: dict,
    *,
    brief: TaskBrief,
    actions: tuple[LegalAction, ...],
    step: int,
    arguments: dict | None = None,
    requires: dict | None = None,
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
        inspected=observation.get("inspected"),
        inflight=list(observation.get("inflight") or []),
        events=list(observation.get("events") or [])[-MAX_EVENTS_SHOWN:],
        actions=actions,
        # The mod's own progress counters. They are part of the declared
        # observation every policy already sees, not evaluator bookkeeping.
        counters=dict(observation.get("task") or {}),
        # Derived by the caller from `env.argument_domains()`, which is a
        # function of the observation -- passed in rather than computed here so
        # this function still takes no environment and cannot reach truth.
        arguments=dict(arguments or {}),
        requires=dict(requires or {}),
    )
