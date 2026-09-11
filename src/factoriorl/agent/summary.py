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

from factoriorl.catalog import ARGUMENT_PREFIX, WAIT_FOR

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
    # Needed because two catalog keys share one mod action. `wait` and
    # `wait_for` both dispatch to the mod's `wait`, and they differ by a factor
    # of sixty in what they buy.
    key = getattr(template, "key", "")

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
        # Three things measured on live runs, none of which "mine <handle> (1x)"
        # conveyed -- and one of which I had wrong until the engine corrected me.
        #
        # It removes standing entities, which is how trees, rocks and a machine
        # you want back get cleared. But `item-on-ground` has `minable = true`
        # and **no products**, so mining a loose pile does nothing at all;
        # building over the pile works instead, and `can_place_entity` on a tile
        # holding seven iron ore returned true for every build-check type.
        #
        # And it is *ongoing*. Several in one reply earns `busy` on the second,
        # which one run did four times running -- including
        # `wait+mine_at+wait+mine_at` and `wait_for+mine_at+...`, because a
        # single wait is far shorter than a mine.
        return (
            f"mine {reference(payload.get('handle'))} ({payload.get('count', 1)}x) -- "
            "a resource tile, or a standing entity (tree, rock, or a machine you "
            "want back), which it REMOVES and hands you. It does NOT pick up "
            "loose items lying on the ground. It runs across several decisions: "
            "send ONE per reply and let it finish. A second one in the same "
            "reply is refused as busy, and interleaving waits does not help. "
            "USE A LARGE count -- one mine_at with count 20 keeps mining until "
            "it has 20, across as many decisions as that takes, and costs you "
            "one reply instead of twenty. If the tile runs out first it stops "
            "and tells you how many it got"
        )
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
        # Which way it turns, and -- the part that matters -- what turning it
        # actually changes. A drill that faces the wrong way produces onto bare
        # ground, and "rotate <handle>" said nothing about the connection
        # between facing and the output tile. There is no absolute "face east":
        # this steps by 90 degrees, so getting a specific facing means up to
        # three of these, checking the printed output tile between them.
        way = "anticlockwise" if payload.get("reverse") else "clockwise"
        return (
            f"turn {reference(payload.get('handle'))} 90 degrees {way} -- this "
            "MOVES THE TILE IT OUTPUTS ONTO, which is how you point a drill at "
            "a furnace or an inserter at a machine. The new output tile is "
            "shown next to that machine afterwards, so turn, look, turn again"
        )
    if action == "wait":
        # Two verbs ride on the mod's `wait` and they are not remotely
        # equivalent: `wait` buys one decision interval -- 30 ticks, half a
        # second -- and `wait_for` blocks on a stated condition for up to
        # thirty. Both were described with this same sentence, so the only
        # visible difference was that one needed arguments and the other did
        # not, and a run spent 71 of its 135 decisions on the cheap one. That is
        # roughly 35 seconds of game time bought with 71 model calls.
        if key == WAIT_FOR:
            return (
                "wait until something happens, or until the seconds you name run "
                "out -- WHICHEVER COMES FIRST. This is the one to use: it buys up "
                "to 30 seconds, against the bare `wait` below which buys half a "
                "second. A stone furnace takes 3.2 seconds per plate, so waiting "
                "the other way costs six decisions and six model calls per plate"
            )
        return (
            "do nothing for half a second of game time. This is the smallest "
            "possible step and is almost never what you want -- it is here so a "
            "legal action always exists. To let a machine work, use wait_for"
        )
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

#: Patches named in the charted-map block. Generous: this is the whole reason
#: an agent knows there is coal to walk to, and it is paid for once.
SURVEY_PATCHES_SHOWN = 40

#: Machines listed in the factory block before it is truncated. Generous: this
#: is the agent's own construction and forgetting a machine is how a furnace
#: ends up unfuelled and abandoned twenty tiles away.
MAX_BUILT_SHOWN = 20

#: Handles listed in full before the list is truncated.
#:
#: Twelve rather than twenty since each one now carries its offset and
#: distance. The annotation is what makes the list usable -- an agent standing
#: on coal picked a handle 10.7 tiles away because a bare list said nothing
#: about which was near -- but it triples the width of a row, and the list is
#: sorted nearest first, so the tail was never the part worth reading. Measured:
#: the annotated list at twenty pushed one prompt from 1,971 to 3,999
#: characters, in a history that is re-sent in full on every later turn.
MAX_TARGETS_SHOWN = 12

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
        # Labelled, because a column of bare coordinates is unreadable and the
        # names were already known when the coordinates were chosen. A run had
        # `[-9.5, -111.5]` in this list for its whole length -- the stone patch
        # it needed to build another furnace -- and never went, having written
        # off expansion in its own plan as "practical maximum without stone".
        labels = getattr(env, "_destination_labels", None) or {}
        values = []
        for point in destinations:
            name = labels.get((float(point[0]), float(point[1])))
            values.append(f"[{point[0]}, {point[1]}]" + (f" {name}" if name else ""))
        rendered["destinations"] = {
            "rule": (
                "an ABSOLUTE world coordinate to walk to -- somewhere you have "
                "seen, which may be far outside arm's reach. Unlike placements "
                "these are not limited to 5 tiles. The name after each one says "
                "what is there: if you are short of something, walk to it"
            ),
            "count": len(destinations),
            "values": values,
        }
    for name in ENUMERATED_DOMAINS:
        values = list(domains.get(name) or [])
        if values:
            rendered[name] = values
    return rendered


def objective_block(env: Any) -> str:
    """The standing instruction, for a world that declares one (roadmap A3.1).

    Empty for every benchmark task, and that is the point: a task's objective is
    its `description`, already rendered per turn under TASK, and adding anything
    here would change a frozen prompt and with it what every existing result
    means. Only a `WorldMode` carries an `objective`, and only an open world has
    a `mode`.

    It lives in the static prefix rather than the turn because it never changes.
    At ~700 characters, repeating it would cost ~175 tokens on every turn of a
    run whose history is re-sent in full -- around 26k tokens by turn 300, for
    text that was identical every time.
    """
    mode = getattr(env, "mode", None)
    return str(getattr(mode, "objective", "") or "")


def survey_block(env: Any) -> str:
    """What the charted map holds, rendered once (roadmap A3.1's world state).

    The agent's sensor reaches 32 tiles and the world charts far further, so
    everything past that was invisible. Measured on seed 20260910: coal 79.7
    tiles away, trees 81.0, stone 104.8 -- and the prompt never mentioned stone
    at all, because stone was outside the sensor. Three paid runs spent most of
    their decisions walking in expanding squares looking for fuel through
    territory the force had already mapped.

    This is map-view information: what is out there and roughly where. It
    prescribes nothing -- no route, no order, no build. And it is invariant, so
    it costs one cache miss rather than a line on every turn.
    """
    survey = getattr(env, "survey", None)
    if not survey:
        return ""
    lines = [
        "=== THE CHARTED MAP ===",
        f"Resource patches within {getattr(env, 'survey_radius', '?')} tiles of the",
        "origin (0, 0), surveyed once when the world was made. Your own sensor",
        "reaches far less than this, so most of what follows you cannot see from",
        "where you are standing -- walk to it. Positions are ABSOLUTE.",
        "",
    ]
    for patch in survey[:SURVEY_PATCHES_SHOWN]:
        position = patch.get("position") or [0.0, 0.0]
        distance = _distance(float(position[0]), float(position[1]))
        lines.append(
            f"  {patch.get('name')}: {patch.get('tiles')} tiles centred on "
            f"({position[0]:.0f}, {position[1]:.0f}), {distance:.0f} tiles from the "
            f"origin {_compass(float(position[0]), float(position[1]))}"
        )
    if len(survey) > SURVEY_PATCHES_SHOWN:
        lines.append(f"  (+{len(survey) - SURVEY_PATCHES_SHOWN} more, further out)")
    return "\n".join(lines)


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

    if objective_block(env):
        # Only a world draws a map, and the part of its key that never changes
        # belongs here rather than in every turn: seven lines of explanation
        # re-sent on each of 300 turns is the same cost A2 measured and moved.
        lines += [
            "",
            "READING THE LOCAL MAP",
            "One character per tile, north at the top, @ is you.",
            "  . open ground   ~ water   t tree   r rock   x debris",
            "  * loose items lying on the ground. A machine whose output tile is",
            "    bare floor drops its work there, where nothing can use it, and then",
            "    jams once the tile is full. A pile does NOT stop you building on",
            "    that tile -- put the machine you wanted there and the drill has",
            "    somewhere to put its next one.",
            "  You cannot build on the tile you are standing on. @ is not a legal",
            "  position; step aside first, or pick a position that is not under you.",
            "  A MACHINE IN CAPITALS IS RUNNING. a machine in lower case is stopped:",
            "    built, standing there, producing nothing. Its line says why.",
            "  > < ^ v  the tile a machine puts its output on. A drill whose output",
            "    tile is not the machine you want fed drops onto the ground instead.",
            "    An inserter is the confusing one: it TAKES FROM BEHIND ITSELF and",
            "    PUTS IN FRONT, so both of its tiles are named on its own line rather",
            "    than left to the word 'facing', which points the wrong way for it.",
            "  Two machines only pass items when one's output tile is the other's",
            "  tile. Touching is not enough; the tiles have to match.",
        ]

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


#: Factorio's direction enum, as the mod publishes it. Rendered as words
#: because "facing 4" is not something a reader can act on, and which way a
#: burner drill faces decides which tile its output lands on.
_DIRECTION_NAMES = {
    0: "north",
    2: "northeast",
    4: "east",
    6: "southeast",
    8: "south",
    10: "southwest",
    12: "west",
    14: "northwest",
}


#: The engine's own status word, plus what to do about it.
#:
#: This is deliberately NOT a second table of status names. An earlier version of
#: this file mapped the numeric codes to names invented from memory, got 18
#: wrong, and told a run its furnace was "waiting for space in destination" when
#: the engine meant `no_ingredients` -- so the names come from the engine's
#: reverse lookup of `defines.entity_status` and are only ever passed through.
#: Each key is echoed verbatim at the front of its own value; if a key is ever
#: wrong the line falls back to the engine's word untouched.
#:
#: The remedies exist because a diagnosis with no remedy is a dead end. A burner
#: already got one -- `FUEL SLOT EMPTY -- give it some`. A run placed a lab, was
#: told `no power`, and left it standing there for the rest of the run: it had
#: never seen electricity, nothing said where power comes from, and the word
#: alone named no action.
_STATUS_REMEDY: dict[str, str] = {
    "no_power": (
        "no power -- this machine is ELECTRIC and nothing on the map is generating "
        "electricity. It will never start on its own: it needs a generator built "
        "and electric poles carrying power to it. It has no fuel slot, so giving "
        "it coal does nothing"
    ),
    "no_fuel": "no fuel -- put coal or wood in its fuel slot",
    "no_ingredients": "no ingredients -- it has nothing to work on; put its input in",
    "no_minable_resources": (
        "no minable resources -- there is no ore left under it. Mine it to pick it "
        "back up and place it where ore is under every tile it covers"
    ),
    "waiting_for_space_in_destination": (
        "waiting for space in destination -- its output tile is full and nothing is "
        "taking the output away"
    ),
    "waiting_for_source_items": "waiting for source items -- nothing is arriving for it to move",
    "item_ingredient_shortage": (
        "item ingredient shortage -- it is missing one of the items its recipe needs"
    ),
}


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
            # The handle of *this* tile. It was computed here and dropped, so
            # the prompt could say "coal, nearest 0.5 tiles, here" while the
            # only addressable handles were an unordered list with no
            # positions -- and an agent that picked one got a tile 10.7 tiles
            # away and `out_of_reach`. Measured on the first paid run, five
            # times in a row, with the model insisting the coal was "beneath
            # me". It was. It just could not name it.
            row["handle"] = tile.get("h") or tile.get("handle")
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
    #: A tile-by-tile picture of the immediate surroundings, for a world whose
    #: profile publishes one. Empty for every benchmark task.
    grid: dict = field(default_factory=dict)
    #: Every machine the agent has built, at any distance. Distinct from
    #: `entities`, which is what the sensor can currently see.
    built: list[dict] = field(default_factory=list)
    #: How many of each recipe the character could craft right now, from what
    #: it is holding. The recipe *domain* is every enabled recipe, because
    #: asking for one you cannot afford is a legal request with an informative
    #: refusal -- but "legal to ask for" and "will work" are different, and only
    #: this says which is which.
    craftable: dict = field(default_factory=dict)
    #: For each recipe that cannot be crafted, which ingredients are short and
    #: by how much. `x0` alone says only that *something* is missing, and the
    #: ingredient lists live in a static block the turn does not repeat -- a run
    #: read `stone-furnace x0`, concluded "practical maximum without stone" in
    #: its own plan, and spent its remaining two hundred decisions hauling coal
    #: by hand rather than walking to the stone the survey had charted for it.
    shortfalls: dict = field(default_factory=dict)
    #: How this run's clock works and how much of it is left. "Decision 503 of
    #: 100000" is not planning guidance for a run about to hit a 30-minute
    #: wall, and the system prompt used to assert a fixed decision interval
    #: while the selected mode was continuous time.
    clock: dict = field(default_factory=dict)
    #: What became of the previous reply, member by member. The executor has
    #: always recorded this; the next turn got a bounded engine event log
    #: instead, which can omit most of an eight-action batch -- so the model had
    #: to reconstruct its own last move from someone else's summary.
    receipt: dict = field(default_factory=dict)
    #: The resource tiles as the observation published them, kept because
    #: `resources` above is per-name aggregates and the *handles* live here.
    #: Rendered nowhere directly; used to say where an addressable tile is.
    raw_resources: dict = field(default_factory=dict)
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

    def _receipt_lines(self) -> list[str]:
        """What happened to the reply before this one.

        A2.3 promises the model is told "which ones ran, which one failed and
        which never started". The executor recorded exactly that and the next
        prompt did not carry it.
        """
        members = (self.receipt or {}).get("actions") or []
        if not members:
            return []
        lines = ["", "YOUR LAST REPLY, ACTION BY ACTION"]
        for member in members:
            status = str(member.get("status") or "unknown")
            detail = member.get("action_error") or member.get("detail") or member.get("failure")
            args = member.get("arguments") or {}
            shown = " ".join(f"{k}={v}" for k, v in sorted(args.items())) if args else ""
            line = f"  {member.get('position')}: {member.get('key')}"
            if member.get("target"):
                line += f" -> {member['target']}"
            if shown:
                line += f" ({shown})"
            line += f" -- {status}"
            if detail:
                line += f": {detail}"
            lines.append(line)
        stopped = (self.receipt or {}).get("stopped")
        if stopped:
            lines.append(f"  the rest never started, because the sequence stopped: {stopped}")
        return lines

    def _clock_lines(self) -> list[str]:
        """The clock rule and what is left of it.

        The system prompt points here rather than asserting a rule of its own,
        because the rule differs by run: exactly-stepped play pauses the world
        between decisions and real-time play does not. It asserted the stepped
        rule unconditionally while every open-world run has been real time.
        """
        if not self.clock:
            return []
        lines = []
        if self.clock.get("realtime"):
            lines.append(
                "  CLOCK: REAL TIME. The world keeps running while you think, so "
                "this observation ages"
            )
            lines.append("    as you read it. Time you spend deciding is time your machines run.")
        else:
            lines.append(
                "  CLOCK: STEPPED. The world is paused between decisions and advances a "
                "fixed interval"
            )
            lines.append("    each time you act, so thinking costs you no game time.")
        remaining = self.clock.get("remaining_seconds")
        if remaining is not None:
            minutes, seconds = divmod(max(0, int(remaining)), 60)
            lines.append(
                f"    {minutes}m {seconds}s of run time left. When it runs out the run "
                "stops where it is."
            )
        budget = self.clock.get("remaining_usd")
        if budget is not None:
            lines.append(f"    ${budget:.2f} of spending allowance left.")
        return lines

    def _map_lines(self) -> list[str]:
        """The local map, drawn.

        The rest of the observation gives distances and compass bearings, which
        answer "how far" and never answer "what is next to what". A burner
        drill drops its ore onto one adjacent tile; an agent that cannot see
        adjacency cannot connect one to a furnace, and on the first paid run it
        built both, left them five tiles apart, and never noticed.
        """
        rows = (self.grid or {}).get("rows") or []
        if not rows:
            return []
        origin = (self.grid or {}).get("origin") or [0, 0]
        lines = [
            "",
            "LOCAL MAP -- one character per tile, north at the top, @ is you.",
            f"  the top-left cell is world ({origin[0]}, {origin[1]}); "
            "x grows east (right), y grows south (down)",
        ]
        lines += [f"  {row}" for row in rows]
        legend = (self.grid or {}).get("legend") or []
        if legend:
            lines.append("  key (the fixed part is in the ACTION REFERENCE above):")
            for entry in legend:
                if entry.get("kind") == "resource":
                    lines.append(f"    {entry['glyph']} {entry['name']} (minable)")
                    continue
                where = entry.get("position") or [0, 0]
                facing = entry.get("direction")
                parts = [
                    f"    {entry['glyph']} {entry.get('name')} [{entry.get('handle')}] "
                    f"at ({where[0]:.1f}, {where[1]:.1f})"
                ]
                if facing is not None:
                    parts.append(f"facing {_DIRECTION_NAMES.get(facing, facing)}")
                # The tile, not the compass word. "facing south" does not tell
                # anyone which tile the ore lands on, and a drill pointed at
                # open ground produces into the ground however well fuelled.
                drop = entry.get("drop")
                if drop:
                    parts.append(f"outputs onto ({drop[0]:.1f}, {drop[1]:.1f})")
                pickup = entry.get("pickup")
                if pickup:
                    parts.append(f"takes from ({pickup[0]:.1f}, {pickup[1]:.1f})")
                fuel = entry.get("fuel")
                if fuel == 0 and entry.get("burns"):
                    # Loud, because a machine with no fuel looks exactly like a
                    # working one on the map and the run before this one built
                    # two and never fuelled either.
                    # The status line below now carries the engine's own
                    # diagnosis, so this says only the thing the status cannot:
                    # the slot is empty and that is fixable by the agent.
                    parts.append("FUEL SLOT EMPTY -- give it some")
                elif fuel:
                    parts.append(f"fuel {fuel}")
                # The engine's own name, never a second table. A hand-written
                # one here called 18 "waiting for space in destination" when 18
                # is `no_ingredients` and 34 is the waiting one, so a single
                # prompt described one furnace two different ways in two blocks
                # -- and a report was then written off the wrong half. An
                # unrecognised code stays a code rather than becoming a guess.
                named = entry.get("st")
                if named:
                    parts.append(str(named).replace("_", " "))
                elif entry.get("status") is not None:
                    parts.append(f"status {entry['status']}")
                if entry.get("stopped"):
                    parts.append("STOPPED")
                lines.append(", ".join(parts))
            debris = (self.grid or {}).get("debris") or []
            if debris:
                # One line for the spawn wreckage instead of nine. It is scenery
                # and it was eating most of the alphabet, pushing the machines
                # the agent had actually built down to letters it had to hunt
                # for -- but some pieces hold starting items, so those are named.
                holding = [d for d in debris if d.get("contents")]
                lines.append(
                    f"    x  {len(debris)} pieces of crash-site debris. Not machines, "
                    "and in the way:"
                )
                lines.append("       mine_at removes a piece and gives you what it was made of.")
                for piece in holding[:6]:
                    held = ", ".join(
                        f"{name} x{count}" for name, count in sorted(piece["contents"].items())
                    )
                    lines.append(f"       [{piece['handle']}] holds {held} -- take_from works")
        return lines

    def _factory_lines(self) -> list[str]:
        """Everything the agent built, however far away it now is.

        The map reaches 8 tiles and the sensor 32, so a machine beyond that
        stopped existing as far as the agent was concerned. Measured on a live
        run: it built a second furnace, walked off to find coal, and never
        returned -- the furnace sat unfuelled and out of range for the rest of
        the run, and no observation could have reminded it the thing was there.

        Two machines pass items only when one's output tile is the other's
        tile, so the output tile is printed for each, which is what makes the
        list a wiring diagram rather than an inventory.
        """
        if not self.built:
            return []
        origin = (self.character.get("position") or [0.0, 0.0]) if self.character else [0.0, 0.0]
        lines = ["", f"YOUR FACTORY ({len(self.built)} machines, nearest first)"]
        for record in self.built[:MAX_BUILT_SHOWN]:
            position = record.get("p") or [0.0, 0.0]
            dx = float(position[0]) - float(origin[0])
            dy = float(position[1]) - float(origin[1])
            parts = [
                f"  {record.get('name')} [{record.get('h')}] at "
                f"({position[0]:.1f}, {position[1]:.1f}), {_distance(dx, dy):.1f} tiles "
                f"{_compass(dx, dy)}"
            ]
            covers = record.get("covers")
            if covers:
                # The tiles it actually occupies. Two machines "next to each
                # other" can still miss by a tile, and the agent should not have
                # to know a stone furnace is 2x2 to work that out.
                parts.append(f"covers x {covers[0]}..{covers[2]}, y {covers[1]}..{covers[3]}")
            drop = record.get("drop")
            if drop:
                # What is standing on the output tile, resolved rather than left
                # as arithmetic. Measured on a live run: a drill outputting onto
                # (-28.7, -0.5) with its furnace covering y 0..2 -- one tile
                # short, ore on the floor, and both numbers already in the
                # prompt.
                into = record.get("drop_into")
                if into == "ground":
                    where = " -> BARE GROUND: its output piles up there and it jams"
                elif into:
                    handle = record.get("drop_into_handle")
                    where = f" -> into {into}" + (f" [{handle}]" if handle else "")
                else:
                    where = ""
                parts.append(f"outputs onto ({drop[0]:.1f}, {drop[1]:.1f}){where}")
            pickup = record.get("pickup")
            if pickup:
                parts.append(f"takes from ({pickup[0]:.1f}, {pickup[1]:.1f})")
            # An *absent* fuel inventory is not an empty one. `sensor` returns
            # `fuel` only for entities that burn something, so a chest and an
            # electric drill both arrive with none -- and both were being
            # rendered "NO FUEL", which is a fabricated blocker on a machine
            # that was working fine. Only a burner that reports an empty
            # inventory earns the warning.
            fuel = record.get("fuel")
            if fuel:
                parts.append("fuel " + ", ".join(f"{k} x{v}" for k, v in sorted(fuel.items())))
            elif record.get("burns"):
                parts.append("fuel inventory empty")
            contents = record.get("contents")
            if contents:
                parts.append("holds " + ", ".join(f"{k} x{v}" for k, v in sorted(contents.items())))
            status = record.get("st")
            if status:
                parts.append(_STATUS_REMEDY.get(str(status), str(status).replace("_", " ")))
            elif record.get("status") is not None:
                parts.append(f"status {record['status']}")
            lines.append(", ".join(parts))
        if len(self.built) > MAX_BUILT_SHOWN:
            lines.append(f"  (+{len(self.built) - MAX_BUILT_SHOWN} more, further away)")
        return lines

    def _handle_offsets(self) -> dict[str, tuple[float, float, float]]:
        """Where each addressable handle is, relative to the character.

        Built from the two places a handle can come from: the entity rows and
        the resource tiles. Both carry a position; only the entity rows were
        ever rendered with one.
        """
        origin = (self.character.get("position") or [0.0, 0.0]) if self.character else [0.0, 0.0]
        found: dict[str, tuple[float, float, float]] = {}
        for row in self.entities:
            handle = row.get("handle") or row.get("h")
            offset = row.get("offset")
            if handle and offset:
                found[str(handle)] = (_distance(offset[0], offset[1]), offset[0], offset[1])
        for tile in (self.raw_resources or {}).get("tiles") or []:
            handle = tile.get("h") or tile.get("handle")
            position = tile.get("p")
            if not handle or not position:
                continue
            dx = float(position[0]) - float(origin[0])
            dy = float(position[1]) - float(origin[1])
            found[str(handle)] = (_distance(dx, dy), dx, dy)
        return found

    def _shortfall(self, recipe: str) -> str:
        """`` (need 5 stone, have 0)`` for a recipe the character cannot afford."""
        missing = (self.shortfalls or {}).get(recipe)
        if not missing:
            return ""
        parts = []
        for entry in missing:
            if not isinstance(entry, dict) or not entry.get("name"):
                continue
            need, have = entry.get("need", "?"), entry.get("have", 0)
            parts.append(f"need {need} {entry['name']}, have {have}")
        return f" ({'; '.join(parts)})" if parts else ""

    def _handle_amounts(self) -> dict[str, int]:
        """How much ore each addressable resource tile still holds.

        The mod has measured this per tile since the sensor was written and
        nothing rendered it, so every ore tile looked alike in the prompt. They
        are not alike: a tile at the edge of a patch holds a fraction of one at
        its centre, and a burner drill mines only the four tiles beneath it. Run
        7 put its drill where one tile had ore, reported `no_minable_resources`
        within seconds, and had to pick the drill back up and start again.
        """
        amounts: dict[str, int] = {}
        for tile in (self.raw_resources or {}).get("tiles") or []:
            handle = tile.get("h") or tile.get("handle")
            amount = tile.get("amount")
            if handle is None or amount is None:
                continue
            try:
                amounts[str(handle)] = int(amount)
            except (TypeError, ValueError):
                continue
        return amounts

    def render(self) -> str:
        """The concise text a model is actually shown."""
        task = self.task
        lines: list[str] = [
            f"TASK {task.get('id')} v{task.get('version')}",
            f"  {task.get('description')}",
            f"  decision {self.step}; game tick {self.tick}",
            *self._clock_lines(),
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
            # The request id is the operation's identity and the only thing
            # `cancel_request` accepts. It was published in the `requests`
            # argument domain and rendered nowhere, which left the verb legal
            # and unusable -- there was no way to learn a value for it.
            identity = entry.get("request_id")
            named = f" [{identity}]" if identity else ""
            lines.append(f"  in flight: {entry.get('action')}{named} (progress {progress:.2f})")

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
        lines += self._receipt_lines()
        lines += self._map_lines()
        lines += self._factory_lines()
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
            # The handle of the nearest tile, named. Without it this line
            # described a tile the model had no way to address: it could read
            # "coal, nearest 0.5 tiles, here" and still have nothing to pass to
            # `mine_at` but an unordered list of handles with no positions.
            named = f" [{row['handle']}]" if row.get("handle") else ""
            # How much is actually there. "100 tiles" says nothing about whether
            # the patch is worth a drill; a patch total and the nearest tile's
            # own stock do, and both were already measured and discarded.
            total = row.get("patch_amount") if row.get("patch_amount") else row.get("amount")
            stock = f", {int(total):,} ore" if isinstance(total, int | float) and total else ""
            lines.append(
                f"  {row['name']}: {row['tiles']} tiles{stock}, nearest{named} at offset "
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
                # Already rendered, label and all, by `argument_domains`. This
                # used to format the pair itself, which meant two places decided
                # what a destination looks like -- and when the labels were
                # added here it raised on the first turn of a live world.
                shown = ", ".join(str(value) for value in destinations.get("values") or ())
                lines.append(f"  destination: {shown}")
            # `directions`, `conditions` and `durations` are fixed lists and
            # live in the static reference; only what actually varies with the
            # world is repeated here.
            for name, label in (
                ("items", "item (to give away)"),
                # What is inside something else, which is what `take_from`
                # accepts. Rendered separately because the two `item` domains
                # genuinely differ and showing only the first made collecting a
                # new product look impossible.
                ("source_items", "item (to take from a machine or chest)"),
                ("amounts", "count"),
                ("recipes", "recipe"),
                # Empty whenever nothing is running, which is most turns.
                ("requests", "target_request_id"),
                # Research the force can start now: prerequisites met, not yet
                # researched. Empty until a profile publishes the frontier,
                # which is what kept `research` unreachable.
                ("technologies", "technology"),
            ):
                values = self.arguments.get(name)
                if not values:
                    continue
                if name == "recipes" and self.craftable:
                    # Affordable first, and every one carrying how many you
                    # could make. A live run asked for a stone furnace it had
                    # no stone for, was told `craftable: 0`, and asked again --
                    # the list said the recipe was available and never said it
                    # was out of reach.
                    ordered = sorted(values, key=lambda v: (-self.craftable.get(v, 0), str(v)))
                    shown = ", ".join(
                        f"{v} x{self.craftable.get(v, 0)}{self._shortfall(str(v))}"
                        for v in ordered[:24]
                    )
                    more = "" if len(ordered) <= 24 else f" (+{len(ordered) - 24} more)"
                    lines.append(f"  {label}: {shown}{more}")
                    lines.append(
                        "    xN is how many you can make right now from what you hold; "
                        "x0 means you are missing ingredients, and the bracket after "
                        "it names them -- go and get them"
                    )
                    continue
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
                # Annotated and ordered by distance. A bare list of handles is
                # unusable for resource tiles: they are not in the ENTITIES
                # block, they arrive in the engine's own order rather than by
                # distance, and `mine` refuses anything past 2.7 tiles. An
                # agent standing *on* coal picked a handle 10.7 tiles away and
                # was refused five times running, because nothing here said
                # which handle was the near one.
                placed = self._handle_offsets()
                amounts = self._handle_amounts()
                ordered = sorted(
                    (str(t) for t in targets),
                    key=lambda handle: placed.get(handle, (float("inf"), 0.0, 0.0))[0],
                )
                rendered = []
                for handle in ordered[:MAX_TARGETS_SHOWN]:
                    found = placed.get(handle)
                    if found is None:
                        rendered.append(handle)
                        continue
                    distance, dx, dy = found
                    # `x312` is how much ore is left in that tile. Without it
                    # every tile in the list looks equally worth mining or
                    # standing a drill on, and the thin ones at a patch edge
                    # look exactly like the rich ones at its centre.
                    held = amounts.get(handle)
                    stock = f" x{held}" if held is not None else ""
                    rendered.append(f"{handle}@({dx:+.1f},{dy:+.1f}) {distance:.1f}t{stock}")
                more = (
                    ""
                    if len(ordered) <= MAX_TARGETS_SHOWN
                    else f" (+{len(ordered) - MAX_TARGETS_SHOWN} more, further away)"
                )
                lines.append(f"  handle / from / to: {', '.join(rendered)}{more}")
                lines.append(
                    "    nearest first; @(dx,dy) is the offset from you, the "
                    "number is the distance in tiles, and xN is how much ore "
                    "that tile still holds"
                )
        return "\n".join(lines)


def summarise(
    observation: dict,
    *,
    brief: TaskBrief,
    actions: tuple[LegalAction, ...],
    step: int,
    arguments: dict | None = None,
    requires: dict | None = None,
    clock: dict | None = None,
    receipt: dict | None = None,
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
        raw_resources=dict(observation.get("resources") or {}),
        grid=dict(observation.get("grid") or {}),
        built=list(observation.get("built") or []),
        clock=dict(clock or {}),
        receipt=dict(receipt or {}),
        craftable={
            entry["name"]: entry.get("craftable", 0)
            for entry in (observation.get("recipes") or [])
            if isinstance(entry, dict)
        },
        shortfalls={
            entry["name"]: entry["missing"]
            for entry in (observation.get("recipes") or [])
            if isinstance(entry, dict) and entry.get("missing")
        },
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
