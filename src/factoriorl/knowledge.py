"""What the game *is*, as opposed to what the world currently looks like.

The mod has always been able to read `prototypes.*` and has never returned any
of it. Three call sites in `actions.lua` look something up -- a recipe's first
product, an item's `place_result` -- and throw it away as soon as the action
answers. The consequences were paid elsewhere: a client had no way to learn that
an iron gear wheel costs two plates except by attempting the craft, and
`tasks/spec.py` carries a hardcoded `ENTITY_TILE_SIZES` table of four prototypes
with a 1x1 fallback for everything else, because the footprint of a stone
furnace was not askable. `mod/factoriorl/knowledge.lua` returns it; this module
fetches, caches and renders it.

Why it is cached on disk
------------------------
The reply is the largest the protocol produces and it does not change while a
worker runs. Fetching it once per run and reusing the file is worth a little
machinery, but only if the key is honest about what can invalidate it: the
**engine build**, because prototypes come from the game, and
`manifest.mod_source_digest()`, because a change to `knowledge.lua` changes what
is returned from the same game. Keying on the build alone would serve a stale
file after any edit to the mod, and the difference would be invisible -- the
JSON would parse and simply be missing a field.

Why `render` must be byte-stable
--------------------------------
Its output goes into a prompt prefix that the provider caches. A reordered line
is indistinguishable from a changed one to a cache key, so a single `pairs`
iteration leaking through would invalidate the entire cached conversation on
every turn and be visible only as a bill. `knowledge.lua` sorts each collection
before it serialises; this module sorts again on the way out rather than trust
that, because a cache file written by an older mod is exactly the input nobody
would think to check.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from factoriorl.manifest import mod_source_digest
from factoriorl.paths import runtime_dir

#: The three tables `knowledge.lua` returns. Each is required to be non-empty:
#: no Factorio install has zero recipes, so an empty one means the Lua walked a
#: table it could not read and returned success anyway.
SECTIONS = ("recipes", "placeable", "technologies")

#: Cache filenames are derived from an engine build string, which comes from the
#: engine rather than from this repository. Anything outside this set is
#: replaced rather than trusted into a path.
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


class KnowledgeUnavailable(RuntimeError):
    """The worker could not supply the static game data.

    Raised rather than defaulted, for the reason `freeplay.FreeplayUnreadable`
    gives: a fabricated recipe graph would make a run look like it was planning
    against the real game while it was not, and no later measurement could tell.
    """


# --------------------------------------------------------------- fetching


def fetch(session: Any) -> dict:
    """One `knowledge` request against a live worker.

    Read-only, so it needs no episode and can be asked before the first reset.
    """
    timed = session.knowledge()
    response = timed.response
    if not response.ok:
        detail = response.error.message if response.error else response.code
        raise KnowledgeUnavailable(f"the worker refused the knowledge request: {detail}")
    result = response.result or {}
    missing = [section for section in SECTIONS if not _rows(result, section)]
    if missing:
        raise KnowledgeUnavailable(
            f"the knowledge reply has no {', '.join(missing)}. Every Factorio install "
            f"has hundreds of each, so an empty table is a read that silently failed "
            f"rather than a game with nothing in it"
        )
    return result


# ---------------------------------------------------------------- caching


def cache_key(engine_build: str, mod_digest: str | None = None) -> str:
    """What a cached answer is allowed to be reused for.

    Both halves matter. The prototypes come from the engine, so a different
    build is a different answer; and `knowledge.lua` decides which fields are
    reported, so a mod edit is a different answer from the same engine.
    """
    digest = mod_digest if mod_digest is not None else mod_source_digest()
    return f"{_UNSAFE.sub('-', engine_build or 'unknown')}-{_UNSAFE.sub('-', digest)}"


def cache_path(engine_build: str, mod_digest: str | None = None) -> Path:
    return runtime_dir() / "knowledge" / f"{cache_key(engine_build, mod_digest)}.json"


def cached(session: Any, engine_build: str, mod_digest: str | None = None) -> dict:
    """Fetch once per (engine build, mod digest) and reuse the file thereafter.

    A cache miss, an unreadable file and a file whose stored key disagrees with
    the one asked for all mean the same thing: fetch. The key is stored inside
    the file as well as in its name because the name is slugged, so two builds
    whose strings differ only in a character this module rewrites would
    otherwise share one entry.
    """
    key = cache_key(engine_build, mod_digest)
    path = cache_path(engine_build, mod_digest)
    stored = _read_cache(path, key)
    if stored is not None:
        return stored

    knowledge = fetch(session)
    document = {
        "key": key,
        "engine_build": engine_build,
        "mod_digest": mod_digest if mod_digest is not None else mod_source_digest(),
        "knowledge": knowledge,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Written whole and moved into place: a crash mid-write would otherwise
        # leave a truncated file that the next run has to recognise and repair.
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(path)
    except OSError:
        # Failing to cache is not failing to fetch. The caller has its answer.
        pass
    return knowledge


def _read_cache(path: Path, key: str) -> dict | None:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(document, dict) or document.get("key") != key:
        return None
    knowledge = document.get("knowledge")
    if not isinstance(knowledge, dict):
        return None
    if any(not _rows(knowledge, section) for section in SECTIONS):
        return None
    return knowledge


# --------------------------------------------------------------- rendering


def render(knowledge: dict) -> str:
    """A prompt-sized plain-text block. Byte-identical for equal input.

    Deterministic by construction: every collection is sorted by name here, not
    only in Lua, and every number goes through `_number` so a float that arrives
    as `0.5` and one that arrives as `0.5000000001` cannot render differently on
    two runs of the same worker.
    """
    lines = [
        "=== STATIC GAME KNOWLEDGE ===",
        "Factorio prototype data: what things cost, what they make, how big they",
        "are, and what unlocks what. The same on every turn, and nothing you do",
        "changes it.",
        "",
        "It says nothing about what is available to you *now*. The recipes you can",
        "currently craft are in the observation under RECIPES, and the research you",
        "can start is under ARGUMENT VALUES. Read availability there, not here.",
        "",
    ]
    lines += _recipe_section(knowledge)
    lines.append("")
    lines += _placeable_section(knowledge)
    lines.append("")
    lines += _technology_section(knowledge)
    return "\n".join(lines)


def _recipe_section(knowledge: dict) -> list[str]:
    rows = _rows(knowledge, "recipes")
    lines = [f"RECIPES ({len(rows)}) -- name: ingredients -> products (craft seconds, category)"]
    for row in rows:
        name = str(row.get("name", ""))
        ingredients = _stacks(row.get("ingredients")) or "-"
        # The product name is dropped when it repeats the recipe name, which is
        # the overwhelming majority of recipes and is the difference between a
        # line the model reads once and a line it reads twice.
        products = _stacks(row.get("products"), omit=name) or "-"
        line = (
            f"{name}: {ingredients} -> {products} "
            f"({_number(row.get('energy'))}s, {row.get('category', 'crafting')})"
        )
        # No lock marker here, and none for entities or technologies below.
        # `enabled`, `unlocked` and `researched` are force state read off
        # `game.forces["player"]`, not prototype state, and they move the
        # moment research completes. This block is captured once and pinned
        # into a prompt prefix that must stay byte-identical for the whole
        # run, so a marker recorded here would be quietly wrong for most of
        # it. Availability is published every turn in the observation --
        # enabled recipes under RECIPES, the researchable frontier under
        # ARGUMENT VALUES -- which is where a reader should look for it.
        if row.get("hidden"):
            # Still craftable by a machine; only kept out of the crafting menu.
            line += " [hidden]"
        lines.append(line)
    return lines


def _placeable_section(knowledge: dict) -> list[str]:
    rows = _rows(knowledge, "placeable")
    lines = [
        f"PLACEABLE ENTITIES ({len(rows)}) -- item: tiles wide x tall, "
        f"footprint centred on the placement position"
    ]
    for row in rows:
        name = str(row.get("name", ""))
        line = f"{name}: {_number(row.get('width', 1))}x{_number(row.get('height', 1))}"
        entity = row.get("entity")
        # Named only when it differs. `place` takes the item name, so repeating
        # an identical entity name on every line would be noise the agent has to
        # read past on all ~200 of them.
        if entity and entity != name:
            line += f" -> {entity}"
        # The stricter rule `actions.lua` applies: placement is refused while a
        # same-named recipe is locked, which base Factorio does not do.
        lines.append(line)
    return lines


def _technology_section(knowledge: dict) -> list[str]:
    rows = _rows(knowledge, "technologies")
    lines = [f"TECHNOLOGIES ({len(rows)}) -- name: cost -> recipes unlocked <- prerequisites"]
    for row in rows:
        name = str(row.get("name", ""))
        line = f"{name}: {_research_cost(row)}"
        unlocks = sorted(str(entry) for entry in _list(row.get("unlocks")))
        if unlocks:
            line += " -> " + ", ".join(unlocks)
        prerequisites = sorted(str(entry) for entry in _list(row.get("prerequisites")))
        if prerequisites:
            line += " <- " + ", ".join(prerequisites)
        lines.append(line)
    return lines


def _research_cost(row: dict) -> str:
    """The cost line for one technology, including the ones that have no cost.

    A trigger technology is reported as its trigger rather than as a science
    cost, because it genuinely has none: 2.0 completes 7 of 196 by crafting or
    mining an item, and they cannot be queued at all. `protocol.py` carries
    `tech_not_selectable` for exactly this case, and an agent told only "not
    selectable" has nothing to act on.
    """
    trigger = row.get("trigger")
    if trigger:
        # What actually satisfies it. `trigger:craft-item` told an agent it
        # could not research the thing and nothing about how to obtain it, which
        # is the half that matters -- these unlock by *doing* something, and the
        # something is knowable.
        subject = row.get("trigger_item") or row.get("trigger_entity")
        count = row.get("trigger_count")
        verb = str(trigger).replace("-", " ")
        if subject and count:
            return f"no science: {verb} {_number(count)}x {subject}"
        if subject:
            return f"no science: {verb} {subject}"
        return f"no science: completes by {verb}"
    count = row.get("unit_count")
    ingredients = _list(row.get("unit_ingredients"))
    if not ingredients:
        return "free" if count is None else f"{_number(count)}x -"
    packs = "+".join(_stack(entry, brief=True) for entry in ingredients)
    return f"{_number(count) if count is not None else '?'}x {packs}"


# ----------------------------------------------------------------- helpers


def _rows(knowledge: dict, section: str) -> list[dict]:
    """One of the three tables, name-sorted, whatever shape the JSON arrived in.

    Lua's empty table serialises as a JSON *object*, not an array, so an install
    with nothing in a section hands back `{}` where every other reply hands back
    a list. Normalised here rather than at each of the five call sites.
    """
    value = knowledge.get(section)
    if isinstance(value, dict):
        value = list(value.values())
    if not isinstance(value, list):
        return []
    rows = [row for row in value if isinstance(row, dict)]
    return sorted(rows, key=lambda row: str(row.get("name", "")))


def _list(value: Any) -> list:
    if isinstance(value, dict):
        return list(value.values())
    return list(value) if isinstance(value, list) else []


def _stacks(entries: Any, omit: str | None = None) -> str:
    return " + ".join(
        _stack(entry, omit=omit) for entry in _list(entries) if isinstance(entry, dict)
    )


def _stack(entry: dict, omit: str | None = None, brief: bool = False) -> str:
    """One ingredient or product: `2 iron-plate`, and the awkward cases.

    A probabilistic product carries a range and a chance instead of an amount --
    uranium processing, every ore-crushing recipe -- and rendering the missing
    `amount` as nothing would read as a recipe that produces one of something.
    """
    name = str(entry.get("name", ""))
    if brief:
        return name
    amount = entry.get("amount")
    low, high = entry.get("amount_min"), entry.get("amount_max")
    if low is not None and high is not None and low != high:
        rendered = f"{_number(low)}-{_number(high)}"
    else:
        rendered = _number(amount)
    label = name
    if entry.get("type") == "fluid":
        label += "(fluid)"
    probability = entry.get("probability")
    if isinstance(probability, int | float) and probability < 1:
        label += f"@{_number(probability * 100)}%"
    # The product of a recipe named after it: `iron-gear-wheel: 2 iron-plate -> 1`.
    # Only when the label is the bare name -- a fluid or a chance is information
    # the recipe's own name does not carry, so it stays.
    if omit is not None and label == omit:
        return rendered
    return f"{rendered} {label}"


def _number(value: Any) -> str:
    """Numbers that render the same way twice.

    Whole floats lose their `.0` -- `2.0 iron-plate` is a token spent on
    nothing -- and the rest are rounded before formatting, because recipe
    energies arrive from the engine as floats and `0.30000000000000004` would be
    both unreadable and a different string on a machine that rounded elsewhere.
    """
    if value is None:
        return "?"
    if isinstance(value, bool) or not isinstance(value, int | float):
        return str(value)
    rounded = round(float(value), 4)
    if rounded == int(rounded):
        return str(int(rounded))
    return f"{rounded:g}"
