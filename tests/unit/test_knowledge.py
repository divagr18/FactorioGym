"""Static game knowledge: the recipe graph, footprints and technology tree.

Engine-free. The knowledge *content* can only come from a running Factorio, so
what is checkable here is the part that has to hold whatever the engine says:
that `render` is byte-stable, that the cache key names both things that can
invalidate an answer, and that an empty reply is refused instead of cached.

The byte-stability tests carry the most weight. `render`'s output goes into a
prompt prefix the provider caches, so a line that moves between two calls
invalidates the whole cached conversation -- and it would do that silently, at
the cost of a bill rather than a failure.
"""

from __future__ import annotations

import json

import pytest

from factoriorl import knowledge as knowledge_module
from factoriorl.knowledge import (
    KnowledgeUnavailable,
    cache_key,
    cache_path,
    cached,
    fetch,
    render,
)
from factoriorl.protocol import ErrorBody, ErrorCode, RequestType, Response, ResultCode

BUILD = "2.0.60"
DIGEST = "abc123def4567890"


def a_knowledge_reply() -> dict:
    """A miniature of what `mod/factoriorl/knowledge.lua` returns.

    Deliberately holds one of each awkward case the renderer has to survive: a
    recipe whose product is named after it, one whose product is not, a fluid
    ingredient, a probabilistic product, a locked recipe, an entity wider than
    one tile, a gated placement and a trigger technology.
    """
    return {
        "recipes": [
            {
                "name": "iron-gear-wheel",
                "ingredients": [{"name": "iron-plate", "amount": 2}],
                "products": [{"name": "iron-gear-wheel", "amount": 1}],
                "energy": 0.5,
                "category": "crafting",
                "enabled": True,
            },
            {
                "name": "electronic-circuit",
                "ingredients": [
                    {"name": "iron-plate", "amount": 1},
                    {"name": "copper-cable", "amount": 3},
                ],
                "products": [{"name": "electronic-circuit", "amount": 1}],
                "energy": 0.5,
                "category": "crafting",
                "enabled": False,
            },
            {
                "name": "basic-oil-processing",
                "ingredients": [{"name": "crude-oil", "amount": 100, "type": "fluid"}],
                "products": [
                    {"name": "petroleum-gas", "amount": 45, "type": "fluid"},
                    {
                        "name": "heavy-oil",
                        "amount_min": 1,
                        "amount_max": 4,
                        "probability": 0.25,
                    },
                ],
                "energy": 5.0,
                "category": "oil-processing",
                "enabled": True,
            },
        ],
        "placeable": [
            {"name": "stone-furnace", "entity": "stone-furnace", "width": 2, "height": 2},
            {
                "name": "electric-mining-drill",
                "entity": "electric-mining-drill",
                "width": 3,
                "height": 3,
                "gated_by": "electric-mining-drill",
                "unlocked": False,
            },
        ],
        "technologies": [
            {
                "name": "automation",
                "prerequisites": ["electronics"],
                "unit_count": 10,
                "unit_ingredients": [{"name": "automation-science-pack", "amount": 1}],
                "unlocks": ["assembling-machine-1", "long-handed-inserter"],
                "researched": False,
            },
            {
                "name": "steam-power",
                "prerequisites": [],
                "unit_ingredients": [],
                "unlocks": ["boiler", "steam-engine"],
                "trigger": "craft-item",
                "researched": True,
            },
        ],
        "counts": {"recipes": 3, "placeable": 2, "technologies": 2},
        "force_state_tick": 1234,
    }


class FakeSession:
    """A worker that answers `knowledge` and counts how often it was asked."""

    def __init__(self, result: dict | None = None, code: ResultCode = ResultCode.OK):
        self.result = a_knowledge_reply() if result is None else result
        self.code = code
        self.calls = 0

    def knowledge(self):
        self.calls += 1
        error = None
        if self.code is not ResultCode.OK:
            error = ErrorBody(code=ErrorCode.ENGINE, message="prototypes unreadable")
        response = Response(
            request_id="knowledge-1",
            episode_id="",
            code=self.code,
            result=self.result,
            error=error,
        )
        return type("Timed", (), {"response": response, "round_trip_ms": 1.0})()


# --- the property the prompt cache depends on -------------------------------


def test_render_is_byte_identical_across_calls() -> None:
    """The one property that cannot be allowed to slip.

    `render`'s output is a cached prompt prefix. A single reordered line is
    indistinguishable from a changed one to the provider's cache key, so an
    unstable render silently re-bills the entire conversation every turn instead
    of failing.
    """
    reply = a_knowledge_reply()
    assert render(reply) == render(reply)
    assert render(reply).encode("utf-8") == render(a_knowledge_reply()).encode("utf-8")


def test_render_ignores_the_order_the_engine_happened_to_return() -> None:
    """`pairs` order is not stable in Lua, so it must not reach the output.

    `knowledge.lua` sorts before it serialises. This asserts the Python side
    sorts too, because a cache file written by an older mod is exactly the input
    nobody would think to re-check.
    """
    forward = a_knowledge_reply()
    reversed_reply = {
        section: list(reversed(rows)) if isinstance(rows, list) else rows
        for section, rows in forward.items()
    }
    assert render(reversed_reply) == render(forward)


def test_render_survives_the_empty_table_lua_serialises_as_an_object() -> None:
    """An empty Lua table becomes `{}` in JSON, not `[]`.

    Every list in the reply is built with integer keys, so an install with
    nothing in a section hands back a dict where every other reply hands back a
    list. Rendering that as a section header with no rows beats raising.
    """
    reply = a_knowledge_reply()
    reply["placeable"] = {}
    assert "PLACEABLE ENTITIES (0)" in render(reply)


# --- the shape of a line ----------------------------------------------------


def test_a_recipe_line_matches_the_worked_example() -> None:
    """~18 tokens: the target the section was budgeted against.

    The product name is dropped when it repeats the recipe name, which is true
    of the overwhelming majority of recipes and is the difference between the
    model reading the name once and reading it twice.
    """
    assert "iron-gear-wheel: 2 iron-plate -> 1 (0.5s, crafting)" in render(a_knowledge_reply())


def test_a_locked_recipe_renders_like_any_other() -> None:
    """Whether a recipe is unlocked is force state, not prototype state.

    It moves as research completes, and this block is captured once and pinned
    into a prompt prefix that must stay byte-identical for the whole run. So the
    lock flag is deliberately absent: a marker recorded here would be wrong for
    most of the run, and wrong in the direction that stops an agent trying
    something it can in fact do. The observation carries the enabled recipes
    every turn.
    """
    rendered = render(a_knowledge_reply())
    assert "electronic-circuit: 1 iron-plate + 3 copper-cable -> 1 (0.5s, crafting)" in rendered
    assert "[locked]" not in rendered


def test_fluids_and_chances_are_not_rendered_as_plain_items() -> None:
    """A fluid cannot be carried or hand-crafted, and a chance is not an amount.

    Both were silently droppable: a probabilistic product carries
    `amount_min`/`amount_max` instead of `amount`, and rendering the missing
    field as nothing would read as "produces one of these".
    """
    rendered = render(a_knowledge_reply())
    assert "100 crude-oil(fluid)" in rendered
    assert "45 petroleum-gas(fluid)" in rendered
    assert "1-4 heavy-oil@25%" in rendered


def test_entity_footprints_are_the_real_ones() -> None:
    """This is what replaces `tasks/spec.py`'s four-prototype guess.

    That table falls back to 1x1 for everything it does not list, and a guessed
    footprint is worse than none: it decides where a task believes an entity
    fits.
    """
    rendered = render(a_knowledge_reply())
    assert "stone-furnace: 2x2" in rendered
    assert "electric-mining-drill: 3x3" in rendered


def test_a_trigger_technology_reports_its_trigger_rather_than_a_cost() -> None:
    """2.0 roots the tech tree in triggers, and they have no science cost.

    Seven of 196 complete by crafting or mining an item and cannot be queued at
    all -- the case `protocol.py` carries `tech_not_selectable` for. An agent
    told only "not selectable" has nothing to act on; the trigger tells it what
    would finish the research.
    """
    rendered = render(a_knowledge_reply())
    assert "steam-power: trigger:craft-item -> boiler, steam-engine" in rendered
    assert "automation: 10x automation-science-pack ->" in rendered


def test_render_labels_itself_static_and_separate_from_the_observation() -> None:
    """It shares a prompt with the per-turn observation and must not be confused
    for one: a model that treats the recipe list as the current world state will
    believe it is holding items it has never seen."""
    rendered = render(a_knowledge_reply())
    assert rendered.startswith("=== STATIC GAME KNOWLEDGE ===")
    assert "observation" in rendered


# --- fetching ---------------------------------------------------------------


def test_fetch_asks_once_and_returns_the_tables() -> None:
    session = FakeSession()
    result = fetch(session)
    assert session.calls == 1
    assert {"recipes", "placeable", "technologies"} <= set(result)


def test_a_refused_request_raises_rather_than_returning_an_empty_graph() -> None:
    """`freeplay.FreeplayUnreadable`'s rule, for the same reason.

    A fabricated or empty recipe graph would make a run look like it planned
    against the real game while it did not, and nothing measured afterwards
    could tell the difference.
    """
    with pytest.raises(KnowledgeUnavailable, match="refused"):
        fetch(FakeSession(code=ResultCode.ERROR))


def test_an_empty_section_is_refused_as_a_failed_read() -> None:
    """No Factorio install has zero recipes.

    An empty table means the Lua walked something it could not read and returned
    success anyway -- which would otherwise be cached to disk and reused for the
    rest of the run.
    """
    reply = a_knowledge_reply()
    reply["recipes"] = []
    with pytest.raises(KnowledgeUnavailable, match="recipes"):
        fetch(FakeSession(result=reply))


# --- caching ----------------------------------------------------------------


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    """Point `paths.runtime_dir` at a temporary tree.

    `knowledge.cache_path` resolves it through the module it imported from, so
    the patch has to land on the name this module actually calls.
    """
    monkeypatch.setattr(knowledge_module, "runtime_dir", lambda: tmp_path)
    return tmp_path


def test_the_cache_key_names_both_things_that_can_invalidate_it(runtime) -> None:
    """The engine build alone is not enough.

    Prototypes come from the game, so a different build is a different answer;
    but `knowledge.lua` decides which fields are reported, so a mod edit is a
    different answer from the same game. Keying on the build alone would serve a
    stale file after any edit to the mod, and the JSON would still parse.
    """
    assert BUILD in cache_key(BUILD, DIGEST)
    assert DIGEST in cache_key(BUILD, DIGEST)
    assert cache_key(BUILD, DIGEST) != cache_key(BUILD, "0000000000000000")
    assert cache_key(BUILD, DIGEST) != cache_key("2.0.61", DIGEST)


def test_a_cached_answer_is_fetched_once_and_reused(runtime) -> None:
    session = FakeSession()
    first = cached(session, BUILD, DIGEST)
    second = cached(session, BUILD, DIGEST)
    assert session.calls == 1
    assert first == second
    assert cache_path(BUILD, DIGEST).is_file()


def test_a_different_mod_digest_does_not_read_the_other_worlds_file(runtime) -> None:
    session = FakeSession()
    cached(session, BUILD, DIGEST)
    cached(session, BUILD, "ffffffffffffffff")
    assert session.calls == 2


def test_a_file_whose_stored_key_disagrees_is_refetched(runtime) -> None:
    """The filename is slugged, so it is not by itself proof of the key.

    Two build strings differing only in a character `_UNSAFE` rewrites would
    land on one path; the key is stored inside the file so that collision is
    detected rather than served.
    """
    session = FakeSession()
    cached(session, BUILD, DIGEST)
    path = cache_path(BUILD, DIGEST)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["key"] = "some-other-key"
    path.write_text(json.dumps(document), encoding="utf-8")
    cached(session, BUILD, DIGEST)
    assert session.calls == 2


def test_a_truncated_cache_file_is_refetched_rather_than_raising(runtime) -> None:
    """The file is written whole and moved into place, so this should not
    happen -- but a half-written file from an older build or an interrupted
    copy must cost a fetch, not the run."""
    session = FakeSession()
    cached(session, BUILD, DIGEST)
    cache_path(BUILD, DIGEST).write_text("{not json", encoding="utf-8")
    assert cached(session, BUILD, DIGEST)["counts"]["recipes"] == 3
    assert session.calls == 2


# --- the protocol seam ------------------------------------------------------


def test_knowledge_is_a_declared_request_type() -> None:
    assert RequestType.KNOWLEDGE.value == "knowledge"


def test_the_prefix_carries_no_force_state() -> None:
    """`enabled`, `unlocked` and `researched` are read off
    `game.forces["player"]`, not from a prototype, and they move the moment
    research completes.

    This block is captured once and pinned into a prompt prefix that must stay
    byte-identical for the whole run -- that is what makes it a cache hit rather
    than a fresh charge on every turn. A lock marker recorded here would be
    quietly wrong for most of the run, and wrong in the direction that stops an
    agent trying something it can in fact do.

    Availability is published every turn instead: enabled recipes in the
    observation's `recipes`, and the researchable frontier in `researchable`.
    """
    rendered = render(a_knowledge_reply())
    for stale in ("[locked", "[researched]"):
        assert stale not in rendered, (
            f"{stale!r} is force state in a block that must not change mid-run"
        )
    # And the block says where to look instead, rather than leaving its silence
    # to be interpreted as "everything is available".
    assert "available to you *now*" in rendered
    assert "RECIPES" in rendered
