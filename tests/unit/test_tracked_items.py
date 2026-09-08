"""What `truth["produced"]` counts, and why it cannot be a fixed list.

`world.lua` counted eight hardcoded items. A task whose objective names
anything else -- and R3's construction chain may -- would read zero from its
own success predicate, with no error raised anywhere. The list is now the
default plus whatever the task's predicates name.

The digest constraint is the reason this is a *difference* rather than the
whole list: `freeze_holdout` stores a `blueprint_digest` per episode and
`prepare_scene` must reproduce it, so adding a key to every payload would
invalidate `holdout_v3` and every result measured against it.
"""

from __future__ import annotations

import pathlib
import re

from factoriorl.env import blueprint_digest
from factoriorl.tasks import all_tasks, get
from factoriorl.tasks.spec import (
    DEFAULT_TRACKED_ITEMS,
    Blueprint,
    Predicate,
    PredicateKind,
    RewardComponent,
    RewardKind,
    TaskSpec,
)

WORLD_LUA = pathlib.Path(__file__).resolve().parents[2] / "mod" / "factoriorl" / "world.lua"


def _lua_default_list() -> set[str]:
    """The literal the mod actually iterates."""
    source = WORLD_LUA.read_text(encoding="utf-8")
    match = re.search(r"local tracked = \{(.*?)\}", source, re.S)
    assert match, "the tracked-item literal moved; this test must be updated"
    return set(re.findall(r'"([^"]+)"', match.group(1)))


def _spec(*predicates, rewards=()):
    return TaskSpec(
        id="t",
        version="1.0.0",
        description="d",
        layout_families=(),
        success=tuple(predicates),
        rewards=tuple(rewards),
    )


def test_the_python_constant_matches_the_lua_literal():
    """Drift here makes a task's output invisible to its own predicate."""
    assert _lua_default_list() == set(DEFAULT_TRACKED_ITEMS)


def test_the_mod_appends_the_extra_items_to_the_default():
    source = WORLD_LUA.read_text(encoding="utf-8")
    assert "scene.extra_tracked_items" in source
    assert "tracked[#tracked + 1] = item" in source


class TestDerivation:
    def test_an_item_outside_the_default_is_reported(self):
        spec = _spec(Predicate(PredicateKind.PRODUCED, item="electronic-circuit"))
        assert spec.extra_tracked_items == ("electronic-circuit",)

    def test_a_default_item_is_not_repeated(self):
        spec = _spec(Predicate(PredicateKind.PRODUCED, item="iron-plate"))
        assert spec.extra_tracked_items == ()

    def test_a_shaping_component_counts_too(self):
        """A reward driven by an uncounted item would read zero forever."""
        spec = _spec(
            Predicate(PredicateKind.CHARACTER_WITHIN, marker="m"),
            rewards=(
                RewardComponent(
                    name="c",
                    kind=RewardKind.HIGH_WATER,
                    predicate=Predicate(PredicateKind.PRODUCED, item="copper-cable"),
                    cap=1.0,
                ),
            ),
        )
        assert spec.extra_tracked_items == ("copper-cable",)

    def test_a_sustained_predicate_is_included(self):
        spec = _spec(Predicate(PredicateKind.SUSTAINED_OUTPUT, item="steel-plate", over_ticks=3600))
        assert spec.extra_tracked_items == ("steel-plate",)

    def test_the_result_is_sorted_and_deduplicated(self):
        spec = _spec(
            Predicate(PredicateKind.PRODUCED, item="pipe"),
            Predicate(PredicateKind.PRODUCED, item="engine-unit"),
            Predicate(PredicateKind.PRODUCED, item="pipe"),
        )
        assert spec.extra_tracked_items == ("engine-unit", "pipe")

    def test_a_predicate_with_no_item_contributes_nothing(self):
        spec = _spec(Predicate(PredicateKind.CHARACTER_WITHIN, marker="dst"))
        assert spec.extra_tracked_items == ()


class TestDigestsDoNotMove:
    def test_no_existing_task_needs_an_extra_item(self):
        """The precondition for every frozen digest staying valid."""
        for name in all_tasks():
            assert get(name).spec.extra_tracked_items == (), name

    def test_an_empty_list_leaves_the_payload_byte_identical(self):
        blueprint = Blueprint()
        assert blueprint.to_dict() == blueprint.to_dict(extra_tracked_items=())
        assert "extra_tracked_items" not in blueprint.to_dict()
        assert blueprint_digest(blueprint.to_dict(extra_tracked_items=())) == blueprint_digest(
            blueprint.to_dict()
        )

    def test_every_task_digest_is_unchanged_by_the_new_argument(self):
        import random

        for name in all_tasks():
            task = get(name)
            for family in task.spec.layout_families:
                blueprint = task.generate(family, random.Random(7))
                plain = blueprint.to_dict(public_markers=task.spec.public_markers)
                wired = blueprint.to_dict(
                    public_markers=task.spec.public_markers,
                    extra_tracked_items=task.spec.extra_tracked_items,
                )
                assert blueprint_digest(plain) == blueprint_digest(wired), (
                    f"{name}/{family.name} digest moved"
                )

    def test_a_task_that_does_need_one_gets_a_different_digest(self):
        """Stated rather than assumed: it is a scene change, and should be."""
        blueprint = Blueprint()
        assert blueprint_digest(blueprint.to_dict()) != blueprint_digest(
            blueprint.to_dict(extra_tracked_items=("steel-plate",))
        )


class TestOnlyProductionKindsCount:
    """`BUILT` reads `truth["built"]`; asking the mod to count production of a
    machine nobody smelts would move that task's digest for no reason."""

    def test_a_built_predicate_does_not_request_tracking(self):
        spec = _spec(Predicate(PredicateKind.BUILT, item="burner-mining-drill", at_least=1))
        assert spec.extra_tracked_items == ()

    def test_a_container_predicate_does_not_request_tracking(self):
        spec = _spec(
            Predicate(PredicateKind.CONTAINER_HOLDS, marker="sink", item="electronic-circuit")
        )
        assert spec.extra_tracked_items == ()

    def test_an_any_working_predicate_does_not_request_tracking(self):
        spec = _spec(Predicate(PredicateKind.ANY_WORKING, item="stone-furnace", at_least=1))
        assert spec.extra_tracked_items == ()

    def test_build_line_asks_for_nothing_despite_naming_two_machines(self):
        """The concrete case that caught this."""
        spec = get("build_line").spec
        named = {p.item for p in spec.success if p.item}
        assert "burner-mining-drill" in named and "stone-furnace" in named
        assert spec.extra_tracked_items == ()
