"""Engine-free coverage for the skill layer (PLAN.md 4b.1, 4b.2).

The criteria that matter here are not about whether skills help -- 4b.3 measures
that -- but about whether they are *legitimate*. A skill library that names a
task's answer would hand the policy evaluator knowledge through the action
space, which is the same violation the action-mask purity test exists to catch.
"""

from __future__ import annotations

import numpy as np
import pytest

from factoriorl import skills as skills_module
from factoriorl.tasks import all_tasks, get

FAMILIES = ("navigate", "deliver", "mine_smelt", "supply_furnace", "repair_belt", "restore_power")


def test_no_skill_names_a_task_or_a_marker():
    """PLAN 4b.1: a skill may address the third-nearest entity; it may not
    'repair the belt gap'. Task ids and marker names are the vocabulary of the
    answer, so their appearance in a skill is the detectable form of cheating."""
    forbidden = set(all_tasks())
    for task_id in all_tasks():
        spec = get(task_id).spec
        for predicate in spec.success + spec.failure:
            if predicate.marker:
                forbidden.add(predicate.marker)
    for skill in skills_module.SKILLS:
        text = f"{skill.key} {skill.description}".lower()
        for word in forbidden:
            assert word.lower() not in text, f"skill {skill.key} names {word!r}"


def test_skills_are_shared_not_per_family():
    """Sharing is the only route by which a skill can transfer. A library that
    grows a branch per family is a lookup table wearing an abstraction's name."""
    keys = [s.key for s in skills_module.SKILLS]
    assert len(keys) == len(set(keys))
    # Every skill is parameterised by an observation slot, never by a family.
    for skill in skills_module.SKILLS:
        assert skill.requires.startswith(("entity_", "resource_"))


def test_skill_availability_is_reconstructible_from_the_observation():
    """The mask a policy sees must be derivable from what the policy sees."""
    observation = {
        "character": {"position": [0.0, 0.0]},
        "entities": [
            {"name": "wooden-chest", "p": [3.0, 0.0], "h": "e1"},
            {"name": "stone-furnace", "p": [9.0, 0.0], "h": "e2"},
        ],
        "resources": {"tiles": [{"p": [-4.0, 0.0], "h": "r1"}]},
        "inventory": {},
    }
    mask = skills_module.skill_masks(observation)
    by_key = dict(zip([s.key for s in skills_module.SKILLS], mask, strict=True))
    assert by_key["approach_entity_0"]
    assert by_key["approach_entity_1"]
    # Only two entities are visible, so ranks 2 and 3 are not addressable.
    assert not by_key["approach_entity_2"]
    assert not by_key["approach_entity_3"]
    assert by_key["approach_resource"]
    assert by_key["mine_batch"]


def test_no_entity_skills_are_legal_in_an_empty_scene():
    empty = {"character": {"position": [0.0, 0.0]}, "entities": [], "resources": {"tiles": []}}
    assert not skills_module.skill_masks(empty).any()


def test_entities_are_ranked_by_distance_from_the_character():
    """Rank is the whole addressing scheme, so it must follow the character
    rather than the scene origin."""
    observation = {
        "character": {"position": [10.0, 0.0]},
        "entities": [
            {"name": "far", "p": [0.0, 0.0], "h": "far"},
            {"name": "near", "p": [14.0, 0.0], "h": "near"},
        ],
        "resources": {"tiles": []},
    }
    context = skills_module.skill_context(observation)
    assert context["entity_0"]["h"] == "near"
    assert context["entity_1"]["h"] == "far"


@pytest.mark.parametrize("task_id", FAMILIES)
def test_skill_env_appends_without_moving_primitive_indices(task_id):
    """A primitive keeps its index when skills are added, so the two arms of
    the 4b.3 ablation disagree only about actions the flat arm never had."""

    class _StubEnv:
        def __init__(self, spec):
            from factoriorl import catalog as catalog_module

            self.catalog = catalog_module.resolve(spec.catalog, spec.catalog_subset)
            self.spec_ = spec
            self.task = None
            self.session = None
            self.observation_space = None
            self._observation = {
                "character": {"position": [0.0, 0.0]},
                "entities": [{"name": "wooden-chest", "p": [6.0, 0.0], "h": "e1"}],
                "resources": {"tiles": []},
            }

        def action_masks(self):
            return np.ones(len(self.catalog), dtype=bool)

    spec = get(task_id).spec
    stub = _StubEnv(spec)
    wrapped = skills_module.SkillEnv(stub)
    assert wrapped.action_space.n == len(stub.catalog) + len(skills_module.SKILLS)
    mask = wrapped.action_masks()
    assert mask.shape == (wrapped.action_space.n,)
    # Primitive half is untouched.
    assert mask[: len(stub.catalog)].all()
    # One entity visible: rank 0 legal, ranks 1..3 not.
    skill_half = dict(
        zip([s.key for s in skills_module.SKILLS], mask[len(stub.catalog) :], strict=True)
    )
    assert skill_half["approach_entity_0"]
    assert not skill_half["approach_entity_1"]


def test_an_entity_underfoot_is_not_addressable():
    """Rank is measured from the character, so arriving at the k-th nearest
    entity makes it the nearest and demotes the one just left. Re-issuing the
    same approach then walks straight back.

    A trained policy was observed alternating `approach_entity_1` between two
    chests for an entire episode -- nine steps out, twelve back, forever --
    because executing the action inverted its meaning. Excluding what the agent
    is already standing at means an approach is never a no-op, and with two
    candidates and one underfoot the second slot stops existing, so the looping
    action is masked illegal rather than merely unhelpful.
    """
    at_the_chest = {
        "character": {"position": [3.0, 0.0]},
        "entities": [
            {"name": "wooden-chest", "type": "container", "p": [3.5, 0.0], "h": "here"},
            {"name": "wooden-chest", "type": "container", "p": [20.0, 0.0], "h": "away"},
        ],
        "resources": {"tiles": []},
    }
    context = skills_module.skill_context(at_the_chest)
    assert context["entity_0"]["h"] == "away", "the chest underfoot is still being offered"
    assert context["entity_1"] is None, "a second slot exists for a two-chest scene"

    mask = skills_module.skill_masks(at_the_chest)
    by_key = dict(zip([s.key for s in skills_module.SKILLS], mask, strict=True))
    assert by_key["approach_entity_0"]
    assert not by_key["approach_entity_1"], "the oscillating action is still legal"


# ------------------------------------- 4b.1, checked semantically


def rankable_counts(task_id: str, split: str, samples: int = 20) -> list[int]:
    """How many entities `approach_entity_k` can actually address, per scene.

    Mirrors `skill_context`: obstacles are dropped from the ranking. Blueprint
    specs carry no engine `type`, and `stone-wall` is the only obstacle any
    family places, so the name is the proxy -- asserted below so a new obstacle
    entity cannot slip past it.
    """
    from factoriorl.seeding import Branch, SeedPlan

    task = get(task_id)
    families = [f for f in task.spec.layout_families if f.split == split]
    if not families:
        return []
    plan = SeedPlan(master=20260908, run_id="holdout-v1")
    counts = []
    for index in range(1000, 1000 + samples):
        rng = plan.generator_rng(Branch.EVAL, index)
        family = families[rng.randrange(len(families))]
        blueprint = task.generate(family, rng)
        entities = list(getattr(blueprint, "entities", []) or [])
        counts.append(sum(1 for e in entities if "wall" not in e.name))
    return counts


def test_wall_is_still_the_only_obstacle_the_name_proxy_has_to_cover():
    assert set(skills_module.OBSTACLE_TYPES) == {"wall"}


@pytest.mark.parametrize("task_id", FAMILIES)
def test_addressable_field_is_pinned(task_id):
    """PLAN 4b.1 forbids a skill that encodes a family's solution, and the
    string test above cannot see the violation that matters.

    `approach_entity_k` addresses the k-th *nearest* entity. When a scene
    contains exactly one rankable entity, that skill has exactly one possible
    referent -- and in `navigate` that referent is the goal marker itself, so a
    single action wins the episode. The measured random floor over skills is
    0.99 on `navigate` and 0.32 on `mine_smelt`, against 0.04 on `deliver`,
    which is the only family that places entities the solution does not need.

    None of that is visible to a test that greps skill descriptions for task
    ids: `approach_entity_0` contains no forbidden substring and is, in five of
    six families, the answer.

    The counts are pinned rather than required to exceed a bound, because
    raising them means changing generators, which changes every blueprint digest
    and invalidates the frozen holdout. A generator edit that trivialises a
    family fails here; so does the fix, deliberately.
    """
    expected = {
        # task: (train, test) rankable entities per scene
        "navigate": (1, 1),  # the goal chest, alone among walls -- floor 0.99
        "mine_smelt": (1, 1),  # the furnace -- floor 0.32
        "supply_furnace": (2, 3),
        "deliver": (4, 4),  # 1 dst, 1 src, 2 decoys -- floor 0.04
        "repair_belt": (13, 12),
        "restore_power": (12, 12),
    }[task_id]
    for split, want in zip(("train", "test"), expected, strict=True):
        counts = rankable_counts(task_id, split)
        assert counts, f"{task_id} has no {split} family"
        assert min(counts) <= want <= max(counts), (
            f"{task_id}/{split} addressable field moved: saw {min(counts)}..{max(counts)}, "
            f"pinned at {want}. A family whose rankable set is a singleton hands "
            f"approach_entity_0 the answer."
        )
