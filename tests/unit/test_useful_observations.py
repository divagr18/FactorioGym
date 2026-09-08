"""R2.3: the task and its interactions must be legible from what the policy sees.

Its gate: "The task and available interactions are understandable from
policy-visible data. Hidden diagnostic state is not smuggled through masks or
errors. Both clients' different encodings and assistance are recorded."

Before this, a stopped machine's *cause* was one bit -- `no_fuel`, `no_power`
and `waiting_for_source_items` encoded identically -- and no recipe was
observable, so `craft` and `set_recipe` had no legal value a policy could name.
"""

from __future__ import annotations

import numpy as np
import pytest

from factoriorl import catalog as catalog_module
from factoriorl import encoders
from factoriorl.env import FactorioEnv
from factoriorl.tasks import get

STATUS_SLOT, WORKING_SLOT, HAS_STATUS_SLOT, FUEL_SLOT, OUTPUT_SLOT = 11, 12, 13, 14, 15


def _entity(**overrides):
    base = {"p": [1.5, 0.5], "type": "furnace", "h": "e1"}
    base.update(overrides)
    return base


def _encode(*entities, **observation):
    payload = {"character": {"position": [0.0, 0.0]}, "entities": list(entities)}
    payload.update(observation)
    return encoders.encode(payload)


class _Env(FactorioEnv):
    def __init__(self, observation):
        self.catalog = catalog_module.resolve("parameterized-v1")
        self.spec_ = get("repair_belt").spec
        self._observation = observation
        self._steps = 0
        self._truth = {}


class TestAStoppedMachineSaysWhy:
    def test_different_causes_encode_differently(self):
        """The whole point: one bit could not distinguish these."""
        seen = set()
        for cause in ("no_fuel", "no_power", "waiting_for_source_items", "full_output"):
            row = _encode(_entity(st=cause, working=False))["entities"][0]
            seen.add(round(float(row[STATUS_SLOT]), 6))
        assert len(seen) == 4, "stop causes collapsed to the same value"

    def test_working_is_stated_not_inferred_from_absence(self):
        working = _encode(_entity(st="working", working=True))["entities"][0]
        stopped = _encode(_entity(st="no_fuel", working=False))["entities"][0]
        assert working[WORKING_SLOT] == 1.0
        assert stopped[WORKING_SLOT] == 0.0

    def test_no_status_at_all_is_distinguishable_from_not_working(self):
        """A belt has no status; a stopped furnace does. These differed by
        an absent field, which encoded the same as `not working`."""
        beltish = _encode(_entity(type="transport-belt"))["entities"][0]
        stopped = _encode(_entity(st="no_fuel", working=False))["entities"][0]
        assert beltish[HAS_STATUS_SLOT] == 0.0
        assert stopped[HAS_STATUS_SLOT] == 1.0
        assert beltish[WORKING_SLOT] == stopped[WORKING_SLOT] == 0.0

    def test_an_unknown_status_name_does_not_shift_the_others(self):
        row = _encode(_entity(st="some-future-status", working=False))["entities"][0]
        assert row[STATUS_SLOT] == 0.0

    def test_fuel_and_output_are_visible(self):
        fuelled = _encode(_entity(fuel={"coal": 20}, output={"iron-plate": 3}))["entities"][0]
        empty = _encode(_entity())["entities"][0]
        assert fuelled[FUEL_SLOT] > 0.0 and fuelled[OUTPUT_SLOT] > 0.0
        assert empty[FUEL_SLOT] == 0.0 and empty[OUTPUT_SLOT] == 0.0

    def test_a_drill_with_no_fuel_is_distinguishable_from_one_with_fuel(self):
        """The concrete diagnosis a repair agent has to make."""
        starved = _encode(_entity(type="mining-drill", st="no_fuel", working=False, fuel={}))
        running = _encode(
            _entity(type="mining-drill", st="working", working=True, fuel={"coal": 10})
        )
        assert not np.array_equal(starved["entities"][0], running["entities"][0])


class TestActionOutcomesReachThePolicy:
    def test_a_refusal_is_visible_in_the_self_vector(self):
        refused = _encode(events=[{"seq": 1, "action": "place", "status": "rejected"}])["self"]
        completed = _encode(events=[{"seq": 1, "action": "place", "status": "completed"}])["self"]
        assert refused[9] == 1.0 and completed[9] == 0.0

    def test_the_refusal_rate_is_visible(self):
        events = [
            {"seq": 1, "status": "rejected"},
            {"seq": 2, "status": "completed"},
            {"seq": 3, "status": "rejected"},
        ]
        assert _encode(events=events)["self"][11] == pytest.approx(2 / 3)

    def test_a_running_action_does_not_count_as_settled(self):
        assert _encode(events=[{"seq": 1, "status": "running"}])["self"][11] == 0.0


class TestRecipesAreSelectable:
    def test_an_observed_recipe_becomes_a_legal_argument(self):
        env = _Env(
            {
                "character": {"position": [0.0, 0.0]},
                "entities": [{"h": "e1", "p": [1.5, 0.5], "type": "furnace"}],
                "resources": {"tiles": []},
                "terrain": {"blocked": []},
                "inventory": {"stone-furnace": 1},
                "inflight": [],
                "recipes": ["iron-plate", "stone-furnace"],
            }
        )
        assert env.argument_domains()["recipes"] == ["iron-plate", "stone-furnace"]
        mask = dict(zip(env.catalog.keys(), env.action_masks(), strict=True))
        assert mask["craft_recipe"], "a craftable recipe must unmask crafting"
        assert mask["set_recipe_at"]

    def test_without_recipes_those_operations_stay_masked(self):
        env = _Env(
            {
                "character": {"position": [0.0, 0.0]},
                "entities": [{"h": "e1", "p": [1.5, 0.5], "type": "furnace"}],
                "resources": {"tiles": []},
                "terrain": {"blocked": []},
                "inventory": {},
                "inflight": [],
            }
        )
        mask = dict(zip(env.catalog.keys(), env.action_masks(), strict=True))
        assert not mask["craft_recipe"]
        assert not mask["set_recipe_at"]
        assert mask["wait"]


class TestNoHiddenStateIsSmuggled:
    def test_the_encoding_never_reads_truth(self):
        """`encode` takes an observation and a goal vector, and nothing else."""
        import inspect

        parameters = set(inspect.signature(encoders.encode).parameters)
        assert parameters == {"observation", "goal", "profile"}

    def test_domains_are_a_pure_function_of_the_observation(self):
        observation = {
            "character": {"position": [0.0, 0.0]},
            "entities": [{"h": "e1", "p": [1.5, 0.5], "type": "furnace"}],
            "resources": {"tiles": []},
            "terrain": {"blocked": []},
            "inventory": {"coal": 3},
            "inflight": [],
            "recipes": ["iron-plate"],
        }
        first = _Env(dict(observation)).argument_domains()
        second = _Env(dict(observation)).argument_domains()
        assert first == second

    def test_a_marker_the_task_did_not_publish_is_not_in_any_domain(self):
        env = _Env(
            {
                "character": {"position": [0.0, 0.0]},
                "entities": [],
                "resources": {"tiles": []},
                "terrain": {"blocked": []},
                "inventory": {},
                "inflight": [],
                # An evaluator-only marker would arrive here if publication
                # leaked; the domains must not pick anything out of `goal`.
                "goal": {"gap": [9.5, -7.5]},
            }
        )
        flat = [v for values in env.argument_domains().values() for v in values]
        assert [9.5, -7.5] not in flat


def test_both_clients_encodings_are_pinned():
    """R2.3's third clause, restated as a check."""
    from factoriorl.agent.summary import SUMMARY_ENCODING_VERSION
    from factoriorl.learn.policy import EXTRACTOR_VERSION

    assert EXTRACTOR_VERSION >= 7
    assert encoders.GOAL_ENCODING_VERSION >= 4
    assert SUMMARY_ENCODING_VERSION >= 1


def test_no_entity_feature_slot_is_structurally_dead():
    """Five slots were zero in every observation of every task."""
    row = _encode(
        _entity(
            st="no_fuel",
            working=False,
            d=4,
            contents={"iron-ore": 4},
            recipe="iron-plate",
            health=0.5,
            fuel={"coal": 1},
            output={"iron-plate": 2},
        )
    )["entities"][0]
    assert (row[:16] != 0).sum() >= 12, f"too many dead slots: {row.tolist()}"
