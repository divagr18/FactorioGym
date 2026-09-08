"""Actions that carry arguments, with domains the policy can see.

`primitive-v1` welds *where* to *which facing*: `place_<item>_<direction>`
targets `floor(position) + PLACE_OFFSETS[direction]` and sets the facing to the
same direction. Filling a belt gap with an east-facing belt therefore means
standing west of it, on the belt line, so the reference solver uses
`place_transport_belt_north` then `rotate_target` -- two ordered actions at tile
precision, which `repair_belt` found 7 times in 221 episodes.

The Lua `place` handler has always accepted an arbitrary in-reach position and
an independent direction, so `parameterized-v1` exposes an existing capability.
"""

from __future__ import annotations

import numpy as np
import pytest

from factoriorl import catalog as catalog_module
from factoriorl.env import PLACEMENT_RADIUS, TRANSFER_AMOUNTS, FactorioEnv
from factoriorl.tasks import all_tasks, get


class _Env(FactorioEnv):
    """Argument domains and masks only; no engine, no session."""

    def __init__(self, observation, catalog_name="parameterized-v1"):
        self.catalog = catalog_module.resolve(catalog_name)
        self._observation = observation
        self._steps = 0
        self._truth = {}
        self.spec_ = get("repair_belt").spec


def _observation(**overrides):
    base = {
        "character": {"position": [0.0, 0.0]},
        "entities": [
            {"h": "e1", "name": "transport-belt", "type": "transport-belt", "p": [4.5, 0.5]},
            {"h": "e2", "name": "wooden-chest", "type": "container", "p": [-3.5, 0.5]},
        ],
        "resources": {"tiles": [{"h": "r1", "name": "iron-ore", "p": [2.5, 3.5]}]},
        "terrain": {"blocked": []},
        "inventory": {"transport-belt": 5, "coal": 0},
        "inflight": [{"request_id": "act-7", "action": "move", "progress": 0.5}],
    }
    base.update(overrides)
    return base


class TestArgumentDeclaration:
    def test_declaring_an_argument_is_writing_it_into_the_payload(self):
        template = catalog_module.ActionTemplate(
            "place_at", "place", {"item": "?item", "position": "?position"}
        )
        assert template.arguments == ("item", "position")
        assert template.parameterized

    def test_the_digest_covers_the_argument_list(self):
        """An action that gains an argument is a different action."""
        one = catalog_module.ResolvedCatalog(
            "x", (catalog_module.ActionTemplate("k", "place", {"item": "?item"}),)
        )
        two = catalog_module.ResolvedCatalog(
            "x",
            (catalog_module.ActionTemplate("k", "place", {"item": "?item", "d": "?direction"}),),
        )
        assert one.digest() != two.digest()

    def test_a_missing_argument_is_an_error_not_a_hole(self):
        template = catalog_module.ActionTemplate("place_at", "place", {"item": "?item"})
        with pytest.raises(ValueError, match="needs argument 'item'"):
            template.bind({}, {})

    def test_legacy_templates_are_byte_identical(self):
        """Old artifacts must remain runnable under their original profile."""
        for template in catalog_module.resolve("primitive-v1").templates:
            assert not template.parameterized, template.key
        context = {"target": "e1", "resource": "r1", "at_north": [0.5, -1.5]}
        legacy = catalog_module.resolve("primitive-v1").templates
        before = [t.bind(context) for t in legacy]
        after = [t.bind(context, None) for t in legacy]
        assert before == after


class TestPlacementIsNowExpressible:
    def test_position_and_facing_are_independent(self):
        """The whole point: a tile east of me, facing north."""
        template = next(
            t for t in catalog_module.resolve("parameterized-v1").templates if t.key == "place_at"
        )
        payload = template.bind(
            {}, {"item": "transport-belt", "position": [1.5, 0.5], "direction": "north"}
        )
        assert payload == {
            "action": "place",
            "item": "transport-belt",
            "position": [1.5, 0.5],
            "direction": "north",
        }

    def test_the_old_catalog_could_not_express_it(self):
        """Every primitive placement couples the tile to the facing."""
        for template in catalog_module.resolve("primitive-v1").templates:
            if template.action != "place":
                continue
            direction = template.payload["direction"]
            assert template.payload["position"] == f"$at_{direction}"


class TestDomainsComeFromTheObservation:
    def test_entity_and_resource_handles_are_both_addressable(self):
        """Resource tiles carry handles that nothing could name before."""
        domains = _Env(_observation()).argument_domains()
        assert set(domains["targets"]) == {"e1", "e2", "r1"}

    def test_items_are_only_what_is_held(self):
        domains = _Env(_observation()).argument_domains()
        # coal is present with count 0 and must not be offered.
        assert domains["items"] == ["transport-belt"]

    def test_placements_exclude_occupied_and_blocked_tiles(self):
        observation = _observation(terrain={"blocked": [[1.5, 1.5]]})
        domains = _Env(observation).argument_domains()
        tiles = {tuple(p) for p in domains["placements"]}
        assert (4.5, 0.5) not in tiles, "a visible entity's tile is not a candidate"
        assert (1.5, 1.5) not in tiles, "a blocked tile is not a candidate"
        assert (0.5, 0.5) in tiles

    def test_placements_are_bounded(self):
        domains = _Env(_observation()).argument_domains()
        span = 2 * PLACEMENT_RADIUS + 1
        assert len(domains["placements"]) <= span * span

    def test_in_flight_requests_are_addressable(self):
        domains = _Env(_observation()).argument_domains()
        assert domains["requests"] == ["act-7"]

    def test_unobservable_domains_are_empty_not_guessed(self):
        domains = _Env(_observation()).argument_domains()
        assert domains["recipes"] == []
        assert domains["technologies"] == []

    def test_amounts_are_a_declared_ladder(self):
        assert _Env(_observation()).argument_domains()["amounts"] == list(TRANSFER_AMOUNTS)


class TestEmptyDomainsAreMasked:
    def _mask(self, observation):
        env = _Env(observation)
        return dict(zip(env.catalog.keys(), env.action_masks(), strict=True))

    def test_an_action_whose_argument_has_no_value_is_illegal(self):
        """sb3 turns an all-false sub-mask into a uniform draw over illegal
        values, silently, so this has to be caught before it gets there."""
        mask = self._mask(_observation())
        # No recipe is observable yet, so neither recipe action can be issued.
        assert not mask["set_recipe_at"]
        assert not mask["craft_recipe"]

    def test_actions_whose_domains_are_populated_are_legal(self):
        mask = self._mask(_observation())
        assert mask["place_at"]
        assert mask["mine_at"]
        assert mask["give_to"]

    def test_an_empty_scene_masks_the_addressed_actions_but_leaves_wait(self):
        empty = _observation(entities=[], resources={"tiles": []}, inflight=[])
        mask = self._mask(empty)
        assert not mask["mine_at"]
        assert not mask["rotate_at"]
        assert not mask["cancel_request"]
        assert mask["wait"], "a legal no-op must always exist"

    def test_holding_nothing_masks_placement(self):
        broke = _observation(inventory={})
        mask = self._mask(broke)
        assert not mask["place_at"]
        assert mask["wait"]

    def test_the_mask_is_reconstructible_from_the_observation(self):
        """Rebuilt here from policy-visible data, the way test_skills does."""
        observation = _observation()
        env = _Env(observation)
        domains = env.argument_domains()
        expected = []
        for template in env.catalog.templates:
            legal = all(
                domains.get(catalog_module.ARGUMENT_DOMAINS[name])
                for name in template.arguments
                if name in catalog_module.ARGUMENT_DOMAINS
            )
            expected.append(bool(legal) or template.action == "wait")
        assert np.array_equal(env.action_masks(), np.array(expected))
        assert not all(expected), "a vacuous mask would prove nothing"


class TestValidation:
    def test_an_out_of_domain_argument_is_refused_with_a_name(self):
        env = _Env(_observation())
        index = env.catalog.keys().index("mine_at")
        with pytest.raises(ValueError, match="not in domain 'targets'"):
            env.step_arguments(index, {"handle": "nope"})

    def test_a_missing_argument_is_refused(self):
        env = _Env(_observation())
        index = env.catalog.keys().index("place_at")
        with pytest.raises(ValueError, match="needs argument"):
            env.step_arguments(index, {"item": "transport-belt"})


class TestPurity:
    def test_no_operation_or_argument_names_a_task_or_marker(self):
        names = {t.id for t in (get(x).spec for x in all_tasks())}
        markers = {"gap", "gap2", "sink", "drill", "dst", "src", "goal"}
        for template in catalog_module.resolve("parameterized-v1").templates:
            words = f"{template.key} {' '.join(template.arguments)}".lower()
            for forbidden in names | markers:
                assert forbidden not in words.split("_"), f"{template.key} names {forbidden}"

    def test_every_argument_has_a_declared_domain(self):
        for template in catalog_module.resolve("parameterized-v1").templates:
            for name in template.arguments:
                assert name in catalog_module.ARGUMENT_DOMAINS, f"{template.key}: {name}"

    def test_the_parameterized_catalog_keeps_a_wait(self):
        assert catalog_module.resolve("parameterized-v1").wait_index >= 0
