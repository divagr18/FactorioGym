"""`parameterized-v2`: a target is a row, a placement is a tile.

The shape of the action vector does not move between the two profiles, so this
pins the only thing that does -- what an index means -- because a policy
trained against one meaning and evaluated against the other reads a different
entity and builds on a different tile, silently.

The simulator these numbers come from implements the same two rules
(`factory-sim`, `csrc/fsim_rl.c`); the action-for-action transfer replay is
what checks the two agree end to end.
"""

from __future__ import annotations

import numpy as np
import pytest

from factoriorl import encoders
from factoriorl.env import PLACEMENT_RADIUS
from factoriorl.parameterized import DIMENSIONS, UNUSED, ParameterizedEnv

pytest.importorskip("gymnasium")

# Bare module name, not `tests.unit....`: there is no `__init__.py` under
# `tests/`, so pytest puts this directory on `sys.path`. Imported rather than
# copied, so the two profiles are exercised against the same fixture.
from test_parameterized_policy import _Inner, _observation  # noqa: E402

SIDE = 2 * PLACEMENT_RADIUS + 1
PLACE = "place_at"


def _v2(**overrides) -> ParameterizedEnv:
    return ParameterizedEnv(_Inner(_observation(**overrides)), profile="v2")


def _v1(**overrides) -> ParameterizedEnv:
    return ParameterizedEnv(_Inner(_observation(**overrides)))


def _slice(env, name: str) -> slice:
    names = [n for n, _ in DIMENSIONS]
    start = int(sum(env.action_space.nvec[: names.index(name)]))
    return slice(start, start + int(env.action_space.nvec[names.index(name)]))


def _spread_out(**extra):
    """Entities whose sweep order is deliberately not their distance order."""
    return _observation(
        entities=[
            {"h": "far", "p": [8.5, 0.5], "type": "container"},
            {"h": "near", "p": [1.5, 0.5], "type": "transport-belt"},
            {"h": "middle", "p": [4.5, 0.5], "type": "container"},
        ],
        **extra,
    )


class TestTheProfileItself:
    def test_an_unknown_profile_is_refused(self):
        with pytest.raises(ValueError, match="profile"):
            ParameterizedEnv(_Inner(_observation()), profile="v9")

    def test_both_profiles_have_the_same_action_space(self):
        assert list(_v1().action_space.nvec) == list(_v2().action_space.nvec)


class TestTargetIsARow:
    def test_target_k_is_row_k_of_the_entity_table(self):
        inner = _Inner(_spread_out())
        env = ParameterizedEnv(inner, profile="v2")
        rows = encoders.entity_row_order(inner._observation, [0.0, 0.0])
        expected = [record["h"] for _distance, record, _remembered in rows]
        assert expected == ["near", "middle", "far"], "the fixture must not be pre-sorted"
        assert env._domain_values()["target"] == expected

    def test_v1_keeps_sweep_order_and_its_resource_tiles(self):
        inner = _Inner(_spread_out())
        env = ParameterizedEnv(inner)
        # Sweep order, unsorted, and the ore tile the grid planes also carry.
        assert env._domain_values()["target"] == ["far", "near", "middle", "r1"]

    def test_a_resource_tile_has_no_row_so_it_cannot_be_targeted(self):
        assert "r1" not in _v2()._domain_values()["target"]
        assert "r1" in _v1()._domain_values()["target"]

    def test_the_target_mask_covers_exactly_the_rows(self):
        env = _v2(entities=_spread_out()["entities"])
        mask = env.action_masks()[_slice(env, "target")]
        assert mask[UNUSED]
        assert int(mask[1:].sum()) == 3


class TestPlacementIsATile:
    def test_slot_index_is_the_fixed_offset_from_the_character(self):
        env = _v2()
        places = env._domain_values()["placement"]
        assert len(places) == SIDE * SIDE
        for index, position in enumerate(places):
            dx, dy = divmod(index, SIDE)
            assert position == [dx - PLACEMENT_RADIUS + 0.5, dy - PLACEMENT_RADIUS + 0.5]

    def test_occupancy_moves_the_mask_and_not_the_meaning(self):
        """The property v1 does not have: a slot names one tile, always."""
        empty = _v2(entities=[], terrain={"blocked": []})
        blocked = _v2(entities=[], terrain={"blocked": [[2.5, 1.5]]})
        assert empty._domain_values()["placement"] == blocked._domain_values()["placement"]
        slot = (2 + PLACEMENT_RADIUS) * SIDE + (1 + PLACEMENT_RADIUS)
        assert empty.action_masks()[_slice(empty, "placement")][slot + 1]
        assert not blocked.action_masks()[_slice(blocked, "placement")][slot + 1]

    def test_v1_renumbers_its_tiles_when_something_is_built(self):
        """The same comparison under v1, which is why v2 exists."""
        empty = _v1(entities=[], terrain={"blocked": []})
        blocked = _v1(entities=[], terrain={"blocked": [[-4.5, -4.5]]})
        assert empty._domain_values()["placement"] != blocked._domain_values()["placement"]

    def test_the_character_tile_is_masked_rather_than_missing(self):
        env = _v2(entities=[], terrain={"blocked": []})
        here = PLACEMENT_RADIUS * SIDE + PLACEMENT_RADIUS
        assert env._domain_values()["placement"][here] == [0.5, 0.5]
        assert not env.action_masks()[_slice(env, "placement")][here + 1]


class TestDecode:
    def _place_vector(self, env, placement):
        names = [n for n, _ in DIMENSIONS]
        operation = next(
            index
            for index, template in enumerate(env.env.catalog.templates)
            if template.key == PLACE
        )
        vector = [UNUSED] * len(names)
        vector[0] = operation
        vector[names.index("placement")] = placement
        vector[names.index("direction")] = 1
        vector[names.index("item")] = list(encoders.ITEMS).index("transport-belt") + 1
        return np.asarray(vector, dtype=np.int64)

    def test_an_occupied_slot_is_a_counted_decode_failure(self):
        env = _v2(entities=[], terrain={"blocked": [[2.5, 1.5]]})
        slot = (2 + PLACEMENT_RADIUS) * SIDE + (1 + PLACEMENT_RADIUS)
        _operation, arguments, failure = env.decode(self._place_vector(env, slot + 1))
        assert arguments == {}
        assert failure is not None and "occupied" in failure

    def test_a_free_slot_decodes_to_its_tile(self):
        env = _v2(entities=[], terrain={"blocked": []})
        slot = (2 + PLACEMENT_RADIUS) * SIDE + (1 + PLACEMENT_RADIUS)
        _operation, arguments, failure = env.decode(self._place_vector(env, slot + 1))
        assert failure is None
        assert arguments["position"] == [2.5, 1.5]

    def test_encode_round_trips_through_decode(self):
        env = _v2(entities=[], terrain={"blocked": []})
        operation = next(
            index
            for index, template in enumerate(env.env.catalog.templates)
            if template.key == PLACE
        )
        arguments = {"position": [2.5, 1.5], "direction": "north", "item": "transport-belt"}
        vector = env.encode(operation, arguments)
        assert env.decode(vector) == (operation, arguments, None)


class TestNoDimensionIsEverEmpty:
    @pytest.mark.parametrize(
        "observation",
        [
            _observation(entities=[], resources={"tiles": []}, inventory={}),
            _observation(inventory={}),
            _observation(terrain={"blocked": [[float(x) + 0.5, 0.5] for x in range(-5, 6)]}),
        ],
    )
    def test_v2_never_produces_an_all_false_dimension(self, observation):
        env = ParameterizedEnv(_Inner(observation), profile="v2")
        mask = env.action_masks()
        offset = 0
        for size in env.action_space.nvec:
            assert mask[offset : offset + int(size)].any()
            offset += int(size)
