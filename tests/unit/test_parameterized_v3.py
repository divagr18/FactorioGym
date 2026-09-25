"""`v3`: the belt-logistics layout and its action profile.

`v1` and `v2` must not move by a bit, so every assertion about them here is an
equality with what they were. `v3` is pinned by what each new slot means,
because the simulator implements the same rules (`factory-sim`,
`csrc/fsim_rl.c`) and its contract test compares the two on every decision of
the golden traces (`tools/v3_contract_golden.py`).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from factoriorl import catalog as catalog_module
from factoriorl import encoders
from factoriorl.env import PLACEMENT_RADIUS_V3
from factoriorl.parameterized import DIMENSIONS, UNUSED, ParameterizedEnv, wrap_for_policy
from factoriorl.tasks import get

pytest.importorskip("gymnasium")

from test_parameterized_policy import _Inner, _observation  # noqa: E402

SIDE = 2 * PLACEMENT_RADIUS_V3 + 1


class _InnerV3(_Inner):
    def __init__(self, observation=None):
        super().__init__(observation, catalog="parameterized-v3")
        self.v3 = True
        self.layout = encoders.LAYOUT_V3
        self.placement_radius = PLACEMENT_RADIUS_V3
        self.observation_space = encoders.observation_space(layout=self.layout)


def _v3(**overrides) -> ParameterizedEnv:
    return ParameterizedEnv(_InnerV3(_observation(**overrides)), profile="v3")


def _slice(env, name: str) -> slice:
    names = [n for n, _ in DIMENSIONS]
    start = int(sum(env.action_space.nvec[: names.index(name)]))
    return slice(start, start + int(env.action_space.nvec[names.index(name)]))


def _ops(env) -> dict[str, bool]:
    mask = env.action_masks()[: len(env.env.catalog)]
    return dict(zip(env.env.catalog.keys(), mask, strict=True))


class TestLayout:
    def test_v1_is_unchanged(self):
        space = encoders.observation_space()
        assert space["entities"].shape == (32, 16)
        assert space["inventory"].shape == (14,)
        assert space["goal"].shape == (12,)
        assert encoders.ITEMS_V3[: len(encoders.ITEMS)] == encoders.ITEMS

    def test_v3_shapes(self):
        encoded = encoders.encode(_observation(), layout=encoders.LAYOUT_V3)
        assert encoded["entities"].shape == (96, 32)
        assert encoded["entity_mask"].shape == (96,)
        assert encoded["inventory"].shape == (18,)
        assert encoded["goal"].shape == (30,)
        assert encoded["grid"].shape == (6, 65, 65)

    def test_the_first_sixteen_features_are_v1s(self):
        observation = _observation(
            entities=[
                {"h": "a", "p": [3.5, 0.5], "type": "transport-belt", "d": 4, "lanes": [2, 1]},
                {"h": "b", "p": [1, 1], "type": "furnace", "fuel": {"coal": 3}, "st": "working",
                 "working": True, "contents": {"iron-ore": 2}},
            ]
        )  # fmt: skip
        v1 = encoders.encode(observation)
        v3 = encoders.encode(observation, layout=encoders.LAYOUT_V3)
        np.testing.assert_array_equal(v3["entities"][:32, :16], v1["entities"])
        np.testing.assert_array_equal(v3["grid"], v1["grid"])
        np.testing.assert_array_equal(v3["self"][:12], v1["self"])
        np.testing.assert_array_equal(v3["inventory"][:14], v1["inventory"])

    def test_the_thirteenth_self_feature_is_the_free_share_of_the_main_inventory(self):
        character = {"position": [0.0, 0.0], "slots": {"free": 20, "total": 80}}
        encoded = encoders.encode(_observation(character=character), layout=encoders.LAYOUT_V3)
        assert encoded["self"].shape == (13,)
        assert encoded["self"][12] == np.float32(0.25)
        # No slot count in the observation: 0, as for a full inventory.
        assert encoders.encode(_observation(), layout=encoders.LAYOUT_V3)["self"][12] == 0.0

    def test_ninety_six_rows(self):
        entities = [
            {"h": f"e{i}", "p": [i + 0.5, 0.5], "type": "transport-belt"} for i in range(120)
        ]
        encoded = encoders.encode(_observation(entities=entities), layout=encoders.LAYOUT_V3)
        assert int(encoded["entity_mask"].sum()) == 96

    def test_logistics_features(self):
        observation = _observation(
            entities=[
                {"h": "belt", "p": [1.5, 0.5], "type": "transport-belt", "lanes": [3, 9],
                 "shape": "right"},
                {"h": "ins", "p": [0.5, 2.5], "type": "inserter", "held": "iron-plate",
                 "pickup": [0.5, 1.5], "drop": [0.5, 3.69921875]},
                {"h": "drill", "p": [-3, -3], "type": "mining-drill",
                 "drop": [-3.5, -4.296875]},
                {"h": "chest", "p": [4.5, 4.5], "type": "container",
                 "contents": {"iron-plate": 7, "coal": 2}},
            ]
        )  # fmt: skip
        encoded = encoders.encode(observation, layout=encoders.LAYOUT_V3)
        rows = {
            record["h"]: encoded["entities"][k]
            for k, (_, record, _) in enumerate(
                encoders.entity_row_order(observation, [0.0, 0.0], 96)
            )
        }
        n = len(encoders.ITEMS_V3)
        belt, ins, drill, chest = rows["belt"], rows["ins"], rows["drill"], rows["chest"]
        assert (belt[16], belt[17], belt[18], belt[19]) == (3 / 8, 1.0, 0.0, 1.0)
        assert ins[20] == 1.0
        assert ins[21] == np.float32((encoders.ITEMS_V3.index("iron-plate") + 1) / n)
        assert (ins[22], ins[23], ins[24]) == (0.0, -0.5, 1.0)
        assert (ins[25], ins[26], ins[27]) == (0.0, np.float32(1.19921875 / 2), 1.0)
        assert (drill[25], drill[26], drill[27]) == (-0.25, np.float32(-1.296875 / 2), 1.0)
        assert drill[24] == 0.0
        assert chest[28] == np.float32((encoders.ITEMS_V3.index("iron-plate") + 1) / n)
        assert not np.stack(list(rows.values()))[:, 29:].any()

    def test_ties_in_contents_go_to_the_lower_index(self):
        assert encoders.dominant_item({"iron-plate": 5, "coal": 5}, encoders.ITEMS_V3) == "coal"
        assert encoders.dominant_item({"stone-wall": 9}, encoders.ITEMS_V3) is None

    def test_remembered_rows_carry_no_logistics(self):
        observation = _observation(
            entities=[],
            remembered=[{"h": "r", "p": [40.5, 0.5], "type": "transport-belt", "d": 4, "age": 60}],
        )
        row = encoders.encode(observation, layout=encoders.LAYOUT_V3)["entities"][0]
        assert row[9] == 1.0 and not row[16:].any()


class TestMarkers:
    def test_markers_follow_public_marker_order(self):
        inner = _InnerV3(_observation(goal={"patch": [64.5, -200.0]}))
        inner.spec_ = get("construct_smelting_line").spec
        slots = inner.marker_slots()
        assert slots.shape == (18,)
        assert inner.spec_.public_markers[0] == "patch"
        assert list(slots[:3]) == [np.float32(64.5 / 128), -1.0, 1.0]
        assert not slots[3:].any()

    def test_v3_goal_is_v1_goal_then_markers(self):
        inner = _InnerV3(_observation(goal={"gap": [3.5, 0.5]}))
        goal = inner._goal_vector()
        assert goal.shape == (30,)
        v1 = _Inner(_observation(goal={"gap": [3.5, 0.5]}))._goal_vector()
        np.testing.assert_array_equal(goal[:12], v1)


class TestActionSpace:
    def test_shape(self):
        assert list(_v3().action_space.nvec) == [25, 97, 226, 5, 19, 4]

    def test_catalog_selects_the_profile(self):
        v1 = catalog_module.resolve("parameterized-v1")
        v3 = catalog_module.resolve("parameterized-v3")
        assert v3.keys() == (*v1.keys(), "mine_tile", "take_fuel", "finish")
        assert v3.digest() != v1.digest()
        assert v1.digest() == "7222fb372fe51f63", "parameterized-v1's frozen digest moved"
        assert wrap_for_policy(_InnerV3()).profile == "v3"
        assert wrap_for_policy(_Inner()).profile == "v1"

    def test_a_v1_window_refuses_v3(self):
        with pytest.raises(ValueError, match="placement window"):
            ParameterizedEnv(_Inner(_observation()), profile="v3")

    def test_placement_slot_is_a_fixed_tile_of_the_15_window(self):
        env = _v3()
        positions, _legal = env.env.placement_grid()
        assert len(positions) == SIDE * SIDE == 225
        dx, dy = 3, -6
        slot = (dx + PLACEMENT_RADIUS_V3) * SIDE + (dy + PLACEMENT_RADIUS_V3)
        assert positions[slot] == [dx + 0.5, dy + 0.5]
        mask = env.action_masks()[_slice(env, "placement")]
        assert mask[UNUSED]
        here = PLACEMENT_RADIUS_V3 * SIDE + PLACEMENT_RADIUS_V3
        assert not mask[here + 1]
        assert mask[slot + 1]

    def test_occupied_tile_is_masked_and_refused(self):
        env = _v3()
        slot = (3 + PLACEMENT_RADIUS_V3) * SIDE + PLACEMENT_RADIUS_V3  # the belt at (3.5, 0.5)
        assert not env.action_masks()[_slice(env, "placement")][slot + 1]
        place = env.env.catalog.keys().index("place_at")
        _, _, failure = env.decode([place, 0, slot + 1, 1, 9, 0])
        assert failure and "occupied" in failure

    def test_items_are_the_v3_vocabulary(self):
        env = _v3(inventory={"boiler": 1, "transport-belt": 2})
        mask = env.action_masks()[_slice(env, "item")]
        assert mask[encoders.ITEMS_V3.index("boiler") + 1]
        assert mask[encoders.ITEMS_V3.index("transport-belt") + 1]
        assert not mask[encoders.ITEMS_V3.index("coal") + 1]
        # A locked recipe: the mod's `place` refuses the item.
        locked = _v3(inventory={"boiler": 1}, recipes=["transport-belt"])
        assert not _ops(locked)["place_at"]

    def test_no_rows_masks_row_operations(self):
        """v3 masks an operation whose target dimension has only the sentinel."""
        env = _v3(entities=[])  # a resource tile is in view, no entity
        ops = _ops(env)
        assert not ops["mine_at"] and not ops["rotate_at"] and not ops["give_to"]
        v2 = ParameterizedEnv(_Inner(_observation(entities=[])), profile="v2")
        assert _ops(v2)["mine_at"], "v2 keeps its masks"

    def test_set_recipe_and_craft_stay_masked(self):
        ops = _ops(_v3())
        assert not ops["set_recipe_at"] and not ops["craft_recipe"] and not ops["cancel_request"]
        assert ops["wait"] and ops["place_at"] and ops["move_north"]


def test_the_local_v3_sensor_carries_what_the_layout_reads():
    """`local-v3` sweeps as many entities as the layout has rows, with the detail."""
    import re

    from factoriorl.paths import mod_source_dir

    source = (mod_source_dir() / "factoriorl" / "profiles.lua").read_text(encoding="utf-8")
    block = re.search(r'\["local-v3"\] = \{(.*?)\n  \},', source, re.DOTALL)
    assert block, "the local-v3 observation profile was not found in profiles.lua"
    body = block.group(1)
    assert re.search(r"^\s*entity_cap = (\d+),", body, re.MULTILINE).group(1) == str(
        encoders.MAX_ENTITIES_V3
    )
    assert re.search(r"^\s*logistics_detail = true,", body, re.MULTILINE)
    assert re.search(r"^\s*radius = (\d+),", body, re.MULTILINE).group(1) == str(
        encoders.LOCAL_V1.radius
    )


def test_the_v3_contract_file_is_what_the_encoder_produces():
    """`docs/evidence/sim-parity/v3_contract.json.xz` is what factory-sim is held to.

    Regenerated from the traces for two short scenarios, so a change to the `v3`
    encoder or masks that forgets to regenerate the file fails here, not in the
    other repository.
    """
    import json
    import lzma
    import sys
    from pathlib import Path

    tools = Path(__file__).resolve().parents[2] / "tools"
    sys.path.insert(0, str(tools))
    try:
        import v3_contract_golden as golden
    finally:
        sys.path.remove(str(tools))
    if not golden.OUT.is_file():
        pytest.skip("no v3 contract file")
    stored = json.loads(lzma.decompress(golden.OUT.read_bytes()))
    index = json.loads((golden.EVIDENCE / "index.json").read_text(encoding="utf-8"))
    for name in ("placement_footprints", "logistics_belt_rotate_and_mine"):
        assert golden.scenario(name, index[name]) == stored["scenarios"][name], name


class TestReachAndHandMining:
    """User decisions 1 and 2 (2026-09-25): hand-mining in v3, and a mask that
    is legal exactly where the game accepts (docs/sim-logistics.md)."""

    def test_placement_needs_build_distance(self):
        # Off-centre character: the far corner of the window is beyond 10.
        env = _v3(character={"position": [0.1, 0.1]}, entities=[])
        positions, legal = env.env.placement_grid()
        for (x, y), ok in zip(positions, legal, strict=True):
            near = ((x - 0.1) ** 2 + (y - 0.1) ** 2) ** 0.5 <= 10
            if (math.floor(x), math.floor(y)) != (0, 0):
                assert ok == near, (x, y)
        assert not all(legal)

    def test_target_rows_need_reach_and_sight(self):
        entities = [
            {"h": "near", "name": "wooden-chest", "p": [3.5, 0.5], "type": "container"},
            # 10.5 - 0.34765625 = 10.15: out of reach.
            {"h": "far", "name": "wooden-chest", "p": [10.5, 0.5], "type": "container"},
            # 10.5 - 0.69921875 = 9.8: a furnace that far is in reach.
            {"h": "furnace", "name": "stone-furnace", "p": [10.5, 0.0], "type": "furnace"},
        ]
        remembered = [{"h": "old", "name": "wooden-chest", "p": [40.5, 0.5], "type": "container"}]
        env = _v3(entities=entities, remembered=remembered)
        handles = env.env.entity_row_handles()
        legal = dict(zip(handles, env.env.target_row_legal(), strict=True))
        assert legal == {"near": True, "far": False, "furnace": True, "old": False}
        mask = env.action_masks()[_slice(env, "target")]
        for k, handle in enumerate(handles):
            assert mask[k + 1] == legal[handle]
        mine = env.env.catalog.keys().index("mine_at")
        _, _, failure = env.decode([mine, handles.index("far") + 1, 0, 0, 0, 0])
        assert failure and "reach" in failure

    def test_mine_tile_names_a_resource_by_its_tile(self):
        tiles = [{"h": "r1", "name": "coal", "p": [2.5, 0.5]},
                 {"h": "r2", "name": "coal", "p": [3.5, 0.5]}]  # fmt: skip
        env = _v3(entities=[], resources={"tiles": tiles})
        op = env.env.catalog.keys().index("mine_tile")
        assert op == 22
        slot = (2 + PLACEMENT_RADIUS_V3) * SIDE + PLACEMENT_RADIUS_V3
        operation, arguments, failure = env.decode([op, 0, slot + 1, 0, 0, 2])
        assert failure is None and arguments == {"tile": "r1", "count": 5}
        assert list(env.encode(op, {"tile": "r1", "count": 5})) == [op, 0, slot + 1, 0, 0, 2]
        # 3.5 from the character: beyond resource reach (2.7).
        far = (3 + PLACEMENT_RADIUS_V3) * SIDE + PLACEMENT_RADIUS_V3
        _, _, failure = env.decode([op, 0, far + 1, 0, 0, 1])
        assert failure and "no resource tile" in failure
        assert env.env.argument_domains()["resource_tiles"] == ["r1"]
        ops = _ops(env)
        assert ops["mine_tile"] and not ops["mine_at"]

    def test_no_resource_in_reach_masks_mine_tile(self):
        env = _v3(resources={"tiles": [{"h": "r", "name": "coal", "p": [6.5, 0.5]}]})
        assert not _ops(env)["mine_tile"]

    def test_the_reach_rule_is_the_engines_on_the_sweep(self):
        """`entity_in_reach` against `can_reach_entity` on every probed position."""
        import json
        import lzma
        from pathlib import Path

        from factoriorl.env import entity_in_reach

        path = Path(__file__).resolve().parents[2] / "docs" / "evidence" / "handmine-reach.json.xz"
        doc = json.loads(lzma.decompress(path.read_bytes()))
        sub, off = doc["sub"], doc["offsets"]
        for spec in doc["types"]:
            ex, ey = float(spec["position"][0]), float(spec["position"][1])
            record = {"name": spec["name"], "p": [ex, ey]}
            for si, bits in enumerate(spec["grid"]):
                sx, sy = (si % sub) / sub, (si // sub) / sub
                k = 0
                for dy in range(-off, off + 1):
                    for dx in range(-off, off + 1):
                        character = (math.floor(ex) + dx + sx, math.floor(ey) + dy + sy)
                        assert entity_in_reach(character, record) == (bits[k] == "1"), (
                            spec["name"],
                            character,
                        )
                        k += 1


def _op_rows(env) -> dict[str, dict[str, np.ndarray]]:
    """Each operation's row of `operation_masks`, split by dimension."""
    masks = env.operation_masks()
    names = [n for n, _ in DIMENSIONS[1:]]
    sizes = env.argument_sizes()
    out = {}
    for op, key in enumerate(env.env.catalog.keys()):
        start, row = 0, {}
        for name, size in zip(names, sizes, strict=True):
            row[name] = masks[op, start : start + size]
            start += size
        out[key] = row
    return out


def _legal(row: np.ndarray) -> list[int]:
    return [int(i) for i in np.flatnonzero(row)]


class TestOperationMasks:
    """User decision "v3 masks: per operation" (option C, 2026-09-25): per
    operation, a mask over each argument dimension, legal where some whole
    combination is one the game accepts; the flat mask folds them."""

    ENTITIES = [
        {"h": "belt", "name": "transport-belt", "p": [2.5, 0.5], "type": "transport-belt"},
        {"h": "chest", "name": "wooden-chest", "p": [-2.5, 0.5], "type": "container",
         "contents": {"iron-plate": 30}},
        {"h": "furnace", "name": "stone-furnace", "p": [0.0, 3.0], "type": "furnace",
         "fuel": {"coal": 4}, "contents": {"iron-ore": 5}, "output": {"iron-plate": 2}},
        {"h": "ins", "name": "burner-inserter", "p": [0.5, -2.5], "type": "inserter",
         "fuel": {"wood": 1}},
        {"h": "pile", "name": "item-on-ground", "p": [-1.5, -1.5], "type": "item-entity",
         "contents": {"coal": 1}},
    ]  # fmt: skip

    def _env(self, **overrides):
        base = {
            "entities": self.ENTITIES,
            "inventory": {"coal": 10, "transport-belt": 3},
            "character": {"position": [0.5, 0.5], "slots": {"free": 70, "total": 80}},
            "recipes": ["transport-belt", "stone-furnace"],
        }
        return _v3(**{**base, **overrides})

    def _rows(self, env):
        return {h: k + 1 for k, h in enumerate(env.env.entity_row_handles())}

    def test_shapes_and_no_empty_dimension(self):
        env = self._env()
        masks = env.operation_masks()
        assert masks.shape == (25, 97 + 226 + 5 + 19 + 4)
        assert env.action_masks().shape == (25 + masks.shape[1],)
        for row in _op_rows(env).values():
            assert all(part.any() for part in row.values())

    def test_each_verb_names_only_what_it_can_act_on(self):
        env = self._env()
        rows, ops = self._rows(env), _op_rows(env)
        item = {name: encoders.ITEMS_V3.index(name) + 1 for name in encoders.ITEMS_V3}
        assert _legal(ops["rotate_at"]["target"]) == sorted([rows["belt"], rows["ins"]])
        # Coal into the furnace's fuel slot or the chest. The inserter's one
        # fuel slot holds wood, so it takes no coal; a belt and a pile have no
        # inventory.
        assert _legal(ops["give_to"]["target"]) == sorted([rows["chest"], rows["furnace"]])
        assert _legal(ops["give_to"]["item"]) == sorted([item["coal"], item["transport-belt"]])
        # take_from reads what some visible entity holds as contents or
        # output (the pile's coal among them): the chest's plates, the
        # furnace's ore, plates and -- its fuel slot is the first inventory
        # holding coal -- coal.
        assert _legal(ops["take_from"]["target"]) == sorted([rows["chest"], rows["furnace"]])
        assert _legal(ops["take_from"]["item"]) == sorted(
            [item["coal"], item["iron-ore"], item["iron-plate"]]
        )
        assert _legal(ops["take_fuel"]["target"]) == sorted([rows["furnace"], rows["ins"]])
        assert _legal(ops["take_fuel"]["item"]) == [UNUSED]
        # A pile off the patch yields nothing to mine_at.
        assert rows["pile"] not in _legal(ops["mine_at"]["target"])
        assert _legal(ops["place_at"]["item"]) == [item["transport-belt"]]
        assert _legal(ops["place_at"]["direction"]) == [1, 2, 3, 4]
        for key in ("finish", "wait", "move_north"):
            assert all(_legal(part) == [UNUSED] for part in ops[key].values())

    def test_the_flat_mask_is_the_union_over_legal_operations(self):
        env = self._env()
        masks, flat = env.operation_masks(), env.action_masks()
        legal = flat[:25]
        assert legal[env.env.catalog.keys().index("finish")]
        np.testing.assert_array_equal(flat[25:], masks[legal].any(axis=0))

    def test_a_running_mine_or_no_free_slot_masks_mining(self):
        busy = self._env(inflight=[{"action": "mine", "request_id": "r1"}])
        assert not _ops(busy)["mine_at"] and not _ops(busy)["mine_tile"]
        full = self._env(character={"position": [0.5, 0.5], "slots": {"free": 0, "total": 80}})
        assert not _ops(full)["mine_at"]
        assert _ops(self._env())["mine_at"]

    def test_no_room_masks_take(self):
        # Every slot holds wood: no room for plates, ore or coal.
        env = self._env(
            inventory={"wood": 8000},
            character={"position": [0.5, 0.5], "slots": {"free": 0, "total": 80}},
        )
        ops = _ops(env)
        assert not ops["take_from"] and not ops["take_fuel"]
        # Room for wood only (the last stack is part full): the inserter's.
        env = self._env(
            inventory={"wood": 7999},
            character={"position": [0.5, 0.5], "slots": {"free": 0, "total": 80}},
        )
        rows = self._rows(env)
        assert _legal(_op_rows(env)["take_fuel"]["target"]) == [rows["ins"]]

    def test_a_pile_on_a_resource_tile_is_mined_through_the_tile(self):
        tile = {"h": "t", "name": "iron-ore", "p": [1.5, 1.5]}
        pile = {"h": "t", "name": "item-on-ground", "p": [1.5, 1.5], "type": "item-entity",
                "contents": {"iron-ore": 1}}  # fmt: skip
        env = self._env(entities=[pile], resources={"tiles": [tile]})
        assert _legal(_op_rows(env)["mine_at"]["target"]) == [1]
        env = self._env(
            entities=[{**pile, "h": "p"}], resources={"tiles": [{**tile, "p": [9.5, 9.5]}]}
        )
        assert not _ops(env)["mine_at"]

    def test_take_fuel_decodes_to_a_transfer_of_what_the_slot_holds(self):
        env = self._env()
        op = env.env.catalog.keys().index("take_fuel")
        assert op == 23 and env.env.catalog.keys().index("finish") == 24
        rows = self._rows(env)
        _, arguments, failure = env.decode([op, rows["furnace"], 0, 0, 0, 3])
        assert failure is None and arguments == {"from": "furnace", "count": 20}


class TestFinish:
    """User decision (2026-09-25): `finish` runs the verification window now and
    ends the episode on its result."""

    def test_finish_verifies_once_and_terminates(self, monkeypatch):
        from test_construct_smelting_line import _stepping_env

        from factoriorl.env import FactorioEnv

        monkeypatch.setattr(encoders, "encode", lambda observation, goal: {})
        env, calls = _stepping_env(
            tick=300, steps=10, verified={"success": True, "reward": 0.7, "machine_output": 7,
                                          "reward_components": {"verified_output": 0.7}},
        )  # fmt: skip
        _, reward, terminated, truncated, info = FactorioEnv.finish(env)
        assert calls == [11], "one decision, and the verifier runs on it"
        assert (terminated, truncated) == (True, False)
        assert reward == pytest.approx(0.7) and info["verification"]["machine_output"] == 7
        assert info["action_key"] == "finish"

    def test_finish_decodes_with_no_arguments(self):
        env = _v3()
        op = env.env.catalog.keys().index("finish")
        assert env.decode([op, 5, 7, 1, 2, 3]) == (op, {}, None)
        assert _ops(env)["finish"]
