"""`belt_smelting`: the scene contract, the delivery verifier and the reference layout."""

from __future__ import annotations

import random

import pytest

from factoriorl.env import FactorioEnv
from factoriorl.rewards import RewardAccountant
from factoriorl.tasks import get
from factoriorl.tasks.families import belt_smelting as family
from factoriorl.tasks.spec import Predicate, PredicateKind, VerificationSpec, entity_tiles

FAMILY_NAMES = ("open", "walled", "split_patch", "obstructed", "far_chest")


def _rect(tile):
    return (tile[0], tile[1], tile[0], tile[1])


class TestTheSpec:
    def test_starting_inventory_is_the_parts_of_a_line(self):
        blueprint = family.generate(family.FAMILIES[0], random.Random(0))
        assert blueprint.character_inventory == {
            "burner-mining-drill": 4,
            "stone-furnace": 4,
            "transport-belt": 40,
            "burner-inserter": 10,
            "coal": 20,
        }
        assert set(blueprint.unlock_recipes) >= {
            "burner-mining-drill",
            "stone-furnace",
            "transport-belt",
            "burner-inserter",
        }

    def test_ten_minutes_of_delivery_after_the_construction_budget(self):
        spec = get("belt_smelting").spec
        assert spec.max_decision_steps == 2500
        assert spec.verification == VerificationSpec(
            "iron-plate", 150, 36000, source="iron-ore", container="output"
        )
        # Construction can use every decision the budget allows.
        assert spec.max_game_ticks - spec.verification.ticks == 2500 * spec.decision_ticks
        assert spec.success == (
            Predicate(PredicateKind.VERIFIED_MACHINE_OUTPUT, item="iron-plate", at_least=150),
        )

    def test_splits(self):
        spec = get("belt_smelting").spec
        assert [f.name for f in spec.families("train")] == ["open", "walled", "split_patch"]
        assert [f.name for f in spec.families("test")] == ["obstructed", "far_chest"]

    def test_all_three_sites_are_published(self):
        spec = get("belt_smelting").spec
        assert set(spec.public_markers) == {"iron", "coal", "output"}

    def test_a_container_is_serialized_only_when_declared(self):
        assert VerificationSpec("iron-plate", 10, 3600).to_dict() == {
            "item": "iron-plate",
            "target": 10,
            "ticks": 3600,
        }
        assert get("belt_smelting").spec.verification.to_dict()["container"] == "output"


@pytest.mark.parametrize("name", FAMILY_NAMES)
def test_success_is_false_at_reset(name):
    spec = get("belt_smelting").spec
    for seed in range(8):
        blueprint = family.generate(family.LayoutFamily(name), random.Random(seed))
        observation, truth = blueprint.initial_state()
        assert truth["containers"]["output"] == {}
        assert not any(p.evaluate(observation, truth) for p in spec.success)


def _verifier(before: dict, after: dict) -> FactorioEnv:
    """A `FactorioEnv` reduced to what `run_verification` reads."""
    spec = get("belt_smelting").spec
    env = object.__new__(FactorioEnv)
    env.spec_ = spec
    env._verification = None
    env._observation = {"tick": 40000}
    env._truth = dict(before)
    env.accountant = RewardAccountant(spec.rewards)
    env.accountant.reset(env._observation, env._truth)

    def advance(ticks: int) -> int:
        assert ticks == 36000
        env._observation = {"tick": 40000 + ticks}
        env._truth = dict(after)
        return ticks

    env.advance = advance
    return env


class TestTheVerifierCountsDeliveries:
    def test_plates_delivered_into_the_chest_count(self):
        env = _verifier(
            before={
                "containers": {"output": {"iron-plate": 3}},
                "machine_produced": {"iron-ore": 4},
            },
            after={
                "containers": {"output": {"iron-plate": 173}},
                # Production counts are ignored: only deliveries score.
                "machine_produced": {"iron-plate": 900, "iron-ore": 200},
            },
        )
        result = env.run_verification()
        assert result["uncapped_output"] == 170
        assert result["machine_output"] == 170
        assert result["container"] == "output"
        assert result["success"] is True
        assert result["reward"] == 1.0

    def test_partial_delivery_is_partial_credit(self):
        env = _verifier(
            before={"containers": {"output": {}}, "machine_produced": {}},
            after={
                "containers": {"output": {"iron-plate": 75}},
                "machine_produced": {"iron-plate": 75, "iron-ore": 80},
            },
        )
        result = env.run_verification()
        assert result["success"] is False
        assert result["reward"] == pytest.approx(0.5)

    def test_smelted_but_undelivered_plates_score_nothing(self):
        env = _verifier(
            before={"containers": {"output": {}}, "machine_produced": {}},
            after={
                "containers": {"output": {}},
                "machine_produced": {"iron-plate": 90, "iron-ore": 90},
            },
        )
        result = env.run_verification()
        assert result["machine_output"] == 0
        assert result["reward"] == 0.0

    def test_plates_from_hand_mined_ore_are_capped_away(self):
        """A furnace stuffed with hand-mined ore, feeding the chest, with no drill."""
        env = _verifier(
            before={"containers": {"output": {}}, "machine_produced": {"iron-ore": 0}},
            after={
                "containers": {"output": {"iron-plate": 70}},
                "machine_produced": {"iron-plate": 70, "iron-ore": 0},
            },
        )
        result = env.run_verification()
        assert result["uncapped_output"] == 70
        assert result["machine_output"] == 0
        assert result["success"] is False


@pytest.mark.parametrize("name", FAMILY_NAMES)
def test_the_generator_is_deterministic_and_varied(name):
    layout = family.LayoutFamily(name)
    first = family.generate(layout, random.Random(11))
    again = family.generate(layout, random.Random(11))
    assert first == again
    distinct = {repr(family.generate(layout, random.Random(seed))) for seed in range(60)}
    assert len(distinct) >= 32


@pytest.mark.parametrize("name", FAMILY_NAMES)
def test_scene_geometry(name):
    """Sites far enough apart that no position reaches two; walls keep off them."""
    for seed in range(40):
        s = family.scene(name, random.Random(seed))
        chest = _rect(s.chest)
        rects = (s.iron_rect, s.coal_rect, chest)
        for i in range(3):
            assert family._in_scene(rects[i])
            for j in range(i + 1, 3):
                assert family._gap_sq(rects[i], rects[j]) >= 21 * 21
                assert 40**2 <= family._centre_dist_sq4(rects[i], rects[j]) <= 80**2
        low, high = family.CHEST_MANHATTAN[name]
        assert low <= family._manhattan_gap(s.iron_rect, chest) <= high
        for wall in s.walls:
            assert family._wall_ok(wall, s.iron_rect, s.coal_rect, chest)
        assert s.start not in set(s.walls) and s.start != s.chest
        assert set(s.iron) <= {
            (x, y)
            for x in range(s.iron_rect[0], s.iron_rect[2] + 1)
            for y in range(s.iron_rect[1], s.iron_rect[3] + 1)
        }
        blueprint = family.generate(family.LayoutFamily(name), random.Random(seed))
        assert blueprint.footprint_conflicts() == []
        assert blueprint.character_obstructed() == []
        assert blueprint.out_of_box() == []


def test_walls_appear_only_where_declared():
    for seed in range(20):
        assert family.scene("open", random.Random(seed)).walls == ()
        assert family.scene("split_patch", random.Random(seed)).walls == ()
    assert any(family.scene("walled", random.Random(seed)).walls for seed in range(20))
    assert any(len(family.scene("obstructed", random.Random(seed)).walls) > 8 for seed in range(20))


def test_far_chest_clutter_stays_out_of_the_belt_corridor():
    """The family measures a long belt, not a detour added on top of one."""
    seen = 0
    for seed in range(30):
        s = family.scene("far_chest", random.Random(seed))
        x0 = min(s.iron_rect[0], s.chest[0]) - family.CORRIDOR_MARGIN
        y0 = min(s.iron_rect[1], s.chest[1]) - family.CORRIDOR_MARGIN
        x1 = max(s.iron_rect[2], s.chest[0]) + family.CORRIDOR_MARGIN
        y1 = max(s.iron_rect[3], s.chest[1]) + family.CORRIDOR_MARGIN
        for x, y in s.walls:
            assert not (x0 <= x <= x1 and y0 <= y <= y1)
        seen += len(s.walls)
    assert seen > 0


def test_split_patch_has_an_ore_free_strip():
    for seed in range(20):
        s = family.scene("split_patch", random.Random(seed))
        x0, y0, x1, y1 = s.iron_rect
        assert len(s.iron) == (x1 - x0 + 1) * (y1 - y0 + 1) - 2 * min(x1 - x0 + 1, y1 - y0 + 1)


def test_far_chest_is_farther_than_any_training_chest():
    train_high = max(family.CHEST_MANHATTAN[name][1] for name in ("open", "walled", "split_patch"))
    assert family.CHEST_MANHATTAN["far_chest"][0] > train_high


# ------------------------------------------------------------------ geometry


class TestMeasuredGeometry:
    """The planner's rotations against values read off the engine (2.0.60).

    A burner drill whose centre is lattice point (21, 21) drops at
    north (20.5, 19.70), east (22.30, 20.5), south (21.5, 22.30),
    west (19.70, 21.5); a burner inserter's direction points at its pickup tile.
    """

    MEASURED_DROP_TILES = {
        "north": (20, 19),
        "east": (22, 20),
        "south": (21, 22),
        "west": (19, 21),
    }

    @pytest.mark.parametrize("turns", range(4))
    def test_drill_drop_tile_rotates_with_the_cell(self, turns):
        facing = family._turn("south", turns)
        dx, dy = family._rotate_tile((0, 1), turns)
        assert (21 + dx, 21 + dy) == self.MEASURED_DROP_TILES[facing]


@pytest.mark.parametrize("name", FAMILY_NAMES)
def test_every_sampled_scene_has_a_line_within_forty_belts(name):
    for seed in range(12):
        s = family.scene(name, random.Random(seed))
        plan = family.plan_line(s)
        assert plan is not None, (name, seed)
        _check_plan(s, plan)


def _check_plan(s, plan):
    ore = set(s.iron)
    walls = set(s.walls)
    assert len(plan.belts) <= family.BELTS
    occupied: dict = {}
    for placement in [*plan.machines, *plan.inserters, *plan.belts, plan.chest_inserter]:
        tiles = entity_tiles(placement.item, placement.centre)
        assert sorted(tiles) == sorted(placement.tiles)
        for tile in tiles:
            assert tile not in occupied and tile not in walls and tile != s.chest
            occupied[tile] = placement
    furnaces = plan.furnaces
    belt_tiles = [p.tiles[0] for p in plan.belts]
    for drill in plan.drills:
        assert all(tile in ore for tile in drill.tiles)
        dx, dy = family.STEP[drill.direction]
        # The drop tile, by the measured rule, rotated with the drill's facing.
        drop = {
            "south": (round(drill.centre[0]), round(drill.centre[1]) + 1),
            "north": (round(drill.centre[0]) - 1, round(drill.centre[1]) - 2),
            "east": (round(drill.centre[0]) + 1, round(drill.centre[1]) - 1),
            "west": (round(drill.centre[0]) - 2, round(drill.centre[1])),
        }[drill.direction]
        assert any(drop in f.tiles for f in furnaces)
    for inserter in plan.inserters:
        tile = inserter.tiles[0]
        dx, dy = family.STEP[inserter.direction]
        pickup, drop = (tile[0] + dx, tile[1] + dy), (tile[0] - dx, tile[1] - dy)
        assert any(pickup in f.tiles for f in furnaces)
        assert drop in belt_tiles
    for here, nxt, belt in zip(belt_tiles, belt_tiles[1:], plan.belts, strict=False):
        dx, dy = family.STEP[belt.direction]
        assert (here[0] + dx, here[1] + dy) == nxt
    chest_ins = plan.chest_inserter.tiles[0]
    dx, dy = family.STEP[plan.chest_inserter.direction]
    assert (chest_ins[0] + dx, chest_ins[1] + dy) == belt_tiles[-1]
    assert (chest_ins[0] - dx, chest_ins[1] - dy) == s.chest
    last = plan.belts[-1]
    ldx, ldy = family.STEP[last.direction]
    assert (belt_tiles[-1][0] + ldx, belt_tiles[-1][1] + ldy) == chest_ins


def test_the_solver_rebuilds_the_installed_scene():
    """`_episode_scene` must replay `prepare_scene`'s draws: family, then scene."""
    from factoriorl.seeding import Branch, SeedPlan

    plan = SeedPlan(master=5, run_id="belt-test")

    class _Env:
        spec_ = get("belt_smelting").spec
        split = "test"
        seed_plan = plan
        branch = Branch.TRAIN
        _episode_index = 7

    rng = plan.generator_rng(Branch.TRAIN, 7)
    families = _Env.spec_.families("test")
    chosen = families[rng.randrange(len(families))]
    installed = family.generate(chosen, rng)
    rebuilt = family._episode_scene(_Env())
    assert installed.character_position == (rebuilt.start[0] + 0.5, rebuilt.start[1] + 0.5)
    assert installed.markers["iron"] == family._patch_centre(rebuilt.iron)


class TestTheTargetNeedsTheCoalPatch:
    """1.1.0's target, against the burn rates it was set from."""

    def test_twenty_coal_cannot_reach_the_target_even_spent_perfectly(self):
        # Every joule of the starting coal, plus the quarter wood each of the
        # three inserters is built burning, spent inside the window, and the
        # chest inserter charged only a chest-to-chest swing: an upper bound.
        per_plate = (
            family.DRILL_J_PER_ORE + family.FURNACE_J_PER_PLATE + 2 * family.INSERTER_J_PER_SWING
        )
        ceiling = (20 * family.COAL_J + 3 * family.INSERTER_BUILT_J) / per_plate
        assert 75 < ceiling < 80
        assert ceiling < 0.6 * family.TARGET_PLATES

    def test_the_split_of_twenty_coal(self):
        """What the reference does with no coal supply: sized for 53 plates."""
        assert family.fuel_plan(20) == family.FuelPlan(
            plates=53, drill=4, furnace=3, output_inserter=1, chest_inserter=3
        )

    def test_the_reference_coal_supply_clears_the_target_by_ten_percent(self):
        plan = family.fuel_plan(20 + family.EXTRA_COAL)
        assert plan.plates >= 1.1 * family.TARGET_PLATES

    @pytest.mark.parametrize("coal", [5, 20, 33, 60, 91, 200])
    def test_a_plan_never_spends_more_coal_than_it_has(self, coal):
        plan = family.fuel_plan(coal)
        spent = 2 * plan.drill + 2 * plan.furnace + 2 * plan.output_inserter + plan.chest_inserter
        assert spent <= coal
        assert plan.plates <= family.CELL_PLATE_CEILING
        # The drills are fuelled for the plates, and the rest for more.
        assert 2 * plan.drill * family.COAL_J >= plan.plates * family.DRILL_J_PER_ORE
        assert (
            2 * plan.furnace * family.COAL_J
            >= plan.plates * family.FUEL_MARGIN * family.FURNACE_J_PER_PLATE
        )

    def test_the_target_fits_the_machines_handed_over(self):
        """Four drills mine 0.25 ore/s each; the window is 600 s."""
        assert family.TARGET_PLATES < 4 * 0.25 * family.VERIFICATION_TICKS / 60
        assert family.TARGET_PLATES < family.CELL_PLATE_CEILING
