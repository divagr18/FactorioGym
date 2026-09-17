"""The line potential factory-sim shapes construct_smelting_line with."""

from __future__ import annotations

import pytest

from factoriorl.tasks.potentials import line_potential

TRUTH = {"markers": {"patch": [0.0, 0.0]}}


def observation(*entities, at=(0.0, 0.0)):
    return {"character": {"position": list(at)}, "entities": list(entities)}


def drill(x, y, d=8, fuel=0):
    record = {"name": "burner-mining-drill", "p": [x, y], "d": d}
    if fuel:
        record["fuel"] = {"coal": fuel}
    return record


def furnace(x, y, fuel=0, ore=0, plates=0):
    record = {"name": "stone-furnace", "p": [x, y]}
    if fuel:
        record["fuel"] = {"coal": fuel}
    if ore:
        record["contents"] = {"iron-ore": ore}
    if plates:
        record["output"] = {"iron-plate": plates}
    return record


def test_approach_alone_scales_with_distance_and_bottoms_out():
    assert line_potential(observation(), TRUTH, "patch") == pytest.approx(0.1)
    assert line_potential(observation(at=(32.0, 0.0)), TRUTH, "patch") == pytest.approx(0.05)
    assert line_potential(observation(at=(100.0, 0.0)), TRUTH, "patch") == 0.0
    assert line_potential(observation(), TRUTH, None) == 0.0


def test_a_south_facing_drill_feeds_the_furnace_two_tiles_below():
    # drop point (1.5, 2.296875): inside [1, 3) x [2, 4) for a furnace at (2, 3)
    assert line_potential(observation(drill(1, 1), furnace(2, 3)), TRUTH, "patch") == (
        pytest.approx(0.1 + 0.2 + 0.3)
    )
    assert line_potential(observation(drill(1, 1), furnace(1, 3)), TRUTH, "patch") == (
        pytest.approx(0.6)
    )
    # (0, 3) does not hold the drop point: measured on the engine, it never feeds
    assert line_potential(observation(drill(1, 1), furnace(0, 3)), TRUTH, "patch") == (
        pytest.approx(0.3)
    )
    # facing north, the same pair is not a line
    assert line_potential(observation(drill(1, 1, d=0), furnace(2, 3)), TRUTH, "patch") == (
        pytest.approx(0.3)
    )


def test_fuel_and_throughput_pay_on_the_best_line():
    full = observation(drill(1, 1, fuel=5), furnace(2, 3, fuel=5, ore=1))
    assert line_potential(full, TRUTH, "patch") == pytest.approx(0.9)
    plates = observation(drill(1, 1), furnace(2, 3, plates=3))
    assert line_potential(plates, TRUTH, "patch") == pytest.approx(0.7)
    # a fuelled drill elsewhere does not lend its fuel to an unfuelled line
    split = observation(drill(1, 1), furnace(2, 3), drill(20, 20, fuel=5))
    assert line_potential(split, TRUTH, "patch") == pytest.approx(0.6)


def test_a_drill_feeding_a_drill_is_not_a_line():
    pair = observation(drill(1, 1), drill(2, 3, fuel=5))
    assert line_potential(pair, TRUTH, "patch") == pytest.approx(0.3)
