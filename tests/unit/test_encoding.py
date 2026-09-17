"""Engine-free coverage for the observation encoder (PLAN.md 3.4).

``encoders.encode`` rasterises sparse wire lists into the dense tensors the
policy consumes, and until now the only thing checking its output was the
space-containment assertion in ``gate_phase3`` -- which needs a live Factorio
worker, so a shape or plane-semantics regression could sit in the tree for a
whole phase before anything noticed.

The radius-agreement test at the bottom is the important one: Python and Lua
declare the sensor radius independently, and a divergence is invisible. The
grid stays the shape ``observation_space`` promises, so ``contains()`` keeps
passing while the encoder's bounds check silently drops every tile beyond the
smaller of the two radii.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from factoriorl.encoders import (
    GOAL_FEATURES,
    ITEMS,
    LOCAL_V1,
    LOCAL_V1_COARSE,
    MAX_ENTITIES,
    RESOURCES,
    ObservationProfile,
    encode,
    observation_space,
)
from factoriorl.paths import mod_source_dir

ORIGIN = [10.0, -4.0]

#: Plane layout: one presence plane per resource, then the amount plane, then
#: the obstacle plane. Derived, never hardcoded -- adding a resource shifts the
#: last two and a test with literal indices would keep passing on the old ones.
AMOUNT_PLANE = len(RESOURCES)
OBSTACLE_PLANE = len(RESOURCES) + 1


def _observation(tiles=(), blocked=()) -> dict:
    """A minimal wire observation with the sensor parked at ``ORIGIN``."""
    return {
        "sensor": {"origin": list(ORIGIN)},
        "resources": {"tiles": list(tiles)},
        "terrain": {"blocked": [list(position) for position in blocked]},
        "entities": [],
        "remembered": [],
        "character": {"position": list(ORIGIN)},
        "inventory": {},
    }


def _at(offset_x: float, offset_y: float) -> list[float]:
    return [ORIGIN[0] + offset_x, ORIGIN[1] + offset_y]


def _cell(offset_x: int, offset_y: int, profile: ObservationProfile) -> tuple[int, int]:
    """The (row, col) a tile-space offset from the sensor origin lands in."""
    return (
        (offset_y + profile.radius) // profile.cell_size,
        (offset_x + profile.radius) // profile.cell_size,
    )


# ------------------------------------------------------------- space contract


def test_encoded_observation_matches_the_declared_space():
    encoded = encode(_observation())
    space = observation_space()

    assert set(encoded) == set(space.spaces), "encoded keys and space keys disagree"
    for key, subspace in space.spaces.items():
        assert encoded[key].shape == subspace.shape, f"{key}: shape disagrees with the space"
        assert encoded[key].dtype == subspace.dtype, f"{key}: dtype disagrees with the space"
    assert space.contains(encoded)


def test_default_profile_grid_is_sixty_five_squared():
    assert LOCAL_V1.grid_size == 65
    assert encode(_observation())["grid"].shape == (len(RESOURCES) + 2, 65, 65)


def test_vector_shapes_follow_the_vocabularies():
    encoded = encode(_observation())
    assert encoded["inventory"].shape == (len(ITEMS),)
    assert encoded["entities"].shape[0] == MAX_ENTITIES
    assert encoded["goal"].shape == (GOAL_FEATURES,)


# ------------------------------------------------------------ plane semantics


def test_resource_amount_and_obstacle_planes_land_where_they_should():
    ore_row, ore_col = _cell(3, -2, LOCAL_V1)
    blocked_row, blocked_col = _cell(-5, 7, LOCAL_V1)
    grid = encode(
        _observation(
            tiles=[{"name": "iron-ore", "p": _at(3, -2), "amount": 1200}],
            blocked=[_at(-5, 7)],
        )
    )["grid"]

    iron = RESOURCES.index("iron-ore")
    assert grid[iron, ore_row, ore_col] == 1.0
    assert grid[AMOUNT_PLANE, ore_row, ore_col] > 0.0
    assert grid[OBSTACLE_PLANE, blocked_row, blocked_col] == 1.0
    assert OBSTACLE_PLANE == grid.shape[0] - 1, "the obstacle plane must stay last"

    # One ore tile must not light up the other resources' planes, and must not
    # bleed into a neighbouring cell: a policy reads presence per plane per cell.
    for index, name in enumerate(RESOURCES):
        if name != "iron-ore":
            assert grid[index].sum() == 0.0, f"{name} plane populated by an iron-ore tile"
    assert grid[iron].sum() == 1.0
    assert grid[OBSTACLE_PLANE].sum() == 1.0


def test_unknown_resource_names_populate_the_amount_plane_only():
    """The mod's resource vocabulary can outgrow ``RESOURCES``; that must not
    scribble on an unrelated plane or raise."""
    ore_row, ore_col = _cell(1, 1, LOCAL_V1)
    grid = encode(_observation(tiles=[{"name": "uranium-ore", "p": _at(1, 1), "amount": 500}]))[
        "grid"
    ]

    assert grid[AMOUNT_PLANE, ore_row, ore_col] > 0.0
    assert grid[: len(RESOURCES)].sum() == 0.0


def test_the_amount_plane_takes_the_richest_tile_in_a_shared_cell():
    """At ``cell_size > 1`` several tiles share a cell. The amount must not
    depend on the order the mod serialised them in."""
    rich = {"name": "iron-ore", "p": _at(0, 0), "amount": 4000}
    poor = {"name": "iron-ore", "p": _at(1, 1), "amount": 1}
    row, col = _cell(0, 0, LOCAL_V1_COARSE)
    assert _cell(1, 1, LOCAL_V1_COARSE) == (row, col), "the fixture tiles must share a cell"

    forward = encode(_observation(tiles=[rich, poor]), profile=LOCAL_V1_COARSE)["grid"]
    reverse = encode(_observation(tiles=[poor, rich]), profile=LOCAL_V1_COARSE)["grid"]

    assert forward[AMOUNT_PLANE, row, col] == reverse[AMOUNT_PLANE, row, col]
    assert forward[AMOUNT_PLANE, row, col] == pytest.approx(
        encode(_observation(tiles=[rich]), profile=LOCAL_V1_COARSE)["grid"][AMOUNT_PLANE, row, col]
    )


# --------------------------------------------------------------------- bounds


@pytest.mark.parametrize(
    "offset",
    [(33, 0), (-33, 0), (0, 33), (0, -33), (500, 500), (-1000, 7)],
)
def test_positions_outside_the_radius_are_dropped(offset):
    """Dropping is deliberate. Wrapping or clamping would paint a distant patch
    onto the grid edge, where the policy reads it as something within reach."""
    grid = encode(
        _observation(
            tiles=[{"name": "iron-ore", "p": _at(*offset), "amount": 900}],
            blocked=[_at(*offset)],
        )
    )["grid"]
    assert not grid.any(), f"offset {offset} lies outside the radius but reached the grid"


def test_the_radius_boundary_itself_is_kept():
    row, col = _cell(32, -32, LOCAL_V1)
    grid = encode(_observation(blocked=[_at(32, -32)]))["grid"]
    assert grid[OBSTACLE_PLANE, row, col] == 1.0


# ------------------------------------------------------------ coarse profile


def test_the_coarse_profile_halves_the_grid_and_still_fits_its_space():
    assert LOCAL_V1_COARSE.cell_size == 2
    assert LOCAL_V1_COARSE.grid_size == 33

    encoded = encode(
        _observation(
            tiles=[{"name": "coal", "p": _at(-6, 4), "amount": 300}],
            blocked=[_at(9, 9)],
        ),
        profile=LOCAL_V1_COARSE,
    )
    space = observation_space(LOCAL_V1_COARSE)

    assert encoded["grid"].shape == (len(RESOURCES) + 2, 33, 33)
    assert space.contains(encoded)
    coal_row, coal_col = _cell(-6, 4, LOCAL_V1_COARSE)
    assert encoded["grid"][RESOURCES.index("coal"), coal_row, coal_col] == 1.0


def test_the_default_profile_is_unchanged_by_the_cell_size_addition():
    """`LOCAL_V1` must stay `cell_size = 1`, or every existing checkpoint is
    reading a different grid than it was trained on."""
    assert LOCAL_V1.cell_size == 1
    assert np.array_equal(
        encode(_observation(tiles=[{"name": "stone", "p": _at(-4, 11), "amount": 77}]))["grid"],
        encode(
            _observation(tiles=[{"name": "stone", "p": _at(-4, 11), "amount": 77}]),
            profile=ObservationProfile(name="local-v1", version=1, radius=32, cell_size=1),
        )["grid"],
    )


# ----------------------------------------------------- cross-language radius


def test_python_and_lua_agree_on_the_sensor_radius():
    """The one number both sides declare independently and nothing checks.

    If Lua's radius shrinks below Python's, the wire simply carries fewer
    tiles and the grid keeps its shape; if it grows, ``to_cell`` drops the
    surplus. Either way ``observation_space().contains()`` still passes and the
    policy trains on a quietly truncated view -- there is no error to trace.
    """
    source = (mod_source_dir() / "factoriorl" / "profiles.lua").read_text(encoding="utf-8")
    block = re.search(r'\["local-v1"\] = \{(.*?)\n  \},', source, re.DOTALL)
    assert block, "the local-v1 observation profile was not found in profiles.lua"
    match = re.search(r"^\s*radius = (\d+),", block.group(1), re.MULTILINE)
    assert match, "radius not found in the local-v1 profile; the regex is wrong"
    assert int(match.group(1)) == LOCAL_V1.radius, (
        "profiles.lua and encoders.LOCAL_V1 disagree on the sensor radius"
    )


def test_the_python_profile_name_matches_the_lua_one():
    source = (mod_source_dir() / "factoriorl" / "profiles.lua").read_text(encoding="utf-8")
    assert f'["{LOCAL_V1.name}"]' in source
    assert re.search(rf"^\s*version = {LOCAL_V1.version},", source, re.MULTILINE)
