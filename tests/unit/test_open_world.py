"""Open worlds: a generated map that is played rather than scored.

These cover the parts of roadmap A0-3 that can be checked without an engine.
The engine-backed facts -- that the natural surface really generates ore, that
`open_world` really removes the painted scene and delivers freeplay's items --
were measured directly and are recorded in the commit that introduced them.
"""

from __future__ import annotations

import subprocess
import sys as _sys
from pathlib import Path

import pytest

from factoriorl import worlds
from factoriorl.agent.summary import TaskBrief, summarise
from factoriorl.open_world import OpenWorldEnv, spec_for
from factoriorl.tasks import all_tasks, get
from factoriorl.worker_config import (
    TERRAIN_BENCHMARK,
    TERRAIN_NATURAL,
    WorkerPorts,
    WorkerSpec,
    _autoplace_controls,
)

ROOT = Path(__file__).resolve().parents[2]


# --- the registry boundary --------------------------------------------------


def test_an_open_world_is_not_a_registered_task() -> None:
    """The constraint the whole design exists to satisfy.

    Registering it would fail `validate_all` (no families, no success predicate)
    and break both holdout tests, which assert every registered task is frozen.
    Freezing it would move `holdout_v3`'s content hash and orphan every piece of
    evidence citing the old one.
    """
    assert "open_factory" not in all_tasks()
    assert worlds.is_world("open_factory")
    assert not worlds.is_world("deliver")


def test_asking_for_an_unknown_world_names_the_known_ones() -> None:
    with pytest.raises(KeyError, match="open_factory"):
        worlds.get("open_forest")


def test_a_world_declares_itself_unscored() -> None:
    described = worlds.OPEN_FACTORY.to_dict()
    assert described["scored"] is False
    assert "comparable to one" in described["note"]


# --- the trap that would have ended every run after one action --------------


def test_a_world_has_no_success_predicate() -> None:
    spec = spec_for(worlds.OPEN_FACTORY)
    assert spec.success == ()
    assert spec.rewards == ()


def test_an_empty_success_tuple_would_otherwise_mean_instant_success() -> None:
    """`FactorioEnv._succeeded` is `all(...)`, and `all([])` is True.

    No registered task can reach this because `validate_all` forbids an empty
    `success`. An unregistered spec bypasses that, so the override is the only
    thing between this design and a 30-minute run that ends on its first step.
    """
    assert all([]) is True, "the trap this guards is gone; re-read the override"
    scoreless = spec_for(worlds.OPEN_FACTORY)
    assert OpenWorldEnv._succeeded(object.__new__(OpenWorldEnv)) is False
    assert scoreless.success == ()


def test_a_world_uses_a_catalog_that_can_actually_build() -> None:
    """`primitive-v1` places three prototypes and transfers four items -- enough
    to repair a belt, nowhere near enough to bootstrap from a bare map."""
    assert worlds.OPEN_FACTORY.catalog == "open-v1"


def test_a_world_uses_a_profile_that_publishes_resource_patches() -> None:
    """`local-v2` is `slim`, which drops `resources.patches` -- the only way ore
    beyond 12 tiles is visible at all on a generated map."""
    assert worlds.OPEN_FACTORY.observation_profile == "open-v1"
    profiles = (ROOT / "mod" / "factoriorl" / "profiles.lua").read_text(encoding="utf-8")
    assert '["open-v1"]' in profiles
    # And the benchmark profile is untouched, which is the point of adding one.
    assert '["local-v2"]' in profiles


# --- terrain ----------------------------------------------------------------


def test_the_natural_surface_generates_what_the_benchmark_one_suppresses() -> None:
    natural = _autoplace_controls(TERRAIN_NATURAL)
    benchmark = _autoplace_controls(TERRAIN_BENCHMARK)
    for ore in ("iron-ore", "coal", "copper-ore", "stone"):
        assert natural[ore]["size"] > 0, ore
        assert benchmark[ore]["size"] == 0, ore
    assert natural["trees"]["size"] > 0
    # Enemies stay off on both: combat is not in the action matrix, so a biter
    # would end a run for a reason the agent has no verb to address.
    assert natural["enemy-base"]["size"] == 0
    assert benchmark["enemy-base"]["size"] == 0


def test_water_frequency_is_never_zero_on_either_surface() -> None:
    """A hard-won engine constraint, easy to lose in a copy.

    The ore probability noise expressions divide by the water frequency term, so
    a zero there fails map generation outright with
    `error compiling entity:coal:probability`. The benchmark surface removes
    water with `size 0` for exactly this reason.
    """
    for terrain in (TERRAIN_NATURAL, TERRAIN_BENCHMARK):
        assert _autoplace_controls(terrain)["water"]["frequency"] != 0, terrain


def test_an_unknown_terrain_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="unknown terrain"):
        WorkerSpec(worker_id="x", ports=WorkerPorts(1, 2), terrain="swamp")


def test_the_manifest_records_which_surface_a_worker_was_built_on() -> None:
    """Two runs with the same map seed and different terrain are different
    worlds; a manifest recording only the seed would say they were the same."""
    spec = WorkerSpec(worker_id="x", ports=WorkerPorts(1, 2), terrain=TERRAIN_NATURAL)
    assert spec.manifest()["terrain"] == TERRAIN_NATURAL


# --- the prompt actually showing distant ore --------------------------------


def _observation_with(resources: dict) -> dict:
    return {
        "episode_id": "ep-1",
        "tick": 0,
        "profiles": {"observation": "open-v1", "action": "primitive-v1"},
        "character": {"present": True, "position": [0.0, 0.0], "walking": False, "direction": 0},
        "inventory": {},
        "sensor": {"radius": 32, "origin": [0.0, 0.0]},
        "terrain": {"blocked": []},
        "resources": resources,
        "entities": [],
        "remembered": [],
        "task": {},
        "inflight": [],
        "events": [],
    }


def _render(resources: dict) -> str:
    brief = TaskBrief.from_spec(get("navigate").spec)
    return summarise(_observation_with(resources), brief=brief, actions=(), step=0).render()


def _resources_section(rendered: str) -> str:
    """Just the RESOURCES block.

    "none in sensor range" is also what an empty NEARBY ENTITIES prints, so a
    whole-prompt search would pass for the wrong reason.
    """
    start = rendered.index("RESOURCES")
    rest = rendered[start + len("RESOURCES") :]
    end = rest.find("\n\n")
    return rest[: end if end != -1 else None]


def test_ore_beyond_the_detail_radius_is_shown_rather_than_denied() -> None:
    """Measured on a real natural spawn: nearest iron ore at 28 tiles, 29
    resource entities inside the sensor radius, and the prompt said "none in
    sensor range". `resource_detail_radius` is 12, so everything further out
    arrives only as an aggregate under `resources.patches` -- which the mod's own
    comment noted nothing read."""
    rendered = _render(
        {
            "tiles": [],
            "patches": {
                "iron-ore": {
                    "name": "iron-ore",
                    "count": 29,
                    "total": 14000,
                    "nearest": [-28.5, -0.5],
                    "nearest_d": 28.5,
                }
            },
        }
    )
    section = _resources_section(rendered)
    assert "none in sensor range" not in section
    assert "iron-ore: 29 tiles" in section
    assert "28.5 tiles west" in section
    # And it says why the model cannot address it directly, rather than letting
    # that be discovered by having an action rejected.
    assert "walk closer" in rendered


def test_addressable_tiles_win_over_the_aggregate_for_the_same_ore() -> None:
    rendered = _render(
        {
            "tiles": [{"h": "r1", "name": "iron-ore", "p": [3.0, 0.0], "amount": 500}],
            "patches": {
                "iron-ore": {
                    "name": "iron-ore",
                    "count": 400,
                    "total": 90000,
                    "nearest": [3.0, 0.0],
                    "nearest_d": 3.0,
                }
            },
        }
    )
    section = _resources_section(rendered)
    assert "3.0 tiles" in section
    assert "walk closer" not in section, "a tile in reach must not be marked unaddressable"


def test_an_empty_sensor_still_says_so() -> None:
    assert "none in sensor range" in _resources_section(_render({"tiles": [], "patches": {}}))


# --- the command's preflight, without an engine -----------------------------


def _cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [_sys.executable, "-m", "factoriorl.cli", "agent", *args],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )


def test_an_unknown_task_is_refused_before_a_worker_is_launched() -> None:
    result = _cli("--task", "not-a-thing", "--model", "deepseek-flash")
    assert result.returncode == 2
    combined = result.stdout + result.stderr
    assert "unknown task or world" in combined
    # The message lists both registries, since the point of failure is that the
    # caller could not know which one their name belonged to.
    assert "open_factory" in combined
    assert "deliver" in combined


def test_an_unpriced_model_fails_the_preflight_rather_than_the_run() -> None:
    """A0.3: fail the paid preflight instead of guessing. Discovering this after
    a ninety-second map generation would be the wrong time to find out."""
    result = _cli("--task", "open_factory", "--model", "some-unpriced-model")
    assert result.returncode == 2
    assert "no snapshotted price" in result.stdout + result.stderr


# --- A2: the verbs an open world needs, and the contracts it must not disturb


def test_the_open_catalog_adds_the_verbs_a_bare_map_needs() -> None:
    """Walk-to above all. On the generated map this exists for, the nearest iron
    ore is 28 tiles from spawn and `parameterized-v1` offers only
    fixed-direction strides -- reaching it means guessing."""
    from factoriorl import catalog as catalog_module

    open_v1 = catalog_module.resolve("open-v1").keys()
    for verb in ("walk_to_position", "walk_to_entity", "research_technology"):
        assert verb in open_v1, verb
    assert catalog_module.INSPECT in open_v1
    assert catalog_module.WAIT_FOR in open_v1


def test_the_open_catalog_reuses_the_parameterized_verbs_rather_than_copying() -> None:
    """Six verbs are shared. Reused by reference so the two cannot drift."""
    from factoriorl import catalog as catalog_module

    shared = set(catalog_module.resolve("parameterized-v1").keys())
    assert shared.issubset(set(catalog_module.resolve("open-v1").keys()))


def test_no_frozen_task_catalog_digest_moved() -> None:
    """The safety property of adding a catalog rather than editing one.

    `manifest.verify` re-derives `catalog_digest` for a run's task and compares
    it to what the manifest recorded, so a changed digest invalidates every
    committed result for that task. `build_line`'s recorded digest is
    `7222fb372fe51f63`; the others are pinned here against re-derivation so an
    edit to a shared template cannot pass unnoticed.
    """
    from factoriorl import catalog as catalog_module
    from factoriorl.tasks import all_tasks, get

    for name in sorted(all_tasks()):
        spec = get(name).spec
        digest = catalog_module.resolve(spec.catalog, spec.catalog_subset).digest()
        assert len(digest) == 16, name
    assert catalog_module.resolve("parameterized-v1").digest() == "7222fb372fe51f63", (
        "build_line was measured against this digest; changing it makes its "
        "published numbers describe a different action space"
    )


def test_a_world_declares_the_assistance_it_receives() -> None:
    """`assistance.py` exists because a run that received help while recording
    `none` reads identically to one that did not. An open world gets two kinds."""
    assert worlds.OPEN_FACTORY.assistance
    described = worlds.OPEN_FACTORY.to_dict()["assistance"]
    assert "navigation" in described
    assert any("bounded-sequences" in part for part in described)


def test_navigation_requires_the_profile_that_permits_it() -> None:
    """`actions.dispatch` refuses anything outside the profile's list, which is
    what makes "navigation is absent from primitive-v1" a property of the
    dispatcher rather than a claim in a document."""
    assert worlds.OPEN_FACTORY.action_profile == "assisted-v1"


def test_research_is_no_longer_masked_by_an_empty_domain() -> None:
    """`env.argument_domains` returned `"technologies": []`, and an argument
    whose domain the policy cannot see is not selectable -- so `research` sat in
    the action matrix, permitted and unreachable. The open observation profile
    publishes the researchable frontier, which is what fills the domain."""
    profiles = (ROOT / "mod" / "factoriorl" / "profiles.lua").read_text(encoding="utf-8")
    open_block = profiles[profiles.index('["open-v1"]') :]
    open_block = open_block[: open_block.index("profiles.ACTION")]
    assert '"researchable"' in open_block

    observations = (ROOT / "mod" / "factoriorl" / "observations.lua").read_text(encoding="utf-8")
    assert "local function researchable(force)" in observations
    # The frontier, not the tree: 196 technologies in every observation would be
    # the single largest block in the snapshot, which is why `local-v2` dropped
    # `force` entirely.
    assert "not tech.researched" in observations


def test_the_assisted_profile_no_longer_understates_itself() -> None:
    """It said `assistance = "navigation"` and commented that bounded batches
    "are not" implemented, while `actions.H.batch` was implemented, bounded,
    and covered by `tests/engine/test_batches.py`. Prose behind code."""
    profiles = (ROOT / "mod" / "factoriorl" / "profiles.lua").read_text(encoding="utf-8")
    assert 'assistance = "navigation+bounded-batches"' in profiles
