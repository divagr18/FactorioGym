"""Observation-profile parity and per-episode independence of the character.

Requires a real Factorio worker (engine marker). Both tests here exist because
they caught a live defect, and both pin a property that regresses silently:

* ``local-v2`` is the default observation profile, and it is only sound because
  it is information-preserving. Nothing in the encoder would complain if it
  stopped being so -- a thinner wire payload still decodes into a well-formed
  tensor, just a different one -- so the equality has to be measured against
  ``local-v1`` on a real worker.
* Reset restored the character's position, inventory, walking, mining and
  crafting queue but never its *facing*, so an episode began pointing wherever
  the previous episode's last move left it.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from factoriorl import tasks as task_registry
from factoriorl.env import FactorioEnv
from factoriorl.seeding import Branch, SeedPlan

pytestmark = pytest.mark.engine

#: The six introductory families. Parametrised by name so a failure says which
#: family diverged rather than only that some family did.
FAMILIES = (
    "navigate",
    "deliver",
    "mine_smelt",
    "supply_furnace",
    "repair_belt",
    "restore_power",
)

#: Long enough for the scene to change under the agent (entities appear in and
#: leave the sensor radius, memory fills up, the entity cap starts binding),
#: short enough that twelve episodes of it stay inside a test run.
PARITY_STEPS = 20


@pytest.fixture(scope="module")
def paced_session(module_session):
    """The module's worker, paced up.

    Pacing only: Factorio is tick-based and the simulation is identical at any
    speed, which is what lets these tests compare observations at all. At the
    default 60 UPS every 30-tick decision costs a hard 500 ms of wall clock,
    which would put the parity sweep alone at over two minutes of pure waiting.
    """
    module_session.configure(speed=60.0)
    return module_session


def _with_profile(task: task_registry.RegisteredTask, profile: str):
    """The same registered task, asking the worker for a different profile.

    The profile lives on the frozen ``TaskSpec``, and ``FactorioEnv`` reads it
    off ``task.spec`` on every reset, so both the spec and the task wrapping it
    have to be replaced.
    """
    return dataclasses.replace(
        task, spec=dataclasses.replace(task.spec, observation_profile=profile)
    )


def _facing(env: FactorioEnv):
    """The character's facing as the policy would see it.

    Read off the raw observation rather than the encoded tensor because that is
    where the value is unambiguous: ``encoders.encode`` divides it by 16 and
    buries it at ``self[3]``.
    """
    return (env._observation.get("character") or {}).get("direction")


def _play(env: FactorioEnv, actions: list[int] | None = None, seed: int = 0):
    """Reset, then play a short episode; return the frames, wire sizes, actions.

    With ``actions`` given the run is a replay of an earlier one, which is how
    two profiles are made to see the same episode: they share a session, so
    they cannot be stepped side by side (either env's reset would invalidate
    the other's episode), and they are run one after the other instead.
    """
    rng = np.random.default_rng(seed)
    frames: list[dict] = []
    sizes: list[int] = []
    played: list[int] = []

    observation, _info = env.reset()
    frames.append(observation)
    sizes.append(len(json.dumps(env._observation)))

    budget = range(PARITY_STEPS) if actions is None else range(len(actions))
    for index in budget:
        if actions is None:
            legal = np.flatnonzero(env.action_masks())
            action = int(rng.choice(legal))
        else:
            action = actions[index]
        played.append(action)
        observation, _reward, terminated, truncated, _info = env.step(action)
        frames.append(observation)
        sizes.append(len(json.dumps(env._observation)))
        if terminated or truncated:
            break
    return frames, sizes, played


@pytest.mark.parametrize("task_id", FAMILIES)
def test_observation_profiles_decode_identically(paced_session, task_id):
    """``local-v2`` must decode to byte-identical tensors, and cost less.

    ``local-v2`` is the default profile: it drops the blocks nothing on the
    Python side reads and caps the entity list at 48 after a distance sort,
    where ``local-v1`` ships 256. Since ``encoders.encode`` keeps only the 32
    nearest entities, an entity at rank 49 already has 48 entities at least as
    close and can never reach the encoder -- so the saving is supposed to be
    free rather than a trade. That argument is about ranks and caps in two
    different files (``mod/factoriorl/profiles.lua`` and
    ``src/factoriorl/encoders.py``); nothing enforces it, and a future change to
    either cap, to the sweep ordering, or to which keys a slim profile omits
    would quietly start feeding the policy a different tensor while every test
    that only checks shapes and spaces kept passing. This measures it instead.
    """
    task = task_registry.get(task_id)
    plan = SeedPlan(master=20260907, run_id=f"profile-parity-{task_id}")

    envs = {}
    for profile in ("local-v1", "local-v2"):
        env = FactorioEnv(
            _with_profile(task, profile),
            paced_session,
            plan,
            branch=Branch.TRAIN,
            split="train",
        )
        # Both envs must draw the same scene from the same seed stream. The
        # episode index is what selects the layout family and the generator
        # RNG, so it is pinned here rather than trusted to the constructor.
        env._episode_index = -1
        envs[profile] = env

    v1_frames, v1_sizes, actions = _play(envs["local-v1"])
    v2_frames, v2_sizes, _replayed = _play(envs["local-v2"], actions=actions)

    assert len(v1_frames) > 1, f"{task_id}: the episode ended before it started"
    assert len(v2_frames) == len(v1_frames), (
        f"{task_id}: local-v2 ran {len(v2_frames)} frames against local-v1's "
        f"{len(v1_frames)}; the same action sequence ended the episode at a "
        "different point, so the two profiles are not seeing the same world"
    )

    for index, (v1, v2) in enumerate(zip(v1_frames, v2_frames, strict=True)):
        assert v1.keys() == v2.keys(), f"{task_id}: frame {index} has different keys"
        for key in v1:
            assert np.array_equal(v1[key], v2[key]), (
                f"{task_id}: frame {index} differs in {key!r} "
                f"({int(np.sum(v1[key] != v2[key]))} of {v1[key].size} values)"
            )

    # The whole point of the profile. Measured reductions were 2.6x to 5.6x
    # depending on the family, but the ratio is a property of the scene and not
    # worth pinning; that v2 is smaller at all is the invariant -- a slim
    # profile that stopped being slim would otherwise pass this file silently.
    assert sum(v2_sizes) < sum(v1_sizes), (
        f"{task_id}: local-v2 sent {sum(v2_sizes)} bytes against local-v1's "
        f"{sum(v1_sizes)}; the profile is no longer saving anything"
    )


def _walk_until_the_facing_changes(env: FactorioEnv, initial, attempts: int = 5):
    """Drive the character with one explicit move until it faces elsewhere.

    An explicit catalog move rather than a masked random action: this test is
    about a single scalar, and a random walk that happened to pick `wait` would
    make it pass or fail on the draw.
    """
    keys = env.catalog.keys()
    assert "move_east" in keys, f"catalog {env.catalog.name} has no move_east: {keys}"
    action = keys.index("move_east")
    facing = initial
    for _ in range(attempts):
        _obs, _reward, terminated, truncated, _info = env.step(action)
        facing = _facing(env)
        if facing != initial or terminated or truncated:
            break
    return facing


def test_reset_restores_the_characters_facing(paced_session):
    """A new episode must start facing where the first one did.

    Reset restored position, inventory, walking, mining and the crafting queue
    but never the facing, so an episode began pointing wherever the previous
    episode's last move had left it. That reaches the policy directly --
    ``encoders.encode`` puts ``direction / 16`` into the self vector -- so
    episodes were not independent, and nothing about the observation looked
    wrong. The fix is to recreate the character at reset
    (``world.recreate_character``): assigning ``ch.direction`` is applied by the
    engine only on the following tick and the world is paused between
    decisions, so the reset observation would still have reported the stale
    value.
    """
    env = FactorioEnv(
        task_registry.get("navigate"),
        paced_session,
        SeedPlan(master=20260907, run_id="facing-single"),
        branch=Branch.TRAIN,
        split="train",
    )
    env._episode_index = -1

    env.reset()
    initial = _facing(env)

    facing = _walk_until_the_facing_changes(env, initial)
    # Without this the test could pass vacuously: if movement stopped changing
    # the facing at all, "the facing was restored" would be trivially true.
    assert facing != initial, (
        f"walking east never moved the facing off {initial!r}; this test can no "
        "longer observe the leak it was written for"
    )

    env.reset()
    assert _facing(env) == initial, (
        f"episode 1 started facing {_facing(env)!r} where episode 0 started at "
        f"{initial!r}: the previous episode's last move leaked across reset"
    )


def test_facing_does_not_leak_between_separate_envs(paced_session):
    """Two envs on one session must not inherit each other's facing.

    This is the shape in which the leak was originally found: a session is
    reused across environments (the gates and the trainer both do it), so one
    env's final heading became the next env's opening observation. Testing it
    with two envs rather than two resets of one env also covers the case where
    the restoring state is held on the Python side rather than in the mod --
    it is not, and this asserts that.
    """
    plan = SeedPlan(master=20260907, run_id="facing-cross-env")
    first, second = (
        FactorioEnv(
            task_registry.get("navigate"),
            paced_session,
            plan,
            branch=Branch.TRAIN,
            split="train",
        )
        for _ in range(2)
    )
    first._episode_index = -1
    second._episode_index = -1

    first.reset()
    initial = _facing(first)
    facing = _walk_until_the_facing_changes(first, initial)
    assert facing != initial, (
        f"walking east never moved the facing off {initial!r}; this test can no "
        "longer observe the leak it was written for"
    )

    second.reset()
    assert _facing(second) == initial, (
        f"a fresh env opened facing {_facing(second)!r} where the first env "
        f"opened at {initial!r}: facing leaked across environments on one session"
    )
