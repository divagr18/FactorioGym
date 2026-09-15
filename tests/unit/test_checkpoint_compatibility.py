"""A checkpoint can load and still be unusable, and it should say which.

SB3 rebuilds a policy's input layers from the observation space pickled
*inside* the checkpoint, so `MaskablePPO.load` succeeds against any
environment. The mismatch surfaces much later, in `obs_to_tensor`, as

    ValueError: cannot reshape array of size 14 into shape (10)

which names neither the key nor the reason. That is not hypothetical: `ITEMS`
grew from 10 to 14 when R4.3 added two task families, so every checkpoint from
before then expects a 10-wide `inventory` while the current environment offers
14. Four archived `restore_power` checkpoints hit exactly this.

`architecture_signature` does not help, and it is worth being precise about
why: it digests the policy's own parameter names and shapes, which are
self-consistent in a checkpoint that loads. What differs is the space the
*environment* will present, which the checkpoint never sees.

So R6's gate -- "another user can evaluate a provided checkpoint" -- needs the
comparison this makes, and needs it before stepping rather than after.
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from factoriorl.learn.policy import observation_compatibility


class Fake:
    def __init__(self, observation_space, action_space=None):
        self.observation_space = observation_space
        self.action_space = action_space


def dict_space(**shapes) -> spaces.Dict:
    return spaces.Dict(
        {key: spaces.Box(-1.0, 1.0, shape, np.float32) for key, shape in shapes.items()}
    )


def test_the_real_incompatibility_is_named_by_key():
    """The `ITEMS` 10 to 14 case, which is what motivated this."""
    checkpoint = Fake(dict_space(inventory=(10,), goal=(12,)), spaces.Discrete(23))
    env = Fake(dict_space(inventory=(14,), goal=(12,)), spaces.Discrete(23))
    report = observation_compatibility(checkpoint, env)
    assert report["compatible"] is False
    assert report["observation_differences"] == {
        "inventory": {"checkpoint": (10,), "environment": (14,)}
    }
    assert "item or entity catalog" in report["hint"]


def test_a_matching_pair_is_compatible_and_carries_no_hint():
    """The hint must not cry wolf: a usable checkpoint gets a clean report."""
    space = dict_space(inventory=(14,), goal=(12,))
    report = observation_compatibility(
        Fake(space, spaces.Discrete(13)), Fake(space, spaces.Discrete(13))
    )
    assert report["compatible"] is True
    assert report["observation_differences"] == {}
    assert report["hint"] is None
    # JSON-safe: `Discrete.n` is a numpy integer, and json.dumps refuses
    # numpy scalars -- so the report would break the CLI that emits it.
    assert type(report["compatible"]) is bool
    assert all(type(v) is int for v in report["actions"].values())


def test_an_action_count_mismatch_is_caught_and_attributed_to_skills():
    """13 primitive actions against 23 with skills. The commonest way to point
    a checkpoint at the wrong environment, and silently truncating a
    skill-trained policy's catalog would score a different agent."""
    space = dict_space(inventory=(14,))
    report = observation_compatibility(
        Fake(space, spaces.Discrete(23)), Fake(space, spaces.Discrete(13))
    )
    assert report["compatible"] is False
    assert report["actions"] == {"checkpoint": 23, "environment": 13}
    assert "skills flag" in report["hint"]


def test_a_key_present_on_only_one_side_is_a_difference():
    """An added or removed observation key is as fatal as a resized one, and
    reporting only shared keys would call it compatible."""
    report = observation_compatibility(
        Fake(dict_space(inventory=(14,)), spaces.Discrete(13)),
        Fake(dict_space(inventory=(14,), extra=(4,)), spaces.Discrete(13)),
    )
    assert report["compatible"] is False
    assert report["observation_differences"]["extra"] == {
        "checkpoint": None,
        "environment": (4,),
    }


def test_every_differing_key_is_reported_not_just_the_first():
    report = observation_compatibility(
        Fake(dict_space(inventory=(10,), goal=(10,)), spaces.Discrete(13)),
        Fake(dict_space(inventory=(14,), goal=(12,)), spaces.Discrete(13)),
    )
    assert set(report["observation_differences"]) == {"inventory", "goal"}


def test_a_missing_space_is_reported_as_not_comparable():
    """Refusing to compare is not the same as reporting compatible, and a
    caller that treated a missing space as agreement would step anyway."""
    report = observation_compatibility(Fake(None), Fake(dict_space(inventory=(14,))))
    assert report["comparable"] is False
    assert "no observation space" in report["why"]
    assert "compatible" not in report


def test_a_plain_box_space_is_handled_rather_than_crashing():
    """Not every policy takes a Dict observation, and a comparison that only
    understood Dict would raise on the simple case."""
    report = observation_compatibility(
        Fake(spaces.Box(-1.0, 1.0, (8,), np.float32), spaces.Discrete(4)),
        Fake(spaces.Box(-1.0, 1.0, (8,), np.float32), spaces.Discrete(4)),
    )
    assert report["compatible"] is True


def test_two_different_factorized_action_spaces_are_not_compatible():
    """`.n` is `None` on every `MultiDiscrete`, so the check compared `None` with
    `None` and called any two parameterized policies compatible."""
    from gymnasium import spaces

    from factoriorl import encoders

    observation = encoders.observation_space()
    model = Fake(observation, spaces.MultiDiscrete([22, 33, 122, 5, 15, 4]))
    env = Fake(observation, spaces.MultiDiscrete([23, 33, 122, 5, 15, 4]))
    result = observation_compatibility(model, env)
    assert result["comparable"] is True
    assert result["compatible"] is False
    assert result["actions"] == {
        "checkpoint": [22, 33, 122, 5, 15, 4],
        "environment": [23, 33, 122, 5, 15, 4],
    }


def test_matching_factorized_action_spaces_are_compatible():
    from gymnasium import spaces

    from factoriorl import encoders

    observation = encoders.observation_space()
    nvec = [22, 33, 122, 5, 15, 4]
    result = observation_compatibility(
        Fake(observation, spaces.MultiDiscrete(nvec)), Fake(observation, spaces.MultiDiscrete(nvec))
    )
    assert result["compatible"] is True
