"""T1's action-locked verifier is a task contract, not a reporting convention."""

from __future__ import annotations

import pytest

from factoriorl.env import FactorioEnv
from factoriorl.rewards import RewardAccountant
from factoriorl.tasks import TaskConfigError, get
from factoriorl.tasks.spec import Predicate, PredicateKind


def test_verified_machine_output_requires_the_evaluator_result():
    predicate = Predicate(
        PredicateKind.VERIFIED_MACHINE_OUTPUT,
        item="iron-plate",
        at_least=10,
    )
    assert not predicate.evaluate({}, {})
    assert not predicate.evaluate({}, {"verification": {"iron-plate": 9}})
    assert predicate.evaluate({}, {"verification": {"iron-plate": 10}})


def test_verification_uses_machine_delta_and_is_one_shot():
    """Handcrafting / prior output cannot be added to the verifier score."""
    spec = get("construct_smelting_line").spec
    env = object.__new__(FactorioEnv)
    env.spec_ = spec
    env._verification = None
    env._observation = {"tick": 1200}
    env._truth = {
        "machine_produced": {"iron-plate": 40},
        # A large ordinary production count must be ignored by the verifier.
        "produced": {"iron-plate": 999},
    }
    env.accountant = RewardAccountant(spec.rewards)
    env.accountant.reset(env._observation, env._truth)

    def advance(ticks: int) -> int:
        assert ticks == 3600
        env._observation = {"tick": 4800}
        env._truth = {
            "machine_produced": {"iron-plate": 50},
            "produced": {"iron-plate": 5000},
        }
        return ticks

    env.advance = advance
    result = env.run_verification()

    assert result["machine_output"] == 10
    assert result["success"] is True
    assert result["reward"] == 1.0

    with pytest.raises(TaskConfigError, match="already run"):
        env.run_verification()


def test_task_reserves_the_final_minute_for_verification():
    spec = get("construct_smelting_line").spec
    assert spec.verification is not None
    assert spec.max_game_ticks - spec.verification.ticks == 18000
    assert spec.verification.item == "iron-plate"
    assert spec.verification.target == 10


def test_verification_returns_normalized_partial_credit():
    spec = get("construct_smelting_line").spec
    env = object.__new__(FactorioEnv)
    env.spec_ = spec
    env._verification = None
    env._observation = {"tick": 0}
    env._truth = {"machine_produced": {"iron-plate": 0}}
    env.accountant = RewardAccountant(spec.rewards)
    env.accountant.reset(env._observation, env._truth)

    def advance(ticks: int) -> int:
        env._observation = {"tick": ticks}
        env._truth = {"machine_produced": {"iron-plate": 4}}
        return ticks

    env.advance = advance
    result = env.run_verification()
    assert result["success"] is False
    assert result["reward"] == 0.4
