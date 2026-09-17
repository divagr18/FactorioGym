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
        "machine_produced": {"iron-plate": 40, "iron-ore": 60},
        # A large ordinary production count must be ignored by the verifier.
        "produced": {"iron-plate": 999},
    }
    env.accountant = RewardAccountant(spec.rewards)
    env.accountant.reset(env._observation, env._truth)

    def advance(ticks: int) -> int:
        assert ticks == 3600
        env._observation = {"tick": 4800}
        env._truth = {
            # A drill-fed line: machines mined the ore the plates came from.
            "machine_produced": {"iron-plate": 50, "iron-ore": 72},
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
    env._truth = {"machine_produced": {"iron-plate": 0, "iron-ore": 0}}
    env.accountant = RewardAccountant(spec.rewards)
    env.accountant.reset(env._observation, env._truth)

    def advance(ticks: int) -> int:
        env._observation = {"tick": ticks}
        env._truth = {"machine_produced": {"iron-plate": 4, "iron-ore": 5}}
        return ticks

    env.advance = advance
    result = env.run_verification()
    assert result["success"] is False
    assert result["reward"] == 0.4


def _verifier(before: dict, after: dict) -> FactorioEnv:
    """A `FactorioEnv` reduced to what `run_verification` reads."""
    spec = get("construct_smelting_line").spec
    env = object.__new__(FactorioEnv)
    env.spec_ = spec
    env._verification = None
    env._observation = {"tick": 18000}
    env._truth = {"machine_produced": dict(before)}
    env.accountant = RewardAccountant(spec.rewards)
    env.accountant.reset(env._observation, env._truth)

    def advance(ticks: int) -> int:
        env._observation = {"tick": 18000 + ticks}
        env._truth = {"machine_produced": dict(after)}
        return ticks

    env.advance = advance
    return env


class TestAHandFedFurnaceIsNotAProductionLine:
    """1.0.0 paid in full for a furnace loaded by hand.

    `machine_produced` is `produced - handcrafted - mined`. That subtracts the
    hand-mined *ore*, but the plates smelted from it are neither handcrafted nor
    mined, so they count as machine output. A furnace, some coal and ten
    hand-mined ore scored 1.0 with no drill on the map --
    `docs/evidence/a4-production.json` already records five such "machine"
    plates from thirteen hand-mined ore. An RL policy trained on this task would
    find that before it found a drill.
    """

    def test_plates_with_no_machine_mined_ore_score_nothing(self):
        # Eighteen plates out of a hand-loaded furnace; no drill mined anything.
        env = _verifier(before={"iron-plate": 0}, after={"iron-plate": 18})
        result = env.run_verification()
        assert result["uncapped_output"] == 18
        assert result["machine_source"] == 0
        assert result["machine_output"] == 0
        assert result["success"] is False
        assert result["reward"] == 0.0

    def test_a_drill_fed_line_still_clears_the_target(self):
        """The measured reference line is drill-limited: about fifteen plates a
        minute from about fifteen drill-mined ore. The cap must not cost it the
        task."""
        env = _verifier(
            before={"iron-plate": 3, "iron-ore": 5},
            after={"iron-plate": 17, "iron-ore": 20},
        )
        result = env.run_verification()
        assert result["machine_output"] == 14
        assert result["success"] is True
        assert result["reward"] == 1.0

    def test_output_is_capped_by_ore_machines_supplied_in_the_window(self):
        """Ore buffered before the window, or hand-loaded on top of a real line,
        cannot lift the count past what the drills fed during it."""
        env = _verifier(
            before={"iron-plate": 0, "iron-ore": 0},
            after={"iron-plate": 18, "iron-ore": 7},
        )
        result = env.run_verification()
        assert result["machine_output"] == 7
        assert result["success"] is False

    def test_the_task_declares_its_source_and_its_new_version(self):
        spec = get("construct_smelting_line").spec
        assert spec.verification.source == "iron-ore"
        assert spec.version == "1.1.1"
        assert spec.verification.to_dict()["source"] == "iron-ore"

    def test_a_verification_without_a_source_keeps_its_old_serialized_shape(self):
        from factoriorl.tasks.spec import VerificationSpec

        assert VerificationSpec("iron-plate", 10, 3600).to_dict() == {
            "item": "iron-plate",
            "target": 10,
            "ticks": 3600,
        }


def _stepping_env(*, tick: int, steps: int, verified: dict):
    """Exactly the attributes `FactorioEnv.step_payload` reads, and no more."""
    import numpy as np

    from factoriorl import encoders

    spec = get("construct_smelting_line").spec
    env = object.__new__(FactorioEnv)
    env.spec_ = spec
    env._verification = None
    env._steps = steps
    env._observation = {"tick": tick}
    env._truth = {}
    env._family = None
    env.accountant = RewardAccountant(spec.rewards)
    env.accountant.reset(env._observation, env._truth)

    class _Response:
        result = {"action": {"status": "completed"}}

    class _Timed:
        response = _Response()

    class _Session:
        def step(self, payload, ticks):
            return _Timed()

    class _Metrics:
        def report(self):
            return {}

    env.session = _Session()
    env.metrics = _Metrics()
    env._adopt = lambda result: None
    env._failed = lambda: False
    env._succeeded = lambda: bool(
        (env._truth.get("verification") or {}).get("iron-plate", 0) >= spec.verification.target
    )
    env._goal_vector = lambda: np.zeros(encoders.GOAL_FEATURES, dtype=np.float32)
    env.action_masks = lambda: np.ones(1, dtype=bool)
    calls: list[int] = []

    def run_verification():
        calls.append(env._steps)
        env._verification = dict(verified)
        env._truth = {"verification": {"iron-plate": verified["machine_output"]}}
        return dict(env._verification)

    env.run_verification = run_verification
    return env, calls


class TestTheVerifierRunsWhenTheEpisodeRunsOut:
    """`run_verification` was only ever called by a driver: the agentic bridge's
    `finish` and the evaluator-only `solve`. An RL episode on this task ran out of
    budget, truncated, and was never measured, so its reward was zero whatever it
    built. A `finish` verb would have changed the frozen `parameterized-v1`
    digest, so the verifier runs at truncation instead."""

    def test_the_last_construction_step_is_verified_and_terminal(self, monkeypatch):
        from factoriorl import encoders

        monkeypatch.setattr(encoders, "encode", lambda observation, goal: {})
        env, calls = _stepping_env(
            tick=18000,
            steps=10,
            verified={"success": True, "reward": 1.0, "machine_output": 12},
        )
        _, reward, terminated, truncated, info = FactorioEnv.step_payload(
            env, {"action": "wait"}, action_key="wait"
        )
        assert calls == [11], "the verifier runs exactly once, on the truncating step"
        assert terminated is True, "the verified score is the outcome; nothing to bootstrap"
        assert truncated is False, "terminated and truncated stay exclusive"
        assert reward == pytest.approx(1.0)
        assert info["verification"]["success"] is True
        assert info["success"] is True

    def test_running_out_of_decisions_verifies_too(self, monkeypatch):
        from factoriorl import encoders

        monkeypatch.setattr(encoders, "encode", lambda observation, goal: {})
        spec = get("construct_smelting_line").spec
        env, calls = _stepping_env(
            tick=300,
            steps=spec.max_decision_steps - 1,
            verified={"success": False, "reward": 0.0, "machine_output": 0},
        )
        _, reward, terminated, truncated, info = FactorioEnv.step_payload(
            env, {"action": "wait"}, action_key="wait"
        )
        assert calls, "a decision budget running out is still the end of construction"
        assert terminated is True and truncated is False
        assert reward == 0.0
        assert info["success"] is False

    def test_an_ordinary_step_is_not_verified(self, monkeypatch):
        from factoriorl import encoders

        monkeypatch.setattr(encoders, "encode", lambda observation, goal: {})
        env, calls = _stepping_env(
            tick=300,
            steps=10,
            verified={"success": True, "reward": 1.0, "machine_output": 12},
        )
        _, _, terminated, truncated, info = FactorioEnv.step_payload(
            env, {"action": "wait"}, action_key="wait"
        )
        assert calls == []
        assert terminated is False and truncated is False
        assert "verification" not in info

    def test_a_driver_that_already_verified_is_not_verified_twice(self, monkeypatch):
        from factoriorl import encoders

        monkeypatch.setattr(encoders, "encode", lambda observation, goal: {})
        env, calls = _stepping_env(
            tick=18000,
            steps=10,
            verified={"success": True, "reward": 1.0, "machine_output": 12},
        )
        env._verification = {"success": True}
        _, _, terminated, truncated, _ = FactorioEnv.step_payload(
            env, {"action": "wait"}, action_key="wait"
        )
        assert calls == []
        assert truncated is True and terminated is False
