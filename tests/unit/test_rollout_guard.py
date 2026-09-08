"""An infrastructure failure must never reach an optimizer update.

`excluded_from_metrics` was a reporting flag only: nothing in the training path
read it, `vecenv.step_wait` dressed a transport failure as a time limit, and
`sb3_contrib` therefore bootstrapped `gamma * V(stale observation)` into the
reward and added the transition to the rollout buffer like any other. A
simulator fault was learned from.

These tests drive a **real** `MaskablePPO` over a stub `VecEnv`, because a test
of the logger alone cannot show what the collector does. The stub mirrors
`FactorioVecEnv`'s public surface -- the same method set, the same observation
and action spaces, the same `action_masks` exposure through `env_method` -- and
`test_the_stub_matches_the_real_vec_env_surface` fails if the real class grows a
method the stub does not have, so a green result here cannot come from a stub
that has drifted from the thing it stands in for.
"""

from __future__ import annotations

import numpy as np
import pytest
from stable_baselines3.common.vec_env import VecEnv

from factoriorl import encoders
from factoriorl.learn.train import RolloutGuard
from factoriorl.vecenv import FactorioVecEnv

PROFILE = encoders.LOCAL_V1
ACTIONS = 6


def _observation():
    space = encoders.observation_space(PROFILE)
    return {key: np.zeros(sub.shape, dtype=sub.dtype) for key, sub in space.spaces.items()}


def _stack(observations):
    return {key: np.stack([obs[key] for obs in observations]) for key in observations[0]}


class StubVecEnv(VecEnv):
    """Enough of `FactorioVecEnv` to exercise the real collector.

    `fail_at` names the vector step (0-based) on which one worker reports an
    infrastructure failure, shaped exactly as `FactorioEnv.step_payload`
    reports one.
    """

    def __init__(self, num_envs: int = 2, fail_at: int | None = None, fail_env: int = 0):
        super().__init__(num_envs, encoders.observation_space(PROFILE), _space())
        self.fail_at = fail_at
        self.fail_env = fail_env
        self.calls = 0
        self._actions = None
        self.emitted_infos: list[dict] = []

    # ---- the surface FactorioVecEnv exposes -----------------------------
    def reset(self):
        return _stack([_observation() for _ in range(self.num_envs)])

    def step_async(self, actions) -> None:
        self._actions = actions

    def step_wait(self):
        index = self.calls
        self.calls += 1
        observations, rewards, dones, infos = [], [], [], []
        for env_index in range(self.num_envs):
            failing = (
                self.fail_at is not None and index == self.fail_at and env_index == self.fail_env
            )
            info: dict = {"action_mask": np.ones(ACTIONS, dtype=bool)}
            if failing:
                # Exactly what env.step_payload returns, and exactly what
                # vecenv.step_wait then does with it: an episode boundary is
                # reported, but no bootstrappable terminal observation.
                info["infrastructure_failure"] = "transport failure: [WinError 10054]"
                info["excluded_from_metrics"] = True
                info["terminal_observation"] = None
                info["TimeLimit.truncated"] = False
                dones.append(True)
                rewards.append(0.0)
            else:
                dones.append(False)
                rewards.append(-0.001)
            observations.append(_observation())
            infos.append(info)
        self.emitted_infos = infos
        return (
            _stack(observations),
            np.array(rewards, dtype=np.float32),
            np.array(dones),
            infos,
        )

    def close(self) -> None:
        return None

    def env_method(self, method_name: str, *args, indices=None, **kwargs) -> list:
        if method_name == "action_masks":
            return [np.ones(ACTIONS, dtype=bool) for _ in range(self.num_envs)]
        raise NotImplementedError(method_name)

    def get_attr(self, attr_name: str, indices=None) -> list:
        return [getattr(self, attr_name, None)] * self.num_envs

    def set_attr(self, attr_name: str, value, indices=None) -> None:
        setattr(self, attr_name, value)

    def env_is_wrapped(self, wrapper_class, indices=None) -> list[bool]:
        return [False] * self.num_envs

    def seed(self, seed: int | None = None) -> list:
        return [seed] * self.num_envs

    def action_masks(self) -> np.ndarray:
        return np.ones((self.num_envs, ACTIONS), dtype=bool)


def _space():
    from gymnasium import spaces

    return spaces.Discrete(ACTIONS)


def _model(env, n_steps: int = 8):
    from sb3_contrib import MaskablePPO

    return MaskablePPO(
        "MultiInputPolicy",
        env,
        n_steps=n_steps,
        batch_size=n_steps * env.num_envs,
        n_epochs=1,
        device="cpu",
        verbose=0,
    )


# ------------------------------------------------------------------ the gate


def test_no_optimizer_update_uses_a_contaminated_rollout():
    """The gate: a failure mid-rollout must stop collection before `train()`."""
    env = StubVecEnv(num_envs=2, fail_at=3)
    model = _model(env, n_steps=8)
    guard = RolloutGuard()

    updates = []
    real_train = model.train
    model.train = lambda *a, **k: updates.append(model.num_timesteps)  # type: ignore[method-assign]

    model.learn(total_timesteps=64, callback=guard, progress_bar=False)

    assert guard.aborted is not None, "the guard did not notice the failure"
    assert "10054" in guard.aborted.cause
    assert updates == [], f"an optimizer update ran on a contaminated rollout: {updates}"
    assert real_train is not None  # the spy replaced a real method, not a typo


def test_a_clean_rollout_still_trains():
    """The guard must not abort runs that have no failure in them."""
    env = StubVecEnv(num_envs=2, fail_at=None)
    model = _model(env, n_steps=8)
    guard = RolloutGuard()

    updates = []
    model.train = lambda *a, **k: updates.append(model.num_timesteps)  # type: ignore[method-assign]

    model.learn(total_timesteps=32, callback=guard, progress_bar=False)

    assert guard.aborted is None
    assert updates, "a clean rollout produced no optimizer update"


class _Future:
    def __init__(self, payload):
        self._payload = payload

    def result(self):
        return self._payload


class _InnerEnv:
    """Stands in for a wrapped `FactorioEnv` inside the real vec env."""

    def __init__(self):
        self.unwrapped = self
        self._assigned_index = 7
        self.resets = 0

    def reset(self):
        self.resets += 1
        return _observation(), {}


def _real_step_wait(payloads):
    """Drive `FactorioVecEnv.step_wait` itself, with no engine.

    Asserting on the stub's own emitted info would test the fixture rather
    than the code -- the mistake the exploring-starts occupancy guard made.
    This calls the real method, bound to hand-built inner envs, so the
    behaviour under test is `vecenv`'s.
    """
    vec = object.__new__(FactorioVecEnv)
    vec.envs = [_InnerEnv() for _ in payloads]
    vec._pending = [_Future(p) for p in payloads]
    vec._cursor = None
    vec._stop = None
    vec._issued = []
    _, _, dones, infos = FactorioVecEnv.step_wait(vec)
    return dones, infos, vec


def test_vecenv_does_not_dress_a_failure_as_a_time_limit():
    """A simulator fault must not look like a legitimate truncation.

    `sb3_contrib` bootstraps `rewards[i] += gamma * V(terminal_observation)`
    when and only when `TimeLimit.truncated` is set with a non-None
    `terminal_observation`. Both must be absent for a failure, or the learner
    fabricates a value target from an observation that never happened -- the
    failed step returns the *stale* observation from before the step, because
    the step never came back.
    """
    failure = (
        _observation(),
        0.0,
        False,
        True,
        {
            "infrastructure_failure": "transport failure: [WinError 10054]",
            "excluded_from_metrics": True,
        },
    )
    dones, infos, _ = _real_step_wait([failure])

    assert dones[0], "the episode boundary must still be reported so the env resets"
    assert infos[0]["TimeLimit.truncated"] is False
    assert infos[0]["terminal_observation"] is None
    assert infos[0]["infrastructure_failure"]


def test_vecenv_still_bootstraps_a_legitimate_truncation():
    """The two cases must not be conflated in either direction."""
    truncation = (_observation(), -0.001, False, True, {})
    dones, infos, _ = _real_step_wait([truncation])

    assert dones[0]
    assert infos[0]["TimeLimit.truncated"] is True
    assert infos[0]["terminal_observation"] is not None


def test_vecenv_marks_a_real_termination_as_not_truncated():
    termination = (_observation(), 1.0, True, False, {"success": True})
    dones, infos, _ = _real_step_wait([termination])

    assert dones[0]
    assert infos[0]["TimeLimit.truncated"] is False
    assert infos[0]["terminal_observation"] is not None


def test_guard_reports_the_timestep_it_aborted_at():
    env = StubVecEnv(num_envs=2, fail_at=2)
    model = _model(env, n_steps=8)
    model.train = lambda *a, **k: None  # type: ignore[method-assign]
    guard = RolloutGuard()
    model.learn(total_timesteps=64, callback=guard, progress_bar=False)

    assert guard.aborted is not None
    # Three vector steps of two envs each.
    assert guard.aborted.timestep == 6
    assert "timestep 6" in str(guard.aborted)


# --------------------------------------------------- the stub must stay honest


def test_the_stub_matches_the_real_vec_env_surface():
    """A stub that has drifted from `FactorioVecEnv` proves nothing.

    Every public method the real vec env defines must exist on the stub, so
    this test fails when the real class grows a method the collector might
    depend on and the stub silently does not provide it.
    """
    real = {
        name
        for name in vars(FactorioVecEnv)
        if callable(getattr(FactorioVecEnv, name)) and not name.startswith("_")
    }
    # `retarget` and `issued_episodes` are evaluation-only helpers the collector
    # never calls; everything else the collector may reach must be present.
    collector_surface = real - {"retarget", "issued_episodes"}
    missing = sorted(name for name in collector_surface if not hasattr(StubVecEnv, name))
    assert not missing, f"StubVecEnv is missing {missing} from FactorioVecEnv's surface"


def test_the_stub_really_drives_masked_action_selection():
    """If masking were unsupported the collector would take a different path."""
    from sb3_contrib.common.maskable.utils import is_masking_supported

    assert is_masking_supported(StubVecEnv(num_envs=2))


@pytest.mark.parametrize("fail_env", [0, 1])
def test_a_failure_in_any_worker_aborts(fail_env):
    """The old logger read `infos[0]`; a guard must not repeat that."""
    env = StubVecEnv(num_envs=2, fail_at=1, fail_env=fail_env)
    model = _model(env, n_steps=8)
    model.train = lambda *a, **k: None  # type: ignore[method-assign]
    guard = RolloutGuard()
    model.learn(total_timesteps=64, callback=guard, progress_bar=False)
    assert guard.aborted is not None, f"a failure in worker {fail_env} was not noticed"
