"""A checkpoint has to restore training, not only inference (PLAN 4.1).

PLAN 4.1's acceptance reads: *"Model checkpoints restore both inference and
resumable training state."* `gate_phase4.py` checks the first half --
`MaskablePPO.load(...)` returns something -- and stops there, so the second
half has been an unverified claim rather than a passing one, on a gate that
otherwise reports 15 of 15.

The two halves fail differently. A checkpoint that deserialises but drops its
optimizer state resumes with a freshly initialised Adam: the moment estimates
are gone, so the first updates after a resume are effectively a re-warm-up and
a "continued" run is not the run it claims to continue. A checkpoint that
resets `num_timesteps` is worse in a quieter way -- the schedule restarts, and
a learning-rate or clip-range schedule silently rewinds while the curve keeps
counting up.

Engine-free: a three-action masked bandit stands in for `FactorioEnv`. What is
under test is SB3's save/load round trip as this repo uses it, not the
environment.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("sb3_contrib")

import torch as th  # noqa: E402
from gymnasium import Env, spaces  # noqa: E402
from sb3_contrib import MaskablePPO  # noqa: E402

#: Small on purpose. This measures a save/load contract, not learning: 128
#: steps is two rollouts of 64 and takes under a second on CPU.
STEPS = 128
ROLLOUT = 64


class MaskedBandit(Env):
    """Three actions, one masked out, reward for picking the best legal one.

    A `FactorioEnv` cannot be used here -- it needs a worker -- and it does not
    need to be: `MaskablePPO.save` serialises the policy and its optimizer
    without consulting the environment.
    """

    metadata: dict = {}

    def __init__(self) -> None:
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self._steps = 0
        self._rng = np.random.default_rng(0)

    def action_masks(self) -> np.ndarray:
        # The third action is always illegal, which is what makes this a
        # *maskable* problem rather than a plain one.
        return np.array([True, True, False], dtype=bool)

    def _observation(self) -> np.ndarray:
        return self._rng.uniform(-1.0, 1.0, size=(4,)).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._steps = 0
        return self._observation(), {}

    def step(self, action):
        self._steps += 1
        reward = 1.0 if int(action) == 0 else 0.0
        return self._observation(), reward, False, self._steps >= 32, {}


def _model() -> MaskablePPO:
    return MaskablePPO(
        "MlpPolicy",
        MaskedBandit(),
        n_steps=ROLLOUT,
        batch_size=ROLLOUT,
        n_epochs=1,
        seed=0,
        device="cpu",
        verbose=0,
    )


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    """One trained-and-saved model, reused: training twice costs nothing but time."""
    directory = tmp_path_factory.mktemp("resume")
    model = _model()
    model.learn(total_timesteps=STEPS)
    path = directory / "model"
    model.save(path)
    return model, path


class TestInferenceStillWorks:
    """The half the gate already checks, kept here so the two are read together."""

    def test_the_checkpoint_loads_and_predicts(self, saved):
        _original, path = saved
        loaded = MaskablePPO.load(path, device="cpu")
        env = MaskedBandit()
        observation, _ = env.reset(seed=1)
        action, _state = loaded.predict(
            observation, action_masks=env.action_masks(), deterministic=True
        )
        assert int(action) in (0, 1), "a masked-out action must never be predicted"


class TestTrainingStateSurvives:
    def test_the_policy_weights_are_identical(self, saved):
        original, path = saved
        loaded = MaskablePPO.load(path, device="cpu")
        before = original.policy.state_dict()
        after = loaded.policy.state_dict()
        assert set(before) == set(after)
        for key, value in before.items():
            assert th.allclose(value.cpu(), after[key].cpu()), key

    def test_the_optimizer_state_comes_back_and_is_not_empty(self, saved):
        """The failure this exists for: Adam's moment estimates are the training
        state. A checkpoint that restores weights and drops these resumes with a
        fresh optimizer, so the first updates after a resume are a re-warm-up
        and the continued run is not the run it claims to continue."""
        original, path = saved
        loaded = MaskablePPO.load(path, device="cpu")
        before = original.policy.optimizer.state_dict()["state"]
        after = loaded.policy.optimizer.state_dict()["state"]
        assert before, "the original trained model has no optimizer state to compare"
        assert set(after) == set(before), "optimizer state was not restored"
        for index, entry in before.items():
            for key, value in entry.items():
                if isinstance(value, th.Tensor):
                    assert th.allclose(value.cpu(), after[index][key].cpu()), (index, key)
                else:
                    assert value == after[index][key], (index, key)

    def test_the_step_counter_survives(self, saved):
        original, path = saved
        loaded = MaskablePPO.load(path, device="cpu")
        assert loaded.num_timesteps == original.num_timesteps
        assert loaded.num_timesteps >= STEPS

    def test_resuming_continues_the_count_rather_than_restarting_it(self, saved):
        """`reset_num_timesteps=False` is what makes a resume a continuation.
        Without it the schedule rewinds while the curve keeps counting up, which
        is the quiet half of this failure."""
        _original, path = saved
        loaded = MaskablePPO.load(path, device="cpu")
        loaded.set_env(MaskedBandit())
        started_at = loaded.num_timesteps
        loaded.learn(total_timesteps=ROLLOUT, reset_num_timesteps=False)
        assert loaded.num_timesteps > started_at

    def test_a_resumed_update_changes_the_weights(self, saved):
        """A resume that loads but cannot train would pass every check above."""
        _original, path = saved
        loaded = MaskablePPO.load(path, device="cpu")
        loaded.set_env(MaskedBandit())
        before = {k: v.clone() for k, v in loaded.policy.state_dict().items()}
        loaded.learn(total_timesteps=ROLLOUT, reset_num_timesteps=False)
        after = loaded.policy.state_dict()
        assert any(
            not th.allclose(before[key].cpu(), after[key].cpu()) for key in before
        ), "no weight moved, so the resumed run did not actually train"
