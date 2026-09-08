"""Behaviour cloning from the scripted reference solutions (PLAN.md 4.2 note).

    BC needs `M >~ ln(|Pi|/delta) gamma^2 / ((1-gamma)^4 eps^2)` expert
    state-action pairs, with **no exploration term at all** and no dependence
    on `|A|^H`.
    -- ABJKS Thm. 13.3 (p.162), Thm. 13.4 (p.163); see
       `docs/research/rl-theory.md`.

Why this exists
---------------
`repair_belt` trained for 50,000 steps and logged zero successes across 190
episodes, because until the belt is repaired nothing pays: the mean episode
reward was -0.300, exactly the step cost. `restore_power` had no shaping
component at all. Undirected exploration cannot find either goal, and the
exploration lower bound for an `H = 300` family is 15-27 M steps -- 55-100 hours
per family per seed -- so the honest options were "spend a week" or "change the
problem".

BC changes the problem. The scripted solvers already solve every family at 1.00
on both splits, so the expert exists and is free to query; cloning it replaces
an `|A|^H` exploration problem with a maximum-likelihood one whose sample
complexity has no horizon exponent. The notes call it the cheapest budget in the
program, and the only mechanism that makes `restore_power` learnable at all.

The line this must not cross
----------------------------
PLAN 4.2: "the agent receives no scripted solution labels during ordinary PPO
training." That rule is what makes the Phase 4 baseline mean something, so BC is
**a separate, declared training mode**: its own entrypoint, its own
`training_mode` in the manifest, and its own column in any results table. It is
never mixed into a PPO run, and a checkpoint produced here is not comparable to
one produced by the baseline. `factoriorl runs show` will say which it was.

The honest caveat
-----------------
Thm. 13.3 carries an error that compounds as `1/(1-gamma)^2` under the learner's
*own* state distribution: a cloned policy that drifts off the expert's
distribution has never seen the states it then has to act in. DAgger-style
relabelling on learner-visited states is the follow-on, and is not implemented
here -- so a BC checkpoint is a starting point, not a finished policy.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

#: Names the training mode in a manifest, beside `skills-v1` and
#: `language-model-v1`. A result produced by cloning an expert is not comparable
#: to one produced by exploration, and a run that does not say which cannot be
#: read later.
TRAINING_MODE = "behaviour-cloning-v1"


@dataclass
class Demonstrations:
    """Expert state-action pairs, and where they came from."""

    observations: list[dict] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    episodes: int = 0
    solved: int = 0
    #: Per-episode outcome, so a demonstration set that is mostly failures
    #: cannot be mistaken for one that is mostly expert.
    outcomes: list[dict] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.actions)

    def summary(self) -> dict:
        return {
            "pairs": len(self.actions),
            "episodes": self.episodes,
            "solved": self.solved,
            "solved_rate": round(self.solved / self.episodes, 4) if self.episodes else 0.0,
            "mean_pairs_per_episode": round(len(self.actions) / max(self.episodes, 1), 1),
        }


def collect(env, solve, episodes: int, *, first_index: int = 0) -> Demonstrations:
    """Run the scripted solver and record what it saw and what it did.

    `solve` is `tasks.reference.solve`, which takes the *environment* and builds
    the `Driver` itself. Passing the per-task solver instead hands it a driver
    it does not have, which is how the first run of this failed: the solver
    asked the recorder for `driver.success` and got an AttributeError from the
    environment underneath.

    Only *solved* episodes contribute pairs. A solver that got stuck produces a
    trajectory of an agent failing, and cloning that teaches failing -- which is
    the one thing a demonstration set must not do. The failures are still
    counted, because a set whose solver only succeeded half the time is a
    different object from one where it always did.
    """
    data = Demonstrations()
    env.unwrapped._episode_index = first_index - 1

    for _ in range(episodes):
        env.reset()
        recorder = _Recorder(env)
        trace = solve(recorder)
        data.episodes += 1
        succeeded = bool(getattr(trace, "succeeded", False))
        data.outcomes.append(
            {
                "solved": succeeded,
                "steps": len(recorder.pairs),
                "stuck_reason": getattr(trace, "stuck_reason", None),
            }
        )
        if not succeeded:
            continue
        data.solved += 1
        for observation, action in recorder.pairs:
            data.observations.append(observation)
            data.actions.append(action)
    return data


class _Recorder:
    """Wraps the env the solver drives, capturing (observation, action).

    The solver calls `step` with catalog indices, so the label is the index the
    expert chose in the state the policy would have seen -- the same encoded
    observation, not the wire one, so a cloned policy trains on exactly the
    tensor it will be given at evaluation.
    """

    def __init__(self, env) -> None:
        self._env = env
        self.pairs: list[tuple[dict, int]] = []

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action: int):
        from factoriorl import encoders

        encoded = encoders.encode(
            self._env.unwrapped._observation, self._env.unwrapped._goal_vector()
        )
        self.pairs.append(({k: np.asarray(v).copy() for k, v in encoded.items()}, int(action)))
        return self._env.step(action)


def _stack(observations: list[dict]) -> dict[str, np.ndarray]:
    return {key: np.stack([o[key] for o in observations]) for key in observations[0]}


def train_bc(
    data: Demonstrations,
    observation_space,
    action_space,
    *,
    epochs: int = 20,
    batch_size: int = 128,
    learning_rate: float = 3e-4,
    seed: int = 0,
    device: str = "auto",
) -> tuple[Any, dict]:
    """Fit a MaskablePPO policy's network to the expert's action labels.

    The result is a `MaskablePPO` whose policy has been trained by supervised
    learning rather than by rollouts. Using the same class matters: the
    checkpoint then loads, evaluates and replays through exactly the same code
    as a PPO checkpoint, so a BC result and a baseline result are compared on
    identical evaluation machinery even though they were produced differently.
    """
    import gymnasium as gym
    import torch
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    from factoriorl.learn.policy import policy_kwargs

    if not len(data):
        raise ValueError(
            "no expert pairs: the solver never finished an episode, so there is nothing to clone"
        )

    # A throwaway vector env purely so MaskablePPO can build its network against
    # the right spaces. It is never stepped, but SB3 type-checks it, so it has
    # to be a real `gymnasium.Env` rather than something duck-shaped.
    class _Shell(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self):
            super().__init__()
            self.observation_space = observation_space
            self.action_space = action_space
            self.render_mode = None

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return observation_space.sample(), {}

        def step(self, _action):
            return observation_space.sample(), 0.0, True, False, {}

        def action_masks(self):
            return np.ones(action_space.n, dtype=bool)

    model = MaskablePPO(
        "MultiInputPolicy",
        DummyVecEnv([lambda: _Shell()]),
        policy_kwargs=policy_kwargs(),
        seed=seed,
        device=device,
        verbose=0,
    )

    torch_device = model.policy.device
    features = _stack(data.observations)
    labels = torch.as_tensor(np.asarray(data.actions), dtype=torch.long, device=torch_device)
    tensors = {
        key: torch.as_tensor(value, dtype=torch.float32, device=torch_device)
        for key, value in features.items()
    }

    optimiser = torch.optim.Adam(model.policy.parameters(), lr=learning_rate)
    generator = np.random.default_rng(seed)
    history: list[dict] = []
    total = len(labels)

    for epoch in range(epochs):
        order = generator.permutation(total)
        losses, correct = [], 0
        for start in range(0, total, batch_size):
            index = order[start : start + batch_size]
            batch_index = torch.as_tensor(index, dtype=torch.long, device=torch_device)
            batch = {key: value[batch_index] for key, value in tensors.items()}
            target = labels[batch_index]

            latent = model.policy.extract_features(batch)
            if isinstance(latent, tuple):
                latent = latent[0]
            latent_pi = model.policy.mlp_extractor.forward_actor(latent)
            logits = model.policy.action_net(latent_pi)

            loss = torch.nn.functional.cross_entropy(logits, target)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            losses.append(float(loss.item()))
            correct += int((logits.argmax(dim=1) == target).sum().item())
        history.append(
            {
                "epoch": epoch,
                "loss": round(float(np.mean(losses)), 5),
                # Accuracy on the expert's own labels. It says the network fits
                # the demonstrations; it says nothing about behaving well off
                # their distribution, which is Thm. 13.3's caveat.
                "expert_action_accuracy": round(correct / total, 4),
            }
        )

    report = {
        "training_mode": TRAINING_MODE,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "demonstrations": data.summary(),
        "history": history,
        "final_expert_action_accuracy": history[-1]["expert_action_accuracy"] if history else 0.0,
        "caveat": (
            "Expert-action accuracy is measured on the expert's own state "
            "distribution. ABJKS Thm. 13.3: cloned error compounds as 1/(1-gamma)^2 "
            "off that distribution, so this is a starting point rather than a "
            "finished policy. DAgger-style relabelling is the follow-on."
        ),
    }
    return model, report


def write_report(run_dir: Path, report: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {**report, "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (run_dir / "bc.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
