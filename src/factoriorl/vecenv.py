"""Threaded vectorised environment (PLAN.md 4.3).

A 30-tick decision interval costs ~16 ms of engine time on this machine -- the
engine plateaus near 1,800 UPS no matter what ``game.speed`` is set to -- and a
round trip adds ~1.5 ms. So a single worker cannot step faster than about 19 ms,
and the only way to go faster is to overlap workers.

**Threads, not subprocesses.** The engines are already separate OS processes and
the expensive part of a step is waiting on a socket while one of them ticks,
which releases the GIL. A subprocess vec env would add N Python interpreters and
pickle every observation for no benefit.

Measured: 41.8 steps/s at one worker, 202 steps/s at eight.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from factoriorl.env import FactorioEnv
from factoriorl.pool import WorkerPool
from factoriorl.rcon import RCONClient
from factoriorl.seeding import Branch, SeedPlan
from factoriorl.session import WorkerSession
from factoriorl.tasks import RegisteredTask

#: Raising this past ~30 buys nothing: the engine is already at its UPS ceiling,
#: but it does keep the paused server loop polling RCON quickly (16.6 ms at
#: speed 1 versus 1.5 ms at speed 60).
DEFAULT_SPEED = 60.0


class FactorioVecEnv(VecEnv):
    """N independent workers stepped in parallel."""

    def __init__(
        self,
        task: RegisteredTask,
        seed_plan: SeedPlan,
        num_workers: int = 4,
        branch: Branch = Branch.TRAIN,
        split: str = "train",
        shaping: bool = True,
        speed: float = DEFAULT_SPEED,
        worker_prefix: str = "vec",
    ) -> None:
        self.pool = WorkerPool()
        self.envs: list[FactorioEnv] = []
        self._executor = ThreadPoolExecutor(max_workers=num_workers)
        self._pending: list[Any] | None = None

        for index in range(num_workers):
            worker = self.pool.start(f"{worker_prefix}-{index}")
            with RCONClient(worker.spec.rcon_endpoint, timeout=30.0) as client:
                client.lua(f"game.speed = {speed} return game.speed")
            session = WorkerSession(worker.handle, timeout=30.0)
            session.status()
            env = FactorioEnv(task, session, seed_plan, branch=branch, split=split, shaping=shaping)
            # Distinct episode streams per worker, so two workers never run the
            # same episode at the same time.
            env._episode_index = index * 100_000 - 1
            self.envs.append(env)
            # Stagger: the launch storm is the worst commit spike.
            time.sleep(0.15)

        first = self.envs[0]
        super().__init__(num_workers, first.observation_space, first.action_space)

    # ------------------------------------------------------------ VecEnv

    def reset(self):
        results = list(self._executor.map(lambda e: e.reset(), self.envs))
        self._last_infos = [info for _, info in results]
        return _stack([obs for obs, _ in results])

    def step_async(self, actions: np.ndarray) -> None:
        self._pending = [
            self._executor.submit(env.step, int(action))
            for env, action in zip(self.envs, actions, strict=True)
        ]

    def step_wait(self):
        observations, rewards, dones, infos = [], [], [], []
        for env, future in zip(self.envs, self._pending, strict=True):
            observation, reward, terminated, truncated, info = future.result()
            done = terminated or truncated
            if done:
                # SB3 expects auto-reset, with the final observation preserved
                # so the learner can bootstrap correctly.
                info = {**info, "terminal_observation": observation}
                info["TimeLimit.truncated"] = truncated and not terminated
                observation, _ = env.reset()
            observations.append(observation)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        self._pending = None
        return _stack(observations), np.array(rewards, dtype=np.float32), np.array(dones), infos

    def close(self) -> None:
        self._executor.shutdown(wait=False)
        for env in self.envs:
            try:
                env.session.close()
            except OSError:
                pass
        self.pool.close(preserve_evidence=False)

    # ---- the bits sb3-contrib's masking and SB3's plumbing reach for -----

    def env_method(self, method_name: str, *args, indices=None, **kwargs) -> list:
        targets = self._targets(indices)
        return [getattr(env, method_name)(*args, **kwargs) for env in targets]

    def get_attr(self, attr_name: str, indices=None) -> list:
        return [getattr(env, attr_name) for env in self._targets(indices)]

    def set_attr(self, attr_name: str, value, indices=None) -> None:
        for env in self._targets(indices):
            setattr(env, attr_name, value)

    def env_is_wrapped(self, wrapper_class, indices=None) -> list[bool]:
        return [False for _ in self._targets(indices)]

    def seed(self, seed: int | None = None) -> list:
        return [seed for _ in self.envs]

    def _targets(self, indices) -> list[FactorioEnv]:
        if indices is None:
            return self.envs
        if isinstance(indices, int):
            return [self.envs[indices]]
        return [self.envs[i] for i in indices]

    def action_masks(self) -> np.ndarray:
        return np.stack([env.action_masks() for env in self.envs])


def _stack(observations: list[dict]) -> dict:
    keys = observations[0].keys()
    return {key: np.stack([o[key] for o in observations]) for key in keys}
