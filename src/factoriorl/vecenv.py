"""Threaded vectorised environment (PLAN.md 4.3).

A 30-tick decision interval costs ~5.5 ms of engine time on this machine and a
round trip adds a couple of milliseconds, so a single worker steps in about
9 ms and the way past that is to overlap workers.

An earlier version of this note claimed the engine "plateaus near 1,800 UPS no
matter what ``game.speed`` is set to". That was wrong, and wrong in the
expensive direction: 1,800 UPS is simply what speed 30 asks for. The engine
tracks the multiplier faithfully to roughly 5,480 UPS.

**Threads, not subprocesses.** The engines are already separate OS processes and
the expensive part of a step is waiting on a socket while one of them ticks,
which releases the GIL. A subprocess vec env would add N Python interpreters and
pickle every observation for no benefit.

Measured: 41.8 steps/s at one worker, 202 steps/s at eight.
"""

from __future__ import annotations

import collections
import itertools
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from factoriorl.env import FactorioEnv
from factoriorl.errors import InfrastructureFailure, ProtocolError
from factoriorl.pool import WorkerPool
from factoriorl.rcon import RCONClient
from factoriorl.seeding import Branch, SeedPlan
from factoriorl.session import WorkerSession
from factoriorl.skills import SkillEnv
from factoriorl.tasks import RegisteredTask

#: Raising this past ~30 buys nothing: the engine is already at its UPS ceiling,
#: but it does keep the paused server loop polling RCON quickly (16.6 ms at
#: speed 1 versus 1.5 ms at speed 60).
#: 90, not 60, and not higher. The engine tracks `game.speed` proportionally
#: only until it hits its own tick-rate ceiling -- measured at ~5,480 UPS on a
#: small scene, reached around speed 90-100 and flat thereafter at 120, 150,
#: 200 and 1000. Below the ceiling the client's predicted advance wait is
#: accurate; above it the prediction is optimistic, the first collect arrives
#: before the world has settled, and the extra polls give back exactly what the
#: faster ticks bought. Measured end to end on `navigate`: 11.16 ms/step at 60,
#: 8.82 ms at 90, 9.73 ms at 120, 10.34 ms at 200.
DEFAULT_SPEED = 90.0


class EpisodeResetFailed(RuntimeError):
    """A scene could not be installed. Carries the identity so it can be retried.

    Previously `_reset_one` called `env.reset()` unguarded, so a transport
    failure there propagated out of `step_wait` and killed the run. That frame
    appears in eight recorded `WinError 10054` failures.
    """

    def __init__(self, index: int | None, cause: str) -> None:
        super().__init__(f"episode {index} failed to reset: {cause}")
        self.index = index
        self.cause = cause


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
        start_curriculum: float = 0.0,
        speed: float = DEFAULT_SPEED,
        worker_prefix: str = "vec",
        skills: bool = False,
    ) -> None:
        self.pool = WorkerPool()
        self.envs: list = []
        self._executor = ThreadPoolExecutor(max_workers=num_workers)
        self._pending: list[Any] | None = None
        # When set, episode indices are handed out from one shared counter
        # instead of each worker owning a private stream. Evaluation needs
        # that: the set of scenes evaluated must be {0..N-1} whatever the
        # worker count, or two arms of an ablation would be scored on
        # different scenes and could not be compared as paired measurements.
        self._cursor: itertools.count | None = None
        self._cursor_lock = threading.Lock()
        #: Frozen indices still owed, when a bounded stream is installed. A
        #: deque rather than a counter so a failed scene can be handed back.
        self._pending_indices: collections.deque[int] | None = None
        self._attempts: collections.Counter = collections.Counter()
        #: Exclusive upper bound on issued episode indices, or None for an
        #: unbounded stream. Set only by a frozen-holdout evaluation.
        self._stop: int | None = None
        #: Episode indices actually handed out since the last retarget.
        #:
        #: A run can cite a frozen holdout's hash without having evaluated a
        #: single one of its episodes, and no downstream check could tell --
        #: a manifest carries whatever hash it is given. Recording what was
        #: issued turns the citation into something falsifiable.
        self._issued: list[int] = []

        for index in range(num_workers):
            worker = self.pool.start(f"{worker_prefix}-{index}")
            with RCONClient(worker.spec.rcon_endpoint, timeout=30.0) as client:
                client.lua(f"game.speed = {speed} return game.speed")
            session = WorkerSession(worker.handle, timeout=30.0)
            session.status()
            env = FactorioEnv(
                task,
                session,
                seed_plan,
                branch=branch,
                split=split,
                shaping=shaping,
                start_curriculum=start_curriculum,
            )
            # Distinct episode streams per worker, so two workers never run the
            # same episode at the same time. Set before wrapping: `SkillEnv`
            # forwards attribute *reads* to the inner environment but an
            # assignment would land on the wrapper, leaving every worker on
            # episode stream zero.
            env._episode_index = index * 100_000 - 1
            if skills:
                env = SkillEnv(env)
            self.envs.append(env)
            # Stagger: the launch storm is the worst commit spike.
            time.sleep(0.15)

        first = self.envs[0]
        super().__init__(num_workers, first.observation_space, first.action_space)

    # ------------------------------------------------------------ VecEnv

    def retarget(
        self,
        split: str,
        branch: Branch,
        start_index: int = 0,
        plan: SeedPlan | None = None,
        stop_index: int | None = None,
    ) -> None:
        """Point every worker at a different split, reusing the live engines.

        Evaluating three splits by building three vectorised environments would
        pay the worker launch storm three times over -- the single most
        expensive moment in a run. The split and branch are plain attributes of
        each environment, so retargeting is assignment.
        """
        for env in self.envs:
            inner = env.unwrapped
            inner.branch = branch
            inner.split = split
            if plan is not None:
                # A frozen holdout defines its own seed stream. Retargeting the
                # split without also retargeting the plan would evaluate the
                # right *families* from the wrong episodes, which is the
                # failure mode a frozen holdout exists to prevent.
                inner.seed_plan = plan
        # An explicit queue rather than `itertools.count` + a bound.
        #
        # The counter could not express "give this index back". Past the bound
        # `_reset_one` returned `index = None` while still playing a real
        # episode, which was never eligible, so `evaluate_parallel`'s
        # `while counted < episodes` spun on genuine engine episodes forever --
        # with no timeout at any level, and `--eval-episodes 100` against
        # `episodes_per_task 100` leaving exactly zero slack. One excluded
        # episode was enough to hang a run.
        if stop_index is None:
            self._cursor = itertools.count(start_index)
            self._pending_indices = None
        else:
            self._cursor = None
            self._pending_indices = collections.deque(range(start_index, stop_index))
        self._stop = stop_index
        self._issued = []
        self._attempts = collections.Counter()

    def _reset_one(self, env):
        """Assign the next episode index, and tag the env with what it got.

        The tag is what makes a frozen evaluation checkable. More episodes are
        *started* than are scored -- with N workers the last N-1 are begun and
        abandoned when the count is reached -- so "which episodes did this run
        touch" and "which episodes did this run score" are different sets, and
        only the second one can be compared across arms.

        Past ``stop_index`` the env is reset without an assignment and tagged
        None. It still has to produce an observation, because the vectorised
        contract has no way to say "this slot is finished", but a None tag tells
        the evaluator not to count it. That is what keeps a bounded holdout from
        being scored on episodes outside itself.
        """
        index = None
        if self._pending_indices is not None:
            with self._cursor_lock:
                index = self._pending_indices.popleft() if self._pending_indices else None
                if index is not None:
                    self._issued.append(index)
                    self._attempts[index] += 1
        elif self._cursor is not None:
            with self._cursor_lock:
                index = next(self._cursor)
                self._issued.append(index)
        if index is not None:
            # reset() increments before use, so seed it one below the target.
            env.unwrapped._episode_index = index - 1
        env.unwrapped._assigned_index = index
        try:
            return env.reset()
        except (InfrastructureFailure, ProtocolError) as exc:
            # A reset failure used to propagate out of `step_wait` and kill the
            # run. That frame -- `_reset_one` -> `env.reset()` -- is in eight
            # recorded `WinError 10054` failures, and is why the crash fired
            # before the hang. The scene is handed back so it can be retried
            # rather than silently lost.
            if index is not None:
                self.release_episode(index)
            raise EpisodeResetFailed(index, str(exc)) from exc

    def reset(self):
        results = list(self._executor.map(self._reset_one, self.envs))
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
            failed = bool(info.get("infrastructure_failure"))
            if done:
                # SB3 expects auto-reset, with the final observation preserved
                # so the learner can bootstrap correctly.
                #
                # Except after an infrastructure failure. `TimeLimit.truncated`
                # plus a `terminal_observation` is precisely the shape
                # `sb3_contrib` looks for to bootstrap
                # `rewards[i] += gamma * V(terminal_obs)`, and the observation
                # a failed step returns is the *stale* one from before the
                # step, because the step never came back. Dressing a transport
                # failure as a time limit therefore fabricated a value target
                # for a legal action off an observation that never happened.
                # The episode boundary is still reported, so the env resets;
                # only the bootstrap is withheld.
                info = {**info, "terminal_observation": None if failed else observation}
                info["TimeLimit.truncated"] = bool(truncated and not terminated and not failed)
                # Read before the reset overwrites it: this names the episode
                # that just finished, not the one about to start.
                info["episode_index"] = getattr(env.unwrapped, "_assigned_index", None)
                observation, _ = self._reset_one(env)
            observations.append(observation)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        self._pending = None
        return _stack(observations), np.array(rewards, dtype=np.float32), np.array(dones), infos

    def close(self) -> None:
        # Sessions first, executor second. A worker thread parked on a socket
        # read does not return until that socket is closed, and
        # ThreadPoolExecutor threads are non-daemon -- the interpreter joins
        # them at exit. Shutting the executor down first therefore left a
        # completed run hanging forever on a thread that could never finish,
        # holding every engine in its pool: eight idle Factorio processes
        # competing with whatever ran next, which silently halves the
        # throughput of the following measurement rather than failing.
        for env in self.envs:
            try:
                env.session.close()
            except OSError:
                pass
        self._executor.shutdown(wait=False, cancel_futures=True)
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

    def issued_episodes(self) -> list[int]:
        """Episode indices handed out since the last retarget.

        More are started than are scored: with N workers the last N-1 episodes
        are begun and abandoned when the count is reached, and an episode
        dropped as an infrastructure failure advances the cursor too. So this
        is the set the evaluation *touched*, which is the honest thing to check
        a frozen holdout against.
        """
        return list(self._issued)

    def release_episode(self, index: int, retry_limit: int = 3) -> bool:
        """Hand a scene identity back so the same scene can be retried.

        Returns True if it was requeued. Beyond `retry_limit` attempts it is
        not, and the caller must report incomplete coverage rather than
        substitute a different scene: a frozen holdout that quietly evaluates
        99 of its 100 episodes, or swaps one for another, is no longer the
        artifact its hash names.
        """
        if self._pending_indices is None:
            return False
        with self._cursor_lock:
            if self._attempts[index] >= retry_limit:
                return False
            self._pending_indices.append(index)
            return True

    def pending_episodes(self) -> int:
        """How many frozen indices are still owed. None-bounded streams: -1."""
        if self._pending_indices is None:
            return -1
        with self._cursor_lock:
            return len(self._pending_indices)

    def episode_attempts(self) -> dict[int, int]:
        return dict(self._attempts)

    def in_flight_episodes(self) -> list[int | None]:
        """Scene identities the workers are currently playing.

        An empty pending queue does not mean nothing more can be scored: with
        N workers, N episodes are already loaded and will report on the next
        step. Terminating on the queue alone dropped those, which showed up as
        a clean 6-scene evaluation reporting two episodes unrecoverable.
        """
        return [getattr(env.unwrapped, "_assigned_index", None) for env in self.envs]

    def action_masks(self) -> np.ndarray:
        return np.stack([env.action_masks() for env in self.envs])


def _stack(observations: list[dict]) -> dict:
    keys = observations[0].keys()
    return {key: np.stack([o[key] for o in observations]) for key in keys}
