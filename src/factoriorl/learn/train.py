"""Training and evaluation (PLAN.md 4.1-4.2).

CSV is the source of truth for curves, not TensorBoard: release curves must be
diffable across runs, and event files are not.

The run manifest is written **before the first step**, so an interrupted run
still has one, and it records the model configuration, the extractor version,
the resolved task, the catalog digest and the seed plan -- everything needed to
tell whether a checkpoint still means what it claimed to.
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

from factoriorl import encoders
from factoriorl import manifest as manifest_module
from factoriorl import rewards as rewards_module
from factoriorl.assistance import describe_assistance
from factoriorl.baselines import cached_random_baseline
from factoriorl.engine_config import resolve_game_speed
from factoriorl.env import FactorioEnv
from factoriorl.learn.buffers import DurationAwareMaskableDictRolloutBuffer
from factoriorl.learn.policy import EXTRACTOR_VERSION, describe, policy_kwargs
from factoriorl.rcon import RCONClient
from factoriorl.seeding import Branch, SeedPlan, seed_everything
from factoriorl.session import WorkerSession
from factoriorl.skills import SKILLS, SkillEnv
from factoriorl.tasks import get
from factoriorl.vecenv import EpisodeResetFailed
from factoriorl.worker import WorkerManager

#: Also sets the *paused* server loop rate, which is what RCON latency
#: actually depends on: 16.6 ms per round trip at speed 1, 1.5 ms at 60.
#: 90, not 60, and not higher. The engine tracks `game.speed` proportionally
#: only until it hits its own tick-rate ceiling -- measured at ~5,480 UPS on a
#: small scene, reached around speed 90-100 and flat thereafter at 120, 150,
#: 200 and 1000. Below the ceiling the client's predicted advance wait is
#: accurate; above it the prediction is optimistic, the first collect arrives
#: before the world has settled, and the extra polls give back exactly what the
#: faster ticks bought. Measured end to end on `navigate`: 11.16 ms/step at 60,
#: 8.82 ms at 90, 9.73 ms at 120, 10.34 ms at 200.
TRAIN_SPEED = resolve_game_speed()
#: Floor for the per-env rollout length, so a large worker count cannot
#: shrink it to something PPO cannot learn from.
MIN_STEPS_PER_ENV = 32


#: Names the deliberation layer in a published manifest. A result produced by
#: a policy over skills is not comparable to one produced over primitives, and
#: a run that does not say which it used cannot be interpreted later.
SKILL_PROFILE = "skills-v1"


def _wrap(env: FactorioEnv, skills: bool):
    """Give an evaluation environment the same action space as training.

    Evaluating a skill-trained policy on a primitive-only environment would
    silently truncate its action space, so this is applied to every evaluation
    and baseline environment, not only the training ones.
    """
    return SkillEnv(env) if skills else env


@dataclass
class TrainConfig:
    task_id: str
    total_steps: int = 50_000
    master_seed: int = 20260907
    learning_rate: float = 3e-4
    n_steps: int = 512
    batch_size: int = 128
    gamma: float = 0.99
    ent_coef: float = 0.01
    shaping: bool = True
    eval_episodes: int = 20
    #: PLAN section 3: validation supports debugging and model selection;
    #: test results use frozen layouts *excluded from tuning*. Routine runs
    #: and sweeps therefore report `val`; only a declared release evaluation
    #: passes `test`, and every result records which split produced it. The
    #: routine loop defaulted to `test`, which spent the holdout on
    #: candidate selection.
    eval_split: str = "val"
    #: Path to a frozen holdout (see `tools/freeze_holdout.py`). When set, the
    #: structural row is evaluated against *that* file's seed plan and episode
    #: range rather than the run's own, and the run records its content hash.
    #:
    #: Without this the citation is theatre: a run builds its evaluation plan
    #: from `--seed` and a per-run id and starts its cursor at 0, so it would
    #: cite a holdout it never evaluated a single episode of, and no check
    #: downstream could tell -- a manifest can carry any hash you like.
    holdout: str | None = None
    #: Add temporally extended actions to the primitive catalog (PLAN 4b).
    #: The 4b.3 ablation runs two arms that differ in this flag alone.
    skills: bool = False
    device: str = "auto"
    #: Parallel workers. A 30-tick interval costs ~16 ms of engine time that no
    #: setting can remove, so overlapping workers is the only way to step
    #: faster: 41.8 steps/s at one, 202 at eight.
    workers: int = 1
    #: Also score the structural row with the stochastic policy, on the same
    #: episodes. Cheap next to training and the only way to tell a policy that
    #: did not learn from one that learned and is being read greedily.
    paired_stochastic_eval: bool = True
    #: Exploring starts (Sutton & Barto §5.3, p. 79): the fraction of training
    #: episodes whose character starts beside the task's focus marker instead
    #: of at the origin. Training stream only -- no evaluated episode, not even
    #: the unfamiliar-seed row over training layouts, is affected.
    start_curriculum: float = 0.0
    run_prefix: str = "train"
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "eval_split": self.eval_split,
            "holdout": self.holdout,
            "skills": self.skills,
            "total_steps": self.total_steps,
            "master_seed": self.master_seed,
            "learning_rate": self.learning_rate,
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "gamma": self.gamma,
            "ent_coef": self.ent_coef,
            "shaping": self.shaping,
            "eval_episodes": self.eval_episodes,
            "workers": self.workers,
            "paired_stochastic_eval": self.paired_stochastic_eval,
            "start_curriculum": self.start_curriculum,
        }


class RolloutAborted(RuntimeError):
    """An infrastructure failure reached the rollout; no update may use it."""

    def __init__(self, cause: str, timestep: int) -> None:
        super().__init__(
            f"infrastructure failure during rollout collection at timestep {timestep}: {cause}"
        )
        self.cause = cause
        self.timestep = timestep


class RolloutGuard(BaseCallback):
    """Abort collection when an infrastructure failure enters the rollout.

    `excluded_from_metrics` used to be a reporting flag only. Nothing in the
    training path read it, so a transport failure was added to the rollout
    buffer like any other transition and included in the next optimizer update
    -- with a value target bootstrapped off the stale observation the failed
    step returned. A simulator fault was, in effect, learned from.

    `OnPolicyAlgorithm.learn` is `continue_training = self.collect_rollouts(...)`
    then `if not continue_training: break` and only then `self.train()`, and
    `collect_rollouts` checks `callback.on_step()` on every step. Returning
    False therefore stops collection *before* any optimizer update touches the
    contaminated buffer, which is what the redirection asks for: explicit
    rollout failure in preference to deleting transitions out of a buffer
    SB3 does not expose per-transition.

    The run is then marked incomplete rather than reporting a rate, and the
    last valid checkpoint is preserved.
    """

    def __init__(self) -> None:
        super().__init__()
        self.aborted: RolloutAborted | None = None

    def _on_step(self) -> bool:
        for info in self.locals.get("infos") or []:
            cause = info.get("infrastructure_failure")
            if cause:
                self.aborted = RolloutAborted(str(cause), int(self.num_timesteps))
                return False
        return True


class DurationRecorder(BaseCallback):
    """Write each transition's primitive duration into the rollout buffer.

    `_on_step` runs after `buffer.add`, so the row just written is at
    `pos - 1`. Using this seam avoids overriding `collect_rollouts`.
    """

    def __init__(self) -> None:
        super().__init__()
        self.total_primitive_steps = 0
        self.total_decisions = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos") or []
        durations = np.array(
            [float(info.get("primitive_steps", 1) or 1) for info in infos], dtype=np.float32
        )
        self.total_primitive_steps += int(durations.sum())
        self.total_decisions += len(durations)
        buffer = getattr(self.model, "rollout_buffer", None)
        setter = getattr(buffer, "set_duration", None)
        if setter is not None and buffer.pos > 0:
            setter(buffer.pos - 1, durations)
        return True


class CurveLogger(BaseCallback):
    """Append one row per episode. CSV, because curves must be diffable.

    Accumulators are **per environment**. The first version of this kept one
    running reward and one step count for the whole vector, summed `rewards`
    across all workers, incremented the step count once per *vector* step, and
    flushed a row whenever `dones.any()` fired -- reading `infos[0]` for the
    success flag. Every column was wrong in a way that looked plausible:

    * `episode_reward` was the N-worker sum of reward since the last any-done,
      which for eight workers over a long family lands near the step cost of a
      single full episode. It reads exactly like "the policy never earned
      anything" and is really "eight workers each paid a third of an episode".
    * `episode_steps` counted vector steps, so a 300-step family reported ~30.
    * `success` was **worker 0's flag alone**. A success in workers 1..N-1 was
      recorded as a zero, shrinking the evidentiary weight of a "no successes"
      curve by a factor of N -- and `repair_belt` and `restore_power` were both
      declared unlearnable off curves that had successes in them.

    One row per real episode, attributed to the worker that finished it.
    """

    def __init__(self, path: Path, started: float, num_envs: int = 1) -> None:
        super().__init__()
        self.path = path
        self.started = started
        self.rows: list[dict] = []
        self._num_envs = max(int(num_envs), 1)
        self._episode_reward = [0.0] * self._num_envs
        self._episode_steps = [0] * self._num_envs

    def _on_step(self) -> bool:
        rewards = np.asarray(self.locals.get("rewards", [])).reshape(-1)
        infos = self.locals.get("infos") or []
        dones = np.asarray(self.locals.get("dones", [])).reshape(-1)

        # A vec env may be wider than the count we were built with (SB3 wraps
        # single envs in a DummyVecEnv of one); grow rather than mis-attribute.
        width = max(rewards.size, dones.size, len(infos))
        if width > self._num_envs:
            self._episode_reward.extend([0.0] * (width - self._num_envs))
            self._episode_steps.extend([0] * (width - self._num_envs))
            self._num_envs = width

        for index in range(rewards.size):
            self._episode_reward[index] += float(rewards[index])
        for index in range(max(rewards.size, dones.size)):
            self._episode_steps[index] += 1

        for index in range(dones.size):
            if not bool(dones[index]):
                continue
            info = infos[index] if index < len(infos) else {}
            self.rows.append(
                {
                    "timestep": int(self.num_timesteps),
                    "wall_s": round(time.perf_counter() - self.started, 2),
                    "worker": index,
                    "episode_reward": round(self._episode_reward[index], 4),
                    "episode_steps": self._episode_steps[index],
                    "success": int(bool(info.get("success"))),
                    "layout_family": info.get("layout_family"),
                    "excluded": int(bool(info.get("excluded_from_metrics"))),
                }
            )
            self._episode_reward[index] = 0.0
            self._episode_steps[index] = 0
        return True

    def summary(self) -> dict:
        """Aggregate the curve honestly.

        `final_train_success_rate` used to be the mean over the last twenty
        rows, which reported 24 real successes as 0.0 whenever the last of them
        landed more than twenty episodes before the end. Both figures are kept
        -- the tail rate is what a learning curve is read off -- but the total
        is what decides whether a family ever succeeded at all.
        """
        scored = [row for row in self.rows if not row["excluded"]]
        successes = sum(row["success"] for row in scored)
        tail = scored[-20:]
        return {
            "episodes_logged": len(self.rows),
            "episodes_scored": len(scored),
            "train_successes": successes,
            "train_success_rate": round(successes / len(scored), 4) if scored else 0.0,
            "final_train_success_rate": (
                round(float(np.mean([row["success"] for row in tail])), 4) if tail else 0.0
            ),
            "mean_episode_reward": (
                round(float(np.mean([row["episode_reward"] for row in scored])), 4)
                if scored
                else 0.0
            ),
            "mean_episode_steps": (
                round(float(np.mean([row["episode_steps"] for row in scored])), 2)
                if scored
                else 0.0
            ),
        }

    def write(self) -> None:
        if not self.rows:
            return
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)


def _make_session(manager: WorkerManager, worker_id: str) -> tuple:
    handle = manager.launch(worker_id)
    with RCONClient(handle.spec.rcon_endpoint, timeout=30.0) as client:
        client.lua(f"game.speed = {TRAIN_SPEED} return game.speed")
    session = WorkerSession(handle, timeout=30.0)
    session.status()
    return handle, session


def evaluate(env: FactorioEnv, model, episodes: int, deterministic: bool = True) -> dict:
    """Evaluate on the env's current split. Wilson interval, not a bare rate."""
    successes = 0
    counted = 0
    lengths: list[int] = []
    rewards: list[float] = []
    for _ in range(episodes):
        observation, _ = env.reset()
        total = 0.0
        steps = 0
        while True:
            masks = env.action_masks()
            action, _ = model.predict(observation, action_masks=masks, deterministic=deterministic)
            observation, reward, terminated, truncated, info = env.step(int(action))
            total += reward
            steps += 1
            if terminated or truncated:
                if info.get("excluded_from_metrics"):
                    break
                counted += 1
                successes += int(bool(info.get("success")))
                lengths.append(steps)
                rewards.append(total)
                break
    rate = successes / counted if counted else 0.0
    return {
        "episodes": counted,
        "successes": successes,
        "success_rate": round(rate, 4),
        "wilson_95": _wilson(successes, counted),
        "mean_episode_reward": round(float(np.mean(rewards)), 4) if rewards else 0.0,
        "mean_episode_steps": round(float(np.mean(lengths)), 1) if lengths else 0.0,
    }


def _wilson(successes: int, total: int, z: float = 1.96) -> list[float]:
    """PLAN.md section 3 wants aggregate uncertainty, not a bare point estimate."""
    if total == 0:
        return [0.0, 0.0]
    p = successes / total
    denominator = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    margin = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    return [round(max(0.0, centre - margin), 4), round(min(1.0, centre + margin), 4)]


def evaluate_parallel(
    vec,
    model,
    episodes: int,
    deterministic: bool = True,
    only_indices: set[int] | None = None,
) -> dict:
    """Evaluate across every worker at once.

    Evaluation was the largest single cost in a run -- on a 25,000-step
    `navigate` run it was 430 of 766 wall-clock seconds -- and it ran on one
    worker while the other seven sat idle. It is embarrassingly parallel: the
    episodes are independent by construction.
    """
    observations = vec.reset()
    count = vec.num_envs
    totals = np.zeros(count, dtype=np.float64)
    steps = np.zeros(count, dtype=np.int64)
    successes = 0
    counted = 0
    lengths: list[int] = []
    rewards: list[float] = []
    # `only_indices` turns "the first N episodes to finish" into "these exact N
    # episodes". Without it the scored set is decided by worker scheduling: the
    # last N-1 episodes are started and abandoned, so a fast policy and a slow
    # one score different subsets of the same window and the comparison PLAN
    # section 3 calls paired quietly stops being paired.
    scored: set[int] = set()
    # One row per scored episode. `evaluate_parallel` recorded counts only, so
    # "how many failed" was answerable and "which scenes failed" was not --
    # synthesis section 10's split between a benchmark number and a diagnostic
    # one needs the second, and a rate with no scene identities cannot be
    # turned into a regression case.
    per_scene: list[dict] = []
    unrecovered: list[int] = []
    incomplete: str | None = None
    reset_failures: list[dict] = []
    while counted < episodes:
        # Termination, which this loop previously had no way to reach. Past the
        # frozen bound `_reset_one` handed out real episodes tagged None, which
        # never became eligible, so the loop spun on genuine engine episodes
        # with no timeout at any level. Now the queue is finite: when nothing
        # is owed and the count is short, say so instead of spinning.
        owed = vec.pending_episodes()
        in_flight = {
            episode
            for episode in vec.in_flight_episodes()
            if episode is not None
            and (only_indices is None or episode in only_indices)
            and episode not in scored
        }
        if owed == 0 and not in_flight and counted < episodes:
            attempts = vec.episode_attempts()
            unrecovered = sorted(set(only_indices or ()) - scored)
            incomplete = (
                f"{len(unrecovered)} of {episodes} frozen episodes could not be scored "
                f"after bounded retries (attempts: "
                f"{ {i: attempts.get(i, 0) for i in unrecovered} }); no substitute scene was "
                "evaluated in their place"
            )
            break
        masks = vec.action_masks()
        actions, _ = model.predict(observations, action_masks=masks, deterministic=deterministic)
        try:
            observations, step_rewards, dones, infos = vec.step(actions)
        except EpisodeResetFailed as exc:
            # `_reset_one` already handed the identity back, so the same scene
            # is retried. Persistent failure exhausts the retry limit, empties
            # the queue and lands on the incomplete-coverage branch above --
            # never on a substituted scene, and never scored as an agent loss.
            reset_failures.append({"episode": exc.index, "cause": exc.cause})
            observations = vec.reset()
            continue
        totals += step_rewards
        steps += 1
        for index, done in enumerate(dones):
            if not done:
                continue
            info = infos[index]
            # `index` is the worker slot and indexes the accumulators;
            # `episode` is which scene that worker just played. Naming both
            # `index` read fine and silently indexed an 8-element array with an
            # episode number.
            episode = info.get("episode_index")
            eligible = only_indices is None or (episode in only_indices and episode not in scored)
            # An infrastructure failure is not a task outcome, so it is dropped
            # rather than counted as a loss -- and the scene identity is handed
            # back so the *same* scene is retried. Consuming its index and
            # moving on silently shortened a frozen evaluation, or hung it.
            if info.get("excluded_from_metrics") and episode is not None and eligible:
                vec.release_episode(episode)
            if not info.get("excluded_from_metrics") and eligible and counted < episodes:
                if episode is not None:
                    scored.add(episode)
                counted += 1
                successes += int(bool(info.get("success")))
                lengths.append(int(steps[index]))
                rewards.append(float(totals[index]))
                row = {
                    "episode_index": episode,
                    "layout_family": info.get("layout_family"),
                    "success": bool(info.get("success")),
                    "steps": int(steps[index]),
                    "reward": round(float(totals[index]), 4),
                    # Why it ended, which pass/fail alone does not say: a policy
                    # that ran out of budget and one that hit a failure
                    # predicate are different diagnoses.
                    "terminated": bool(info.get("terminated", not info.get("TimeLimit.truncated"))),
                    "truncated": bool(info.get("TimeLimit.truncated")),
                    "action_error": info.get("action_error"),
                }
                if info.get("production") is not None:
                    row["production"] = info["production"]
                per_scene.append(row)
            totals[index] = 0.0
            steps[index] = 0
    rate = successes / counted if counted else 0.0
    return {
        "episodes": counted,
        "successes": successes,
        "success_rate": round(rate, 4),
        "wilson_95": _wilson(successes, counted),
        "mean_episode_reward": round(float(np.mean(rewards)), 4) if rewards else 0.0,
        "mean_episode_steps": round(float(np.mean(lengths)), 1) if lengths else 0.0,
        "deterministic": deterministic,
        "scored_episodes": sorted(scored) if only_indices is not None else None,
        # Sorted by identity, not by finish order: worker scheduling decides
        # the latter, so two runs over the same frozen set would produce rows
        # in different orders and diff as if the outcomes had changed.
        "per_scene": sorted(
            per_scene, key=lambda r: (r["episode_index"] is None, r["episode_index"])
        ),
        "failed_scenes": sorted(
            r["episode_index"]
            for r in per_scene
            if not r["success"] and r["episode_index"] is not None
        ),
        "incomplete_coverage": incomplete,
        "reset_failures": reset_failures or None,
        "unrecovered_episodes": unrecovered or None,
        "episode_attempts": ({k: v for k, v in vec.episode_attempts().items() if v > 1} or None),
    }


def random_baseline_parallel(vec, episodes: int, rng: np.random.Generator) -> dict:
    """The random floor, across every worker."""
    vec.reset()
    successes = 0
    counted = 0
    while counted < episodes:
        masks = vec.action_masks()
        actions = [int(rng.choice(np.flatnonzero(row))) for row in masks]
        _, _, dones, infos = vec.step(np.array(actions))
        for index, done in enumerate(dones):
            if not done:
                continue
            if not infos[index].get("excluded_from_metrics") and counted < episodes:
                counted += 1
                successes += int(bool(infos[index].get("success")))
    return {
        "episodes": counted,
        "successes": successes,
        "success_rate": round(successes / counted, 4) if counted else 0.0,
        "wilson_95": _wilson(successes, counted),
    }


def random_baseline(env: FactorioEnv, episodes: int, rng: np.random.Generator) -> dict:
    """Uniform over the masked catalog. The floor every curve is read against."""
    successes = 0
    for _ in range(episodes):
        env.reset()
        while True:
            legal = np.flatnonzero(env.action_masks())
            _, _, terminated, truncated, info = env.step(int(rng.choice(legal)))
            if terminated or truncated:
                successes += int(bool(info.get("success")))
                break
    return {
        "episodes": episodes,
        "successes": successes,
        "success_rate": round(successes / episodes, 4),
        "wilson_95": _wilson(successes, episodes),
    }


def _load_frozen_holdout(config: TrainConfig, task) -> dict | None:
    """Read and check the frozen holdout, or return None when none was asked for.

    Every rejection here is a property of the file and the task, so none of it
    needs an engine, a worker or a trained policy -- which is the whole reason
    it runs before any of those exist.

    The task table is nested under ``holdout``, beside ``seed_plan`` and
    ``start_index``. Reading it from the top level instead returned ``None`` for
    every task, so `--holdout` raised "does not cover task" for tasks the file
    plainly covered, and the frozen-holdout path had therefore never once
    executed successfully.
    """
    if not config.holdout:
        return None
    frozen = json.loads(Path(config.holdout).read_text(encoding="utf-8"))
    spec = frozen["holdout"]
    if spec["split"] != "test":
        raise ValueError(
            f"holdout {config.holdout} freezes the {spec['split']!r} split; "
            "the release evaluation is defined on the structural (test) split"
        )
    recorded = (spec.get("tasks") or {}).get(config.task_id)
    if recorded is None:
        covered = sorted((spec.get("tasks") or {}).keys())
        raise ValueError(
            f"holdout {config.holdout} does not cover task {config.task_id!r}; it covers {covered}"
        )
    if recorded.get("task_version") != task.spec.version:
        raise ValueError(
            f"holdout was frozen against {config.task_id} "
            f"v{recorded.get('task_version')} but this run is v{task.spec.version}; "
            "a holdout is only meaningful for the task it was frozen against"
        )
    # The file must hash to what it says it hashes to. Without this the
    # `content_hash` a run records is a copied string rather than a checksum.
    # Cheap, engine-free, and it prevents a guaranteed hang: the evaluation
    # scores exactly the frozen indices, so asking for more episodes than the
    # holdout contains can never be satisfied. `--eval-episodes` defaults to
    # 100 and `episodes_per_task` is 100, so this also documents that there is
    # zero slack -- every excluded episode must be recovered by retry or
    # reported as incomplete coverage.
    per_task = spec.get("episodes_per_task")
    if per_task is not None and config.eval_episodes > per_task:
        raise ValueError(
            f"eval_episodes={config.eval_episodes} exceeds the {per_task} episodes "
            f"holdout {spec.get('holdout_id')} freezes per task; the evaluation could "
            "never reach that count without evaluating scenes outside the frozen set"
        )
    computed = holdout_content_hash(spec)
    if computed != frozen.get("content_hash"):
        raise ValueError(
            f"holdout {config.holdout} records content_hash "
            f"{frozen.get('content_hash')!r} but its content hashes to {computed!r}; "
            "the file has been edited since it was frozen"
        )
    # A whole-file hash covers every task at once, so re-freezing one family
    # changes the hash cited by all of them -- and `holdout_v2` was in fact
    # rewritten five times, each bump silently invalidating the hash recorded
    # by every earlier run while leaving the untouched families' episodes
    # identical. The per-task digest is what a cell should cite: it changes
    # exactly when the episodes that produced the number change.
    frozen = dict(frozen)
    frozen["task_entry_hash"] = holdout_entry_hash(recorded)
    return frozen


def holdout_content_hash(spec: dict) -> str:
    """The hash a frozen holdout file records for itself."""
    canonical = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def holdout_entry_hash(entry: dict) -> str:
    """The digest of one task's frozen episode set.

    Scoped to a single task on purpose: this is the value that identifies the
    scenes a number was measured on, and it must not move when an unrelated
    family is re-frozen.
    """
    canonical = json.dumps(entry, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()[:16]


def train(config: TrainConfig) -> dict:
    started = time.perf_counter()
    run_id = manifest_module.new_run_id(config.run_prefix)
    run_dir = manifest_module.runs_dir() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # The accountant discounts potential-based shaping with its own module
    # constant while the learner discounts with the configured gamma. The
    # shaping is policy-invariant only when the two agree, and nothing tied
    # them together -- a changed gamma would have silently voided the
    # invariance rather than failed.
    seeded = seed_everything(config.master_seed)
    plan = SeedPlan(master=config.master_seed, run_id=run_id)
    task = get(config.task_id)

    # A family may declare its own discount, because the useful horizon is
    # 1/(1-gamma) and it has to cover the episode. At 0.99 that horizon is 100
    # steps: a 300-step family shaping toward a goal pays more drag for
    # approaching than the approach is worth, whatever weight the component
    # carries, and the `w` cancels out of the comparison entirely.
    #
    # Whichever value is used, the learner and the accountant must use the *same*
    # one or the shaping stops being policy-invariant -- which is what this guard
    # has always been for, now checked against the value actually in play.
    gamma = task.spec.gamma or config.gamma
    accountant_gamma = task.spec.gamma or rewards_module.GAMMA
    if abs(accountant_gamma - gamma) > 1e-9:
        raise ValueError(
            f"reward shaping discounts at gamma={accountant_gamma} while the "
            f"learner uses gamma={gamma}; potential-based shaping is only "
            "policy-invariant when they match"
        )

    # Checked here, beside the gamma guard, because everything it can reject is
    # knowable before a single step is taken. It used to be validated after
    # `model.learn` returned, so a one-word mistake in a holdout path cost a
    # full training run to discover: three release cells trained 25,000 steps
    # each, twenty-one minutes apiece, and then raised on a config error.
    frozen = _load_frozen_holdout(config, task)

    manager = WorkerManager()
    # Run-scoped worker id. A fixed "train-<task>" name collides the moment two
    # runs of the same task overlap -- the second dies on the first one's
    # write-data lock, which is the same failure a stale orphan caused in
    # Phase 1.
    handle, session = _make_session(manager, f"train-{config.task_id}-{run_id[-8:]}")
    status = {"run_id": run_id, "state": "running"}
    (run_dir / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")

    vec_env = None
    try:
        single = _wrap(
            FactorioEnv(
                task,
                session,
                plan,
                branch=Branch.TRAIN,
                split="train",
                shaping=config.shaping,
                start_curriculum=config.start_curriculum,
            ),
            config.skills,
        )
        if config.workers > 1:
            from factoriorl.vecenv import FactorioVecEnv

            vec_env = FactorioVecEnv(
                task,
                plan,
                num_workers=config.workers,
                shaping=config.shaping,
                worker_prefix=f"vec-{config.task_id}-{run_id[-8:]}",
                skills=config.skills,
                start_curriculum=config.start_curriculum,
            )
            env = vec_env
        else:
            env = single
        # The catalog is a property of the task, identical across workers, so
        # the manifest reads it from the single env either way.
        catalog_env = single
        # n_steps is *per environment*, so the rollout buffer is
        # n_steps x workers. Left alone, six workers collect six times as much
        # experience per update and therefore take six times fewer gradient
        # steps for the same env budget -- which showed up as a collapse from
        # 1.00 to 0.05 training success at an unchanged step count. Divide it
        # so the buffer, and hence the update count, stays put.
        steps_per_env = max(MIN_STEPS_PER_ENV, config.n_steps // max(config.workers, 1))
        model = MaskablePPO(
            "MultiInputPolicy",
            env,
            policy_kwargs=policy_kwargs(),
            learning_rate=config.learning_rate,
            n_steps=steps_per_env,
            batch_size=min(config.batch_size, steps_per_env * max(config.workers, 1)),
            gamma=gamma,
            ent_coef=config.ent_coef,
            seed=config.master_seed,
            device=config.device,
            # Durations only differ from 1 under skills, and the buffer is
            # exactly stock GAE when they are all 1.
            rollout_buffer_class=DurationAwareMaskableDictRolloutBuffer,
            verbose=0,
        )

        # Written before the first step, so an interrupted run still has one.
        manifest_module.RunManifest(
            run_id=run_id,
            engine=handle.engine.to_dict(),
            task={
                "id": task.spec.id,
                "version": task.spec.version,
                "config_digest": manifest_module.config_digest(task.spec.to_dict()),
                "resolved": task.spec.to_dict(),
            },
            profiles={
                "observation": task.spec.observation_profile,
                "action": task.spec.action_profile,
                # Not a constant. A task whose goal geometry is chosen for it by
                # the environment received an assistance, and a result carrying
                # "none" is indistinguishable from one produced without it.
                "assistance": describe_assistance(task.spec),
                # Which deliberation layer chose the actions. A policy over
                # primitives and a policy over skills produce results that are
                # not comparable, and a published result that does not say
                # which one it used cannot be interpreted later.
                "deliberation": (
                    SKILL_PROFILE if config.skills else task.spec.deliberation_profile
                ),
                "skill_library": [s.key for s in SKILLS] if config.skills else [],
                "goal_encoding": encoders.GOAL_ENCODING_VERSION,
                "catalog": catalog_env.catalog.name,
                "catalog_digest": catalog_env.catalog.digest(),
                "resolved_catalog": list(catalog_env.catalog.keys()),
            },
            reward={
                "shaping_enabled": config.shaping,
                "config_digest": manifest_module.config_digest([r.name for r in task.spec.rewards]),
                "components": [
                    {"name": r.name, "kind": r.kind.value, "weight": r.weight, "shaping": r.shaping}
                    for r in task.spec.rewards
                ],
            },
            seeds={**plan.to_dict(), "seeded": seeded},
            budgets={
                "max_decision_steps": task.spec.max_decision_steps,
                "max_game_ticks": task.spec.max_game_ticks,
                "total_steps": config.total_steps,
            },
            workers=[handle.spec.manifest()],
            model={**describe(model), "extractor_version": EXTRACTOR_VERSION},
            extra={
                "config": config.to_dict(),
                # A citation `freeze_holdout.manifests_citing()` can actually
                # find. It looked for a top-level `holdout` dict with an id and
                # a hash, while runs recorded only `config.holdout` as a path
                # string -- so it returned zero citations for every holdout and
                # `stale_citations()` checked nothing at all.
                **(
                    {
                        "holdout": {
                            "id": frozen["holdout"].get("holdout_id"),
                            "content_hash": frozen["content_hash"],
                            "task_entry_hash": frozen.get("task_entry_hash"),
                            "task": config.task_id,
                            "start_index": frozen["holdout"]["start_index"],
                            "episodes_per_task": frozen["holdout"]["episodes_per_task"],
                            "seed_plan": frozen["holdout"]["seed_plan"],
                        }
                    }
                    if frozen is not None
                    else {}
                ),
                "time_units": {
                    "note": (
                        "policy_decisions is what total_steps and n_steps count. "
                        "A skill spans many primitive steps, so flat and "
                        "skill-augmented arms matched on total_steps are matched "
                        "on decisions and not on environment experience."
                    ),
                    "objective_time_unit": "primitive_step",
                    "trace_decay_per": "primitive_step",
                },
                "rollout": {
                    "steps_per_env": steps_per_env,
                    "buffer": steps_per_env * max(config.workers, 1),
                    "updates": config.total_steps // max(steps_per_env * max(config.workers, 1), 1),
                },
            },
        ).write()

        curve = CurveLogger(run_dir / "curve.csv", started, num_envs=config.workers)
        guard = RolloutGuard()
        durations = DurationRecorder()
        model.learn(
            total_timesteps=config.total_steps,
            callback=[guard, durations, curve],
            progress_bar=False,
        )
        # Order matters: the curve and the checkpoint are written before the
        # abort is raised, so an aborted run still leaves its evidence and its
        # last valid checkpoint on disk rather than only a traceback.
        curve.write()
        model.save(run_dir / "model")
        checkpoint = run_dir / "model.zip"
        if checkpoint.is_file():
            manifest_module.amend(
                run_id,
                {
                    "model": {
                        "checkpoint": checkpoint.name,
                        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                        "checkpoint_bytes": checkpoint.stat().st_size,
                    }
                },
            )
        if guard.aborted is not None:
            (run_dir / "status.json").write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "state": "aborted_infrastructure",
                        "cause": guard.aborted.cause,
                        "timestep": guard.aborted.timestep,
                        "note": (
                            "Collection stopped before the next optimizer update, so no "
                            "gradient step used the contaminated rollout. No success rate is "
                            "reported: an infrastructure fault is not a task outcome."
                        ),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            raise guard.aborted

        # A frozen holdout replaces the *structural* row's seed plan and start
        # index. The other two rows stay on the run's own plan: they are
        # diagnostics for this run, not the published result. Loaded and
        # validated at the top of `train`; see `_load_frozen_holdout`.

        # Three rows, because one number cannot say which failure happened.
        # PLAN section 3 puts the threshold on unfamiliar *structures* and
        # requires the unfamiliar-*seed* rate published beside it: a policy
        # that scores well on seeds and badly on structures learned the task
        # and failed to transfer, while one that fails both never learned it.
        # Every row uses the EVAL seed branch, so no evaluation episode is one
        # the policy trained on, whatever split it came from.
        #
        # Retargeting mutates the training vec env: its workers are pointed at
        # the evaluation branch and another split and are never pointed back.
        # That is only acceptable because evaluation is terminal -- training
        # has finished and the model is already saved -- so nothing may be
        # added below that resumes learning on `vec_env`.
        rows: dict[str, dict] = {}
        # The stream each row was actually scored on. `master_seed` alone does
        # not name the scenes, so two runs sharing it are not replications.
        eval_streams: dict = {}
        for label, split in (
            ("seeds", "train"),
            ("val", "val"),
            ("structures", "test"),
        ):
            if not task.spec.families(split):
                continue
            row_plan, row_start = plan, 0
            row_stop: int | None = None
            frozen_indices: set[int] | None = None
            if frozen is not None and split == "test":
                seeds_spec = frozen["holdout"]["seed_plan"]
                row_plan = SeedPlan(master=seeds_spec["master"], run_id=seeds_spec["run_id"])
                row_start = frozen["holdout"]["start_index"]
                # The frozen set is a closed range, so the cursor stops at its
                # end and the evaluation scores exactly those indices. Without
                # both, a 100-episode holdout could never be covered by a
                # 100-episode evaluation: the workers start more episodes than
                # are scored, so the run touched 1000..1111 for a range ending
                # at 1100 and reported itself out of range -- correctly.
                row_stop = row_start + frozen["holdout"]["episodes_per_task"]
                frozen_indices = set(range(row_start, row_stop))
            if config.workers > 1 and vec_env is not None:
                # Retarget rather than rebuild. `split` and `branch` are plain
                # attributes of `FactorioEnv`, so pointing the live engines at
                # another split is assignment; building a vectorised
                # environment per split would pay the worker launch storm --
                # the most expensive moment in a run -- three times over.
                vec_env.retarget(split, Branch.EVAL, row_start, plan=row_plan, stop_index=row_stop)
                rows[label] = evaluate_parallel(
                    vec_env, model, config.eval_episodes, only_indices=frozen_indices
                )
                rows[label]["episodes_touched"] = vec_env.issued_episodes()
                # The greedy policy is not guaranteed to be at least as good as
                # the one that was trained. On a partially observed task the
                # deterministic memoryless policies are a strictly weaker
                # class, so argmax can score below the stochastic policy that
                # earned the training successes -- and a checkpoint with real
                # successes reading 0.00 under argmax is the textbook symptom.
                # Scored on the *same* episodes, so the pair is a pair.
                if config.paired_stochastic_eval:
                    vec_env.retarget(
                        split, Branch.EVAL, row_start, plan=row_plan, stop_index=row_stop
                    )
                    rows[label]["stochastic"] = evaluate_parallel(
                        vec_env,
                        model,
                        config.eval_episodes,
                        deterministic=False,
                        only_indices=frozen_indices,
                    )
            else:
                # The single-worker path needs the frozen plan and start index
                # as much as the vectorised one. Left on the run's own plan it
                # would cite a holdout and evaluate entirely different scenes,
                # with nothing downstream able to tell -- the exact failure a
                # frozen holdout exists to prevent, reintroduced in the branch
                # nobody looks at.
                serial = _wrap(
                    FactorioEnv(
                        task,
                        session,
                        row_plan,
                        branch=Branch.EVAL,
                        split=split,
                        shaping=config.shaping,
                    ),
                    config.skills,
                )
                serial.unwrapped._episode_index = row_start - 1
                rows[label] = evaluate(serial, model, config.eval_episodes)
            rows[label]["split"] = split

            # Loop variables are bound as defaults, not captured: the closure
            # is called inside this iteration today, but a late-bound
            # row_plan would silently measure the baseline against a different
            # holdout than the policy was scored on.
            def measure_baseline(
                split: str = split, row_plan=row_plan, row_start: int = row_start
            ) -> dict:
                # The baseline shares the evaluation budget: a 10-episode
                # baseline has a Wilson interval so wide that almost no
                # measured rate can clear its upper bound, which turns an
                # underpowered comparison into a false negative.
                rng = np.random.default_rng(config.master_seed)
                if config.workers > 1 and vec_env is not None:
                    # Rewound to episode zero so the floor is measured on the
                    # same scenes the policy just ran, not on whatever the
                    # shared cursor happened to reach.
                    vec_env.retarget(split, Branch.EVAL, row_start, plan=row_plan)
                    return random_baseline_parallel(vec_env, config.eval_episodes, rng)
                floor_env = _wrap(
                    FactorioEnv(task, session, row_plan, branch=Branch.EVAL, split=split),
                    config.skills,
                )
                floor_env.unwrapped._episode_index = row_start - 1
                return random_baseline(floor_env, config.eval_episodes, rng)

            eval_streams[label] = {
                "split": split,
                "branch": Branch.EVAL.value,
                "master": row_plan.master,
                "run_id": row_plan.run_id,
                "start_index": row_start,
                "stop_index": row_stop,
                "deterministic": True,
                "stochastic_arm": bool(config.paired_stochastic_eval),
                "excluded_episodes": rows[label].get("reset_failures") or [],
            }
            rows[label]["random_baseline"] = cached_random_baseline(
                task,
                split,
                config.master_seed,
                config.eval_episodes,
                measure_baseline,
                # The floor belongs to the action space, not just the task: a
                # random policy over skills is a different agent from a random
                # policy over primitives.
                action_space=SKILL_PROFILE if config.skills else task.spec.action_profile,
                # The stream the floor was actually measured on. `master_seed`
                # alone does not name the scenes: an episode's scene comes from
                # blake2b(master | run_id | branch | index), and every frozen
                # holdout here shares master=20260908 while differing only in
                # run_id and start_index. Without these a floor measured on one
                # holdout was served for another whenever the task version had
                # not moved between them.
                seed_run_id=row_plan.run_id,
                start_index=row_start,
            )

        # Coverage, not just citation. If this run claims a frozen holdout, say
        # whether the episodes it actually touched fall inside the frozen range
        # -- otherwise the hash in the manifest is decoration.
        if frozen is not None and "structures" in rows:
            spec = frozen["holdout"]
            low = spec["start_index"]
            high = low + spec["episodes_per_task"]
            touched = rows["structures"].get("episodes_touched") or []
            # Coverage is a claim about the episodes that produced the number,
            # so it is judged on what was *scored*. `episodes_touched` stays in
            # the record because the gap between the two is worth seeing, but
            # it always overshoots: workers start episodes that are abandoned
            # when the count is reached.
            scored = rows["structures"].get("scored_episodes") or []
            rows["structures"]["holdout"] = {
                "id": spec.get("holdout_id"),
                "content_hash": frozen["content_hash"],
                "task_entry_hash": frozen.get("task_entry_hash"),
                "frozen_range": [low, high],
                "episodes_touched": len(touched),
                "episodes_scored": len(scored),
                "within_frozen_range": bool(scored) and all(low <= i < high for i in scored),
                "covers_frozen_set": sorted(scored) == list(range(low, high)),
            }

        # `held_out` stays the acceptance number and remains the structural
        # row; `eval_split` names the split that selection may look at.
        held_out = rows.get("structures") or rows.get(config.eval_split) or {}
        baseline = held_out.get("random_baseline", {})

        result = {
            "run_id": run_id,
            "task": config.task_id,
            "eval_split": config.eval_split,
            "total_steps": config.total_steps,
            "wall_seconds": round(time.perf_counter() - started, 1),
            "steps_per_second": round(
                config.total_steps / max(time.perf_counter() - started, 1e-9), 2
            ),
            "held_out": held_out,
            "random_baseline": baseline,
            "evaluation": rows,
            **curve.summary(),
            # The four units the redirection asks to be reported separately.
            "policy_decisions": durations.total_decisions,
            "primitive_transitions": durations.total_primitive_steps,
            "simulated_ticks": durations.total_primitive_steps * task.spec.decision_ticks,
            "optimizer_updates": model.num_timesteps
            // max(steps_per_env * max(config.workers, 1), 1),
            "objective_time_unit": "primitive_step",
        }
        manifest_module.amend(run_id, {"seeds": {"eval_streams": eval_streams}})
        (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        status = {"run_id": run_id, "state": "completed"}
        return result
    except Exception as exc:  # noqa: BLE001 - a failed run must stay inspectable
        import traceback

        status = {
            "run_id": run_id,
            "state": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        raise
    finally:
        # Never delete a failed run; its directory is the evidence.
        (run_dir / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        if vec_env is not None:
            try:
                vec_env.close()
            except Exception:  # noqa: BLE001 - teardown must not mask a result
                pass
        try:
            session.close()
        except OSError:
            pass
        manager.cleanup(handle)
