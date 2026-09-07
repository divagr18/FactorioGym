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
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

from factoriorl import encoders
from factoriorl import manifest as manifest_module
from factoriorl.env import FactorioEnv
from factoriorl.learn.policy import EXTRACTOR_VERSION, describe, policy_kwargs
from factoriorl.rcon import RCONClient
from factoriorl.seeding import Branch, SeedPlan, seed_everything
from factoriorl.session import WorkerSession
from factoriorl.tasks import get
from factoriorl.worker import WorkerManager

#: Also sets the *paused* server loop rate, which is what RCON latency
#: actually depends on: 16.6 ms per round trip at speed 1, 1.5 ms at 60.
TRAIN_SPEED = 60.0
#: Floor for the per-env rollout length, so a large worker count cannot
#: shrink it to something PPO cannot learn from.
MIN_STEPS_PER_ENV = 32


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
    device: str = "auto"
    #: Parallel workers. A 30-tick interval costs ~16 ms of engine time that no
    #: setting can remove, so overlapping workers is the only way to step
    #: faster: 41.8 steps/s at one, 202 at eight.
    workers: int = 1
    run_prefix: str = "train"
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "eval_split": self.eval_split,
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
        }


class CurveLogger(BaseCallback):
    """Append one row per episode. CSV, because curves must be diffable."""

    def __init__(self, path: Path, started: float) -> None:
        super().__init__()
        self.path = path
        self.started = started
        self.rows: list[dict] = []
        self._episode_reward = 0.0
        self._episode_steps = 0

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        infos = self.locals.get("infos") or []
        dones = self.locals.get("dones")
        if rewards is not None:
            self._episode_reward += float(np.asarray(rewards).sum())
        self._episode_steps += 1
        if dones is not None and bool(np.asarray(dones).any()):
            info = infos[0] if infos else {}
            self.rows.append(
                {
                    "timestep": int(self.num_timesteps),
                    "wall_s": round(time.perf_counter() - self.started, 2),
                    "episode_reward": round(self._episode_reward, 4),
                    "episode_steps": self._episode_steps,
                    "success": int(bool(info.get("success"))),
                    "layout_family": info.get("layout_family"),
                    "excluded": int(bool(info.get("excluded_from_metrics"))),
                }
            )
            self._episode_reward = 0.0
            self._episode_steps = 0
        return True

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


def train(config: TrainConfig) -> dict:
    started = time.perf_counter()
    run_id = manifest_module.new_run_id(config.run_prefix)
    run_dir = manifest_module.runs_dir() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    seeded = seed_everything(config.master_seed)
    plan = SeedPlan(master=config.master_seed, run_id=run_id)
    task = get(config.task_id)

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
        single = FactorioEnv(
            task, session, plan, branch=Branch.TRAIN, split="train", shaping=config.shaping
        )
        if config.workers > 1:
            from factoriorl.vecenv import FactorioVecEnv

            vec_env = FactorioVecEnv(
                task,
                plan,
                num_workers=config.workers,
                shaping=config.shaping,
                worker_prefix=f"vec-{config.task_id}-{run_id[-8:]}",
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
            gamma=config.gamma,
            ent_coef=config.ent_coef,
            seed=config.master_seed,
            device=config.device,
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
                "assistance": "none",
                # Which deliberation layer chose the actions. A policy over
                # primitives and a policy over skills produce results that are
                # not comparable, and a published result that does not say
                # which one it used cannot be interpreted later.
                "deliberation": task.spec.deliberation_profile,
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
                "rollout": {
                    "steps_per_env": steps_per_env,
                    "buffer": steps_per_env * max(config.workers, 1),
                    "updates": config.total_steps // max(steps_per_env * max(config.workers, 1), 1),
                },
            },
        ).write()

        curve = CurveLogger(run_dir / "curve.csv", started)
        model.learn(total_timesteps=config.total_steps, callback=curve, progress_bar=False)
        curve.write()
        model.save(run_dir / "model")

        # Three rows, because one number cannot say which failure happened.
        # PLAN section 3 puts the threshold on unfamiliar *structures* and
        # requires the unfamiliar-*seed* rate published beside it: a policy
        # that scores well on seeds and badly on structures learned the task
        # and failed to transfer, while one that fails both never learned it.
        # Every row uses the EVAL seed branch, so no evaluation episode is one
        # the policy trained on, whatever split it came from.
        rows: dict[str, dict] = {}
        for label, split in (
            ("seeds", "train"),
            ("val", "val"),
            ("structures", "test"),
        ):
            if not task.spec.families(split):
                continue
            rows[label] = evaluate(
                FactorioEnv(
                    task,
                    session,
                    plan,
                    branch=Branch.EVAL,
                    split=split,
                    shaping=config.shaping,
                ),
                model,
                config.eval_episodes,
            )
            rows[label]["split"] = split
            rows[label]["random_baseline"] = random_baseline(
                FactorioEnv(task, session, plan, branch=Branch.EVAL, split=split),
                # The baseline shares the evaluation budget: a 10-episode
                # baseline has a Wilson interval so wide that almost no
                # measured rate can clear its upper bound, which turns an
                # underpowered comparison into a false negative.
                config.eval_episodes,
                np.random.default_rng(config.master_seed),
            )

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
            "episodes_logged": len(curve.rows),
            "final_train_success_rate": round(
                float(np.mean([r["success"] for r in curve.rows[-20:]])), 4
            )
            if curve.rows
            else 0.0,
        }
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
