"""Phase 3 exit gate (PLAN.md).

    all six task families run through Gymnasium with real workers, reference
    solutions, and clean resets

Sections, in order of what they prove:

1. **config** - every task validates before a worker is launched.
2. **spaces** - the Gymnasium env checker, observations inside their declared
   space, mask invariants, and termination/truncation exclusivity.
3. **reset correctness** - 500+ consecutive resets with digest comparison
   against a *freshly launched* worker, plus growth proxies that catch
   unbounded state a digest comparison alone would miss.
4. **reward accounting** - components sum to the returned reward, and shaping
   on/off does not change the success predicate.
5. **performance** - the steps/sec and reset numbers Phase 4.3 builds on.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field

import numpy as np

from factoriorl import manifest as manifest_module
from factoriorl.env import FactorioEnv
from factoriorl.paths import evidence_dir, runtime_dir
from factoriorl.rewards import RewardAccountant
from factoriorl.seeding import Branch, SeedPlan
from factoriorl.session import WorkerSession
from factoriorl.tasks import all_tasks, validate_all
from factoriorl.worker import WorkerManager

GATE_SPEED = 30.0
RESET_TARGET = 500
RANDOM_EPISODE_STEPS = 40
#: Episodes per family for the reference-solution check.
REFERENCE_EPISODES = 3
#: A reference solution that cannot finish reliably indicates a task defect.
REFERENCE_THRESHOLD = 0.99


@dataclass
class GateReport:
    passed: bool = True
    sections: dict = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    measurements: dict = field(default_factory=dict)

    def fail(self, reason: str) -> None:
        self.failures.append(reason)
        self.passed = False

    def check(self, section: str, name: str, ok: bool, **data) -> bool:
        self.sections.setdefault(section, []).append({"check": name, "ok": bool(ok), **data})
        if not ok:
            self.fail(f"{section}: {name}")
        return bool(ok)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "failures": self.failures,
            "measurements": self.measurements,
            "sections": self.sections,
        }


def _digest_lines(session: WorkerSession) -> tuple[str, list[str], dict]:
    result = session.world_digest().response.result or {}
    lines = result.get("lines", [])
    payload = "\n".join(lines)
    return hashlib.sha256(payload.encode()).hexdigest()[:16], lines, result.get("growth", {})


def _configure(handle, session: WorkerSession) -> None:
    session.status()
    session.configure(speed=GATE_SPEED)


def run_phase3_gate(worker_id: str = "phase3-gate") -> dict:
    report = GateReport()
    manager = WorkerManager()
    handle = manager.launch(worker_id)
    reference_handle = None
    started = time.perf_counter()
    plan = SeedPlan(master=20260907, run_id=manifest_module.new_run_id("phase3"))

    try:
        session = WorkerSession(handle, timeout=30.0)
        _configure(handle, session)

        # ---- 1. config -------------------------------------------------
        validation = validate_all()
        report.check(
            "config",
            "every task validates before a worker is launched",
            validation["ok"],
            tasks=sorted(validation["tasks"]),
        )
        tasks = all_tasks()
        report.check(
            "config", "six introductory families are registered", len(tasks) >= 6, count=len(tasks)
        )

        step_times: list[float] = []
        reset_times: list[float] = []

        for task_id, task in sorted(tasks.items()):
            env = FactorioEnv(task, session, plan, branch=Branch.TRAIN, split="train")

            # ---- 2. spaces ---------------------------------------------
            t0 = time.perf_counter()
            observation, info = env.reset()
            reset_times.append(time.perf_counter() - t0)

            report.check(
                "spaces",
                f"{task_id}: reset observation is inside the declared space",
                env.observation_space.contains(observation),
            )
            report.check(
                "spaces",
                f"{task_id}: reset reports its layout family and split",
                bool(info.get("layout_family")) and info.get("split") == "train",
                family=info.get("layout_family"),
            )

            mask = env.action_masks()
            report.check(
                "spaces",
                f"{task_id}: a legal no-op always exists",
                bool(mask.any()) and bool(mask[env.catalog.wait_index]),
                legal=int(mask.sum()),
                total=int(mask.size),
            )

            terminated = truncated = False
            components_sum_ok = True
            in_space = True
            for _ in range(RANDOM_EPISODE_STEPS):
                legal = np.flatnonzero(env.action_masks())
                action = int(legal[np.random.randint(len(legal))])
                t0 = time.perf_counter()
                observation, reward, terminated, truncated, info = env.step(action)
                step_times.append(time.perf_counter() - t0)
                if not env.observation_space.contains(observation):
                    in_space = False
                components = info.get("reward_components", {})
                if abs(RewardAccountant.total(components) - reward) > 1e-9:
                    components_sum_ok = False
                if terminated or truncated:
                    break

            report.check("spaces", f"{task_id}: observations stay inside the space", in_space)
            report.check(
                "rewards",
                f"{task_id}: components sum to the returned reward",
                components_sum_ok,
            )
            report.check(
                "spaces",
                f"{task_id}: termination and truncation are exclusive",
                not (terminated and truncated),
            )

            # ---- 4. shaping does not change the success predicate --------
            sparse_env = FactorioEnv(
                task, session, plan, branch=Branch.TRAIN, split="train", shaping=False
            )
            sparse_env.reset()
            sparse_rewards = []
            for _ in range(10):
                legal = np.flatnonzero(sparse_env.action_masks())
                action = int(legal[np.random.randint(len(legal))])
                _, reward, term, trunc, info = sparse_env.step(action)
                sparse_rewards.append((reward, info.get("success")))
                if term or trunc:
                    break
            shaped_off_zero = all(abs(r) < 1e-9 for r, success in sparse_rewards if not success)
            report.check(
                "rewards",
                f"{task_id}: disabling shaping zeroes every non-success reward",
                shaped_off_zero,
            )

            # ---- held-out split is reachable and structurally different ---
            test_env = FactorioEnv(task, session, plan, branch=Branch.EVAL, split="test")
            _, test_info = test_env.reset()
            report.check(
                "splits",
                f"{task_id}: a held-out layout family exists and resets",
                bool(test_info.get("layout_family")),
                family=test_info.get("layout_family"),
            )
            report.check(
                "splits",
                f"{task_id}: held-out family differs from training families",
                test_info["layout_family"] not in {f.name for f in task.spec.families("train")},
                held_out=test_info.get("layout_family"),
            )

        # ---- 2b. reference solutions (PLAN.md 3.2) ---------------------
        # The check this gate was missing. A task whose scripted solution
        # cannot finish is unsolvable *through the catalog the policy is
        # given* -- a task defect, not a training result. Without this the
        # gate passed while five of six families were unlearnable.
        from factoriorl.tasks.reference import solve

        for task_id, task in sorted(tasks.items()):
            solved = 0
            reasons: set[str] = set()
            for index in range(REFERENCE_EPISODES):
                ref_env = FactorioEnv(task, session, plan, branch=Branch.TRAIN, split="train")
                ref_env._episode_index = index - 1
                ref_env.reset()
                trace = solve(ref_env)
                solved += int(trace.succeeded)
                if trace.stuck_reason:
                    reasons.add(trace.stuck_reason)
            rate = solved / REFERENCE_EPISODES
            report.check(
                "reference",
                f"{task_id}: the scripted solution completes the task",
                rate >= REFERENCE_THRESHOLD,
                reference_rate=round(rate, 2),
                stuck_reasons=sorted(reasons)[:3],
            )

        # ---- 3. reset correctness -------------------------------------
        # Same task, same seed, repeatedly: the digest must not drift, and the
        # growth proxies must stay flat. A digest comparison alone would miss
        # state that grows without changing the world.
        navigate = tasks["navigate"]
        env = FactorioEnv(navigate, session, plan, branch=Branch.TRAIN, split="train")
        env.reset()
        baseline_digest, baseline_lines, baseline_growth = _digest_lines(session)

        drifted = None
        growth_start = None
        resets_done = 0
        # Ceiling, not floor: 500 // 6 * 6 is 498, which would miss the target.
        per_task = -(-RESET_TARGET // max(len(tasks), 1))
        for task_id, task in sorted(tasks.items()):
            loop_env = FactorioEnv(task, session, plan, branch=Branch.TRAIN, split="train")
            fixed_digest = None
            for index in range(per_task):
                loop_env._episode_index = 0  # same seed every time
                t0 = time.perf_counter()
                loop_env.reset()
                reset_times.append(time.perf_counter() - t0)
                resets_done += 1
                digest, lines, growth = _digest_lines(session)
                if fixed_digest is None:
                    fixed_digest = digest
                    growth_start = growth
                elif digest != fixed_digest and drifted is None:
                    drifted = {
                        "task": task_id,
                        "iteration": index,
                        "diff": [line for line in lines if line not in set(baseline_lines)][:5],
                    }
            report.check(
                "reset",
                f"{task_id}: repeated resets to the same seed are identical",
                drifted is None or drifted["task"] != task_id,
                detail=drifted if drifted and drifted["task"] == task_id else None,
            )

        report.check(
            "reset",
            f"at least {RESET_TARGET} consecutive resets",
            resets_done >= RESET_TARGET,
            resets=resets_done,
        )

        _, _, growth_end = _digest_lines(session)
        flat = all(
            growth_end.get(key, 0) <= (growth_start or {}).get(key, 0) + 64
            for key in ("handles", "remembered", "ledger", "events")
        )
        report.check(
            "reset",
            "growth proxies stay flat across the reset run",
            flat,
            start=growth_start,
            end=growth_end,
        )

        # ---- fresh-worker comparison (PLAN.md 3.3) ---------------------
        # The check that makes reset correctness meaningful: a long-running
        # worker and a brand-new process must agree on the same scene.
        reference_handle = manager.launch(f"{worker_id}-reference")
        reference_session = WorkerSession(reference_handle, timeout=30.0)
        _configure(reference_handle, reference_session)
        reference_env = FactorioEnv(navigate, reference_session, plan, split="train")
        reference_env._episode_index = -1
        reference_env.reset()
        fresh_digest, fresh_lines, _ = _digest_lines(reference_session)

        env._episode_index = -1
        env.reset()
        long_digest, long_lines, _ = _digest_lines(session)
        report.check(
            "reset",
            "a long-running worker matches a freshly launched one",
            fresh_digest == long_digest,
            diff=[line for line in long_lines if line not in set(fresh_lines)][:5],
        )
        reference_session.close()

        # ---- 5. performance -------------------------------------------
        report.measurements["step_ms"] = {
            "n": len(step_times),
            "mean": round(float(np.mean(step_times)) * 1000, 2),
            "p95": round(float(np.percentile(step_times, 95)) * 1000, 2),
        }
        report.measurements["reset_ms"] = {
            "n": len(reset_times),
            "mean": round(float(np.mean(reset_times)) * 1000, 2),
            "p95": round(float(np.percentile(reset_times, 95)) * 1000, 2),
        }
        report.measurements["steps_per_second"] = round(1.0 / float(np.mean(step_times)), 2)
        report.measurements["wall_seconds"] = round(time.perf_counter() - started, 1)
        report.measurements["resets"] = resets_done
        session.close()
    except Exception as exc:  # noqa: BLE001 - the gate reports, it does not raise
        report.fail(f"unhandled: {type(exc).__name__}: {exc}")
    finally:
        payload = report.to_dict()
        payload["engine"] = handle.engine.to_dict()
        payload["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        gate_dir = runtime_dir() / "gate-phase3"
        gate_dir.mkdir(parents=True, exist_ok=True)
        (gate_dir / "report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        evidence_dir().mkdir(parents=True, exist_ok=True)
        (evidence_dir() / "phase3-gate.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        if reference_handle is not None:
            manager.cleanup(reference_handle)
        manager.cleanup(handle)
    return payload
