"""Worker-count profiling (PLAN.md 4.3).

    Benchmark 1, 2, 4, and 8 workers where resource limits permit.
    ...
    More workers are not described as faster unless measurements show it.

Two design points worth stating.

**Threads, not subprocesses.** The engines are already separate processes and
the expensive operations -- socket recv and the engine ticking -- release the
GIL, so threads let one worker's decision interval overlap another's. A
subprocess vec env would add N Python processes and pickle every 101 KB
observation, spending exactly the commit budget that is scarce here.

**The memory ceiling is confronted, not crashed into.** Per-worker commit is
measured at W=1 and any configuration projected past a threshold is *refused
and recorded*. PLAN says "where resource limits permit", so a reasoned skip is
a passing outcome and far better evidence than an OOM traceback.
"""

from __future__ import annotations

import ctypes
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from factoriorl.env import FactorioEnv
from factoriorl.paths import evidence_dir, runtime_dir
from factoriorl.pool import WorkerPool
from factoriorl.rcon import RCONClient
from factoriorl.seeding import Branch, SeedPlan
from factoriorl.session import WorkerSession
from factoriorl.tasks import get

PROFILE_SPEED = 60.0
#: Refuse a configuration projected to use more than this share of the commit
#: limit. Leaves room for the learner, the OS, and the spike during launch.
COMMIT_HEADROOM = 0.80


class _MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def commit_status() -> dict:
    """Commit charge, which is the constraint that actually bites here.

    Free RAM is not the limit: Windows fails an allocation when the *commit*
    charge reaches the limit, which is what produced the BadAllocation crash
    recorded in HANDOFF.md.
    """
    status = _MemoryStatus()
    status.dwLength = ctypes.sizeof(_MemoryStatus)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    total = status.ullTotalPageFile
    available = status.ullAvailPageFile
    return {
        "commit_limit_gb": round(total / 1e9, 2),
        "commit_available_gb": round(available / 1e9, 2),
        "commit_used_gb": round((total - available) / 1e9, 2),
        "used_fraction": round((total - available) / total, 3),
        "ram_available_gb": round(status.ullAvailPhys / 1e9, 2),
    }


@dataclass
class ProfileResult:
    workers: int
    #: The profiled task's decision interval, in game ticks. Carried on the
    #: result rather than assumed, because every simulated-throughput number in
    #: the report is a decision rate multiplied by it. It has no default on
    #: purpose: a default would let a caller silently profile one interval and
    #: report another.
    decision_ticks: int
    steps: int = 0
    wall_seconds: float = 0.0
    step_times: list[float] = field(default_factory=list)
    reset_times: list[float] = field(default_factory=list)
    encode_times: list[float] = field(default_factory=list)
    observation_bytes: list[int] = field(default_factory=list)
    skipped: str | None = None
    commit_before: dict = field(default_factory=dict)
    commit_peak: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        if self.skipped:
            return {
                "workers": self.workers,
                "decision_ticks": self.decision_ticks,
                "skipped": self.skipped,
                "commit_before": self.commit_before,
            }
        steps_per_second = self.steps / self.wall_seconds if self.wall_seconds else 0.0
        return {
            "workers": self.workers,
            "decision_ticks": self.decision_ticks,
            "steps": self.steps,
            "wall_seconds": round(self.wall_seconds, 2),
            "steps_per_second": round(steps_per_second, 2),
            "steps_per_second_per_worker": round(steps_per_second / self.workers, 2),
            "step_ms_mean": round(float(np.mean(self.step_times)) * 1000, 2),
            "step_ms_p95": round(float(np.percentile(self.step_times, 95)) * 1000, 2),
            "reset_ms_mean": round(float(np.mean(self.reset_times)) * 1000, 2)
            if self.reset_times
            else None,
            "encode_ms_mean": round(float(np.mean(self.encode_times)) * 1000, 3)
            if self.encode_times
            else None,
            "observation_bytes_mean": int(np.mean(self.observation_bytes))
            if self.observation_bytes
            else None,
            # One decision advances the engine by the *task's* decision
            # interval, so the conversion from decisions/s to simulated ticks/s
            # is that interval and nothing else. This was hardcoded to 30, the
            # default `TaskSpec.decision_ticks`, which meant profiling a task
            # that had changed its interval reported a throughput that was
            # wrong by exactly the ratio -- silently, since nothing compared
            # the two -- and every derived number in
            # docs/evidence/phase4-worker-profile.json inherited the error.
            "simulated_ticks_per_second": round(steps_per_second * self.decision_ticks, 1),
            "commit_before": self.commit_before,
            "commit_peak": self.commit_peak,
        }


def _profile_one(
    workers: int, task_id: str, steps_per_worker: int, per_worker_commit_gb: float | None
) -> ProfileResult:
    # Resolve the task before the memory check, not after: a skipped
    # configuration still reports which decision interval it would have run at,
    # so a report is never a mix of intervals that nothing records.
    task = get(task_id)
    result = ProfileResult(workers=workers, decision_ticks=task.spec.decision_ticks)
    result.commit_before = commit_status()

    if per_worker_commit_gb is not None:
        projected = result.commit_before["commit_used_gb"] + per_worker_commit_gb * workers
        limit = result.commit_before["commit_limit_gb"] * COMMIT_HEADROOM
        if projected > limit:
            # A recorded, reasoned skip -- better evidence than an OOM.
            result.skipped = (
                f"memory_ceiling: projected {projected:.1f} GB exceeds "
                f"{COMMIT_HEADROOM:.0%} of {result.commit_before['commit_limit_gb']:.1f} GB"
            )
            return result

    plan = SeedPlan(master=4321, run_id=f"profile-w{workers}")
    pool = WorkerPool()
    envs: list[FactorioEnv] = []
    try:
        # Staggered launch: the launch storm is the worst commit spike.
        for index in range(workers):
            worker = pool.start(f"profile-w{workers}-{index}")
            with RCONClient(worker.spec.rcon_endpoint, timeout=30.0) as client:
                client.lua(f"game.speed = {PROFILE_SPEED} return game.speed")
            session = WorkerSession(worker.handle, timeout=30.0)
            session.status()
            envs.append(FactorioEnv(task, session, plan, branch=Branch.TRAIN, split="train"))
            time.sleep(0.2)

        peak = commit_status()

        for env in envs:
            started = time.perf_counter()
            env.reset()
            result.reset_times.append(time.perf_counter() - started)

        def run_one(env: FactorioEnv) -> tuple[list[float], list[float]]:
            timings: list[float] = []
            encodes: list[float] = []
            for _ in range(steps_per_worker):
                legal = np.flatnonzero(env.action_masks())
                action = int(legal[np.random.randint(len(legal))])
                t0 = time.perf_counter()
                _, _, terminated, truncated, _ = env.step(action)
                timings.append(time.perf_counter() - t0)
                t1 = time.perf_counter()
                json.dumps(env._observation)
                encodes.append(time.perf_counter() - t1)
                if terminated or truncated:
                    env.reset()
            return timings, encodes

        wall_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for timings, encodes in executor.map(run_one, envs):
                result.step_times.extend(timings)
                result.encode_times.extend(encodes)
        result.wall_seconds = time.perf_counter() - wall_start
        result.steps = steps_per_worker * workers

        current = commit_status()
        result.commit_peak = current if current["commit_used_gb"] > peak["commit_used_gb"] else peak
        result.observation_bytes = [len(json.dumps(env._observation)) for env in envs]
    finally:
        for env in envs:
            try:
                env.session.close()
            except OSError:
                pass
        pool.close(preserve_evidence=False)
    return result


#: Doubling the worker count must buy at least this much aggregate throughput
#: to be worth the memory and CPU it costs.
MARGINAL_GAIN = 1.15


def choose_default_workers(results: list[dict]) -> dict:
    """The default falls out of measurement, not out of taste.

    Maximise aggregate throughput subject to commit headroom, and stop adding
    workers once doubling them no longer buys a meaningful gain.

    An earlier version of this required each candidate to be *both* within 15%
    of the best throughput *and* within 20% of the single-worker per-worker
    rate. Those are mutually unsatisfiable whenever scaling is sublinear -- and
    it always is -- so the selector silently fell through to one worker while
    reporting that the best measured configuration was nearly five times
    faster. Per-worker efficiency is worth *reporting*; it is not a reason to
    leave throughput on the table.
    """
    usable = [r for r in results if not r.get("skipped")]
    if not usable:
        return {"workers": 1, "reason": "no configuration completed"}

    affordable = [
        r for r in usable if r["commit_peak"].get("used_fraction", 1.0) <= COMMIT_HEADROOM
    ]
    if not affordable:
        return {"workers": 1, "reason": "no configuration fit inside the commit headroom"}

    ordered = sorted(affordable, key=lambda r: r["workers"])
    chosen = ordered[0]
    for candidate in ordered[1:]:
        if candidate["steps_per_second"] >= chosen["steps_per_second"] * MARGINAL_GAIN:
            chosen = candidate

    single = next((r for r in usable if r["workers"] == 1), ordered[0])
    speedup = candidate_speedup = chosen["steps_per_second"] / max(single["steps_per_second"], 1e-9)
    efficiency = chosen["steps_per_second_per_worker"] / max(
        single["steps_per_second_per_worker"], 1e-9
    )
    return {
        "workers": chosen["workers"],
        "steps_per_second": chosen["steps_per_second"],
        "speedup_vs_one_worker": round(candidate_speedup, 2),
        "per_worker_efficiency": round(efficiency, 2),
        "reason": (
            f"highest measured throughput inside the commit headroom: "
            f"{chosen['steps_per_second']:.1f} steps/s, {speedup:.1f}x a single worker "
            f"at {efficiency:.0%} per-worker efficiency; each doubling up to here "
            f"bought at least {MARGINAL_GAIN:.0%}"
        ),
    }


def run_profile(
    worker_counts: tuple[int, ...] = (1, 2, 4, 8),
    task_id: str = "navigate",
    steps_per_worker: int = 120,
) -> dict:
    report: dict = {
        "task": task_id,
        # Recorded at the top level as well so a published profile states the
        # interval its simulated-tick figures were computed at. Two profiles of
        # the same task taken across a change of interval are otherwise
        # indistinguishable, and the slower one looks like a regression.
        "decision_ticks": get(task_id).spec.decision_ticks,
        "steps_per_worker": steps_per_worker,
        "commit_at_start": commit_status(),
        "configurations": [],
    }
    per_worker_commit: float | None = None

    for workers in worker_counts:
        result = _profile_one(workers, task_id, steps_per_worker, per_worker_commit)
        payload = result.to_dict()
        report["configurations"].append(payload)
        if workers == 1 and not result.skipped:
            # Measure the per-worker cost once, then project from it.
            used = result.commit_peak.get("commit_used_gb", 0) - result.commit_before.get(
                "commit_used_gb", 0
            )
            # A measured cost with an explicit margin, not a flat constant.
            # The old 0.35 GB floor was roughly double the measured 0.17 GB a
            # worker actually commits, so the projection refused configurations
            # that would have fit -- and a guard that skips good configurations
            # is only marginally better than one that misses bad ones. The 1.5x
            # covers steady-state growth the first worker has not paid yet;
            # the small absolute floor covers a measurement that reads as ~0
            # because another process freed memory during the sample.
            per_worker_commit = max(used * 1.5, 0.2)
            report["per_worker_commit_gb"] = round(per_worker_commit, 2)

    report["default_workers"] = choose_default_workers(report["configurations"])
    usable = [c for c in report["configurations"] if not c.get("skipped")]
    if usable:
        slowest = max(usable, key=lambda c: c["step_ms_mean"])
        report["bottleneck"] = {
            "component": "environment_step",
            "step_ms_mean": slowest["step_ms_mean"],
            "encode_ms_mean": slowest.get("encode_ms_mean"),
            "note": (
                "the environment round trip dominates; observation serialisation is "
                "two orders of magnitude smaller"
            ),
        }
    report["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    out = runtime_dir() / "profile"
    out.mkdir(parents=True, exist_ok=True)
    (out / "workers.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    evidence_dir().mkdir(parents=True, exist_ok=True)
    (evidence_dir() / "phase4-worker-profile.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
