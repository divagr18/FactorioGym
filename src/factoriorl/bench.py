"""Transport measurement (PLAN.md 11.4: every optimization has a before/after).

This is the measurement that Phase T's optimizations are chosen and judged by.
It answers four questions with numbers rather than reasoning:

1. **Does a zero-output console command produce a response packet?** The
   sentinel framing fix depends on it. A command with no ``rcon.print`` should
   return an empty body promptly.
2. **How much of a normal request is the continuation stall?** An empty
   response short-circuits the continuation loop (``rcon.py``), while a
   non-empty one always runs it out to ``CONTINUATION_TIMEOUT``. The difference
   between the two, measured on the same connection, *is* the stall.
3. **What does a 30-tick advance actually cost in wall time?** ``_settle``
   reports only its first dispatch plus its last probe, so the committed
   figures understate it.
4. **Does ``game.speed`` remove the tick floor, and does it change outcomes?**
   A headless server targets 60 UPS, so 30 ticks is ~500 ms at speed 1.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

from factoriorl.paths import runtime_dir
from factoriorl.rcon import RCONClient
from factoriorl.session import WorkerSession
from factoriorl.worker import WorkerManager

DEFAULT_REPS = 15
DEFAULT_SPEEDS = (1.0, 30.0)
ADVANCE_TICKS = 30


@dataclass
class Timing:
    """Summary of a repeated measurement, in milliseconds."""

    label: str
    samples: list[float] = field(default_factory=list)

    def add(self, seconds: float) -> None:
        self.samples.append(seconds * 1000.0)

    def to_dict(self) -> dict[str, Any]:
        if not self.samples:
            return {"label": self.label, "n": 0}
        ordered = sorted(self.samples)
        return {
            "label": self.label,
            "n": len(ordered),
            "mean_ms": round(statistics.fmean(ordered), 2),
            "p50_ms": round(ordered[len(ordered) // 2], 2),
            "p95_ms": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], 2),
            "min_ms": round(ordered[0], 2),
            "max_ms": round(ordered[-1], 2),
        }


def _time_calls(label: str, reps: int, call) -> Timing:
    timing = Timing(label)
    for _ in range(reps):
        start = time.perf_counter()
        call()
        timing.add(time.perf_counter() - start)
    return timing


def _probe_zero_output(client: RCONClient, reps: int) -> tuple[Timing, dict[str, Any]]:
    """Does a command that prints nothing answer, and how fast?

    ``command`` reads the first response packet unconditionally, so if the
    server sent nothing this would block until the socket timeout. Returning
    promptly with an empty body is the evidence the sentinel design needs.
    """
    body = "/c local _ = 1"
    first = client.command(body)
    verdict = {
        "command": body,
        "responded": True,
        "body_empty": first == "",
        "body_repr": repr(first[:80]),
    }
    return _time_calls("zero_output_command", reps, lambda: client.command(body)), verdict


def run_transport_bench(
    worker_id: str = "bench-transport",
    reps: int = DEFAULT_REPS,
    speeds: tuple[float, ...] = DEFAULT_SPEEDS,
) -> dict[str, Any]:
    manager = WorkerManager()
    handle = manager.launch(worker_id)
    report: dict[str, Any] = {
        "engine": handle.engine.to_dict(),
        "worker": handle.spec.manifest(),
        "reps": reps,
        "measurements": {},
        "advance": {},
        "notes": [],
    }
    try:
        client = RCONClient(handle.spec.rcon_endpoint, timeout=20.0)
        client.connect()
        try:
            # 1 + 2: the stall, isolated on one connection.
            zero_timing, zero_verdict = _probe_zero_output(client, reps)
            report["zero_output_probe"] = zero_verdict
            report["measurements"]["zero_output_command"] = zero_timing.to_dict()

            printing = '/c rcon.print("x")'
            printing_timing = _time_calls(
                "printing_command", reps, lambda: client.command(printing)
            )
            report["measurements"]["printing_command"] = printing_timing.to_dict()

            empty_mean = statistics.fmean(zero_timing.samples)
            print_mean = statistics.fmean(printing_timing.samples)
            report["stall_ms"] = round(print_mean - empty_mean, 2)
            report["real_transport_ms"] = round(empty_mean, 2)
            report["notes"].append(
                "stall_ms is the continuation-loop cost: the same connection answering an "
                "empty body (short-circuits the loop) versus a non-empty one (runs it out)."
            )
        finally:
            client.close()

        # 3 + 4: true advance wall time, per game.speed.
        with WorkerSession(handle, timeout=20.0) as session:
            session.status()
            for speed in speeds:
                set_speed(handle, speed)
                session.reset()
                timing = _time_calls(
                    f"advance_{ADVANCE_TICKS}t_speed_{speed:g}",
                    max(3, reps // 3),
                    lambda: session.advance(ADVANCE_TICKS),
                )
                observed = session.observe().response.result
                report["advance"][f"speed_{speed:g}"] = {
                    "ticks": ADVANCE_TICKS,
                    "wall": timing.to_dict(),
                    "absolute_tick_after": observed.get("absolute_tick"),
                }
            set_speed(handle, 1.0)
    finally:
        report["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        out_dir = runtime_dir() / "bench"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "transport.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        manager.cleanup(handle)
    return report


def set_speed(handle, speed: float) -> float:
    """Set ``game.speed`` through the evaluator channel.

    Evaluator-only for now; Phase 2 gives it a typed ``configure`` request.
    Ticks are identical at any speed -- only real-time pacing changes -- which
    the Phase 0 gate proves by comparing cycle records across speeds.
    """
    with RCONClient(handle.spec.rcon_endpoint, timeout=20.0) as client:
        result = client.lua(f"game.speed = {speed} return game.speed")
    return float(result)


#: Fields of an episode-cycle record that must not depend on wall-clock pacing.
DETERMINISM_KEYS = (
    "episode_tick",
    "advance_result",
    "character_position",
    "character_inventory",
    "src_contents",
    "dst_contents",
    "task",
)


def run_speed_determinism(
    worker_id: str = "bench-speed",
    speeds: tuple[float, ...] = DEFAULT_SPEEDS,
    cycles: int = 5,
) -> dict[str, Any]:
    """Prove `game.speed` changes pacing only, never outcomes.

    `game.speed` is the largest single throughput lever (30 ticks costs 500 ms
    of wall clock at the default 60 UPS), but it is only usable if the
    simulation is identical at any speed. Factorio is tick-based, so it should
    be -- and the Phase 0 gate already has the machinery to prove it, comparing
    episode-cycle records field by field.

    Runs the same reset/move/advance/transfer cycle at each speed on one worker
    and asserts the records match exactly.
    """
    from factoriorl.gate_phase0 import _episode_cycle

    manager = WorkerManager()
    handle = manager.launch(worker_id)
    report: dict[str, Any] = {
        "engine": handle.engine.to_dict(),
        "speeds": list(speeds),
        "cycles_per_speed": cycles,
        "records": {},
        "wall_ms": {},
        "mismatches": [],
    }
    try:
        with WorkerSession(handle, timeout=20.0) as session:
            session.status()
            for speed in speeds:
                set_speed(handle, speed)
                key = f"speed_{speed:g}"
                started = time.perf_counter()
                records = [_episode_cycle(session, i) for i in range(cycles)]
                report["wall_ms"][key] = round((time.perf_counter() - started) * 1000.0, 2)
                report["records"][key] = [{k: rec[k] for k in DETERMINISM_KEYS} for rec in records]
            set_speed(handle, 1.0)

        baseline_key = f"speed_{speeds[0]:g}"
        baseline = report["records"][baseline_key]
        for speed in speeds[1:]:
            key = f"speed_{speed:g}"
            for index, (want, got) in enumerate(zip(baseline, report["records"][key], strict=True)):
                for field_name in DETERMINISM_KEYS:
                    if want[field_name] != got[field_name]:
                        report["mismatches"].append(
                            {
                                "speed": key,
                                "cycle": index,
                                "field": field_name,
                                "baseline": want[field_name],
                                "observed": got[field_name],
                            }
                        )
        report["identical"] = not report["mismatches"]
        base_ms = report["wall_ms"][baseline_key]
        report["speedup"] = {k: round(base_ms / v, 2) for k, v in report["wall_ms"].items() if v}
    finally:
        report["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        out_dir = runtime_dir() / "bench"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "speed-determinism.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        manager.cleanup(handle)
    return report
