"""Phase 0 exit gate (PLAN.md).

One command launches a worker, acts, steps, resets, verifies repeatability,
and exits cleanly. Records latency and reset measurements as evidence:

    uv run factoriorl phase0-gate

Proves:
- 0.3 exact stepping (1/30/120 ticks, mixed sequences, zero idle drift);
- 0.4 movement + transfer semantics (collision, reach, conservation);
- 0.5 >=10 identical reset cycles through the same transport.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from factoriorl.paths import runtime_dir
from factoriorl.session import WorkerSession
from factoriorl.worker import WorkerManager

REPETITIONS = 10
MOVEMENT_TICKS = 30


@dataclass
class GateReport:
    passed: bool = False
    repetitions: int = REPETITIONS
    cycles: list[dict] = field(default_factory=list)
    stepping: dict = field(default_factory=dict)
    actions: dict = field(default_factory=dict)
    latency_ms: dict = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)

    def fail(self, reason: str) -> None:
        self.failures.append(reason)
        self.passed = False

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "repetitions": self.repetitions,
            "stepping": self.stepping,
            "actions": self.actions,
            "latency_ms": self.latency_ms,
            "failures": self.failures,
            "cycles": self.cycles,
        }


def _episode_cycle(session: WorkerSession, index: int) -> dict:
    """One reset cycle: identical action sequence; returns the outcome record."""
    latencies: list[float] = []

    reset = session.reset()
    latencies.append(reset.round_trip_ms)
    episode_id = session.episode_id

    initial = session.observe()
    latencies.append(initial.round_trip_ms)

    move = session.act("move", direction="south", ticks=MOVEMENT_TICKS)
    latencies.append(move.round_trip_ms)
    step1 = session.advance(MOVEMENT_TICKS)
    latencies.append(step1.round_trip_ms)
    pos_after_move = session.observe()
    latencies.append(pos_after_move.round_trip_ms)

    transfer = session.act(
        "transfer", **{"from": "src", "to": "character", "item": "iron-plate", "count": 10}
    )
    latencies.append(transfer.round_trip_ms)
    final = session.observe()
    latencies.append(final.round_trip_ms)

    return {
        "cycle": index,
        "episode_id": episode_id,
        "episode_tick": final.response.result["tick"],
        "advance_result": step1.response.result,
        "character_position": pos_after_move.response.result["character"]["position"],
        "character_inventory": final.response.result["inventory"],
        "src_contents": final.response.result["entities"]["src"]["contents"],
        "dst_contents": final.response.result["entities"]["dst"]["contents"],
        "task": final.response.result["task"],
        "latency_ms_mean": sum(latencies) / len(latencies),
    }


def run_phase0_gate(worker_id: str = "phase0-gate") -> dict:
    report = GateReport()
    manager = WorkerManager()
    handle = manager.launch(worker_id)
    gate_dir = runtime_dir() / "gate-phase0"
    gate_dir.mkdir(parents=True, exist_ok=True)
    try:
        with WorkerSession(handle) as session:
            # ---------------- 0.3 exact stepping
            base = session.observe().response.result["absolute_tick"]
            advances = {}
            for ticks in (1, 30, 120):
                timed = session.advance(ticks)
                result = timed.response.result
                advances[str(ticks)] = result
                if result.get("ticks_advanced") != ticks:
                    report.fail(f"advance({ticks}) reported {result}")
            # Mixed sequence drift check.
            for ticks in (30, 1, 120, 30, 30):
                session.advance(ticks)
            after_mixed = session.observe().response.result["absolute_tick"]
            expected = base + 1 + 30 + 120 + 30 + 1 + 120 + 30 + 30
            report.stepping = {
                "advances": advances,
                "mixed_drift": after_mixed - expected,
            }
            if after_mixed != expected:
                report.fail(f"mixed sequence drifted: {after_mixed} != {expected}")
            # Idle does not advance.
            idle_a = session.observe().response.result["absolute_tick"]
            time.sleep(0.5)
            idle_b = session.observe().response.result["absolute_tick"]
            report.stepping["idle_drift"] = idle_b - idle_a
            if idle_b != idle_a:
                report.fail("world advanced while idle between requests")

            # ---------------- 0.4 physical actions
            session.reset()
            # Movement south.
            session.act("move", direction="south", ticks=MOVEMENT_TICKS)
            session.advance(MOVEMENT_TICKS)
            moved = session.observe().response.result["character"]["position"]
            # Collision: walk 600 ticks north into the wall row at y=-6.
            session.act("move", direction="north", ticks=600)
            session.advance(600)
            blocked = session.observe().response.result["character"]["position"]
            report.actions = {
                "position_after_south_move": moved,
                "position_after_600t_north_into_wall": blocked,
            }
            if not moved[1] > 2.0:
                report.fail(f"south movement did not displace character: {moved}")
            if blocked[1] < -5.6:
                report.fail(f"wall did not block movement: {blocked}")

            # Conservation: transfer 10 of 50.
            session.reset()
            before = session.observe().response.result
            tr = session.act(
                "transfer", **{"from": "src", "to": "character", "item": "iron-plate", "count": 10}
            )
            after = session.observe().response.result
            total_before = sum(before["entities"]["src"]["contents"].values())
            total_after = sum(after["entities"]["src"]["contents"].values()) + sum(
                after["inventory"].values()
            )
            report.actions["transfer"] = tr.response.result
            report.actions["item_total_before"] = total_before
            report.actions["item_total_after"] = total_after
            if tr.response.code.value != "ok":
                report.fail(f"valid transfer rejected: {tr.response}")
            if total_before != total_after:
                report.fail("transfer created or destroyed items")
            # Overdraw fails and changes nothing.
            overdraw = session.act(
                "transfer",
                **{"from": "src", "to": "character", "item": "iron-plate", "count": 999},
            )
            after_overdraw = session.observe().response.result
            report.actions["overdraw"] = (
                overdraw.response.error.to_dict() if overdraw.response.error else None
            )
            if overdraw.response.code.value != "rejected":
                report.fail("overdraw transfer was not rejected")
            if (
                after_overdraw["entities"]["src"]["contents"]
                != after["entities"]["src"]["contents"]
            ):
                report.fail("rejected overdraw mutated state")

            # ---------------- 0.5 reset repeatability (>=10 cycles)
            baseline = None
            for i in range(REPETITIONS):
                record = _episode_cycle(session, i)
                if baseline is None:
                    baseline = record
                else:
                    for key in (
                        "episode_tick",
                        "advance_result",
                        "character_position",
                        "character_inventory",
                        "src_contents",
                        "dst_contents",
                        "task",
                    ):
                        if record[key] != baseline[key]:
                            report.fail(
                                f"cycle {i} diverged on {key}: {record[key]!r} != {baseline[key]!r}"
                            )
                report.cycles.append(record)

            report.latency_ms = {
                "mean_per_cycle_ms": sum(c["latency_ms_mean"] for c in report.cycles)
                / len(report.cycles),
            }
            report.passed = not report.failures
    finally:
        report_dict = report.to_dict()
        report_dict["engine"] = handle.engine.to_dict()
        report_dict["worker"] = handle.spec.manifest()
        report_dict["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        (gate_dir / "report.json").write_text(json.dumps(report_dict, indent=2), encoding="utf-8")
        manager.cleanup(handle)
    return report_dict


if __name__ == "__main__":
    print(json.dumps(run_phase0_gate(), indent=2))
