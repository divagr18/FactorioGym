"""FactorioRL CLI.

Commands:
  factoriorl doctor         probe engine and print pinned version info
  factoriorl worker start   launch one supervised worker, handshake, print status
  factoriorl worker reap    terminate engines orphaned by a failed parent run
  factoriorl phase0-gate    run the Phase 0 exit gate end to end
  factoriorl phase1-gate    run the Phase 1 exit gate and record its evidence
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

from factoriorl.engine_config import resolve_engine_config
from factoriorl.errors import FactorioRLError
from factoriorl.paths import evidence_dir, workspace_root

PHASE1_EVIDENCE_NAME = "phase1-engine-suite.txt"


def cmd_doctor(_args) -> int:
    try:
        engine = resolve_engine_config()
    except FactorioRLError as exc:
        print(f"engine check failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(engine.to_dict(), indent=2))
    return 0


def cmd_worker_start(_args) -> int:
    """Launch through the pool, so the supervised path is the demonstrated one."""
    from factoriorl.pool import WorkerPool

    with WorkerPool() as pool:
        worker = pool.start("cli-demo")
        print(json.dumps(worker.session.status().response.result, indent=2))
        print(json.dumps(worker.session.observe().response.result, indent=2))
        print(json.dumps({"liveness": pool.heartbeat(force=True)}, indent=2))
    return 0


def cmd_worker_reap(args) -> int:
    from factoriorl.pool import reap_orphans

    orphans = reap_orphans(kill=not args.dry_run)
    print(json.dumps({"orphans": orphans, "count": len(orphans)}, indent=2))
    return 0


def cmd_bench_transport(args) -> int:
    from factoriorl.bench import run_transport_bench

    report = run_transport_bench(reps=args.reps)
    print(json.dumps(report, indent=2))
    return 0


def cmd_bench_speed(_args) -> int:
    from factoriorl.bench import run_speed_determinism

    report = run_speed_determinism()
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))
    return 0 if report.get("identical") else 1


def cmd_phase0_gate(_args) -> int:
    from factoriorl.gate_phase0 import run_phase0_gate

    report = run_phase0_gate()
    print(json.dumps(report, indent=2))
    return 0 if report.get("passed") else 1


def cmd_phase1_gate(_args) -> int:
    """Run the engine suite and record its output as gate evidence.

    PLAN.md 1 exit gate: protocol contract tests and lifecycle fault tests
    passing against actual Factorio workers. Recording the transcript is part
    of the gate, not an afterthought -- a gate whose evidence is not written
    down cannot be re-checked later.
    """
    evidence_dir().mkdir(parents=True, exist_ok=True)
    target = evidence_dir() / PHASE1_EVIDENCE_NAME
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/engine",
        "-v",
        "--tb=short",
        # A gate must never pass by skipping its own evidence (PLAN.md sec. 4).
        "--require-engine",
        "-p",
        "no:cacheprovider",
    ]
    print(f"$ {' '.join(cmd)}")
    completed = subprocess.run(
        cmd,
        cwd=str(workspace_root()),
        capture_output=True,
        text=True,
        check=False,
    )
    transcript = completed.stdout + completed.stderr
    print(transcript)
    header = (
        f"# Phase 1 exit gate evidence\n"
        f"# command: {' '.join(cmd)}\n"
        f"# exit code: {completed.returncode}\n\n"
    )
    target.write_text(header + transcript, encoding="utf-8")
    print(f"evidence written to {target} (exit code {completed.returncode})")
    return completed.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="factoriorl")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="probe engine and pinned version")

    worker = sub.add_parser("worker", help="worker commands")
    worker_sub = worker.add_subparsers(dest="worker_command", required=True)
    worker_sub.add_parser("start", help="launch one worker and print handshake status")
    reap = worker_sub.add_parser("reap", help="terminate engines orphaned by a dead parent")
    reap.add_argument("--dry-run", action="store_true", help="report orphans without killing")

    bench = sub.add_parser("bench", help="measurement harnesses")
    bench_sub = bench.add_subparsers(dest="bench_command", required=True)
    bench_transport = bench_sub.add_parser("transport", help="RCON round-trip decomposition")
    bench_transport.add_argument("--reps", type=int, default=15)
    bench_sub.add_parser("speed", help="prove game.speed changes pacing, not outcomes")

    sub.add_parser("phase0-gate", help="run the Phase 0 exit gate")
    sub.add_parser("phase1-gate", help="run the Phase 1 exit gate and record evidence")

    args = parser.parse_args(argv)
    if args.command == "doctor":
        return cmd_doctor(args)
    if args.command == "worker":
        if args.worker_command == "reap":
            return cmd_worker_reap(args)
        return cmd_worker_start(args)
    if args.command == "bench":
        if args.bench_command == "speed":
            return cmd_bench_speed(args)
        return cmd_bench_transport(args)
    if args.command == "phase1-gate":
        return cmd_phase1_gate(args)
    return cmd_phase0_gate(args)


if __name__ == "__main__":
    raise SystemExit(main())
