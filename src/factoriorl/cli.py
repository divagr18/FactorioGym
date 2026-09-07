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


def cmd_spike_control(_args) -> int:
    from factoriorl.spike_control import run_control_spike

    report = run_control_spike()
    print(json.dumps(report, indent=2))
    return 0


def cmd_phase0_gate(_args) -> int:
    from factoriorl.gate_phase0 import run_phase0_gate

    report = run_phase0_gate()
    print(json.dumps(report, indent=2))
    return 0 if report.get("passed") else 1


def cmd_phase2_gate(_args) -> int:
    from factoriorl.gate_phase2 import run_phase2_gate

    report = run_phase2_gate()
    summary = {k: v for k, v in report.items() if k != "steps"}
    print(json.dumps(summary, indent=2))
    for step in report.get("steps", []):
        mark = "ok " if step.get("ok", True) else "FAIL"
        print(f"  [{mark}] {step['criterion']}: {step['note']}")
    return 0 if report.get("passed") else 1


def cmd_phase3_gate(_args) -> int:
    from factoriorl.gate_phase3 import run_phase3_gate

    report = run_phase3_gate()
    print(json.dumps({k: v for k, v in report.items() if k != "sections"}, indent=2))
    for section, checks in report.get("sections", {}).items():
        for check in checks:
            mark = "ok " if check["ok"] else "FAIL"
            print(f"  [{mark}] {section}: {check['check']}")
    return 0 if report.get("passed") else 1


def cmd_tasks(args) -> int:
    from factoriorl.tasks import all_tasks, validate_all

    if args.tasks_command == "list":
        for task_id, task in sorted(all_tasks().items()):
            spec = task.spec
            families = ", ".join(f"{f.name}({f.split})" for f in spec.layout_families)
            print(f"{task_id:16s} v{spec.version}  {families}")
            print(f"                 {spec.description}")
        return 0
    report = validate_all()
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


def cmd_runs(args) -> int:
    from factoriorl import manifest as manifest_module

    if args.runs_command == "list":
        for row in manifest_module.list_runs():
            print(f"{row['run_id']}  {row['created_at']}  {row['task']} v{row['task_version']}")
        return 0
    if args.runs_command == "show":
        print(json.dumps(manifest_module.load(args.run_id), indent=2))
        return 0
    result = manifest_module.verify(args.run_id)
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


def cmd_profile(args) -> int:
    from factoriorl.profiling import run_profile

    counts = tuple(int(x) for x in args.workers.split(","))
    report = run_profile(worker_counts=counts, task_id=args.task, steps_per_worker=args.steps)
    print(json.dumps(report, indent=2))
    return 0


def cmd_phase4_gate(args) -> int:
    from factoriorl.gate_phase4 import run_phase4_gate

    report = run_phase4_gate(mode=args.mode)
    print(json.dumps({k: v for k, v in report.items() if k != "checks"}, indent=2))
    for check in report.get("checks", []):
        mark = "ok " if check["ok"] else "FAIL"
        print(f"  [{mark}] {check['check']}")
    return 0 if report.get("passed") else 1


def cmd_train(args) -> int:
    from factoriorl.learn.train import TrainConfig, train

    result = train(
        TrainConfig(
            task_id=args.task,
            total_steps=args.steps,
            master_seed=args.seed,
            shaping=not args.no_shaping,
            workers=args.workers,
            eval_episodes=args.eval_episodes,
            run_prefix=args.prefix,
        )
    )
    print(json.dumps(result, indent=2))
    return 0


def cmd_doctor_train(_args) -> int:
    """Fail loudly on a CPU-only wheel: the commonest Windows setup failure."""
    try:
        import torch
    except ImportError:
        print("torch is not installed; run: uv sync --group train", file=sys.stderr)
        return 1
    info = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        info["device"] = properties.name
        info["vram_gb"] = round(properties.total_memory / 1e9, 1)
        info["capability"] = f"sm_{properties.major}{properties.minor}"
    print(json.dumps(info, indent=2))
    if not torch.cuda.is_available():
        print("CUDA is unavailable: this is a CPU-only torch wheel", file=sys.stderr)
        return 1
    return 0


def cmd_action_matrix(args) -> int:
    from factoriorl.action_matrix import to_markdown
    from factoriorl.paths import workspace_root

    target = workspace_root() / "docs" / "ACTION_MATRIX.md"
    rendered = to_markdown()
    if args.check:
        current = target.read_text(encoding="utf-8") if target.is_file() else ""
        if current.strip() != rendered.strip():
            print("docs/ACTION_MATRIX.md is stale; run: factoriorl action-matrix", file=sys.stderr)
            return 1
        print("docs/ACTION_MATRIX.md is current")
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(rendered, encoding="utf-8")
    print(f"wrote {target}")
    return 0


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

    sub.add_parser("spike-control", help="probe LuaControl on a player-less character")
    sub.add_parser("phase0-gate", help="run the Phase 0 exit gate")
    sub.add_parser("phase1-gate", help="run the Phase 1 exit gate and record evidence")
    sub.add_parser("phase2-gate", help="run the Phase 2 exit gate (scripted embodied agent)")
    sub.add_parser("phase3-gate", help="run the Phase 3 exit gate (tasks, spaces, resets)")

    train_cmd = sub.add_parser("train", help="train a policy on one task")
    train_cmd.add_argument("--task", required=True)
    train_cmd.add_argument("--steps", type=int, default=50_000)
    train_cmd.add_argument("--seed", type=int, default=20260907)
    train_cmd.add_argument("--eval-episodes", type=int, default=20)
    train_cmd.add_argument("--no-shaping", action="store_true")
    train_cmd.add_argument("--prefix", default="train")
    train_cmd.add_argument("--workers", type=int, default=1, help="parallel envs")

    sub.add_parser("doctor-train", help="check the training stack and CUDA")
    prof = sub.add_parser("profile", help="worker-count throughput profiling (PLAN 4.3)")
    prof.add_argument("--workers", default="1,2,4,8")
    prof.add_argument("--task", default="navigate")
    prof.add_argument("--steps", type=int, default=120)
    gate4 = sub.add_parser("phase4-gate", help="run the Phase 4 exit gate")
    gate4.add_argument("--mode", choices=("reproduce", "full"), default="reproduce")

    tasks_cmd = sub.add_parser("tasks", help="task registry")
    tasks_sub = tasks_cmd.add_subparsers(dest="tasks_command", required=True)
    tasks_sub.add_parser("list", help="list registered tasks")
    tasks_sub.add_parser("validate", help="validate every task before training")

    runs_cmd = sub.add_parser("runs", help="run manifests")
    runs_sub = runs_cmd.add_subparsers(dest="runs_command", required=True)
    runs_sub.add_parser("list", help="list recorded runs")
    show_cmd = runs_sub.add_parser("show", help="print a run manifest")
    show_cmd.add_argument("run_id")
    verify_cmd = runs_sub.add_parser("verify", help="re-check a manifest against current code")
    verify_cmd.add_argument("run_id")

    matrix_cmd = sub.add_parser("action-matrix", help="generate docs/ACTION_MATRIX.md")
    matrix_cmd.add_argument("--check", action="store_true", help="fail if the doc is stale")

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
    if args.command == "spike-control":
        return cmd_spike_control(args)
    if args.command == "phase1-gate":
        return cmd_phase1_gate(args)
    if args.command == "phase2-gate":
        return cmd_phase2_gate(args)
    if args.command == "phase3-gate":
        return cmd_phase3_gate(args)
    if args.command == "train":
        return cmd_train(args)
    if args.command == "doctor-train":
        return cmd_doctor_train(args)
    if args.command == "profile":
        return cmd_profile(args)
    if args.command == "phase4-gate":
        return cmd_phase4_gate(args)
    if args.command == "tasks":
        return cmd_tasks(args)
    if args.command == "runs":
        return cmd_runs(args)
    if args.command == "action-matrix":
        return cmd_action_matrix(args)
    return cmd_phase0_gate(args)


if __name__ == "__main__":
    raise SystemExit(main())
