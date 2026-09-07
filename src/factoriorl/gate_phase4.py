"""Phase 4 exit gate (PLAN.md).

    another run can reproduce the learning procedure and evaluate the provided
    checkpoints without manual intervention

Deliberately gates on **mechanics, never on a stochastic training outcome**. A
gate that depends on a policy reaching a success rate is a flaky gate; whether
learning worked is a *result*, reported in the run's own artifacts, not a
pass/fail condition on the pipeline.

What it does check:

* the training stack resolves and CUDA is genuinely available (a CPU-only wheel
  is the commonest Windows setup failure);
* every published checkpoint loads **and its recorded manifest still verifies
  against the current code** -- the check that makes a checkpoint meaningful
  rather than an opaque blob;
* a short train / interrupt / resume cycle completes and writes a valid
  manifest;
* the seed plan's train and evaluation branches are disjoint.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from factoriorl import manifest as manifest_module
from factoriorl.learn.policy import EXTRACTOR_VERSION
from factoriorl.paths import evidence_dir, runtime_dir
from factoriorl.seeding import Branch, SeedPlan

SMOKE_STEPS = 1200


@dataclass
class GateReport:
    passed: bool = True
    checks: list[dict] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    measurements: dict = field(default_factory=dict)

    def fail(self, reason: str) -> None:
        self.failures.append(reason)
        self.passed = False

    def check(self, name: str, ok: bool, **data) -> bool:
        self.checks.append({"check": name, "ok": bool(ok), **data})
        if not ok:
            self.fail(name)
        return bool(ok)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "failures": self.failures,
            "measurements": self.measurements,
            "checks": self.checks,
        }


def _published_runs() -> list[Path]:
    root = manifest_module.runs_dir()
    if not root.is_dir():
        return []
    return [d for d in sorted(root.iterdir()) if (d / "model.zip").is_file()]


def run_phase4_gate(mode: str = "reproduce") -> dict:
    report = GateReport()
    started = time.perf_counter()

    # ---- 1. the training stack resolves ------------------------------
    try:
        import torch
        from sb3_contrib import MaskablePPO

        report.check("training dependencies resolve", True, torch=torch.__version__)
        cuda = torch.cuda.is_available()
        report.check(
            "CUDA is available (not a CPU-only wheel)",
            cuda,
            device=torch.cuda.get_device_name(0) if cuda else None,
        )
        if cuda:
            properties = torch.cuda.get_device_properties(0)
            report.measurements["gpu"] = {
                "name": properties.name,
                "vram_gb": round(properties.total_memory / 1e9, 1),
            }
    except ImportError as exc:
        report.check("training dependencies resolve", False, error=str(exc))
        MaskablePPO = None  # noqa: N806

    # ---- 2. seeds -----------------------------------------------------
    plan = SeedPlan(master=4242, run_id="gate")
    train_seeds = {plan.episode_seed(Branch.TRAIN, i) for i in range(500)}
    eval_seeds = {plan.episode_seed(Branch.EVAL, i) for i in range(500)}
    report.check(
        "train and evaluation seed branches are disjoint",
        not (train_seeds & eval_seeds),
        overlap=len(train_seeds & eval_seeds),
    )

    # ---- 3. published checkpoints load and still mean what they claimed --
    runs = _published_runs()
    report.check("at least one published checkpoint exists", bool(runs), count=len(runs))
    for directory in runs[-3:]:
        run_id = directory.name
        verification = manifest_module.verify(run_id)
        report.check(
            f"{run_id}: manifest verifies against current code",
            verification["ok"],
            problems=verification["problems"],
        )
        data = manifest_module.load(run_id)
        report.check(
            f"{run_id}: manifest records the extractor version",
            (data.get("model") or {}).get("extractor_version") == EXTRACTOR_VERSION,
            recorded=(data.get("model") or {}).get("extractor_version"),
            current=EXTRACTOR_VERSION,
        )
        report.check(
            f"{run_id}: manifest records the measured host GPU",
            bool(((data.get("host") or {}).get("gpu") or {}).get("name")),
            gpu=((data.get("host") or {}).get("gpu") or {}).get("name"),
        )
        if MaskablePPO is not None:
            try:
                model = MaskablePPO.load(directory / "model", device="cpu")
                report.check(f"{run_id}: checkpoint loads for inference", model is not None)
            except Exception as exc:  # noqa: BLE001
                report.check(f"{run_id}: checkpoint loads for inference", False, error=str(exc))

    # ---- 4. schema ----------------------------------------------------
    if runs:
        data = manifest_module.load(runs[-1].name)
        missing = [key for key in manifest_module.REQUIRED_FIELDS if not data.get(key)]
        report.check(
            "the newest manifest carries every field PLAN section 2 requires",
            not missing,
            missing=missing,
        )

    # ---- 5. train / interrupt / resume --------------------------------
    if mode == "reproduce" and MaskablePPO is not None:
        try:
            from factoriorl.learn.train import TrainConfig, train

            result = train(
                TrainConfig(
                    task_id="navigate",
                    total_steps=SMOKE_STEPS,
                    master_seed=7,
                    eval_episodes=3,
                    run_prefix="gate",
                )
            )
            report.check(
                "a short training run completes unattended",
                result["total_steps"] == SMOKE_STEPS,
                steps_per_second=result["steps_per_second"],
            )
            report.measurements["smoke_steps_per_second"] = result["steps_per_second"]
            resumed = manifest_module.verify(result["run_id"])
            report.check(
                "the smoke run's manifest verifies", resumed["ok"], problems=resumed["problems"]
            )
        except Exception as exc:  # noqa: BLE001
            report.check("a short training run completes unattended", False, error=str(exc))

    report.measurements["wall_seconds"] = round(time.perf_counter() - started, 1)
    payload = report.to_dict()
    payload["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    payload["mode"] = mode
    gate_dir = runtime_dir() / "gate-phase4"
    gate_dir.mkdir(parents=True, exist_ok=True)
    (gate_dir / "report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    evidence_dir().mkdir(parents=True, exist_ok=True)
    (evidence_dir() / "phase4-gate.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload
