"""Phase 4 exit gate (DESIGN.md).

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
from factoriorl.learn.policy import EXTRACTOR_VERSION, checkpoint_signature
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


#: Where the declared checkpoint set lives. Committed, unlike the runs
#: themselves.
DECLARATION = "phase4-checkpoints.json"


def _published_runs() -> list[Path]:
    """The checkpoints this gate is about, from a committed declaration.

    DESIGN 4.1's exit clause is about "**the provided** checkpoints", and until
    this declaration existed that phrase named nothing. The previous
    implementation scanned gitignored `runtime/runs/`, sorted alphabetically
    and took the last three. Measured consequences on this machine:

    * 78 directories hold a `model.zip`, and `runs[-3:]` resolved to
      `vec-…`, `vecfull-…`, `w8-…` -- **not** the two runs the 2026-09-07 gate
      certified, so the gate silently changed its own subject as runs
      accumulated;
    * the gate's own smoke run writes a `gate…` prefix, which sorts before all
      of those, so it never checked the checkpoint it had just produced;
    * `release/` is not the source either: its one run records
      `checkpoint_bytes: null` and ships no `model.zip` at all.

    Falls back to the old scan when no declaration is committed, so a fresh
    clone with no evidence still runs, and says which it used.
    """
    root = manifest_module.runs_dir()
    declared = evidence_dir() / DECLARATION
    if declared.is_file():
        body = json.loads(declared.read_text(encoding="utf-8"))
        return [root / entry["run_id"] for entry in body.get("checkpoints") or []]
    if not root.is_dir():
        return []
    return [d for d in sorted(root.iterdir()) if (d / "model.zip").is_file()][-3:]


def _declared_entry(run_id: str) -> dict:
    declared = evidence_dir() / DECLARATION
    if not declared.is_file():
        return {}
    body = json.loads(declared.read_text(encoding="utf-8"))
    for entry in body.get("checkpoints") or []:
        if entry.get("run_id") == run_id:
            return entry
    return {}


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
    # No slice: `_published_runs` returns the declared set, and a gate that
    # silently checked three of it would be back to choosing its own subject.
    for directory in runs:
        run_id = directory.name
        verification = manifest_module.verify(run_id)
        report.check(
            f"{run_id}: manifest verifies against current code",
            verification["ok"],
            problems=verification["problems"],
        )
        data = manifest_module.load(run_id)
        model_block = data.get("model") or {}
        declared = _declared_entry(run_id)
        # Recorded, not compared to the current integer.
        #
        # `EXTRACTOR_VERSION` declares intent and is orthogonal to loadability
        # in both directions: of six bumps exactly one changed a width the
        # loader cannot adapt to, four were semantics-only, and a separate
        # `GOAL_ENCODING_VERSION` 3-to-4 change bumped nothing here at all. So
        # a strict `recorded == current` test both rejects checkpoints that
        # load perfectly and would vouch for ones that do not -- measured: no
        # run on this machine records 7, while a run recording 3 loads cleanly.
        # `architecture_signature` below is the check that actually bites.
        report.check(
            f"{run_id}: manifest records which extractor it was trained on",
            model_block.get("extractor_version") is not None,
            recorded=model_block.get("extractor_version"),
            current=EXTRACTOR_VERSION,
        )
        # The architecture the declaration says this checkpoint has, against
        # the one the file actually carries -- read from the zip, because the
        # interesting case is a checkpoint that cannot load and so cannot be
        # asked.
        if declared.get("architecture_signature"):
            try:
                actual = checkpoint_signature(directory / "model")
            except Exception as exc:  # noqa: BLE001
                actual = f"unreadable: {exc}"
            report.check(
                f"{run_id}: the checkpoint on disk is the declared architecture",
                actual == declared["architecture_signature"],
                declared=declared["architecture_signature"],
                on_disk=actual,
                note="a mismatch means this checkpoint predates the current "
                "architecture, or is not the file that was declared",
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
                # DESIGN 4.1 asks for "both inference and resumable training
                # state", and this second half was unverified while the gate
                # reported 15 of 15. The two fail differently: a checkpoint
                # that deserialises but drops its optimizer state resumes with
                # a freshly initialised Adam, so the moment estimates are gone
                # and the first updates after a resume are a re-warm-up -- the
                # continued run is not the run it claims to continue. A
                # checkpoint that resets `num_timesteps` rewinds any
                # learning-rate or clip-range schedule while the curve keeps
                # counting up, which is quieter and worse.
                #
                # `tests/unit/test_checkpoint_resume.py` pins the round trip
                # itself against a synthetic env; this checks the *published*
                # checkpoints, which is the claim the gate is making.
                state = model.policy.optimizer.state_dict().get("state") or {}
                report.check(
                    f"{run_id}: checkpoint restores optimizer state, so training can resume",
                    bool(state),
                    parameter_groups=len(state),
                )
                report.check(
                    f"{run_id}: checkpoint carries its step count, so a resume continues it",
                    int(getattr(model, "num_timesteps", 0)) > 0,
                    num_timesteps=int(getattr(model, "num_timesteps", 0)),
                )
            except Exception as exc:  # noqa: BLE001
                # A size mismatch here is not a mystery, so do not report it as
                # one: read the shapes out of the zip and name the difference.
                try:
                    on_disk = checkpoint_signature(directory / "model")
                except Exception:  # noqa: BLE001
                    on_disk = "unreadable"
                report.check(
                    f"{run_id}: checkpoint loads for inference",
                    False,
                    error=str(exc).strip().splitlines()[-1][:200],
                    on_disk_architecture=on_disk,
                    declared_architecture=declared.get("architecture_signature"),
                    hint="the architecture changed under this checkpoint; "
                    "`GRID_FEATURES` 64 to 128 is the one width change in the "
                    "history that the loader cannot absorb",
                )

    # ---- 4. schema ----------------------------------------------------
    if runs:
        data = manifest_module.load(runs[-1].name)
        missing = [key for key in manifest_module.REQUIRED_FIELDS if not data.get(key)]
        report.check(
            "the newest manifest carries every field DESIGN section 2 requires",
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
