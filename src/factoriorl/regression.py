"""Adversarially found scenes, kept as named cases (synthesis section 10).

Section 10 asks for a benchmark number and a diagnostic one to be separate
things. A representative rate answers "how good is this policy on this
distribution"; a regression case answers "does this specific known-bad scene
still behave". Pooling the second into the first destroys both: a hand-picked
set of hard scenes drags the rate down without describing any distribution,
and a defect that reappears in one scene out of a hundred moves the rate by a
point and reads as noise.

So this suite is deliberately **not a split** and deliberately **not averaged**.

Not a split, because `train`/`val`/`test` are sampled distributions with a
declared structural holdout, and these are individual scenes chosen precisely
because something was wrong with them. It cannot be `val` either:
`tools/freeze_holdout.py` refuses to freeze `val` on purpose, and a case that
is not frozen is not a regression case at all.

Not averaged, because the only useful summary is which cases fail and why. This
module therefore reports per case and has no `rate` field. That is enforced by
a test rather than left as a convention.

A case names a task, a layout family, and an episode index under the seed plan
recorded with it. `blueprint_digest` pins the scene, so a generator edit that
changes the scene fails loudly instead of silently retargeting the case at
different content -- the same guarantee `holdout_v3` gives its episodes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from factoriorl.paths import evidence_dir
from factoriorl.seeding import Branch, SeedPlan

SCHEMA = "factoriorl.regression/1"

#: The seed plan cases are enumerated under. Fixed, and distinct from both the
#: holdout's plan and any training run's, so a case is a stable object.
REGRESSION_MASTER = 777
REGRESSION_RUN_ID = "generator-diagnostics"


class RegressionError(ValueError):
    """A regression case no longer names the scene it was recorded against."""


@dataclass(frozen=True)
class Case:
    """One named scene, and why it is here."""

    case_id: str
    task: str
    layout_family: str
    episode_index: int
    blueprint_digest: str
    #: What was found, in a sentence. Required: a case with no reason is a
    #: number nobody can act on, and will be deleted by the next person who
    #: cannot tell whether it still matters.
    finding: str
    #: What the case asserts. `scene_property` means the defect is in the
    #: generated content and no policy is involved; `policy_outcome` means the
    #: case is about behaviour on that scene.
    kind: str = "scene_property"
    #: Whether the defect is currently present. A case is kept when the finding
    #: was deliberately not fixed, so that "still open" is recorded rather than
    #: implied by a failing check nobody expects to pass.
    expected_present: bool = False

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "task": self.task,
            "layout_family": self.layout_family,
            "episode_index": self.episode_index,
            "blueprint_digest": self.blueprint_digest,
            "finding": self.finding,
            "kind": self.kind,
            "expected_present": self.expected_present,
        }


def plan() -> SeedPlan:
    return SeedPlan(master=REGRESSION_MASTER, run_id=REGRESSION_RUN_ID)


def blueprint_for(task: Any, case: Case) -> Any:
    """Regenerate the case's scene, exactly as the environment would.

    `FactorioEnv.prepare_scene` draws `rng.randrange(len(families_in_split))`
    before handing the same RNG to the generator, so the discarded draw is
    replayed. Without it the case would name a scene no run installs.
    """
    family = next(f for f in task.spec.layout_families if f.name == case.layout_family)
    split_size = max(1, len(task.spec.families(family.split)))
    rng = plan().generator_rng(Branch.TRAIN, case.episode_index)
    rng.randrange(split_size)
    return task.generate(family, rng)


def suite_path() -> Path:
    return evidence_dir() / "regression_scenes.json"


def load(path: Path | None = None) -> list[Case]:
    document = json.loads((path or suite_path()).read_text(encoding="utf-8"))
    if document.get("schema") != SCHEMA:
        raise RegressionError(f"unexpected schema {document.get('schema')!r}")
    return [Case(**row) for row in document["cases"]]


def save(cases: list[Case], path: Path | None = None) -> Path:
    target = path or suite_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "seed_plan": {
                    "master": REGRESSION_MASTER,
                    "run_id": REGRESSION_RUN_ID,
                    "branch": Branch.TRAIN.value,
                },
                "note": (
                    "Individually named scenes, not a distribution. Reported per "
                    "case and never averaged: a hand-picked set of hard scenes "
                    "makes a rate that describes nothing."
                ),
                "cases": [case.to_dict() for case in sorted(cases, key=lambda c: c.case_id)],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return target


def verify(cases: list[Case]) -> list[str]:
    """Every case must still name the scene it was recorded against."""
    from factoriorl.tasks import coverage, get

    problems: list[str] = []
    for case in cases:
        try:
            task = get(case.task)
        except Exception as exc:  # noqa: BLE001 - report, do not raise
            problems.append(f"{case.case_id}: task {case.task} is not registered ({exc})")
            continue
        if case.layout_family not in {f.name for f in task.spec.layout_families}:
            problems.append(
                f"{case.case_id}: {case.task} declares no layout family {case.layout_family!r}"
            )
            continue
        blueprint = blueprint_for(task, case)
        digest = coverage._digest(blueprint.to_dict(public_markers=task.spec.public_markers))
        if digest != case.blueprint_digest:
            problems.append(
                f"{case.case_id}: recorded digest {case.blueprint_digest} but the "
                f"generator now produces {digest}; this case no longer names the "
                "scene the finding was about"
            )
    return problems


def report(cases: list[Case]) -> dict:
    """Per-case outcomes. Deliberately carries no pooled rate."""
    problems = verify(cases)
    stale = {p.split(":")[0] for p in problems}
    return {
        "schema": SCHEMA,
        "cases": [
            {
                **case.to_dict(),
                "scene_still_matches": case.case_id not in stale,
            }
            for case in sorted(cases, key=lambda c: c.case_id)
        ],
        "stale_cases": sorted(stale),
        "open_findings": sorted(c.case_id for c in cases if c.expected_present),
        "problems": problems,
        # No `success_rate`, and no `mean_*`. See the module docstring; a test
        # asserts the absence.
        "note": "per case; not a distribution and not averaged",
    }
