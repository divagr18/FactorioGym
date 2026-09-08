"""Held-out *combinations*, not held-out distributions (synthesis section 9).

Section 9 asks for a holdout whose descriptor combination is unseen while every
marginal is covered in training. The existing overlap rule in
`tools/generator_diagnostics.py` cannot express that, and worse, it inverts it:
a 12-bin histogram over {1} against one over {2} scores 0.0 overlap, which the
separation rule reads as *good* separation.

Both distribution surprises this repo actually had were exactly that shape.
`restore_power` trained on interior gaps in 600/600 scenes and tested on
terminal ones in 600/600; `repair_belt` trained on one gap and tested on two.
So the rule that would have caught them is set containment, not a retuned
threshold -- and it is deliberately a *separate, separately named* check,
because `SEPARATION_MAX_OVERLAP` requires a structural descriptor at overlap
<= 0.40 and therefore rewards the very disjointness that caused both. The two
rules want opposite things about different descriptors; collapsing them would
silence one.

Lives here rather than in the tool because it is a property of a task, and
because `gate_phase3.py` has to read it -- a gate importing from `tools/` would
be the wrong direction.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from factoriorl.seeding import Branch, SeedPlan
from factoriorl.tasks.spec import Blueprint

#: The categorical descriptors, published as a *cell* rather than as marginals.
#: Checking marginals independently would pass a split whose every individual
#: value is covered while the pair it tests never appears in training.
COVERAGE_DESCRIPTORS = ("fault_count", "fault_neighbours", "fault_marker_published")


def fault_positions(task: Any, blueprint: Blueprint) -> list[tuple[float, float]]:
    """Declared fault markers this scene actually places, in declared order."""
    markers = blueprint.all_markers()
    return [tuple(markers[name]) for name in task.spec.fault_markers if name in markers]


def flanking_sides(blueprint: Blueprint, fault: tuple[float, float]) -> int:
    """How many sides of `fault` carry more of *its own chain*.

    2 means the fault sits inside the chain, 1 that it is at an end, 0 that no
    chain lines up with it. That is the interior-versus-terminal distinction,
    and it is invisible to every other descriptor: moving a missing pole from
    the middle of a chain to its end changes no count, no footprint and no
    occupancy.

    "Its own chain" is load-bearing, and the first version of this got it
    wrong. Counting *any* collinear entity reported 2 for a terminal
    `restore_power` gap as well as an interior one, because the drill sits at
    the chain's end and occupies the outer side -- so the descriptor scored the
    two regimes identically and the containment rule built on it would have
    passed the exact surprise it exists to catch. Restricting to the prototype
    the chain is made of (the most numerous collinear prototype) separates
    them: measured on the real generators, `pole_gap` scores 2 and
    `gap_near_drill` scores 1.

    The axis is chosen by measurement rather than declared, because a chain's
    orientation is a generator detail.
    """
    best = 0
    for axis in (0, 1):
        other = 1 - axis
        collinear = [
            entity
            for entity in blueprint.entities
            if abs(entity.position[other] - fault[other]) <= 1.5
        ]
        if not collinear:
            continue
        counts: dict[str, int] = {}
        for entity in collinear:
            counts[entity.name] = counts.get(entity.name, 0) + 1
        chain = max(counts, key=lambda name: (counts[name], name))
        positions = [e.position for e in collinear if e.name == chain]
        before = any(position[axis] < fault[axis] - 0.5 for position in positions)
        after = any(position[axis] > fault[axis] + 0.5 for position in positions)
        best = max(best, int(before) + int(after))
    return best


def coverage_cell(task: Any, blueprint: Blueprint) -> tuple[int, int, int]:
    """The categorical cell this scene occupies."""
    faults = fault_positions(task, blueprint)
    published = set(task.spec.public_markers)
    return (
        len(faults),
        max((flanking_sides(blueprint, f) for f in faults), default=0),
        int(all(name in published for name in task.spec.fault_markers))
        if task.spec.fault_markers
        else 0,
    )


def _digest(payload: dict) -> str:
    """The algorithm `factoriorl.env.blueprint_digest` installs scenes by.

    Recomputed rather than imported for the reason
    `tools/generator_diagnostics.py` gives for the same duplication:
    `factoriorl.env` pulls in gymnasium and numpy, and this analysis must stay
    runnable without the `rl` extra. A test pins the two together.
    """
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def occupancy(task: Any, family: Any, samples: int, plan: SeedPlan) -> dict[tuple, dict]:
    """Cells one layout family occupies, sampled the way the environment does.

    `FactorioEnv.prepare_scene` draws `rng.randrange(len(families_in_split))`
    to choose the family and hands that *same, already-advanced* RNG to
    `task.generate`, so the discarded draw is replayed here. Sampling with a
    fresh RNG would explore states the environment never produces.
    """
    split_size = max(1, len(task.spec.families(family.split)))
    cells: dict[tuple, dict] = {}
    for index in range(samples):
        rng = plan.generator_rng(Branch.TRAIN, index)
        rng.randrange(split_size)
        blueprint = task.generate(family, rng)
        cell = coverage_cell(task, blueprint)
        row = cells.setdefault(cell, {"count": 0, "example": None})
        row["count"] += 1
        if row["example"] is None:
            # One example per cell, kept rather than discarded: "the test split
            # occupies a cell training never does" is only actionable with a
            # scene to look at.
            row["example"] = {
                "episode_index": index,
                "blueprint_digest": _digest(
                    blueprint.to_dict(public_markers=task.spec.public_markers)
                ),
            }
    return cells


def containment(task: Any, per_family: dict[str, dict[tuple, dict]]) -> dict:
    """Every cell the test split occupies must also be occupied in training.

    `per_family` maps layout family name to that family's cell occupancy.
    Reported for every task; a task declaring no `fault_markers` has one cell
    and trivially passes, which is stated as `vacuous` rather than hidden -- a
    vacuous pass must not read as a verified one.
    """
    spec = task.spec

    def merge(split: str) -> dict[tuple, dict]:
        merged: dict[tuple, dict] = {}
        for family in spec.families(split):
            for cell, body in per_family.get(family.name, {}).items():
                row = merged.setdefault(
                    cell, {"count": 0, "example": body["example"], "families": []}
                )
                row["count"] += body["count"]
                row["families"].append(family.name)
        return merged

    train = merge("train")
    test = merge("test")
    held_out = sorted(set(test) - set(train))

    def rows(cells: dict[tuple, dict]) -> list[dict]:
        return [
            {"cell": dict(zip(COVERAGE_DESCRIPTORS, cell, strict=True)), **body}
            for cell, body in sorted(cells.items())
        ]

    return {
        "descriptors": list(COVERAGE_DESCRIPTORS),
        "declared_fault_markers": list(spec.fault_markers),
        "vacuous": not spec.fault_markers,
        "train_cells": rows(train),
        "test_cells": rows(test),
        "held_out_cells": [
            {
                "cell": dict(zip(COVERAGE_DESCRIPTORS, cell, strict=True)),
                "test_count": test[cell]["count"],
                "test_families": test[cell]["families"],
                "example": test[cell]["example"],
            }
            for cell in held_out
        ],
        "contained": not held_out,
    }


def analyse(task: Any, samples: int, plan: SeedPlan | None = None) -> dict:
    """Sample every layout family and apply the containment rule."""
    plan = plan or SeedPlan(master=777, run_id="coverage")
    per_family = {
        family.name: occupancy(task, family, samples, plan) for family in task.spec.layout_families
    }
    result = containment(task, per_family)
    result["samples_per_family"] = samples
    return result


def failures(task_id: str, result: dict) -> list[str]:
    """One message per uncovered cell, naming a scene to look at."""
    return [
        f"coverage: {task_id}'s test split occupies "
        + ", ".join(f"{k}={v}" for k, v in cell["cell"].items())
        + f" in {cell['test_count']} scenes ({', '.join(cell['test_families'])}) "
        "and no training family ever does; example scene "
        f"{cell['example']['blueprint_digest']} at episode index "
        f"{cell['example']['episode_index']}"
        for cell in result["held_out_cells"]
    ]


__all__ = [
    "COVERAGE_DESCRIPTORS",
    "analyse",
    "containment",
    "coverage_cell",
    "failures",
    "flanking_sides",
    "occupancy",
]
