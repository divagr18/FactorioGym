"""R5.1's report: what the published record already diagnoses, and what it cannot.

R5.1's gate is a prohibition before it is a request: *"Reports identify where
attempts fail. Neither zero success nor high training success alone is called an
algorithmic explanation."* So this assembles the diagnosis from the committed
evidence and attaches no algorithmic claim to any number in it.

The separating measurement is the unfamiliar-seed row
----------------------------------------------------
A held-out score of 0.00 is two completely different findings depending on one
other number, and the record contains both. `restore_power` scores 0.00 on the
structural holdout with an unfamiliar-seed rate of **1.00**: it executes its
training distribution perfectly and does not transfer. `repair_belt` scores
0.00 on the holdout with an unfamiliar-seed rate of **0.00**: there is nothing
to transfer, and no amount of transfer analysis explains it. Reporting both as
"0.00 held-out" would be true and useless, which is the failure R5.1's gate
names.

Four readings, from the same pair of rows:

  * a **transfer** question -- clears the bar on familiar layouts, misses
    structurally. `restore_power` at 50k steps: 1.00 familiar, 0.66 structural.
  * a **learning** question -- misses on familiar layouts too, so the structural
    miss is not a transfer result. `repair_belt`: 0.12 familiar, 0.06 structural.
  * **both open** -- clears neither bar, so it is neither a clean transfer miss
    nor a clean learning miss. `restore_power` at 25k steps: 0.70 familiar, 0.37
    structural. This band matters because the first version of this tool folded
    it into "transfer", which claimed the policy had learned its distribution
    when 0.70 does not establish that.
  * **no open question** -- clears the bar structurally. Nothing here does.

The per-seed spread is read separately, because it answers a different question
and needs the seeds merged across files to be visible at all.

Two findings that are properties, not defects
---------------------------------------------
**The argmax penalty is large and systematic.** Greedy evaluation scores below
sampled in every row of every run here, and on one `restore_power` run by 0.03
against 0.70 -- a factor of 23. Sutton & Barto's Example 13.1 is the reference
case: where states are aliased under function approximation the optimal policy
is *stochastic*, and the two deterministic choices score -44 and -82 against
the stochastic optimum's -11.6. So the gap is evidence about argmax under
partial observability rather than a defect to explain away, and R5.1's gate
requires both arms labelled with neither substituted after seeing the better
score. This report therefore never prints one without the other.

**A three-seed spread of 0.77/0.00/0.00 is not transfer variance.** All three
seeds scored 0.975 or better on unfamiliar seeds, so all three learned. What
differed is *which* policy they learned, and `restore_power.py`'s own comment
gives the two candidates that fit training equally well: "fill the hole between
two poles", which scores exactly 0.00 on that holdout by construction, and
"walk to the published gap and place", which transfers. A spread that bimodal
between 0.77 and 0.00 -- with nothing between -- is the shape of a lottery over
two policies, not a distribution around a mean. Reporting the mean of 0.26
would describe a policy no seed found.

Seeing this at all requires merging the release files first, which is why the
spread is computed on (task, holdout) rather than per file. Seed 1's 0.77 lives
in `phase4-release-v3-restore_power.json` and seeds 2 and 3's 0.00s live in
`-seeds23.json`, so a per-file reading sees a single 0.77 and a pair of 0.00s
and reports both as unremarkable. The finding exists only in the union.

`deliver` is the contrast that makes the distinction worth drawing: 0.54, 0.85,
0.92 across three seeds, span 0.38 with a value in the middle. That is variance
around a mean, and its mean of 0.77 does describe something a seed produced.

What this report cannot do, and why
-----------------------------------
It cannot localise *where* in an episode the attempts failed, because the
per-scene rows do not exist for any of these runs. `evaluate_parallel` has
emitted `per_scene` since `623f7dd`, committed at 02:22 on 2026-09-09, and every
result cited here predates it. So the honest scope of this half of R5-B is the
aggregate rows; the localisation half needs a re-run, and
`tools/diagnose_run.py` is the consumer waiting for it.

Two provenance limits bound every number, and neither is small. The three-seed
spreads were measured against **`holdout_v2`**, not the live `holdout_v3`, so
they are not comparable to anything measured today. And `restore_power` yields
only **70 distinct scenes across its 100 frozen episodes**, so its effective
sample is smaller than the count suggests.

Run: uv run python tools/learning_diagnosis.py [--out docs/evidence/r5-learning-diagnosis.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs" / "evidence"

#: Published run results (copies of a `result.json`) carrying both evaluation
#: arms. The runs themselves are gitignored and absent from this machine, so
#: these copies are the only surviving form.
RUN_FILES = (
    "restore_power-curriculum-holdout_v3.json",
    "repair_belt-curriculum-holdout_v3.json",
    "restore_power-v1.6.0-holdout_v3.json",
)

#: Release matrices, which carry the per-seed structural rates and the
#: unfamiliar-seed row that separates a transfer miss from a learning miss.
RELEASE_FILES = (
    "phase4-release-v3-restore_power.json",
    "phase4-release-v3-restore_power-seeds23.json",
    "phase4-release-v3-repair_belt.json",
    "phase4-release-v3-deliver.json",
)

ROWS = ("seeds", "val", "structures")

#: PLAN 4.5's bar. Used here only to say which side of it a row falls on, never
#: as a pass mark for this report -- nothing here is an acceptance claim.
THRESHOLD = 0.8


def load(name: str) -> dict | None:
    path = EVIDENCE / name
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def arms(row: dict | None) -> dict:
    """Both evaluation arms, always together.

    R5.1's gate: both are labeled and neither is substituted after seeing a
    better score. Returning them as one object is the mechanism -- a caller
    cannot print the flattering one without the other in hand.
    """
    row = row or {}
    greedy = row.get("success_rate")
    sampled = (row.get("stochastic") or {}).get("success_rate")
    gap = round(sampled - greedy, 4) if greedy is not None and sampled is not None else None
    return {
        "greedy": greedy,
        "sampled": sampled,
        "sampled_minus_greedy": gap,
        "scored_episodes": row.get("episodes"),
        # Absent for every run in this report, which is the reason the
        # localisation half of R5-B needs a re-run rather than more reading.
        "per_scene_rows": len(row.get("per_scene") or []),
    }


def diagnose_run(data: dict) -> dict:
    evaluation = data.get("evaluation") or {}
    rows = {name: arms(evaluation.get(name)) for name in ROWS}
    familiar = rows["seeds"]["sampled"]
    structural = rows["structures"]["sampled"]
    # The classification R5.1 asks for, and the only inference in this file.
    # It is a statement about *which* question a number poses, not about any
    # mechanism -- naming a mechanism is what the gate forbids.
    if familiar is None or structural is None:
        question = "unclassifiable: a row is missing an arm"
    elif familiar < 0.2:
        question = (
            "a learning question: the policy misses on unfamiliar seeds over its "
            "own training layouts, so the structural miss is not a transfer result"
        )
    elif structural >= THRESHOLD:
        question = "no open question on this row: it clears the threshold structurally"
    elif familiar >= THRESHOLD:
        question = (
            "a transfer question: the policy executes its training distribution and "
            "does not generalise structurally"
        )
    else:
        # The band the first version of this collapsed into "transfer". A run
        # at 0.70 on familiar layouts has not established that it executes its
        # training distribution, so calling its structural miss a transfer
        # result claims something the numbers do not support.
        question = (
            "both questions are open: the policy clears neither threshold, so it "
            "is neither a clean transfer miss nor a clean learning miss"
        )
    return {
        "run_id": data.get("run_id"),
        "task": data.get("task"),
        "total_steps": data.get("total_steps"),
        "final_train_success_rate": data.get("final_train_success_rate"),
        "rows": rows,
        "which_question": question,
    }


def classify_spread(rates: list[float], familiar: list[float]) -> dict:
    """Whether a spread is a distribution around a mean, or a lottery.

    A spread that is bimodal -- widely separated with nothing between -- is not
    variance around its mean, and reporting the mean describes a policy no seed
    found. `restore_power`'s 0.77/0.00/0.00 is the case: all three seeds cleared
    0.975 on unfamiliar seeds, so all three *learned*, and what differed is
    which of two policies they learned. Only one of them transfers.
    """
    if len(rates) < 2:
        return {
            "spread": "one seed, so there is no spread to read",
            "mean_is_representative": None,
        }
    span = max(rates) - min(rates)
    middle = [value for value in rates if 0.2 < value < 0.6]
    all_learned = bool(familiar) and min(familiar) >= 0.9
    if span > 0.5 and not middle:
        return {
            "spread": (
                "bimodal, so not transfer variance: widely separated with nothing "
                "between"
                + (
                    ". Every seed cleared 0.9 on unfamiliar seeds, so every seed "
                    "learned -- what differed is which policy it learned"
                    if all_learned
                    else ""
                )
            ),
            "mean_is_representative": False,
            "span": round(span, 4),
        }
    return {
        "spread": "unimodal: the mean describes something a seed could produce",
        "mean_is_representative": True,
        "span": round(span, 4),
    }


def seed_spread(data: dict) -> list[dict]:
    """Per-seed structural rates, with the unfamiliar-seed row beside them."""
    out = []
    for family, block in (data.get("families") or {}).items():
        per_seed = block.get("structural_success_rate_per_seed") or []
        declared = data.get("seeds") or []
        out.append(
            {
                "task": family,
                "holdout": data.get("cites_holdout_id"),
                "seeds_declared": declared,
                "structural_per_seed": per_seed,
                "structural_mean": block.get("structural_success_rate_mean"),
                "unfamiliar_seed_rate": block.get("unfamiliar_seed_rate_mean"),
                "random_floor": block.get("random_floor_mean"),
                "meets_threshold": block.get("meets_threshold"),
            }
        )
    return out


def build() -> dict:
    runs, missing = [], []
    for name in RUN_FILES:
        data = load(name)
        if data is None:
            missing.append(name)
            continue
        runs.append({"source": name, **diagnose_run(data)})

    # Merged on (task, holdout) *before* the spread is judged. The three-seed
    # restore_power result is split across two files -- seed 1 at 0.77 in one,
    # seeds 2 and 3 at 0.00 in the other -- so a per-file view can never see
    # the spread that is the whole finding, and reported each file as
    # unremarkable.
    per_file = []
    for name in RELEASE_FILES:
        data = load(name)
        if data is None:
            missing.append(name)
            continue
        for row in seed_spread(data):
            per_file.append({"source": name, **row})

    merged: dict[tuple, dict] = {}
    for row in per_file:
        key = (row["task"], row["holdout"])
        block = merged.setdefault(
            key,
            {
                "task": row["task"],
                "holdout": row["holdout"],
                "sources": [],
                "seeds_declared": [],
                "structural_per_seed": [],
                "unfamiliar_seed_rate": [],
            },
        )
        block["sources"].append(row["source"])
        block["seeds_declared"].extend(row["seeds_declared"])
        block["structural_per_seed"].extend(row["structural_per_seed"])
        if row["unfamiliar_seed_rate"] is not None:
            block["unfamiliar_seed_rate"].append(row["unfamiliar_seed_rate"])

    spreads = []
    for block in merged.values():
        rates = sorted(block["structural_per_seed"])
        spreads.append(
            {
                **block,
                "structural_per_seed": rates,
                "seeds_declared": sorted(set(block["seeds_declared"])),
                "unfamiliar_seed_rate": (
                    round(min(block["unfamiliar_seed_rate"]), 4)
                    if block["unfamiliar_seed_rate"]
                    else None
                ),
                "structural_mean": round(sum(rates) / len(rates), 4) if rates else None,
                **classify_spread(rates, block["unfamiliar_seed_rate"]),
                "rates_reported_for_every_declared_seed": len(rates)
                == len(set(block["seeds_declared"])),
            }
        )

    gaps = [
        {
            "source": run["source"],
            "task": run["task"],
            "row": name,
            "greedy": row["greedy"],
            "sampled": row["sampled"],
            "ratio": round(row["sampled"] / row["greedy"], 1)
            if row["greedy"]
            else "greedy scored zero, so the ratio is undefined",
        }
        for run in runs
        for name, row in run["rows"].items()
        if row["greedy"] is not None and row["sampled"] is not None
    ]
    argmax_never_better = all(
        row["sampled_minus_greedy"] >= 0
        for run in runs
        for row in run["rows"].values()
        if row["sampled_minus_greedy"] is not None
    )

    localisable = [
        f"{run['source']}/{name}"
        for run in runs
        for name, row in run["rows"].items()
        if row["per_scene_rows"]
    ]

    return {
        "report": "R5.1 -- interpretable learning results, from the committed record",
        "gate": (
            "Reports identify where attempts fail. Neither zero success nor high "
            "training success alone is called an algorithmic explanation."
        ),
        "attaches_no_algorithmic_claim": (
            "Every entry classifies which *question* a pair of rows poses. None "
            "names a mechanism, because the gate forbids naming one from a rate"
        ),
        "the_separating_measurement": (
            "the unfamiliar-seed row. A 0.00 structural score with an "
            "unfamiliar-seed rate of 1.00 is a transfer miss; the same 0.00 with a "
            "rate of 0.00 is a learning miss. Reporting both as '0.00 held-out' "
            "would be true and useless"
        ),
        "runs": runs,
        "seed_spreads": spreads,
        "both_arms": {
            "requirement": (
                "both sampled and argmax evaluations are labeled; neither is "
                "substituted after seeing a better score"
            ),
            "sampled_never_scored_below_greedy": argmax_never_better,
            "largest_gaps": sorted(
                (row for row in gaps if isinstance(row["ratio"], (int, float))),
                key=lambda row: -row["ratio"],
            )[:5],
            "reading": (
                "the gap is evidence about argmax under partial observability, not a "
                "defect: where states are aliased under function approximation the "
                "optimal policy is stochastic (Sutton & Barto, Example 13.1)"
            ),
        },
        "cannot_localise_within_an_episode": {
            "rows_with_per_scene_data": localisable,
            "why": (
                "evaluate_parallel has emitted per_scene since 623f7dd (02:22 on "
                "2026-09-09) and every result cited here predates it, so no row "
                "carries scene-level outcomes. The localisation half of R5-B needs "
                "a re-run; tools/diagnose_run.py is the consumer waiting for it"
            ),
        },
        "provenance_limits": [
            "the three-seed spreads were measured against holdout_v2, not the live "
            "holdout_v3, so they are not comparable to anything measured today",
            "restore_power yields only 70 distinct scenes across its 100 frozen "
            "episodes, so its effective sample is smaller than the count suggests",
            "the curriculum fraction is recorded only in manifest.json, and those "
            "manifests are absent from this machine, so no result here can be tied "
            "to the curriculum setting that produced it",
        ],
        "missing_sources": missing,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=None, help="write the report as JSON")
    args = parser.parse_args()
    report = build()
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
