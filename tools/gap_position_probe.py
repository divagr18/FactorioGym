"""Name the matched scene sets R5.2's shortcut probe needs, with no engine.

R5.2 asks whether a `restore_power` policy learned *"walk to the published gap
and place"* or the shortcut *"fill the hole between two poles"*. The two fit
training equally well and only the first transfers, which is why one seed
scored 0.77 and two scored 0.00 -- a lottery over which policy a seed found,
not transfer variance. Answering it needs the same checkpoint evaluated on
scenes that differ in **where the gap sits** and in as little else as possible.

Where the frozen holdout cannot help
------------------------------------
The obvious matched set does not exist. `holdout_v3` scores `restore_power` on
100 episodes that are **all** `gap_near_drill`, and that family is 600/600
terminal in `holdout-distribution-audit.json`: there is no interior scene in the
holdout to match a terminal one against. Neither can the training families
supply the other half -- `pole_gap` and `pole_gap_far` are 600/600 *interior*,
so pairing them against the holdout would confound gap position with the
train/test split and with two different generators. That comparison would
answer nothing.

`two_gaps` is the family where both regimes coexist. It places two faults, and
the flanking-side signature splits three ways -- both gaps interior, one of
each, both terminal -- inside one family, one split, one generator and one task
version. That looked like the matched design R5.2 wants, and better than the
binary version, because the mixed class separates "needs an interior gap" from
"needs *only* interior gaps".

Measuring the confounds killed most of it, which is the finding
---------------------------------------------------------------
The three classes are not matched on anything else. Over 1,200 scenes, keyed on
(class, chain length, fault separation):

    both_terminal  chain 5  sep 4.0   135      both_interior  chain 6  sep 8.0   95
    both_terminal  chain 5  sep 8.0   261      both_interior  chain 7  sep 8.0  164
    both_terminal  chain 6  sep 4.0   101      one_of_each    chain 6  sep 8.0  202
    both_terminal  chain 7  sep 4.0    83      one_of_each    chain 7  sep 8.0  159

`both_terminal` never appears with a separation of 8 unless the chain is length
5, and `both_interior` never appears with a chain of length 5 at all. The
entanglement is structural rather than a sampling accident: on a short chain
both gaps are necessarily at its ends. **So the interior-versus-terminal
comparison R5.2 asks for cannot be built from `two_gaps` either** -- any such
contrast varies chain length and fault separation alongside gap position, and a
difference in outcome would be unattributable.

One matched contrast does survive, and the tool reports it as the only one:
`both_interior` against `one_of_each` at a **fixed** chain length and
separation -- 164 against 159 at chain 7, 95 against 202 at chain 6. Same chain,
same separation, differing only in whether one of the two gaps sits at an end.
That tests whether a policy needs *both* gaps interior, which is a weaker
question than R5.2's but is the strongest one this generator can pose. Widening
it needs a generator change: a family that places a long chain with both gaps
terminal.

Why this is a scene definition and nothing more
-----------------------------------------------
This tool emits scene sets. It makes no claim about any policy, and it cannot:
R5.1's gate forbids naming an algorithmic explanation from a rate, and a
generator distribution is not evidence about behaviour either -- R5.2 says so
outright, requiring that policies' actual actions be confirmed "rather than
inferring them solely from generator distributions".

What it does provide is the confound check above, which is why the design
changed before a run was launched rather than after. Each scene records the
character's distance to each fault, the separation between faults, and the
chain length; scenes are then grouped into strata that hold the confounds
fixed, and a contrast is reported as usable only where two classes actually
co-occupy a stratum with enough scenes in each. A class pair that shares no
stratum is reported as unusable and named, rather than compared anyway across
differing chain lengths.

The subsets are declared, not frozen holdout entries. They are pinned to an
explicit seed plan -- `master`, `run_id`, `branch` -- because a scene is a pure
function of those plus the episode index, and `run_id` enters the seed. Reusing
`holdout_v3`'s plan would draw the probe's scenes from the same stream as a
scored holdout, so the probe declares its own `run_id` and the blueprint digests
recorded here are what a later run must reproduce.

Run:
  uv run python tools/gap_position_probe.py
  uv run python tools/gap_position_probe.py --task restore_power --family two_gaps \\
      --samples 400 --out docs/evidence/r5-gap-position-probe.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from factoriorl import tasks  # noqa: E402
from factoriorl.seeding import Branch, SeedPlan  # noqa: E402
from factoriorl.tasks.coverage import fault_positions, flanking_sides  # noqa: E402

#: The probe's own seed stream. Deliberately not `holdout-v3`: a scene is a
#: pure function of (master, run_id, branch, index), so sharing a `run_id` with
#: a scored holdout would draw probe scenes from the same stream as reported
#: results. A distinct `run_id` keeps the two disjoint by construction.
PROBE_RUN_ID = "r5-gap-position-probe-v1"
PROBE_MASTER = 20260909

#: What a flanking-side count means (`coverage.flanking_sides`): 2 = more of the
#: fault's own chain on both sides, 1 = on one side only, 0 = no chain aligned.
CLASS_NAMES = {
    (2,): "interior",
    (1,): "terminal",
    (2, 2): "both_interior",
    (1, 2): "one_of_each",
    (1, 1): "both_terminal",
}


def class_name(signature: tuple[int, ...]) -> str:
    return CLASS_NAMES.get(signature, "sides_" + "_".join(str(s) for s in signature))


def digest(payload: dict) -> str:
    return hashlib.blake2b(
        json.dumps(payload, sort_keys=True).encode("utf-8"), digest_size=8
    ).hexdigest()


def describe_scene(task, family, plan: SeedPlan, index: int) -> dict | None:
    """One scene's gap-position class, plus the descriptors it might confound with.

    The RNG dance mirrors `FactorioEnv.prepare_scene`: it draws
    `randrange(len(families_in_split))` to pick the family and hands the *same,
    already-advanced* generator to `task.generate`. Sampling with a fresh RNG
    would explore scenes the environment never produces.
    """
    split_size = max(1, len(task.spec.families(family.split)))
    rng = plan.generator_rng(Branch.EVAL, index)
    rng.randrange(split_size)
    blueprint = task.generate(family, rng)
    faults = fault_positions(task, blueprint)
    if not faults:
        return None

    sides = sorted(flanking_sides(blueprint, fault) for fault in faults)
    character = blueprint.character_position
    distances = [round(math.dist(character, fault), 3) for fault in faults]
    # The chain the gaps sit in. Its length is what a terminal gap is terminal
    # *of*, so a class that systematically has shorter chains differs in more
    # than gap position.
    prototypes = Counter(entity.name for entity in blueprint.entities)
    chain, chain_length = prototypes.most_common(1)[0] if prototypes else (None, 0)
    return {
        "episode_index": index,
        "layout_family": family.name,
        "split": family.split,
        "flanking_sides": sides,
        "class": class_name(tuple(sides)),
        "blueprint_digest": digest(blueprint.to_dict(public_markers=task.spec.public_markers)),
        # Confounds, recorded so a later comparison can be read. Gap position
        # correlates with distance if terminal gaps sit at the far end.
        "distance_to_faults": distances,
        "nearest_fault": min(distances),
        "farthest_fault": max(distances),
        "fault_separation": (
            round(math.dist(faults[0], faults[1]), 3) if len(faults) > 1 else None
        ),
        "chain_prototype": chain,
        "chain_length": chain_length,
    }


def summarise(scenes: list[dict], keys: tuple[str, ...]) -> dict:
    """Median and range of each confound descriptor, for one class."""
    out: dict = {"scenes": len(scenes)}
    for key in keys:
        values = [s[key] for s in scenes if isinstance(s.get(key), (int, float))]
        if not values:
            continue
        out[key] = {
            "median": round(statistics.median(values), 3),
            "min": round(min(values), 3),
            "max": round(max(values), 3),
        }
    return out


CONFOUNDS = ("nearest_fault", "farthest_fault", "fault_separation", "chain_length")

#: Held fixed within a stratum. These are the two descriptors measured to vary
#: with the gap-position class, so a contrast is only readable inside one cell
#: of them. Distance to the faults is *not* held fixed: it is continuous and
#: partly a consequence of the other two, and stratifying on it would empty
#: every cell.
STRATUM_KEYS = ("chain_length", "fault_separation")


def stratum_of(scene: dict) -> tuple:
    return tuple(scene.get(key) for key in STRATUM_KEYS)


def build(
    task_id: str,
    family_name: str,
    samples: int,
    per_class: int | None,
    minimum: int = 20,
) -> dict:
    task = tasks.get(task_id)
    families = {family.name: family for family in task.spec.layout_families}
    if family_name not in families:
        raise SystemExit(f"{task_id} has no family {family_name!r}; it has {sorted(families)}")
    family = families[family_name]
    if not task.spec.fault_markers:
        raise SystemExit(
            f"{task_id} declares no fault markers, so there is no gap position to classify"
        )

    plan = SeedPlan(master=PROBE_MASTER, run_id=PROBE_RUN_ID)
    scenes = [
        scene
        for scene in (describe_scene(task, family, plan, index) for index in range(samples))
        if scene is not None
    ]

    by_class: dict[str, list[dict]] = defaultdict(list)
    for scene in scenes:
        by_class[scene["class"]].append(scene)

    # Strata hold the confounds fixed. Grouping by class alone produced classes
    # that differ in chain length and fault separation as much as in gap
    # position, so a contrast across them measures three things at once.
    by_stratum: dict[tuple, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for scene in scenes:
        by_stratum[stratum_of(scene)][scene["class"]].append(scene)

    strata: dict[str, dict] = {}
    usable: list[dict] = []
    unusable: list[dict] = []
    for key, classes in sorted(by_stratum.items(), key=lambda item: str(item[0])):
        label = f"chain={key[0]},separation={key[1]}"
        eligible = {name: rows for name, rows in classes.items() if len(rows) >= minimum}
        size = min((len(rows) for rows in eligible.values()), default=0)
        if per_class:
            size = min(size, per_class)
        comparable = len(eligible) > 1
        strata[label] = {
            "classes_present": {name: len(rows) for name, rows in sorted(classes.items())},
            "classes_usable": sorted(eligible) if comparable else [],
            "subset_size": size if comparable else 0,
            "subsets": (
                {
                    name: {
                        "episode_indices": [row["episode_index"] for row in rows[:size]],
                        "blueprint_digests": [row["blueprint_digest"] for row in rows[:size]],
                        "confounds": summarise(rows[:size], CONFOUNDS),
                    }
                    for name, rows in sorted(eligible.items())
                }
                if comparable
                else {}
            ),
        }
        if comparable:
            for first, second in combinations(sorted(eligible), 2):
                usable.append({"stratum": label, "contrast": [first, second], "each": size})

    # Every class pair that shares no stratum, named. These are the comparisons
    # a reader would otherwise assume are available.
    shared = {
        frozenset(pair)
        for block in strata.values()
        for pair in combinations(block["classes_usable"], 2)
    }
    for first, second in combinations(sorted(by_class), 2):
        if frozenset((first, second)) not in shared:
            unusable.append(
                {
                    "contrast": [first, second],
                    "why": (
                        "no stratum holds both at a fixed chain length and fault "
                        "separation, so any comparison varies those alongside gap "
                        "position and its result is unattributable"
                    ),
                }
            )

    return {
        "probe": "does a restore_power policy need the gap to be interior?",
        "declares": (
            "matched scene sets differing in where the fault sits, within one "
            "layout family, one split and one generator"
        ),
        "claims_nothing_about_any_policy": (
            "These are scene definitions. R5.2 requires that policies' actual "
            "actions be confirmed rather than inferred from generator "
            "distributions, so nothing here is evidence about behaviour"
        ),
        "task": {"id": task_id, "version": task.spec.version},
        "family": {"name": family_name, "split": family.split},
        "fault_markers": list(task.spec.fault_markers),
        "faults_are_published": [
            name for name in task.spec.fault_markers if name in task.spec.public_markers
        ],
        "seed_plan": {
            "master": PROBE_MASTER,
            "run_id": PROBE_RUN_ID,
            "branch": Branch.EVAL.value,
            "episode_seed_algorithm": "blake2b(master|run_id|branch|index)[:8]",
            "why_not_the_holdout_plan": (
                "a scene is a pure function of (master, run_id, branch, index), so "
                "sharing holdout_v3's run_id would draw probe scenes from the same "
                "stream as scored results"
            ),
        },
        "sampled": len(scenes),
        "class_counts": {name: len(rows) for name, rows in sorted(by_class.items())},
        "stratified_by": list(STRATUM_KEYS),
        "minimum_per_class": minimum,
        "strata": strata,
        "usable_contrasts": usable,
        "unusable_contrasts": unusable,
        "confound_check": {
            "descriptors": list(CONFOUNDS),
            "why_stratified": (
                "grouped by class alone, both_terminal scenes have a median chain "
                "length of 5 against both_interior's 7 and a median fault separation "
                "of 4 against 8. Those are structurally different scenes, not "
                "repositioned gaps, so the classes are compared only within a "
                "stratum that holds both fixed"
            ),
            "structural_entanglement": (
                "on a short chain both gaps are necessarily at its ends, so "
                "both_terminal never occurs with a separation of 8 unless the chain "
                "is length 5, and both_interior never occurs at length 5 at all. "
                "That is why the interior-versus-terminal contrast R5.2 asks for is "
                "absent here and not merely undersampled"
            ),
        },
        "not_the_frozen_holdout": (
            "holdout_v3 scores restore_power on 100 episodes that are all "
            "gap_near_drill, and that family is 600/600 terminal, so the holdout "
            "contains no interior scene to match against. The training families are "
            "600/600 interior, so pairing them against the holdout would confound "
            "gap position with the split and with two different generators"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", default="restore_power")
    parser.add_argument(
        "--family",
        default="two_gaps",
        help="the family both regimes coexist in; the default is the only one that does",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1200,
        help="stratifying splits the sample across cells, so the default is larger "
        "than a flat classification would need",
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=None,
        help="cap each subset at this many scenes; default is the smallest class, "
        "so the subsets are equal-sized without discarding more than necessary",
    )
    parser.add_argument(
        "--minimum",
        type=int,
        default=20,
        help="a class needs at least this many scenes in a stratum to be compared "
        "there; below it the cell is reported but not offered as a contrast",
    )
    parser.add_argument("--out", default=None, help="write the report as JSON")
    args = parser.parse_args()

    report = build(args.task, args.family, args.samples, args.per_class, args.minimum)
    print(json.dumps(report, indent=2))
    if args.out:
        path = Path(args.out)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
