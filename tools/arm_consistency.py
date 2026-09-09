"""Do two arms solve the *same* scenes, or the same *number* of scenes?

A held-out rate cannot tell these apart, and `shaping_comparison`'s own pairing
check does not either: it verifies the arms scored the identical frozen episode
set -- necessary, and the reason a verdict is refused without it -- but it never
compares outcomes scene by scene. So two arms can post identical rates while
disagreeing about a third of the holdout, and nothing in the pipeline would
say so.

The 4.4 comparison is the case that motivated this. Seed 1's shaped and sparse
arms both scored **0.84** held-out with byte-identical Wilson intervals, which
reads as one result measured twice. They agreed on 82 of 100 scenes: 9 solved
only by shaped, 9 only by sparse. Seed 2 disagreed on only 5. That difference
between seeds is itself information a pooled rate discards.

Why per-seed McNemar is not enough
----------------------------------
Run per seed, McNemar answers "did one arm win *this* draw", and on 9-against-9
it correctly answers no. But it is blind to the question that matters here:
whether the arms consistently differ on the *same* scenes across seeds. Scenes
3002 and 3065 were solved by sparse and missed by shaped in both finished
seeds. Three independent McNemar tests cannot see that, because each throws
away the scene identities the moment it reduces to two counts.

The test here conditions on each scene's own difficulty. For every frozen scene
it counts how many of the n seeds each arm solved it in, then asks whether the
arms differ scene-by-scene more than exchangeable arm labels would produce.
Under the null -- shaping changes nothing about which scenes are solvable --
the 2n outcomes for a scene are exchangeable between arms, so shuffling arm
labels *within* each scene, preserving that scene's total solves, gives the
exact conditional null. That is a permutation test, not an approximation, and
it is conditioned so that easy and hard scenes contribute on their own terms.

The statistic is the mean absolute per-scene disagreement. It is deliberately
insensitive to the direction of any difference: a rate comparison already
answers "which arm is better", and this answers the different question of
whether the arms are solving different problems.

Read the perfect-split count against the right baseline
-------------------------------------------------------
A scene solved by every run of one arm and none of the other looks like a
finding and usually is not. Under the null it arises with probability
2/C(2n, n) -- 0.1 at n=3 -- but *only for scenes that could split at all*, and
that is the part worth getting right. The within-scene shuffle preserves each
scene's total solves, so a scene every run solved cannot split under any
shuffle. Only scenes whose total is exactly n are at risk.

On this holdout most scenes are solved by every run, so the unconditional
figure is badly wrong: 100 scenes x 0.333 at n=2 gives 33 expected splits
against 2 observed, which reads as the arms agreeing far more than chance. The
conditional count compares 2 observed against the handful of scenes that were
actually at risk, which is a different and much less comfortable reading. The
report gives the conditional number and shows the unconditional one beside it
so the difference is visible.

The named scenes are leads to inspect, not findings. The p-value is over the
aggregate statistic.

Run:
  uv run python tools/arm_consistency.py --arm shaped=id1,id2,id3 --arm sparse=id4,id5,id6
  uv run python tools/arm_consistency.py --from docs/evidence/phase4-shaping-comparison.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from factoriorl import manifest as manifest_module  # noqa: E402

#: Enough shuffles that a p-value near the conventional cut is stable to ~0.005,
#: and cheap: the statistic is a sum over at most a few hundred scenes.
SHUFFLES = 20_000

#: The row this is about. `structures` is the frozen structural holdout, the
#: only row where every arm is guaranteed to have scored the same scenes.
DEFAULT_ROW = "structures"


def load(run: str) -> dict:
    candidate = Path(run)
    if candidate.is_file():
        return json.loads(candidate.read_text(encoding="utf-8"))
    path = manifest_module.runs_dir() / run / "result.json"
    if not path.is_file():
        raise SystemExit(f"no result for {run!r}: not a file, and no {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def outcomes(run: str, row: str) -> dict[int, bool]:
    """Scene identity -> solved, for one run's evaluation row."""
    data = load(run)
    scenes = ((data.get("evaluation") or {}).get(row) or {}).get("per_scene") or []
    found = {
        int(scene["episode_index"]): bool(scene.get("success"))
        for scene in scenes
        if scene.get("episode_index") is not None
    }
    if not found:
        raise SystemExit(
            f"{run} carries no per_scene rows for {row!r}. Runs before 623f7dd "
            "recorded none, so a per-scene comparison is not available for them"
        )
    return found


def solves_per_scene(arms: dict[str, list[str]], row: str) -> tuple[dict, list[int]]:
    """For each arm, how many of its runs solved each scene.

    Only scenes every run scored are kept. A scene one arm never saw cannot be
    evidence about a difference between arms, and silently treating an absent
    scene as a failure would manufacture disagreement.
    """
    per_run: dict[str, dict[int, dict[int, bool]]] = {}
    for arm, runs in arms.items():
        per_run[arm] = {index: outcomes(run, row) for index, run in enumerate(runs)}

    scene_sets = [set(result) for arm_runs in per_run.values() for result in arm_runs.values()]
    shared = sorted(set.intersection(*scene_sets)) if scene_sets else []

    counts = {
        arm: {scene: sum(int(result[scene]) for result in arm_runs.values()) for scene in shared}
        for arm, arm_runs in per_run.items()
    }
    return counts, shared


def permutation_test(first: list[list[bool]], second: list[list[bool]], seed: int = 0) -> dict:
    """Do the arms disagree scene-by-scene more than exchangeable labels would?

    `first[i]` and `second[i]` are one scene's per-seed outcomes under each arm.
    Under the null the 2n outcomes for a scene are exchangeable between arms, so
    the shuffle is *within* each scene -- which conditions on that scene's own
    difficulty and lets easy and hard scenes contribute on their own terms. A
    shuffle across scenes would instead test whether the arms differ on
    average, which is what the rate comparison already answers.
    """

    def statistic(a: list[list[bool]], b: list[list[bool]]) -> float:
        return sum(abs(sum(x) - sum(y)) for x, y in zip(a, b, strict=True)) / max(len(a), 1)

    observed = statistic(first, second)
    rng = random.Random(seed)
    at_least = 0
    for _ in range(SHUFFLES):
        shuffled_a, shuffled_b = [], []
        for x, y in zip(first, second, strict=True):
            pool = list(x) + list(y)
            rng.shuffle(pool)
            shuffled_a.append(pool[: len(x)])
            shuffled_b.append(pool[len(x) :])
        if statistic(shuffled_a, shuffled_b) >= observed:
            at_least += 1
    # +1 to numerator and denominator: the observed arrangement is itself one
    # sample from the null, so a p-value of exactly 0 is not attainable and
    # should not be reported.
    return {
        "statistic": "mean absolute per-scene disagreement in solve count",
        "observed": round(observed, 4),
        "shuffles": SHUFFLES,
        "p_value": round((at_least + 1) / (SHUFFLES + 1), 4),
        "null": (
            "shaping changes nothing about which scenes are solvable, so a scene's "
            "outcomes are exchangeable between arms. Shuffled within each scene, "
            "which conditions on that scene's difficulty"
        ),
        "two_sided": (
            "the statistic is absolute, so this asks whether the arms solve "
            "different scenes -- not which arm is better, which the rates answer"
        ),
    }


def build(arms: dict[str, list[str]], row: str) -> dict:
    if len(arms) != 2:
        raise SystemExit(f"exactly two arms, got {sorted(arms)}")
    counts, shared = solves_per_scene(arms, row)
    (first_name, first_counts), (second_name, second_counts) = sorted(counts.items())
    sizes = {arm: len(runs) for arm, runs in arms.items()}

    per_run = {arm: [outcomes(run, row) for run in runs] for arm, runs in arms.items()}
    first = [[result[scene] for result in per_run[first_name]] for scene in shared]
    second = [[result[scene] for result in per_run[second_name]] for scene in shared]

    disagreement = defaultdict(list)
    for scene in shared:
        gap = first_counts[scene] - second_counts[scene]
        disagreement[gap].append(scene)

    # A scene solved by every run of one arm and no run of the other. Named as
    # leads, not findings: with n=3 a perfect split arises under the null with
    # probability 0.1, so ~10 of 100 scenes split this way by chance.
    always_first = [
        scene
        for scene in shared
        if first_counts[scene] == sizes[first_name] and second_counts[scene] == 0
    ]
    always_second = [
        scene
        for scene in shared
        if second_counts[scene] == sizes[second_name] and first_counts[scene] == 0
    ]

    # How many scenes could have split perfectly at all, and how many would be
    # expected to. Conditioning matters more than the arithmetic here: on this
    # holdout most scenes are solved by every run, and those cannot split under
    # any within-scene shuffle. Comparing the observed count against an
    # unconditional expectation reads the evidence backwards -- it made two
    # consistent disagreements look like far fewer than chance, when the honest
    # comparison is against the scenes that were actually at risk.
    total_runs = sizes[first_name] + sizes[second_name]
    per_arm = sizes[first_name]
    splittable = sum(1 for scene in shared if first_counts[scene] + second_counts[scene] == per_arm)
    expected_splits = round(splittable * 2 / _choose(total_runs, per_arm), 2)

    return {
        "question": "do the arms solve the same scenes, or the same number of scenes?",
        "row": row,
        "arms": {arm: {"runs": runs, "seeds": len(runs)} for arm, runs in arms.items()},
        "shared_scenes": len(shared),
        "solved_by_arm": {arm: sum(scene_counts.values()) for arm, scene_counts in counts.items()},
        "rate_by_arm": {
            arm: round(sum(scene_counts.values()) / (len(shared) * sizes[arm]), 4)
            if shared
            else None
            for arm, scene_counts in counts.items()
        },
        "scenes_by_disagreement": {
            str(gap): {"scenes": len(scenes), "episode_indices": sorted(scenes)[:20]}
            for gap, scenes in sorted(disagreement.items())
        },
        "always_only_in": {
            first_name: sorted(always_first),
            second_name: sorted(always_second),
        },
        "perfect_splits": {
            "observed": len(always_first) + len(always_second),
            "expected_under_null": expected_splits,
            "splittable_scenes": splittable,
            "why_conditional": (
                "a scene can only split perfectly if its total solves across both "
                "arms is exactly the number of seeds per arm. A scene every run "
                "solved cannot split under any shuffle, so the unconditional "
                f"count -- {len(shared)} scenes x 2/C({total_runs},{sizes[first_name]}) "
                f"= {round(len(shared) * 2 / _choose(total_runs, sizes[first_name]), 1)} "
                "-- overstates the null by counting scenes that were never at risk"
            ),
        },
        "permutation_test": permutation_test(first, second),
        "why_not_per_seed_mcnemar": (
            "run per seed, McNemar asks whether one arm won that draw and correctly "
            "answers no on a 9-against-9 split. It cannot see whether the arms "
            "differ on the *same* scenes across seeds, because it discards the "
            "scene identities when it reduces to two counts"
        ),
    }


def _choose(n: int, k: int) -> int:
    import math

    return math.comb(n, k)


def parse_arm(value: str) -> tuple[str, list[str]]:
    if "=" not in value:
        raise SystemExit(f"--arm wants name=run1,run2 -- got {value!r}")
    name, runs = value.split("=", 1)
    return name.strip(), [run.strip() for run in runs.split(",") if run.strip()]


def arms_from_report(path: str) -> dict[str, list[str]]:
    """Recover the arms from a `shaping_comparison` report.

    Its cells are named `shaped-…` / `sparse-…`, so the arm is the run id's own
    prefix. Read from the report rather than from `runtime/runs/` so the set is
    the one the comparison actually published.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    arms: dict[str, list[str]] = defaultdict(list)
    for run in data.get("runs") or []:
        run_id = run.get("run_id") if isinstance(run, dict) else run
        if not run_id or "-" not in run_id:
            continue
        arms[run_id.split("-", 1)[0]].append(run_id)
    if len(arms) != 2:
        raise SystemExit(
            f"expected two arm prefixes in {path}, found {sorted(arms)}. Pass --arm "
            "explicitly instead"
        )
    return dict(arms)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--arm",
        action="append",
        default=[],
        help="name=run1,run2,run3 -- give this twice, once per arm",
    )
    parser.add_argument(
        "--from",
        dest="from_report",
        default=None,
        help="recover the arms from a shaping_comparison report",
    )
    parser.add_argument("--row", default=DEFAULT_ROW, help=f"default {DEFAULT_ROW}")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.from_report:
        arms = arms_from_report(args.from_report)
    elif args.arm:
        arms = dict(parse_arm(value) for value in args.arm)
    else:
        raise SystemExit("pass --arm twice, or --from a shaping_comparison report")

    report = build(arms, args.row)
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
