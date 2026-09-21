"""Where did the attempts fail, and did two arms differ on the same scenes?

R5.1's gate: *"Reports identify where attempts fail. Neither zero success nor
high training success alone is called an algorithmic explanation."* A success
rate cannot do that, and until now nothing could: `evaluate_parallel` has
written `per_scene` rows since `623f7dd` and **no code in the repository reads
them** -- a grep over `src/` and `tools/` matches only the producer.

Two things this does that a rate cannot.

**It says how episodes ended, not just how many succeeded.** A run that
exhausted its budget and one that tripped a failure predicate are different
diagnoses, and a failure at 17 decisions is a different diagnosis again from
one at the 120th. Those live in `terminated` / `truncated` / `steps` /
`action_error`, per scene.

**It compares two arms on the scenes they actually shared.** DESIGN.md section 3
requires this and gives the reason: *"Paired analysis whenever two arms are
evaluated on the same generated scenes; analysing paired measurements as
independent discards the pairing and widens intervals for no reason."* Two arms
scored on one frozen holdout are paired binary outcomes, so the informative
quantity is the **discordant** scenes -- solved by one arm and not the other --
not the difference of two rates. 84 against 84 can mean "the same 84 scenes" or
"84 each with 30 disagreements", and those are different findings.

The test on discordant pairs is McNemar's, computed exactly rather than with
the chi-square approximation: the counts here are small, and the approximation
is unreliable below about 25 discordant pairs.

**It says which budget ran out, or refuses to say.** `env.py:743` truncates on
*either* of two budgets -- `self._steps >= max_decision_steps` **or** `tick >=
max_game_ticks` -- and `per_scene` records only the boolean, so both arrive as
one undifferentiated `truncated`. They are different diagnoses, and which one
happened looks recoverable without touching the env: truncation requires one of
the two, so a step count short of the decision budget implicates ticks.

On `deliver` that inference runs into a wall, and the wall is the interesting
part. `step` advances exactly `decision_ticks` per decision (`env.py:702`), so
reaching `max_game_ticks` takes at least `max_game_ticks / decision_ticks`
decisions -- 15000/30 = **500** on `deliver`, against a decision budget of
**120**. The tick budget is unreachable by construction. Yet across the two
finished cells of the 4.4 comparison, **103 of 113 failures truncated below 120
steps**, some as low as 12. Neither budget explains those, so the tool reports
them as `truncated_unexplained` rather than picking whichever budget is left.

That is a lead, not a diagnosis, and the candidates are worth writing down
because two of them would invalidate any report built on these rows: the flag
`per_scene` reads is `TimeLimit.truncated`, which `env.py`'s own docstring
records as having once been synthesised for *infrastructure failures* -- making
a worker crash indistinguishable from a time limit -- and the step count is the
evaluation loop's own accumulator, not the env's `info["steps"]`, so the two
could disagree. Neither is confirmed here.

Do not read `simulated_ticks` as evidence about this. It is derived as
`primitive_steps * decision_ticks` (`train.py:1083`), not measured game time,
so ratios computed from it describe two counters and not the clock.

Deliberately not here: any verdict about *why*. This tool localises; naming a
cause from a localisation is the thing R5.1's gate forbids doing from a rate,
and it is not improved by doing it from a histogram.

Run:
  uv run python tools/diagnose_run.py <run_id>
  uv run python tools/diagnose_run.py <shaped_run_id> <sparse_run_id> --row structures
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from factoriorl import manifest as manifest_module  # noqa: E402
from factoriorl import tasks  # noqa: E402

#: Evaluation rows `train.py` writes, in the order a reader should read them:
#: familiar layouts with unfamiliar seeds, then validation, then the structural
#: holdout. Separating the first from the last is R5.1's "familiar-distribution
#: execution versus structural transfer".
ROWS = ("seeds", "val", "structures")


def load(run: str) -> dict:
    """A run id under `runtime/runs/`, or a path to a result file.

    Published evidence under `docs/evidence/` is a copy of a `result.json`, and
    the runs behind those copies are gitignored and mostly absent from any one
    machine -- both curriculum runs, for instance, exist only on the desktop. A
    tool that could read only live run directories therefore could not read the
    committed record at all, which is the only form most of these results
    survive in.
    """
    candidate = Path(run)
    if candidate.is_file():
        return json.loads(candidate.read_text(encoding="utf-8"))
    path = manifest_module.runs_dir() / run / "result.json"
    if not path.is_file():
        raise SystemExit(
            f"no result for {run!r}: not a file, and no {path}. Pass a run id "
            "under runtime/runs/ or a path to a result.json"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def budgets(run_id: str, task_id: str | None) -> dict | None:
    """The budgets *this run* used, preferring the run's own manifest.

    `result.json` records neither the budgets nor even the task version, so
    the registry is the wrong first choice: it would answer with today's spec
    for a run made against an older one, and silently mislabel every
    truncation. `manifest.json` records `task.resolved.budgets` as resolved at
    launch, which is the number the run was actually held to. The registry is
    the fallback, and says so, because a mislabelled split is worse than an
    absent one.
    """
    # A result file passed by path may sit beside its manifest, or may be a
    # published copy with no manifest at all. Both are tried before the
    # registry, since either beats today's spec for an older run.
    candidate = Path(run_id)
    manifest_path = (
        candidate.with_name("manifest.json")
        if candidate.is_file()
        else manifest_module.runs_dir() / run_id / "manifest.json"
    )
    if manifest_path.is_file():
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            task = data.get("task") or {}
            spec = task.get("resolved") or {}
            resolved = spec.get("budgets") or {}
            if resolved.get("max_decision_steps"):
                return _budget_block(
                    int(resolved["max_decision_steps"]),
                    int(resolved.get("max_game_ticks") or 0),
                    int(spec.get("decision_ticks") or 0),
                    task.get("version"),
                    "this run's manifest",
                )
        except (ValueError, KeyError, TypeError):
            pass
    if not task_id:
        return None
    try:
        spec = tasks.get(task_id).spec
    except Exception:
        # A result may name a task this tree no longer registers. That costs
        # the budget split and nothing else, so it must not be fatal.
        return None
    return _budget_block(
        int(spec.max_decision_steps),
        int(spec.max_game_ticks),
        int(getattr(spec, "decision_ticks", 0) or 0),
        spec.version,
        "the registry as it stands today, because this run has no manifest. "
        "If the task version moved since the run, the split is wrong",
    )


def _budget_block(
    decisions: int, ticks: int, decision_ticks: int, version: str | None, source: str
) -> dict:
    """The budgets, plus whether the tick budget is reachable at all.

    A decision advances exactly `decision_ticks` (`env.py:702`), so the tick
    budget cannot bind before `max_game_ticks / decision_ticks` decisions. When
    that exceeds the decision budget the tick budget is dead: it can never be
    the cause of a truncation, and a short truncation is therefore explained by
    *neither* budget. Computing this is what stops the tool from labelling
    those episodes "ran out of game time" by elimination.
    """
    block = {
        "max_decision_steps": decisions,
        "max_game_ticks": ticks,
        "decision_ticks": decision_ticks,
        "task_version": version,
        "source": source,
    }
    if decision_ticks > 0 and ticks > 0:
        needed = math.ceil(ticks / decision_ticks)
        block["decisions_to_exhaust_ticks"] = needed
        block["tick_budget_reachable"] = needed <= decisions
        if needed > decisions:
            block["note"] = (
                f"the tick budget needs {needed} decisions at {decision_ticks} ticks "
                f"each but the decision budget stops at {decisions}, so it can never "
                "bind -- a truncation below the decision budget is explained by "
                "neither budget and is reported as unexplained"
            )
    return block


def outcome(
    row: dict,
    decision_budget: int | None = None,
    tick_budget_reachable: bool = True,
) -> str:
    """How one episode ended, as mutually exclusive answers.

    `truncated` splits by which budget expired. `env.py` truncates on either,
    so a `truncated` episode whose step count is short of the decision budget
    implicates game ticks -- *unless* the tick budget is unreachable, in which
    case nothing explains it and the honest answer is to say so rather than
    name the only remaining option.
    """
    if row.get("success"):
        return "solved"
    if row.get("truncated"):
        if decision_budget is None:
            return "out_of_budget"
        if int(row.get("steps") or 0) >= decision_budget:
            return "out_of_decisions"
        return "out_of_game_time" if tick_budget_reachable else "truncated_unexplained"
    if row.get("terminated"):
        return "failure_predicate"
    return "unknown"


def diagnose(scenes: list[dict], task_budgets: dict | None = None) -> dict:
    """Where the attempts in one evaluation row ended up."""
    if not scenes:
        return {"scenes": 0, "note": "no per_scene rows; this run predates 623f7dd"}
    decision_budget = task_budgets.get("max_decision_steps") if task_budgets else None
    reachable = task_budgets.get("tick_budget_reachable", True) if task_budgets else True
    outcomes = Counter(outcome(scene, decision_budget, reachable) for scene in scenes)
    solved = [s for s in scenes if s.get("success")]
    failed = [s for s in scenes if not s.get("success")]

    def steps(rows: list[dict]) -> dict:
        values = sorted(int(r.get("steps") or 0) for r in rows)
        if not values:
            return {}
        return {
            "min": values[0],
            "median": values[len(values) // 2],
            "max": values[-1],
        }

    return {
        "scenes": len(scenes),
        "outcomes": dict(outcomes),
        "budgets": task_budgets
        or {"note": "no budgets recoverable, so truncations are not split by budget"},
        # A failure that ends early is not a failure that ran out of time, and
        # the budget cap is the number that tells them apart.
        "steps_when_solved": steps(solved),
        "steps_when_failed": steps(failed),
        # The last action error before the episode ended. Only the terminal
        # one is recorded, so this is a hint about the final state and not a
        # census of everything that went wrong.
        "terminal_action_error": dict(Counter(s.get("action_error") for s in failed)),
        "by_layout_family": {
            family: {
                "scenes": total,
                "solved": sum(
                    1 for s in scenes if s.get("layout_family") == family and s.get("success")
                ),
            }
            for family, total in Counter(s.get("layout_family") for s in scenes).items()
        },
    }


def mcnemar_exact(only_a: int, only_b: int) -> dict:
    """Two-sided exact McNemar on the discordant pairs.

    Under the null the discordant scenes split 50/50, so the count favouring
    one arm is Binomial(n, 1/2). Exact rather than the chi-square
    approximation because that approximation is unreliable below roughly 25
    discordant pairs and these counts are routinely smaller.
    """
    total = only_a + only_b
    if total == 0:
        return {
            "discordant": 0,
            "p_value": 1.0,
            "note": "the arms agreed on every scene, so there is nothing to test",
        }
    smaller = min(only_a, only_b)
    tail = sum(math.comb(total, k) for k in range(smaller + 1)) / (2**total)
    return {"discordant": total, "p_value": round(min(1.0, 2 * tail), 6)}


def compare(scenes_a: list[dict], scenes_b: list[dict]) -> dict:
    """Paired comparison over the scenes both arms actually scored."""
    by_a = {
        s["episode_index"]: bool(s.get("success"))
        for s in scenes_a
        if s.get("episode_index") is not None
    }
    by_b = {
        s["episode_index"]: bool(s.get("success"))
        for s in scenes_b
        if s.get("episode_index") is not None
    }
    shared = sorted(set(by_a) & set(by_b))
    if not shared:
        return {
            "paired": False,
            "reason": (
                "the arms share no scored episode index, so nothing here compares "
                "them. Both arms must be scored on the same frozen holdout"
            ),
        }
    both = [i for i in shared if by_a[i] and by_b[i]]
    neither = [i for i in shared if not by_a[i] and not by_b[i]]
    only_a = [i for i in shared if by_a[i] and not by_b[i]]
    only_b = [i for i in shared if by_b[i] and not by_a[i]]
    return {
        "paired": True,
        "shared_scenes": len(shared),
        "only_in_a": len(only_a),
        "only_in_b": len(only_b),
        "both": len(both),
        "neither": len(neither),
        # The scene identities, not just the counts: a discordant scene is a
        # regression case someone can look at.
        "only_in_a_episodes": only_a[:20],
        "only_in_b_episodes": only_b[:20],
        "mcnemar": mcnemar_exact(len(only_a), len(only_b)),
        "note": (
            "equal rates with many discordant scenes is a different finding from "
            "equal rates on the same scenes, and only the pairing distinguishes them"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "runs",
        nargs="+",
        help="one to diagnose, or two to compare. Each is a run id under "
        "runtime/runs/ or a path to a result.json (published evidence works)",
    )
    parser.add_argument(
        "--row",
        default=None,
        choices=ROWS,
        help="only this evaluation row; default reports all three, because "
        "separating familiar execution from structural transfer is the point",
    )
    parser.add_argument("--out", default=None, help="write the report as JSON")
    args = parser.parse_args()

    if len(args.runs) > 2:
        raise SystemExit("one run to diagnose, or two to compare")
    rows = (args.row,) if args.row else ROWS
    results = [load(run_id) for run_id in args.runs]

    report: dict = {
        "runs": [
            {
                "source": source,
                "run_id": r.get("run_id"),
                "task": r.get("task"),
                "total_steps": r.get("total_steps"),
                "final_train_success_rate": r.get("final_train_success_rate"),
            }
            for source, r in zip(args.runs, results, strict=True)
        ],
        "rows": {},
    }

    for row in rows:
        block: dict = {}
        for run_id, result in zip(args.runs, results, strict=True):
            evaluation = (result.get("evaluation") or {}).get(row) or {}
            scenes = evaluation.get("per_scene") or []
            task_budgets = budgets(run_id, result.get("task"))
            block[run_id] = {
                "success_rate": evaluation.get("success_rate"),
                "stochastic_success_rate": (evaluation.get("stochastic") or {}).get("success_rate"),
                # Both arms, always. R5.1's gate: "Both sampled and argmax
                # evaluations are labeled; neither is substituted after seeing
                # a better score."
                "diagnosis": diagnose(scenes, task_budgets),
            }
        if len(args.runs) == 2:
            first, second = args.runs
            block["paired"] = compare(
                ((results[0].get("evaluation") or {}).get(row) or {}).get("per_scene") or [],
                ((results[1].get("evaluation") or {}).get(row) or {}).get("per_scene") or [],
            )
            block["paired"]["a"] = first
            block["paired"]["b"] = second
        report["rows"][row] = block

    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
