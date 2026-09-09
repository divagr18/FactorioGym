"""The diagnosis tool must localise, and must not invent a localisation.

Three claims are worth a test here, and they are the three that would be
silently wrong rather than loudly broken.

**Pairing.** Two arms scored on one frozen holdout can post identical rates
while disagreeing about which scenes they solve; `PLAN.md` section 3 requires
the paired reading for exactly this reason. A comparison that pooled the two
arms would report "no difference" for both a perfect agreement and a 50/50
shuffle, so the discordant counts are the thing under test.

**Which budget expired, and when to refuse to say.** `env.py` truncates on
either the decision budget or the game-tick budget and records only a boolean,
so the split is an inference. Two things can make that inference wrong with no
error raised, and both are tested here. It compares against a budget, so it is
only as good as the budget's provenance -- reading today's registry for a run
made against an older spec would mislabel every truncation, which is why the
run's own manifest is preferred and why the fallback admits it may be stale.
And the elimination step only holds if both budgets can actually expire: on
`deliver` the tick budget needs 500 decisions and the decision budget stops at
120, so "not the decision budget" does not leave ticks as the answer -- it
leaves no answer, and the tool has to say `truncated_unexplained` instead of
filling the gap.

**Refusing to compare.** Two runs scored on *different* scene sets have no
paired reading at all, and the tool has to say so rather than intersect
whatever indices happen to coincide.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _tool(name: str):
    """Load a tool by path; `tools/` is not an importable package."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def diag():
    return _tool("diagnose_run")


def scene(index, success: bool, steps: int = 10, **extra) -> dict:
    row = {
        "episode_index": index,
        "success": success,
        "steps": steps,
        "terminated": not success,
        "truncated": False,
        "layout_family": "f",
    }
    row.update(extra)
    return row


# --- how an episode ended ---------------------------------------------------


def test_a_short_truncation_is_game_time_when_ticks_can_bind(diag):
    """Truncation requires one of the two budgets, so a step count short of the
    decision budget implicates ticks -- but only where ticks can actually
    expire first."""
    row = scene(1, False, steps=17, truncated=True, terminated=False)
    assert diag.outcome(row, 120, True) == "out_of_game_time"


def test_a_short_truncation_is_unexplained_when_ticks_cannot_bind(diag):
    """The anomaly that motivated all of this, and the reason it is not simply
    relabelled. On `deliver` a decision advances exactly 30 ticks, so the
    15000-tick budget needs 500 decisions and the decision budget stops at
    120: ticks can never expire first. 103 of 113 failures across the two
    finished 4.4 cells truncated below 120 steps anyway. Naming ticks as the
    cause by elimination would have manufactured a diagnosis out of a
    contradiction."""
    row = scene(1, False, steps=17, truncated=True, terminated=False)
    assert diag.outcome(row, 120, False) == "truncated_unexplained"


def test_a_truncation_at_the_budget_is_decisions(diag):
    row = scene(1, False, steps=120, truncated=True, terminated=False)
    assert diag.outcome(row, 120) == "out_of_decisions"


def test_without_a_budget_the_split_is_withheld_not_guessed(diag):
    """No budget recovered means no inference. Naming one anyway would be a
    fabricated localisation, which is worse than an absent one."""
    row = scene(1, False, steps=17, truncated=True, terminated=False)
    assert diag.outcome(row, None) == "out_of_budget"


def test_an_unreachable_tick_budget_is_detected_and_explained(diag, tmp_path, monkeypatch):
    """`deliver`'s own numbers: 15000 ticks at 30 per decision needs 500
    decisions, and the budget stops at 120."""
    run = tmp_path / "r1"
    run.mkdir()
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "task": {
                    "id": "deliver",
                    "version": "1.3.0",
                    "resolved": {
                        "decision_ticks": 30,
                        "budgets": {"max_decision_steps": 120, "max_game_ticks": 15000},
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(diag.manifest_module, "runs_dir", lambda: tmp_path)
    found = diag.budgets("r1", "deliver")
    assert found["decisions_to_exhaust_ticks"] == 500
    assert found["tick_budget_reachable"] is False
    assert "never bind" in found["note"]


def test_a_reachable_tick_budget_carries_no_warning(diag, tmp_path, monkeypatch):
    """The arithmetic must not cry wolf: a task whose tick budget really can
    expire first gets the split, not the refusal."""
    run = tmp_path / "r2"
    run.mkdir()
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "task": {
                    "id": "t",
                    "resolved": {
                        "decision_ticks": 30,
                        "budgets": {"max_decision_steps": 500, "max_game_ticks": 6000},
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(diag.manifest_module, "runs_dir", lambda: tmp_path)
    found = diag.budgets("r2", "t")
    assert found["decisions_to_exhaust_ticks"] == 200
    assert found["tick_budget_reachable"] is True
    assert "note" not in found


def test_a_failure_predicate_is_not_a_truncation(diag):
    assert diag.outcome(scene(1, False, steps=5), 120, True) == "failure_predicate"


def test_success_wins_over_every_other_flag(diag):
    """A solved episode that also carries `truncated` is solved: the success
    predicate fired. Checking truncation first would have hidden it."""
    row = scene(1, True, steps=120, truncated=True)
    assert diag.outcome(row, 120, True) == "solved"


def test_the_budget_comes_from_the_runs_own_manifest(diag, tmp_path, monkeypatch):
    """`result.json` records neither the budgets nor the task version, so the
    registry would answer with today's spec for a run made against an older
    one -- and mislabel every truncation without erroring."""
    run = tmp_path / "r1"
    run.mkdir()
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "task": {
                    "id": "deliver",
                    "version": "0.9.0",
                    "resolved": {"budgets": {"max_decision_steps": 42, "max_game_ticks": 999}},
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(diag.manifest_module, "runs_dir", lambda: tmp_path)
    found = diag.budgets("r1", "deliver")
    assert found["max_decision_steps"] == 42, "the run's own budget, not the registry's"
    assert found["task_version"] == "0.9.0"
    assert "manifest" in found["source"]


def test_the_registry_fallback_says_it_may_be_wrong(diag, tmp_path, monkeypatch):
    monkeypatch.setattr(diag.manifest_module, "runs_dir", lambda: tmp_path)
    found = diag.budgets("absent", "deliver")
    assert found is not None, "deliver is registered, so a fallback exists"
    assert "registry" in found["source"] and "wrong" in found["source"], (
        "a fallback that does not admit it may be stale invites the exact "
        "mislabelling the manifest lookup exists to avoid"
    )


def test_an_unregistered_task_costs_the_split_and_nothing_else(diag, tmp_path, monkeypatch):
    monkeypatch.setattr(diag.manifest_module, "runs_dir", lambda: tmp_path)
    assert diag.budgets("absent", "no-such-task-anywhere") is None


# --- pairing ----------------------------------------------------------------


def test_identical_rates_can_still_disagree_on_every_scene(diag):
    """The finding this tool exists for. Both arms solve 2 of 4 -- identical
    rates, identical Wilson intervals -- and share not one solved scene. A
    pooled comparison calls these two arms the same."""
    a = [scene(1, True), scene(2, True), scene(3, False), scene(4, False)]
    b = [scene(1, False), scene(2, False), scene(3, True), scene(4, True)]
    result = diag.compare(a, b)
    assert result["both"] == 0 and result["neither"] == 0
    assert result["only_in_a"] == 2 and result["only_in_b"] == 2
    assert result["mcnemar"]["discordant"] == 4


def test_perfect_agreement_is_reported_as_nothing_to_test(diag):
    a = [scene(1, True), scene(2, False)]
    result = diag.compare(a, list(a))
    assert result["mcnemar"]["discordant"] == 0
    assert result["mcnemar"]["p_value"] == 1.0


def test_only_the_shared_scenes_are_compared(diag):
    """An arm evaluated on more scenes than the other must not have its extra
    scenes counted against it."""
    a = [scene(1, True), scene(2, True), scene(99, True)]
    b = [scene(1, True), scene(2, False)]
    result = diag.compare(a, b)
    assert result["shared_scenes"] == 2, "99 is not in both, so it is not evidence"
    assert result["only_in_a"] == 1 and result["only_in_b"] == 0


def test_disjoint_scene_sets_refuse_to_compare(diag):
    """Not an empty comparison -- a refusal. Two arms on different holdouts
    produce a paired number that means nothing."""
    result = diag.compare([scene(1, True)], [scene(2, True)])
    assert result["paired"] is False
    assert "same frozen holdout" in result["reason"]


def test_a_one_sided_sweep_is_significant_and_a_split_is_not(diag):
    """Exact McNemar, not the chi-square approximation, because these counts
    are routinely under the ~25 discordant pairs the approximation needs."""
    one_sided = diag.mcnemar_exact(10, 0)
    even = diag.mcnemar_exact(9, 9)
    assert one_sided["p_value"] < 0.01, "10 of 10 disagreements favouring one arm"
    assert even["p_value"] == 1.0, "the observed 9/9 split is exactly the null"
    assert diag.mcnemar_exact(1, 0)["p_value"] == 1.0, "a single pair proves nothing"


def test_the_discordant_scenes_are_named_not_only_counted(diag):
    """A count says a disagreement exists; an episode index is something a
    person can go and look at, which is what turns this into a probe."""
    a = [scene(7, True), scene(8, False)]
    b = [scene(7, False), scene(8, False)]
    result = diag.compare(a, b)
    assert result["only_in_a_episodes"] == [7]


def test_scenes_without_an_identity_are_dropped_rather_than_paired(diag):
    """`episode_index` is None on the unfrozen path. Two such rows are not the
    same scene, and pairing them would fabricate agreement."""
    a = [scene(None, True), scene(1, True)]
    b = [scene(None, False), scene(1, True)]
    result = diag.compare(a, b)
    assert result["shared_scenes"] == 1
    assert result["both"] == 1 and result["only_in_b"] == 0


# --- the aggregate ----------------------------------------------------------


def test_diagnose_separates_the_two_step_distributions(diag):
    """Solved-in-4 against failed-at-the-cap is the shape that says "recognises
    the situation or does not", and it is invisible in a mean over all
    episodes."""
    scenes = [scene(1, True, steps=4), scene(2, True, steps=4)] + [
        scene(i, False, steps=120, truncated=True, terminated=False) for i in range(3, 6)
    ]
    out = diag.diagnose(
        scenes,
        {"max_decision_steps": 120, "max_game_ticks": 15000, "tick_budget_reachable": True},
    )
    assert out["steps_when_solved"] == {"min": 4, "median": 4, "max": 4}
    assert out["steps_when_failed"]["min"] == 120
    assert out["outcomes"] == {"solved": 2, "out_of_decisions": 3}


def test_a_run_without_per_scene_says_so_instead_of_reporting_zeroes(diag):
    """Both curriculum runs predate `623f7dd` and carry no rows. Reporting
    "0 failures" for them would be a false clean bill."""
    out = diag.diagnose([], None)
    assert out["scenes"] == 0
    assert "predates" in out["note"]
