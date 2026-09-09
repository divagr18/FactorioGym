"""The R5.1 report must classify honestly and merge before it judges.

Three things here would be wrong in a way no exception reports.

**The band boundaries.** R5.1's whole point is that a 0.00 held-out score means
two different things depending on the familiar-layout row. A classifier that
called every structural miss a "transfer question" would claim the policy had
learned its training distribution, which is precisely the unsupported
algorithmic reading the gate forbids -- and is what the first version of this
tool did to a run sitting at 0.70.

**Merging before judging.** The three-seed `restore_power` result is split
across two evidence files: seed 1's 0.77 in one, seeds 2 and 3's 0.00s in the
other. Judged per file it is a lone 0.77 and a pair of 0.00s, both
unremarkable. The bimodality -- the finding -- exists only in the union, so the
merge is not a convenience.

**Both arms, together.** The gate requires greedy and sampled labeled with
neither substituted after seeing the better score, so they are returned as one
object rather than as two fields a caller can pick between.
"""

from __future__ import annotations

import importlib.util
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
    return _tool("learning_diagnosis")


@pytest.fixture(scope="module")
def report(diag):
    return diag.build()


def run(familiar: float | None, structural: float | None, greedy: float = 0.0) -> dict:
    return {
        "task": "t",
        "evaluation": {
            "seeds": {"success_rate": greedy, "stochastic": {"success_rate": familiar}},
            "val": {"success_rate": greedy, "stochastic": {"success_rate": 0.1}},
            "structures": {"success_rate": greedy, "stochastic": {"success_rate": structural}},
        },
    }


# --- which question a pair of rows poses ------------------------------------


def test_a_familiar_layout_miss_is_a_learning_question(diag):
    """`repair_belt`'s case. No amount of transfer analysis explains 0.12 on the
    policy's own training layouts."""
    assert "learning question" in diag.diagnose_run(run(0.12, 0.06))["which_question"]


def test_a_familiar_layout_pass_with_a_structural_miss_is_a_transfer_question(diag):
    """`restore_power` at 50k steps: 1.00 familiar, 0.66 structural."""
    assert "transfer question" in diag.diagnose_run(run(1.0, 0.66))["which_question"]


def test_clearing_neither_bar_is_not_called_a_transfer_result(diag):
    """The band the first version collapsed into "transfer". At 0.70 familiar the
    policy has not established that it executes its training distribution, so
    calling its structural miss a transfer result claims more than the numbers
    support."""
    question = diag.diagnose_run(run(0.7, 0.37))["which_question"]
    assert "both questions are open" in question
    assert "transfer question" not in question


def test_clearing_the_structural_bar_leaves_no_open_question(diag):
    assert "no open question" in diag.diagnose_run(run(1.0, 0.85))["which_question"]


def test_a_missing_arm_is_unclassifiable_rather_than_assumed(diag):
    """Guessing here would put a fabricated diagnosis in an evidence file."""
    assert "unclassifiable" in diag.diagnose_run(run(1.0, None))["which_question"]


# --- the spread -------------------------------------------------------------


def test_a_bimodal_spread_is_not_reported_as_variance(diag):
    """0.77/0.00/0.00 with every seed above 0.9 on familiar layouts: all three
    learned, and what differed is which policy they learned."""
    verdict = diag.classify_spread([0.0, 0.0, 0.77], [0.975, 1.0])
    assert verdict["mean_is_representative"] is False
    assert "bimodal" in verdict["spread"]
    assert "which policy it learned" in verdict["spread"]


def test_a_unimodal_spread_keeps_its_mean(diag):
    """`deliver`'s 0.54/0.85/0.92 -- a value in the middle, so the mean
    describes something a seed produced."""
    verdict = diag.classify_spread([0.54, 0.85, 0.92], [0.9833])
    assert verdict["mean_is_representative"] is True
    assert "unimodal" in verdict["spread"]


def test_a_wide_spread_with_a_middle_value_is_not_called_bimodal(diag):
    """The gap alone is not the signal. Without this, any wide spread would be
    labelled a lottery."""
    verdict = diag.classify_spread([0.0, 0.45, 0.9], [1.0])
    assert verdict["mean_is_representative"] is True


def test_one_seed_is_not_a_spread(diag):
    verdict = diag.classify_spread([0.0], [0.0])
    assert verdict["mean_is_representative"] is None
    assert "no spread to read" in verdict["spread"]


def test_the_learned_claim_needs_every_seed_to_have_learned(diag):
    """A bimodal spread where one seed never learned is a different finding, so
    the "which policy it learned" clause must not appear."""
    verdict = diag.classify_spread([0.0, 0.0, 0.77], [0.1, 1.0])
    assert verdict["mean_is_representative"] is False
    assert "which policy it learned" not in verdict["spread"]


# --- against the real evidence ----------------------------------------------


def test_the_restore_power_seeds_are_merged_across_files(report):
    """The finding exists only in the union: seed 1's 0.77 is in one file and
    seeds 2 and 3's 0.00s in another."""
    block = next(
        row
        for row in report["seed_spreads"]
        if row["task"] == "restore_power" and len(row["structural_per_seed"]) == 3
    )
    assert block["structural_per_seed"] == [0.0, 0.0, 0.77]
    assert len(block["sources"]) == 2, "a single source means the merge did not happen"
    assert block["mean_is_representative"] is False


def test_deliver_and_restore_power_get_opposite_spread_verdicts(report):
    """If both came back the same, the classifier is not discriminating."""
    verdicts = {
        row["task"]: row["mean_is_representative"]
        for row in report["seed_spreads"]
        if len(row["structural_per_seed"]) == 3
    }
    assert verdicts["deliver"] is True
    assert verdicts["restore_power"] is False


def test_the_three_published_runs_get_three_different_questions(report):
    """The point of the report. If they collapsed to one label it would be the
    "0.00 held-out" reading the gate calls useless."""
    questions = {run["source"]: run["which_question"] for run in report["runs"]}
    assert len(questions) == 3
    assert len(set(questions.values())) == 3


def test_no_row_reports_one_arm_without_the_other(report):
    for row in report["runs"]:
        for arm in row["rows"].values():
            assert "greedy" in arm and "sampled" in arm


def test_sampled_never_scored_below_greedy_in_the_record(report):
    """Recorded as a measured property rather than assumed: it is the evidence
    that the argmax penalty is systematic and not a single run's noise."""
    assert report["both_arms"]["sampled_never_scored_below_greedy"] is True


def test_the_largest_argmax_gap_is_surfaced(report):
    """0.03 greedy against 0.70 sampled on one `restore_power` run. A report
    that buried this would be understating a property of the evaluation."""
    largest = report["both_arms"]["largest_gaps"][0]
    assert largest["ratio"] >= 20


def test_the_report_says_it_cannot_localise_and_why(report):
    """Every cited run predates `623f7dd`, so no row carries scene-level
    outcomes. Silence here would read as "nothing to localise"."""
    block = report["cannot_localise_within_an_episode"]
    assert block["rows_with_per_scene_data"] == []
    assert "623f7dd" in block["why"]


def test_the_provenance_limits_are_carried_with_the_numbers(report):
    """holdout_v2 and the 70-distinct-scene ceiling bound every rate above, so
    they belong in the artifact rather than in a commit message."""
    text = " ".join(report["provenance_limits"])
    assert "holdout_v2" in text and "70 distinct scenes" in text


def test_every_declared_source_was_found(report):
    """A silently missing evidence file would shrink the report without
    changing any claim in it."""
    assert report["missing_sources"] == []


def test_the_report_disclaims_an_algorithmic_reading(report):
    assert "forbids naming one from a rate" in report["attaches_no_algorithmic_claim"]
