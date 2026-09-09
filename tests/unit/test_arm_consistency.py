"""Two arms can post the same rate and solve different scenes.

The statistical content here is the conditioning, and getting it wrong is not
an error anything raises -- it just reads the evidence backwards.

The permutation shuffles arm labels **within each scene**, which preserves that
scene's total solves. That is the exact conditional null for "shaping changes
nothing about which scenes are solvable", and it means a scene every run solved
cannot split under any shuffle. So the baseline for a perfectly-split scene has
to be conditional too: on the real holdout most scenes are solved by every run,
and the unconditional figure gave 33 expected splits against 2 observed --
which reads as the arms agreeing far more than chance, when in fact only 2
scenes were ever at risk and both of them split.

Also under test: the tool must not manufacture disagreement out of missing
data. A scene one arm never scored is not a failure by that arm, and a run
predating `623f7dd` carries no per-scene rows at all.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _tool(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def arm():
    return _tool("arm_consistency")


def write_run(directory: pathlib.Path, name: str, solved: dict[int, bool], row="structures"):
    path = directory / f"{name}.json"
    path.write_text(
        json.dumps(
            {
                "run_id": name,
                "task": "deliver",
                "evaluation": {
                    row: {
                        "per_scene": [
                            {"episode_index": index, "success": ok, "steps": 4}
                            for index, ok in sorted(solved.items())
                        ]
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return str(path)


# --- the conditioning -------------------------------------------------------


def test_a_scene_every_run_solved_cannot_split_and_is_not_counted_as_at_risk(arm, tmp_path):
    """The correction. Scene 0 is solved by all four runs, so no within-scene
    shuffle can split it; counting it toward the expected number of perfect
    splits inflates the null by scenes that were never at risk."""
    a1 = write_run(tmp_path, "a1", {0: True, 1: True})
    a2 = write_run(tmp_path, "a2", {0: True, 1: True})
    b1 = write_run(tmp_path, "b1", {0: True, 1: False})
    b2 = write_run(tmp_path, "b2", {0: True, 1: False})
    report = arm.build({"a": [a1, a2], "b": [b1, b2]}, "structures")
    splits = report["perfect_splits"]
    assert splits["splittable_scenes"] == 1, "only scene 1 has a total of 2 solves"
    assert splits["observed"] == 1
    assert splits["expected_under_null"] == round(1 * 2 / 6, 2)


def test_the_unconditional_figure_is_shown_so_the_difference_is_visible(arm, tmp_path):
    runs = {
        "a": [write_run(tmp_path, f"a{i}", {j: True for j in range(10)}) for i in range(2)],
        "b": [write_run(tmp_path, f"b{i}", {j: True for j in range(10)}) for i in range(2)],
    }
    report = arm.build(runs, "structures")
    assert report["perfect_splits"]["splittable_scenes"] == 0
    assert report["perfect_splits"]["expected_under_null"] == 0.0
    assert "overstates the null" in report["perfect_splits"]["why_conditional"]


def test_a_perfectly_split_scene_is_attributed_to_the_arm_that_solved_it(arm, tmp_path):
    a1 = write_run(tmp_path, "a1", {0: False})
    a2 = write_run(tmp_path, "a2", {0: False})
    b1 = write_run(tmp_path, "b1", {0: True})
    b2 = write_run(tmp_path, "b2", {0: True})
    report = arm.build({"a": [a1, a2], "b": [b1, b2]}, "structures")
    assert report["always_only_in"]["b"] == [0]
    assert report["always_only_in"]["a"] == []


# --- the permutation test ---------------------------------------------------


def test_identical_arms_give_no_evidence_of_a_difference(arm, tmp_path):
    solved = {index: index % 3 != 0 for index in range(30)}
    runs = {
        "a": [write_run(tmp_path, f"a{i}", solved) for i in range(2)],
        "b": [write_run(tmp_path, f"b{i}", solved) for i in range(2)],
    }
    report = arm.build(runs, "structures")
    assert report["permutation_test"]["observed"] == 0.0
    assert report["permutation_test"]["p_value"] == 1.0, (
        "zero disagreement is the least extreme outcome possible, so every shuffle ties it"
    )


def test_total_disagreement_on_every_scene_is_detected(arm, tmp_path):
    """One arm solves everything, the other nothing. If this did not come back
    significant the test would be measuring nothing."""
    everything = {index: True for index in range(30)}
    nothing = {index: False for index in range(30)}
    runs = {
        "a": [write_run(tmp_path, f"a{i}", everything) for i in range(2)],
        "b": [write_run(tmp_path, f"b{i}", nothing) for i in range(2)],
    }
    report = arm.build(runs, "structures")
    assert report["permutation_test"]["observed"] == 2.0
    assert report["permutation_test"]["p_value"] < 0.001


def test_the_statistic_ignores_direction(arm, tmp_path):
    """A rate comparison already answers which arm is better. This answers
    whether the arms solve different scenes, so swapping the arms must not
    change the statistic."""
    first = write_run(tmp_path, "f", {0: True, 1: False, 2: True})
    second = write_run(tmp_path, "s", {0: False, 1: True, 2: False})
    one = arm.build({"a": [first], "b": [second]}, "structures")
    other = arm.build({"a": [second], "b": [first]}, "structures")
    assert one["permutation_test"]["observed"] == other["permutation_test"]["observed"]


def test_a_p_value_of_exactly_zero_is_never_reported(arm, tmp_path):
    """The observed arrangement is itself one draw from the null, so 0 is not
    attainable -- reporting it would claim more than a finite number of
    shuffles can support."""
    everything = {index: True for index in range(20)}
    nothing = {index: False for index in range(20)}
    runs = {
        "a": [write_run(tmp_path, "a1", everything)],
        "b": [write_run(tmp_path, "b1", nothing)],
    }
    report = arm.build(runs, "structures")
    assert report["permutation_test"]["p_value"] > 0


# --- refusing to invent data ------------------------------------------------


def test_a_scene_only_one_arm_scored_is_dropped_not_counted_as_a_failure(arm, tmp_path):
    """Treating an unscored scene as a miss would manufacture disagreement out
    of an absence."""
    a1 = write_run(tmp_path, "a1", {0: True, 1: True, 99: True})
    b1 = write_run(tmp_path, "b1", {0: True, 1: True})
    report = arm.build({"a": [a1], "b": [b1]}, "structures")
    assert report["shared_scenes"] == 2
    assert report["permutation_test"]["observed"] == 0.0


def test_a_run_without_per_scene_rows_is_refused(arm, tmp_path):
    """Both curriculum runs predate 623f7dd. Returning an empty comparison for
    them would read as "the arms agree everywhere"."""
    path = tmp_path / "old.json"
    path.write_text(
        json.dumps({"run_id": "old", "evaluation": {"structures": {"success_rate": 0.5}}}),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="623f7dd"):
        arm.build({"a": [str(path)], "b": [str(path)]}, "structures")


def test_more_than_two_arms_is_refused(arm, tmp_path):
    run = write_run(tmp_path, "r", {0: True})
    with pytest.raises(SystemExit, match="exactly two arms"):
        arm.build({"a": [run], "b": [run], "c": [run]}, "structures")


# --- the plumbing -----------------------------------------------------------


def test_arms_are_recovered_from_a_comparison_report_by_run_id_prefix(arm, tmp_path):
    report = tmp_path / "cmp.json"
    report.write_text(
        json.dumps(
            {
                "runs": [
                    {"run_id": "shaped-20260909T1-a"},
                    {"run_id": "sparse-20260909T2-b"},
                    {"run_id": "shaped-20260909T3-c"},
                ]
            }
        ),
        encoding="utf-8",
    )
    arms = arm.arms_from_report(str(report))
    assert sorted(arms) == ["shaped", "sparse"]
    assert len(arms["shaped"]) == 2 and len(arms["sparse"]) == 1


def test_a_report_with_one_arm_prefix_is_refused(arm, tmp_path):
    report = tmp_path / "cmp.json"
    report.write_text(json.dumps({"runs": [{"run_id": "shaped-1-a"}]}), encoding="utf-8")
    with pytest.raises(SystemExit, match="expected two arm prefixes"):
        arm.arms_from_report(str(report))


def test_an_arm_argument_needs_a_name(arm):
    with pytest.raises(SystemExit, match="name=run1"):
        arm.parse_arm("just-a-run-id")
