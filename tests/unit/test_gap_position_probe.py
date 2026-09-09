"""The probe must refuse the comparison it was written to make.

R5.2 wants a `restore_power` policy evaluated on interior-gap against
terminal-gap scenes, matched otherwise. This tool set out to name those scene
sets and found the comparison is not constructible: in `two_gaps` a terminal
gap only occurs alongside a shorter pole chain and a smaller fault separation,
because on a short chain both gaps are necessarily at its ends. So the
interesting behaviour under test is the *refusal* -- the tool must decline to
compare `both_terminal` against anything, name why, and still offer the one
contrast that does hold its confounds fixed.

The rest is provenance. A scene is a pure function of (master, run_id, branch,
index), so the probe's declared seed plan is what makes its subsets
reproducible, and sharing `run_id` with a scored holdout would draw probe
scenes from the same stream as reported results.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
HOLDOUT = ROOT / "docs" / "evidence" / "holdout_v3.json"


def _tool(name: str):
    """Load a tool by path; `tools/` is not an importable package."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def probe():
    return _tool("gap_position_probe")


@pytest.fixture(scope="module")
def report(probe):
    """One real build, shared: generating 600 scenes is the slow part."""
    return probe.build("restore_power", "two_gaps", samples=600, per_class=None)


# --- the refusal ------------------------------------------------------------


def test_the_terminal_contrast_is_refused_not_reported(report):
    """The whole point. `both_terminal` shares no stratum with either other
    class, so comparing it would vary chain length and fault separation
    alongside gap position -- and R5.2's question would come back
    unattributable rather than answered."""
    refused = {tuple(row["contrast"]) for row in report["unusable_contrasts"]}
    assert ("both_interior", "both_terminal") in refused
    assert ("both_terminal", "one_of_each") in refused
    offered = {frozenset(row["contrast"]) for row in report["usable_contrasts"]}
    assert not any("both_terminal" in pair for pair in offered)


def test_the_refusal_says_why(report):
    for row in report["unusable_contrasts"]:
        assert "chain length" in row["why"] and "unattributable" in row["why"]


def test_the_surviving_contrast_is_offered(report):
    """Refusing everything would be useless. `both_interior` against
    `one_of_each` at a fixed chain length and separation is weaker than R5.2's
    question -- it asks whether a policy needs *both* gaps interior -- but it
    is the strongest one this generator can pose."""
    offered = {frozenset(row["contrast"]) for row in report["usable_contrasts"]}
    assert frozenset(("both_interior", "one_of_each")) in offered
    assert report["usable_contrasts"], "at least one contrast must survive"


def test_every_offered_contrast_lives_inside_one_stratum(report):
    """A contrast that spanned two strata would be exactly the confounded
    comparison this tool exists to prevent."""
    for row in report["usable_contrasts"]:
        stratum = report["strata"][row["stratum"]]
        for name in row["contrast"]:
            assert name in stratum["classes_usable"]


def test_a_stratum_holds_its_confounds_fixed(report):
    """The stratum key is the guarantee. Within one, both classes must report
    identical chain length and fault separation -- otherwise the label is
    decorative."""
    for label, block in report["strata"].items():
        if not block["subsets"]:
            continue
        seen = set()
        for name, subset in block["subsets"].items():
            confounds = subset["confounds"]
            chain = confounds["chain_length"]
            separation = confounds["fault_separation"]
            assert chain["min"] == chain["max"], f"{label}/{name} chain length varies"
            assert separation["min"] == separation["max"], f"{label}/{name} separation varies"
            seen.add((chain["min"], separation["min"]))
        assert len(seen) == 1, f"{label} classes disagree on the confounds they fix"


def test_subsets_in_a_stratum_are_equal_sized(report):
    """Otherwise the comparison is also a comparison of sample size."""
    for label, block in report["strata"].items():
        sizes = {len(subset["episode_indices"]) for subset in block["subsets"].values()}
        assert len(sizes) <= 1, f"{label} offers unequal subsets"
        if block["subsets"]:
            assert sizes == {block["subset_size"]}


def test_a_class_too_thin_to_compare_is_reported_but_not_offered(probe):
    """A stratum with one scene in a class is not a contrast. Reporting the
    count while withholding the contrast is the distinction."""
    thin = probe.build("restore_power", "two_gaps", samples=600, per_class=None, minimum=10_000)
    assert thin["usable_contrasts"] == []
    assert any(block["classes_present"] for block in thin["strata"].values()), (
        "the counts must still be reported, or a reader cannot see why nothing was offered"
    )


# --- provenance -------------------------------------------------------------


def test_the_probe_does_not_share_the_holdouts_seed_stream(probe):
    """A scene is a pure function of (master, run_id, branch, index). Sharing
    `run_id` with a scored holdout would draw probe scenes from the same stream
    as reported results."""
    frozen = json.loads(HOLDOUT.read_text(encoding="utf-8"))["holdout"]["seed_plan"]
    assert probe.PROBE_RUN_ID != frozen["run_id"]


def test_the_declared_plan_is_the_one_that_generated_the_scenes(report, probe):
    assert report["seed_plan"]["run_id"] == probe.PROBE_RUN_ID
    assert report["seed_plan"]["master"] == probe.PROBE_MASTER


def test_the_same_arguments_reproduce_the_same_scenes(probe):
    """Without this the declared episode indices name nothing a later run can
    replay."""
    first = probe.build("restore_power", "two_gaps", samples=200, per_class=None, minimum=5)
    second = probe.build("restore_power", "two_gaps", samples=200, per_class=None, minimum=5)
    assert first == second


def test_scenes_carry_the_digest_a_later_run_must_match(report):
    for block in report["strata"].values():
        for subset in block["subsets"].values():
            assert len(subset["blueprint_digests"]) == len(subset["episode_indices"])
            assert all(subset["blueprint_digests"]), "a digest is what pins the scene"


def test_the_report_records_the_task_version_it_classified(report, probe):
    """A generator change moves every class boundary, so a probe without a
    version is a probe about nothing in particular."""
    from factoriorl import tasks

    assert report["task"]["version"] == tasks.get("restore_power").spec.version


def test_the_report_disclaims_any_finding_about_a_policy(report):
    """R5.2 requires that policies' actual actions be confirmed rather than
    inferred from generator distributions. This tool only defines scenes, and
    saying so in the artifact is what stops the next reader treating a class
    count as a result."""
    assert "claims_nothing_about_any_policy" in report


def test_the_holdout_cannot_supply_the_matched_set(report):
    """Recorded because it is the reason the probe declares its own scenes:
    holdout_v3 scores restore_power entirely on gap_near_drill, which is
    600/600 terminal, so it holds no interior scene to match against."""
    frozen = json.loads(HOLDOUT.read_text(encoding="utf-8"))["holdout"]
    families = {
        episode["layout_family"] for episode in frozen["tasks"]["restore_power"]["episodes"]
    }
    assert families == {"gap_near_drill"}, (
        "if the holdout ever gains an interior family, the probe should use it "
        "instead of declaring its own scenes"
    )
    assert "not_the_frozen_holdout" in report


# --- classification ---------------------------------------------------------


def test_a_scene_is_classified_from_its_own_chain(probe):
    """`flanking_sides` counts the fault's own chain, not any collinear entity.
    Both readings are recorded per scene so the class can be re-derived."""
    from factoriorl import tasks
    from factoriorl.seeding import SeedPlan

    task = tasks.get("restore_power")
    family = next(f for f in task.spec.layout_families if f.name == "two_gaps")
    plan = SeedPlan(master=probe.PROBE_MASTER, run_id=probe.PROBE_RUN_ID)
    scene = probe.describe_scene(task, family, plan, 0)
    assert scene["flanking_sides"] == sorted(scene["flanking_sides"])
    assert scene["class"] == probe.class_name(tuple(scene["flanking_sides"]))
    assert scene["split"] == family.split == "val"


def test_a_task_without_fault_markers_is_refused(probe):
    """`navigate` has no gap to position, so there is nothing here to measure
    and an empty report would read as "no confound found"."""
    with pytest.raises(SystemExit, match="fault markers"):
        probe.build("navigate", "open", samples=10, per_class=None)


def test_an_unknown_family_lists_the_real_ones(probe):
    with pytest.raises(SystemExit, match="two_gaps"):
        probe.build("restore_power", "no_such_family", samples=10, per_class=None)
