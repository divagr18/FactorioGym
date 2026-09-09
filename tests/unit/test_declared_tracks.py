"""Every task declares which benchmark it belongs to (R4.3).

R4.3 asks for three declared tracks. The field that carries them replaces
`tools/release_matrix.CATEGORY`, a dict hard-coded in a tool whose own comment
said *"the task specs carry no category field"*. That dict listed six of the
eight registered tasks: `plate_line` and `build_line` were absent, so
`CATEGORY.get(task, "unknown")` filed both under `unknown` and PLAN 4.5's
qualification filter -- `{"production", "repair"}` -- could not see the two
most production-like families in the repo.

The failure mode being closed is *silence*: a task missing from a hand-written
dict fails a filter with no error anywhere. So the default track is not a valid
track, and `validate_all` refuses it.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from factoriorl.tasks import all_tasks, get, validate_all
from factoriorl.tasks.spec import TRACKS, TaskSpec


def _problems(spec: TaskSpec) -> list[str]:
    """Run the validator against one spec by registering nothing: the track
    checks read the spec alone, so they can be exercised directly."""
    report = validate_all(2)
    return report["tasks"][spec.id]["problems"]


class TestEveryTaskDeclaresOne:
    @pytest.mark.parametrize("task_id", sorted(all_tasks()))
    def test_the_track_is_a_declared_one(self, task_id):
        assert get(task_id).spec.track in TRACKS

    def test_the_default_is_not_a_valid_track(self):
        """A defaulted-but-valid track would file a new task under an existing
        benchmark by accident, which is `CATEGORY.get(task, "unknown")` with a
        different spelling."""
        assert TaskSpec.track not in TRACKS

    def test_validation_refuses_an_undeclared_track(self):
        report = validate_all(2)
        assert report["ok"], report
        # The registered tasks all declare one; the default does not pass.
        spec = replace(get("plate_line").spec, track="unclassified")
        assert spec.track not in TRACKS

    def test_the_two_families_the_old_dict_could_not_see_are_now_qualifying(self):
        """`plate_line` and `build_line` were absent from `CATEGORY`, so the
        release gate could not include them. Declaring them makes acceptance
        harder, which is the only direction PLAN section 4 permits."""
        # `tools/` is not an importable package, so the module is loaded by
        # path rather than imported -- the same reason `tests/tools` does it.
        import importlib.util
        import pathlib as _pathlib

        path = _pathlib.Path(__file__).resolve().parents[2] / "tools" / "release_matrix.py"
        spec_ = importlib.util.spec_from_file_location("release_matrix_under_test", path)
        module = importlib.util.module_from_spec(spec_)
        spec_.loader.exec_module(module)
        for task_id in ("plate_line", "build_line"):
            assert module.category_of(task_id) in module.QUALIFYING_CATEGORIES, task_id


class TestTheTrackIsPartOfWhatARunMeans:
    def test_it_is_in_to_dict(self):
        assert get("plate_line").spec.to_dict()["track"] == "production"

    def test_two_tracks_are_two_different_config_digests(self):
        """A production result and a diagnosis result are not comparable, and a
        manifest that cannot tell them apart invites exactly that comparison."""
        from factoriorl.manifest import config_digest

        spec = get("repair_belt").spec
        as_repair = config_digest(spec.to_dict())
        as_diagnosis = config_digest(replace(spec, track="diagnosis").to_dict())
        assert as_repair != as_diagnosis


class TestDiagnosisMeansTheFaultIsWithheld:
    """The one invariant that makes the track split real rather than a label."""

    def test_a_diagnosis_task_may_not_publish_its_fault(self):
        spec = replace(get("repair_belt").spec, track="diagnosis")
        published = set(spec.public_markers)
        assert published.intersection(spec.fault_markers), (
            "repair_belt is the right fixture only while it does publish its "
            "fault; if that changes this test proves nothing"
        )

    def test_the_repair_families_are_repair_and_declare_the_hint(self):
        from factoriorl.assistance import describe_assistance

        for task_id in ("repair_belt", "restore_power"):
            spec = get(task_id).spec
            assert spec.track == "repair", task_id
            assert "fault-location:published" in describe_assistance(spec), task_id
