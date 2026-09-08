"""Synthesis section 8's leading clause: symmetry, tested rather than assumed.

Section 8 proposes scoring placement candidates by their resulting local
structure, and its gate says to *"treat symmetry as a tested property: not
every Factorio mechanic is rotation/reflection invariant"* and to *"defer a new
value-learning algorithm until the simpler representation test is
informative"*.

`tools/symmetry_probe.py` ran that test on a real engine. These tests pin its
published conclusion, because the conclusion is the deliverable: a candidate
scorer may share parameters across **rotations** of a local structure and must
**not** share them across reflections. A scorer assuming the full dihedral
group would be wrong on half its orbit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import symmetry_probe as probe  # noqa: E402

EVIDENCE = ROOT / "docs" / "evidence" / "section8-symmetry.json"


@pytest.fixture(scope="module")
def report() -> dict:
    return json.loads(EVIDENCE.read_text(encoding="utf-8"))


class TestTheTransforms:
    """Pure arithmetic, so the conclusion cannot rest on a bad rotation."""

    def test_four_quarter_turns_is_the_identity(self):
        assert probe.rotate((3, -7), 4) == (3, -7)

    def test_a_quarter_turn_is_clockwise(self):
        assert probe.rotate((1, 0), 1) == (0, 1)
        assert probe.rotate((0, 1), 1) == (-1, 0)

    def test_two_quarter_turns_negates_both_axes(self):
        assert probe.rotate((2, -3), 2) == (-2, 3)

    def test_reflection_flips_one_axis_only(self):
        assert probe.reflect((2, -3), "x") == (-2, -3)
        assert probe.reflect((2, -3), "y") == (2, 3)


class TestTheMeasuredResult:
    def test_every_facing_admits_exactly_two_productive_placements(self, report):
        counts = {
            facing: len(body["productive"]) for facing, body in report["facings"].items()
        }
        assert counts == {"north": 2, "east": 2, "south": 2, "west": 2}, counts

    def test_rotation_is_invariant(self, report):
        """All three rotations of south's productive set match exactly."""
        base = {tuple(o) for o in report["facings"]["south"]["productive"]}
        for quarters, facing in ((1, "west"), (2, "north"), (3, "east")):
            actual = {tuple(o) for o in report["facings"][facing]["productive"]}
            assert {probe.rotate(o, quarters) for o in base} == actual, facing

    def test_reflection_is_not_invariant(self, report):
        """The finding. A 2x2 entity at centre (cx, cy) occupies tiles
        {cx-1, cx} x {cy-1, cy}: it extends one tile in the negative direction
        and none in the positive. That bias survives a 90-degree rotation and
        does not survive a flip."""
        base = {tuple(o) for o in report["facings"]["south"]["productive"]}
        north = {tuple(o) for o in report["facings"]["north"]["productive"]}
        reflected = {probe.reflect(o, "y") for o in base}
        assert reflected != north
        # Off by exactly one tile in x, which is the footprint's bias.
        assert reflected == {(0, -2), (1, -2)}
        assert north == {(-1, -2), (0, -2)}

    def test_the_report_says_the_mechanic_is_not_symmetric(self, report):
        assert report["symmetric"] is False
        failed = [c["check"] for c in report["checks"] if not c["holds"]]
        assert failed == ["reflecting south's productive set in y gives north's"], failed

    def test_the_gate_held_the_candidate_set_information_and_budget_fixed(self, report):
        """The gate's own wording, so the comparison is a comparison."""
        fixed = report["held_fixed"]
        assert "candidate_set" in fixed
        assert "information" in fixed and "no evaluator truth" in fixed["information"]
        assert fixed["budget_ticks"] == probe.MEASURE_TICKS

    def test_the_productive_set_matches_the_geometry_build_line_relies_on(self, report):
        """`build_line` places the furnace at `(drill.x, drill.y + 2)` for a
        south-facing drill; the sweep must agree, or one of the two is wrong."""
        south = {tuple(o) for o in report["facings"]["south"]["productive"]}
        assert (0, 2) in south
        assert (1, 2) in south
        assert (0, 3) not in south


class TestTranslation:
    def test_translation_is_invariant(self, report):
        """The other half of "translated/rotated". Odd and even drill offsets
        both, because a 2x2 footprint has a parity."""
        if not report.get("translations"):
            pytest.skip("this evidence file predates the translation arm")
        base = {tuple(o) for o in report["facings"]["south"]["productive"]}
        for origin, body in report["translations"].items():
            assert {tuple(o) for o in body["productive"]} == base, origin

    def test_both_parities_were_tested(self, report):
        if not report.get("translations"):
            pytest.skip("this evidence file predates the translation arm")
        origins = [
            tuple(int(v) for v in key.split(",")) for key in report["translations"]
        ]
        parities = {(x % 2, y % 2) for x, y in origins}
        assert len(parities) >= 2, f"only one parity tested: {origins}"


def test_the_conclusion_is_recorded_where_a_scorer_author_will_read_it():
    """A finding nobody meets is a finding nobody uses."""
    status = (ROOT / "docs" / "CURRENT_STATUS.md").read_text(encoding="utf-8")
    assert "section8-symmetry.json" in status
    assert "reflection" in status.lower()
