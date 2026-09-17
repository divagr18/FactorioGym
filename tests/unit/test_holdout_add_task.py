"""Adding a task to a frozen holdout without destroying its evidence.

`--refreeze` rebuilds the document from scratch, which drops
`declaration.declared_at` -- and that timestamp is the entire evidence that the
candidate set was chosen before any held-out result was read. Adding an eighth
task must not destroy the evidence about the other seven.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import freeze_holdout as fh  # noqa: E402

COMMITTED = ROOT / "docs" / "evidence" / "holdout_v3.json"


@pytest.fixture(scope="module")
def document() -> dict:
    return json.loads(COMMITTED.read_text(encoding="utf-8"))


class TestBuildLineIsFrozenButNotDeclared:
    def test_it_is_present(self, document):
        assert "build_line" in document["holdout"]["tasks"]
        assert len(document["holdout"]["tasks"]["build_line"]["episodes"]) == 100

    def test_it_is_not_a_declared_release_candidate(self, document):
        """Declaring a candidate after freezing it is what PLAN 4.5 forbids."""
        assert "build_line" not in (document["declaration"]["candidate_families"] or [])

    def test_the_declaration_timestamp_survived(self, document):
        declaration = document["declaration"]
        assert declaration["candidate_families"] == [
            "deliver",
            "repair_belt",
            "restore_power",
        ]
        assert declaration["declared_at"]
        assert declaration["declared_at_commit"]

    def test_it_was_frozen_under_the_files_own_seed_plan(self, document):
        """A task added under a different plan would sit in the same file, under
        the same holdout id, naming scenes from a different seed stream."""
        plan = fh.SeedPlan(
            master=document["holdout"]["seed_plan"]["master"],
            run_id=document["holdout"]["seed_plan"]["run_id"],
        )
        expected = fh.freeze_task(
            "build_line",
            plan,
            document["holdout"]["start_index"],
            document["holdout"]["episodes_per_task"],
        )
        assert document["holdout"]["tasks"]["build_line"] == expected


class TestTheSevenExistingEntriesAreUntouched:
    #: Read off the committed file before `build_line` was added.
    ENTRY_HASHES = {
        "deliver": "63dd42500ba435bd",
        "mine_smelt": "a11e26d64afdd678",
        "navigate": "22088ee7c7b098ac",
        # plate_line reached v1.2.0 when the character-on-a-wall defect was
        # fixed, so its entry was legitimately re-frozen; the other six are the
        # pre-`build_line` values and must never move.
        "plate_line": "18bdee7ecedcdc2a",
        "repair_belt": "32f8a13a2b4772b1",
        "restore_power": "dcab54a130c4e091",
        "supply_furnace": "24eb644a3f90d3a6",
    }

    @pytest.mark.parametrize(("task", "digest"), sorted(ENTRY_HASHES.items()))
    def test_the_per_task_digest_did_not_move(self, document, task, digest):
        """`stale_citations` judges a run on this, so a published result still
        describes the scenes it was measured on."""
        assert document["summary"][task]["entry_hash"] == digest

    def test_the_recomputed_digest_agrees_with_the_summary(self, document):
        for task, entry in document["holdout"]["tasks"].items():
            assert fh.entry_hash(entry) == document["summary"][task]["entry_hash"]


class TestVerifyStillGuardsTheGap:
    def test_a_registered_task_missing_from_the_file_is_still_an_error(self, document):
        """`pending` exists only for `--add-task`; without that guarantee,
        "registered but never frozen" would stop being reported."""
        trimmed = json.loads(json.dumps(document))
        del trimmed["holdout"]["tasks"]["build_line"]
        trimmed["content_hash"] = fh.content_hash(trimmed["holdout"])
        problems = fh.verify(trimmed)
        assert any("build_line" in p and "never frozen" in p for p in problems), problems

    def test_naming_it_pending_is_the_only_way_past_that(self, document):
        trimmed = json.loads(json.dumps(document))
        del trimmed["holdout"]["tasks"]["build_line"]
        trimmed["content_hash"] = fh.content_hash(trimmed["holdout"])
        assert fh.verify(trimmed, pending={"build_line"}) == []

    def test_a_hand_edited_file_is_still_caught(self, document):
        tampered = json.loads(json.dumps(document))
        tampered["holdout"]["tasks"]["build_line"]["episodes"][0]["blueprint_digest"] = "0" * 16
        problems = fh.verify(tampered)
        assert problems
