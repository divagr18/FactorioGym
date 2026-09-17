"""The parity recorder's normalisation, and the committed traces it produced.

A trace hash is only a determinism claim if what it hashes is the world. The
recorder removes three things that differ between two recordings of one world
-- the episode id, absolute engine ticks, session request ids -- and these
tests pin that it removes exactly those. The last test re-derives every
committed trace's hash from its file, so the index cannot drift from the
evidence it describes.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs" / "evidence" / "sim-parity"


def _tool():
    """Load the tool by path; `tools/` is not an importable package."""
    spec = importlib.util.spec_from_file_location(
        "record_parity_trace", ROOT / "tools" / "record_parity_trace.py"
    )
    module = importlib.util.module_from_spec(spec)
    # Registered first: `dataclass` resolves annotations through
    # `sys.modules[cls.__module__]`.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tool():
    return _tool()


def _normaliser(tool, absolute_tick: int, tick: int = 30):
    normaliser = tool.Normaliser()
    normaliser.begin({"absolute_tick": absolute_tick, "tick": tick})
    return normaliser


def _frame(absolute_start: int, session: str) -> dict:
    return {
        "episode_id": "ep-7",
        "tick": 60,
        "absolute_tick": absolute_start + 60,
        "events": [
            {
                "request_id": f"step-{session}-4:act",
                "tick": 60,
                "action": {"action": "mine", "cancelled_at_tick": absolute_start + 45},
            }
        ],
        "inflight": [{"request_id": f"step-{session}-9:act", "started_tick": absolute_start + 30}],
    }


class TestNormalisation:
    def test_two_sessions_at_different_engine_ticks_normalise_equal(self, tool):
        a = _normaliser(tool, absolute_tick=1_000 + 30)
        b = _normaliser(tool, absolute_tick=90_000 + 30)
        assert a.observation(_frame(1_000, "07d5db43")) == b.observation(_frame(90_000, "197d6cb5"))

    def test_relative_ticks_are_left_alone(self, tool):
        normaliser = _normaliser(tool, absolute_tick=5_030)
        body = normaliser.observation(_frame(5_000, "07d5db43"))
        assert body["tick"] == 60
        assert body["events"][0]["tick"] == 60
        assert body["events"][0]["action"]["cancelled_at_tick"] == 45
        assert body["inflight"][0]["started_tick"] == 30
        assert "episode_id" not in body and "absolute_tick" not in body

    def test_request_ids_are_renamed_by_first_appearance_everywhere(self, tool):
        normaliser = _normaliser(tool, absolute_tick=30)
        body = normaliser.observation(_frame(0, "07d5db43"))
        assert body["events"][0]["request_id"] == "r1"
        assert body["inflight"][0]["request_id"] == "r2"
        hidden = normaliser.hidden(
            {
                "tick": 60,
                "inflight": {
                    "entries": [{"request_id": "step-07d5db43-9:act", "started_tick": 30}]
                },
            }
        )
        assert hidden["inflight"]["entries"][0]["request_id"] == "r2"

    def test_a_string_that_only_resembles_an_id_is_kept(self, tool):
        normaliser = _normaliser(tool, absolute_tick=30)
        assert normaliser.observation({"name": "burner-mining-drill"})["name"] == (
            "burner-mining-drill"
        )

    def test_empty_engine_tables_become_the_lists_they_are(self, tool):
        normaliser = _normaliser(tool, absolute_tick=130, tick=30)
        hidden = normaliser.hidden(
            {
                "tick": 190,
                "entities": [{"name": "stone-furnace", "inventories": {}}],
                "ground_items": {},
                "resources": {},
                "character": {"main": {"size": 80, "stacks": {}}},
                "handles": {"next_id": 2, "order": [{"handle": "h1", "first_seen": 100}]},
                "inflight": {"next_seq": 0, "entries": {}},
            }
        )
        assert hidden["tick"] == 90
        assert hidden["ground_items"] == [] and hidden["resources"] == []
        assert hidden["character"]["main"]["stacks"] == []
        assert hidden["entities"][0]["inventories"] == {}
        assert hidden["handles"]["order"][0]["first_seen"] == 0
        assert hidden["inflight"]["entries"] == []


class TestHashing:
    def test_the_hash_moves_with_any_field(self, tool):
        header = {"scenario": "x"}
        records = [{"decision": 0, "hidden": {"tick": 0}}]
        base = tool.trace_hash(header, records)
        assert tool.trace_hash(header, [{"decision": 0, "hidden": {"tick": 1}}]) != base
        assert tool.trace_hash({"scenario": "y"}, records) != base
        assert tool.trace_hash(header, [dict(records[0])]) == base

    def test_the_first_difference_is_named(self, tool):
        found = tool.locate_difference(
            {"a": 1}, [{"x": [1, 2]}, {"x": [1, 2]}], {"a": 1}, [{"x": [1, 2]}, {"x": [1, 3]}]
        )
        assert found == {"decision": 1, "path": ".x[1]: 2 != 3"}


class TestCommittedTraces:
    def test_every_scenario_is_recorded_replayed_identically_and_matches_its_file(self, tool):
        index = json.loads((EVIDENCE / "index.json").read_text(encoding="utf-8"))
        assert set(index) == {scenario.name for scenario in tool.SCENARIOS}
        for name, entry in index.items():
            assert entry["error"] is None, name
            assert entry["replay_identical"] is True, name
            assert entry["unencodable_actions"] == 0, name
            header, records = tool.read_trace(EVIDENCE / entry["trace"])
            assert tool.trace_hash(header, records) == entry["trace_sha256"], name
            assert len(records) == entry["decisions"] + 1, name

    def test_every_tick_trace_repeats_and_agrees_with_its_decision_trace(self, tool):
        index = json.loads((EVIDENCE / "index.json").read_text(encoding="utf-8"))
        ticked = {name for name, entry in index.items() if entry.get("ticks")}
        assert ticked == set(tool.TICK_SCENARIOS)
        for name in ticked:
            ticks = index[name]["ticks"]
            assert ticks["repeat_identical"] is True, name
            assert ticks["matches_decision_trace"] is True, name
            header, rows = tool.read_trace(EVIDENCE / ticks["trace"])
            assert tool.trace_hash(header, rows) == ticks["trace_sha256"], name
            assert len(rows) == ticks["ticks"] + 1, name
            assert ticks["ticks"] == index[name]["decisions"] * header["ticks_per_decision"], name

    def test_the_verifier_scores_the_reference_and_refuses_the_exploit(self):
        index = json.loads((EVIDENCE / "index.json").read_text(encoding="utf-8"))
        reference = index["construct_smelting_line_reference"]["verification"]
        exploit = index["construct_smelting_line_exploit"]["verification"]
        assert reference["success"] and reference["machine_output"] >= 10
        assert exploit["uncapped_output"] > 0 and exploit["machine_output"] == 0
