"""A report must not turn a missing file into a claim about the agent.

`tools/run_report.py` read almost everything from `summary.json`, which is
written during finalization. A run that is stopped never reaches finalization,
so the file does not exist -- and the report then said, in order:

* "gameplay Nones of a Nones limit" and "spend $None of a $None cap", which read
  as measurements rather than as their own absence;
* "actions by verb: none", two lines under "381 tool actions";
* "none: the agent never used the `plan` field", for a run that stated a plan on
  126 of its 253 decisions -- including the one line that explains the whole
  run, "Practical maximum without stone".

The last one is the serious one. Every other field was merely unhelpful; that
one was false, and it was false in the direction of blaming the agent for a gap
in the harness. Everything it needed was already on disk: `decisions.jsonl`,
`tool_events.jsonl` and `production.jsonl` are appended as the run goes.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import run_report  # noqa: E402


class TestPlansComeFromWhatWasActuallySent:
    def test_plans_are_recovered_from_the_decisions_of_a_stopped_run(self):
        plans = run_report.stated_plans(
            [
                {"step": 2, "plan": "bootstrap iron"},
                {"step": 3, "reason": "no plan this turn"},
                {"step": 4, "plan": "bootstrap iron"},
                {"step": 9, "plan": "practical maximum without stone"},
            ]
        )
        assert [p["plan"] for p in plans] == [
            "bootstrap iron",
            "practical maximum without stone",
        ]

    def test_a_plan_held_across_turns_is_reported_once_with_its_span(self):
        """126 plan statements over 253 decisions are mostly the same few plans
        restated. Listing every restatement buries the two or three moments the
        agent actually changed its mind, which is the only thing worth reading."""
        plans = run_report.stated_plans(
            [
                {"step": 1, "plan": "haul coal"},
                {"step": 2, "plan": "haul coal"},
                {"step": 7, "plan": "haul coal"},
            ]
        )
        assert len(plans) == 1
        assert plans[0]["stated_at"] == 1
        assert plans[0]["held_until"] == 7

    def test_a_run_that_truly_stated_none_still_says_so(self):
        assert run_report.stated_plans([{"step": 1, "reason": "x"}]) == []


class TestAMissingNumberSaysSoRatherThanNone:
    def test_an_absent_value_is_marked_unrecorded(self):
        assert run_report._or_unrecorded(None) == run_report.UNRECORDED
        assert run_report._or_unrecorded(None, "s") == run_report.UNRECORDED

    def test_a_present_value_keeps_its_units(self):
        assert run_report._or_unrecorded(12.5, "s") == "12.5s"

    def test_zero_is_a_measurement_and_not_an_absence(self):
        """`or`-style defaulting would have swallowed this: a run that spent
        nothing and a run that did not record its spend are different facts."""
        assert run_report._or_unrecorded(0) == "0"
        assert run_report._or_unrecorded(0.0, "s") == "0.0s"


class TestFirstMachineOutput:
    def test_it_is_the_tick_of_the_first_sample_with_machine_output(self):
        assert (
            run_report.first_machine_output(
                [
                    {"tick": 100, "machine_produced": {}},
                    {"tick": 3385, "machine_produced": {"iron-plate": 1}},
                    {"tick": 4000, "machine_produced": {"iron-plate": 9}},
                ]
            )
            == 3385
        )

    def test_a_run_that_produced_nothing_reports_nothing_rather_than_zero(self):
        assert run_report.first_machine_output([{"tick": 100, "machine_produced": {}}]) is None
        assert run_report.first_machine_output([]) is None
