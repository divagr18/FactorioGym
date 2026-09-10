"""Bounded action sequences (roadmap A2.3) and their per-action clocks (A2.4).

Everything here runs offline against the same stub environment and scripted
adapter the rest of the agent suite uses -- no engine, no provider, no key.

Two properties carry the section. The first is that batching is an *extension*:
`{"action": i}` still means what it always meant, and a `decisions.jsonl`
written by a batching run is still readable by `tools/replay.py`, which reads
`action_index`/`action_key`/`target`/`arguments` as scalars. The second is that
a batch is honest about partial failure -- it stops at the first refusal, never
undoes what already happened, and reports completed, failed and unexecuted
counts that add back up to what was asked for.
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import pytest

from factoriorl.agent import loop as loop_module
from factoriorl.agent.adapters import ScriptedAdapter
from factoriorl.agent.parsing import (
    MAX_ACTIONS_PER_SEQUENCE,
    DecisionFailure,
    ParsedAction,
    ParsedSequence,
    ParseFailure,
    parse_action,
    parse_sequence,
)
from factoriorl.agent.summary import action_vocabulary, legal_actions

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import diagnose_run  # noqa: E402
import replay  # noqa: E402

# The offline harness is defined once, in the loop's own test module, and
# imported rather than copied: a second `StubEnv` would be a second definition
# of what the environment surface is, and the two would drift. `_light_host`
# comes with it -- it is autouse, and without it every artifact written here
# would import torch to describe a GPU nothing in this file is about.
from test_agent_loop import (  # noqa: E402, F401
    StubEnv,
    _light_host,
    loop_for,
)

#: The chest `deliver` transfers into is the only reason the transfer actions
#: are legal in the stub, so removing it is how a later action of a batch gets
#: masked out between decisions.
TRANSFER = "take_iron-plate_5"


class SequencedEnv(StubEnv):
    """StubEnv with control over what fails, and over what stops being legal.

    Both are what a batch has to survive, and neither is expressible with the
    plain stub: the world a sequence was planned against is not the world its
    later actions run in, which is the entire reason A2.3 requires validation
    before every action rather than once per decision.
    """

    def __init__(
        self,
        *,
        fail_on: tuple[int, ...] = (),
        lose_entities_after: int | None = None,
        horizon: int = 64,
        **kwargs,
    ) -> None:
        super().__init__(horizon=horizon, **kwargs)
        #: Zero-based environment steps the engine refuses, as an out-of-reach
        #: rejection -- the shape `env.py` gives a real refusal.
        self.fail_on = frozenset(fail_on)
        self.lose_entities_after = lose_entities_after

    def step(self, index: int):
        ordinal = self.steps
        _, reward, terminated, truncated, info = super().step(index)
        moved = dict(self._observation)
        # A decision is a fixed interval of game time, so the observation clock
        # has to move for the freshness record to mean anything.
        moved["tick"] = int(moved["tick"]) + int(self.spec_.decision_ticks or 30)
        if self.lose_entities_after is not None and self.steps >= self.lose_entities_after:
            moved["entities"] = []
        self._observation = moved
        if ordinal in self.fail_on:
            info = {
                **info,
                "action_status": "rejected",
                "action_error": "out_of_reach",
                "success": False,
            }
        return None, reward, terminated, truncated, info


def vocabulary_and_legal(env: StubEnv):
    vocabulary = action_vocabulary(env)
    return vocabulary, legal_actions(vocabulary, env.action_masks())


def batch(*steps: dict, reason: str = "a plan") -> str:
    return json.dumps({"actions": list(steps), "reason": reason})


def segment(loop, **kwargs) -> dict:
    """Run one segment on an already-ready stub.

    `run_segment` streams to `decisions.jsonl` as it goes, and only `run()`
    creates the directory, so a test that drives the segment directly has to.
    """
    loop.run_dir.mkdir(parents=True, exist_ok=True)
    return loop.run_segment(0, fresh_memory=True, **kwargs)


# ------------------------------------------------------------------ the format


class TestTheReplyFormat:
    def test_a_single_action_is_a_sequence_of_one(self):
        """The old contract has to survive as a special case, not as a branch:
        every reply already written is `{"action": i}`."""
        env = StubEnv()
        vocabulary, legal = vocabulary_and_legal(env)
        parsed = parse_sequence('{"action": "wait", "reason": "stall"}', legal, vocabulary)
        assert isinstance(parsed, ParsedSequence)
        assert len(parsed) == 1
        assert parsed.first.key == "wait"
        assert parsed.first.reason == "stall"

    def test_the_one_action_entry_point_still_returns_one_action(self):
        """`parse_action` has callers of its own; batching must not retype it."""
        env = StubEnv()
        vocabulary, legal = vocabulary_and_legal(env)
        parsed = parse_action(
            batch({"action": "wait"}, {"action": "move_north"}), legal, vocabulary
        )
        assert isinstance(parsed, ParsedAction)
        assert parsed.key == "wait"

    def test_a_batch_keeps_the_order_it_was_written_in(self):
        env = StubEnv()
        vocabulary, legal = vocabulary_and_legal(env)
        parsed = parse_sequence(
            batch({"action": 3}, {"action": "wait"}, {"action": 1}), legal, vocabulary
        )
        assert isinstance(parsed, ParsedSequence)
        assert [a.index for a in parsed.actions] == [3, vocabulary_index(vocabulary, "wait"), 1]

    def test_a_batch_carries_targets_and_arguments_per_action(self):
        env = StubEnv()
        vocabulary, legal = vocabulary_and_legal(env)
        parsed = parse_sequence(
            batch({"action": "wait"}, {"action": TRANSFER, "target": "e1"}), legal, vocabulary
        )
        assert isinstance(parsed, ParsedSequence)
        assert parsed.actions[1].target == "e1"

    def test_each_action_may_give_its_own_reason(self):
        """Otherwise a replay of a batch is eight copies of one sentence."""
        env = StubEnv()
        vocabulary, legal = vocabulary_and_legal(env)
        parsed = parse_sequence(
            batch({"action": "wait", "reason": "let it smelt"}, {"action": 1}, reason="head east"),
            legal,
            vocabulary,
        )
        assert isinstance(parsed, ParsedSequence)
        assert parsed.reason == "head east"
        assert parsed.actions[0].reason == "let it smelt"
        # No reason of its own falls back to the batch's, so nothing is blank.
        assert parsed.actions[1].reason == "head east"

    def test_exactly_the_cap_is_accepted(self):
        env = StubEnv()
        vocabulary, legal = vocabulary_and_legal(env)
        parsed = parse_sequence(
            batch(*([{"action": "wait"}] * MAX_ACTIONS_PER_SEQUENCE)), legal, vocabulary
        )
        assert isinstance(parsed, ParsedSequence)
        assert len(parsed) == MAX_ACTIONS_PER_SEQUENCE

    def test_more_than_the_cap_is_refused_rather_than_truncated(self):
        """Silently running the first eight of twelve would execute a plan the
        model did not write and record it as the one it did."""
        env = StubEnv()
        vocabulary, legal = vocabulary_and_legal(env)
        outcome = parse_sequence(
            batch(*([{"action": "wait"}] * (MAX_ACTIONS_PER_SEQUENCE + 1))), legal, vocabulary
        )
        assert isinstance(outcome, ParseFailure)
        assert outcome.failure is DecisionFailure.TOO_MANY_ACTIONS
        assert str(MAX_ACTIONS_PER_SEQUENCE) in outcome.detail

    def test_the_prompt_and_the_parser_agree_on_the_cap(self):
        """A prompt inviting nine actions and a parser refusing them is a
        rejection the model has no way to act on."""
        assert f"up to {MAX_ACTIONS_PER_SEQUENCE} of them" in loop_module.SYSTEM_PROMPT
        assert "$MAX" not in loop_module.SYSTEM_PROMPT

    def test_an_empty_batch_is_not_an_action(self):
        env = StubEnv()
        vocabulary, legal = vocabulary_and_legal(env)
        outcome = parse_sequence('{"actions": [], "reason": "nothing"}', legal, vocabulary)
        assert isinstance(outcome, ParseFailure)
        assert outcome.failure is DecisionFailure.UNPARSEABLE

    def test_a_reply_carrying_both_shapes_is_refused(self):
        """There is no reading of `action` plus `actions` that is not a guess,
        and guessing here runs a plan nobody wrote."""
        env = StubEnv()
        vocabulary, legal = vocabulary_and_legal(env)
        outcome = parse_sequence(
            json.dumps({"action": 1, "actions": [{"action": 2}]}), legal, vocabulary
        )
        assert isinstance(outcome, ParseFailure)
        assert "both" in outcome.detail

    def test_bare_indices_in_a_batch_are_understood(self):
        """`{"actions": [3, 7]}` is unambiguous; filing it as malformed would
        repeat the nested-object defect that cost 37 well-formed replies."""
        env = StubEnv()
        vocabulary, legal = vocabulary_and_legal(env)
        parsed = parse_sequence('{"actions": [3, 1], "reason": "go"}', legal, vocabulary)
        assert isinstance(parsed, ParsedSequence)
        assert [a.index for a in parsed.actions] == [3, 1]

    def test_a_refusal_says_which_step_it_was_about(self):
        """A correction that does not name the step leaves the model guessing
        which of eight actions to change."""
        env = StubEnv()
        vocabulary, legal = vocabulary_and_legal(env)
        outcome = parse_sequence(
            batch({"action": "wait"}, {"action": "wait"}, {"action": "teleport_to_goal"}),
            legal,
            vocabulary,
        )
        assert isinstance(outcome, ParseFailure)
        assert outcome.failure is DecisionFailure.UNKNOWN_ACTION
        assert "action 3 of 3" in outcome.detail

    def test_a_one_action_reply_keeps_the_message_it_always_had(self):
        """The position prefix must not appear on the shape that predates it."""
        env = StubEnv()
        vocabulary, legal = vocabulary_and_legal(env)
        outcome = parse_sequence('{"action": "teleport_to_goal"}', legal, vocabulary)
        assert isinstance(outcome, ParseFailure)
        assert not outcome.detail.startswith("action 1 of")

    def test_only_the_first_action_is_checked_against_the_current_mask(self):
        """A plan whose later steps become legal only once the earlier ones have
        run is the point of batching; refusing it at parse time would leave the
        feature able to express nothing but repetition."""
        env = StubEnv(entities=False)
        vocabulary, legal = vocabulary_and_legal(env)
        masked = vocabulary_index(vocabulary, TRANSFER)
        assert masked not in {a.index for a in legal}, "the case would be vacuous"
        parsed = parse_sequence(batch({"action": "wait"}, {"action": masked}), legal, vocabulary)
        assert isinstance(parsed, ParsedSequence)
        assert parsed.actions[1].index == masked

    def test_the_first_action_is_still_checked_against_the_current_mask(self):
        env = StubEnv(entities=False)
        vocabulary, legal = vocabulary_and_legal(env)
        masked = vocabulary_index(vocabulary, TRANSFER)
        outcome = parse_sequence(batch({"action": masked}, {"action": "wait"}), legal, vocabulary)
        assert isinstance(outcome, ParseFailure)
        assert outcome.failure is DecisionFailure.ILLEGAL_ACTION


def vocabulary_index(vocabulary, key: str) -> int:
    return [k for k, _ in vocabulary].index(key)


# --------------------------------------------------------------- execution


class TestASequenceRunsSerially:
    def test_every_action_runs_in_the_order_it_was_given(self, tmp_path):
        env = SequencedEnv()
        adapter = ScriptedAdapter([batch({"action": 3}, {"action": "wait"}, {"action": 1})])
        loop = loop_for(env, adapter, tmp_path)
        segment(loop, budget=3)

        assert env.taken == [3, env.catalog.wait_index, 1]

    def test_a_batch_is_one_decision_and_as_many_steps_as_it_ran(self, tmp_path):
        """Counting a batch of three as one step would let it spend three times
        the task's tick budget while the segment believed it had used one."""
        env = SequencedEnv()
        adapter = ScriptedAdapter([batch({"action": 3}, {"action": 1}, {"action": "wait"})])
        loop = loop_for(env, adapter, tmp_path)
        result = segment(loop, budget=3)

        assert len(loop.decisions) == 1
        assert adapter.calls == 1
        assert result["steps"] == 3
        assert loop.decisions[0].result["steps"] == 3

    def test_the_counts_add_up_to_what_was_requested(self, tmp_path):
        """A2.3 asks for completed, failed and unexecuted. Three numbers that do
        not sum to the request describe some other run."""
        env = SequencedEnv(fail_on=(1,))
        adapter = ScriptedAdapter([batch(*([{"action": "wait"}] * 5))] * 4)
        loop = loop_for(env, adapter, tmp_path)
        segment(loop, budget=8)

        counted = loop.decisions[0].sequence
        assert counted["requested"] == 5
        assert counted["completed"] + counted["failed"] + counted["unexecuted"] == 5
        assert (counted["completed"], counted["failed"], counted["unexecuted"]) == (1, 1, 3)

    def test_a_failure_stops_the_rest(self, tmp_path):
        env = SequencedEnv(fail_on=(1,))
        adapter = ScriptedAdapter([batch(*([{"action": "wait"}] * 5))] * 4)
        loop = loop_for(env, adapter, tmp_path)
        segment(loop, budget=8)

        outcomes = loop.decisions[0].outcomes
        assert [o["status"] for o in outcomes] == [
            "completed",
            "failed",
            "unexecuted",
            "unexecuted",
            "unexecuted",
        ]
        assert outcomes[1]["action_error"] == "out_of_reach"
        assert loop.decisions[0].sequence_stopped == "action_failed"

    def test_nothing_that_already_ran_is_undone(self, tmp_path):
        """There is no rollback in this environment, and inventing one would
        mean issuing *more* game actions in the agent's name to reverse its
        own -- a different world state, recorded as something the agent did."""
        env = SequencedEnv(fail_on=(2,))
        adapter = ScriptedAdapter([batch({"action": 3}, {"action": 1}, {"action": 2})] * 4)
        loop = loop_for(env, adapter, tmp_path)
        segment(loop, budget=3)

        # Exactly the three that were dispatched, and no compensating fourth.
        assert env.taken == [3, 1, 2]

    def test_the_mask_is_checked_again_between_actions(self, tmp_path):
        """Action 2 is dispatched into the world action 1 left behind, which may
        no longer permit it -- mining the last ore in reach masks `mine` out
        again, and a chest that has gone out of sight is not transferable."""
        env = SequencedEnv(lose_entities_after=1)
        transfer = vocabulary_index(action_vocabulary(env), TRANSFER)
        adapter = ScriptedAdapter([batch({"action": "wait"}, {"action": transfer})] * 4)
        loop = loop_for(env, adapter, tmp_path)
        segment(loop, budget=4)

        outcomes = loop.decisions[0].outcomes
        assert outcomes[0]["status"] == "completed"
        assert outcomes[1]["status"] == "failed"
        assert outcomes[1]["failure"] == DecisionFailure.ILLEGAL_ACTION.value
        assert loop.decisions[0].sequence_stopped == "refused"
        # Refused before dispatch, so the environment never saw it at all.
        assert transfer not in env.taken

    def test_step_arguments_revalidates_rather_than_reusing_a_bound_payload(self):
        """The re-validation A2.3 wants for a parameterized action is already a
        property of the shared entry point, and this is what makes it one:
        `step_arguments` re-derives the domains and the binding context on every
        call, so action 2 of a batch is checked against the world action 1 left.
        A cached payload or a hoisted domain would silently undo that."""
        from factoriorl.env import FactorioEnv

        source = inspect.getsource(FactorioEnv.step_arguments)
        assert "self.argument_domains()" in source
        assert "self._context()" in source

    def test_a_batch_cannot_spend_steps_the_episode_does_not_have(self, tmp_path):
        env = SequencedEnv()
        adapter = ScriptedAdapter([batch(*([{"action": "wait"}] * 6))] * 4)
        loop = loop_for(env, adapter, tmp_path)
        result = segment(loop, budget=2)

        assert result["steps"] == 2
        assert env.taken == [env.catalog.wait_index] * 2
        counted = loop.decisions[0].sequence
        assert (counted["completed"], counted["unexecuted"]) == (2, 4)
        assert loop.decisions[0].sequence_stopped == "step_budget"

    def test_a_terminal_action_stops_the_rest(self, tmp_path):
        env = SequencedEnv(horizon=2)
        adapter = ScriptedAdapter([batch(*([{"action": "wait"}] * 4))] * 4)
        loop = loop_for(env, adapter, tmp_path)
        result = segment(loop, budget=6)

        assert result["stopped"] == "terminated"
        assert loop.decisions[0].sequence_stopped == "terminated"
        assert loop.decisions[0].sequence["completed"] == 2


class TestASequenceIsBoundedInTime:
    def test_the_bound_is_thirty_seconds(self):
        """A2.3 names the number; a batch that could run for minutes would be an
        escape from the deadline the whole run is held to."""
        assert loop_module.SEQUENCE_WALL_SECONDS == 30.0

    def test_the_batch_stops_at_its_own_wall_clock(self, tmp_path, monkeypatch):
        monkeypatch.setattr(loop_module, "SEQUENCE_WALL_SECONDS", 0.0)
        env = SequencedEnv()
        adapter = ScriptedAdapter([batch(*([{"action": "wait"}] * 4))] * 4)
        loop = loop_for(env, adapter, tmp_path)
        segment(loop, budget=8)

        first = loop.decisions[0]
        assert first.sequence_stopped == "sequence_deadline"
        # The first action is exempt: it was validated against the state the
        # model was shown, and refusing to run anything at all would turn an
        # expired batch into a decision the model never got to make.
        assert first.sequence["completed"] == 1
        assert first.sequence["unexecuted"] == 3

    def test_the_run_deadline_stops_a_batch_it_falls_inside(self, tmp_path):
        """Whichever comes first. `until` is the run's own wall clock -- the CLI
        passes `clock.expired` -- and a batch must not outlive it."""
        env = SequencedEnv()
        adapter = ScriptedAdapter([batch(*([{"action": "wait"}] * 4))] * 4)
        loop = loop_for(env, adapter, tmp_path)
        result = segment(loop, budget=8, until=lambda: True)

        first = loop.decisions[0]
        assert first.sequence_stopped == "run_deadline"
        assert first.sequence["completed"] == 1
        assert result["stopped"] == "until"


class TestEachActionCarriesItsOwnClocks:
    def test_the_two_ticks_and_the_wall_time_are_recorded_per_action(self, tmp_path):
        """A2.4. A record carrying only one of the three cannot tell a slow
        provider from a slow world, and a batch is where they diverge: three
        actions on one decision advance the game clock three times."""
        env = SequencedEnv()
        adapter = ScriptedAdapter([batch(*([{"action": "wait"}] * 3))] * 4)
        loop = loop_for(env, adapter, tmp_path)
        segment(loop, budget=3)

        interval = int(env.spec_.decision_ticks or 30)
        outcomes = loop.decisions[0].outcomes
        for position, outcome in enumerate(outcomes):
            assert outcome["observation_tick"] == 120 + interval * position
            assert outcome["execution_tick"] == outcome["observation_tick"] + interval
            assert outcome["elapsed_ms"] >= 0.0

    def test_a_later_action_is_dispatched_against_a_fresher_observation(self, tmp_path):
        """The freshness claim in one assertion: action 2 was chosen against the
        tick action 1 produced, not against the one the batch was planned in."""
        env = SequencedEnv()
        adapter = ScriptedAdapter([batch({"action": "wait"}, {"action": "wait"})] * 4)
        loop = loop_for(env, adapter, tmp_path)
        segment(loop, budget=2)

        first, second = loop.decisions[0].outcomes
        assert second["observation_tick"] == first["execution_tick"]


# ----------------------------------------------------------- compatibility


class TestTheRecordStaysReadable:
    def test_the_scalar_fields_still_name_the_first_action(self, tmp_path):
        """`tools/replay.py` reads `action_index`, `action_key`, `target` and
        `arguments` as scalars -- it colours a timeline by `action_key` and
        prints `action_key #action_index` as "chosen". Making them lists would
        leave every run written so far unreadable by the same tool."""
        env = SequencedEnv()
        adapter = ScriptedAdapter([batch({"action": 3}, {"action": 1})] * 4)
        loop = loop_for(env, adapter, tmp_path)
        segment(loop, budget=2)

        body = loop.decisions[0].to_dict()
        assert body["action_index"] == 3
        assert body["action_key"] == env.catalog.keys()[3]
        assert isinstance(body["arguments"], dict)
        assert body["target"] is None
        # The rest of the sequence sits beside them, never inside them.
        assert [a["index"] for a in body["actions"]] == [3, 1]
        assert body["sequence"]["requested"] == 2

    def test_a_fallback_still_executes_and_still_reads_as_a_fallback(self, tmp_path):
        """The fallback is a sequence of one so there is a single executor, but
        `resolution` -- not the sequence -- is what says nobody chose it."""
        env = SequencedEnv(entities=False, horizon=1)
        adapter = ScriptedAdapter(["I would rather not."] * 3)
        loop = loop_for(env, adapter, tmp_path)
        segment(loop, budget=1)

        assert env.taken == [env.catalog.wait_index]
        assert loop.decisions[0].resolution == "fallback_wait"
        assert loop.decisions[0].sequence["completed"] == 1

    def test_replay_still_builds_a_page_from_a_batched_run(self, tmp_path):
        env = SequencedEnv()
        adapter = ScriptedAdapter([batch({"action": 3}, {"action": 1})] * 8)
        loop = loop_for(env, adapter, tmp_path, episodes=1, max_steps=4)
        loop.run()

        page = replay.build(loop.run_dir, tmp_path / "replay.html")
        text = page.read_text(encoding="utf-8")
        assert "decisions" in text
        # The counts line is built from `resolution` and `result.success`, and
        # the timeline from `action_key`: all three survive batching.
        assert env.catalog.keys()[3] in text

    def test_replay_renders_every_member_of_a_batch_and_its_own_outcome(self, tmp_path):
        """Gate A4: "replay reconstructs a mixed batch with partial failure."

        The test above only asserted the page *builds* and contains the first
        action's key -- which it did while rendering exactly one row per
        decision and hiding the other seven members. This asserts the second
        member is on the page, that the sequence's stop reason is on the page,
        and that the counts are over actions rather than decisions.
        """
        env = SequencedEnv(fail_on=(1,))
        adapter = ScriptedAdapter([batch(*([{"action": "wait"}] * 4))] * 8)
        loop = loop_for(env, adapter, tmp_path, episodes=1, max_steps=4)
        loop.run()

        counted = loop.decisions[0].sequence
        assert (counted["completed"], counted["failed"], counted["unexecuted"]) == (1, 1, 2)

        page = replay.build(loop.run_dir, tmp_path / "replay.html")
        text = page.read_text(encoding="utf-8")
        # The renderer reads these; a page that carries the data and displays
        # none of it is what this replaces.
        for needed in ("function members(", "sequence (", "refused before execution"):
            assert needed in text, needed
        # And the record it reads from actually holds the members.
        assert '"unexecuted"' in text
        assert '"actions":' in text

    def test_tool_events_are_written_one_line_per_action(self, tmp_path):
        """A4.1 asks for a tool-event log. The data existed only nested inside
        a decision, so every reader that did not know a decision may hold eight
        actions under-reported by up to 8x -- and `tools/replay.py` was one."""
        env = SequencedEnv()
        adapter = ScriptedAdapter([batch({"action": 3}, {"action": 1})] * 8)
        loop = loop_for(env, adapter, tmp_path, episodes=1, max_steps=4)
        loop.run()

        lines = (loop.run_dir / "tool_events.jsonl").read_text(encoding="utf-8").splitlines()
        rows = [json.loads(line) for line in lines if line.strip()]

        assert len(rows) > len(loop.decisions), "a batch is more actions than decisions"
        assert {row["key"] for row in rows} == {env.catalog.keys()[3], env.catalog.keys()[1]}
        assert all("decision" in row and "position" in row for row in rows)

    def test_the_run_writes_the_artifacts_a4_1_asks_for(self, tmp_path):
        env = SequencedEnv()
        adapter = ScriptedAdapter([batch({"action": 3}, {"action": 1})] * 8)
        loop = loop_for(env, adapter, tmp_path, episodes=1, max_steps=4)
        loop.run()

        for name in (
            "config.json",
            "manifest.json",
            "status.json",
            "decisions.jsonl",
            "tool_events.jsonl",
            "result.json",
            "summary.json",
        ):
            assert (loop.run_dir / name).exists(), name

        summary = json.loads((loop.run_dir / "summary.json").read_text(encoding="utf-8"))
        # A decision is what the model was asked; an action is what the world
        # was asked. Reporting one as the other is the defect this fixes.
        assert summary["tool_actions"] > summary["decisions"]
        assert summary["interventions"] == []

    def test_diagnose_run_still_reads_the_manifest_and_the_result(self, tmp_path):
        """It reads `result.json` and `manifest.json`, never `decisions.jsonl`,
        so the check that matters is that neither of those two changed shape."""
        env = SequencedEnv()
        adapter = ScriptedAdapter([batch({"action": 3}, {"action": 1})] * 8)
        loop = loop_for(env, adapter, tmp_path, episodes=1, max_steps=4)
        loop.run()

        result_path = loop.run_dir / "result.json"
        loaded = diagnose_run.load(str(result_path))
        assert loaded["run_id"] == loop.run_id
        found = diagnose_run.budgets(str(result_path), loaded["task"])
        assert found is not None
        assert found["source"] == "this run's manifest"
        assert found["max_decision_steps"] == env.spec_.max_decision_steps

    def test_a_decisions_line_is_still_one_json_object_per_decision(self, tmp_path):
        env = SequencedEnv()
        adapter = ScriptedAdapter([batch({"action": 3}, {"action": 1})] * 8)
        loop = loop_for(env, adapter, tmp_path, episodes=1, max_steps=4)
        loop.run()

        lines = [
            json.loads(line)
            for line in (loop.run_dir / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert len(lines) == len(loop.decisions)
        for line in lines:
            assert line["sequence"]["requested"] == len(line["actions"])
            # `parsed` still means one action, because that is what it means in
            # every decisions.jsonl written before batching existed.
            assert line["attempts"][0]["parsed"]["index"] == line["action_index"]
            assert len(line["attempts"][0]["parsed_actions"]) == 2


@pytest.mark.parametrize("phrase", loop_module.FORBIDDEN_IN_PROMPT)
def test_the_sequence_instructions_leak_no_reference_geometry(phrase):
    """R3.2 again, because the prompt grew: the batch instructions must not
    smuggle in the build sequence the reference solver had to measure."""
    assert phrase.lower() not in loop_module.SYSTEM_PROMPT.lower()
