"""Gate A3: persistent agency and a useful objective.

The gate asks for offline multi-turn fixtures showing that the plan survives
compaction, stale facts stay labelled, failures enter memory, a stopped batch
returns control, and evaluator data does not enter the prompt -- and that one
interface serves both a single action and a batch.

Everything here runs against the same stub environment and scripted adapter as
the rest of the agent suite: no engine, no provider, no key.

The three defects this section closes were all *declared but unreachable*
rather than absent, which is why the tests are written against behaviour a
reader can see in a rendered prompt rather than against the presence of fields:

  - `Memory.record_plan` / `close_plan` had zero callers, so `CURRENT PLAN` and
    the closed-plan heading were rendered and permanently empty.
  - `Fact.source` admitted only observation-sourced entries, so a claim the
    model wanted to keep had nowhere to go that was not a lie about provenance.
  - Nothing counted identical failures, so an agent could send the same refused
    action until its wall clock ran out.
"""

from __future__ import annotations

import sys
from pathlib import Path

from factoriorl import worlds
from factoriorl.agent.loop import FORBIDDEN_IN_OBJECTIVE, FORBIDDEN_IN_PROMPT
from factoriorl.agent.memory import STALL_THRESHOLD, Memory
from factoriorl.agent.parsing import ParsedSequence, parse_sequence
from factoriorl.agent.summary import action_vocabulary, legal_actions, objective_block

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from test_agent_loop import (  # noqa: E402, F401
    StubEnv,
    _light_host,
    loop_for,
)

from factoriorl.agent.adapters import ScriptedAdapter  # noqa: E402


def _segment(loop, **kwargs) -> dict:
    loop.run_dir.mkdir(parents=True, exist_ok=True)
    return loop.run_segment(0, fresh_memory=True, **kwargs)


def _parsed(env: StubEnv, text: str):
    vocabulary = action_vocabulary(env)
    legal = legal_actions(vocabulary, env.action_masks())
    return parse_sequence(text, legal, vocabulary)


# --------------------------------------------------------------- A3.1 objective


def test_the_open_world_states_an_objective_and_a_benchmark_task_does_not():
    """A3.1's instruction belongs to the world, not to the shared prompt.

    A task's objective is its `description`, already rendered per turn. Putting
    the open-world objective anywhere a frozen task could see it would change a
    prompt every existing result was produced against.
    """
    assert objective_block(worlds.get("open_factory")) == ""  # a mode is not an env
    assert objective_block(StubEnv()) == ""

    objective = worlds.get("open_factory").objective
    assert "Build and expand a factory" in objective
    # The direction the 0.4.0 rewrite names, where the old text left the goal
    # open. Pinned because it is the substantive change: an objective that stops
    # naming it has quietly gone back to "expand production that is useful to
    # you", which two runs showed resolves to "keep the current loop running".
    assert "research" in objective.lower()
    assert "progress is measured" in objective.lower()
    assert "until the controller" in objective


def test_the_objective_prescribes_nothing_a3_1_forbids():
    """No coordinates, no machine ordering, no build sequence, no threshold.

    A digit is the tell for the first and the fourth: a coordinate, a count, or
    a target number cannot be written without one.
    """
    objective = worlds.get("open_factory").objective
    lowered = objective.lower()

    assert not [phrase for phrase in FORBIDDEN_IN_OBJECTIVE if phrase in lowered]
    assert not [phrase for phrase in FORBIDDEN_IN_PROMPT if phrase in lowered]
    assert not [character for character in objective if character.isdigit()]


def test_the_objective_is_invariant_so_it_belongs_in_the_cached_prefix():
    """It never changes, and a line repeated per turn is paid for every turn.

    The A2 measurement behind this: hoisting 1,440 invariant characters out of
    the turn cut a 300-turn run's context at turn 900 from 680k tokens to 400k.
    """
    first = worlds.get("open_factory").objective
    second = worlds.get("open_factory").objective
    assert first == second and first


# ------------------------------------------------------- A3.2 plan and note


def test_one_interface_serves_a_single_action_and_a_batch():
    """The gate's last clause. Both shapes carry the standing fields."""
    env = StubEnv()
    single = _parsed(env, '{"action": 0, "reason": "go", "plan": "walk north", "note": "ore east"}')
    batch = _parsed(
        env,
        '{"actions": [{"action": 0}, {"action": 0}], "reason": "go", '
        '"plan": "walk north", "note": "ore east"}',
    )
    for outcome in (single, batch):
        assert isinstance(outcome, ParsedSequence), outcome
        assert outcome.plan == "walk north"
        assert outcome.note == "ore east"
    assert len(single) == 1 and len(batch) == 2


def test_a_reply_without_the_new_fields_is_unchanged():
    """Every fixture written before A3 has to keep meaning what it meant."""
    outcome = _parsed(StubEnv(), '{"action": 0, "reason": "go"}')
    assert isinstance(outcome, ParsedSequence)
    assert outcome.plan == "" and outcome.note == ""


def test_a_plan_stated_once_is_still_in_the_prompt_forty_turns_later():
    """A3.2: compaction must not drop outstanding work.

    Forty turns of successful actions is exactly the traffic that compaction
    exists to shed, so a plan that survives it survives the realistic case.
    """
    memory = Memory()
    memory.record_plan(1, "smelt iron, then build a second furnace")
    for step in range(2, 42):
        memory.record_action(step, "walk_to_position", status="completed")
        memory.compact()
    memory.step = 41

    rendered = memory.render()
    assert "CURRENT PLAN" in rendered
    assert "smelt iron, then build a second furnace" in rendered
    # And the successful tail really was shed, so the test is not vacuous.
    assert len([a for a in memory.attempts if not a.failed]) < 40


def test_an_unresolved_failure_survives_compaction_and_a_repeated_success_does_not():
    memory = Memory()
    memory.record_action(1, "place_at", error="collision", arguments={"position": [1.5, 2.5]})
    for step in range(2, 30):
        memory.record_action(step, "walk_to_position", status="completed")
    memory.compact()

    assert any(a.error == "collision" for a in memory.attempts)
    assert len([a for a in memory.attempts if not a.failed]) < 28


def test_a_note_is_never_rendered_as_something_the_game_confirmed():
    """A3.2's separation, checked where it can act on anything: the prompt."""
    memory = Memory()
    memory.record_note(4, "the coal is north-east, about forty tiles")
    memory.step = 4

    rendered = memory.render()
    assert "THE AGENT'S OWN NOTES (unverified -- you wrote these)" in rendered
    assert "REMEMBERED" not in rendered
    assert memory.notes[0].source == "model"


# ------------------------------------------------------- A3.3 stall handling


def test_three_identical_failures_are_reported_as_a_stall():
    memory = Memory()
    for step in range(STALL_THRESHOLD):
        memory.record_action(
            step, "place_at", error="collision", arguments={"position": [12.5, -3.5]}
        )
    memory.step = STALL_THRESHOLD

    rendered = memory.render()
    assert "REPEATED FAILURE -- this is not working" in rendered
    assert f"has failed {STALL_THRESHOLD} times: collision" in rendered
    assert "Do something different" in rendered
    assert '"plan"' in rendered


def test_two_identical_failures_are_not_yet_a_stall():
    """Two is a coincidence. Crying stall at it would train the agent to give
    up on something that was about to work."""
    memory = Memory()
    for step in range(STALL_THRESHOLD - 1):
        memory.record_action(
            step, "place_at", error="collision", arguments={"position": [1.5, 2.5]}
        )
    memory.step = 2

    assert memory.repeated_failures() == []
    assert "REPEATED FAILURE" not in memory.render()


def test_the_same_verb_at_different_arguments_is_not_one_repeated_failure():
    """`place_at` at three different tiles is three things tried once."""
    memory = Memory()
    for step in range(STALL_THRESHOLD):
        memory.record_action(
            step, "place_at", error="collision", arguments={"position": [float(step), 0.5]}
        )

    assert memory.repeated_failures() == []


def test_the_loop_never_substitutes_an_action_for_the_agent():
    """A3.3 forbids silently selecting something that works.

    Checked against the source, because the property is an *absence*: there is
    no code path from a repeated failure to a chosen action, and the only thing
    a stall does is write text the model reads.
    """
    import inspect

    from factoriorl.agent.loop import AgentLoop

    source = inspect.getsource(AgentLoop)
    stall_block = source[source.index("for entry in self.memory.repeated_failures()") :]
    stall_block = stall_block[: stall_block.index("self.memory.compact()")]
    # It may close a plan and label the decision. It may not pick an action.
    assert "close_plan" in stall_block
    assert "env.step" not in stall_block
    assert "action_index" not in stall_block


# ------------------------------------------------- the loop, end to end


def test_a_plan_stated_on_turn_one_is_in_turn_twos_prompt(tmp_path):
    """The whole point, and the half that memory alone cannot show.

    `record_plan` had no caller before A3, so a `Memory` that renders a plan
    correctly proves nothing about whether a plan ever reaches a prompt.
    """
    env = StubEnv()
    adapter = ScriptedAdapter(
        [
            '{"action": 0, "reason": "start", "plan": "walk north then mine",'
            ' "note": "ore is east of spawn"}',
            '{"action": 0, "reason": "keep going"}',
            '{"action": 0, "reason": "keep going"}',
        ]
    )
    loop = loop_for(env, adapter, tmp_path)
    _segment(loop, budget=3)

    assert loop.decisions[0].plan == "walk north then mine"
    assert loop.decisions[0].note == "ore is east of spawn"
    # The second decision never restated either, and both are still in front of
    # the model when it is asked again.
    later = loop.decisions[1].summary.render() + loop.memory.render()
    assert "walk north then mine" in later
    assert "ore is east of spawn" in later
    assert "THE AGENT'S OWN NOTES" in later


def test_the_reply_reason_reaches_the_decision_record(tmp_path):
    """It was recorded only inside `attempts[].parsed_actions[]`, which is why
    `tools/replay.py` never showed it despite offering to filter on it."""
    env = StubEnv()
    adapter = ScriptedAdapter(['{"action": 0, "reason": "because the ore is there"}'])
    loop = loop_for(env, adapter, tmp_path)
    _segment(loop, budget=1)

    assert loop.decisions[0].reason == "because the ore is there"
    assert loop.decisions[0].to_dict()["reason"] == "because the ore is there"


def test_a_stall_closes_the_open_plan_exactly_once(tmp_path):
    """A plan whose actions keep failing identically is not still running.

    Closed once per signature: closing it every turn would fill the closed-plan
    list with copies of one plan and evict the ones that actually differ.
    """
    memory = Memory()
    memory.record_plan(0, "place a furnace right here")
    for step in range(STALL_THRESHOLD + 4):
        memory.record_action(
            step, "place_at", error="collision", arguments={"position": [1.5, 2.5]}
        )

    stalls = memory.repeated_failures()
    assert len(stalls) == 1
    assert stalls[0]["count"] == STALL_THRESHOLD + 4


# --------------------------------- R5: failures are conditions, not verdicts


def test_a_success_clears_the_repeated_failure_warning():
    """A three-strike rule that never resets turns a temporary condition into a
    permanent prohibition. Measured case: three failed furnace crafts for want
    of stone, then stone arrives and one succeeds -- and the agent was still
    being told it would fail again."""
    memory = Memory()
    for step in range(STALL_THRESHOLD):
        memory.record_action(
            step,
            "craft_recipe",
            error="no_items",
            arguments={"recipe": "stone-furnace", "count": 1},
        )
    assert memory.repeated_failures(), "three identical failures should warn"

    memory.record_action(
        9, "craft_recipe", status="completed", arguments={"recipe": "stone-furnace", "count": 1}
    )

    assert memory.repeated_failures() == []
    assert "REPEATED FAILURE" not in memory.render()


def test_a_success_clears_only_its_own_signature():
    memory = Memory()
    for step in range(STALL_THRESHOLD):
        memory.record_action(
            step, "place_at", error="collision", arguments={"position": [1.5, 2.5]}
        )
        memory.record_action(
            step, "craft_recipe", error="no_items", arguments={"recipe": "stone-furnace"}
        )
    memory.record_action(
        9, "craft_recipe", status="completed", arguments={"recipe": "stone-furnace"}
    )

    stuck = memory.repeated_failures()
    assert [entry["action"] for entry in stuck] == ["place_at"]


def test_an_ongoing_action_is_not_a_failure():
    """A walk, a mine and a craft all answer `running` and finish later. Calling
    that refused work filled the record with imaginary failures and pushed the
    real ones out of a bounded list."""
    memory = Memory()
    for step in range(STALL_THRESHOLD + 2):
        memory.record_action(step, "mine_at", status="running", arguments={"handle": "h7"})

    assert memory.failures() == []
    assert memory.repeated_failures() == []
    assert "REPEATED FAILURE" not in memory.render()
