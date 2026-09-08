"""Persistent agent memory (PLAN.md 5.4)."""

from __future__ import annotations

import pytest

from factoriorl.agent.memory import (
    EVALUATOR_FIELDS,
    SOURCE_ACTION,
    SOURCE_OBSERVATION,
    Fact,
    Memory,
)


def observation(step: int, handles: list[str], contents: dict | None = None) -> dict:
    return {
        "entities": [
            {
                "h": handle,
                "name": "wooden-chest",
                "p": [float(index), 0.0],
                "contents": (contents or {}).get(handle) or {},
            }
            for index, handle in enumerate(handles)
        ],
        "character": {"position": [0.0, 0.0]},
        "step": step,
    }


def test_every_fact_cites_a_source_the_agent_actually_received():
    """A memory entry with no provenance is indistinguishable from one the
    agent invented, so it cannot be constructed."""
    Fact(subject="h1", statement="x", source=SOURCE_OBSERVATION, step=0)
    Fact(subject="h1", statement="x", source=SOURCE_ACTION, step=0)
    with pytest.raises(ValueError, match="must cite"):
        Fact(subject="h1", statement="x", source="inferred", step=0)


def test_a_fact_the_observation_stopped_confirming_reads_as_stale():
    """An entity the sensor cannot see is unobserved, not gone. Rendering the
    two the same way invites the model to act on a fact that may have expired."""
    memory = Memory()
    memory.observe(0, observation(0, ["h1", "h2"]))
    memory.observe(3, observation(3, ["h1"]))

    assert not memory.entities["h1"].stale_at(3)
    assert memory.entities["h2"].stale_at(3)
    rendered = memory.render()
    assert "REMEMBERED (not in the current observation)" in rendered
    assert "h2" in rendered
    assert "not confirmed since" in rendered
    # The one still visible is not re-listed as remembered.
    assert rendered.count("[h1]") == 0


def test_a_tried_target_is_remembered_so_the_probe_loop_can_end():
    """The measured failure this exists for: seventeen takes alternating with
    sixteen gives on one container, because nothing recorded that it had
    already been tried."""
    memory = Memory()
    memory.observe(0, observation(0, ["h1", "h2", "h3"]))
    memory.record_action(1, "give_iron-plate_20", target="h3", status="completed")
    memory.record_action(2, "take_iron-plate_20", target="h3", status="completed")

    assert memory.ruled_out() == ["h3"]
    rendered = memory.render()
    assert "ALREADY TRIED" in rendered
    assert "h3" in rendered


def test_a_failed_plan_survives_compaction():
    """PLAN 5.4: 'a failed plan does not disappear from the record'. It is the
    only thing that stops the agent proposing it again."""
    memory = Memory()
    for step in range(20):
        memory.record_action(step, "move_north", status="completed")
    memory.record_action(20, "give_iron-plate_20", target="h9", error="out_of_reach")
    memory.record_plan(1, "deliver to h9")
    memory.close_plan(21, "h9 is out of reach")

    memory.compact(keep=3)

    assert any(a.error == "out_of_reach" for a in memory.attempts), "compaction dropped the failure"
    assert len([a for a in memory.attempts if not a.failed]) == 3
    rendered = memory.render()
    assert "REFUSED ACTIONS" in rendered
    assert "PLANS THAT DID NOT WORK" in rendered
    assert "h9 is out of reach" in rendered


def test_memory_never_ingests_the_evaluator_s_verdict():
    """The rendered memory goes into the next prompt, so anything memory can
    read, the model can see. `record_action` therefore takes the action's
    status and error and has no parameter for reward or success -- the clause
    is enforced by the signature rather than by discipline."""
    import inspect

    parameters = set(inspect.signature(Memory.record_action).parameters)
    assert not (parameters & EVALUATOR_FIELDS), parameters & EVALUATOR_FIELDS

    memory = Memory()
    memory.observe(0, observation(0, ["h1"]))
    memory.record_action(1, "give_iron-plate_20", target="h1", status="completed")
    memory.record_action(2, "give_iron-plate_20", target="h1", error="no_items")
    memory.compact()

    rendered = memory.render().lower()
    for field in EVALUATOR_FIELDS:
        assert field not in rendered, f"{field} reached the prompt through memory"
    serialised = str(memory.to_dict()).lower()
    for field in EVALUATOR_FIELDS:
        assert field not in serialised


def test_an_empty_memory_renders_nothing():
    """A block of headings with no content is prompt tokens spent on nothing."""
    assert Memory().render() == ""


# ------------------------------------------- memory as the loop uses it


def test_the_loop_resolves_which_entity_a_transfer_acted_on():
    """The catalog binds `$target` to the nearest entity and the agent never
    names it, so without this "already tried" degrades to "already did
    something" and cannot end a probe loop."""
    from factoriorl.agent.loop import _target_of

    scene = {
        "character": {"position": [0.0, 0.0]},
        "entities": [
            {"h": "far", "p": [10.0, 0.0]},
            {"h": "near", "p": [2.0, 0.0]},
        ],
    }
    assert _target_of("give_iron-plate_20", scene) == "near"
    assert _target_of("take_iron-plate_5", scene) == "near"
    # Movement acts on nothing, so recording a target for it would be a claim
    # the observation does not support.
    assert _target_of("move_north", scene) is None
    assert _target_of("approach_entity_1", scene) is None


def test_memory_does_not_survive_an_episode_boundary(tmp_path):
    """Carrying a record across episodes would put one evaluation scene's
    containers into the next scene's prompt. That is contamination, not
    competence, and it would flatter every multi-episode agent result."""
    import inspect

    from factoriorl.agent.loop import AgentLoop

    source = (
        inspect.getsource(AgentLoop.run) + inspect.getsource(AgentLoop._play_episode)
        if hasattr(AgentLoop, "_play_episode")
        else inspect.getsource(AgentLoop)
    )
    assert "self.memory = Memory()" in source
