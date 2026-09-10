"""A running action's identity has to reach the prompt (roadmap A2.3).

A2.3 requires that "an action still running is reported with its operation
identity and can be inspected/cancelled". Both halves were half-built:
`env.argument_domains` published a `requests` domain built from the in-flight
entries, and `cancel_request` declared `target_request_id` as drawing from it --
but the renderer printed neither. The in-flight line said
`in flight: navigate (progress 0.31)` and the argument-values block skipped
`requests` entirely, so the verb was permanently legal and permanently
unusable: nothing in the prompt carried a value it would accept.

That is the eighth instance of this repository's recurring defect -- a field
computed, transmitted and read by nothing -- so it gets a test rather than a
fix alone. Proved against the engine in `docs/evidence/a2-bootstrap.json`,
where a 28-tile walk is cancelled by an id read out of this render.
"""

from __future__ import annotations

from factoriorl.agent.summary import TaskBrief, summarise

BRIEF = TaskBrief(
    id="open_factory", version="0.2.0", description="build a factory", max_decision_steps=400
)


def _render(inflight: list[dict], arguments: dict | None = None) -> str:
    observation = {
        "tick": 3600,
        "character": {"present": True, "position": [0.0, 0.0]},
        "inventory": {},
        "entities": [],
        "inflight": inflight,
    }
    return summarise(
        observation, brief=BRIEF, actions=(), step=0, arguments=arguments or {}
    ).render()


def test_a_running_action_is_rendered_with_its_request_id() -> None:
    text = _render([{"action": "navigate", "progress": 0.31, "request_id": "step-7:act"}])
    assert "in flight: navigate" in text
    # The identity, not merely the verb: `cancel_request` takes the former.
    assert "step-7:act" in text


def test_a_running_action_without_an_id_still_renders() -> None:
    """The mod is the only source of the id, so its absence must not crash."""
    text = _render([{"action": "mine", "progress": 0.5}])
    assert "in flight: mine" in text
    assert "[]" not in text


def test_the_requests_domain_is_offered_as_an_argument_value() -> None:
    text = _render(
        [{"action": "navigate", "progress": 0.1, "request_id": "step-7:act"}],
        arguments={"requests": ["step-7:act"]},
    )
    assert "target_request_id: step-7:act" in text


def test_nothing_running_offers_no_request_to_cancel() -> None:
    text = _render([], arguments={"requests": []})
    assert "target_request_id" not in text
