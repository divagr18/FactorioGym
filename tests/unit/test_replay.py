"""Replay inspection (PLAN.md 5.5), checked without an engine or a provider."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import replay  # noqa: E402


def write_run(tmp_path: Path) -> Path:
    run = tmp_path / "llm-test"
    run.mkdir()
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "llm-test",
                "task": {"id": "deliver", "version": "1.3.0"},
                # As the adapter redacts it: the variable's name and whether one
                # was present, never the credential.
                "model": {
                    "model": "gpt-5.6-luna",
                    "api_key_env": "OPENAI_API_KEY",
                    "credential_present": True,
                },
            }
        ),
        encoding="utf-8",
    )
    decisions = [
        {
            "episode": 0,
            "step": 0,
            "action_index": 14,
            "action_key": "approach_entity_1",
            "target": "h1",
            "resolution": "model",
            "inference_ms": 2100,
            "prompt": "TASK deliver\nOBJECTIVE\n  dst at offset (+3.0, +4.0)",
            "observation": {
                "character": {"position": [0.0, 0.0]},
                "entities": [{"handle": "h1", "name": "wooden-chest", "p": [3.0, 4.0]}],
                "resources": {"tiles": []},
                "inventory": {"iron-plate": 20},
                "counters": {"transfers": 1},
                "goal": {"dst": [3.0, 4.0]},
                "tick": 90,
            },
            "legal_actions": [{"index": 14, "key": "approach_entity_1", "description": "walk"}],
            "attempts": [
                {"attempt": 1, "text": '{"action":14}', "latency_ms": 2100, "usage": {}},
            ],
            "result": {
                "reward": 0.1,
                "success": False,
                "terminated": False,
                "truncated": False,
                "action_status": "completed",
                "action_error": None,
                "skill": "approach_entity_1",
                "skill_outcome": "arrived",
                "skill_steps": 3,
                "skill_trace": [
                    {"key": "move_north", "status": "completed", "error": None},
                    {"key": "move_east", "status": "completed", "error": None},
                    {"key": "nudge_east", "status": "rejected", "error": "blocked"},
                ],
            },
        },
        {
            "episode": 0,
            "step": 1,
            "action_index": 12,
            "action_key": "wait",
            "resolution": "fallback",
            "inference_ms": 0,
            "prompt": "TASK deliver",
            "observation": {"character": {"position": [3.0, 4.0]}, "entities": []},
            "legal_actions": [],
            "attempts": [{"attempt": 1, "text": "nonsense", "failure": "unparseable"}],
            "result": {"reward": -0.001, "success": False, "action_status": "completed"},
        },
    ]
    (run / "decisions.jsonl").write_text(
        "\n".join(json.dumps(d) for d in decisions), encoding="utf-8"
    )
    return run


def test_the_page_reaches_no_network(tmp_path):
    """The acceptance clause is 'inspectable without contacting the model
    provider'. That is a property of a file containing no way to make a request,
    not a promise about how it is opened."""
    page = replay.build(write_run(tmp_path), None).read_text(encoding="utf-8")
    for pattern in (r"https?://", r"\bfetch\s*\(", r"XMLHttpRequest", r"<script\s+src", r"<link"):
        assert not re.search(pattern, page), f"page reaches the network via {pattern}"


def test_selecting_an_event_has_its_observation_and_the_prompt_as_sent(tmp_path):
    page = replay.build(write_run(tmp_path), None).read_text(encoding="utf-8")
    embedded = json.loads(re.search(r"const DECISIONS = (\[.*?\]);\n", page, re.S).group(1))
    assert len(embedded) == 2
    # The prompt actually sent, not one re-rendered later from the observation:
    # re-rendering with changed code answers a different question than the model
    # was asked.
    assert "OBJECTIVE" in embedded[0]["prompt"]
    assert embedded[0]["observation"]["goal"] == {"dst": [3.0, 4.0]}


def test_an_assisted_action_can_be_expanded(tmp_path):
    """One skill is one decision to the model and many to the engine. A step
    count cannot show that it walked out and back, which is the failure that
    cost a whole episode and had to be reconstructed by hand."""
    page = replay.build(write_run(tmp_path), None).read_text(encoding="utf-8")
    embedded = json.loads(re.search(r"const DECISIONS = (\[.*?\]);\n", page, re.S).group(1))
    trace = embedded[0]["result"]["skill_trace"]
    assert [t["key"] for t in trace] == ["move_north", "move_east", "nudge_east"]
    assert trace[-1]["error"] == "blocked"
    assert "assisted action" in page


def test_evaluator_information_is_labelled_and_kept_out_of_the_observation(tmp_path):
    page = replay.build(write_run(tmp_path), None).read_text(encoding="utf-8")
    assert "evaluator information" in page
    assert "not visible to the agent" in page
    # success and reward are the evaluator's verdict, so they must be rendered
    # inside the labelled block rather than beside the wire observation.
    block = page[page.index("evaluator information") :]
    assert 'class="ev"' in block
    for field in ("success: r.success", "reward: r.reward"):
        assert field in block, f"{field} is not inside the labelled evaluator block"
    # And the observation panel must not carry them: the agent never saw either.
    observation_panel = page[page.index("map view (observation only)") : page.index("evaluator")]
    assert "r.success" not in observation_panel


def test_an_intervention_is_not_rendered_as_a_model_decision(tmp_path):
    page = replay.build(write_run(tmp_path), None).read_text(encoding="utf-8")
    assert "intervention" in page


def test_a_training_run_is_refused_with_a_reason(tmp_path):
    run = tmp_path / "train-x"
    run.mkdir()
    (run / "curve.csv").write_text("timestep\n1\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="decisions.jsonl"):
        replay.build(run, None)


def test_no_credential_reaches_the_page(tmp_path):
    page = replay.build(write_run(tmp_path), None).read_text(encoding="utf-8")
    assert "OPENAI_API_KEY" not in page
    assert "credential_present" not in page
    assert "gpt-5.6-luna" in page


def test_capture_is_not_reachable_from_a_policy_action_space():
    """PLAN 5.6: 'screenshot capture is disabled in ordinary RL training'.

    Capture is a client-side facility rather than an action in the matrix, so a
    training run cannot request a frame even by accident and the action-mask
    purity tests keep meaning what they say. Asserted on the catalogs a policy
    can actually be given.
    """
    from factoriorl import catalog as catalog_module
    from factoriorl.tasks import all_tasks, get

    for task_id in all_tasks():
        spec = get(task_id).spec
        keys = catalog_module.resolve(spec.catalog, spec.catalog_subset).keys()
        assert not [k for k in keys if "screenshot" in k or "capture" in k], task_id

    matrix = (ROOT / "mod" / "factoriorl" / "matrix.lua").read_text(encoding="utf-8")
    assert "take_screenshot" not in matrix


def test_an_addressed_decision_is_distinguishable_from_a_default_one(tmp_path):
    """`target` is the difference between "act on that chest" and "act on
    whichever is nearest", and those produce different worlds. A replay that
    rendered them the same could not explain either."""
    page = replay.build(write_run(tmp_path), None).read_text(encoding="utf-8")
    embedded = json.loads(re.search(r"const DECISIONS = (\[.*?\]);\n", page, re.S).group(1))
    assert embedded[0]["target"] == "h1"
    assert embedded[1].get("target") is None
    # Named in the action panel, marked in the timeline, and ringed on the map.
    assert "(addressed)" in page
    assert "nearest entity (catalog default)" in page
    assert "t-addr" in page


def test_the_viewer_can_be_driven_without_a_mouse(tmp_path):
    """Stepping a hundred decisions by clicking is how a trace goes unread."""
    page = replay.build(write_run(tmp_path), None).read_text(encoding="utf-8")
    assert "keydown" in page
    for key in ("ArrowDown", "ArrowUp", "'j'", "'k'"):
        assert key in page


def test_the_timeline_can_be_narrowed(tmp_path):
    """A failure three hundred decisions in is only findable if the list can be
    reduced to failures."""
    page = replay.build(write_run(tmp_path), None).read_text(encoding="utf-8")
    for name in ("addressed", "assisted", "refused", "interventions", "solved"):
        assert name in page
    assert 'type="search"' in page


def test_the_map_draws_where_the_character_has_been(tmp_path):
    """A single frame cannot show pacing. The trail is what made a policy
    walking nine tiles out and twelve back legible as a loop."""
    page = replay.build(write_run(tmp_path), None).read_text(encoding="utf-8")
    assert "trail" in page
    # Built from earlier decisions in the same episode, never across episodes.
    assert "DECISIONS[i].episode === DECISIONS[index].episode" in page


# ------------------------------------------- 6.4: honesty about what ships


def test_limitations_document_covers_every_deferred_or_broken_thing():
    """PLAN 6.4: 'deferred functionality is not advertised as available'.

    A limitations file that drifts is worse than none, because a reader cannot
    tell which entries are still true. This pins the ones that are load-bearing
    for a release: each is a measured failure recorded elsewhere in the repo,
    and if one is fixed this test should fail and the entry be removed.
    """
    text = (ROOT / "docs" / "LIMITATIONS.md").read_text(encoding="utf-8")
    required = [
        # engine
        "2.0.60",
        "CaptureUnsupported",
        "absolute paths",
        # action catalog
        "cannot name what it acts on",
        "not reliably expressible in this catalog",
        # skills
        "approach_entity_0",
        "1.00",
        # rewards
        "pay a policy for stopping",
        "withdrawn",
        # methodology
        "does not reproduce a run's scenes",
        "does not decontaminate the generators",
        "property of the action space",
        # results
        "no accepted learning result",
    ]
    missing = [phrase for phrase in required if phrase not in text]
    assert not missing, f"limitations document no longer states: {missing}"


def test_the_readme_does_not_claim_an_accepted_learning_result():
    """The one claim a research preview must not make."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8").lower()
    for phrase in ("phase 4 accepted", "phases 0-4 accepted", "phases 0–4 accepted"):
        assert phrase not in readme, f"README claims {phrase!r}"
