"""Engine-free coverage for the model-adapter agent loop (DESIGN.md 5.3).

Every test here runs offline: no network, no API key, no Factorio. That is not a
concession -- it is the design. A scripted adapter lives in the package rather
than in this file precisely so the loop can be exercised in CI, and the two HTTP
adapters are tested by capturing the request they would have sent.

The criteria being checked are DESIGN 5.3's, verbatim:

* provider responses become validated typed actions;
* malformed outputs produce bounded retries or a recorded failure;
* credentials remain outside run artifacts;
* local and API models use the same observation and action contracts;
* inference latency and available usage data are recorded.
"""

from __future__ import annotations

import inspect
import json
import subprocess
import sys

import pytest

from factoriorl import catalog as catalog_module
from factoriorl import manifest as manifest_module
from factoriorl.agent import summary as summary_module
from factoriorl.agent.adapters import (
    AnthropicMessagesAdapter,
    ModelRequest,
    OpenAICompatibleAdapter,
    ScriptedAdapter,
)
from factoriorl.agent.loop import AgentConfig, AgentLoop
from factoriorl.agent.parsing import DecisionFailure, ParsedAction, ParseFailure, parse_action
from factoriorl.agent.summary import TaskBrief, action_vocabulary, legal_actions, summarise
from factoriorl.tasks import get

#: A value that exists only in the evaluator's truth dict. If it ever appears in
#: a prompt or an artifact, the policy/evaluator boundary has been crossed.
TRUTH_SENTINEL = "TRUTH-ONLY-MARKER-DO-NOT-LEAK"

#: A value that exists only in the process environment as a credential.
KEY_SENTINEL = "sk-test-DO-NOT-RECORD-0123456789"


@pytest.fixture(autouse=True)
def _light_host(monkeypatch):
    """Keep the unit suite off the multi-GB CUDA import path.

    ``manifest.host_info`` imports torch to describe the GPU, which costs about
    five seconds and several GB of commit per manifest written. That behaviour
    is already covered by ``test_manifest_and_isolation.py``; nothing in this
    file is about host detection, and paying for it on every artifact test would
    make the unit suite slow enough that people stop running it.
    """
    monkeypatch.setattr(
        manifest_module, "host_info", lambda: {"node": "test", "gpu": {"name": None}}
    )


def observation(entities: bool = True) -> dict:
    """One wire observation in the shape the mod emits."""
    return {
        "episode_id": "ep-1",
        "tick": 120,
        "absolute_tick": 5120,
        "profiles": {"observation": "local-v2", "action": "primitive-v1"},
        "character": {
            "present": True,
            "position": [4.0, -2.0],
            "walking": False,
            "direction": 0,
        },
        "inventory": {"iron-plate": 12, "coal": 3},
        "sensor": {"radius": 32, "origin": [4.0, -2.0]},
        "terrain": {"blocked": []},
        "resources": {
            "tiles": [
                {"name": "iron-ore", "p": [10.0, -2.0], "amount": 500, "h": "r1"},
                {"name": "iron-ore", "p": [11.0, -2.0], "amount": 480, "h": "r2"},
            ]
        },
        "entities": (
            [
                {
                    "h": "e1",
                    "name": "wooden-chest",
                    "type": "container",
                    "p": [7.0, -2.0],
                    "contents": {"iron-plate": 50},
                }
            ]
            if entities
            else []
        ),
        "remembered": [
            {
                "h": "e9",
                "name": "stone-furnace",
                "type": "furnace",
                "p": [-20.0, 6.0],
                "age": 420,
            }
        ],
        "task": {"transfers": 0, "items_moved": 0},
        "inflight": [],
        "events": [{"seq": 1, "action": "place", "status": "failed", "tick": 90}],
    }


class StubEnv:
    """The environment surface the loop uses, with no engine behind it.

    The mask is built the way ``FactorioEnv.action_masks`` builds it -- from the
    observation alone -- so a test that manipulates legality manipulates it the
    same way the real environment would.
    """

    def __init__(self, task_id: str = "deliver", *, entities: bool = True, horizon: int = 3):
        self.spec_ = get(task_id).spec
        self.catalog = catalog_module.resolve(self.spec_.catalog, self.spec_.catalog_subset)
        self._observation = observation(entities=entities)
        # Evaluator state. The loop is handed the environment, so this is
        # reachable in principle; the point of the boundary tests is that
        # nothing in the agent package reads it.
        self._truth = {"marker": TRUTH_SENTINEL, "distance_to_goal": 3.0}
        self.horizon = horizon
        self.steps = 0
        self.taken: list[int] = []

    def reset(self):
        self.steps = 0
        self.taken = []
        return None, {}

    def _context(self) -> dict:
        entities = self._observation.get("entities") or []
        tiles = (self._observation.get("resources") or {}).get("tiles") or []
        context = {
            "target": entities[0]["h"] if entities else None,
            "resource": tiles[0]["h"] if tiles else None,
        }
        for direction in catalog_module.PLACE_OFFSETS:
            context[f"at_{direction}"] = [0.5, 0.5]
        return context

    def action_masks(self):
        context = self._context()
        mask = [
            not template.requires or context.get(template.requires) is not None
            for template in self.catalog.templates
        ]
        mask[self.catalog.wait_index] = True
        return mask

    def step(self, index: int):
        self.taken.append(int(index))
        self.steps += 1
        terminated = self.steps >= self.horizon
        info = {
            "success": terminated,
            "action_status": "completed",
            "action_error": None,
            "action_key": self.catalog.keys()[int(index)],
        }
        return None, 1.0 if terminated else -0.001, terminated, False, info


def summary_for(env: StubEnv, step: int = 0):
    vocabulary = action_vocabulary(env)
    legal = legal_actions(vocabulary, env.action_masks())
    return summarise(
        env._observation, brief=TaskBrief.from_spec(env.spec_), actions=legal, step=step
    )


def loop_for(env: StubEnv, adapter, tmp_path, **config_kwargs) -> AgentLoop:
    config = AgentConfig(task_id=env.spec_.id, **config_kwargs)
    return AgentLoop(env, adapter, config, run_id="test-agent-run", run_dir=tmp_path / "run")


# ------------------------------------------------------------------ summary


def test_the_summary_offers_only_actions_the_environment_would_accept():
    """DESIGN 5.3: a model must never be offered an action the env would reject."""
    env = StubEnv(entities=False)
    rendered = summary_for(env).render()
    mask = env.action_masks()
    keys = env.catalog.keys()
    for index, legal in enumerate(mask):
        line = f"{index}: {keys[index]}"
        if legal:
            assert line in rendered, f"{keys[index]} is legal but was not offered"
        else:
            assert line not in rendered, f"{keys[index]} is masked out but was offered"
    # `deliver` addresses a chest through `$target`; with nothing in sensor range
    # those transfers are genuinely unavailable, so the test is not vacuous.
    assert not all(mask)


def test_the_summary_reads_the_observation_and_nothing_else():
    """The boundary is the signature: there is no parameter for task truth.

    `arguments` and `requires` were added for `parameterized-v1`. Neither is a
    hole in this guarantee: `requires` is the catalog's own declaration and
    `arguments` comes from `env.argument_domains()`, which reads the
    observation. The next test asserts that rather than asserting it here.
    """
    parameters = set(inspect.signature(summarise).parameters)
    assert parameters == {
        "observation",
        "brief",
        "actions",
        "step",
        "arguments",
        "requires",
        # The run's own clock and spending allowance. Not evaluator truth and
        # not task state: it is how much wall time and money the *controller*
        # will allow before it stops, which is the agent's own situation and
        # nothing about the world or the answer. The prompt previously asserted
        # a fixed decision interval while every open-world run was continuous
        # time, and offered "decision 503 of 100000" as planning guidance to a
        # run about to hit a thirty-minute wall.
        "clock",
        # The previous reply's per-action outcome. Recorded by the executor
        # since A2 and never shown back, so the model had to reconstruct its
        # own last move from a bounded engine event log that can omit most of
        # an eight-action batch.
        "receipt",
    }
    env = StubEnv()
    assert TRUTH_SENTINEL not in summary_for(env).render()


def test_the_argument_domains_are_a_function_of_the_observation():
    """The new summary fields must not become a truth channel."""
    import inspect as _inspect

    from factoriorl.agent.summary import argument_domains

    source = _inspect.getsource(argument_domains)
    assert "_truth" not in source
    assert "truth" not in source.replace("# ", "").split('"""')[2]

    from factoriorl.env import FactorioEnv

    domains = _inspect.getsource(FactorioEnv.argument_domains)
    assert "_truth" not in domains, (
        "argument_domains must read the observation, or the agent prompt "
        "would carry evaluator state"
    )


def test_the_brief_carries_the_declared_description_not_the_success_predicate():
    """Marker names are the vocabulary of the answer, as `test_skills` puts it.

    A brief built from success predicates would hand the model the marker the
    evaluator scores against, which is the language-model form of the leak the
    action-mask purity test exists to catch.
    """
    spec = get("navigate").spec
    brief = TaskBrief.from_spec(spec)
    markers = {p.marker for p in spec.success + spec.failure if p.marker}
    body = json.dumps(brief.to_dict())
    for marker in markers:
        assert marker not in body, f"the brief names the success marker {marker!r}"


def test_remembered_entities_are_marked_stale_with_their_age():
    """DESIGN section 2 forbids presenting distant machine state as current."""
    rendered = summary_for(StubEnv()).render()
    assert "REMEMBERED" in rendered
    assert "420 ticks ago" in rendered


def test_the_summary_states_positions_relative_to_the_character():
    rendered = summary_for(StubEnv()).render()
    # The chest sits three tiles east of the character at (4, -2).
    assert "wooden-chest [e1] at offset (+3.0, +0.0)" in rendered
    assert "3.0 tiles east" in rendered


def test_action_descriptions_come_from_the_catalog_not_a_lookup_table():
    """Every catalog entry describes itself, so a new action is never invisible."""
    catalog = catalog_module.resolve("primitive-v1")
    for template in catalog.templates:
        text = summary_module.describe_template(template)
        assert text and "$" not in text, f"{template.key} rendered a raw runtime reference"


# ------------------------------------------------------------------ parsing


def _vocabulary_and_legal(env: StubEnv):
    vocabulary = action_vocabulary(env)
    return vocabulary, legal_actions(vocabulary, env.action_masks())


def test_a_well_formed_response_becomes_a_typed_action():
    env = StubEnv()
    vocabulary, legal = _vocabulary_and_legal(env)
    parsed = parse_action('{"action": 0, "reason": "walk north"}', legal, vocabulary)
    assert isinstance(parsed, ParsedAction)
    assert parsed.index == 0
    assert parsed.key == vocabulary[0][0]
    assert parsed.reason == "walk north"


def test_an_action_may_be_named_as_well_as_indexed():
    env = StubEnv()
    vocabulary, legal = _vocabulary_and_legal(env)
    parsed = parse_action('{"action": "wait"}', legal, vocabulary)
    assert isinstance(parsed, ParsedAction)
    assert parsed.key == "wait"


def test_the_last_object_wins_when_a_model_restates_the_format():
    """A model that echoes the format example before answering must not have the
    example executed; taking the first object did exactly that."""
    env = StubEnv()
    vocabulary, legal = _vocabulary_and_legal(env)
    text = 'The format is {"action": <index>}. My answer: {"action": "wait"}'
    parsed = parse_action(text, legal, vocabulary)
    assert isinstance(parsed, ParsedAction)
    assert parsed.key == "wait"


def test_the_four_ways_a_response_fails_are_distinguishable():
    """Each names a different fix, so they must not collapse into one bucket."""
    env = StubEnv(entities=False)
    vocabulary, legal = _vocabulary_and_legal(env)
    masked = next(
        index for index, (key, _) in enumerate(vocabulary) if index not in {a.index for a in legal}
    )
    cases = {
        "I think we should walk east for a while.": DecisionFailure.UNPARSEABLE,
        '{"action": "teleport_to_goal"}': DecisionFailure.UNKNOWN_ACTION,
        '{"action": 9999}': DecisionFailure.OUT_OF_RANGE,
        json.dumps({"action": masked}): DecisionFailure.ILLEGAL_ACTION,
    }
    for text, expected in cases.items():
        outcome = parse_action(text, legal, vocabulary)
        assert isinstance(outcome, ParseFailure), text
        assert outcome.failure is expected, f"{text!r} was filed as {outcome.failure}"


def test_a_boolean_is_not_silently_executed_as_action_one():
    """`bool` is an `int` in Python; `{"action": true}` reaching the environment
    as action 1 would be a silent misexecution rather than a rejected answer."""
    env = StubEnv()
    vocabulary, legal = _vocabulary_and_legal(env)
    outcome = parse_action('{"action": true}', legal, vocabulary)
    assert isinstance(outcome, ParseFailure)
    assert outcome.failure is DecisionFailure.UNPARSEABLE


# --------------------------------------------------------------------- loop


def test_a_valid_response_is_executed_and_recorded(tmp_path):
    env = StubEnv(horizon=2)
    adapter = ScriptedAdapter(['{"action": "wait", "reason": "stall"}'] * 4, usage={"tokens": 7})
    loop = loop_for(env, adapter, tmp_path, episodes=1)
    result = loop.run()

    assert result["episodes"][0]["stopped"] == "terminated"
    assert env.taken == [env.catalog.wait_index] * 2
    assert result["decisions"] == 2
    assert result["fallback_decisions"] == 0
    assert result["usage_totals"] == {"tokens": 14}


def test_retries_are_bounded_and_every_attempt_is_recorded(tmp_path):
    """DESIGN 5.3: malformed output produces bounded retries *or* a recorded
    failure. Here the bound is reached, so both halves are visible: three
    attempts with three distinguishable reasons, and a decision that is recorded
    as a fallback rather than as something the model chose."""
    env = StubEnv(entities=False, horizon=1)
    adapter = ScriptedAdapter(
        ["nothing here", '{"action": "teleport"}', '{"action": 9999}'],
    )
    loop = loop_for(env, adapter, tmp_path, episodes=1, max_attempts=3)
    loop.run()

    decision = loop.decisions[0]
    assert adapter.calls == 3, "the retry bound is three attempts, not unbounded"
    assert decision.resolution == "fallback_wait"
    assert decision.action_index == env.catalog.wait_index
    reasons = [a.to_dict()["failure"] for a in decision.attempts]
    assert reasons == [
        DecisionFailure.UNPARSEABLE.value,
        DecisionFailure.UNKNOWN_ACTION.value,
        DecisionFailure.OUT_OF_RANGE.value,
    ]


def test_a_retry_tells_the_model_what_was_wrong(tmp_path):
    """A retry that re-sends the identical prompt is a repeat, not a retry."""
    env = StubEnv(horizon=1)
    adapter = ScriptedAdapter(["not an action", '{"action": "wait"}'])
    loop_for(env, adapter, tmp_path, episodes=1).run()
    assert "rejected" in adapter.requests[1].user
    assert DecisionFailure.UNPARSEABLE.value in adapter.requests[1].user


def test_a_failed_call_is_distinct_from_a_malformed_answer(tmp_path):
    """A provider that times out is a result, not the model getting it wrong."""
    env = StubEnv(horizon=1)
    adapter = ScriptedAdapter(
        [TimeoutError("read timed out"), '{"action": "wait", "reason": "ok"}']
    )
    loop = loop_for(env, adapter, tmp_path, episodes=1)
    result = loop.run()

    attempts = [a.to_dict() for a in loop.decisions[0].attempts]
    assert attempts[0]["failure"] == DecisionFailure.PROVIDER_ERROR.value
    assert attempts[0]["parsed"] is None
    assert attempts[1]["failure"] is None
    assert result["failures"] == {DecisionFailure.PROVIDER_ERROR.value: 1}


def test_latency_is_recorded_for_every_call_including_the_failed_one(tmp_path):
    """A provider that times out is the case anyone reads the record to find, so
    dropping its latency would misreport exactly the runs that matter."""
    env = StubEnv(horizon=1)
    adapter = ScriptedAdapter([TimeoutError("boom"), '{"action": "wait"}'])
    loop = loop_for(env, adapter, tmp_path, episodes=1)
    result = loop.run()

    attempts = loop.decisions[0].attempts
    assert len(attempts) == 2
    assert all(a.reply.latency_ms >= 0.0 for a in attempts)
    assert result["model_calls"] == 2
    assert result["latency_ms"]["total"] >= 0.0
    assert result["latency_ms"]["max"] >= result["latency_ms"]["p50"]


def test_a_run_stops_and_records_failure_when_the_model_never_answers(tmp_path):
    """The other half of 'bounded retries or a recorded failure': a broken
    provider must not produce a full-length episode of waiting that reads back
    as a played episode."""
    env = StubEnv(horizon=50)
    adapter = ScriptedAdapter(lambda request, index: "I refuse to pick an index.")
    loop = loop_for(env, adapter, tmp_path, episodes=1, max_attempts=2, max_consecutive_fallbacks=2)
    result = loop.run()

    assert result["episodes"][0]["stopped"] == "model_failure"
    assert result["episodes"][0]["steps"] == 2
    assert result["fallback_decisions"] == 2
    assert json.loads((tmp_path / "run" / "status.json").read_text())["state"] == "completed"


def test_the_loop_never_executes_an_action_the_mask_forbids(tmp_path):
    """A model insisting on a masked action must not have it silently applied."""
    env = StubEnv(entities=False, horizon=1)
    masked = next(index for index, legal in enumerate(env.action_masks()) if not legal)
    adapter = ScriptedAdapter([json.dumps({"action": masked})] * 3)
    loop = loop_for(env, adapter, tmp_path, episodes=1)
    loop.run()

    assert env.taken == [env.catalog.wait_index]
    assert loop.decisions[0].resolution == "fallback_wait"


# ----------------------------------------------------------- run artifacts


def test_a_run_writes_the_same_artifact_shape_as_a_training_run(tmp_path):
    """An agent run and a policy run must be inspectable the same way."""
    env = StubEnv(horizon=2)
    adapter = ScriptedAdapter(['{"action": "wait"}'] * 4, usage={"prompt_tokens": 3})
    loop = loop_for(env, adapter, tmp_path, episodes=1)
    loop.run()

    run_dir = tmp_path / "run"
    for name in ("manifest.json", "decisions.jsonl", "result.json", "status.json"):
        assert (run_dir / name).is_file(), f"a run must write {name}"

    written = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert written["task"]["id"] == env.spec_.id
    assert written["profiles"]["deliberation"] == "language-model-v1"
    assert written["profiles"]["catalog_digest"] == env.catalog.digest()
    assert written["model"]["adapter"] == "scripted"
    assert written["model"]["system_prompt_digest"], "the prompt is part of the run's identity"
    assert written["budgets"]["max_attempts_per_decision"] == loop.config.max_attempts

    lines = (run_dir / "decisions.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    # Renamed from `prompt`, which claimed to be "the prompt actually sent" and
    # was not: the request also carries the static prefix, the memory block and
    # any retry corrections. An external review read 235 of these looking for a
    # feature and could not tell whether it was missing or merely unlogged.
    assert first["observation_block"], "the observation half is part of the record"
    # And what was really sent lives beside it, appended before dispatch so a
    # request that never came back is still on disk.
    sent = (run_dir / "messages.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert sent, "the sent request stream is recorded"
    request = json.loads(sent[0])
    assert request["kind"] == "request"
    assert request["messages"][0]["role"] == "system"
    assert request["digest"], "a canonical digest, so a replay can prove it matches"
    assert first["attempts"][0]["text"], "the raw model response is part of the record"
    assert "latency_ms" in first["attempts"][0]
    assert first["result"]["action_status"] == "completed"
    assert first["legal_actions"], "what the model was offered is part of the record"


def test_the_run_artifact_carries_no_evaluator_state(tmp_path):
    """The loop holds the environment, so this is reachable in principle; the
    boundary only holds if nothing in the package actually reads it.

    The scan is asserted to have happened, not assumed. This check used to
    iterate `(tmp_path / "run").iterdir()` and nothing else, which passes
    silently in two ways a refactor could easily produce: an empty run
    directory, and artifacts moved one level down, since `iterdir` does not
    recurse. the project ledger makes the general point about a different check
    -- "a leakage check that never fires is not evidence of a clean reset" --
    and this is that check for the agent artifacts.
    """
    env = StubEnv(horizon=2)
    # First: the leak has to be reachable at all. If the stub stopped putting
    # the sentinel in truth, every assertion below would pass while testing
    # nothing.
    assert TRUTH_SENTINEL in str(env._truth), "the stub must carry the sentinel in truth"

    adapter = ScriptedAdapter(['{"action": "wait"}'] * 4)
    loop_for(env, adapter, tmp_path, episodes=1).run()

    run_dir = tmp_path / "run"
    scanned = [path for path in run_dir.rglob("*") if path.is_file()]
    assert scanned, f"nothing was scanned under {run_dir}, so this proves nothing"
    # The artifacts the boundary actually has to cover. Named, so that renaming
    # one is a test failure rather than a quietly narrower check.
    names = {path.name for path in scanned}
    for required in ("decisions.jsonl", "manifest.json", "result.json"):
        assert required in names, f"{required} missing from {sorted(names)}"
    for path in scanned:
        assert TRUTH_SENTINEL not in path.read_text(encoding="utf-8"), path.name


# ------------------------------------------------------------- credentials


class _EchoingEndpoint(OpenAICompatibleAdapter):
    """A hostile-but-plausible endpoint: one that echoes its own auth header.

    This is the failure the redaction gate exists for. A 401 body quoting the
    key, or a model repeating a prompt that contained it, would otherwise land
    verbatim in a run directory that gets shared as evidence.
    """

    def _post(self, url, headers, body):
        return (
            {
                "choices": [
                    {
                        "message": {
                            "content": f'{{"action": "wait"}} auth={headers["authorization"]}'
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
            None,
            "",
        )


def test_a_credential_is_read_from_the_environment_and_never_written(tmp_path, monkeypatch):
    """DESIGN 5.3: credentials remain outside run artifacts."""
    monkeypatch.setenv("FACTORIORL_TEST_MODEL_KEY", KEY_SENTINEL)
    adapter = _EchoingEndpoint(
        base_url="http://127.0.0.1:1/v1",
        model="echo",
        api_key_env="FACTORIORL_TEST_MODEL_KEY",
    )
    env = StubEnv(horizon=2)
    loop = loop_for(env, adapter, tmp_path, episodes=1)
    loop.run()

    run_dir = tmp_path / "run"
    for path in run_dir.iterdir():
        text = path.read_text(encoding="utf-8")
        assert KEY_SENTINEL not in text, f"the credential reached {path.name}"
    # The endpoint really did echo it, so the assertion above is not vacuous.
    assert "<redacted>" in (run_dir / "decisions.jsonl").read_text(encoding="utf-8")

    written = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert written["model"]["api_key_env"] == "FACTORIORL_TEST_MODEL_KEY"
    assert written["model"]["credential_present"] is True
    assert KEY_SENTINEL not in json.dumps(adapter.describe())


def test_an_unreachable_endpoint_is_a_recorded_failure_not_a_crash(tmp_path, monkeypatch):
    """Exercises the real HTTP path offline: nothing listens on port 1."""
    monkeypatch.setenv("FACTORIORL_TEST_MODEL_KEY", KEY_SENTINEL)
    adapter = OpenAICompatibleAdapter(
        base_url="http://127.0.0.1:1/v1",
        model="unreachable",
        api_key_env="FACTORIORL_TEST_MODEL_KEY",
        timeout=2.0,
    )
    env = StubEnv(horizon=5)
    loop = loop_for(env, adapter, tmp_path, episodes=1, max_attempts=1, max_consecutive_fallbacks=1)
    result = loop.run()

    assert result["failures"] == {DecisionFailure.PROVIDER_ERROR.value: 1}
    assert result["episodes"][0]["stopped"] == "model_failure"
    assert loop.decisions[0].attempts[0].reply.latency_ms >= 0.0
    for path in (tmp_path / "run").iterdir():
        assert KEY_SENTINEL not in path.read_text(encoding="utf-8"), path.name


def test_an_absent_credential_is_absent_rather_than_empty(monkeypatch):
    """An empty `Authorization` header reads as a wrong key, not a missing one."""
    monkeypatch.setenv("FACTORIORL_TEST_MODEL_KEY", "")
    adapter = OpenAICompatibleAdapter(api_key_env="FACTORIORL_TEST_MODEL_KEY")
    assert adapter.has_credential is False
    assert adapter.describe()["credential_present"] is False


# ------------------------------------------------ one contract, two providers


def _capture(adapter) -> dict:
    """Replace one adapter's transport and return what it would have sent."""
    seen: dict = {}

    def fake_post(url, headers, body):
        seen.update({"url": url, "headers": headers, "body": body})
        return (
            {
                "choices": [{"message": {"content": '{"action": 0}'}, "finish_reason": "stop"}],
                "content": [{"type": "text", "text": '{"action": 0}'}],
                "usage": {"input_tokens": 11, "output_tokens": 3, "prompt_tokens": 11},
                "stop_reason": "end_turn",
                "model": "captured",
            },
            None,
            "",
        )

    adapter._post = fake_post
    return seen


def test_a_local_endpoint_and_an_api_provider_send_the_same_observation():
    """DESIGN 5.3: local and API models use the same observation contract.

    Structural rather than aspirational: the loop builds one `ModelRequest` and
    both adapters carry its text through unchanged, so there is no route by
    which a provider could be handed a different view of the world.
    """
    request = ModelRequest(system="SYSTEM RULES", user=summary_for(StubEnv()).render())
    local = OpenAICompatibleAdapter(base_url="http://127.0.0.1:1/v1", model="local")
    api = AnthropicMessagesAdapter(model="claude-opus-5", api_key_env="FACTORIORL_UNSET_KEY")
    local_seen, api_seen = _capture(local), _capture(api)

    local_reply, api_reply = local.complete(request), api.complete(request)
    assert local_reply.ok and api_reply.ok
    assert local_reply.text == api_reply.text == '{"action": 0}'

    assert local_seen["body"]["messages"][1]["content"] == request.user
    assert api_seen["body"]["messages"][0]["content"] == request.user
    assert local_seen["body"]["messages"][0]["content"] == request.system
    assert api_seen["body"]["system"] == request.system


def test_the_api_adapter_omits_sampling_parameters():
    """Sampling parameters were removed on the current model family and a
    request carrying one is rejected with a 400, so forwarding the loop's
    determinism preference would fail every call."""
    api = AnthropicMessagesAdapter(api_key_env="FACTORIORL_UNSET_KEY")
    seen = _capture(api)
    api.complete(ModelRequest(system="s", user="u"))
    assert "temperature" not in seen["body"]
    assert "top_p" not in seen["body"]
    assert seen["headers"]["anthropic-version"] == AnthropicMessagesAdapter.API_VERSION


def test_usage_data_is_recorded_when_the_provider_returns_it():
    """DESIGN 5.3: available usage data is recorded."""
    api = AnthropicMessagesAdapter(api_key_env="FACTORIORL_UNSET_KEY")
    _capture(api)
    reply = api.complete(ModelRequest(system="s", user="u"))
    assert reply.usage["input_tokens"] == 11
    assert reply.usage["output_tokens"] == 3
    assert reply.latency_ms >= 0.0


def test_every_adapter_answers_the_same_call(tmp_path):
    """The interface is one method taking strings; an adapter never sees the
    environment, so it cannot acquire a private channel to the game."""
    signature = inspect.signature(ScriptedAdapter.complete)
    for cls in (OpenAICompatibleAdapter, AnthropicMessagesAdapter):
        assert inspect.signature(cls.complete) == signature
    assert set(inspect.signature(ModelRequest).parameters) == {
        "system",
        "user",
        "max_tokens",
        # An append-only history, for providers that price a cached prefix
        # differently from a fresh one. Still only strings.
        "messages",
    }


def test_a_request_carries_nothing_an_adapter_could_reach_the_game_through():
    """The property the field list above is a proxy for.

    Naming the fields pins the shape; this pins the *reason*. Every field is a
    string, an int, or a sequence of string-to-string mappings, so there is no
    object on a request through which an adapter could reach the environment,
    the catalog or the mask -- whatever fields get added later.
    """
    allowed = {"str", "int", "tuple[dict[str, str], ...]"}
    annotations = inspect.get_annotations(ModelRequest)
    assert annotations, "ModelRequest lost its annotations; this check is now vacuous"
    for name, annotation in annotations.items():
        rendered = (
            annotation
            if isinstance(annotation, str)
            else getattr(annotation, "__name__", str(annotation))
        )
        assert rendered in allowed, (
            f"ModelRequest.{name} is {rendered!r}, which is not a plain string, "
            f"int or mapping of strings. An adapter must not be handed anything "
            f"it could use to reach the game directly"
        )


# ------------------------------------------------------------- isolation


AGENT_IMPORT_PROBE = """
import sys
import factoriorl.agent  # noqa: F401
import factoriorl.agent.loop  # noqa: F401
import factoriorl.agent.adapters  # noqa: F401
banned = [m for m in ("torch", "stable_baselines3", "sb3_contrib") if m in sys.modules]
print(",".join(banned))
"""


def test_the_agent_loop_never_imports_the_training_stack():
    """The environment side of this repository stays free of the training stack
    so that evaluation cannot be changed by having torch installed, and an agent
    run is evaluation. The same probe the environment uses, applied here."""
    result = subprocess.run(
        [sys.executable, "-c", AGENT_IMPORT_PROBE],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if result.returncode != 0:
        if "paging file" in result.stderr or "Memory allocation" in result.stderr:
            pytest.skip(f"host under memory pressure: {result.stderr.strip()[:120]}")
        pytest.fail(result.stderr)
    assert not result.stdout.strip(), f"the agent path imported: {result.stdout.strip()}"


# ------------------------------- 5.3: a real local endpoint, same contracts


class _LocalServer:
    """A minimal OpenAI-compatible chat-completions endpoint on localhost.

    DESIGN 5.3 asks that "local and API models use the same observation and action
    contracts". Until this existed only the API path had ever run, so the claim
    rested on the adapters looking similar rather than on the local one having
    worked. This is a real HTTP server the real adapter really posts to, so the
    request shape it receives is the request shape a hosted provider receives.

    It is not a stand-in for a model: it echoes back a fixed choice. What is
    under test is the transport and the contract, not the reply's quality.
    """

    def __init__(self, reply: str):
        import http.server
        import json as _json
        import threading

        self.requests: list[dict] = []
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802 - http.server's interface
                length = int(self.headers.get("Content-Length") or 0)
                body = _json.loads(self.rfile.read(length) or b"{}")
                outer.requests.append(
                    {"path": self.path, "body": body, "headers": dict(self.headers)}
                )
                payload = _json.dumps(
                    {
                        "choices": [{"message": {"content": reply}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14},
                        "model": "local-model",
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *_args):
                return

        self._server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_exc):
        self._server.shutdown()
        self._server.server_close()


def test_a_local_endpoint_drives_the_loop_through_the_same_contracts(tmp_path):
    from factoriorl.agent.adapters import OpenAICompatibleAdapter

    with _LocalServer('{"action": 0, "reason": "walk north"}') as server:
        adapter = OpenAICompatibleAdapter(
            base_url=f"http://127.0.0.1:{server.port}/v1",
            model="local-model",
            api_key_env=None,
            temperature=0.0,
        )
        env = StubEnv()
        loop = loop_for(env, adapter, tmp_path, episodes=1, max_steps=2)
        result = loop.run()

    assert result["model_calls"] >= 1
    assert result["fallback_decisions"] == 0
    assert not result["failures"]

    sent = server.requests[0]
    assert sent["path"].endswith("/chat/completions")
    # The same message shape a hosted provider is sent: a system prompt that
    # states the action contract, then the rendered observation.
    messages = sent["body"]["messages"]
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "LEGAL ACTIONS" in messages[1]["content"]
    assert sent["body"]["model"] == "local-model"
    # No credential is invented for an endpoint that was configured without one.
    assert "Authorization" not in sent["headers"]

    # And latency and usage are recorded for a local endpoint exactly as they
    # are for an API one -- 5.3 requires it of both.
    assert result["latency_ms"]["mean"] >= 0
    assert result["usage_totals"]["total_tokens"] > 0


def test_a_local_endpoint_and_an_api_endpoint_send_the_same_prompt(tmp_path):
    """The contract claim is that only the transport differs.

    Both adapters are pointed at the same local server, so any difference in
    what the model is shown would appear here as a difference in the recorded
    request.
    """
    from factoriorl.agent.adapters import AnthropicMessagesAdapter, OpenAICompatibleAdapter

    prompts = {}
    for label, build in (
        (
            "local",
            lambda port: OpenAICompatibleAdapter(
                base_url=f"http://127.0.0.1:{port}/v1", model="local-model", api_key_env=None
            ),
        ),
        (
            "api",
            lambda port: OpenAICompatibleAdapter(
                base_url=f"http://127.0.0.1:{port}/v1",
                model="hosted-model",
                api_key_env=None,
                token_parameter="max_completion_tokens",
                temperature=None,
            ),
        ),
    ):
        with _LocalServer('{"action": 0, "reason": "walk north"}') as server:
            loop = loop_for(
                StubEnv(), build(server.port), tmp_path / label, episodes=1, max_steps=1
            )
            loop.run()
            prompts[label] = server.requests[0]["body"]["messages"]

    assert prompts["local"][0]["content"] == prompts["api"][0]["content"]
    assert prompts["local"][1]["content"] == prompts["api"][1]["content"]
    assert AnthropicMessagesAdapter is not None  # the third transport exists, unused here


# --------------------------------- 5.2: naming the entity an action acts on


def _addressing(env):
    from factoriorl.agent.summary import targetable_actions, visible_handles

    return targetable_actions(env), visible_handles(env._observation)


def test_only_actions_that_act_on_an_entity_are_targetable():
    """`$target` in a template is exactly what "acts on the nearest entity"
    means, so it is what decides whether naming one is meaningful."""
    from factoriorl.agent.summary import targetable_actions

    env = StubEnv()
    targetable = targetable_actions(env)
    assert targetable, "no catalog action binds a target"
    for key in targetable:
        assert key.startswith(("take", "give"))
    assert "move_north" not in targetable
    assert "wait" not in targetable


def test_a_named_target_is_carried_through_parsing():
    from factoriorl.agent.parsing import ParsedAction, parse_action

    env = StubEnv()
    vocabulary = _vocabulary_and_legal(env)[0]
    legal = _vocabulary_and_legal(env)[1]
    targetable, handles = _addressing(env)
    key = sorted(targetable)[0]
    index = [k for k, _ in vocabulary].index(key)
    handle = sorted(handles)[0]

    parsed = parse_action(
        f'{{"action": {index}, "target": "{handle}", "reason": "that one"}}',
        legal,
        vocabulary,
        targetable=targetable,
        handles=handles,
    )
    assert isinstance(parsed, ParsedAction), parsed
    assert parsed.target == handle
    assert parsed.to_dict()["target"] == handle


def test_a_target_on_an_action_that_acts_on_nothing_is_refused():
    """Dropping it silently would send the action to the nearest entity --
    exactly the behaviour the model was trying to override."""
    from factoriorl.agent.parsing import DecisionFailure, ParseFailure, parse_action

    env = StubEnv()
    vocabulary, legal = _vocabulary_and_legal(env)
    targetable, handles = _addressing(env)
    index = [k for k, _ in vocabulary].index("wait")

    outcome = parse_action(
        f'{{"action": {index}, "target": "{sorted(handles)[0]}"}}',
        legal,
        vocabulary,
        targetable=targetable,
        handles=handles,
    )
    assert isinstance(outcome, ParseFailure)
    assert outcome.failure is DecisionFailure.UNKNOWN_TARGET


def test_legacy_handle_target_is_normalized_to_the_canonical_arguments_shape():
    """A model trained on the previous prompt keeps its intended mine action."""
    from factoriorl.agent.parsing import ParsedAction, parse_action
    from factoriorl.agent.summary import LegalAction

    parsed = parse_action(
        '{"action": 0, "target": "h41"}',
        (LegalAction(0, "mine_at", "mine"),),
        (("mine_at", "mine"),),
        requires={"mine_at": ("handle",)},
    )
    assert isinstance(parsed, ParsedAction), parsed
    assert parsed.target is None
    assert parsed.arguments == {"handle": "h41"}


def test_a_target_the_agent_cannot_see_is_refused():
    from factoriorl.agent.parsing import DecisionFailure, ParseFailure, parse_action

    env = StubEnv()
    vocabulary, legal = _vocabulary_and_legal(env)
    targetable, handles = _addressing(env)
    key = sorted(targetable)[0]
    index = [k for k, _ in vocabulary].index(key)

    outcome = parse_action(
        f'{{"action": {index}, "target": "h999"}}',
        legal,
        vocabulary,
        targetable=targetable,
        handles=handles,
    )
    assert isinstance(outcome, ParseFailure)
    assert outcome.failure is DecisionFailure.UNKNOWN_TARGET
    assert "h999" in outcome.detail


def test_addressing_adds_no_verb_the_catalog_did_not_already_have():
    """Rebinding the catalog's own template is what keeps this from becoming a
    new capability: the verb, item and count are the discrete action's, and only
    `$target` changes."""
    env = StubEnv()
    template = env.catalog.templates[
        [t.key for t in env.catalog.templates].index(sorted(_addressing(env)[0])[0])
    ]
    default = template.bind({"target": "h1"})
    addressed = template.bind({"target": "h2"})
    assert default["action"] == addressed["action"]
    assert set(default) == set(addressed)
    differing = {k for k in default if default[k] != addressed[k]}
    assert differing <= {"from", "to"}, differing


# ------------------------------------------------- prefix caching (roadmap A0)


def _bodies(adapter) -> list[list[dict]]:
    """The message list actually sent on each call."""
    return [[dict(m) for m in request.messages] for request in adapter.requests]


def test_each_request_body_is_a_prefix_of_the_next(tmp_path):
    """The invariant a prefix-caching provider bills against.

    DeepSeek caches on an exact prefix and charges 50x less for the part that
    matches, so over a long run this single property is the difference between
    about $1.19 and about $34. It is asserted structurally rather than hoped for.
    """
    env = StubEnv(horizon=4)
    adapter = ScriptedAdapter([json.dumps({"action": "wait", "reason": "hold"})] * 4)
    loop_for(env, adapter, tmp_path, episodes=1).run()

    bodies = _bodies(adapter)
    assert len(bodies) >= 3, "need several calls for this to mean anything"
    for earlier, later in zip(bodies, bodies[1:], strict=False):
        assert later[: len(earlier)] == earlier, (
            "a request was not a prefix of the next: something rewrote history, "
            "and every token after the edit reverts to the cache-miss rate"
        )
        assert len(later) > len(earlier), "the history did not grow"


def test_a_retry_appends_instead_of_rewriting_the_turn_already_sent(tmp_path):
    """The loop used to splice the correction into the same user string. That is
    a prefix mutation: it invalidates the cache for the whole conversation."""
    env = StubEnv(horizon=1)
    adapter = ScriptedAdapter(["not an action", json.dumps({"action": "wait"})])
    loop_for(env, adapter, tmp_path, episodes=1).run()

    first, second = _bodies(adapter)
    assert second[: len(first)] == first
    # The rejected answer, then the reason -- both appended.
    assert [m["role"] for m in second[len(first) :]] == ["assistant", "user"]
    assert second[len(first)]["content"] == "not an action"
    assert "rejected" in second[-1]["content"]


def test_a_transport_failure_resends_a_byte_identical_body(tmp_path):
    """The call never reached the model, so there is no answer to record and no
    correction to make. Resending the identical body is a complete cache hit."""
    env = StubEnv(horizon=1)
    adapter = ScriptedAdapter(
        [TimeoutError("read timed out"), json.dumps({"action": "wait", "reason": "ok"})]
    )
    loop_for(env, adapter, tmp_path, episodes=1).run()

    first, second = _bodies(adapter)
    assert first == second, "a failed call must not change what is sent next"


def test_a_new_episode_starts_a_new_transcript(tmp_path):
    """Contamination beats cost here.

    `docs/LIMITATIONS.md` is explicit that memory is per episode, because
    carrying it over would put one evaluation scene's contents into the next
    scene's prompt. The transcript is the same argument with the same answer,
    even though resetting it throws away the cached prefix.
    """
    env = StubEnv(horizon=1)
    adapter = ScriptedAdapter([json.dumps({"action": "wait", "reason": "ok"})] * 2)
    loop = loop_for(env, adapter, tmp_path, episodes=2)
    loop.run()

    bodies = _bodies(adapter)
    assert len(bodies) == 2
    # Each episode's first request is system + one user turn, and nothing more.
    assert len(bodies[0]) == len(bodies[1]) == 2
    assert loop.transcript.to_dict()["turns"] == 2, "the second episode's own turns"


def test_no_reasoning_content_is_ever_sent_back(tmp_path):
    """Without `tools`, DeepSeek ignores a returned `reasoning_content` and does
    not concatenate it. Sending one anyway would inflate the context for nothing;
    the transcript has no field for it, so this is structural."""
    env = StubEnv(horizon=2)
    adapter = ScriptedAdapter([json.dumps({"action": "wait", "reason": "ok"})] * 2)
    loop_for(env, adapter, tmp_path, episodes=1).run()

    for body in _bodies(adapter):
        for message in body:
            assert set(message) == {"role", "content"}, message
