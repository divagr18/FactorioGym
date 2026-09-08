"""The agent path over `parameterized-v1` (R3.2).

`build_line` is the first task whose catalog carries arguments, and the agent
loop was written for a discrete one. Two defects showed up on the first real
run and both are pinned here, because both produced a *plausible-looking*
failure rather than an error:

* `_OBJECT = re.compile(r"\\{[^{}]*\\}")` cannot match a nested object, so once
  `arguments` became an object the regex matched only the inner `{...}` -- which
  has no `"action"` key. 37 well-formed replies in a row were recorded as
  `unparseable`, which reads as "the model cannot follow the format".
* The prompt said "positions are given as offsets from the character" while the
  placement domain is absolute world coordinates, so the model supplied offsets
  and every placement was refused as out of domain.
"""

from __future__ import annotations

import pytest

from factoriorl.agent.loop import FORBIDDEN_IN_PROMPT, SYSTEM_PROMPT
from factoriorl.agent.parsing import (
    DecisionFailure,
    ParsedAction,
    ParseFailure,
    _extract,
    _json_objects,
    parse_action,
)
from factoriorl.agent.summary import LegalAction

VOCAB = (
    ("place_at", "place an item"),
    ("give_to", "move items"),
    ("wait", "do nothing"),
)
LEGAL = tuple(LegalAction(index=i, key=k, description=d) for i, (k, d) in enumerate(VOCAB))
REQUIRES = {
    "place_at": ("item", "position", "direction"),
    "give_to": ("to", "item", "count"),
}
DOMAINS = {
    "placements": [[0.5, 0.5], [1.5, 0.5], [-2.5, 3.5]],
    "directions": ["north", "east", "south", "west"],
    "items": ["burner-mining-drill", "coal"],
    "amounts": [1, 5, 20],
    "targets": ["h1", "r7"],
}


def _parse(text: str):
    return parse_action(
        text, LEGAL, VOCAB, requires=REQUIRES, domains=DOMAINS, handles=frozenset({"h1"})
    )


class TestNestedObjectsExtract:
    """The defect that made 37 correct replies read as unparseable."""

    REAL_REPLY = (
        '{"action":0,"arguments":{"item":"burner-mining-drill",'
        '"position":[0.5,0.5],"direction":"north"},"reason":"place it on the ore"}'
    )

    def test_a_nested_reply_yields_one_span(self):
        assert len(_json_objects(self.REAL_REPLY)) == 1

    def test_the_action_and_the_arguments_both_come_out(self):
        extracted = _extract(self.REAL_REPLY)
        assert extracted is not None
        action, _reason, _target, arguments = extracted
        assert action == 0
        assert arguments["position"] == [0.5, 0.5]

    def test_it_parses_into_an_action(self):
        outcome = _parse(self.REAL_REPLY)
        assert isinstance(outcome, ParsedAction), getattr(outcome, "detail", outcome)
        assert outcome.arguments["item"] == "burner-mining-drill"
        assert outcome.arguments["direction"] == "north"

    def test_a_brace_inside_a_string_does_not_split_the_span(self):
        assert _json_objects('{"reason": "}{ not a brace"}') == ['{"reason": "}{ not a brace"}']

    def test_an_escaped_quote_does_not_end_the_string(self):
        text = '{"action": 2, "reason": "he said \\"go\\" then {"}'
        assert len(_json_objects(text)) == 1

    def test_prose_around_the_object_is_ignored(self):
        outcome = _parse(f"Here is my choice:\n{self.REAL_REPLY}\nThanks.")
        assert isinstance(outcome, ParsedAction)

    def test_the_last_object_wins_when_a_model_reconsiders(self):
        first = '{"action": 2, "reason": "wait"}'
        outcome = _parse(f"{first}\nActually:\n{self.REAL_REPLY}")
        assert isinstance(outcome, ParsedAction) and outcome.index == 0

    def test_a_flat_reply_still_parses(self):
        outcome = _parse('{"action": 2, "reason": "nothing to do"}')
        assert isinstance(outcome, ParsedAction) and outcome.key == "wait"


class TestArgumentsAreValidatedAgainstTheObservedDomain:
    def test_a_missing_argument_is_named_not_guessed(self):
        outcome = _parse('{"action": 0, "arguments": {"item": "coal"}, "reason": "x"}')
        assert isinstance(outcome, ParseFailure)
        assert outcome.failure is DecisionFailure.MISSING_ARGUMENT
        assert "position" in outcome.detail

    def test_a_value_outside_the_domain_is_named(self):
        """The offset-versus-absolute defect, as a test: a plausible position
        that is simply not in the domain."""
        outcome = _parse(
            '{"action": 0, "arguments": {"item": "coal", "position": [-99.5, 4.5],'
            ' "direction": "north"}, "reason": "x"}'
        )
        assert isinstance(outcome, ParseFailure)
        assert outcome.failure is DecisionFailure.BAD_ARGUMENT
        assert "placements" in outcome.detail

    def test_the_failure_shows_examples_so_a_retry_can_be_corrected(self):
        outcome = _parse(
            '{"action": 0, "arguments": {"item": "coal", "position": [-99.5, 4.5],'
            ' "direction": "north"}, "reason": "x"}'
        )
        assert "0.5" in outcome.detail

    def test_an_item_the_agent_does_not_hold_is_refused(self):
        outcome = _parse(
            '{"action": 0, "arguments": {"item": "steel-plate", "position": [0.5, 0.5],'
            ' "direction": "north"}, "reason": "x"}'
        )
        assert isinstance(outcome, ParseFailure)
        assert outcome.failure is DecisionFailure.BAD_ARGUMENT

    def test_an_extra_argument_is_refused_rather_than_dropped(self):
        outcome = _parse('{"action": 2, "arguments": {"item": "coal"}, "reason": "waiting"}')
        assert isinstance(outcome, ParseFailure)
        assert outcome.failure is DecisionFailure.BAD_ARGUMENT

    def test_integer_positions_are_accepted_as_floats(self):
        outcome = _parse(
            '{"action": 0, "arguments": {"item": "coal", "position": [1.5, 0.5],'
            ' "direction": "east"}, "reason": "x"}'
        )
        assert isinstance(outcome, ParsedAction)
        assert outcome.arguments["position"] == [1.5, 0.5]

    def test_an_action_needing_no_arguments_takes_none(self):
        outcome = _parse('{"action": 2, "reason": "wait"}')
        assert isinstance(outcome, ParsedAction) and outcome.arguments == {}

    @pytest.mark.parametrize("count", [1, 5, 20])
    def test_a_legal_transfer_amount_is_accepted(self, count):
        outcome = _parse(
            '{"action": 1, "arguments": {"to": "h1", "item": "coal", "count": '
            f"{count}"
            '}, "reason": "fuel it"}'
        )
        assert isinstance(outcome, ParsedAction), getattr(outcome, "detail", outcome)

    def test_an_illegal_transfer_amount_is_refused(self):
        outcome = _parse(
            '{"action": 1, "arguments": {"to": "h1", "item": "coal", "count": 7},'
            ' "reason": "fuel it"}'
        )
        assert isinstance(outcome, ParseFailure)
        assert outcome.failure is DecisionFailure.BAD_ARGUMENT


class TestThePromptIsUnambiguousAndWithholdsTheGeometry:
    def test_it_says_argument_positions_are_absolute(self):
        """The prompt previously said only "positions are offsets", which is
        true of the entity list and false of the placement domain."""
        assert "ABSOLUTE world coordinates" in SYSTEM_PROMPT
        assert "offsets from the character" in SYSTEM_PROMPT

    def test_it_explains_the_arguments_object(self):
        assert '"arguments"' in SYSTEM_PROMPT
        assert "[needs: ...]" in SYSTEM_PROMPT

    @pytest.mark.parametrize("phrase", FORBIDDEN_IN_PROMPT)
    def test_it_withholds_the_reference_geometry(self, phrase):
        """R3.2: the reference build sequence must not be embedded in runtime
        assistance. Which furnace centre catches a drill's drop is what the
        reference had to measure on an engine."""
        assert phrase.lower() not in SYSTEM_PROMPT.lower()

    def test_the_forbidden_list_is_not_empty(self):
        """Otherwise the check above passes by having nothing to check."""
        assert len(FORBIDDEN_IN_PROMPT) >= 5


class TestTheLoopIssuesParameterizedActionsThroughTheSharedEntryPoint:
    def test_execute_uses_step_arguments_for_a_parameterized_template(self):
        from factoriorl import catalog as catalog_module
        from factoriorl.agent.loop import AgentConfig, AgentLoop, Decision

        class _Catalog:
            templates = catalog_module.resolve("parameterized-v1").templates
            wait_index = catalog_module.resolve("parameterized-v1").wait_index

        class _Env:
            catalog = _Catalog()
            spec_ = None
            called: list = []

            def step_arguments(self, index, arguments):
                self.called.append(("step_arguments", index, dict(arguments)))
                return None, 0.0, False, False, {}

            def step(self, index):
                self.called.append(("step", index))
                return None, 0.0, False, False, {}

        from factoriorl.tasks import get

        env = _Env()
        env.spec_ = get("build_line").spec
        loop = AgentLoop.__new__(AgentLoop)
        loop.env = env
        loop.config = AgentConfig(task_id="build_line")
        index = [t.key for t in env.catalog.templates].index("place_at")
        decision = Decision(
            episode=0,
            step=0,
            summary=None,
            attempts=[],
            action_index=index,
            action_key="place_at",
            resolution="model",
            arguments={"item": "coal", "position": [0.5, 0.5], "direction": "north"},
        )
        loop._execute(decision)
        assert env.called == [
            (
                "step_arguments",
                index,
                {"item": "coal", "position": [0.5, 0.5], "direction": "north"},
            )
        ], env.called

    def test_a_discrete_action_still_goes_through_step(self):
        from factoriorl import catalog as catalog_module
        from factoriorl.agent.loop import AgentConfig, AgentLoop, Decision
        from factoriorl.tasks import get

        class _Env:
            catalog = catalog_module.resolve("parameterized-v1")
            spec_ = get("build_line").spec
            called: list = []

            def step(self, index):
                self.called.append(("step", index))
                return None, 0.0, False, False, {}

        env = _Env()
        loop = AgentLoop.__new__(AgentLoop)
        loop.env = env
        loop.config = AgentConfig(task_id="build_line")
        index = env.catalog.keys().index("wait")
        loop._execute(
            Decision(
                episode=0,
                step=0,
                summary=None,
                attempts=[],
                action_index=index,
                action_key="wait",
                resolution="model",
            )
        )
        assert env.called == [("step", index)]
