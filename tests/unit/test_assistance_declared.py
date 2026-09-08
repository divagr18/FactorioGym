"""An assistance the environment gives must appear in the record.

`env._focus_target` picks the nearest *unrepaired* declared fault and hands it
to the policy as three floats, which resolves the fault-selection sub-problem
of the only two multi-fault families. Both manifest writers previously recorded
`assistance: "none"`, and `task.resolved.focus_marker` reported a static marker
the code overrode -- so a run with the selector read identically to one without.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from factoriorl import encoders
from factoriorl.assistance import STATIC, describe_assistance
from factoriorl.env import FactorioEnv
from factoriorl.tasks import all_tasks, get

SELECTING = "nearest_unrepaired"


class _Env(FactorioEnv):
    """Just the goal-geometry half; no engine, no session."""

    def __init__(self, spec, observation):
        self.spec_ = spec
        self._observation = observation
        self._steps = 0
        self._truth = {}


def _observation():
    return {
        "character": {"position": [0.0, 0.0]},
        # gap is filled, gap2 is open.
        "entities": [{"p": [4.5, 0.5], "h": "e1", "type": "transport-belt"}],
        "goal": {"gap": [4.5, 0.5], "gap2": [8.5, 0.5]},
    }


class TestDescribeAssistance:
    def test_a_selecting_policy_is_named(self):
        spec = replace(get("repair_belt").spec, focus_policy=SELECTING)
        assert describe_assistance(spec) == f"goal-focus:{SELECTING}"

    def test_a_static_policy_reports_none(self):
        spec = replace(get("repair_belt").spec, focus_policy="static")
        assert describe_assistance(spec) == STATIC

    def test_assistances_compose_rather_than_replace(self):
        """A second assistance must not silently overwrite the first."""
        spec = replace(get("repair_belt").spec, focus_policy=SELECTING)
        described = describe_assistance(spec, extra=("navigation",))
        assert "goal-focus" in described and "navigation" in described


class TestFocusPolicyIsDeclared:
    def test_the_two_multi_fault_families_declare_the_selector(self):
        for task_id in ("repair_belt", "restore_power"):
            spec = get(task_id).spec
            assert spec.focus_policy == SELECTING, task_id
            assert len(spec.extra_public_markers) > 1, task_id

    def test_every_other_family_is_static(self):
        for task_id in sorted(all_tasks()):
            spec = get(task_id).spec
            if task_id in {"repair_belt", "restore_power"}:
                continue
            assert spec.focus_policy == "static", task_id

    def test_it_enters_the_config_digest(self):
        """Two runs differing only in the assistance are different experiments."""
        from factoriorl.manifest import config_digest

        spec = get("repair_belt").spec
        selecting = config_digest(spec.to_dict())
        static = config_digest(replace(spec, focus_policy="static").to_dict())
        assert selecting != static

    def test_a_selecting_family_declares_more_than_one_fault(self):
        """Selecting among one marker would be theatre."""
        for task_id in sorted(all_tasks()):
            spec = get(task_id).spec
            if spec.focus_policy == SELECTING:
                assert len(spec.extra_public_markers) > 1, task_id


class TestTheCodeHonoursTheDeclaration:
    def test_a_selecting_policy_skips_the_filled_fault(self):
        spec = replace(get("repair_belt").spec, focus_policy=SELECTING)
        env = _Env(spec, _observation())
        target = env._focus_target(_observation()["goal"], [0.0, 0.0])
        assert target == [8.5, 0.5], "should point at the open fault, not the filled one"

    def test_a_static_policy_always_points_at_the_focus_marker(self):
        spec = replace(get("repair_belt").spec, focus_policy="static")
        env = _Env(spec, _observation())
        target = env._focus_target(_observation()["goal"], [0.0, 0.0])
        assert target == [4.5, 0.5], "static must not retarget, even to an open fault"

    def test_the_two_policies_disagree_on_the_same_scene(self):
        """If they agreed, declaring the difference would be pointless."""
        goal = _observation()["goal"]
        selecting = _Env(
            replace(get("repair_belt").spec, focus_policy=SELECTING), _observation()
        )._focus_target(goal, [0.0, 0.0])
        static = _Env(
            replace(get("repair_belt").spec, focus_policy="static"), _observation()
        )._focus_target(goal, [0.0, 0.0])
        assert selecting != static


def test_the_goal_encoding_version_was_bumped_for_the_behaviour_change():
    """Version 3 documented "the nearest published marker", which is not what
    the code does on a selecting task. Leaving it at 3 meant a v3 checkpoint and
    a v3 run could mean different things in the same three slots."""
    assert encoders.GOAL_ENCODING_VERSION >= 4


def test_both_clients_pin_their_own_encoding():
    """The LLM client had no encoding version while the RL client had two."""
    from factoriorl.agent.summary import SUMMARY_ENCODING_VERSION
    from factoriorl.learn.policy import EXTRACTOR_VERSION

    assert SUMMARY_ENCODING_VERSION >= 1
    assert EXTRACTOR_VERSION >= 4
    assert encoders.GOAL_ENCODING_VERSION >= 4


@pytest.mark.parametrize("task_id", ["repair_belt", "restore_power"])
def test_the_resolved_spec_no_longer_claims_a_static_focus_alone(task_id):
    resolved = get(task_id).spec.to_dict()
    assert resolved["focus_marker"] == "gap"
    # ...but the manifest now also says the focus is selected at runtime.
    assert resolved["focus_policy"] == SELECTING
