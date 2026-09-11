"""The cross-host bridge may expose policy state, never evaluator internals."""

from __future__ import annotations

import numpy as np
import pytest

from factoriorl.agentic.bridge import BridgeRequestError, FactorioBridge


class _Template:
    def __init__(self, key: str, arguments=()):
        self.key = key
        self.arguments = tuple(arguments)


class _Catalog:
    templates = (_Template("wait"), _Template("give", ("to", "item")))


class _Env:
    catalog = _Catalog()

    def __init__(self):
        self._observation = {"tick": 0, "inventory": {"coal": 4}}
        self._truth = {"machine_produced": {"iron-plate": 99}}
        self.calls = []

    def reset(self, seed=None):
        self.calls.append(("reset", seed))

    def action_masks(self):
        return np.array([True, True])

    def argument_domains(self):
        return {"targets": ["h1"], "items": ["coal"]}

    def step_arguments(self, index, arguments):
        self.calls.append(("act", index, dict(arguments)))
        return None, 0.0, False, False, {"success": False, "action_key": "wait"}

    def run_verification(self):
        self.calls.append(("finish",))
        return {"success": True, "reward": 1.0}


def test_observation_contract_does_not_leak_truth():
    bridge = FactorioBridge(_Env(), "construct_smelting_line")
    state = bridge.reset(7)
    assert state["observation"] == {"tick": 0, "inventory": {"coal": 4}}
    assert "truth" not in state
    assert "machine_produced" not in repr(state)


def test_only_registered_actions_with_exact_arguments_are_forwarded():
    env = _Env()
    bridge = FactorioBridge(env, "construct_smelting_line")
    bridge.reset()
    bridge.act(1, {"to": "h1", "item": "coal"})
    assert env.calls[-1] == ("act", 1, {"to": "h1", "item": "coal"})
    with pytest.raises(BridgeRequestError, match="exactly"):
        bridge.act(1, {"to": "h1"})
    with pytest.raises(BridgeRequestError, match="outside"):
        bridge.act(9, {})


def test_finish_is_one_shot_and_blocks_later_actions():
    env = _Env()
    bridge = FactorioBridge(env, "construct_smelting_line")
    bridge.reset()
    assert bridge.finish()["verification"]["success"] is True
    with pytest.raises(BridgeRequestError, match="ended"):
        bridge.finish()
    with pytest.raises(BridgeRequestError, match="ended"):
        bridge.act(0, {})
