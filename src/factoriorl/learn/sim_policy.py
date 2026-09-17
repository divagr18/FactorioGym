"""A policy trained on factory-sim, run against the real engine.

factory-sim (github.com/divagr18/factory-sim) reproduces this project's
`parameterized-v1` action space and `local-v1` tensor layout tick for tick, and
trains on them. Its `train.py` exports a TorchScript module taking the six
observation tensors, the flat 201-entry action mask and a `greedy` flag, and
returning the MultiDiscrete vector. Nothing of factory-sim is imported here:
the file is the whole interface, which is what makes this a transfer test and
not a shared-code test.

`predict` has the signature `learn.train.evaluate` calls, so a `SimPolicy`
evaluates exactly as an SB3 checkpoint does.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

KEYS = ("grid", "entities", "entity_mask", "self", "inventory", "goal")


class SimPolicy:
    def __init__(self, path: str | Path, seed: int = 0) -> None:
        import torch

        self.path = Path(path)
        self.digest = hashlib.sha256(self.path.read_bytes()).hexdigest()[:16]
        self.module = torch.jit.load(str(self.path), map_location="cpu").eval()
        self._torch = torch
        self._generator_seed = seed
        torch.manual_seed(seed)

    def predict(self, observation: dict, action_masks=None, deterministic: bool = True):
        torch = self._torch
        tensors = [torch.from_numpy(np.asarray(observation[k])[None]) for k in KEYS]
        if action_masks is None:
            raise ValueError("a factory-sim policy needs the action mask")
        mask = torch.from_numpy(np.asarray(action_masks, dtype=bool)[None])
        with torch.no_grad():
            action = self.module(*tensors, mask, bool(deterministic))
        return action[0].numpy().astype(np.int64), None

    def describe(self) -> dict:
        return {"policy": "factory-sim TorchScript", "file": self.path.name, "sha256": self.digest}
