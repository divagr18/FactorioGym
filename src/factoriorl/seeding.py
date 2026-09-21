"""Deterministic seed derivation (DESIGN.md section 2: every run records its seeds).

Before this, the only seed in the project was ``map_seed = 424242``, a literal
repeated in four files, with no RNG object and no per-episode seed.

Episode seeds are derived from ``(master, run_id, episode_index)`` rather than
drawn from a stream, so replaying episode 731 reproduces it exactly regardless
of which worker ran it or what order the episodes were dispatched in. That
matters as soon as episodes are handed to a pool of workers.

Training and evaluation draw from **disjoint** branches, so an evaluation
episode can never coincide with one the policy trained on -- a test asserts it.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from enum import StrEnum

#: Kept only as the historical default; configs are expected to set their own.
DEFAULT_MAP_SEED = 424242

_MASK64 = (1 << 64) - 1


class Branch(StrEnum):
    """Disjoint seed streams. Never reuse one branch's seeds in another."""

    TRAIN = "train"
    EVAL = "eval"
    WORKER = "worker"
    GENERATOR = "generator"


def _derive(master: int, *parts: object) -> int:
    payload = "|".join([str(master), *(str(p) for p in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") & _MASK64


@dataclass(frozen=True)
class SeedPlan:
    """Every seed a run uses, derivable from one number and recorded in full."""

    master: int
    run_id: str

    def episode_seed(self, branch: Branch, index: int) -> int:
        """The seed for one episode. Deterministic in the index, not the order."""
        return _derive(self.master, self.run_id, branch.value, index)

    def worker_map_seed(self, worker_index: int) -> int:
        # Factorio map seeds are 32-bit.
        return _derive(self.master, self.run_id, Branch.WORKER.value, worker_index) % (2**32)

    def generator_rng(self, branch: Branch, index: int) -> random.Random:
        """A generator RNG bound to one episode.

        stdlib `random`, not numpy, so `factoriorl.tasks` stays importable
        without the rl extra.
        """
        return random.Random(self.episode_seed(branch, index))

    def to_dict(self) -> dict:
        return {
            "master": self.master,
            "run_id": self.run_id,
            "episode_seed_algorithm": "blake2b(master|run_id|branch|index)[:8]",
            "branches": [b.value for b in Branch],
        }


def seed_everything(master: int) -> dict:
    """Seed every RNG that is actually installed, and report what was seeded.

    Deliberately does *not* enable torch deterministic algorithms:
    reproducibility here comes from the recorded seed plan and manifest, not
    from bitwise GPU determinism, and the throughput cost is not worth paying.
    """
    seeded = {"python": master}
    random.seed(master)
    try:
        import numpy as np

        np.random.seed(master % (2**32))
        seeded["numpy"] = master % (2**32)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(master)
        seeded["torch"] = master
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(master)
            seeded["torch_cuda"] = master
    except ImportError:
        pass
    return seeded
