"""Compact policy encoder (PLAN.md 4.1).

A small CNN over the spatial grid, a masked per-entity MLP, and an MLP over the
inventory/self/goal vectors, concatenated into an SB3 features extractor driving
MaskablePPO.

The extractor is **versioned** and that version goes into the run manifest: a
silent change to the encoder invalidates every checkpoint trained on it, and a
checkpoint whose provenance cannot be checked is an opaque blob.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

#: Bump whenever the tensor layout or the network shape changes.
#: Spatial resolution the grid is pooled to before flattening, and the width
#: of the resulting grid embedding.
GRID_POOL = 4
GRID_FEATURES = 128

#: Bumped to 2: the grid encoder keeps spatial structure and the goal vector
#: carries landmarks. Both change what a checkpoint means, so a v1 checkpoint
#: is not comparable to a v2 one and the gate checks this against the manifest.
#:
#: Bumped to 3: `ObservationProfile.cell_size` makes the grid resolution
#: settable, so one grid cell no longer necessarily means one world tile. This
#: version is the *only* thing that catches that change. Because `grid_net`
#: ends in `AdaptiveAvgPool2d(GRID_POOL)`, it absorbs any spatial input size:
#: a checkpoint trained on the 65x65 grid loads against a 33x33 one without a
#: single shape error, runs, and produces garbage -- with every gate green,
#: because nothing else compares the two geometries.
#: 5: the `self` vector's last three slots became the action-outcome signal
#: (last action refused, last action completed, refusal rate), fed by the
#: `events` block that `local-v2` now publishes. Same shape, different meaning,
#: which is exactly the case this version exists to catch.
#: 6: the inventory vector gained four item slots, so its width changed.
#: 7: entity slots 11-15, previously always zero, carry the machine stop
#:    cause, a working flag, whether a status exists at all, and fuel and
#:    output totals.
EXTRACTOR_VERSION = 7


class FactorioExtractor(BaseFeaturesExtractor):
    """Grid + entities + vectors -> one feature vector."""

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256) -> None:
        super().__init__(observation_space, features_dim)

        grid_channels = observation_space["grid"].shape[0]
        entity_rows, entity_features = observation_space["entities"].shape
        vector_dim = (
            observation_space["self"].shape[0]
            + observation_space["inventory"].shape[0]
            + observation_space["goal"].shape[0]
        )

        # Each of the three strided convolutions halves both spatial dimensions
        # (65x65 down to 9x9 for `local-v1`, 33x33 down to 5x5 for the coarse
        # profile), and this used
        # to end in `AdaptiveAvgPool2d(1)`: a global average that collapsed the
        # whole map to 64 numbers and destroyed every trace of *where* anything
        # was. Walls survived that only because they also appear in the entity
        # list with relative coordinates. Resources do not -- ore reaches the
        # policy through the grid planes alone -- so on `mine_smelt` the policy
        # could see how much ore was nearby and not which direction it lay in.
        # Pooling to 4x4 keeps a coarse spatial layout at a cost of one linear
        # layer.
        self.grid_net = nn.Sequential(
            nn.Conv2d(grid_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(GRID_POOL),
            nn.Flatten(),
            nn.Linear(64 * GRID_POOL * GRID_POOL, GRID_FEATURES),
            nn.ReLU(),
        )
        self.entity_net = nn.Sequential(
            nn.Linear(entity_features, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU()
        )
        self.vector_net = nn.Sequential(
            nn.Linear(vector_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU()
        )
        # GRID_FEATURES (grid) + 128 (entity mean+max) + 128 (vectors)
        self.head = nn.Sequential(
            nn.Linear(GRID_FEATURES + 128 + 128, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU(),
        )
        self.entity_rows = entity_rows

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        grid = self.grid_net(observations["grid"])

        entities = self.entity_net(observations["entities"])
        mask = observations["entity_mask"].float().unsqueeze(-1)
        # Masked pooling: padding rows must not drag the mean toward zero or
        # win the max, or a scene with three entities would encode differently
        # from the same scene padded to thirty-two.
        counts = mask.sum(dim=1).clamp(min=1.0)
        mean_pool = (entities * mask).sum(dim=1) / counts
        max_pool = (entities.masked_fill(mask == 0, -1e9)).max(dim=1).values
        max_pool = torch.nan_to_num(max_pool, neginf=0.0)

        vectors = self.vector_net(
            torch.cat(
                [observations["self"], observations["inventory"], observations["goal"]], dim=1
            )
        )
        return self.head(torch.cat([grid, mean_pool, max_pool, vectors], dim=1))


def policy_kwargs(features_dim: int = 256) -> dict:
    return {
        "features_extractor_class": FactorioExtractor,
        "features_extractor_kwargs": {"features_dim": features_dim},
        "net_arch": {"pi": [256], "vf": [256]},
    }


def architecture_signature(model) -> str:
    """Digest of every parameter's name and shape. What `load` actually compares.

    `EXTRACTOR_VERSION` is a declaration of intent and is orthogonal to whether
    a checkpoint loads, in both directions. Measured over the whole history:

    * Of six bumps, exactly **one** changed a width the loader cannot adapt to
      -- `GRID_FEATURES` 64 to 128, which took `head.0` from `(256, 320)` to
      `(256, 384)` and added a `grid_net.8` layer that v1 checkpoints do not
      have at all. Four bumps were semantics-only, and one changed a width the
      loader *does* adapt to.
    * `GOAL_ENCODING_VERSION` went 3 to 4 in a commit that never touched this
      file, so that meaning change carries no extractor bump at all.

    So a matching integer does not vouch for a checkpoint and a mismatched one
    does not condemn it: a checkpoint recording version 3 loads cleanly against
    today's 7, carrying a ten-item inventory against the current fourteen,
    because SB3 rebuilds the *input* layers from the observation space pickled
    inside the checkpoint. Only the widths derived from the constants in this
    module -- `GRID_FEATURES`, `features_dim`, and the hardcoded entity and
    vector head widths -- can ever break a load.

    Hashing the state dict's `(name, shape)` pairs records exactly that, so a
    gate can say "this checkpoint predates the current architecture" instead of
    letting `MaskablePPO.load` raise a size-mismatch traceback.
    """
    return _digest(model.policy.state_dict())


def _digest(state_dict) -> str:
    pairs = sorted(
        (name, tuple(int(dim) for dim in tensor.shape)) for name, tensor in state_dict.items()
    )
    return hashlib.sha256(json.dumps(pairs, separators=(",", ":")).encode("utf-8")).hexdigest()[:16]


def checkpoint_signature(path) -> str:
    """The same digest, read out of a saved checkpoint **without loading it**.

    Needed because the interesting case is a checkpoint that *cannot* load:
    computing the signature from a live model would require the very load that
    raises. An SB3 zip carries the policy's tensors in `policy.pth`, so the
    shapes are readable directly and a gate can report "this checkpoint's
    architecture is X, the declared one is Y" instead of a size-mismatch
    traceback.

    `weights_only=True`: this reads an artefact to describe it, and nothing in
    a shape comparison needs arbitrary pickle execution.
    """
    import io
    import zipfile

    import torch as th

    candidate = Path(path)
    if candidate.suffix != ".zip":
        candidate = candidate.with_suffix(".zip")
    with zipfile.ZipFile(candidate) as bundle:
        with bundle.open("policy.pth") as member:
            blob = io.BytesIO(member.read())
    return _digest(th.load(blob, map_location="cpu", weights_only=True))


def describe(model) -> dict:
    """Model configuration for the run manifest (PLAN.md section 2)."""
    parameters = sum(p.numel() for p in model.policy.parameters())
    return {
        "algorithm": type(model).__name__,
        "extractor": FactorioExtractor.__name__,
        "extractor_version": EXTRACTOR_VERSION,
        # The mechanical check, beside the declared one. See
        # `architecture_signature`.
        "architecture_signature": architecture_signature(model),
        "parameters": parameters,
        "device": str(model.device),
        "learning_rate": getattr(model, "learning_rate", None),
        "n_steps": getattr(model, "n_steps", None),
        "batch_size": getattr(model, "batch_size", None),
        "gamma": getattr(model, "gamma", None),
    }
