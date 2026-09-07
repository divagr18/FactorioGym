"""Compact policy encoder (PLAN.md 4.1).

A small CNN over the spatial grid, a masked per-entity MLP, and an MLP over the
inventory/self/goal vectors, concatenated into an SB3 features extractor driving
MaskablePPO.

The extractor is **versioned** and that version goes into the run manifest: a
silent change to the encoder invalidates every checkpoint trained on it, and a
checkpoint whose provenance cannot be checked is an opaque blob.
"""

from __future__ import annotations

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
EXTRACTOR_VERSION = 2


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

        # The three strided convolutions take 65x65 down to 9x9, and this used
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


def describe(model) -> dict:
    """Model configuration for the run manifest (PLAN.md section 2)."""
    parameters = sum(p.numel() for p in model.policy.parameters())
    return {
        "algorithm": type(model).__name__,
        "extractor": FactorioExtractor.__name__,
        "extractor_version": EXTRACTOR_VERSION,
        "parameters": parameters,
        "device": str(model.device),
        "learning_rate": getattr(model, "learning_rate", None),
        "n_steps": getattr(model, "n_steps", None),
        "batch_size": getattr(model, "batch_size", None),
        "gamma": getattr(model, "gamma", None),
    }
