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


#: Recorded by a run started from someone else's weights. Not a cosmetic
#: label: PLAN 4.2 forbids scripted-solution labels during ordinary PPO
#: training, and a BC-initialised run *inherits* parameters fitted on exactly
#: those labels. It is therefore not comparable to a scratch PPO run and must
#: never appear in the same column, which is only enforceable if the run says
#: what it is.
INITIALISED_MODE = "ppo-initialised-from-checkpoint-v1"


def initialise_from(model, path) -> dict:
    """Start `model` from a saved policy's weights, keeping a fresh optimizer.

    R5.3's third arm. The comparison it asks for is scratch PPO against
    behaviour cloning against BC-initialised PPO, and the third did not exist.

    **Only the policy's parameters are copied.** `MaskablePPO.load` would
    replace the optimizer state too, and BC's optimizer moments were
    accumulated under a supervised cross-entropy loss -- carrying Adam's
    running averages from that objective into a policy-gradient objective
    conditions the first PPO updates on gradients of a different loss. The
    point of the arm is the *starting parameters*, so the optimizer starts
    fresh.

    **An architecture mismatch is refused, not absorbed.** `load_state_dict`
    with `strict=False` would silently leave every mismatched layer at its
    random initialisation, producing a policy that is neither scratch nor
    cloned and a result attributable to nothing. The signatures compare name
    and shape for every parameter, which is what SB3's own load compares, and
    the checkpoint's is read without loading it so an unloadable file gets a
    diagnosis rather than a traceback.

    Returns what the manifest should record, including a digest of the weights
    before and after: identical digests mean nothing was transferred, and a run
    that silently trained from scratch under this flag would otherwise be
    indistinguishable from one that did not.
    """
    candidate = Path(path)
    if candidate.suffix != ".zip":
        candidate = candidate.with_suffix(".zip")
    if not candidate.is_file():
        raise FileNotFoundError(f"no checkpoint to initialise from at {candidate}")

    target = architecture_signature(model)
    source = checkpoint_signature(candidate)
    if target != source:
        raise ValueError(
            "refusing to initialise from a checkpoint with a different "
            f"architecture: this model is {target}, {candidate.name} is {source}. "
            "A partial load would leave the mismatched layers randomly "
            "initialised and the result attributable to neither arm"
        )

    before = _digest_values(model.policy.state_dict())

    import io
    import zipfile

    with zipfile.ZipFile(candidate) as bundle:
        with bundle.open("policy.pth") as member:
            blob = io.BytesIO(member.read())
    # `weights_only=True`: loading parameters needs no arbitrary pickle
    # execution, and this file may not be one we produced.
    state = torch.load(blob, map_location=model.device, weights_only=True)
    model.policy.load_state_dict(state, strict=True)
    after = _digest_values(model.policy.state_dict())

    return {
        "initialised_from": candidate.name,
        "initialised_from_path": str(candidate),
        "source_architecture_signature": source,
        "architecture_signature": target,
        "policy_weights_digest_before": before,
        "policy_weights_digest_after": after,
        # The check that the flag did anything. Equal digests mean the load was
        # a no-op and the run is really a scratch run wearing a label.
        "weights_changed": before != after,
        "optimizer_state": (
            "fresh. BC's Adam moments were accumulated under a supervised loss, "
            "so carrying them into a policy-gradient objective would condition "
            "the first updates on gradients of a different loss"
        ),
        "training_mode": INITIALISED_MODE,
        "not_comparable_to": (
            "a scratch PPO run. These parameters were fitted on scripted-solution "
            "labels, which PLAN 4.2 forbids during ordinary PPO training"
        ),
    }


def _digest_values(state_dict) -> str:
    """Digest of the parameter *values*, unlike `_digest`'s names and shapes.

    Two policies with identical architecture have identical `_digest`; this
    separates them, which is what makes "did the load actually transfer
    anything" a checkable question rather than an assumption.
    """
    hasher = hashlib.blake2b(digest_size=8)
    for name in sorted(state_dict):
        value = state_dict[name]
        if isinstance(value, torch.Tensor):
            hasher.update(name.encode("utf-8"))
            hasher.update(value.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


def observation_compatibility(model, env) -> dict:
    """Whether this checkpoint can be fed by this environment.

    A checkpoint can load and still be unusable. SB3 rebuilds a policy's input
    layers from the observation space pickled *inside* the checkpoint, so
    `load` succeeds against any environment; the failure surfaces later, deep
    in `obs_to_tensor`, as `cannot reshape array of size 14 into shape (10)`
    with no indication of which key or why.

    That is not hypothetical. `ITEMS` grew from 10 to 14 when R4.3 added two
    task families, so every checkpoint trained before it expects a 10-wide
    `inventory` and today's environment offers 14. `architecture_signature`
    does not catch this either -- it compares the policy's own tensors, not the
    space the environment will present.

    So this compares the two spaces key by key and names what differs, which is
    the difference between "this checkpoint predates the R4.3 item catalog" and
    a reshape traceback.
    """
    theirs = getattr(model, "observation_space", None)
    ours = getattr(env, "observation_space", None)
    if theirs is None or ours is None:
        return {"comparable": False, "why": "one side exposes no observation space"}

    def shapes(space) -> dict:
        spaces_dict = getattr(space, "spaces", None)
        if spaces_dict is None:
            return {"<box>": tuple(getattr(space, "shape", ()) or ())}
        return {key: tuple(getattr(sub, "shape", ()) or ()) for key, sub in spaces_dict.items()}

    mine, yours = shapes(ours), shapes(theirs)
    differences = {
        key: {"checkpoint": yours.get(key), "environment": mine.get(key)}
        for key in sorted(set(mine) | set(yours))
        if mine.get(key) != yours.get(key)
    }
    # `int` and `bool` deliberately: `Discrete.n` is a numpy integer, so the
    # comparison yields `np.bool_` and this whole dict goes into JSON, which
    # refuses numpy scalars.
    action_theirs = _action_signature(getattr(model, "action_space", None))
    action_ours = _action_signature(getattr(env, "action_space", None))
    return {
        "comparable": True,
        "compatible": bool(not differences and action_theirs == action_ours),
        "observation_differences": differences,
        "actions": {"checkpoint": action_theirs, "environment": action_ours},
        "hint": (
            "an action-count mismatch is usually the skills flag: a policy trained "
            "with --skills has a larger catalog. An observation mismatch usually "
            "means the checkpoint predates a change to the item or entity catalog, "
            "and cannot be evaluated against this tree at all"
        )
        if differences or action_theirs != action_ours
        else None,
    }


def _action_signature(space) -> int | list[int] | None:
    """What an action space's *shape* is, in a form two checkpoints can compare.

    Read as `.n` alone, this was `None` for every `MultiDiscrete` space, so any
    two parameterized policies compared equal -- a checkpoint trained on one
    catalog would have been declared compatible with an environment exposing a
    different one. `int` and `list` deliberately: numpy scalars break the JSON
    this result is written into.
    """
    if space is None:
        return None
    nvec = getattr(space, "nvec", None)
    if nvec is not None:
        return [int(n) for n in nvec]
    n = getattr(space, "n", None)
    return None if n is None else int(n)


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
