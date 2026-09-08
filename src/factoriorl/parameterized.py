"""A factorized action space over the shared semantic contract (R2.2).

The discrete catalog cannot carry an argument, so `$target` binds to whatever
is nearest and placement is welded to four adjacent tiles. `parameterized-v1`
gives the arguments names; this gives a policy a way to *choose* them.

One `MultiDiscrete` dimension per argument, masked per dimension. Factorized
rather than autoregressive, which R2.2 allows and stock `sb3_contrib` supports:
`MaskableMultiCategoricalDistribution` splits a flat mask columnwise by
`nvec`, so sampling, log-probability and PPO updates all work unmodified.

**The limitation that follows, stated rather than hidden.** Masks are fetched
once before the forward pass, so a dimension's mask cannot depend on the
operation actually sampled. Index 0 of every argument dimension is therefore an
explicit `UNUSED` sentinel, and an operation that needs an argument left unused
decodes to a no-op with `decode_failure` naming the missing argument. That is
observable and countable, not silently substituted. An autoregressive
distribution would remove it and needs a custom distribution class -- three
things block it in stock sb3: masks are fetched before the forward pass, all
sub-distributions come from one flat head and are sampled independently, and
`log_prob` sums independent factors, so autoregressive sampling scored through
it yields a wrong PPO ratio.

`docs/research/book-synthesis-2026-09-09.md` §8 suggests a further step: score
placement candidates by their resulting local structure rather than giving each
index an unrelated logit, which may generalize across translated and rotated
layouts. That is a representation change on top of this interface, not a
replacement for it, and it is left to a separate experiment with its own gate.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from factoriorl import catalog as catalog_module

#: Index 0 of every argument dimension. Always legal, so no dimension can have
#: an all-false mask -- sb3's masked categorical turns that into a *uniform*
#: draw over illegal values, silently.
UNUSED = 0

#: Bounds per argument dimension, excluding the sentinel. Targets match the
#: encoder's `MAX_ENTITIES` so a policy can address anything it can see.
MAX_TARGETS = 32
MAX_PLACEMENTS = 121

#: Which domain each dimension draws from, and the payload argument it fills.
#: `operation` is the catalog index and takes no domain.
DIMENSIONS: tuple[tuple[str, str | None], ...] = (
    ("operation", None),
    ("target", "targets"),
    ("placement", "placements"),
    ("direction", "directions"),
    ("item", "items"),
    ("amount", "amounts"),
)

#: Which dimension supplies each payload argument name.
ARGUMENT_DIMENSION: dict[str, str] = {
    "handle": "target",
    "from": "target",
    "to": "target",
    "position": "placement",
    "direction": "direction",
    "item": "item",
    "count": "amount",
    # Not observable yet (R2.3): no dimension can offer a value, so any
    # operation needing one stays masked.
    "recipe": "recipe",
    "technology": "technology",
    "target_request_id": "request",
}


class ParameterizedEnv(gym.Env):
    """Wraps a `FactorioEnv` whose catalog declares arguments.

    A wrapper rather than a change to `FactorioEnv`, following `SkillEnv`: the
    discrete path keeps its meaning, so a flat checkpoint and a parameterized
    one disagree only about actions the flat one never had.
    """

    metadata: dict = {}

    def __init__(
        self, env: Any, max_targets: int = MAX_TARGETS, max_placements: int = MAX_PLACEMENTS
    ):
        self.env = env
        self.max_targets = max_targets
        self.max_placements = max_placements
        self._sizes = {
            "operation": len(env.catalog),
            "target": max_targets + 1,
            "placement": max_placements + 1,
            "direction": len(catalog_module.DIRECTIONS) + 1,
            "item": self._item_slots() + 1,
            "amount": self._amount_slots() + 1,
        }
        self.action_space = spaces.MultiDiscrete([self._sizes[name] for name, _ in DIMENSIONS])
        self.observation_space = env.observation_space
        self.decode_failures = 0

    # ---- sizes that must not move between runs ------------------------
    def _item_slots(self) -> int:
        from factoriorl import encoders

        return len(encoders.ITEMS)

    def _amount_slots(self) -> int:
        from factoriorl.env import TRANSFER_AMOUNTS

        return len(TRANSFER_AMOUNTS)

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def __getattr__(self, name):
        return getattr(self.env, name)

    # ---- masks --------------------------------------------------------
    def _domain_values(self) -> dict[str, list]:
        domains = self.env.argument_domains()
        from factoriorl import encoders

        return {
            "target": list(domains["targets"])[: self.max_targets],
            "placement": list(domains["placements"])[: self.max_placements],
            "direction": list(domains["directions"]),
            # Fixed slots, so an item's index means the same thing in every
            # scene; availability is what the mask expresses.
            "item": list(encoders.ITEMS),
            "amount": list(domains["amounts"]),
        }

    def action_masks(self) -> np.ndarray:
        """Flat concatenation, in `nvec` order -- the shape sb3 splits."""
        values = self._domain_values()
        available = self.env.argument_domains()
        parts: list[np.ndarray] = [np.asarray(self.env.action_masks(), dtype=bool)]
        for name, _domain in DIMENSIONS[1:]:
            size = self._sizes[name]
            mask = np.zeros(size, dtype=bool)
            mask[UNUSED] = True  # never an all-false dimension
            legal = values.get(name, [])
            if name == "item":
                held = set(available["items"])
                for index, item in enumerate(legal):
                    mask[index + 1] = item in held
            else:
                for index in range(min(len(legal), size - 1)):
                    mask[index + 1] = True
            parts.append(mask)
        return np.concatenate(parts)

    # ---- decode -------------------------------------------------------
    def decode(self, action) -> tuple[int, dict, str | None]:
        """Turn one action vector into a catalog index and its arguments.

        Returns the operation, the arguments it needs, and a failure reason
        when a needed argument was left unused -- which a factorized space
        cannot prevent, because the argument masks were built before the
        operation was sampled.
        """
        vector = [int(v) for v in np.asarray(action).reshape(-1)]
        operation = vector[0]
        chosen = dict(zip([name for name, _ in DIMENSIONS], vector, strict=True))
        values = self._domain_values()
        template = self.env.catalog.templates[operation]

        arguments: dict = {}
        for argument in template.arguments:
            dimension = ARGUMENT_DIMENSION.get(argument)
            if dimension not in self._sizes:
                return operation, {}, f"{argument}: no dimension offers a value"
            index = chosen[dimension]
            if index == UNUSED:
                return operation, {}, f"{argument}: left unused"
            legal = values.get(dimension, [])
            if index - 1 >= len(legal):
                return operation, {}, f"{argument}: index {index} past the domain"
            arguments[argument] = legal[index - 1]
        return operation, arguments, None

    def step(self, action):
        operation, arguments, failure = self.decode(action)
        if failure is not None:
            self.decode_failures += 1
            wait = self.env.catalog.wait_index
            observation, reward, terminated, truncated, info = self.env.step(wait)
            info = {**info, "decode_failure": failure, "requested_operation": operation}
            return observation, reward, terminated, truncated, info
        try:
            observation, reward, terminated, truncated, info = self.env.step_arguments(
                operation, arguments
            )
        except ValueError as exc:
            # The domain moved between mask and action, or an argument was
            # sampled that this operation cannot use. A named no-op, not a
            # crashed rollout.
            self.decode_failures += 1
            wait = self.env.catalog.wait_index
            observation, reward, terminated, truncated, info = self.env.step(wait)
            return (
                observation,
                reward,
                terminated,
                truncated,
                {**info, "decode_failure": str(exc), "requested_operation": operation},
            )
        return (
            observation,
            reward,
            terminated,
            truncated,
            {**info, "decode_failure": None, "arguments": arguments},
        )

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
