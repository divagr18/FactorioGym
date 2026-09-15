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

    def decodable(self, operation: int) -> bool:
        """Whether every argument this operation needs has a dimension to come from.

        `craft_recipe`, `set_recipe_at` and `cancel_request` take a `recipe` or a
        `target_request_id`, and no dimension offers either. They could never be
        decoded, only sampled and turned into a counted no-op -- yet the operation
        mask was the catalog's own, which leaves `craft_recipe` legal whenever
        `recipes` is non-empty. The comment on `ARGUMENT_DIMENSION` said these
        "stay masked"; nothing made that true until this.
        """
        template = self.env.catalog.templates[operation]
        return all(ARGUMENT_DIMENSION.get(name) in self._sizes for name in template.arguments)

    def action_masks(self) -> np.ndarray:
        """Flat concatenation, in `nvec` order -- the shape sb3 splits."""
        values = self._domain_values()
        available = self.env.argument_domains()
        operations = np.asarray(self.env.action_masks(), dtype=bool).copy()
        for index in range(len(operations)):
            if operations[index] and not self.decodable(index):
                operations[index] = False
        parts: list[np.ndarray] = [operations]
        for name, _domain in DIMENSIONS[1:]:
            size = self._sizes[name]
            mask = np.zeros(size, dtype=bool)
            mask[UNUSED] = True  # never an all-false dimension
            legal = values.get(name, [])
            if name == "item":
                # Everything an item argument could legally name: what the
                # character holds (`give_to`, `place_at`) *and* what visible
                # entities hold (`take_from`). A dimension mask is built before
                # the operation is sampled, so it cannot tell which of the two
                # applies; the union is the only mask that leaves both reachable.
                #
                # Held-only was the previous rule, and it made `take_from`
                # unable to collect anything the character was not already
                # carrying -- including the first plate out of a furnace. A
                # `give_to` of an item it does not hold is still refused, by
                # `step_arguments`, as a counted decode failure.
                nameable = set(available["items"]) | set(available.get("source_items") or ())
                for index, item in enumerate(legal):
                    mask[index + 1] = item in nameable
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

    def encode(self, operation: int, arguments: dict) -> np.ndarray:
        """The action vector that `decode` turns into this operation and arguments.

        For recording a scripted action in the form a policy emits it, so a
        second backend can replay the vector rather than the semantic call. It
        raises when no vector exists -- a target past `max_targets`, an item
        outside `encoders.ITEMS` -- because that action is one a policy cannot
        take, and a trace that silently substituted a nearby one would compare
        two backends on different actions.
        """
        vector = [UNUSED] * len(DIMENSIONS)
        vector[0] = int(operation)
        names = [name for name, _ in DIMENSIONS]
        values = self._domain_values()
        template = self.env.catalog.templates[int(operation)]
        for argument in template.arguments:
            dimension = ARGUMENT_DIMENSION.get(argument)
            if dimension not in self._sizes:
                raise ValueError(f"{template.key}: {argument} has no dimension")
            if argument not in arguments:
                raise ValueError(f"{template.key} needs argument {argument!r}")
            legal = values.get(dimension, [])
            if arguments[argument] not in legal:
                raise ValueError(
                    f"{template.key}: {argument}={arguments[argument]!r} is not among the "
                    f"{len(legal)} values the {dimension} dimension offers"
                )
            index = legal.index(arguments[argument]) + 1
            position = names.index(dimension)
            if vector[position] not in (UNUSED, index):
                # `from` and `to` share the target dimension. No template in
                # `parameterized-v1` takes both, and this is where one would show.
                raise ValueError(f"{template.key}: two arguments need the {dimension} dimension")
            vector[position] = index
        return np.asarray(vector, dtype=np.int64)

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


def wrap_for_policy(env: Any, skills: bool = False) -> Any:
    """Give a `FactorioEnv` the action space its catalog actually has.

    One function, used everywhere a policy meets an environment -- training,
    evaluation, baselines and every vectorised worker -- because they must
    agree, and they used to be wrapped in four places by hand. Each of those
    only knew about `SkillEnv`, so a task on `parameterized-v1` (`build_line`,
    `construct_smelting_line`) reached the policy as `Discrete(len(catalog))`.
    Stepping any template with a `?` argument through that path raises, so no
    RL run on those tasks could get past its first placement.
    """
    if any(template.parameterized for template in env.catalog.templates):
        if skills:
            raise ValueError(
                f"{env.spec_.id} uses a parameterized catalog; skills are defined over "
                "the discrete primitive catalog and cannot be layered on top of it"
            )
        return ParameterizedEnv(env)
    if skills:
        from factoriorl.skills import SkillEnv

        return SkillEnv(env)
    return env


def as_env_action(space: spaces.Space, action):
    """A policy's output in the form the environment's `step` takes.

    `int(action)` was hard-coded at both places a learned policy's action
    reaches an environment -- the vectorised worker and serial evaluation --
    which is right for `Discrete` and raises on every `MultiDiscrete` vector.
    The first real training run on a parameterized task died on it, on its very
    first step.
    """
    if isinstance(space, spaces.MultiDiscrete):
        return np.asarray(action, dtype=np.int64).reshape(-1)
    return int(np.asarray(action).reshape(-1)[0])


def sample_masked(space: spaces.Space, mask: np.ndarray, rng: np.random.Generator):
    """One uniformly random legal action, for either kind of action space.

    The random floor used to be `rng.choice(np.flatnonzero(mask))`, which is a
    catalog index for `Discrete` and nonsense for `MultiDiscrete`: the flat mask
    is every dimension concatenated, so a "random action" was an index into that
    concatenation, fed to the environment as if it were one operation. A
    factorized action is uniform per dimension over that dimension's legal
    values, which is the distribution an untrained masked policy starts from.
    """
    mask = np.asarray(mask, dtype=bool)
    if isinstance(space, spaces.MultiDiscrete):
        action, offset = [], 0
        for size in (int(n) for n in space.nvec):
            legal = np.flatnonzero(mask[offset : offset + size])
            # `UNUSED` is always legal, so an empty dimension means the mask is
            # broken. Raise rather than pick something: sb3 turns an all-false
            # sub-mask into a uniform draw over illegal values, silently, and a
            # sampler that papered over it would hide the same bug.
            if legal.size == 0:
                raise ValueError("a dimension had no legal value")
            action.append(int(rng.choice(legal)))
            offset += size
        return np.asarray(action, dtype=np.int64)
    return int(rng.choice(np.flatnonzero(mask)))
