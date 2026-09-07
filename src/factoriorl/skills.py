"""Temporally extended actions (PLAN.md 4b.1, 4b.2).

A skill is a small closed-loop policy over the primitive catalog that runs to
its own termination condition and reports one transition. The point is horizon,
not convenience: walking to a chest costs fifteen or twenty primitive decisions,
and a policy that must discover that sequence by exploration spends its whole
budget rediscovering it in every layout.

**Skills must not encode a task's solution.** PLAN 4b.1 draws the line where
section 2 already draws it for action masks: a skill may say "approach the
third-nearest entity", because rank-by-distance is something the observation
states outright; it may not say "repair the belt gap", because that names the
answer and moves evaluator knowledge into the action space. Everything here is
parameterised by *observation slots* -- entity rank, nearest resource -- and
nothing here knows what a task is or which family it is running in.
`tests/unit/test_skills.py` asserts both halves of that: no skill mentions a
task or marker name, and every skill's availability is reconstructible from a
policy-visible observation.

Two consequences worth stating plainly, because they shape how 4b.3 must be
read:

* **A skill's reward is the undiscounted sum of the primitive rewards it
  collects.** Step cost is charged per underlying primitive step, so a skill
  that walks twenty tiles pays twenty steps of cost. Without that, skills would
  be cheaper than primitives by construction and the ablation would measure the
  discount rather than the abstraction.
* **The learner still discounts per decision, not per tick.** Proper treatment
  of variable-duration actions discounts by elapsed time, which the stock
  learner does not express, and PLAN 4b.2 forbids introducing a new algorithm
  here. Elapsed decisions are recorded on every transition so the size of that
  approximation is measurable rather than hidden.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from factoriorl.env import FactorioEnv

#: How many of the nearest entities the policy may address by rank. Four covers
#: a source, a destination, a machine and a distractor without turning the
#: action space into a list of everything visible.
ADDRESSABLE_ENTITIES = 4

#: A skill gives up after this many primitive decisions. A skill that cannot
#: finish is a failed action, not a hung episode.
SKILL_BUDGET = 40

#: Close enough to interact: entity reach is 10 tiles, but stopping adjacent
#: keeps the *nearest-entity* binding that the primitive catalog resolves
#: `$target` against unambiguous.
INTERACT_RANGE = 2.0

#: Resource reach is 2.7 tiles, tighter than entity reach.
RESOURCE_RANGE = 2.0


@dataclass(frozen=True)
class Skill:
    """One temporally extended action."""

    key: str
    #: Observation feature that must be present for this skill to be legal,
    #: checked the same way primitive templates declare `requires`.
    requires: str
    description: str


SKILLS: tuple[Skill, ...] = tuple(
    [
        *[
            Skill(
                key=f"approach_entity_{rank}",
                requires=f"entity_{rank}",
                description=(
                    f"Walk until the {rank}th-nearest visible entity is within interaction range."
                ),
            )
            for rank in range(ADDRESSABLE_ENTITIES)
        ],
        Skill(
            key="approach_resource",
            requires="resource_tile",
            description="Walk until the nearest visible resource tile is within mining reach.",
        ),
        Skill(
            key="mine_batch",
            requires="resource_tile",
            description="Mine the nearest resource repeatedly until the batch lands or stalls.",
        ),
    ]
)


def skill_context(observation: dict) -> dict:
    """Which skills are legal, from the observation alone.

    Mirrors `FactorioEnv._context`: it returns a mapping whose *keys* are the
    features skills declare in `requires`, so availability is a presence check
    and never a computation over hidden state.
    """
    entities = observation.get("entities") or []
    origin = (observation.get("character") or {}).get("position") or [0.0, 0.0]

    def distance(record) -> float:
        point = record.get("p") or [0.0, 0.0]
        return float(math.hypot(point[0] - origin[0], point[1] - origin[1]))

    ordered = sorted(entities, key=distance)
    context: dict = {}
    for rank in range(ADDRESSABLE_ENTITIES):
        context[f"entity_{rank}"] = ordered[rank] if rank < len(ordered) else None
    tiles = (observation.get("resources") or {}).get("tiles") or []
    context["resource_tile"] = min(tiles, key=distance) if tiles else None
    return context


def _movement_key(env: FactorioEnv, direction: str, error: float) -> str | None:
    """The finest stride the task's catalog offers for the distance remaining.

    Each tier must be finer than the error it closes or the walk oscillates on
    that stride's lattice -- the defect that made every family unsolvable
    before a 2-tick stride existed.
    """
    if error > 5.0:
        order = ("move", "step", "nudge")
    elif error > 1.5:
        order = ("step", "nudge", "move")
    else:
        order = ("nudge", "step", "move")
    keys = env.catalog.keys()
    for prefix in order:
        candidate = f"{prefix}_{direction}"
        if candidate in keys:
            return candidate
    return None


def _character(env: FactorioEnv) -> tuple[float, float]:
    position = (env._observation.get("character") or {}).get("position") or [0.0, 0.0]
    return float(position[0]), float(position[1])


class SkillRunner:
    """Executes a skill as a loop of primitive steps on a live environment."""

    def __init__(self, env: FactorioEnv) -> None:
        self.env = env

    def run(self, skill: Skill) -> dict:
        """Run `skill` to termination.

        Returns the accumulated transition. `steps` is the number of primitive
        decisions consumed, which is what makes the per-decision discounting
        approximation measurable.
        """
        context = skill_context(self.env._observation)
        target = context.get(skill.requires)
        if target is None:
            # Masking should have prevented this; treat it as a failed action
            # rather than an exception so a masking bug is visible in the
            # traces instead of crashing a training run.
            return self._blank("unavailable")

        if skill.key == "mine_batch":
            return self._mine_batch()
        goal = tuple(target.get("p") or [0.0, 0.0])
        reach = RESOURCE_RANGE if skill.requires == "resource_tile" else INTERACT_RANGE
        return self._walk_to(goal, reach)

    # ------------------------------------------------------------ internals

    def _blank(self, outcome: str) -> dict:
        observation, _, terminated, truncated, info = self.env.step(self.env.catalog.wait_index)
        return {
            "observation": observation,
            "reward": 0.0,
            "terminated": terminated,
            "truncated": truncated,
            "info": info,
            "steps": 1,
            "outcome": outcome,
        }

    def _walk_to(self, goal: tuple[float, float], reach: float) -> dict:
        total, steps = 0.0, 0
        result: dict = {}
        stalled = 0
        for _ in range(SKILL_BUDGET):
            here = _character(self.env)
            dx, dy = goal[0] - here[0], goal[1] - here[1]
            error = math.hypot(dx, dy)
            if error <= reach:
                return self._finish(result, total, steps, "arrived")
            if abs(dx) >= abs(dy):
                direction = "east" if dx > 0 else "west"
            else:
                direction = "south" if dy > 0 else "north"
            key = _movement_key(self.env, direction, max(abs(dx), abs(dy)))
            if key is None:
                return self._finish(result, total, steps, "no_movement_action")
            result = self._primitive(key)
            total += result["reward"]
            steps += 1
            if result["terminated"] or result["truncated"]:
                return self._finish(result, total, steps, "episode_ended")
            if math.dist(here, _character(self.env)) < 0.05:
                stalled += 1
                if stalled >= 3:
                    return self._finish(result, total, steps, "blocked")
            else:
                stalled = 0
        return self._finish(result, total, steps, "budget")

    def _mine_batch(self) -> dict:
        """Mine until the batch lands, the resource is gone, or progress stops.

        Mining is ongoing and slow -- roughly four decision intervals per ore --
        so issuing one mine and moving on collects nothing. This is precisely
        the sequence a flat policy has to rediscover in every episode.
        """
        keys = self.env.catalog.keys()
        mine_key = next((k for k in ("mine_nearest_5", "mine_nearest") if k in keys), None)
        if mine_key is None:
            return self._blank("no_mine_action")
        total, steps = 0.0, 0
        result = self._primitive(mine_key)
        total += result["reward"]
        steps += 1
        if result["terminated"] or result["truncated"]:
            return self._finish(result, total, steps, "episode_ended")
        held = self._carried()
        idle = 0
        for _ in range(SKILL_BUDGET - 1):
            result = self._primitive(self.env.catalog.wait_index)
            total += result["reward"]
            steps += 1
            if result["terminated"] or result["truncated"]:
                return self._finish(result, total, steps, "episode_ended")
            # The mod already knows when the batch is done: a mine runs as an
            # in-flight operation and settles when the count is reached or the
            # resource is gone. Waiting for the inventory to stop rising
            # instead cost eight idle decisions on every batch -- pure overhead
            # charged to the skill, which would have shown up in the ablation
            # as skills being slower rather than as this bug.
            inflight = self.env._observation.get("inflight") or []
            if not any(entry.get("action") == "mine" for entry in inflight):
                return self._finish(result, total, steps, "complete")
            now = self._carried()
            if now > held:
                held = now
                idle = 0
            else:
                idle += 1
                if idle >= 8:
                    return self._finish(result, total, steps, "stalled")
        return self._finish(result, total, steps, "budget")

    def _carried(self) -> int:
        return sum((self.env._observation.get("inventory") or {}).values())

    def _primitive(self, action) -> dict:
        index = action if isinstance(action, int) else self.env.catalog.keys().index(action)
        observation, reward, terminated, truncated, info = self.env.step(index)
        return {
            "observation": observation,
            "reward": float(reward),
            "terminated": terminated,
            "truncated": truncated,
            "info": info,
        }

    def _finish(self, result: dict, total: float, steps: int, outcome: str) -> dict:
        if not result:
            return self._blank(outcome)
        return {
            "observation": result["observation"],
            "reward": total,
            "terminated": result["terminated"],
            "truncated": result["truncated"],
            "info": result["info"],
            "steps": max(steps, 1),
            "outcome": outcome,
        }


def skill_masks(observation: dict, skills: tuple[Skill, ...] = SKILLS) -> np.ndarray:
    """Availability for each skill, from the observation alone."""
    context = skill_context(observation)
    return np.array([context.get(s.requires) is not None for s in skills], dtype=bool)


class SkillEnv:
    """`FactorioEnv` with skills appended to its action space.

    A wrapper rather than a change to `FactorioEnv`, for two reasons. Phase 3
    is accepted and its environment contract should not move underneath it; and
    the 4b.3 ablation wants flat and skill-augmented arms that differ in exactly
    one switch, which a wrapper gives literally.

    Action indices are the primitive catalog's, unchanged, followed by the
    skills in `SKILLS` order. Primitive indices therefore keep their meaning,
    so a flat checkpoint and a skill checkpoint disagree only about actions the
    flat one never had.
    """

    def __init__(self, env: FactorioEnv, skills: tuple[Skill, ...] = SKILLS) -> None:
        from gymnasium import spaces

        self.env = env
        self.skills = skills
        self.runner = SkillRunner(env)
        self.primitive_count = len(env.catalog)
        self.action_space = spaces.Discrete(self.primitive_count + len(skills))
        self.observation_space = env.observation_space
        self.spec_ = env.spec_
        self.task = env.task
        self.session = env.session

    # ----------------------------------------------------------- gym API

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action: int):
        index = int(action)
        if index < self.primitive_count:
            observation, reward, terminated, truncated, info = self.env.step(index)
            info = {**info, "skill": None, "skill_steps": 1}
            return observation, reward, terminated, truncated, info

        skill = self.skills[index - self.primitive_count]
        result = self.runner.run(skill)
        info = {
            **result["info"],
            "skill": skill.key,
            "skill_steps": result["steps"],
            "skill_outcome": result["outcome"],
        }
        return (
            result["observation"],
            result["reward"],
            result["terminated"],
            result["truncated"],
            info,
        )

    def action_masks(self) -> np.ndarray:
        primitive = self.env.action_masks()
        skills = skill_masks(self.env._observation, self.skills)
        return np.concatenate([primitive, skills])

    def close(self) -> None:
        self.env.close()

    def __getattr__(self, name):
        # Everything else -- _observation, accountant, seed_plan -- belongs to
        # the wrapped environment.
        return getattr(self.env, name)
