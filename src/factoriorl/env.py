"""Gymnasium environment (PLAN.md 3.4).

One transition is one fused ``step`` request: apply the action, run the decision
interval, return the observation. Two round trips is the floor for exact
stepping over RCON, and that is what the transport work in Phase T bought --
before it, a step was four round trips at 267 ms each.

Three properties are enforced rather than hoped for:

* **masks always leave a legal no-op**, asserted every step;
* **masks use only policy-visible information** -- they are built from the
  observation, never from the blueprint, the task truth, or the reward state,
  so PLAN section 3's "masks must not use hidden solutions" is checkable;
* **termination and truncation are exclusive**, and an infrastructure failure is
  neither: a dead worker is not a task outcome (PLAN section 2), so it is
  reported with ``excluded_from_metrics`` and the trainer must drop it.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from factoriorl import catalog as catalog_module
from factoriorl import encoders
from factoriorl.errors import FactorioRLError, InfrastructureFailure, ProtocolError
from factoriorl.rewards import RewardAccountant
from factoriorl.seeding import Branch, SeedPlan
from factoriorl.session import WorkerSession
from factoriorl.tasks import RegisteredTask
from factoriorl.tasks.spec import LayoutFamily


def blueprint_digest(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


class FactorioEnv(gym.Env):
    """One task on one worker."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        task: RegisteredTask,
        session: WorkerSession,
        seed_plan: SeedPlan,
        branch: Branch = Branch.TRAIN,
        split: str = "train",
        shaping: bool = True,
    ) -> None:
        self.task = task
        self.spec_ = task.spec
        self.session = session
        self.seed_plan = seed_plan
        self.branch = branch
        self.split = split
        self.catalog = catalog_module.resolve(self.spec_.catalog, self.spec_.catalog_subset)
        self.action_space = spaces.Discrete(len(self.catalog))
        self.observation_space = encoders.observation_space()
        self.accountant = RewardAccountant(self.spec_.rewards, shaping_enabled=shaping)

        self._episode_index = -1
        self._steps = 0
        self._installed: set[str] = set()
        self._observation: dict = {}
        self._truth: dict = {}
        self._family: LayoutFamily | None = None
        self._last_action_ok = True

    # ------------------------------------------------------------ helpers

    def _families(self) -> tuple[LayoutFamily, ...]:
        families = self.spec_.families(self.split)
        if not families:
            raise FactorioRLError(f"task {self.spec_.id} has no '{self.split}' layout family")
        return families

    def _install(self, blueprint) -> str:
        # The scene carries every marker, including decoys, because the
        # evaluator scores against them. Which of those the *observation* may
        # show is a property of the objective, and only the task knows it, so
        # the allowlist is declared here rather than inferred in the mod.
        payload = blueprint.to_dict(public_markers=self.spec_.public_markers)
        digest = blueprint_digest(payload)
        if digest not in self._installed:
            self.session.define_scenario(payload, digest)
            self._installed.add(digest)
        return digest

    def _refresh_truth(self) -> None:
        self._truth = self.session.truth().response.result or {}

    def _context(self) -> dict:
        """Runtime bindings for catalog templates, from the observation only."""
        entities = self._observation.get("entities", [])
        origin = (self._observation.get("sensor") or {}).get("origin", [0.0, 0.0])

        def distance(record) -> float:
            p = record.get("p", [0, 0])
            return float(np.hypot(p[0] - origin[0], p[1] - origin[1]))

        nearest_entity = min(entities, key=distance) if entities else None
        tiles = (self._observation.get("resources") or {}).get("tiles", [])
        nearest_resource = min(tiles, key=distance) if tiles else None
        character = self._observation.get("character") or {}
        position = character.get("position", [0, 0])
        context = {
            "target": nearest_entity["h"] if nearest_entity else None,
            "resource": nearest_resource["h"] if nearest_resource else None,
        }
        # One placement position per facing, tile-aligned. A single fixed offset
        # made a belt gap fillable from exactly one standing position.
        # floor, not round: positions are tile centres and banker's rounding
        # would send adjacent placements to the wrong tile half the time.
        tile_x, tile_y = math.floor(position[0]), math.floor(position[1])
        for direction, (dx, dy) in catalog_module.PLACE_OFFSETS.items():
            context[f"at_{direction}"] = [tile_x + dx + 0.5, tile_y + dy + 0.5]
        return context

    def action_masks(self) -> np.ndarray:
        """sb3-contrib's masking protocol.

        Built from the observation alone. A test constructs the mask from a
        policy-visible observation and asserts equality with this, which is what
        makes "no privileged information in masks" checkable rather than
        aspirational.
        """
        context = self._context()
        mask = np.ones(len(self.catalog), dtype=bool)
        for index, template in enumerate(self.catalog.templates):
            if template.requires and context.get(template.requires) is None:
                mask[index] = False
        # A legal no-op always exists.
        mask[self.catalog.wait_index] = True
        return mask

    # ------------------------------------------------------------ gym API

    def prepare_scene(self, episode_index: int) -> str:
        """Choose and install the scene for an episode, returning its digest.

        Kept separate from beginning the episode because the two have
        different lifetimes: a scene is installed once and referenced by hash
        thereafter, while an episode begins many times against the same scene.
        Phase 7's persistent stages advance the goal without reinstalling, and
        anything that restores a mid-episode state needs the same seam.
        """
        families = self._families()
        rng = self.seed_plan.generator_rng(self.branch, episode_index)
        self._family = families[rng.randrange(len(families))]
        blueprint = self.task.generate(self._family, rng)
        return self._install(blueprint)

    def begin_episode(self, digest: str) -> dict:
        """Return the world to an installed scene and start scoring."""
        # Drain anything still in flight before resetting, so a reset never
        # lands on top of a running advance.
        #
        # The profile has to be re-sent on every reset, not just the first: the
        # worker's observation profile is episode state, and a task that
        # declares a non-default profile got the worker's default instead
        # because this call omitted it. That failure is silent -- the wrong
        # profile still produces a well-formed observation the encoder happily
        # reads -- so the task's `observation_profile` was decorative until it
        # was passed here.
        # `action_profile` had the identical bug and it went unnoticed because
        # every current task declares the worker's default: the field was
        # recorded in manifests, resolved into the catalog digest, and never
        # sent. The first task to declare `assisted-v1` would have run under
        # `primitive-v1` while its manifest said otherwise.
        self.session.reset(
            blueprint_hash=digest,
            observation_profile=self.spec_.observation_profile,
            action_profile=self.spec_.action_profile,
        )
        self._observation = self.session.observe().response.result
        self._refresh_truth()
        self.accountant.reset(self._observation, self._truth)
        return self._observation

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._episode_index += 1
        self._steps = 0

        digest = self.prepare_scene(self._episode_index)
        self.begin_episode(digest)

        info = {
            "task": self.spec_.id,
            "task_version": self.spec_.version,
            "layout_family": self._family.name,
            "split": self.split,
            "blueprint": digest,
            "episode_index": self._episode_index,
            "action_mask": self.action_masks(),
        }
        return encoders.encode(self._observation, self._goal_vector()), info

    def _goal_vector(self) -> np.ndarray:
        """Budget fraction, success predicates, then observable landmarks.

        Success predicates are all false until the episode is essentially over,
        so on their own they leave most of this vector constant and tell a
        long-horizon policy nothing about where it is. Landmarks are the
        intermediate conditions a solution passes through. They are evaluated
        against the observation with **empty truth**, so a landmark cannot
        smuggle evaluator state into the policy input; a task declaring a
        truth-dependent landmark fails `test_landmarks_do_not_read_truth`
        rather than silently leaking.
        """
        goal = np.zeros(encoders.GOAL_FEATURES, dtype=np.float32)
        goal[0] = min(1.0, self._steps / max(self.spec_.max_decision_steps, 1))
        cursor = 1
        limit = encoders.GOAL_FEATURES - encoders.GOAL_GEOMETRY_SLOTS
        for predicate in self.spec_.success[: limit - cursor]:
            goal[cursor] = 1.0 if predicate.evaluate(self._observation, self._truth) else 0.0
            cursor += 1
        for predicate in self.spec_.landmarks[: limit - cursor]:
            goal[cursor] = 1.0 if predicate.evaluate(self._observation, {}) else 0.0
            cursor += 1

        # Where the objective is, not merely whether it has been met. The
        # success flags above are zero until the episode is essentially over,
        # which told a policy nothing about which of four identical containers
        # it was scored on. This comes from the *observation*, so it carries no
        # evaluator state: the mod publishes only the markers the task declared
        # public, and decoys are absent from that set by construction.
        published = self._observation.get("goal") or {}
        if published:
            position = (self._observation.get("character") or {}).get("position") or [0.0, 0.0]
            name = next(iter(sorted(published)))
            target = published[name]
            scale = float(max(encoders.LOCAL_V1.radius, 1))
            goal[-3] = float(np.clip((target[0] - position[0]) / scale, -1.0, 1.0))
            goal[-2] = float(np.clip((target[1] - position[1]) / scale, -1.0, 1.0))
            goal[-1] = 1.0
        return goal

    def _succeeded(self) -> bool:
        return all(p.evaluate(self._observation, self._truth) for p in self.spec_.success)

    def _failed(self) -> bool:
        return any(p.evaluate(self._observation, self._truth) for p in self.spec_.failure)

    def step(self, action: int):
        """Take one action by catalog index -- the policy's interface."""
        template = self.catalog.templates[int(action)]
        return self.step_payload(template.bind(self._context()), action_key=template.key)

    def step_payload(self, payload: dict, *, action_key: str | None = None):
        """Take one action by typed payload, with identical accounting.

        The discrete catalog binds `$target` to the *nearest* entity, because a
        discrete index cannot carry an argument. That is the whole of a defect
        seen three ways: a policy alternating `approach_entity_k` between two
        chests, `navigate` solved 99 times in 100 by one skill because its scene
        holds a single addressable entity, and an agent fuelling a mining drill
        four times over while the furnace two tiles away stayed empty. In every
        case the agent could see which thing it wanted and had no way to say so.

        The mod has always accepted addressed actions -- `transfer` takes
        `from` and `to` handles, `navigate` takes a handle -- so this exposes
        an existing capability rather than adding one, and the action profile
        still decides what is permitted: the payload goes through the same
        dispatcher, and a request outside the profile is refused there.

        Not reachable from a policy. `step` is the discrete interface and is
        unchanged; this is for the language-model deliberation profile, which
        the manifest records, so a result produced with addressed actions can
        never be mistaken for one produced without them.
        """
        self._steps += 1
        try:
            timed = self.session.step(payload, ticks=self.spec_.decision_ticks)
        except (InfrastructureFailure, ProtocolError) as exc:
            # A worker crash is an infrastructure failure, never a task outcome
            # (PLAN.md section 2). Report it as truncated and excluded, so the
            # trainer drops the transition instead of learning from it.
            observation = encoders.encode(self._observation, self._goal_vector())
            return (
                observation,
                0.0,
                False,
                True,
                {
                    "infrastructure_failure": str(exc),
                    "excluded_from_metrics": True,
                    "action_mask": self.action_masks(),
                },
            )

        result = timed.response.result or {}
        action_result = result.get("action") or {}
        self._last_action_ok = action_result.get("status") == "completed"
        self._observation = result.get("observation") or self.session.observe().response.result
        # The step response carries truth with it; only fall back to a separate
        # request if an older worker did not send it.
        truth = result.get("truth")
        if truth is None:
            self._refresh_truth()
        else:
            self._truth = truth

        succeeded = self._succeeded()
        # Termination is decided *before* the transition is scored: the
        # potential-based shaping needs to know whether s' is absorbing.
        terminated = succeeded or self._failed()
        components = self.accountant.step(
            self._observation, self._truth, succeeded, terminated=terminated
        )
        reward = RewardAccountant.total(components)

        truncated = (not terminated) and (
            self._steps >= self.spec_.max_decision_steps
            or self._observation.get("tick", 0) >= self.spec_.max_game_ticks
        )
        info: dict[str, Any] = {
            "reward_components": components,
            "success": succeeded,
            "action_key": action_key or payload.get("action"),
            "action_status": action_result.get("status"),
            "action_error": (action_result.get("error") or {}).get("code"),
            "layout_family": self._family.name if self._family else None,
            "action_mask": self.action_masks(),
            "steps": self._steps,
        }
        return (
            encoders.encode(self._observation, self._goal_vector()),
            reward,
            terminated,
            truncated,
            info,
        )

    def close(self) -> None:
        try:
            self.session.close()
        except OSError:
            pass
