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
from dataclasses import replace
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from factoriorl import catalog as catalog_module
from factoriorl import encoders
from factoriorl import rewards as rewards_module
from factoriorl.errors import FactorioRLError, InfrastructureFailure, ProtocolError
from factoriorl.production import ProductionMetrics
from factoriorl.rewards import RewardAccountant
from factoriorl.seeding import Branch, SeedPlan
from factoriorl.session import WorkerSession
from factoriorl.tasks import MAX_ADVANCE_TICKS, RegisteredTask, TaskConfigError
from factoriorl.tasks.spec import LayoutFamily


def blueprint_digest(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


#: How far a placement candidate may be from the character. The character's
#: build distance is 10 and `character.reach` is not in `local-v2`, so this is
#: a deliberate constant well inside it: a candidate is never offered only to
#: be refused for range. The runtime still enforces the real distance.
PLACEMENT_RADIUS = 5

#: How many destinations to offer. An enumerated domain is rendered into the
#: prompt, so an unbounded one would bury the rest of the observation; the
#: nearest few dozen are what an agent can act on anyway.
DESTINATION_CAP = 24

#: The transfer counts a policy may name. Unbounded counts would make the
#: argument domain unbounded; the runtime accepts 1..10000.
TRANSFER_AMOUNTS = (1, 5, 20)

#: Production samples kept per episode. One per decision, and the longest
#: budget is 400 decisions, so this holds a whole episode with room over --
#: a predicate's window must never fall off the front of the buffer.
WINDOW_SAMPLES = 512


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
        start_curriculum: float = 0.0,
    ) -> None:
        self.task = task
        self.spec_ = task.spec
        self.session = session
        self.seed_plan = seed_plan
        self.branch = branch
        self.split = split
        # Exploring starts, and a *run* setting rather than a task one. It
        # rewrites the character's start on the training stream only, so it
        # cannot alter a single test-split scene -- which is precisely why it
        # does not belong in `TaskSpec.to_dict`. Putting it there bumped the
        # task version and invalidated two frozen holdouts for a change no
        # evaluated episode can observe. Here it is recorded in the run config
        # instead, and curriculum-on and curriculum-off can be compared against
        # the same frozen holdout.
        self.start_curriculum = float(start_curriculum)
        self.catalog = catalog_module.resolve(self.spec_.catalog, self.spec_.catalog_subset)
        self.action_space = spaces.Discrete(len(self.catalog))
        self.observation_space = encoders.observation_space()
        self.accountant = RewardAccountant(
            self.spec_.rewards,
            shaping_enabled=shaping,
            gamma=self.spec_.gamma or rewards_module.GAMMA,
        )
        #: Benchmark metrics, in simulated ticks. Deliberately separate from
        #: the accountant: these describe the run and must not be able to
        #: change what the agent is paid.
        self.metrics = ProductionMetrics.for_task(self.spec_)

        self._episode_index = -1
        self._steps = 0
        #: (tick, produced) samples for the episode. Cleared with it, or a
        #: sustained objective would be satisfied by the previous episode.
        self._window: list[tuple[int, dict]] = []
        #: Every out-of-band refresh, with the tick and the declared reason.
        #: An evaluator write is not a decision, so it must not be invisible in
        #: the record either.
        self._resyncs: list[dict] = []
        #: Which declared disruptions have fired this episode, by index,
        #: and what the engine reported for each.
        self._disrupted: set[int] = set()
        self._disruptions_applied: list[dict] = []
        self._installed: set[str] = set()
        self._observation: dict = {}
        self._truth: dict = {}
        self._family: LayoutFamily | None = None

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
        payload = blueprint.to_dict(
            public_markers=self.spec_.public_markers,
            extra_tracked_items=self.spec_.extra_tracked_items,
        )
        digest = blueprint_digest(payload)
        if digest not in self._installed:
            self.session.define_scenario(payload, digest)
            self._installed.add(digest)
        return digest

    def _refresh_truth(self) -> None:
        self._truth = self.session.truth().response.result or {}
        self._record_window()

    def _record_window(self) -> None:
        """Keep a bounded production history, and expose it as `truth["window"]`.

        `PRODUCED` is a monotone counter with no timestamp, so nothing could
        distinguish sustained output from a burst that stopped. Sampling
        `(tick, produced)` per step and handing the history to the predicate
        keeps `Predicate.evaluate` a pure function of its inputs -- so a
        sustained objective is testable with no engine.

        Bounded by count rather than by window length: a predicate declares its
        own `over_ticks`, and the buffer must outlive the longest one a task
        might ask for.
        """
        tick = int(self._observation.get("tick") or 0)
        produced = dict(self._truth.get("produced") or {})
        if self._window and self._window[-1][0] == tick:
            self._window[-1] = (tick, produced)
        else:
            self._window.append((tick, produced))
            if len(self._window) > WINDOW_SAMPLES:
                del self._window[0]
        self._truth = {**self._truth, "window": list(self._window)}

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

    def argument_domains(self) -> dict[str, list]:
        """Legal values for each declared argument, from the observation alone.

        The same discipline the masks follow: every domain is reconstructible
        from policy-visible data, so a parameterized action cannot smuggle
        evaluator state through its argument list. Domains a task cannot yet
        observe -- recipes, technologies -- come back empty rather than guessed,
        because an argument the policy cannot see the values of is not
        selectable.
        """
        observation = self._observation
        origin = (observation.get("character") or {}).get("position") or [0.0, 0.0]

        targets = [
            str(record["h"]) for record in (observation.get("entities") or []) if record.get("h")
        ]
        targets += [
            str(tile["h"])
            for tile in ((observation.get("resources") or {}).get("tiles") or [])
            if tile.get("h")
        ]

        return {
            "targets": targets,
            "placements": self._placement_candidates(origin, observation),
            "destinations": self._destinations(origin, observation),
            "directions": list(catalog_module.DIRECTIONS),
            # Only what the agent is holding: proposing an item it does not
            # have would be refused by the runtime for `no_items`, and the
            # inventory is in the observation.
            "items": sorted(
                item for item, count in (observation.get("inventory") or {}).items() if count
            ),
            "amounts": list(TRANSFER_AMOUNTS),
            # Now observable (local-v2 v4), so `craft_recipe` and
            # `set_recipe_at` stop being permanently masked.
            "recipes": list(observation.get("recipes") or []),
            # Researchable *now*: prerequisites met, not already researched.
            # This was `[]` until an open world published the list, which made
            # `research` permanently masked -- an argument whose domain the
            # policy cannot see is not selectable, so the verb existed in the
            # mod's action matrix and could never be chosen.
            "technologies": list(observation.get("researchable") or []),
            "conditions": list(catalog_module.WAIT_CONDITIONS),
            "durations": list(catalog_module.WAIT_SECONDS),
            "requests": [
                str(entry["request_id"])
                for entry in (observation.get("inflight") or [])
                if isinstance(entry, dict) and entry.get("request_id")
            ],
        }

    def _destinations(self, origin, observation) -> list[list[float]]:
        """Somewhere worth walking to, from what the agent can currently see.

        Separate from `placements` because the two want opposite things: a
        placement must be inside build distance or it is refused, and a
        destination is only interesting when it is *outside* arm's reach. The
        generated map this serves puts the nearest iron ore 28 tiles from spawn,
        so a domain capped at `PLACEMENT_RADIUS` could not name it.

        Everything here comes from the observation -- resource patches the
        sensor aggregated, visible entities, remembered ones -- so it proposes
        no destination the agent has not seen. Unexplored ground stays
        unnameable, which is what keeps navigation assistance rather than
        knowledge.
        """
        seen: list[list[float]] = []
        resources = observation.get("resources") or {}
        for tile in resources.get("tiles") or []:
            if tile.get("p"):
                seen.append([float(tile["p"][0]), float(tile["p"][1])])
        patches = resources.get("patches") or {}
        rows = patches.values() if isinstance(patches, dict) else patches
        for patch in rows:
            nearest = isinstance(patch, dict) and patch.get("nearest")
            if nearest:
                seen.append([float(nearest[0]), float(nearest[1])])
        for record in (observation.get("entities") or []) + (observation.get("remembered") or []):
            if record.get("p"):
                seen.append([float(record["p"][0]), float(record["p"][1])])

        # Rounded to tile centres and de-duplicated, because a route ends on one
        # and two destinations inside the same tile are the same destination.
        unique: dict[tuple[float, float], list[float]] = {}
        for point in seen:
            tile = (math.floor(point[0]) + 0.5, math.floor(point[1]) + 0.5)
            unique.setdefault(tile, [tile[0], tile[1]])
        ordered = sorted(
            unique.values(),
            key=lambda p: (p[0] - float(origin[0])) ** 2 + (p[1] - float(origin[1])) ** 2,
        )
        return ordered[:DESTINATION_CAP]

    def _placement_candidates(self, origin, observation) -> list[list[float]]:
        """Tile centres near the character with nothing visible standing there.

        Bounded to `PLACEMENT_RADIUS`, which is inside the character's build
        distance, so a candidate is never proposed only to be refused for
        range. Occupancy and water come from the observation; the runtime still
        enforces the real `can_place_entity`, so this narrows the choice rather
        than deciding it.
        """
        occupied = {
            (math.floor(record["p"][0]), math.floor(record["p"][1]))
            for record in (observation.get("entities") or [])
            if record.get("p")
        }
        occupied |= {
            (math.floor(point[0]), math.floor(point[1]))
            for point in ((observation.get("terrain") or {}).get("blocked") or [])
        }
        here = (math.floor(origin[0]), math.floor(origin[1]))
        # The character's own tile, which `can_place_entity` refuses and this
        # domain was offering anyway.
        #
        # `here` was computed and never read. The sweep deliberately excludes
        # the agent's own body from `entities` (`sensor.sweep` skips
        # `storage.frrl_character`), so its tile never entered `occupied` and
        # `dx = dy = 0` sat in the candidate list like any other.
        #
        # Measured: `can_place_entity` for a stone furnace on the character's
        # position is **false**, and **true** four tiles away. Every paid run so
        # far opened by walking to a tile and then trying to build on it, and
        # every one of them was refused `collision` for a reason nothing in the
        # prompt could explain -- the map even draws `@` there, which reads as
        # "you are here", not as "this tile is unavailable".
        occupied.add(here)
        candidates = []
        for dx in range(-PLACEMENT_RADIUS, PLACEMENT_RADIUS + 1):
            for dy in range(-PLACEMENT_RADIUS, PLACEMENT_RADIUS + 1):
                tile = (here[0] + dx, here[1] + dy)
                if tile in occupied:
                    continue
                candidates.append([tile[0] + 0.5, tile[1] + 0.5])
        return candidates

    def step_arguments(self, action: int, arguments: dict):
        """Take a parameterized action, validating each argument's domain.

        Refused here rather than at the engine when the value is not in a
        domain the policy could see, so an out-of-domain argument is a client
        error with a name, not a generic rejection.
        """
        template = self.catalog.templates[int(action)]
        domains = self.argument_domains()
        for name in template.arguments:
            if name not in arguments:
                raise ValueError(f"{template.key} needs argument {name!r}")
            domain_name = catalog_module.ARGUMENT_DOMAINS.get(name)
            legal = domains.get(domain_name) if domain_name else None
            if legal is not None and arguments[name] not in legal:
                raise ValueError(
                    f"{template.key}: {name}={arguments[name]!r} is not in "
                    f"domain {domain_name!r} ({len(legal)} values)"
                )
        if template.key in catalog_module.ENV_HANDLED:
            # Answered here rather than forwarded. Both verbs consume the step
            # they are given -- looking around and waiting take time -- but
            # neither changes the world, so both ride on the mod's `wait`, which
            # is exactly a no-op that consumes a step. Their arguments are
            # validated above like any other, and then never reach the wire: the
            # mod declares `wait` with an empty payload and would refuse them.
            return self._env_action(template, arguments)
        payload = template.bind(self._context(), arguments)
        return self.step_payload(payload, action_key=template.key)

    def _wait_once(self, action_key: str):
        """One `wait`, which is a no-op that spends the step's game time."""
        index = self.catalog.wait_index
        return self.step_payload(
            self.catalog.templates[index].bind(self._context()), action_key=action_key
        )

    def _inventory_total(self) -> int:
        return sum((self._observation.get("inventory") or {}).values())

    def _coarse_signature(self) -> tuple:
        """What must change before `world_changes` is satisfied.

        Observation-only and deliberately coarse: entity identity and working
        state plus the inventory. A finer signature would fire on the tick
        counter and make the condition meaningless.
        """
        entities = tuple(
            sorted(
                (str(e.get("h")), str(e.get("st")), bool(e.get("working")))
                for e in (self._observation.get("entities") or [])
            )
        )
        return entities, tuple(sorted((self._observation.get("inventory") or {}).items()))

    def _condition_met(self, until: str, before: tuple, before_items: int) -> bool:
        if until == "inventory_grows":
            return self._inventory_total() > before_items
        if until == "nothing_in_flight":
            return not (self._observation.get("inflight") or [])
        return self._coarse_signature() != before

    def _env_action(self, template, arguments: dict):
        """`inspect` and `wait_for`, neither of which the mod implements.

        `wait_for` repeats the underlying `wait` until its condition holds or
        its bound is spent. One plain `wait` buys `decision_ticks` of game time
        -- half a second at the default -- and a stone furnace needs 3.2 seconds
        per plate, so waiting for a plate one decision at a time costs six model
        calls. This spends game time without spending decisions.
        """
        if template.key == catalog_module.INSPECT:
            result = self._wait_once(template.key)
            observation, reward, terminated, truncated, info = result
            record = {
                **self._describe_handle(str(arguments["handle"])),
                # Stamped with the step it was taken at, because the summary
                # carries it forward into later prompts and an inspection from
                # forty steps ago must not read as a current sighting.
                "at_step": self._steps,
            }
            # Kept on the observation as well as in `info` so the *next* prompt
            # can show it. An inspection the model cannot read afterwards is a
            # step spent on nothing.
            self._observation["inspected"] = record
            info = {**info, "inspected": record}
            return observation, reward, terminated, truncated, info

        until = str(arguments["until"])
        seconds = float(arguments["seconds"])
        # Ticks are the unit the environment actually spends, and one step is
        # `decision_ticks` of them. Rounded up so a one-second wait is never
        # zero steps, which would make the verb a silent no-op.
        per_step = max(self.spec_.decision_ticks, 1) / 60.0
        steps = max(int(math.ceil(seconds / per_step)), 1)

        before, before_items = self._coarse_signature(), self._inventory_total()
        waited, met = 0, False
        result = None
        for _ in range(steps):
            result = self._wait_once(template.key)
            waited += 1
            if result[2] or result[3]:
                break
            if self._condition_met(until, before, before_items):
                met = True
                break
        observation, reward, terminated, truncated, info = result
        info = {
            **info,
            "waited_steps": waited,
            "waited_condition": until,
            # Reported either way: a wait that timed out looks identical to one
            # that succeeded unless the record says which it was, and an agent
            # that cannot tell will wait again.
            "condition_met": met,
        }
        return observation, reward, terminated, truncated, info

    def _describe_handle(self, handle: str) -> dict:
        """Everything the observation holds about one entity.

        The prompt shows the twelve nearest (`summary.MAX_ENTITIES_SHOWN`), so
        on a map with anything built on it most of what exists is not in the
        prompt. Remembered entities are searched too, and are returned with the
        `age` the observation gave them -- a remembered entity is not a current
        sighting and must not be presented as one.
        """
        for entity in self._observation.get("entities") or []:
            if str(entity.get("h")) == handle:
                return {"seen": "now", **entity}
        for tile in (self._observation.get("resources") or {}).get("tiles") or []:
            if str(tile.get("h")) == handle:
                return {"seen": "now", "kind": "resource", **tile}
        for entity in self._observation.get("remembered") or []:
            if str(entity.get("h")) == handle:
                return {"seen": "remembered", **entity}
        return {"seen": "unknown", "handle": handle, "detail": "no such handle in view or memory"}

    def action_masks(self) -> np.ndarray:
        """sb3-contrib's masking protocol.

        Built from the observation alone. A test constructs the mask from a
        policy-visible observation and asserts equality with this, which is what
        makes "no privileged information in masks" checkable rather than
        aspirational.
        """
        context = self._context()
        mask = np.ones(len(self.catalog), dtype=bool)
        # Computed once: a parameterized catalog asks for the same domains on
        # every template.
        domains = (
            self.argument_domains() if any(t.parameterized for t in self.catalog.templates) else {}
        )
        for index, template in enumerate(self.catalog.templates):
            if template.requires and context.get(template.requires) is None:
                mask[index] = False
                continue
            # An action one of whose arguments has no legal value cannot be
            # issued. Masking it here is the generalisation of the wait-index
            # fallback: sb3's masked categorical turns an all-false sub-mask
            # into a *uniform* distribution over illegal values, silently, so
            # the empty domain has to be handled before it ever gets there.
            for name in template.arguments:
                domain = catalog_module.ARGUMENT_DOMAINS.get(name)
                if domain is not None and not domains.get(domain):
                    mask[index] = False
                    break
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
        blueprint = self._apply_start_curriculum(blueprint, rng)
        return self._install(blueprint)

    def _apply_start_curriculum(self, blueprint, rng):
        """Exploring starts, training stream only (Sutton & Barto §5.3, p. 79).

        Gated on `Branch.TRAIN`, not on the split. The unfamiliar-seed row is
        measured with `Branch.EVAL` over the *training* split, so gating on
        `split == "train"` would move the curriculum into a reported number and
        inflate it. Both conditions are required and the branch is the one that
        matters.

        Returns the blueprint unchanged when the run declares no curriculum, so
        the default path and every frozen holdout digest are byte-identical to
        before.
        """
        fraction = self.start_curriculum
        if not fraction:
            return blueprint
        if self.branch is not Branch.TRAIN or self.split != "train":
            return blueprint
        marker = self.spec_.focus_marker
        target = (blueprint.markers or {}).get(marker) if marker else None
        if target is None:
            return blueprint
        # Drawn from the same generator stream, so a scene is still a pure
        # function of (branch, episode_index) and stays reproducible.
        if rng.random() >= fraction:
            return blueprint

        # Compared as *tiles*, not as coordinates. Entities are declared
        # pre-snap at (x, y) while a 1x1 entity occupies the tile centred on
        # (x+0.5, y+0.5), and markers are declared post-snap -- so comparing
        # the raw floats could never match and the guard silently admitted
        # every candidate. The first version of this did exactly that, and the
        # test that was meant to catch it built its entities at post-snap
        # coordinates, so it validated the fixture rather than the code.
        occupied = {
            (math.floor(e.position[0]), math.floor(e.position[1])) for e in blueprint.entities
        }
        for dx, dy in ((0.0, 2.0), (0.0, -2.0), (2.0, 0.0), (-2.0, 0.0), (0.0, 3.0), (0.0, -3.0)):
            candidate = (target[0] + dx, target[1] + dy)
            if (math.floor(candidate[0]), math.floor(candidate[1])) not in occupied:
                return replace(blueprint, character_position=candidate)
        return blueprint

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
        installed = self.session.reset(
            blueprint_hash=digest,
            observation_profile=self.spec_.observation_profile,
            action_profile=self.spec_.action_profile,
        )
        # The scene the engine built has to be the scene the task declared.
        # `LuaEntity.insert` chooses an inventory by what an item is *for*, and
        # a furnace's own product belongs to neither its source nor its fuel
        # slot -- so `diagnose_line` declared a hundred-plate output jam in
        # every scene, the engine accepted none of them, and the fault the
        # family was built around did not exist. It survived four solvability
        # runs and a test that asserted the declaration rather than the
        # installation. Raising is right: a scene that is not the declared one
        # produces numbers that describe no task.
        undelivered = (installed.response.result or {}).get("undelivered") or []
        if undelivered:
            raise TaskConfigError(
                f"{self.spec_.id}: the engine would not accept contents this scene "
                f"declares: {undelivered}. Each entry is "
                "prototype|item|placed/declared. The installed scene is not the "
                "declared one, so anything measured on it describes no task."
            )
        self._observation = self.session.observe().response.result
        self._refresh_truth()
        self.accountant.reset(self._observation, self._truth)
        self.metrics.reset()
        self.metrics.record(self._observation, self._truth)
        return self._observation

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._episode_index += 1
        self._steps = 0
        #: (tick, produced) samples for the episode. Cleared with it, or a
        #: sustained objective would be satisfied by the previous episode.
        self._window: list[tuple[int, dict]] = []
        # Cleared for the same reason: an intervention in the previous episode
        # is not part of this one's record. Disruptions too, or a task
        # declaring one would fire it in the first episode only.
        self._resyncs = []
        self._disrupted = set()
        self._disruptions_applied = []

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
            target = self._focus_target(published, position)
            scale = float(max(encoders.LOCAL_V1.radius, 1))
            goal[-3] = float(np.clip((target[0] - position[0]) / scale, -1.0, 1.0))
            goal[-2] = float(np.clip((target[1] - position[1]) / scale, -1.0, 1.0))
            goal[-1] = 1.0
        return goal

    def _focus_target(self, published: dict, position) -> list:
        """The next fault to fix, not merely the first one declared.

        Only three slots carry geometry, so exactly one marker can be pointed
        at. A task with two faults used to publish both and aim at whichever
        the spec listed first -- `min(gaps)` for `repair_belt` -- so the second
        fault had no coordinate anywhere the policy could read, and a run that
        had perfectly learned the task scored zero on a two-fault holdout by
        construction.

        The nearest *unrepaired* declared fault is chosen instead, which also
        makes the vector advance as faults are fixed rather than continuing to
        name a tile that is already done. Occupancy is read from the
        observation's own entity list, never from truth, so this leaks nothing:
        a marker is "repaired" exactly when the policy can see something
        standing on it.
        """
        faults = (
            [n for n in self.spec_.extra_public_markers if n in published]
            if self.spec_.focus_policy == "nearest_unrepaired"
            else []
        )
        if faults:
            occupied = {
                (math.floor(e["p"][0]), math.floor(e["p"][1]))
                for e in (self._observation.get("entities") or [])
                if e.get("p")
            }
            open_faults = [
                published[n]
                for n in faults
                if (math.floor(published[n][0]), math.floor(published[n][1])) not in occupied
            ]
            if open_faults:
                return min(
                    open_faults,
                    key=lambda t: (t[0] - position[0]) ** 2 + (t[1] - position[1]) ** 2,
                )
            # Every declared fault is filled; keep naming the last one rather
            # than dropping to an unrelated landmark mid-episode.
            return published[faults[-1]]
        name = self.spec_.focus_marker
        if name not in published:
            name = next(iter(sorted(published)))
        return published[name]

    def _succeeded(self) -> bool:
        return all(p.evaluate(self._observation, self._truth) for p in self.spec_.success)

    def _failed(self) -> bool:
        return any(p.evaluate(self._observation, self._truth) for p in self.spec_.failure)

    def step(self, action: int):
        """Take one action by catalog index -- the policy's interface."""
        template = self.catalog.templates[int(action)]
        return self.step_payload(template.bind(self._context()), action_key=template.key)

    def _apply_due_disruptions(self) -> None:
        """Fire any declared disruption whose tick the world has reached.

        Called from the one place every observation refresh goes through, so a
        disruption cannot be skipped by a step that advances past its tick: the
        comparison is `at_tick <= now`, not equality. `plate_line`'s outage was
        fired by a driver at a hard-coded point instead, which is why nothing
        in that run's trace said what had been done to the world.

        The refresh afterwards is `resync`, R4.1's path: the disruption is an
        evaluator-side write the env issued no step for, so the cached
        observation is stale until it re-reads. Without it the first
        post-disruption prompt would describe the world as it was before --
        measured at 3,600 ticks of staleness in the Phase 5 demonstration.

        Announces nothing else. No marker, no event: R4.3 asks for a
        disruption detectable only by monitoring, and status, fuel and output
        are already in the observation.
        """
        if not self.spec_.disruptions:
            return
        now = int(self._observation.get("tick") or 0)
        for index, disruption in enumerate(self.spec_.disruptions):
            if index in self._disrupted or disruption.at_tick > now:
                continue
            self._disrupted.add(index)
            timed = self.session.disrupt(disruption.kind, disruption.targets)
            applied = timed.response.result or {}
            self._disruptions_applied.append(
                {
                    "index": index,
                    "declared_at_tick": disruption.at_tick,
                    "applied_at_tick": int(applied.get("tick") or now),
                    "kind": disruption.kind,
                    "targets": list(disruption.targets),
                    "touched": applied.get("touched") or [],
                    # A disruption that matched nothing is a defect in the
                    # task, not a quieter disruption, so the trace carries the
                    # engine's own answer rather than an assumption.
                    "ok": bool(timed.response.ok),
                    "error": None if timed.response.ok else str(timed.response.error),
                }
            )
            self.resync(f"declared disruption {disruption.kind} on {','.join(disruption.targets)}")

    def _adopt(self, result: dict) -> None:
        """Take a settled step response as the env's current view of the world.

        Extracted from `step_payload` because it was the one place the cache was
        refreshed, and it was reachable only by issuing an env step. Anything
        that advanced time or wrote to the world another way -- the Phase 5
        demonstration driver did both -- left `_observation`, `_truth`, the
        production window and the metrics describing a world that no longer
        existed. Measured: that driver's first post-outage prompt said "game
        tick 450" while the world was at 4050, and reported a drill holding 48
        coal and `working` at the moment it was empty.
        """
        self._observation = result.get("observation") or self.session.observe().response.result
        # The step response carries truth with it; only fall back to a separate
        # request if an older worker did not send it.
        truth = result.get("truth")
        if truth is None:
            self._refresh_truth()
        else:
            self._truth = truth
            self._record_window()
        self.metrics.record(self._observation, self._truth)
        self._apply_due_disruptions()

    def advance(self, ticks: int) -> int:
        """Advance game time without taking a task action, staying in sync.

        Two things make this the honest way to do it:

        **It goes through `session.step`, not `session.advance`.** Only a
        settled *step* carries `observation` and `truth` back (`runtime.lua`
        attaches them for `request_type == "step"` and nothing else), so
        `advance` would return a tick count and leave the env blind. The
        demonstration driver already called `session.step` and discarded
        exactly the pair this adopts.

        **It advances in `decision_ticks` chunks and adopts each one.** A single
        3,600-tick jump would leave the production history with two samples
        3,630 ticks apart, and `Predicate.window_sampled` refuses those -- for
        the good reason that such a history cannot place its own output in
        time. Chunking keeps the window *sampled*, so a measurement window that
        spans an external advance still means a rate.

        Returns the ticks actually advanced. Does not count decision steps: no
        action was taken, and charging the task budget for evaluator time would
        make an intervention shorten the episode.
        """
        if ticks <= 0:
            return 0
        chunk = max(1, min(self.spec_.decision_ticks, MAX_ADVANCE_TICKS))
        advanced = 0
        wait = self.catalog.templates[self.catalog.wait_index]
        payload = wait.bind(self._context())
        while advanced < ticks:
            step = min(chunk, ticks - advanced)
            timed = self.session.step(payload, ticks=step)
            self._adopt(timed.response.result or {})
            advanced += step
        return advanced

    def resync(self, reason: str) -> None:
        """Re-read the world after something outside the env changed it.

        For an evaluator-side write -- a declared disruption, a scripted
        intervention -- which the env cannot see because it issued no request.
        `reason` is recorded so a trace says why the cache was refreshed
        outside a step rather than leaving it to be inferred.
        """
        self._observation = self.session.observe().response.result or {}
        self._refresh_truth()
        self.metrics.record(self._observation, self._truth)
        self._resyncs.append({"tick": int(self._observation.get("tick") or 0), "reason": reason})

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
            # (PLAN.md section 2).
            #
            # This comment used to claim the trainer "drops the transition
            # instead of learning from it". It did not. `excluded_from_metrics`
            # was read only by reporting code, while `vecenv.step_wait`
            # synthesised `TimeLimit.truncated` for this return -- making it
            # indistinguishable from a legitimate time limit -- so MaskablePPO
            # bootstrapped `gamma * V(stale observation)` into the reward and
            # added the transition to the rollout buffer unconditionally.
            #
            # `infrastructure_failure` is now the load-bearing key: `vecenv`
            # refuses to dress it as a truncation, and `RolloutGuard` aborts
            # collection before the next optimizer update.
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
        self._adopt(result)
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
            # Duration of this transition in primitive steps (R1.3).
            "primitive_steps": 1,
        }
        if terminated or truncated:
            info["production"] = self.metrics.report()
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
