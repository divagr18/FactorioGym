"""An environment for a generated map that is played rather than scored.

`FactorioEnv` does almost everything an open world needs -- observation
encoding, the action mask, the catalog, stepping, production metrics -- and
exactly two things it must not do: build a declared scene, and decide that the
episode has been won. So this subclasses it and replaces those two, rather than
growing a second environment beside it.

Two traps this exists to avoid
------------------------------
**An empty success tuple means instant success.** `FactorioEnv._succeeded` is
``all(p.evaluate(...) for p in self.spec_.success)``, and ``all([])`` is
``True``. A scoreless spec would therefore terminate on its first step, with a
reward and a `success: true` in the record. No registered task can hit this
because `tasks.validate_all` requires a non-empty `success`; an unregistered
spec bypasses that check, so the override below is the thing standing between
this design and a 30-minute run that ends after one action.

**`reset` is destructive.** `FactorioEnv.reset` calls `prepare_scene` and then
`session.reset`, which sweeps neutral-force entities inside the scene box --
and natural ore, trees and rocks are neutral. Reusing it would delete the
resources the agent is meant to find. `session.open_world` sweeps only the
player force, which removes the reference scene the mod paints into every new
save while leaving the map alone.
"""

from __future__ import annotations

import gymnasium as gym

from factoriorl import encoders
from factoriorl.env import FactorioEnv
from factoriorl.production import ProductionMetrics
from factoriorl.seeding import Branch, SeedPlan
from factoriorl.session import WorkerSession
from factoriorl.tasks import RegisteredTask, TaskConfigError
from factoriorl.tasks.spec import TaskSpec
from factoriorl.worlds import WorldMode

#: Ticks a world may run before the environment truncates. Deliberately enormous:
#: the real limits on an open run are the wall clock and the spend cap, both of
#: which are enforced above this layer. This exists only so that a loop with a
#: broken deadline cannot run until the machine is switched off.
MAX_WORLD_TICKS = 1_000_000_000


def _no_scene(family, rng):  # pragma: no cover - never called
    raise TaskConfigError(
        "an open world has no layout families and generates no scene; if this ran, "
        "something called prepare_scene() on a world instead of a task"
    )


def spec_for(mode: WorldMode) -> TaskSpec:
    """A `TaskSpec`-shaped description of a world. **Never registered.**

    `factoriorl.worlds` explains why registration is refused: `validate_all`
    would reject it, and two holdout tests assert that every registered task is
    frozen. This object exists so the environment machinery has the fields it
    reads, not so the world can pretend to be a benchmark task.
    """
    return TaskSpec(
        id=mode.id,
        version=mode.version,
        description=mode.description,
        layout_families=(),
        # Empty, and the reason `_succeeded` is overridden below.
        success=(),
        # No shaping and no sparse terminal reward: nothing here is scored, and
        # a reward signal on an unscored world would be a number with no
        # referent that later analysis could mistake for a result.
        rewards=(),
        max_decision_steps=mode.max_decision_steps,
        max_game_ticks=MAX_WORLD_TICKS,
        catalog=mode.catalog,
        action_profile=mode.action_profile,
        observation_profile=mode.observation_profile,
        decision_ticks=mode.decision_ticks,
        track="construction",
    )


def task_for(mode: WorldMode) -> RegisteredTask:
    return RegisteredTask(spec=spec_for(mode), generate=_no_scene)


class OpenWorldEnv(FactorioEnv):
    """A generated map, initialised without destroying it."""

    def __init__(
        self,
        mode: WorldMode,
        session: WorkerSession,
        *,
        inventory: dict[str, int] | None = None,
        seed: int = 20260910,
        fresh: bool = True,
    ) -> None:
        self.mode = mode
        self.inventory = dict(inventory or {})
        self.fresh = fresh
        super().__init__(
            task_for(mode),
            session,
            SeedPlan(master=seed, run_id=mode.id),
            branch=Branch.TRAIN,
            split="train",
            # No reward components exist, so shaping has nothing to shape. Said
            # explicitly rather than left to default, because "shaping is on" in
            # a manifest for a world with no rewards would read as a mistake.
            shaping=False,
        )
        # Replaces the task-derived metrics `FactorioEnv.__init__` just built.
        # `for_task` reads its item list off the spec's PRODUCED/SUSTAINED_OUTPUT
        # predicates, and `spec_for` declares none -- so an open world's
        # production report was `{}` in every field, for the whole run.
        self.metrics = ProductionMetrics.for_world(mode)

    def _succeeded(self) -> bool:
        """Never. See the module docstring -- `all([])` is `True`.

        An open world has no terminal state. It ends when the wall clock or the
        spend cap says so, and both of those live above the environment.
        """
        return False

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        # gym.Env.reset, not FactorioEnv.reset: the latter is the destructive
        # scene path this class exists to avoid.
        gym.Env.reset(self, seed=seed)
        self._episode_index += 1
        self._steps = 0
        self._window = []
        self._resyncs = []
        self._disrupted = set()
        self._disruptions_applied = []

        opened = self.session.open_world(
            inventory=self.inventory if self.fresh else None,
            observation_profile=self.spec_.observation_profile,
            action_profile=self.spec_.action_profile,
            chart_radius=self.mode.chart_radius,
            fresh=self.fresh,
        )
        result = opened.response.result or {}
        # Every resource patch inside the charted area, surveyed once at
        # creation. Kept on the environment rather than in the observation
        # because patches do not move: it is constant for the run, so it belongs
        # in the cached prompt prefix and not in every turn.
        self.survey = list(result.get("survey") or [])
        self.survey_radius = result.get("survey_radius")
        undelivered = result.get("undelivered") or []
        if undelivered:
            # The same rule a declared scene gets: a world that is not the world
            # that was asked for produces observations describing something else.
            raise TaskConfigError(
                f"{self.spec_.id}: the engine would not accept the declared "
                f"starting inventory: {undelivered}. The character does not hold "
                f"what freeplay would give it, so this is not the start it claims"
            )

        self._observation = self.session.observe().response.result
        self._refresh_truth()
        self.accountant.reset(self._observation, self._truth)
        self.metrics.reset()
        self.metrics.record(self._observation, self._truth)

        info = {
            "world": self.spec_.id,
            "world_version": self.spec_.version,
            "terrain": self.mode.terrain,
            "fresh": self.fresh,
            "starting_inventory": dict(self.inventory) if self.fresh else None,
            "removed_prebuilt_entities": result.get("destroyed"),
            "character_position": result.get("position"),
            "episode_index": self._episode_index,
            "action_mask": self.action_masks(),
            "scored": False,
        }
        return encoders.encode(self._observation, self._goal_vector()), info
