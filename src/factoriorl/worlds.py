"""Open worlds: generated maps with no declared scene and no success predicate.

`docs/AGENTIC_ROADMAP-2026-09-10.md` A0.1 asks for an `open_factory` mode "without
changing existing frozen task definitions or holdouts". This module is how that
constraint is met: an open world is **not** a `RegisteredTask`.

Why it must not be a task
-------------------------
Registering it would break four things at once, and each of them exists for a
reason worth keeping:

* `tasks.validate_all` requires at least one `train` family and one `test`
  family, exactly one `sparse_success` reward, a `track` from the declared list,
  and a success predicate that is **False at reset**. An open world has no
  families, no success predicate and no terminal state -- it is not scored, it is
  played.
* `tests/unit/test_holdout.py` and `test_holdout_add_task.py` both assert
  `sorted(holdout["tasks"]) == sorted(all_tasks())`. A new registered task fails
  those until it is frozen.
* Freezing it would move `holdout_v3`'s `content_hash`, and `docs/AUTHORING_TASKS.md`
  is explicit that this orphans every piece of evidence citing the old hash. The
  benchmark's provenance would be disturbed by something that is not a benchmark
  task.
* `manifest.verify` re-derives the catalog digest for `task.id`, so an id that is
  not in the registry would make every open-world run fail verification.

So an open world is a parallel concept resolved *before* the task registry. The
CLI spells it `--task open_factory` because that is the roadmap's interface, and
looks here first.

What one is
-----------
A declared world configuration -- terrain, profiles, catalog, starting inventory
source -- and nothing else. No objective is stated to the environment; the
agent's goal lives in its prompt, where a person can read it.
"""

from __future__ import annotations

from dataclasses import dataclass

from factoriorl.worker_config import TERRAIN_NATURAL

#: Bumped when a change would make two runs of the same id incomparable --
#: different terrain, profiles, catalog or starting inventory. Same contract as
#: `TaskSpec.version`, for the same reason.
OPEN_FACTORY_VERSION = "0.2.0"


@dataclass(frozen=True)
class WorldMode:
    """One playable world that is not a scored task."""

    id: str
    version: str
    description: str
    terrain: str
    observation_profile: str
    action_profile: str
    catalog: str
    decision_ticks: int
    #: Ceiling on primitive steps. Not a scoring budget -- the real limits on an
    #: open run are the wall clock and the spend cap -- but an unbounded loop
    #: with a broken deadline would run until the machine was turned off.
    max_decision_steps: int
    chart_radius: int
    #: What this world hands the agent beyond the bare observation, composed the
    #: way `factoriorl.assistance` composes a task's. A run that receives help
    #: and records `none` is the failure that module exists to prevent, and an
    #: open world receives two kinds.
    assistance: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "version": self.version,
            "kind": "world",
            "terrain": self.terrain,
            "observation_profile": self.observation_profile,
            "action_profile": self.action_profile,
            "catalog": self.catalog,
            "decision_ticks": self.decision_ticks,
            "max_decision_steps": self.max_decision_steps,
            "chart_radius": self.chart_radius,
            "assistance": list(self.assistance),
            "scored": False,
            "note": (
                "an open world has no success predicate and no layout families; "
                "it is not a benchmark task and no rate measured on it is "
                "comparable to one"
            ),
        }


OPEN_FACTORY = WorldMode(
    id="open_factory",
    version=OPEN_FACTORY_VERSION,
    description=(
        "A fresh natural map with freeplay's ordinary starting items and no "
        "enemies. Build and expand automated production as far as time allows."
    ),
    terrain=TERRAIN_NATURAL,
    # `open-v1`, not `local-v2`: the benchmark profile is `slim` and drops
    # `resources.patches`, which on a generated map is the only way ore more
    # than 12 tiles away is visible at all. `mod/factoriorl/profiles.lua`
    # explains why this is a new profile rather than a change to that one.
    observation_profile="open-v1",
    # These two names look like they should match and do not, which is worth
    # stating once: `action_profile` is the *mod's* capability gate
    # (`mod/factoriorl/profiles.lua`), while `catalog` is the Python-side list of
    # addressable actions.
    #
    # `assisted-v1` is what permits `navigate` at all -- `actions.dispatch`
    # refuses anything outside the profile's list, which is what makes
    # "navigation is absent from primitive-v1" a property of the dispatcher
    # rather than a claim in a document. On a map whose nearest ore is 28 tiles
    # away, a catalog of fixed-direction strides reaches it only by guessing.
    action_profile="assisted-v1",
    # `open-v1`, not `parameterized-v1`: the latter is what `build_line` was
    # measured against and `manifest.verify` re-derives its digest, so adding
    # verbs to it would change what an existing result means.
    catalog="open-v1",
    assistance=("navigation", "bounded-sequences:8"),
    decision_ticks=30,
    max_decision_steps=100_000,
    chart_radius=96,
)

MODES: dict[str, WorldMode] = {OPEN_FACTORY.id: OPEN_FACTORY}


def get(mode_id: str) -> WorldMode:
    if mode_id not in MODES:
        known = ", ".join(sorted(MODES))
        raise KeyError(f"unknown world {mode_id!r}; known worlds: {known}")
    return MODES[mode_id]


def is_world(name: str) -> bool:
    """Whether `name` names an open world rather than a benchmark task."""
    return name in MODES
