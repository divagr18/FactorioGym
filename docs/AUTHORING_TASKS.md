# Authoring a task

R6's gate says a user should be able to "author a small task without editing
internals". This document is honest about how far that is true. Most of a task
is one new file. Two things are not, and one of them was fixed to get here.

## The minimum

Drop a module into `src/factoriorl/tasks/families/`. Discovery is import-based
(`pkgutil.iter_modules` over that package), so nothing lists your module and no
registry needs editing. Every family follows the same four-part shape:

```python
from factoriorl.tasks import RegisteredTask, register
from factoriorl.tasks.spec import (
    Blueprint, EntitySpec, LayoutFamily, Predicate, PredicateKind,
    RewardComponent, RewardKind, TaskSpec,
)

FAMILIES = (
    LayoutFamily("open", "train"),
    LayoutFamily("cluttered", "train"),
    LayoutFamily("screened", "test"),      # a test family is required
)

SPEC = TaskSpec(
    id="fetch_plate",
    version="1.0.0",
    description="Carry one iron plate to the marked chest.",
    layout_families=FAMILIES,
    success=(Predicate(PredicateKind.CONTAINER_HOLDS, marker="dst", item="iron-plate"),),
    rewards=(
        RewardComponent("delivered", RewardKind.SPARSE_SUCCESS, weight=1.0, shaping=False),
        RewardComponent("step", RewardKind.STEP_COST, weight=-0.001, shaping=False),
    ),
    track="logistics",                     # must be one of TRACKS
    max_decision_steps=120,
    catalog_subset=("move_north", "move_east", "move_south", "move_west",
                    "take_iron-plate_5", "give_iron-plate_5"),
)

def generate(family, rng) -> Blueprint:
    ...

def solve(driver) -> None:                 # optional but effectively required
    ...

register(RegisteredTask(spec=SPEC, generate=generate, solve=solve))
```

### `TaskSpec`: six required fields

`id`, `version`, `description`, `layout_families`, `success`, `rewards`.
Everything else has a default, with three worth knowing:

- **`track`** defaults to `"unclassified"`, which is *deliberately invalid* and
  fails validation. Pick from `TRACKS`: movement, logistics, production, repair,
  diagnosis, persistent_operation.
- **`decision_ticks`** defaults to 30 and must be **≥ 30**
  (`catalog.LONG_MOVE_TICKS`). Shorter and a walk is still running when the
  next action executes, so placements land on whatever tile the character
  happened to reach. The invariant holds with zero margin, which is why
  validation asserts it.
- **`version`** is task identity. A bump invalidates the frozen holdout entry on
  its own, even with byte-identical scenes.

### `generate(family, rng) -> Blueprint`

Four rules, each of which has been violated at least once here:

1. **Branch on every `family.name` you declared.** Two families that generate
   identical scenes is a real defect the split audit now catches.
2. **Be a pure function of `rng`.** Use only the `rng` you are given — the
   environment hands you the same, already-advanced generator it used to pick
   the family, and the holdout freezer and coverage tools replay that exact
   draw sequence to reproduce your scenes.
3. **Admit at least 32 distinct scenes per family** over 200 seeds. A
   one-scene generator evaluated 100 times reports a Wilson interval as though
   there were 100 draws.
4. **Emit every marker your predicates name, in every scene.** Marker
   coordinates for 1×1 entities are post-snap tile centres (`+0.5`), and any
   item you hand the agent needs its recipe in `unlock_recipes` or the
   placement guard refuses it.

Keep everything inside `radius`, footprints non-overlapping, and the character
off anything non-walkable. `Blueprint` has `footprint_conflicts()`,
`character_obstructed()` and `out_of_box()` so you can check before running.

### `solve(driver) -> None`

Not strictly required, but a task without one is marked **defective** by
`tools/solvability.py` and fails the Phase 3 gate's 0.99 reference threshold —
so in practice you need it. Return nothing; success is read off
`driver.success`.

`Driver` (`src/factoriorl/tasks/reference.py`) gives you `do(key, **args)`,
`walk_to(target)`, `position`, `marker(name)`, `container(name)`,
`inventory()`, `entities_of_type()`, `nearest_resource()` and
`placement_for(tile)`. It reads evaluator truth, which is allowed — a reference
solver is not an agent. Note `walk_to` is greedy axis-first with a sidestep on
stall, not a pathfinder.

**This is the part that was internals until R6.** `RegisteredTask.solve` had
existed since the class was written and nothing read it; dispatch went through
a hard-coded `SOLVERS` dict in `reference.py`, so declaring your own solver did
nothing and you had to edit that dict. A task's own `solve` now wins, with
`SOLVERS` as the fallback for the ten shipped families whose solvers share
`Driver` helpers.

## What still is not one file

Be aware of these before you start; the first is unavoidable.

1. **The frozen holdout.** Two unit tests assert that every registered task
   appears in `docs/evidence/holdout_v3.json`, so adding a family without
   freezing it breaks the suite. There is a supported mode for exactly this:

   ```
   uv run python tools/freeze_holdout.py --add-task fetch_plate --write
   ```

   It freezes your task under the *file's own* seed plan and copies every
   existing entry verbatim, so no other task's `entry_hash` moves. Commit the
   regenerated file. Your task is deliberately **not** added to the declared
   candidate families.

2. **Prototype footprints**, if you place anything larger than 1×1 that is not
   already known. An unknown prototype is silently treated as 1×1, which
   corrupts overlap and reachability checks — so `ENTITY_TILE_SIZES`
   (`tasks/spec.py`) and `MEASURED_FOOTPRINTS`
   (`tests/unit/test_reset_and_footprints.py`) both need the entry.

3. **The action catalog**, if your task needs an item the primitive catalog
   does not cover. `catalog.py` currently knows `stone-furnace`,
   `transport-belt` and `small-electric-pole` for placement and
   `iron-ore`, `coal`, `iron-plate`, `stone` for transfer. A `catalog_subset`
   naming anything else raises at environment construction — and editing the
   shared list moves the catalog digest for every task.

4. **Reward measurement**, if you want shaping driven by `BUILT`,
   `ANY_WORKING` or `SUSTAINED_OUTPUT`. Those three fall through to `0.0` in
   `rewards._measure`, so the component is declared, recorded in the manifest,
   and pays nothing — silently. `SPARSE_SUCCESS`, `HIGH_WATER`, `POTENTIAL`
   and `STEP_COST` all work.

5. **A new *track***, as opposed to a new task on an existing track, also needs
   `TRACKS` and `gate_r4.py`'s track list.

## Checking your work

Engine-free, in this order:

```
uv run factoriorl tasks validate                     # declarations, budgets, structure
uv run pytest tests/unit tests/contract -q           # ~25 registry-wide sweeps
uv run python tools/generator_diagnostics.py         # distinct scenes, reachability, splits
```

Then with an engine:

```
uv run python tools/solvability.py --tasks fetch_plate
```

Your solver must reach **≥ 0.80 on every split** or the task is defective. The
random floor is reported in *both* action spaces and flagged above 0.10 —
reported rather than enforced, because a family whose skills floor is 0.80
cannot carry an 80% claim even though nothing fails.

`generator_diagnostics.py` is the strictest of these. It wants ≥ 32 distinct
scenes, a goal reachable by a 4-connected walk in every scene, no two families
byte-identical, difficulty parity ≥ 0.60 across train and test on route
descriptors, and structural separation ≤ 0.40 on at least one structural
descriptor. Those thresholds are labelled in the source as a starting point
rather than a principled value.

## Sweeps that will silently skip you

Five tests use hard-coded six-element family tuples, so a new task is exempt
until you add it: `tests/unit/test_tasks_and_rewards.py`,
`tests/unit/test_skills.py`, `tests/engine/test_observation_profiles.py` and
`tools/feasibility_sweep.py`. Nothing fails if you leave them; you just lose
the determinism, structural-soundness, landmark and decision-interval checks
for your family. Add yourself.

## Two silent ceilings

The goal vector is 12 features with 3 geometry slots, so success clauses plus
landmarks beyond **8 total** are truncated without warning, and only one marker
can carry geometry. Neither raises.

## Worked examples

- `families/navigate.py` — the simplest, 48-line generator, 6-line solver.
- `families/deliver.py` — containers, decoys, a screen.
- `families/repair_belt.py` — 167-line generator that reasons about fault count
  visibility and injects the holdout's regime into training so coverage
  containment passes. Read this one before designing a repair task.
