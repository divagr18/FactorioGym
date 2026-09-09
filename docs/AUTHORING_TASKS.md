# Authoring a task

R6's gate says a user should be able to "author a small task without editing
internals". This document is honest about how far that is true, and every
listed trap below was hit by someone following an earlier draft of it.

Most of a task is one new file. Two things are not.

## A worked example that actually runs

Drop this into `src/factoriorl/tasks/families/fetch_plate.py`. Discovery is
import-based (`pkgutil.iter_modules` over that package), so nothing lists your
module and no registry needs editing.

This exact file reaches `reference=1.00` on both splits. Every comment in it
marks something an earlier version of this document got wrong.

```python
"""Carry one iron plate to the marked chest."""

from factoriorl.tasks import RegisteredTask, register
from factoriorl.tasks.spec import (
    Blueprint,
    EntitySpec,
    LayoutFamily,
    Predicate,
    PredicateKind,
    RewardComponent,
    RewardKind,
    TaskSpec,
)

FAMILIES = (
    LayoutFamily("open", "train"),
    LayoutFamily("cluttered", "train"),
    LayoutFamily("screened", "test"),  # a test family is required
)

SPEC = TaskSpec(
    id="fetch_plate",
    version="1.0.0",
    description="Carry one iron plate to the marked chest.",
    layout_families=FAMILIES,
    success=(Predicate(PredicateKind.CONTAINER_HOLDS, marker="dst", item="iron-plate"),),
    rewards=(
        RewardComponent("delivered", RewardKind.SPARSE_SUCCESS, weight=1.0, shaping=False),
        # Required, and easy to miss: every family must carry something payable
        # *before* success, or `test_every_family_has_something_that_pays_before
        # _success` fails with "a policy has nothing to ascend".
        RewardComponent(
            "carried",
            RewardKind.HIGH_WATER,
            weight=0.002,
            cap=0.04,
            predicate=Predicate(PredicateKind.INVENTORY_HOLDS, item="iron-plate"),
        ),
        # Positive weight. `rewards.py` applies -abs(weight); writing a negative
        # here is harmless but every shipped family writes it positive.
        RewardComponent("step", RewardKind.STEP_COST, weight=0.001, shaping=False),
    ),
    track="logistics",  # must be one of TRACKS; the default is invalid on purpose
    max_decision_steps=120,
    # `step_*` is not optional. `Driver.walk_to`'s coarsest stride is ~4.45
    # tiles against an arrival tolerance of 1.4, so a move-only subset never
    # converges and every walk exhausts its budget.
    catalog_subset=(
        "move_north",
        "move_east",
        "move_south",
        "move_west",
        "step_north",
        "step_east",
        "step_south",
        "step_west",
        "take_iron-plate_5",
        "give_iron-plate_5",
        "wait",
    ),
)


def generate(family, rng) -> Blueprint:
    radius = 12
    dst_x = rng.randint(3, 8)
    dst_y = rng.randint(-8, -3)
    entities = [EntitySpec("wooden-chest", (dst_x + 0.5, dst_y + 0.5), marker="dst")]
    if family.name == "cluttered":
        taken = {(dst_x, dst_y)}
        while len(taken) < 5:
            tile = (rng.randint(-8, 8), rng.randint(2, 8))
            if tile in taken:
                continue
            taken.add(tile)
            entities.append(EntitySpec("wooden-chest", (tile[0] + 0.5, tile[1] + 0.5)))
    if family.name == "screened":
        for x in range(-2, 3):
            entities.append(EntitySpec("stone-wall", (x + 0.5, -1.5)))
    return Blueprint(
        radius=radius,
        entities=tuple(entities),
        character_position=(0.5, 0.5),
        # Five, not one, despite the task's name: `give_iron-plate_5` is an
        # all-or-nothing transfer of five and fails `no_items` on a smaller
        # stack. Pick the transfer size and the inventory together.
        character_inventory={"iron-plate": 5},
        markers={"dst": (dst_x + 0.5, dst_y + 0.5)},
        unlock_recipes=("iron-plate",),
    )


def solve(driver) -> None:
    driver.walk_to(driver.marker("dst"))
    driver.do("give_iron-plate_5")


register(RegisteredTask(spec=SPEC, generate=generate, solve=solve))
```

`Blueprint`'s fields are `entities`, `resources`, `character_position`,
`character_inventory`, `markers`, `unlock_recipes`, `radius` — there is no
`character` field. `EntitySpec` takes `name`, `position`, and optionally
`direction`, `force`, `contents`, `recipe`, `marker`. Read
`src/factoriorl/tasks/spec.py` for the rest; this document does not restate the
signatures and will drift if it tries.

### `TaskSpec`: six required fields

`id`, `version`, `description`, `layout_families`, `success`, `rewards`.
Everything else defaults, with three worth knowing:

- **`track`** defaults to `"unclassified"`, which is deliberately invalid and
  fails validation.
- **`decision_ticks`** defaults to 30 and must be **≥ 30**
  (`catalog.LONG_MOVE_TICKS`), or a walk is still running when the next action
  executes and placements land wherever the character reached.
- **`version`** is task identity, and bumping it invalidates the frozen holdout
  entry on its own.

### `generate(family, rng) -> Blueprint`

1. **Branch on every `family.name` you declared.** Two families generating
   identical scenes is a defect the split audit catches.
2. **Be a pure function of `rng`.** The environment hands you the same,
   already-advanced generator it used to pick the family, and the holdout
   freezer replays that draw sequence exactly.
3. **Admit at least 32 distinct scenes per family** over 200 seeds.
4. **Emit every marker your predicates name, in every scene**, at post-snap
   tile centres (`+0.5`) for 1×1 entities, and declare `unlock_recipes` for
   anything you hand the agent.

`tasks validate` does **not** check footprint overlap despite listing
"structure" — `Blueprint.footprint_conflicts()` exists and validation does not
call it. `pytest` catches overlaps two steps later.

### `solve(driver) -> None`

A task without one is marked **defective** by `tools/solvability.py` and fails
the Phase 3 gate's 0.99 threshold, so in practice it is required. Return
nothing; success is read off `driver.success`. `Driver` gives you `do(key,
**args)`, `walk_to(target)`, `marker(name)`, `container(name)`, `inventory()`,
`position`, `entities_of_type()`, `nearest_resource()` and
`placement_for(tile)`. It reads evaluator truth, which is allowed.

**This part was internals until R6.** `RegisteredTask.solve` existed and
nothing read it; dispatch went through a hard-coded `SOLVERS` dict, so
declaring your own solver did nothing. A task's own `solve` now wins.

## Checking your work — do this before freezing anything

Order matters, and an earlier draft got it backwards. Every check below can
send you back to the generator, and changing the generator after freezing means
re-freezing.

```
uv run factoriorl tasks validate                     # declarations and budgets
uv run pytest tests/unit tests/contract -q           # ~25 registry-wide sweeps
uv run python tools/generator_diagnostics.py         # distinct scenes, reachability
uv run python tools/solvability.py --tasks fetch_plate    # needs an engine
```

Your solver must reach **≥ 0.80 on every split**. The random floor is reported
in both action spaces and flagged above 0.10 — reported, not enforced, because
a family whose skills floor is 0.80 cannot carry an 80% claim even though
nothing fails.

**These commands rewrite tracked evidence files.** `pytest` regenerates
`docs/evidence/reward-audit.json` through a subprocess, and the two tools
rewrite their own audits. Expect a dirty tree you did not edit.

## Freezing the holdout — last, and it has a trap

Two files and five tests assert that every registered task appears in
`docs/evidence/holdout_v3.json`, so you cannot skip this.

```
uv run python tools/freeze_holdout.py --add-task fetch_plate --write
```

**`--add-task` refuses once your scenes have changed**, with
`refusing to add a task to a holdout that does not verify`. It verifies the
existing file first, and if you already froze an earlier version of your
generator, all 100 of your entries now differ. The escape is
`--replace-task`, which exists and is documented only in `--help`. You do not
need to bump `version` to hit this — any generator change does it.

**Freezing moves the file's `content_hash`, which orphans evidence that cites
it.** Per-task `entry_hash` values are preserved, but the whole-file hash is
not, and that is the hash `phase4-*.json` evidence records. Adding a task
therefore reproduces, in one command, the same defect `docs/LIMITATIONS.md` §8
describes: seven of eleven published evidence files citing a hash nothing has.
There is no clean answer to this today; know that you are doing it.

## The rest of what is not one file

1. **Prototype footprints**, if you place anything larger than 1×1 that is not
   already known. An unknown prototype is silently treated as 1×1, corrupting
   overlap and reachability checks — `ENTITY_TILE_SIZES` (`tasks/spec.py`) and
   `MEASURED_FOOTPRINTS` (`tests/unit/test_reset_and_footprints.py`).
2. **The action catalog**, if you need an item it does not cover. It knows
   `stone-furnace`, `transport-belt`, `small-electric-pole` for placement and
   `iron-ore`, `coal`, `iron-plate`, `stone` for transfer.
3. **Reward measurement**, if you want shaping on `BUILT`, `ANY_WORKING` or
   `SUSTAINED_OUTPUT`. Those fall through to `0.0` in `rewards._measure`, so
   the component is declared, manifested, and pays nothing, silently.
4. **A new *track***, which also needs `TRACKS` (`tasks/spec.py`) and the track
   list in `src/factoriorl/gate_r4.py`.

## Sweeps that will silently skip you

Hard-coded family tuples in `tests/unit/test_tasks_and_rewards.py`,
`tests/unit/test_skills.py`, `tests/engine/test_observation_profiles.py` and
`tools/feasibility_sweep.py` (a tool, not a test). Nothing fails if you leave
them; you lose the determinism, structural-soundness, landmark and
decision-interval checks for your family.

## Two silent ceilings

The goal vector is 12 features with 3 geometry slots, so success clauses plus
landmarks beyond **8 total** are truncated without warning, and only one marker
can carry geometry. Neither raises.

## Worked examples in the tree

- `families/navigate.py` — simplest: 48-line generator, 6-line solver.
- `families/deliver.py` — containers, decoys, a screen.
- `families/repair_belt.py` — 167 lines that reason about fault-count
  visibility and inject the holdout's regime into training so coverage
  containment passes. Read it before designing a repair task.
