# FactorioRL current status

**Maintained entrypoint.** `docs/LEDGER.md` is history; `HANDOFF.md` is a 2026-09-07 snapshot
and carries its own staleness banner. Direction comes from
[`docs/DEVELOPMENT_REDIRECTION.md`](DEVELOPMENT_REDIRECTION.md); this file is day-to-day
state. Read this before treating any older summary as current fact.

**Last reconciled:** 2026-09-09, after R3.1 (`623f7dd`).

---

## 1. Live now

**Nothing is running.** Both machines idle.

Results against the live holdout (`holdout_v3`, now `ff22dedd` -- it was `4227eb56`
when these were measured; `build_line` was added and then re-frozen, and every
existing per-task `entry_hash` is unchanged, so the rows below still describe the
scenes they were measured on. See §3.3):

| task | held-out greedy / stochastic | floor | training |
|---|---|---|---|
`restore_power` v1.6.0 + curriculum, 50k | **0.26 / 0.66** `[0.563,0.745]` | 0.00 | 668/908 |
`restore_power` v1.6.0, 25k | 0.00 / 0.37 `[0.282,0.468]` | 0.02 | 101/227 |
`repair_belt` v1.6.0 + curriculum, 50k | 0.00 / 0.06 | 0.00 | 7/221 |

`restore_power`'s best is the strongest held-out number any repair family has
produced, and it still misses 0.80. **Two things changed between its 0.37 and its
0.66** -- the curriculum and 25k → 50k steps -- so per R5.2 neither gets the credit.

Three caveats that travel with all of these:

- **Greedy and stochastic disagree by a wide margin on every repair-family run**,
  with non-overlapping intervals. Read greedily, two of the three say "did not
  learn". Both arms score identical episodes; neither was picked after the fact.
- **One seed each.** `restore_power` at v1.5.0 gave 0.77 / 0.00 / 0.00 across three
  seeds, so a single cell says little. Per book-synthesis §10, re-evaluating one
  checkpoint on many scenes does not estimate training instability.
- **All measured with the goal-focus assistance on** (`aeae6a4` names it).

## 2. Filename warning — "v3" in an evidence filename is **not** `holdout_v3`

`phase4-release-v3-*.json` are the **third revision of that evidence**. All four cite
**`holdout_v2`** (`df72fe29…`). The live holdout is `holdout_v3` (`4227eb56…`).

| evidence file | holdout it actually cites |
|---|---|
`phase4-release-v3-deliver.json` | `holdout_v2` |
`phase4-release-v3-repair_belt.json` | `holdout_v2` |
`phase4-release-v3-restore_power.json` | `holdout_v2` |
`phase4-release-v3-restore_power-seeds23.json` | `holdout_v2` |
`repair_belt-curriculum-holdout_v3.json` | `holdout_v3` ← the only one |

Do not infer a holdout from a filename. Read `holdout.id` and `holdout.task_entry_hash`.

---

## 3. Versions

### 3.1 Task families

| id | version | gamma | steps | layout families (train / val / **test**) | published markers |
|---|---|---|---|---|---|
`build_line` | **1.1.0** | unset | 600 | `square_patch` / `offset_patch` / **`narrow_patch`** | `patch` |
`deliver` | 1.3.0 | unset | 120 | `two_chest`,`decoy_chest` / `stacked_depot` / **`screened_depot`** | `dst` |
`mine_smelt` | 1.2.0 | unset | 400 | `near_patch`,`fuelled_furnace` / `split_patch` / **`screened_patch`** | — |
`navigate` | 1.2.0 | 0.997 | 150 | `open`,`wall_corridor` / `pillar_field` / **`wall_arc`** | `goal` |
`plate_line` | 1.1.0 | unset | 400 | `commissioning` / `commissioning_far` / **`commissioning_walled`** | — |
`repair_belt` | **1.6.0** | 0.999 | 300 | `gap`,`gap_far` / `misrotation` / **`double_gap`** | `sink`,`gap`,`gap2` |
`restore_power` | **1.6.0** | 0.999 | 250 | `pole_gap`,`pole_gap_far` / `two_gaps` / **`gap_near_drill`** | `drill`,`gap`,`gap2` |
`supply_furnace` | 1.2.0 | unset | 300 | `linear_row`,`two_furnaces` / `far_ore` / **`shared_input`** | — |

All eight: observation `local-v2`, deliberation `flat-v1`, `decision_ticks` 30. Seven
use action profile `primitive-v1`; **`build_line` uses `parameterized-v1`**, and it is
the only task whose success is a conjunction (both machines `BUILT` **and**
`SUSTAINED_OUTPUT` of 10 plates per 3600 ticks, not before tick 7200).

### 3.2 Stack

| item | value |
|---|---|
Engine | Factorio 2.0.60 build 83512 win64 |
Mod | 0.1.0, source digest `8fb6913497f291c9` (readable only from run manifests) |
Protocol | 2 |
Observation profiles | `local-v1`, **`local-v2`** (radius 32, entity_cap 48, grid 65×65) |
Action catalog | `primitive-v1`, 46 templates; **`parameterized-v1`**, 22 templates |
Skills | `approach_entity_0..3`, `approach_resource`, `mine_batch` (`SKILL_BUDGET` 60) |
Policy extractor | `EXTRACTOR_VERSION` **7** (was 4; checkpoints do not cross that boundary) |
`assisted-v1` action profile | declared in the mod, **not implemented** on the Python side |
Tests | **642 engine-free** (`tests/unit` + `tests/contract`), plus engine-marked |

### 3.3 Holdouts

| | `holdout_v1` | `holdout_v2` | **`holdout_v3`** (live) |
|---|---|---|---|
content hash | `4d8b9507…` | `df72fe29…` | `ff22dedd…` (was `4227eb56…`, then `97a190f3…`) |
seed run_id / start | `holdout-v1` / 1000 | `holdout-v2` / 2000 | `holdout-v3` / 3000 |
declared candidates | navigate, deliver, mine_smelt | deliver, repair_belt, restore_power | deliver, repair_belt, restore_power |
tasks frozen | 3 | 6 | **8** (`build_line` added 2026-09-09, deliberately **not** a candidate) |
`--verify` vs source | **fails** (repair/restore now 1.6.0) | **fails** (same) | **OK** |
status | closed | closed | live |

All three share `master = 20260908` and differ only by `run_id` and `start_index`, so their
episode streams are disjoint. Per-task `entry_hash` is recorded inside `holdout_v3` only; v1
and v2 predate the field.

**`holdout_v3`'s content hash moved twice on 2026-09-09 and no existing result was
invalidated.** `build_line` was added with `--add-task`, which copies existing entries
verbatim; then its own entry was re-frozen with `--replace-task` when it reached v1.1.0.
Both modes exist because `--refreeze` rebuilds the document and would drop
`declaration.declared_at` -- the timestamp that is the entire evidence the candidate set
predates any held-out result. `--replace-task` refuses an entry any manifest cites.
All seven pre-existing `entry_hash` values are byte-identical, verified against a copy
taken before the change, and `stale_citations` reports no run as stale.

---

## 4. Results, with provenance

**No family meets PLAN 4.5** (three families ≥0.80 on the structural split, three seeds).

| family | best held-out | seeds | holdout cited | status |
|---|---|---|---|---|
`deliver` | 0.77 mean (0.92 / 0.85 / 0.54) | 3 | `holdout_v2` | misses; **does not reproduce** — an earlier 3-seed run at the same fixed seeds gave 0.92 mean |
`restore_power` | 0.257 mean (0.77 / 0.00 / 0.00) | 3 | `holdout_v2` | misses; the spread had an identified cause (§5) |
`repair_belt` | 0.06 stochastic | 1 | **`holdout_v3`** | misses |

### 4.1 Provenance defects in published evidence

- **7 of 11 `phase4-release-*.json` cite a `content_hash` no committed holdout now has**
  (`f77aa2c2…` in four v1-era files, `b41e4f0c…` in three v2-era files). Those numbers cannot
  be tied to a scene set that still exists.
- **Two files disagree with their own headers at cell level.**
  `phase4-release-navigate.json` cites one hash in its header and two different ones in its
  seed-2 and seed-3 cells; `phase4-release-v2-restore_power.json` likewise. Each of those two
  files reports numbers measured against at least three different scene sets.
- Only the two `-v2-repair_belt`/`-v2-restore_power` files carry a `superseded` marker. The
  four v1-era files do not, despite citing a vanished hash.
- **Fixed** `a69aacb`: `freeze_holdout.manifests_citing()` used to return zero citations for
  every holdout, because manifests recorded `config.holdout` as a *path string* while the tool
  looked for a dict. Runs now write a top-level citation and the tool finds it. Runs predating
  that commit remain uncitable.
- **Fixed** `9a24cba`: `reward-audit.json`'s withdrawn "190 episodes, zero successes, −0.300"
  note is now labelled as withdrawn in the file itself.
- The four run directories cited by the `-v3-` files are **not on this laptop** because those
  cells ran on the desktop; they exist at `D:\FactorioRL\runtime\runs\` there. `runtime/` is
  gitignored on both machines.

### 4.2 A correction to an earlier claim in this repo's history

Earlier today I reported that two run manifests cited **commits that no longer exist**, and
concluded the old task versions could not be faithfully reconstructed. **That was wrong.**
`e2e2d3aaf52e` and `ba932625a195` both exist and are ancestors of `HEAD`; all 40 distinct
manifest commits do. The conclusion drawn from the error — declining to re-evaluate the
existing checkpoints — was therefore unfounded.

Relatedly, `docs/LEDGER.md`'s heading "3.3 — Six introductory families (**incomplete**: no
scripted solutions)" is misleading: `reference.SOLVERS` has a solver for **all seven**
families. `RegisteredTask.solve` is `None` for all seven, which is a different attachment
point, not an absence. `docs/evidence/phase3-solvability.json` covers only `plate_line`, so
the other six have no *published* solver rate — but `repair_belt`'s solver was measured at
6/6 this session.

---

## 5. Known defects, open and closed

### 5.1 Fixed this session

| defect | fix |
|---|---|
`CurveLogger` summed rewards across the worker vector and read `success` from `infos[0]` only | `8b68296`, `tests/unit/test_curve_logger.py` |
Evaluation was greedy-only | paired stochastic arm on identical episodes, same commit |
`holdout_v2` re-frozen five times, whole-file hash cited by all tasks | per-task `entry_hash`; `holdout_v3` bumped rather than re-frozen |
Exploring-starts occupancy guard compared pre-snap coords to post-snap and never matched | `5ba9b14` |
Both repair holdouts tested regimes training never showed | v1.6.0 of both families, `c911f90`; audit in [`holdout-distribution-audit.json`](evidence/holdout-distribution-audit.json) |

The last one explains `restore_power`'s 0.77 / 0.00 / 0.00: over 600 scenes per family the
missing pole was flanked by poles in **600/600** training scenes and **0/600** held-out ones,
where it is always terminal. Two rules fit training equally well and only one transfers, so
which a seed learned was a lottery. That is not transfer variance.

### 5.2 Open, confirmed by inspection, not yet fixed

| id | defect |
|---|---|
**R1.1** | An infrastructure failure is **trained on**. `excluded_from_metrics` is read only by reporting code; `vecenv.step_wait` sets `TimeLimit.truncated` + `terminal_observation` for it, so `sb3_contrib` bootstraps `gamma * V(stale obs)` and adds the transition to the buffer unconditionally |
**R1.2** | Frozen evaluation **cannot terminate** once the episode cursor is exhausted — `_reset_one` past the bound plays real episodes tagged `None`, which never become eligible. No timeout anywhere, and `--eval-episodes` 100 against `episodes_per_task` 100 leaves zero slack |
**R1.2** | `_reset_one` calls `env.reset()` unguarded; this frame is in **eight** recorded `WinError 10054` run failures, and is why the crash fires before the hang |
**R1.3** | Skills sum primitive rewards **undiscounted** while PPO discounts per decision, so potential-based shaping leaves a residual `−w(1−γ)Σφ` charged to the skills that make progress. `info["skill_steps"]` already carries the duration and nothing in the learner reads it |
**R0.2** | `baselines.cache_key` omits the seed plan's `run_id`/`start_index`, so a floor measured on one holdout can be served for another at the same `master_seed` |
**R0.2** | Manifest lacks: dirty-tree identity, the holdout's own seed stream, per-episode scene hashes, checkpoint hash, exclusion counts |

### 5.3 Benchmark-design defects, still open

- `navigate` random floor reaches **0.98–1.00** in the skills action space; `mine_smelt`
  **0.24–0.32**. Clearing 0.80 on either demonstrates little. Both are excluded from the
  current declaration for that reason, not for failing.
- `supply_furnace` scored **0.04** held out against a **0.28** floor — worse than random.
- `phase4b-ablation-deliver.json` carries `holdout: null` and a suspected baseline-cache
  collision.

---

## 6. Capability status

| capability | status | evidence |
|---|---|---|
Exact stepping, worker lifecycle, protocol v2 | verified | `phase1-engine-suite.txt`, `phase2-gate.json` |
Task engine, RL spaces, reset correctness | verified | `phase3-gate.json` |
Reference solvers, 8 families | implemented; **1.00 on every task and split** | `phase3-solvability.json` |
PPO training pipeline | runs; R1 complete | R1 |
Frozen holdout evaluation | terminates under injected failure; records **per-scene outcomes** | R1.2, §10 |
Paired greedy/stochastic evaluation | verified | `repair_belt-curriculum-holdout_v3.json` |
Three-family mastery (PLAN 4.5) | **not met** | §4 |
Split audit over every task | published for **all eight**; `plate_line` fails on a real finding | `phase3-generator-diagnostics.json` |
Held-out *combination* coverage (§9) | checked by set containment; both v1.6.0 fixes confirmed | §7 |
Regression suite, separate from any rate (§10) | 10 cases, all open findings | `regression_scenes.json` |
LLM agent loop, addressed actions | works on `deliver` and `plate_line` commissioning | `phase5-agent-runs.json`, `phase5-demonstration.json` |
Construction (**reference** builder builds machinery) | **R3.1 gate passes**, 15/15 on a real engine | `r3-construction.json` |
Construction (**agent** builds machinery) | **not demonstrated** — R3.2 needs a bounded LLM baseline; the gate stays unmet until an agent passes | redirection R3.2 |
Recovery (validated disruption + no-action control) | **not demonstrated** | redirection R4 |
`assisted-v1` action profile | **not implemented** | §3.2 |
Release packaging | `release/` snapshot is **stale** (predates the curve-logger and holdout_v3 corrections) | — |

---

## 7. What R3.1 delivered, and what is next

**R0, R1, R2 and R3.1 are complete.** 642 engine-free tests, lint and format clean.
R3.1's gate passes on a real engine: `docs/evidence/r3-construction.json`, 15/15.

| | |
|---|---|
Truth channel | `working_counts`, `placed_counts` and `built` are prototype-keyed. `containers`/`working` key on `scene.aliases`, bound at install, so **no predicate could name a machine the agent placed**. `BUILT`/`ANY_WORKING` can |
Sustained output | `SUSTAINED_OUTPUT(item, at_least, over_ticks, not_before_tick)` reads a windowed `(tick, produced)` history. `PRODUCED` is a timestamp-free counter: "30 plates, 15 in the last window" and "30 plates, dead since tick 4000" were the same number |
§12 metrics | `ProductionMetrics` reports time to first sustained output, cumulative production and final rate **per simulated tick**, never per decision, and says which denominator it used in the payload |
`build_line` | Ore on the ground, machines in the inventory, nothing built. Success = both machines `BUILT` **and** `SUSTAINED_OUTPUT`. Reference 1.00 on all three splits, random floor 0.00 |
Validators | Success-false-at-reset and multi-tile footprint overlap now **exist**; `validate_all`'s docstring had claimed both and grep found only the sentence |
§9 coverage | Set containment over categorical cells, in `factoriorl.tasks.coverage`, published for all eight tasks. Both v1.6.0 fixes confirmed by the rule, not by their commit messages |
§10 split | `evaluate_parallel` records per-scene rows; `factoriorl.regression` holds named cases and carries **no pooled rate** |
R3.1's "must not complete the evaluated run" | `reference.solve` raises on `Branch.EVAL` |

### 7.1 Two things the gate caught in work done the same day

Both are recorded because a gate that only confirms is not doing its job.

- **`ANY_WORKING` is unusable as an acceptance conjunct.** Over 102,451 samples a
  *healthy* plate line reads status `working` on **7.5%** of ticks: one burner drill
  outpaces one stone furnace, so it sits in `waiting_for_space_in_destination` 87%
  of the time. Conjoining it would have failed correct builds nine times in ten.
- **A trailing window that begins at tick 0 spans construction.** `build_line` was
  accepted at tick 3600 with a drill mined out 90 ticks earlier, because the window
  still held every plate the line had ever made -- so at those parameters
  `SUSTAINED_OUTPUT` was doing almost nothing `PRODUCED` does not.
  `not_before_tick = 2 * over_ticks` moves the earliest acceptable window to
  `[3600, 7200]`, entirely post-construction. Reference solvability went 120 → 240
  steps, which is tick 7200 exactly.

### 7.2 Open findings

- **`plate_line` fails the split audit**: 10 of 200 `commissioning_walled` scenes
  start the character **on a wall tile**. The screen is at x = 6 and the start is
  drawn at radius 5..9, so 5% of scenes land inside the neutral wall, and
  `build_blueprint` teleports without a collision check. Recorded as ten regression
  cases, all `expected_present`. **Deliberately unfixed**: the generator's bytes are
  what this task's commissioning evidence and its `holdout_v3` entry describe, the
  reference still commissions at 1.00, and the defect has not been shown to change
  any measurement.
- **`restore_power`'s test split admits only 101 distinct scenes over 200 seeds**
  (`gap_near_drill`), and `two_gaps` 117. Below `deliver`'s 200 but above
  `MIN_DISTINCT_SCENES`; it means a Wilson interval over 200 episodes there is
  narrower than the content justifies.
- **`build_line` has no agent result.** R3.2 asks for a bounded LLM baseline
  reporting supplied knowledge, assistance, game time, decisions and inference
  usage. **A failed agent is a valid baseline result; the construction
  demonstration gate stays unmet until an agent actually passes.**

### 7.3 Next

**R3.2** — the bounded LLM baseline on `build_line`, then **R4** (recovery, which
needs the intervention-freshness fix first) and **R5** (interpretable learning
experiments).

### 7.4 Synthesis §8: the symmetry is C4, not D4

§8 proposes scoring placement candidates by their resulting local structure
instead of one unrelated logit per index. Its gate leads with *"treat symmetry as
a tested property: not every Factorio mechanic is rotation/reflection
invariant"*, and *"defer a new value-learning algorithm until the simpler
representation test is informative"*. So the symmetry was measured first, on a
real engine, with the candidate set (every integer offset within 4 tiles), the
information (`can_place_entity` plus measured production, no evaluator truth) and
the budget (3600 ticks per candidate) held fixed:
`docs/evidence/section8-symmetry.json`.

**Rotation is invariant. Reflection is not.** Each drill facing admits exactly two
productive furnace centres, and all three rotations of south's set match exactly.
Reflecting south's `{(0,2), (1,2)}` in y gives `{(0,-2), (1,-2)}`, while north's
measured set is `{(-1,-2), (0,-2)}` — off by exactly one tile in x.

**Translation is invariant**, checked at all four parities of drill centre, so
the productive offsets are a property of the local structure and not of where it
sits in the world.

The cause is a parity: a 2×2 entity at integer centre `(cx, cy)` occupies tiles
`{cx-1, cx} × {cy-1, cy}`, extending one tile in the negative direction and none
in the positive. That bias is relative to the entity, which is why translation
survives it; it survives a 90° rotation, and it does not survive a flip.

**Design consequence, and the reason this test came first.** A candidate scorer
may share parameters across **rotations and translations** of a local structure
and must **not** share them across reflections: a scorer assuming the full dihedral group would be
wrong on half its orbit, scoring `(1,-2)` valid for a north-facing drill where the
engine starves it. The representation test is therefore informative, and the
scorer itself remains deferred — it bumps `EXTRACTOR_VERSION`, and §8 asks for the
value-learning step only after this.

Two interface limitations carried forward, both stated in code:

- **Argument masks cannot depend on the sampled operation.** Index 0 of each
  argument dimension is an `UNUSED` sentinel and a needed-but-unused argument
  decodes to a *counted* no-op. Removing it needs a custom autoregressive
  distribution: stock sb3 fetches masks before the forward pass, samples all
  sub-distributions independently from one flat head, and sums independent
  factors in `log_prob`.
- **`assisted-v1` is declared in the mod and not implemented on the Python side.**

**Every skill-augmented number predating `8e6bdb6` used a different objective**,
and every checkpoint predating `e749e9c` a different observation encoding
(`EXTRACTOR_VERSION` 4 → 7). Neither is comparable across those boundaries.
