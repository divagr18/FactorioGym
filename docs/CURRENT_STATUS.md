# FactorioRL current status

**Maintained entrypoint.** `docs/LEDGER.md` is history; `HANDOFF.md` is a 2026-09-07 snapshot
and carries its own staleness banner. Direction comes from
[`docs/DEVELOPMENT_REDIRECTION.md`](DEVELOPMENT_REDIRECTION.md); this file is day-to-day
state. Read this before treating any older summary as current fact.

**Last reconciled:** 2026-09-08, after R0 and R1 (`1a780ab`).

---

## 1. Live now

**Nothing is running.** Both machines idle.

Results measured against the live holdout (`holdout_v3`, `4227eb56`):

| task | run | held-out greedy / stochastic | floor | training successes |
|---|---|---|---|---|
`restore_power` v1.6.0, 25k, seed 1 | `rp16-…078a96f0` | **0.00 / 0.37** `[0.282,0.468]` | 0.02 | 101 / 227 |
`repair_belt` v1.6.0 + curriculum, 50k, seed 1 | `cur16-…b1e2eb75` | **0.00 / 0.06** | 0.00 | 7 / 221 |

Both single-seed, both `covers_frozen_set: true`. Evidence:
[`restore_power-v1.6.0-holdout_v3.json`](evidence/restore_power-v1.6.0-holdout_v3.json),
[`repair_belt-curriculum-holdout_v3.json`](evidence/repair_belt-curriculum-holdout_v3.json).

Three caveats that belong with these numbers:

- **Greedy and stochastic disagree sharply on both**, with non-overlapping
  intervals on `restore_power`. Read greedily either run says "did not learn".
  Both arms scored identical episodes; neither was chosen after seeing the other.
- **One seed each.** `restore_power` at v1.5.0 gave 0.77 / 0.00 / 0.00 across three
  seeds, so a single cell says little about the family.
- **`repair_belt` had two changes at once** (layout redesign + curriculum), so per
  R5.2 the movement is not attributable to either alone.

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
`deliver` | 1.3.0 | unset | 120 | `two_chest`,`decoy_chest` / `stacked_depot` / **`screened_depot`** | `dst` |
`mine_smelt` | 1.2.0 | unset | 400 | `near_patch`,`fuelled_furnace` / `split_patch` / **`screened_patch`** | — |
`navigate` | 1.2.0 | 0.997 | 150 | `open`,`wall_corridor` / `pillar_field` / **`wall_arc`** | `goal` |
`plate_line` | 1.1.0 | unset | 400 | `commissioning` / `commissioning_far` / **`commissioning_walled`** | — |
`repair_belt` | **1.6.0** | 0.999 | 300 | `gap`,`gap_far` / `misrotation` / **`double_gap`** | `sink`,`gap`,`gap2` |
`restore_power` | **1.6.0** | 0.999 | 250 | `pole_gap`,`pole_gap_far` / `two_gaps` / **`gap_near_drill`** | `drill`,`gap`,`gap2` |
`supply_furnace` | 1.2.0 | unset | 300 | `linear_row`,`two_furnaces` / `far_ore` / **`shared_input`** | — |

All seven: observation `local-v2`, action `primitive-v1`, deliberation `flat-v1`,
`decision_ticks` 30.

### 3.2 Stack

| item | value |
|---|---|
Engine | Factorio 2.0.60 build 83512 win64 |
Mod | 0.1.0, source digest `8fb6913497f291c9` (readable only from run manifests) |
Protocol | 2 |
Observation profiles | `local-v1`, **`local-v2`** (radius 32, entity_cap 48, grid 65×65) |
Action catalog | `primitive-v1`, 46 templates |
Skills | `approach_entity_0..3`, `approach_resource`, `mine_batch` (`SKILL_BUDGET` 60) |
Policy extractor | `EXTRACTOR_VERSION` 4 |
`assisted-v1` action profile | declared in the mod, **not implemented** on the Python side |
Tests | 440 total: 376 engine-free, 64 engine-marked |

### 3.3 Holdouts

| | `holdout_v1` | `holdout_v2` | **`holdout_v3`** (live) |
|---|---|---|---|
content hash | `4d8b9507…` | `df72fe29…` | `4227eb56…` |
seed run_id / start | `holdout-v1` / 1000 | `holdout-v2` / 2000 | `holdout-v3` / 3000 |
declared candidates | navigate, deliver, mine_smelt | deliver, repair_belt, restore_power | deliver, repair_belt, restore_power |
`--verify` vs source | **fails** (repair/restore now 1.6.0) | **fails** (same) | **OK** |
status | closed | closed | live |

All three share `master = 20260908` and differ only by `run_id` and `start_index`, so their
episode streams are disjoint. Per-task `entry_hash` is recorded inside `holdout_v3` only; v1
and v2 predate the field.

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
Reference solvers, 7 families | implemented; **published rate for `plate_line` only** | `phase3-solvability.json` |
PPO training pipeline | runs; **correctness open** (R1.1, R1.3) | §5.2 |
Frozen holdout evaluation | runs; **termination unproven** (R1.2) | §5.2 |
Paired greedy/stochastic evaluation | verified | `repair_belt-curriculum-holdout_v3.json` |
Three-family mastery (PLAN 4.5) | **not met** | §4 |
LLM agent loop, addressed actions | works on `deliver` and `plate_line` commissioning | `phase5-agent-runs.json`, `phase5-demonstration.json` |
Construction (agent builds machinery) | **not demonstrated** — the existing demo starts from preplaced machines | redirection §2 |
Recovery (validated disruption + no-action control) | **not demonstrated** | redirection R4 |
`assisted-v1` action profile | **not implemented** | §3.2 |
Release packaging | `release/` snapshot is **stale** (predates the curve-logger and holdout_v3 corrections) | — |

---

## 7. Next work package

**R0 and R1 are complete.** 376 engine-free tests, lint and format clean.
Verified end to end on a 2,000-step engine run (`r0r1smoke-…e1a766e8`): the
holdout citation, per-task entry hash, checkpoint hash and eval streams all
populate, `manifests_citing` finds the citation where it previously found zero
for every holdout, and the run reports **2048 policy decisions against 3517
primitive transitions** — the budget confound R1.3 exposes, now measured.

Next is **R2** — make construction and stable interaction expressible:

1. **R2.1** canonical parameterized actions, reusing the existing addressed LLM path.
2. **R2.2** an RL action adapter over operation / target / placement / orientation.
3. **R2.3** observations that make the task and its failures legible.

Then R3 (construction), R4 (recovery), R5 (controlled learning), R6 (release).

**Every skill-augmented number predating `8e6bdb6` used a different objective**
(undiscounted sum, one discount per decision) and is not comparable with runs
after it.
