# FactorioRL current status

**Maintained entrypoint.** `docs/LEDGER.md` is history; `HANDOFF.md` is a 2026-09-07 snapshot
and carries its own staleness banner. Direction comes from
[`docs/DEVELOPMENT_REDIRECTION.md`](DEVELOPMENT_REDIRECTION.md); this file is day-to-day
state. Read this before treating any older summary as current fact.

**Last reconciled:** 2026-09-08, at commit `fe51792`.

---

## 1. Live now

| what | machine | state | cites |
|---|---|---|---|
`rp16-20260908T205825-078a96f0` — `restore_power` v1.6.0, 1 seed, 25k steps | laptop | **running** (training finished at timestep 25088; in held-out evaluation) | `holdout_v3` |

**No score is attached to that run.** It has no `result.json` yet.

Completed minutes earlier, and the only result measured against the live holdout:

| run | task | held-out (greedy / stochastic) | floor | training successes |
|---|---|---|---|---|
`cur16-20260908T204530-b1e2eb75` | `repair_belt` v1.6.0 + exploring starts 1/3 | **0.00 / 0.06** | 0.00 | 7 of 221 episodes |

Evidence: [`repair_belt-curriculum-holdout_v3.json`](evidence/repair_belt-curriculum-holdout_v3.json),
citing `holdout_v3` `4227eb56…`, task entry hash `32f8a13a2b4772b1`, `covers_frozen_set: true`.

Two caveats that belong with that number, not after it:

- **Greedy and stochastic disagree on this family** (0.00 vs 0.06 held out, 0.00 vs 0.12 on
  training layouts). On `restore_power` they agreed exactly. Neither arm is the "real" number;
  both are reported, and neither was chosen after seeing the other.
- **Two changes landed together** — the v1.6.0 layout/marker redesign and the exploring-starts
  curriculum. Per redirection R5.2, the gain cannot be attributed to either individually
  without a separate controlled experiment. It is 7 training successes against a prior 1 in
  198, and 0.06 held out against a prior 0.00; that is movement, not a result.

---

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
Tests | 395 total: 331 engine-free (`tests/unit` 279, `tests/contract` 52), 64 engine-marked |

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
- **`freeze_holdout.manifests_citing()` returns zero citations for every holdout**, because
  manifests record `config.holdout` as a *path string* while the tool looks for a dict with
  `id`/`content_hash`/`task_entry_hash`. `stale_citations()` therefore checks nothing. This is
  an open R0.2 item.
- `docs/evidence/reward-audit.json` still contains the **withdrawn** note "repair_belt: 190
  episodes, zero successes, mean reward −0.300". Open item.
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

Per redirection §7: **R0 and R1.1–R1.2 first; do not launch another release matrix.**

1. **R0.1** — this file. Done.
2. **R1.1** — stop infrastructure failures reaching an optimizer update; failure-injection test
   through the real collector.
3. **R1.2** — bounded retry of the same scene identity or explicit incomplete coverage; guard
   the reset path; validate `eval_episodes <= episodes_per_task`.
4. **R0.2** — manifest identity fields and the baseline cache key.
5. **R1.3** — primitive-time objective with duration-aware discounting (chosen convention).

Then R2 (parameterized actions shared by RL and LLM clients), R3 (construction), R4
(recovery), R5 (controlled learning), R6 (release).
