# FactorioRL current status

**Maintained entrypoint.** `docs/LEDGER.md` is history; `HANDOFF.md` is a 2026-09-07 snapshot
and carries its own staleness banner. Direction comes from
[`docs/DEVELOPMENT_REDIRECTION.md`](DEVELOPMENT_REDIRECTION.md); this file is day-to-day
state. Read this before treating any older summary as current fact.

**Last reconciled:** 2026-09-09, after R4.

---

## 1. Live now

**Nothing is running.** Both machines idle.

### R4 changed two things every earlier number should be read against

**A withdrawn claim.** `phase5-demonstration.json`'s
`sustained_production_restored` and `all_clauses_met` are **withdrawn**, in a
`withdrawal` block inside that file; every other byte of it is the original
run. The disruption emptied fuel *inventories* only, and
`docs/evidence/r4-burner-decay.json` measures a burner running **1,170 ticks**
afterwards on stored energy the clear does not touch -- against a 90-tick
"recovery". There was no outage criterion and no control arm. What replaces it
is `docs/evidence/r4-recovery-arms.json`: control 5 plates, agent 5/39/43,
undisturbed ceiling 43, from a digest-validated common state, with the control
arm's outage confirmed at tick 5970. Two of three agent runs are `loss_averted`
rather than `recovery_demonstrated` -- their own line never stopped, so there
was nothing for them to recover from.

**A prompt that could not state a cause.** `summary._entity_row` never copied
the per-entity status *name*, `working`, `fuel` or `output` that `local-v2` v4
was bumped to publish, so every language-model prompt this repo ever rendered
showed `status 18` and nothing else. Fixed;
`SUMMARY_ENCODING_VERSION` is **3**. **R3.2's LLM baseline (0/2, 539 decisions)
and R4.2's first agent arm were both measured against the broken renderer** and
do not compare across the boundary.

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
`plate_line` | **1.2.0** | unset | 400 | `commissioning` / `commissioning_far` / **`commissioning_walled`** | — |
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
Observation profiles | `local-v1`, **`local-v2` v5** (radius 32, entity_cap 48, grid 65×65; v5 drops the ten `parameter-N` recipe placeholders, which were 10 of 22 values in the `recipe` argument domain and craftable by nobody) |
Action catalog | `primitive-v1`, 46 templates; **`parameterized-v1`**, 22 templates |
Skills | `approach_entity_0..3`, `approach_resource`, `mine_batch` (`SKILL_BUDGET` 60) |
Policy extractor | `EXTRACTOR_VERSION` **7** (was 4; checkpoints do not cross that boundary) |
`assisted-v1` action profile | declared in the mod, **not implemented** on the Python side |
Tests | **694 engine-free** (`tests/unit` + `tests/contract`), plus engine-marked |

### 3.3 Holdouts

| | `holdout_v1` | `holdout_v2` | **`holdout_v3`** (live) |
|---|---|---|---|
content hash | `4d8b9507…` | `df72fe29…` | `5a00330b…` (was `4227eb56…`, `97a190f3…`, `ff22dedd…`) |
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

**Phase 4 is not done.** Its mechanical exit gate passes -- `phase4-gate.json`, 15 checks
on 2026-09-07: training dependencies resolve, CUDA is a real device, train and eval seed
branches are disjoint, published checkpoints load, manifests verify, and a short run
completes unattended -- but two sub-sections are unmet and a third is only half verified:

| sub-section | state |
|---|---|
4.1 baseline policy | met, **except** "checkpoints restore both inference and *resumable training* state": the gate checks `checkpoint loads for inference` only, and no test anywhere exercises resume |
4.2 single-task learning | met. Learning exceeds the random floor (`deliver` 0.77 against 0.00, `restore_power` 0.26 against 0.00), eval is on a disjoint seed branch, PPO gets no scripted labels, failed runs are retained |
4.3 profiling | met. Dominant bottleneck identified and it is not close: `environment_step` 56.44 ms against `encode` 0.081 ms. 8 workers chosen at 261 steps/s, 3.43x one worker at 43% per-worker efficiency, described as measured rather than as "faster". Deviations: the sweep ran 1/4/8/12/16 rather than the plan's 1/2/4/8, and the file predates this week's profiling changes |
4.4 shaping dependence | **not met.** The headline is **withdrawn**: the two arms were never scored on the same episodes, so the gap (shaped 35/50, sparse 48/50) cannot be separated from their different scene draws. `--seed` fixes torch and numpy, not the scenes. The tooling now pairs arms by holdout; **the experiment has not been re-run** |
4.5 release learning result | **not met.** No family reaches 0.80 on the structural split |

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

## 5b. R4 additions

| what | where | state |
|---|---|---|
| one refresh path: `FactorioEnv._adopt`, `advance`, `resync` | `env.py` | done, engine-checked by `gate_r4` |
| a sampled window, not merely a spanned one | `tasks/spec.py` | done |
| declared outage/recovery criteria, before any arm ran | `recovery.py` | done |
| burner decay curve | `tools/burner_decay.py` | measured, published |
| three arms, agent arm repeated | `tools/recovery_arms.py` | measured, published |
| `TaskSpec.track`, replacing `release_matrix.CATEGORY` | `tasks/spec.py` | done; `plate_line` and `build_line` are now inside the release qualification filter they were invisible to |
| `fault-location:published` assistance | `assistance.py` | done |
| `TaskSpec.disruptions` + typed `disrupt` request | `spec.py`, `session.py`, `world.lua` | done |
| `diagnose_line` (diagnosis track) | `tasks/families/diagnose_line.py` | registered, frozen, split audit passes; **solvability still short of the floor ceiling in the skills action space -- see below** |
| `keep_line_running` (persistent-operation track) | `tasks/families/keep_line_running.py` | registered, frozen, split audit passes; solvability not yet measured |
| `gate_r4.py` | `factoriorl/gate_r4.py` | written, not yet run on an engine |

**Both new families measure `solvable`, none `defective`** (10 episodes per
split):

| family | split | reference | random, primitives | random, skills |
|---|---|---|---|---|
| `diagnose_line` | train | 1.00 | 0.10 | 0.80 |
| `diagnose_line` | test | 1.00 | 0.00 | 0.80 |
| `keep_line_running` | train | 1.00 | 0.00 | 0.00 |
| `keep_line_running` | test | 1.00 | 0.00 | 0.00 |

`diagnose_line`'s primitive floor took three measured iterations to come down
from 0.40; its **skills** floor is 0.80, which puts it in the same class as
three already-accepted families -- `tools/solvability.py` records `mine_smelt`
at 0.80, `supply_furnace` at 0.88 and `deliver` at 0.80 there. It is a
**primitive-space** benchmark and a skills-space result on it would demonstrate
nothing. `keep_line_running` is discriminative in both spaces.

**`python -m factoriorl.gate_r4` passes**, 20 clauses on a real engine
(`docs/evidence/r4-tracks.json`). It failed two of them on its first run and
found a real defect: `world.disrupt` passed marker aliases to
`find_entities_filtered`, which raises on a name that is not a prototype, so
the declared outage never landed -- reported as `touched: []`, `ok: false`
beside an observation still showing 47 coal. Fixed and re-run.

Neither new family is a declared holdout candidate, and no agent or policy
result is published for either.

**First language-model success on `plate_line`.** `gpt-5.6-luna` finished it --
30 plates, 249 decisions, tick 7470 -- against R3.2's 0/2 with
`gpt-4.1-mini`. Two caveats that keep it a demonstration rather than a
benchmark row: `--wait-batch 8` was set, which is a declared assistance, and
the provider rejects `temperature: 0.0`, so the run sampled at its default and
is not reproducible from its seed. Both are recorded in the run manifest;
`runtime/runs/watch-20260909T131905-376d37f0/replay.html` renders it.

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
Construction (**agent** builds machinery) | **not demonstrated** — the R3.2 baseline is measured and reported; the gate stays unmet until an agent passes | `r3-agent-baseline.json` |
Recovery (validated disruption + no-action control) | **demonstrated**: control arm's outage confirmed at tick 5970, agent arm repeated 3x, digest-validated common state. Two of three agent runs are `loss_averted`, not recovery -- their own line never stopped | `r4-recovery-arms.json`, `r4-tracks.json` |
Declared disruptions, applied by the env | **implemented and gated**: typed `disrupt` request, `TaskSpec.disruptions`, fresh post-disruption observation by construction | `r4-tracks.json` |
Three declared tracks (repair / diagnosis / persistent operation) | **implemented**, each with its own hints, assistance and outcome; no aggregate across them | `r4-tracks.json` |
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

- ~~`plate_line` fails the split audit~~ **fixed in v1.2.0.** 10 of 200
  `commissioning_walled` scenes started the character on a wall tile.
  `Blueprint.character_obstructed` now checks the general form of the defect --
  draw a start, then place obstacles -- for **every** family, so the next
  generator to do it fails `tasks validate` rather than needing an audit;
  `plate_line` was the only one affected. `_clear_of` pushes a colliding start
  outward, consumes no randomness and is a no-op when the start is clear, so
  **1 of the 100 frozen holdout episodes changed** and the other 99 are
  byte-identical. The ten regression cases are now guards: they assert the
  character is *not* on a blocked tile. **All eight tasks pass the split
  audit.**
- **`restore_power`'s test split admits only 101 distinct scenes over 200 seeds**
  (`gap_near_drill`), and `two_gaps` 117. Below `deliver`'s 200 but above
  `MIN_DISTINCT_SCENES`; it means a Wilson interval over 200 episodes there is
  narrower than the content justifies.
- **`build_line`'s agent baseline is measured and it fails.** See §7.3. A
  failed baseline is a valid result; the construction demonstration gate stays
  unmet until an agent actually passes.

### 7.3 R3.2: the LLM baseline, measured

`gpt-4.1-mini` over `parameterized-v1`, 2 episodes on `build_line`'s train
split, `max_steps` 300, assistance `wait-batch:12`:
`docs/evidence/r3-agent-baseline.json`, replay in the run directory.

| | episode 0 | episode 1 |
|---|---|---|
decisions | 300 (`step_ceiling`) | 300 (`step_ceiling`) |
game ticks | 9,000 | 9,000 |
plates in the final 3,600-tick window | **0.0** | **5.0** |
cumulative plates | 0 | 5 |
time to first sustained output | never | never |

Across both: **539 decisions, 681 provider calls, 1,308,177 tokens**, 30
fallbacks, 61 batched waits, **16 of 71 placements accepted**, 110
`bad_argument` and 62 `unknown_target` decision failures. Model latency p50
1,099 ms, p95 1,600 ms.

**Why it failed, from the replay.** The agent places drills successfully, then
repeatedly tries to put the furnace *adjacent* to a 2x2 drill and is refused
`collision` — the geometry the reference builder had to measure on an engine and
which is deliberately withheld from the prompt (`FORBIDDEN_IN_PROMPT`, and the
tool refuses to run if it leaks). It then spreads its 60 coal across dozens of
one-item transfers and starves the line. Episode 1's 5 plates per 3,600 ticks
against the required 10, and against 15 for a correct line, is the useful
number: the criterion discriminates on a real agent and not only on a fixture.

Both episodes reached tick 9,000, past `build_line`'s 7,200-tick settling
point, so the agent had its full measurement window and failed on **production**
rather than on time.

**Assistance, stated.** `wait-batch:12` repeats a wait the model just chose,
stopping as soon as the observation changes, and is recorded in the manifest.
It exists because the window is 3,600 ticks against a 30-tick decision, so most
of a construction run is waiting for a furnace; its stopping rule reads the
observation only, never truth.

### 7.4 Next

**R4** (recovery, which needs the intervention-freshness fix first) and **R5**
(interpretable learning experiments). `build_line` is frozen in `holdout_v3` but
**deliberately not a declared release candidate**.

### 7.5 Synthesis §8: the symmetry is C4, not D4

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
