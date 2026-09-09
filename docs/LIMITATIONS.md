# Known limitations

Everything here is measured, not suspected. Each item says what was observed and
where the evidence is, because a limitations document that hedges is worse than
none: a reader cannot tell which entries are real.

PLAN.md 6.4 requires that deferred functionality is not advertised as available.
This file is where that obligation is discharged.

---

## 1. Engine and platform

**Factorio 2.0.60 (build 83512) is pinned and other builds are refused.**
`src/factoriorl/engine_config.py` checks the build and stops. This is deliberate
-- entity geometry, statuses and the collision layers the mod reads all move
between builds -- but it means the environment does not run on whatever Factorio
a user already has.

**Headless only, and therefore no screenshots.** PLAN 5.6 is implemented and, on
this build, always fails. Measured: `game.take_screenshot` is defined, accepts a
well-formed request, returns success, and writes nothing, because a headless
server has no renderer. `factoriorl.capture` verifies the frame appeared and
raises `CaptureUnsupported` when it did not, rather than attaching a missing or
stale file to a tick. Frames need a Factorio with graphics.

**Windows only in practice.** Nothing is deliberately Windows-specific, but
nothing has been run anywhere else.

**Run manifests embed absolute paths.** A manifest records the worker's
write-data directory and the engine executable, e.g. `D:\FactorioRL\runtime\...`
and `D:\Factorio\bin\x64\factorio.exe`. They are machine-specific and must be
redacted before a manifest is published as a release artifact.

---

## 2. The action catalog

**A discrete action cannot name what it acts on.** Catalog templates bind
`$target` to the *nearest* entity, because an action index carries no argument.
This one limitation produced three failures that looked unrelated:

* a trained policy alternating `approach_entity_1` between two chests for an
  entire episode, nine steps out and twelve back;
* `navigate` solved 99 times in 100 by a random policy over skills, because its
  scenes hold exactly one addressable entity and it is the goal;
* an agent putting 47 of its 50 coal into a mining drill while the furnace two
  tiles away stayed empty, in four runs out of five.

Addressed actions exist now, but **only for the language-model deliberation
profile** (`AgentConfig.addressed_actions`, recorded in the manifest). A policy
still cannot express them, so the rank-addressing weakness is unfixed for RL.
Doing so needs a pointer-style action head, which is not built.

**Placement is quantised to the character's position.** `place_<item>_<dir>`
builds one tile from wherever the character stands, and 2x2 machines snap to
integer centres. A burner mining drill at (1, 1) drops ore at (1.5, 2.30); the
only furnace centre that catches that is (1, 3); and that footprint covers the
tile the builder would have to stand on. **Aligning a furnace to a drill's drop
tile is not reliably expressible in this catalog**, which is why the Phase 5
demonstration has the agent commission a placed line rather than build one from
parts. `restore_power`'s solvability run shows the milder form:
`place_small_electric_pole_south: collision`.

---

## 3. The skill layer

`approach_entity_k` addresses by **rank**, and rank collapses into "the answer"
whenever a scene has few non-obstacle entities. Measured random floors over
skills, on the structural split:

| family | rankable entities | random | random + skills |
|---|---|---|---|
| `navigate` | 1 (the goal) | 0.20 | **1.00** |
| `supply_furnace` | 2-3 | 0.20 | **0.80** |
| `mine_smelt` | 1 (the furnace) | 0.00 | **0.20-0.40** |
| `deliver` | 4 (2 are decoys) | 0.00 | 0.00-0.11 |
| `repair_belt` | 12-13 | 0.00 | 0.00 |
| `restore_power` | 12-13 | 0.00 | 0.00 |
| `plate_line` | 2 | 0.00 | 0.00 |
| `diagnose_line` | 2 | 0.10 train, 0.00 test | **0.80** |
| `keep_line_running` | 2-4 | 0.00 | 0.00 |

**`navigate`, `supply_furnace`, `mine_smelt` and `diagnose_line` cannot carry
an 80% claim under the skill action space**: a uniform random policy already
clears or approaches it. Only families that place entities the solution does
not need -- decoys -- have a defensible floor. `tools/solvability.py` reports
both floors and flags any family above 0.10.

`diagnose_line` joined that list on measurement rather than by design, and the
reason generalises. Its difficulty is *which* fix to apply beside *which*
machine, and a macro whose whole job is to stand the character beside a machine
removes most of it: 0.10 over primitives against **0.80** over skills. Cutting
its budget from 500 decisions to 200 moved that number from 1.00 to 0.80, which
was worth doing and was not enough. It is a primitive-space benchmark, and a
skills-space result on it would demonstrate nothing.

`keep_line_running` is the one family measured discriminative in **both**
spaces -- 0.00 and 0.00, reference 1.00 on train and test. A disruption the
agent has to notice is not something a rank-addressing macro stumbles into.

PLAN 4b.1 forbids a skill that encodes a family's solution, and the test for it
originally grepped skill descriptions for task ids and marker names.
`approach_entity_0` contains no forbidden substring and *is* the answer in five
of the six families that existed when that was measured; there are ten now, and the
two added in R4.3 have not been re-measured against that check. The semantic check now pins the addressable field per family.

---

## 4. Rewards

**Only three of this repo's shaping components preserve the optimal policy.**
Ng, Harada & Russell's condition (*Algorithms for Decision Making* p.365): a
policy optimal under the original reward stays optimal under a shaped reward
**iff** the shaping has the form `F(s,a,s') = gamma*B(s') - B(s)`.

| kind | components | preserves the optimum? |
|---|---|---|
`potential` | `navigate.approach`, `repair_belt.toward_gap`, `restore_power.toward_gap` | **yes** -- can change learning speed only |
`high_water` | `plates_produced`, `carried`, `at_destination`, `ore_mined`, `ore_carried` | **no** -- can change the optimal policy |

`rewards.py`'s `POTENTIAL` computes `weight * (gamma * phi_next - previous)`
with `phi_next = 0` on termination, which is that form exactly, and its comment
already gives the reason termination must zero it.

A `high_water` mark is not of that form, and the consequence is the plateau
recorded below: shaping earnable while the success predicate is false creates a
*new* optimum, which is what paid a policy for stopping. **The caps that fixed
it bound the damage; they do not restore invariance.** So a shaping result on a
family whose shaping is `high_water` is an empirical question, and one on a
`potential` family is answered by the theorem -- worth stating whenever a
shaping comparison is quoted, because `deliver`, the family PLAN 4.4's
comparison runs on, has `high_water` shaping only.


**Three families used to pay a policy for stopping.** Shaping earnable while the
success predicate is false is a plateau: bank it, idle, and the only opposition
is the step cost. `deliver` returned +0.23 for reaching the plateau and standing
still, `mine_smelt` +0.45, `supply_furnace` +0.50. `deliver` seed 3 is the
worked example -- it converged on `take_iron-plate_20`, ran 95.9 steps of a 120
budget for +0.062, and scored 0.00 on all three evaluation rows.

Fixed by grading rather than shrinking: each progress component's weight is now
its cap divided by the quantity success requires. `tools/reward_audit.py`
enforces it and a test pins the empty violation set.

**The shaped-versus-sparse comparison is withdrawn.** It reported sparse 0.96
against shaped 0.70 and could not support it: the arms were scored on
*different scenes*. See §5.

---

## 5. Evaluation methodology

**`--seed` does not reproduce a run's scenes.** An episode seed is
`blake2b(master | run_id | branch | index)` and `run_id` is a fresh timestamp
plus salt per run, so two runs launched with the same `--seed` train and
evaluate on entirely different content. `--seed` fixes the torch and numpy
streams and nothing else. A frozen holdout is the only way to compare two runs
on identical episodes, and it must be passed explicitly.

**Comparisons are unpaired unless a holdout is passed.** Every arm of the first
4.4 and 4b runs reported touching indices 0, 1, 2... and none saw the same
worlds. That is fatal to a small effect: 4.4's result was withdrawn, and 4b's
conclusion survives only because flat 0.01 against skills 0.79 is far outside
what two 50-episode draws can produce. Both comparison tools now pass
`--holdout` by default and **refuse to report a verdict** unless the arms'
recorded `scored_episodes` are identical.

**A frozen holdout does not decontaminate the generators.** It makes the
*scenes* unseen; it does not make the generator unseen. `mine_smelt` and
`supply_furnace` had targets and layouts adjusted after their test-split numbers
were read, so those families were partly tuned against observed held-out
outcomes. No index offset undoes that.

**`restore_power` yields 70 distinct scenes across its 100 frozen episodes**, so
its effective sample size is smaller than the count suggests. `mine_smelt` is 92.

**Episode length in `per_scene` contradicts the truncation rule that produced
it, and the contradiction is unresolved.** Every failed episode on `deliver` is
flagged `truncated` with a step count *below* the decision budget: 103 of 113
across the two finished cells of the 4.4 comparison, as low as 12 against a
budget of 120, and in 2,734 training episodes not one reached 120 (the maximum
is 114). `env.py` truncates on exactly two conditions, and neither fits. The
decision budget is 120, so a 12-step truncation is not that. The tick budget is
15,000 and a decision advances exactly `decision_ticks` = 30, so exhausting it
takes ~500 decisions -- more than the decision budget allows, making the tick
budget unreachable by construction on this family. These have been ruled out as
explanations: an infrastructure failure dressed as a time limit (`vecenv.py:261`
excludes those, and `reset_failures` is empty for all three rows), `_steps`
surviving a reset (`reset` zeroes it), and a `gymnasium` `TimeLimit` wrapper
imposing a smaller cap (there is none).

What this does and does not affect: **success rates are unaffected**, because
they come from `info["success"]` and not from any step count, so the 4.4
comparison's rates stand. What is unreliable is `per_scene.steps`,
`mean_episode_steps`, and therefore any diagnosis that reads episode length --
which is most of what R5.1 asks a report to say. `tools/diagnose_run.py`
detects the unreachable-budget case and labels these `truncated_unexplained`
rather than attributing them to the only budget left standing.

`production.episode_ticks` is not a substitute: it exceeds `30 x steps` by a
variable 2 to 108 ticks because it spans the reset settle as well as the
decisions. Nor is `simulated_ticks`, which is derived as `primitive_steps x
decision_ticks` rather than measured, so ratios computed from it describe two
counters and not the clock. Resolving this needs a live episode instrumented to
report `_steps` and `tick` at the truncating step.

**A random floor is a property of the action space, not the task.** Quoting a
result without the floor measured over the same action space is meaningless, and
this was got wrong four times in one day -- including a baseline cache keyed on
the catalog digest that served a primitive floor to three skill arms.

---

## 6. Learning results

Phase 4 acceptance (4.5) requires at least three families at 80% on the
structural split with at least one production or repair family, evaluated over
100 frozen held-out episodes per family per training seed.

**WITHDRAWN (2026-09-08).** This section previously stated that `repair_belt`
and `restore_power` had no reward gradient and no training successes, citing
"190 training episodes, zero successes, mean episode reward -0.300". Both
halves were wrong.

`docs/evidence/reward-audit.json` reports `has_gradient: true` for both at
v1.5.0. And the numbers came from `CurveLogger`, which kept one reward
accumulator and one step counter for the entire worker vector and read the
success flag from `infos[0]` only -- so a success in any worker but the first
was written down as a zero, and the "-0.300" was an eight-worker sum that
happened to resemble one episode's step cost. The curves on disk contain 24 and
19 successes on `restore_power` and 1 on `repair_belt`. Fixed in `8b68296`,
covered by `tests/unit/test_curve_logger.py`.

On the fixed instrumentation, a 6,000-step `restore_power` run logs 51 episodes
and 5 successes -- **none in worker 0** -- and scores 0.25 stochastic against a
0.10 random floor on training seeds while scoring 0.00 on both held-out splits.
The failure is transfer, not learning. Two related corrections: the
`15-27 M` step lower bound (`docs/research/rl-theory.md` Anchor 1) is
explicitly "a floor with no ceiling attached" and cannot license a negative
claim; and `holdout_v2` was re-frozen five times, so the `phase4-release-v2-*`
scores for these two families were measured on scenes the current holdout no
longer contains. `deliver`'s frozen entry is byte-identical across the last
four freezes, so its 0.92 stands.


**Status: see `docs/LEDGER.md` and `docs/evidence/phase4-release-v2-*.json`.**
The first declaration (`holdout_v1`: navigate, deliver, mine_smelt) produced no
qualifying family -- `navigate` is disqualified on its 0.99 skill floor,
`mine_smelt` measured 0.32, and `deliver` scored 0.35. A second holdout
(`holdout_v2`) declares `deliver`, `repair_belt` and `restore_power`, chosen on
floors measured off the test split.

Until a release matrix is published and accepted in the ledger, **this project
has no accepted learning result**, and any checkpoint shipped with it is a
demonstration that training runs, not evidence that it works.

---

## 7. The agent interface

**Memory is per episode.** Carrying it across would put one evaluation scene's
contents into the next scene's prompt. That is contamination, not competence.

**Agent runs need a provider.** The adapters are provider-independent and a
local OpenAI-compatible endpoint is supported and tested, so no paid API is
required -- but a local model must be running, or `ScriptedAdapter` used, for
any agent command to do anything.

**The Phase 5 demonstration commissions a line; it does not build one.** See §2.
It meets all four of PLAN 5.7's clauses, and the scoping is a catalog limitation
rather than a shortcut.

---

## 8. Throughput

Eight workers is the shipped default, measured: 8 workers 261 steps/s, 12
workers 279, 16 workers 272 -- the last is a regression. Per-worker efficiency
collapses to about 60% at eight, because the machine has eight physical cores
and a Factorio tick loop is latency-bound single-threaded work.

On a machine whose commit is already two thirds used, a training run is close to
the ceiling: the release matrix lost six of nine cells to a memory wall that
reported itself as an empty error string, because Windows refused to start the
subprocess at all. `tools/release_matrix.py` waits for headroom and skips with a
named reason rather than spawning into it.
