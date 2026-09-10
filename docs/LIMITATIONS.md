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

**Greedy evaluation can livelock on a rejected action, and it costs real
success.** This is the concrete mechanism behind the argmax penalty, not a
restatement of it.

Four of `deliver`'s frozen holdout scenes -- 3007, 3026, 3033, 3040 -- were
missed by all six cells of the paired 4.4 comparison, every one of the 24
attempts burning exactly 120 decisions and ending on a `precondition`
rejection. A per-decision trace of one declared checkpoint on those scenes
shows what happened: the policy issues `take_iron-plate_20` **120 times in a
row**, is rejected 120 times, and never moves. Closest approach to the source
is 8.6-9.2 tiles, which is exactly where the character started.

The loop is a fixed point, and the reason is structural. A rejected action does
not change the world, so the next observation is identical to the last, so a
*deterministic* policy selects the same rejected action again. There is no
escape without stochasticity, and the episode is guaranteed to exhaust its
budget. That is why `out_of_decisions` sits at exactly 5 in five of the six
cells: a livelock always burns the full budget, so the count is a property of
how many scenes trigger it rather than of the policy's competence.

**The same checkpoint solves all four scenes when sampled** -- 29, 24, 8 and 16
decisions. So the scenes are solvable, the achievable ceiling on this row is
not capped at 0.96, and an earlier version of this entry which said they "may
be unsolvable" was wrong. It was written from the failure pattern alone, before
the trace existed; the pattern was real and the inference from it was not.

What this adds to the argmax finding recorded elsewhere. Sutton & Barto's
Example 13.1 explains why a stochastic policy can be *better* under state
aliasing. This is stronger and more specific: under action rejection a greedy
policy has an **absorbing** failure mode, so the greedy/sampled gap on the
structural row (0.84 against 0.98) is not only a scoring difference but partly
a count of scenes where argmax cannot act at all. Any report quoting a greedy
rate on a task whose actions can be refused should say so.

Traced with `tools/trace_scenes.py`; evidence in
`docs/evidence/r5-deliver-hardcore-trace.json`. Note the trace needed
`--skills` to match the checkpoint: `shaping_comparison` enables skills by
default, so the 4.4 cells ran a 19-action space rather than the 13 primitives,
and the action space is part of what a checkpoint was trained against.

**`max_decision_steps` is a primitive-step budget, so under the skills action
space a policy gets far fewer decisions than the name implies.** This entry
previously claimed a defect in episode-length reporting. There was no defect;
the claim is withdrawn and this is what was actually going on.

Two step counts exist and they are not the same quantity. `evaluate_parallel`
counts one per `vec.step`, which is a **policy decision**; the environment's
`_steps` counts **primitive actions**. In the primitive action space they
agree. Under skills they do not, because `SkillEnv.step` runs a whole skill
through `runner.run(skill)` and one decision expands into many primitive steps
-- a solved `deliver` episode reads **4 decisions against 18 primitive steps**.

`env.py:744` truncates on `_steps >= max_decision_steps`, so that budget is
consumed in primitive units. A `deliver` episode nominally has 120, and a
skills policy whose average skill expands into ~4 primitives therefore gets
about 30 decisions. That is why published rows showed truncation at 12-118
decisions against a "120" budget: consistent all along.

What was actually wrong was the diagnosis. Reading a decision count against a
primitive budget, I recorded 103 of 113 failures as "explained by neither
budget", added an `EpisodeResetFailed` mechanism that does not fire in those
runs, and changed `per_scene["steps"]` to report primitives. All three are
reverted. `steps` is the decision count, because "where did the attempt fail"
is a question about decisions the policy made, and `primitive_steps` sits
beside it, because their ratio is what lets a reader interpret either.

Two things this leaves standing. The instrumented episode was still worth
running: it confirmed 30 ticks per decision, `info["steps"]` tracking
`env._steps` exactly, and truncation firing on the decision condition. And the
budget's units are a genuine trap worth documenting -- comparing a skills run's
episode lengths against a primitive run's without the ratio compares different
quantities.

Dead ends still worth recording. `production.episode_ticks` is 30 x the
*primitive* count, not the decision count. And `simulated_ticks` is derived as
`primitive_steps x decision_ticks` rather than measured, so ratios computed
from it describe two counters and not the clock.

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

## 8. Distribution and adoption

Written for R6's environment alpha. Everything here is a gap a stranger will
hit, listed so nothing unbuilt is implied to work (PLAN 6.4).

**A source checkout is the only supported install.** `pip install factoriorl`
resolves and imports, and then cannot start a worker. Two independent reasons:
`workspace_root()` is `Path(__file__).parents[2]`, which from
`site-packages/factoriorl/` resolves into the Python installation rather than a
project, so `mod_source_dir()` and `runtime_dir()` point at nothing useful; and
`mod/` is outside `src/` so it is not in the wheel at all, which makes
`package_mod` raise before the engine is launched. `factoriorl demo` and
`factoriorl replay` additionally shell out to `tools/`, which is also unpackaged.
The `rl` extra is genuine and `FactorioEnv` does import under it -- what does
not work is the part that needs the mod and the binary.

Until R6 the README claimed a unit test proved the runtime imports cleanly
without numpy and gymnasium. **No such test exists.** The nearest one imports
`factoriorl.env` -- requiring both -- and asserts only that torch and
sb3-contrib did not leak in. The claim is withdrawn rather than backfilled,
because the property it described was never checked.

**A watched run is a demonstration, not a measurement.** `--launch-client`
attaches a real Factorio client. The joiner is a spectator with no character, so
it cannot act — but it still creates a `LuaPlayer` no measured run has, it can
run console commands and pause the world (`allow_commands`,
`only_admins_can_pause_the_game: False`), and it reorders RCON replies. The run
records `measurement: demonstration` and the observed inversion count. Bind is
`127.0.0.1`, so only a same-machine viewer can join.

**`--launch-client` is wired for open worlds only.** A benchmark task can still
be watched through `tools/watch_agent.py`, which has its own client handling —
including the hardcoded `"watched": true` that the new path deliberately does
not copy.

**Replay does not cover trained policies.** `factoriorl replay` reads
`decisions.jsonl`, which only the language-model loop writes, and refuses a
training run with a message saying so. A trained policy's per-decision actions
are recorded nowhere in the ordinary path; `tools/trace_scenes.py` records them
for named scenes, deliberately outside the evaluation path, and in an aggregate
shape replay cannot read. So of R6's five gate clauses, replay serves the agent
half and not the learning half.

**`factoriorl agent` now exists** (roadmap A0). It runs a registered task or an
open world, selects either adapter with `--adapter`, enforces a spend cap and a
wall clock, and writes run-local artifacts.

**`demo` is still hardwired to `plate_line`, and deliberately so.** It drives
five phases around an injected fuel outage, which `factoriorl agent` has no way
to express; rebuilding that on the new command would have risked changing what
PLAN 5.7's published claim means. What was fixed instead: it writes run-local by
default (`--publish-evidence` to touch the tracked evidence file), it runs under
a spend cap, and its flags are declared on the subcommand rather than swallowed
by `nargs=REMAINDER`, which made `factoriorl demo --help` print nothing.

**Every entrypoint that can bill a provider now goes through one accounting
layer.** `factoriorl.agent.provider.build()` returns an adapter already wrapped
in its cap; there is no argument to it that yields an unguarded one.

**An open world is not a benchmark task, and nothing measured on it is
comparable to one.** `open_factory` has no success predicate, no layout
families and no reward components. It is deliberately outside the task registry
-- see `src/factoriorl/worlds.py` for why registering it would break
`validate_all` and both holdout tests, and why freezing it would orphan every
piece of evidence citing `holdout_v3`'s content hash.

**Resource patches were computed, transmitted and read by nothing.** Anything
further from the character than `resource_detail_radius` (12 tiles) reaches the
observation only as a per-name aggregate under `resources.patches`, and the
mod's own comment recorded that "`encoders.encode` and every task predicate read
`resources.tiles`; nothing reads `patches`". On a painted benchmark scene this
was invisible, because a declared scene puts its ore within a few tiles. On a
generated map it was fatal: measured on a natural spawn, the nearest iron ore
was 28 tiles away and 29 resource entities sat inside the sensor radius while
the agent's prompt said "RESOURCES: none in sensor range". The prompt now
renders the aggregate, marked as too far to address directly. **The RL encoder
still reads only `resources.tiles`**, so a trained policy remains blind to
anything beyond 12 tiles -- unchanged, and now written down.

**The open world now saves and resumes** (roadmap A1.3): before the first
action, every five wall-clock minutes, and at termination, with
`--resume-from <run id>` continuing from a run's newest verified checkpoint.
Measured end to end in `docs/evidence/a1-world-lifecycle.json` — a placed
machine, the character's inventory and the force's research all survive a
save, a worker teardown and a relaunch.

Three things about it are worth knowing before relying on it:

*A checkpoint is verified, not assumed.* `game.server_save` is deferred to the
end of a tick and there is no completion event, and the mod cannot check —
Lua has no `io` and no `game.file_exists`. So the mod reports only that the
request was *issued*, and Python waits for the file to appear and stop growing
before recording anything. A save that never materialises is recorded as a
failure and the last verified checkpoint is kept.

*A resume does not restore the agent's knowledge.* `open_world(fresh=False)`
leaves the world, the inventory and the research exactly as the save holds
them, but handles are episode-scoped and the agent's memory is rebuilt, so the
resumed segment does not remember what it built. It has to look.

*A pre-save request in flight is unresolvable after a resume.* `begin_episode`
clears the dedup ledger, so `request_status` for a request issued before the
save answers `unknown`. It has never mattered because nothing produced a save;
it can now.

**Autosaves remain off.** Their interval is *game* time, so at `game.speed 90`
a 60-minute interval fires roughly every 40 seconds of wall clock. Checkpoints
are driven from Python's wall clock instead.

**A run's cost is conservative accounting, not an invoice.** Token counts come
from the provider's own `usage` block and prices from a table snapshotted on a
recorded date (`src/factoriorl/pricing.py`). Every rate is the peak rate, an
unreported cache split is priced as all-miss, and a call whose usage never
arrives is charged at its worst-case reservation and flagged. The number is an
upper bound this repository can defend.

**The engine build is pinned exactly and cannot be relaxed.** 2.0.60 build
83512, refused otherwise by `factoriorl doctor` and again at worker launch.
This is not conservatism: the observation and action contracts are written
against one build's prototype table, so a different build is a different
environment. It does mean the project will not run on whatever Factorio a user
already has, and that the download has to come from the release archive rather
than from Steam.

**Windows only in practice.** Nothing is deliberately Windows-specific and the
engine discovery chain has no non-Windows default, but nothing has been run
anywhere else, so treat any other platform as untested rather than supported.

**Seven of eleven published release-evidence files cannot be traced to the
scenes that produced them.** Each `phase4-release-*.json` cites its holdout by
`content_hash`, and seven cite a hash no committed holdout has -- earlier
revisions that were regenerated instead of versioned. They are excluded from
the release bundle, and `MANIFEST.json` lists them by name with the hash they
cite so the exclusion is legible rather than silent. The four `-v3-` files
resolve, against `holdout_v2`.

---

## 9. Throughput

Eight workers is the shipped default, measured: 8 workers 261 steps/s, 12
workers 279, 16 workers 272 -- the last is a regression. Per-worker efficiency
collapses to about 60% at eight, because the machine has eight physical cores
and a Factorio tick loop is latency-bound single-threaded work.

On a machine whose commit is already two thirds used, a training run is close to
the ceiling: the release matrix lost six of nine cells to a memory wall that
reported itself as an empty error string, because Windows refused to start the
subprocess at all. `tools/release_matrix.py` waits for headroom and skips with a
named reason rather than spawning into it.
