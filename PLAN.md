# FactorioRL: strategy and phased implementation handoff

> **Development amendment — 2026-09-08:** Read [the development redirection handoff](docs/DEVELOPMENT_REDIRECTION.md) before executing this plan. Its R0–R6 work packages and acceptance gates govern the current development sequence. The amendment separates runtime readiness, reproducible learning, and benchmark mastery; it does not retrospectively accept old experiments or demonstrations. Historical thresholds remain recorded below. Current task versions and pending experiments must be reconciled from artifacts, not inferred from phase labels.

## 1. Project goal and locked decisions

Build an independent Factorio environment for training compact RL policies, developing language-model agents, and comparing hybrid systems in the same game world.

The flagship demonstration is an embodied agent that builds a factory, increases production, recovers from disruption, and continues toward a peaceful base-game rocket launch.

The first public release must contain:

- A reproducible learning result on the user’s PC. **Hardware correction (2026-09-07):** this said “RTX 4060”; the machine is an RTX 3050 Laptop GPU, 4.3 GB. See the training-machine row below.
- An agent that constructs and repairs a small automated production line.
- A shared runtime enforcing the same underlying game rules.
- An inspectable replay explaining what happened.
- Installation, evaluation, and task-authoring instructions that another researcher can follow.

### Confirmed constraints

| Area | Decision |
|---|---|
| Initial platform | Native Windows |
| Training machine | **Measured 2026-09-07:** RTX 3050 Laptop GPU (4.3 GB, sm_86) and Ryzen 7 5800H. This row said “RTX 4060 and Ryzen 5 5600” until then, and every document in the repository repeated it; the GPU was wrong and the CPU claim was the wrong Ryzen. Confirmed against `torch.cuda`, `Win32_VideoController` and `Win32_Processor`, and run manifests now capture GPU name, VRAM, capability and torch version at runtime so it cannot drift again. No further hardware discovery |
| Repository | Independent project in `D:\FactorioRL` |
| Embodiment | Physical character from the beginning |
| Initial learning | Compact policies, followed by learned skills and hybrid planning |
| Initial observations | Local structured state |
| Images | Optional screenshots for agents and replay inspection |
| Agent models | Local inference plus optional API providers |
| Initial assistance | Navigation and bounded typed interactions |
| Main game target | Peaceful base-game rocket launch |
| Existing frameworks | No dependency on FLE architecture or compatibility |
| Development model | One owner coordinating many coding agents |

Keep combat, Space Age, multiplayer agents, pixel-only RL, approximate simulators, Linux support, and FLE adapters outside the initial release sequence.

### Definition of success

The project must become useful at several scales:

1. An individual can reproduce a learning curve.
2. An agent developer can connect a model without modifying the runtime.
3. A researcher can create a new task and obtain trustworthy measurements.
4. A contributor can add a mechanic without breaking existing experiments.
5. Longer episodes expose meaningful planning, recovery, and adaptation challenges.

Adoption, reproducibility, and experiment quality are release requirements.

## 2. Architecture and behavioral contracts

### Runtime ownership

Use a Factorio Lua mod for authoritative game interaction, observations, task state, and outcome measurement.

Use Python for:

- Worker lifecycle and transport.
- Gymnasium integration.
- Task configuration and experiment manifests.
- Training and evaluation.
- Agent orchestration.
- Replay generation and inspection.

Start with one Factorio server process per environment. Keep worker configurations, ports, saves, and logs isolated. Run training and workers together on the PC; a network execution service is unnecessary initially.

Use versioned JSON requests over RCON. Optimize round trips before considering a different transport.

Multiple surfaces are a later optimization experiment. They must demonstrate both isolation and a measured advantage before replacing process isolation.

### Embodied actions

The policy-accessible runtime supports:

- Directional movement.
- Mining.
- Crafting.
- Placement and rotation.
- Inventory transfer.
- Recipe selection.
- Research selection.
- Waiting.
- Cancellation of supported ongoing actions.

Actions must respect the relevant collision, reach, inventory, technology, and time constraints.

Scenario initialization, inventory provisioning, fault injection, and unrestricted state inspection belong to an evaluator-only interface. Agent clients must not receive arbitrary Lua execution or evaluator commands.

Provide two initial profiles:

| Profile | Capabilities |
|---|---|
| Primitive | Individual typed interactions and directional movement |
| Assisted | Known-terrain navigation and bounded batches expanded into primitive execution |

Navigation may find a route. It must not design the factory, choose a production strategy, or inspect unexplored obstacles.

### Simulation time

Use exact stepping, with a default decision interval of 30 game ticks.

- The world pauses between decisions.
- Primitive RL transitions use a fixed interval.
- Ongoing actions retain progress across transitions.
- Assisted tools may execute several transitions, but preserve every underlying action and tick.
- Long actions expose running, completed, failed, or cancelled status.
- Benchmark game time, wall-clock time, and inference time are separate measurements.

Use native game behavior wherever it can express the required character interaction. Any necessary approximation must be documented in the action profile and covered by tests.

The runtime must also expose `snapshot` and `restore` for a *mid-episode* state, not only reset-to-scenario. Restoring a scenario proves reset correctness; it cannot return to a state reached partway through an episode, and without that every decision-time search, teacher, or counterfactual method is unimplementable — including using our own simulator to label its own data. A restore must reproduce what the world digest reports, so the existing digest is the acceptance test. Belt contents, crafting progress, burner fuel, production statistics, entity handles, and the evaluator-side reward state are all part of the state or the snapshot is not one.

### Observations

The initial profile exposes a 32-tile-radius local sensor region, character state, inventory, task description, and declared recipe/technology information.

Explored-map memory records last-observed timestamps. It must not present distant machine state as current.

One canonical structured observation supplies:

- A spatial tensor and padded entity representation for compact policies.
- Typed objects and concise summaries for language-model agents.
- Replay overlays.

Keep full observations authoritative. Add deltas only after reconstruction tests pass.

Entity references are stable within an episode. Destruction and reset invalidate them explicitly.

Screenshots must be aligned to a recorded tick and obey the selected observation profile. Evaluator views and hidden fault annotations cannot enter policy inputs.

### Public interfaces

Expose:

- `FactorioEnv.reset(seed, options)`
- `FactorioEnv.step(action)`
- `FactorioEnv.render()`
- `FactorioEnv.close()`
- An agent client for observation, typed actions, navigation, and execution status.
- Task specifications for setup, goals, interventions, budgets, rewards, and success conditions.
- CLI commands for diagnostics, demonstrations, training, evaluation, and replay.

Every run records the game build, mod version, protocol version, task version, observation profile, assistance profile, reward configuration, model configuration, and seeds.

### Error handling

- Invalid actions return structured reasons and do not silently become another action.
- Batches report partial execution; they do not promise rollback.
- Duplicate requests must not repeat mutations.
- Stale episode requests fail explicitly.
- Worker crashes are infrastructure failures, not ordinary task failures.
- Uncertain transport outcomes must be resolved through request status or worker recovery before further mutation.
- Recovery must preserve evidence of the interrupted run.

## 3. Research and evaluation requirements

### Training defaults

Start with PyTorch and Maskable PPO using bounded, versioned action catalogs for introductory tasks.

Use compact spatial and entity encoders. Do not introduce language-model fine-tuning, recurrent policies, or custom RL algorithms before the basic baseline works.

Masks and candidate enumeration may use declared observations and public mechanics. They must not use hidden solutions, evaluator fault labels, or privileged path information.

Separate:

- Sparse task outcomes.
- Optional shaping.
- Evaluation metrics.

Log each reward component individually.

### Task sequence

The introductory task families are:

1. Navigate to a work site.
2. Retrieve and deliver items.
3. Mine ore and produce plates.
4. Supply an existing furnace.
5. Repair a belt defect.
6. Restore a power connection.

Later tasks combine these into production, growth, recovery, and adaptation in persistent worlds.

Every generated task needs a solvability argument backed by generation constraints and a reference solution. Randomness alone does not establish task quality.

Each family also declares a random-policy ceiling, and a family whose random baseline approaches the acceptance threshold is not a usable benchmark at that difficulty; raise the requirement rather than report the number.

No shaping component's achievable cumulative total may reach the sparse success weight. Shaping that can pay as much as finishing turns the subgoal into the goal, which is the classic failure of rewarding progress instead of outcome. Bound every event-derived component and assert the bound in a test.

### Generalization

Maintain separate training, validation, and test splits.

- Training uses the curriculum distribution.
- Validation supports debugging and model selection.
- Test results use frozen layouts and seeds excluded from tuning.
- Structural holdouts vary layout families, resource arrangements, production combinations, and fault combinations.

Report unfamiliar seeds and unfamiliar structures separately.

Declared splits are a claim about content, so they must be measured as content rather than asserted by name. For every family, publish:

- The count of distinct generated scenes per layout family, with a declared minimum. A holdout whose generator admits one scene is evaluated once, no matter how many episodes are run against it.
- Measured difficulty descriptors (path length, reference-solution length, budget slack) for training and test splits. Match difficulty when isolating structural transfer; explicitly label deliberately harder compositional tests rather than treating them as invalid by definition.
- Structural separation on at least one descriptor that is not difficulty, so the split is doing the work its name claims.

A gate that compares layout-family *names* does not check any of this. Two families that generate identical content, or a test family that falls through to the training layout, must fail validation rather than be reported as transfer.

### Measurements

For construction, require sustained output over a measurement window.

For recovery, report:

- Time to restore production.
- Production lost during disruption and repair.
- Repair resources.
- Damage to other production.
- Final sustained output.

For progression, report milestones reached under declared game-time and compute budgets.

For agent systems, also report inference latency, token usage when available, and API cost when applicable.

Do not combine these into one undocumented composite score.

State the statistical method alongside the numbers:

- A declared policy for multiple comparisons, since families and seeds are compared together and an uncorrected per-comparison interval overstates confidence across a matrix.
- Paired analysis whenever two arms are evaluated on the same generated scenes; analysing paired measurements as independent discards the pairing and widens intervals for no reason.
- Whether an evaluation samples the policy or takes its argmax, because the training objective is defined over the sampled policy and the two do not measure the same quantity.
- Enough evaluation episodes in every arm, including baselines, that the comparison can resolve the effect claimed. An underpowered baseline makes a negative result an artifact of the budget rather than a finding about the task.

### First learning acceptance target

**Amended role:** The following is the structural mastery milestone, not a prerequisite for developing later gameplay or distributing an environment alpha. First-release learning evidence requires a reproducible, declared experiment showing learning relative to appropriate baselines, with per-seed results and uncertainty; it does not require this entire mastery milestone. A miss remains a miss and is not retrospectively relabeled a pass.

At least three introductory families must achieve 80% held-out success across 100 evaluation episodes per training seed, with three training seeds.

**For this mastery target, held-out means unfamiliar structures.** New instances of familiar layouts measure within-distribution generalization; they are not sufficient evidence of structural transfer and are not necessarily memorization. Every run also publishes the unfamiliar-seed rate on training layouts. Report new instances, new compositions, and new mechanisms/structures separately, with generator changes disclosed.

No general result guarantees that a policy trained on one set of structures performs on a disjoint set; the training distribution puts no mass there. That is a reason to expect a flat baseline to miss this bar and to record the miss, not a reason to move it. A missed threshold is reported under section 4's rules for failed gates.

A structural threshold is only meaningful if the held-out split is genuinely distinct content, so the generator diagnostics above are a precondition for reporting against it: a holdout whose generator admits one scene is evaluated once regardless of the episode count.

At least one qualifying family must involve production or repair. Navigation and delivery alone are insufficient.

Publish per-seed results, aggregate uncertainty, random and scripted baselines, training time, and failure examples.

The target is an overnight introductory training recipe on the training machine. This is a goal to verify, not an assumed capability.

**Hardware correction (2026-09-07).** This sentence said “on the 4060”, which names hardware that is not present: the machine is an RTX 3050 Laptop GPU with 4.3 GB. Any figure published against “the 4060” would be a false provenance record.

**And the target is not currently met, by arithmetic rather than by attempt.** `docs/research/rl-theory.md` puts a defensible negative at ~0.4 M steps for `navigate` and ~15–27 M for the H=300 families — 55–100 h per family per seed at 75 steps/s, so “~2,000 h ≈ 3 months” for six families at three seeds. Measured end-to-end throughput is 25–43 steps/s at 8 workers, and the published 50k-step cells are 0.2–0.3% of that floor. So those numbers are a **budget-limited** miss, which PLAN section 3 already distinguishes from a finding about the task: “an underpowered baseline makes a negative result an artifact of the budget rather than a finding about the task.” The route that fits the machine is a shorter-horizon variant or teacher-seeding (R5.3), not a lower bar.

## 4. Coding-agent operating rules

### Work package format

Every agent assignment must contain:

```text
Task ID:
Objective:
Dependencies:
Owned components:
Inputs and contracts:
Required behavior:
Explicit exclusions:
Deliverables:
Verification commands:
Acceptance evidence:
Handoff notes:
```

The assigning agent must identify dependencies before dispatch. An agent must not redefine an upstream contract merely to make its own implementation easier.

### Ownership lanes

| Lane | Responsibility |
|---|---|
| Integration owner | Contracts, dependency order, merges, release gates |
| Runtime agent | Lua mod, character actions, stepping, observations |
| Python systems agent | Workers, transport, Gymnasium, experiment lifecycle |
| Learning agent | Encoders, policies, training, evaluation |
| Tasks agent | Generators, reference solutions, rewards, interventions |
| Experience agent | Agent client, replay, CLI, documentation |
| Verification agent | Independent tests, leakage checks, reproducibility review |

These are logical roles. One agent may fill several roles sequentially.

Use additional agents for bounded independent work. Do not create overlapping ownership of core runtime files or protocol definitions.

### Parallelism

Parallel work begins after the relevant interface fixtures exist.

Examples:

- Observation encoders and replay rendering can proceed against the same frozen observation fixtures.
- Task generators and RL wrappers can proceed after reset and task contracts stabilize.
- Local and API agent adapters can proceed against the same client contract.
- Documentation can proceed alongside implementation using verified commands.

Dependent work may use clearly marked fixtures, but a phase cannot pass on fixtures alone.

### Definition of done

A task is complete only when:

- Required behavior is implemented.
- Verification runs against the intended environment.
- Outputs and commands are recorded.
- Public behavior is documented.
- Known limitations are stated.
- No placeholder remains on the required path.
- An integration check confirms compatibility with its dependencies.

“Implemented but not run,” “works with a mock,” and “tests skipped because Factorio was unavailable” are incomplete states for engine-dependent work.

### Handling failed gates

When a gate fails:

1. Preserve the failing run and configuration.
2. Identify whether the failure is a runtime defect, task defect, learning limitation, or performance limitation.
3. Create a bounded corrective task.
4. Re-run the failed gate and affected upstream checks.
5. Record any proposed scope change separately.

A valid task with an underperforming learner need not be modified. Separate runtime/task validity failures from unmet competence targets. Subsequent infrastructure and gameplay work may proceed on valid contracts while the competence result remains below target.

Agents must not lower success thresholds, weaken embodiment, expose privileged state, or replace learning with scripted behavior to pass a gate.

## 5. Release sequence

| Release | Required outcome |
|---|---|
| Engine proof | Deterministic embodied interaction and reliable reset |
| Environment alpha | Six task families and usable Gymnasium integration |
| First public release | Reproducible learning, construction-and-repair agent, replay |
| Persistent-factory release | Build, expand, disrupt, recover within one episode |
| Hybrid-learning release | Transferable learned skills and manager/planner comparisons |
| Long-horizon release | Documented progression toward a peaceful rocket |
| Research reference release | Stable task authoring, benchmark submissions, reusable datasets |

Each release remains runnable while later phases are developed.

## 6. Appended phase-by-phase build plan

### Phase 0 — Repository foundation and engine feasibility

**Purpose:** establish a working engine boundary before building abstractions on assumptions.

**Dependencies:** none.

**Primary owners:** integration, runtime, Python systems.

#### 0.1 — Establish the project skeleton

Deliver:

- Python package managed with `uv`.
- Lua mod source.
- Test directories separating unit, contract, and engine integration checks.
- Example configurations.
- A contribution guide and agent assignment template.
- A progress ledger tracking phase and gate status.

Use standard Python formatting and linting tools. Keep runtime dependencies separate from training and development dependencies.

**Acceptance:**

- A clean environment installs the Python package.
- Lint and non-engine tests run through documented commands.
- Importing the package does not launch Factorio.
- Generated saves, credentials, checkpoints, and large trajectories are excluded from source control.

#### 0.2 — Establish worker launch and shutdown

Implement explicit game executable configuration, isolated worker directories, process launch, startup detection, connection establishment, and shutdown.

Pin the actual game build used for integration tests. Do not silently switch builds during development.

**Acceptance:**

- One worker starts from a documented command.
- Startup failures identify the missing executable, invalid save, unavailable port, or engine error.
- Closing the worker releases its process and connection.
- Cleanup targets only processes and directories owned by that worker.

#### 0.3 — Prove exact stepping

Implement a minimal request that advances a known number of ticks and returns the resulting tick.

**Acceptance:**

- Test 1, 30, and 120-tick advances.
- Repeat a mixed sequence of advances without accumulated drift.
- Waiting between requests does not advance entity simulation.
- Report observations only after the requested interval completes.

#### 0.4 — Prove a physical action

Implement movement and one inventory interaction using a character.

**Acceptance:**

- Collision blocks movement.
- An out-of-reach interaction fails.
- A valid transfer changes both inventories by the same quantity.
- The action cannot create items.

#### 0.5 — Prove reset and replay

Load a reference starting state, execute a short action sequence, reset, and repeat.

**Acceptance:**

- Character state, inventories, relevant entities, task state, and recorded outcomes match.
- Repeat for at least ten resets.
- The proof runs through the same transport intended for the environment.

**Phase 0 exit gate:** one command launches, acts, steps, resets, verifies repeatability, and exits cleanly. Record latency and reset measurements, without claiming scalability yet.

---

### Phase 1 — Protocol and reliable worker lifecycle

**Purpose:** make the engine boundary safe to build against.

**Dependencies:** Phase 0.

**Primary owners:** Python systems and runtime.

#### 1.1 — Freeze protocol version 1

Define typed requests, responses, errors, request identifiers, episode identifiers, and action execution status.

Create representative fixtures for successful actions, invalid actions, ongoing actions, reset, and observation.

**Acceptance:**

- Python and Lua agree on the fixtures.
- Unknown protocol versions fail clearly.
- Unsupported action types cannot reach arbitrary game execution.
- Missing required data produces a useful error.

#### 1.2 — Add duplicate and stale-request handling

Track applied requests within the current episode.

**Acceptance:**

- Resending a placement or transfer request does not apply it twice.
- Requests from a previous episode are rejected.
- A timeout after execution can be resolved without guessing whether the action happened.

#### 1.3 — Implement worker supervision

Add configurable deadlines, liveness checks, captured engine output, and restart from a known state.

**Acceptance:**

- Forced worker termination produces a classified infrastructure failure.
- A replacement worker starts without overwriting the failed run’s evidence.
- Ports and directories are not accidentally shared across workers.

#### 1.4 — Add multi-worker orchestration

Support independent workers before adding a training vector wrapper.

**Acceptance:**

- Two workers accept different actions and maintain different states.
- Resetting one worker does not affect the other.
- Both close cleanly after a failed parent run.

**Phase 1 exit gate:** protocol contract tests and lifecycle fault tests pass against actual Factorio workers.

---

### Phase 2 — Embodied actions and partial observations

**Purpose:** establish the game semantics that all agents share.

**Dependencies:** Phase 1.

**Primary owners:** runtime; verification reviews independently.

#### 2.1 — Implement the action matrix

Complete movement, mining, crafting, placement, rotation, transfer, recipe selection, research selection, waiting, and supported cancellation.

Maintain an action matrix stating:

- Preconditions.
- Inventory effects.
- Reach requirements.
- Time behavior.
- Failure outcomes.
- Cancellation behavior.

**Acceptance:**

- Every action has a successful example and relevant invalid examples.
- Mining and crafting cannot bypass their required time.
- Placement cannot bypass cost, collision, or technology restrictions.
- Failed actions do not leave undocumented partial changes.

#### 2.2 — Implement ongoing actions

Represent operations that span multiple decision intervals.

**Acceptance:**

- Progress survives repeated `step` calls.
- A target becoming unavailable causes a recorded failure.
- Cancellation stops the operation at a documented boundary.
- Completion is reported once.

#### 2.3 — Implement local structured observations

Expose terrain, resources, visible entities, character state, inventory, and declared public information.

**Acceptance:**

- Objects outside the local sensor region are absent unless represented as remembered observations.
- Remembered observations include age.
- Moving into and out of visibility updates records correctly.
- Evaluator-only information is absent.

#### 2.4 — Add entity identity and lifecycle

Implement stable episode-scoped references and invalidation.

**Acceptance:**

- An unchanged visible entity retains its reference.
- Destroyed references fail explicitly.
- Rebuilt entities do not inherit stale identity accidentally.
- A reset creates a new episode identity boundary.

#### 2.5 — Implement observation and action profiles

Store profile names and versions in configurations and run manifests.

**Acceptance:**

- Primitive and assisted capabilities are distinguishable.
- Tests identify which profile supplied information or assistance.
- Results cannot omit their profile metadata.

**Phase 2 exit gate:** an embodied scripted agent can navigate, gather resources, produce an item, and interact with machinery without privileged mutations.

---

### Phase 3 — Task engine, reset correctness, and RL spaces

**Purpose:** provide solvable, reusable learning tasks.

**Dependencies:** Phase 2.

**Primary owners:** tasks and Python systems.

#### 3.1 — Implement task specifications

Support setup, task description, randomization, budgets, success checks, reward components, and intervention schedules.

**Acceptance:**

- A task can be added without editing worker management.
- Configuration errors fail before training begins.
- Task version and fully resolved configuration appear in the run manifest.

#### 3.2 — Build the six introductory families

For each family provide:

- A fixed reference scenario.
- A randomized generator.
- A scripted solution.
- A random baseline.
- A success predicate.
- A failure and timeout definition.

Use held-out generators or layout families where necessary, not only a different seed.

**Acceptance:**

- Reference solutions pass all fixed scenarios.
- Randomized generation passes a declared solvability suite.
- Task success cannot be satisfied by initial inventory alone unless that is explicitly the task.

#### 3.3 — Implement fast bounded-scenario reset

Reset terrain/entity setup as required, character inventory, crafting state, technology, task counters, action state, observations, and identifiers.

**Acceptance:**

- Run at least 500 consecutive resets across introductory tasks.
- Compare selected resets with freshly loaded reference workers.
- No cumulative item, technology, timer, or reward leakage appears.
- Failed fast reset falls back explicitly to full reload.

#### 3.4 — Implement Gymnasium spaces and wrappers

Use fixed-shape tensors and padded entity features for each versioned training profile.

Define bounded action catalogs for introductory tasks. Catalog membership may be task-specific but must be declared and reproducible.

**Acceptance:**

- Environment checker passes.
- Observations remain within declared spaces.
- All masks have a valid wait/no-op fallback.
- Termination and truncation are distinguishable.
- Reset and close work after an interrupted episode.

#### 3.5 — Implement reward accounting

Provide sparse outcomes and configurable shaping.

**Acceptance:**

- Reward components sum to the returned reward.
- Repeated observation does not generate reward.
- Item transfer loops and dismantle/rebuild loops do not produce unintended progress.
- Success remains unchanged when shaping is disabled.

**Phase 3 exit gate:** all six task families run through Gymnasium with real workers, reference solutions, and clean resets.

---

### Phase 4 — Compact RL baseline and compute characterization

**Purpose:** show practical learning before expanding the game scope.

**Dependencies:** Phase 3.

**Primary owners:** learning and verification.

#### 4.1 — Establish the baseline policy

Implement a compact encoder for local spatial state, entities, inventory, and goal information. Use Maskable PPO initially.

Pin training dependencies and record hyperparameters in resolved configurations.

**Acceptance:**

- Forward passes and action sampling work for every introductory task.
- Model checkpoints restore both inference and resumable training state.
- Evaluation runs without training dependencies changing environment behavior.

#### 4.2 — Run single-task learning experiments

Start with one navigation task and one interaction task, then add production or repair.

**Acceptance:**

- Learning curves exceed random behavior.
- Evaluation uses separate seeds.
- The agent receives no scripted solution labels during ordinary PPO training.
- Failed runs remain available for inspection.

#### 4.3 — Profile end-to-end training

Benchmark 1, 2, 4, and 8 workers where resource limits permit.

Measure:

- Steps and simulated ticks per second.
- Reset time.
- Worker and learner waiting time.
- Observation serialization time.
- Peak GPU memory.
- Wall-clock time to the acceptance score.

Select the default worker count from measured results.

**Acceptance:**

- A reproducible profiling report identifies the dominant bottleneck.
- Worker count is configurable.
- More workers are not described as faster unless measurements show it.

#### 4.4 — Test shaping dependence

Compare sparse and shaped training on at least one tractable task.

**Acceptance:**

- Both use the same success predicate.
- The report distinguishes faster learning from changed task difficulty.
- Any shaping exploitation becomes a task or reward defect with a regression test.

#### 4.5 — Produce the release learning result

**Amendment:** This subsection records the three-family mastery experiment. Its numerical acceptance criteria remain unchanged; the amended first-release learning requirement above is separate. Do not run repeated matrices solely to unblock construction or agent-interface development.

Run three training seeds and frozen held-out evaluation.

**Acceptance:**

- At least three task families meet the agreed 80% threshold on the **test (structural) split**.
- At least one is production or repair.
- Evaluate 100 held-out episodes per family per training seed.
- Publish the unfamiliar-seed rate on training layouts beside every structural rate, and never select a candidate family or a hyperparameter on the test split.
- Publish checkpoints, resolved configurations, curves, and failure examples.
- Record whether the overnight training target was met.

**Phase 4 exit gate:** another run can reproduce the learning procedure and evaluate the provided checkpoints without manual intervention.

---

### Phase 4b — Temporal abstraction ablation

**Purpose:** measure the effect of authored temporal abstraction relative to the flat catalog under explicitly declared objectives, assistance, and budgets. A comparison can establish an effect in its tested setting; it cannot generally separate abstraction from optimization or establish that more flat training could never work.

Run this when the action interface and temporal accounting are stable. Existing reference controllers make a bounded comparison practical. Horizon, exploration, representation, task distributions, and optimization can all affect learning; no favorable result here is required before building valid persistent tasks.

**Dependencies:** Phase 4. The flat baseline is the comparison point and must exist first.

**Primary owners:** learning, tasks.

#### 4b.1 — Extract task-agnostic skills

Turn the recurring phases of the reference solutions into named skills with declared preconditions and effects.

**Acceptance:**

- Every skill is expressed in terms the observation already exposes.
- No skill encodes a family's solution. A skill may be "walk to the nearest entity of a named type" or "insert N of an item into a target"; it may not be "repair the belt gap". A skill library that names the answer moves privileged information into the action space, which section 2 forbids for masks and which is forbidden here for the same reason.
- Skills are shared across families rather than defined per family, since sharing is the only route by which they can transfer.
- A test reconstructs each skill's availability from a policy-visible observation alone.

#### 4b.2 — Expose skills as temporally extended actions

Offer skills alongside the primitive catalog, with availability expressed through the existing action mask.

**Acceptance:**

- The primitive catalog remains available and unchanged; skills are added, not substituted.
- A skill reports running, completed, failed, or cancelled like any other ongoing action, and its underlying ticks and actions are preserved in the trace.
- The catalog hash, action-profile version, and skill-library version are recorded in the manifest.
- No new learning algorithm is introduced. Section 3's ordering rule stands: the baseline works first.

#### 4b.3 — Run the ablation

Train flat and skill-augmented policies on paired evaluation scenes with declared seed streams and budget conventions. Follow R1.3 in the development handoff for duration-aware objective accounting. Report primitive transitions, simulated ticks, policy decisions, optimizer updates, and wall time; equal policy-decision counts do not imply equal environment experience.

**Acceptance:**

- Both arms are evaluated on the same held-out episodes and reported with the seed and structural rows required above.
- The result is reported whichever way it comes out, including no improvement.
- Wall-clock and sample cost are reported per arm, since a skill layer that only wins on wall clock is a different claim from one that wins on samples.

**Phase 4b exit gate:** a published comparison of flat and skill-augmented policies on identical holdouts, with the skill library's independence from task solutions demonstrated by test.

---

### Phase 5 — Assisted agent interface and replay

**Purpose:** give language-model agents a practical way to play while preserving embodiment.

**Dependencies:** Phases 2–3. May proceed alongside Phase 4 after contracts stabilize.

**Primary owners:** experience and runtime.

#### 5.1 — Implement known-terrain navigation

Provide navigation to a declared destination or reachable interaction position.

**Acceptance:**

- Routes use only known terrain.
- Newly discovered obstacles trigger replanning or a useful failure.
- Movement consumes game time.
- Navigation cannot teleport or complete the target interaction implicitly.

#### 5.2 — Implement bounded batches

Allow sequences of typed interactions with explicit execution limits.

**Acceptance:**

- Every underlying action is recorded.
- Errors identify the failed operation and completed prefix.
- Batch execution respects character reach and time.
- Cancellation and time limits behave predictably.

#### 5.3 — Implement model adapters

Use a provider-independent agent loop with adapters for a configurable local inference endpoint and optional API providers.

**Acceptance:**

- Provider responses become validated typed actions.
- Malformed outputs produce bounded retries or a recorded failure.
- Credentials remain outside run artifacts.
- Local and API models use the same observation and action contracts.
- Inference latency and available usage data are recorded.

#### 5.4 — Implement persistent agent memory

Maintain explored terrain, last-seen entities, action outcomes, and the current plan separately from authoritative world state.

**Acceptance:**

- Memory updates cite observations or action results.
- Stale facts remain distinguishable from current observations.
- A failed plan does not disappear from the record.
- Context compaction does not introduce evaluator information.

#### 5.5 — Build replay inspection

Create a local viewer with timeline, map view, character actions, inventories, throughput, errors, and interventions.

**Acceptance:**

- Selecting an event shows the relevant observation and action.
- Assisted actions can be expanded.
- Replay can be inspected without contacting the model provider.
- Debug overlays are identified as evaluator information.

#### 5.6 — Add optional screenshots

Capture selected frames for agent requests and replay checkpoints.

**Acceptance:**

- Frames are associated with a verified game tick.
- The viewport respects the observation profile.
- Screenshot capture is disabled in ordinary RL training.
- Unsupported capture configurations fail clearly rather than returning stale images.

#### 5.7 — Produce the agent demonstration

**Evidence correction:** The existing preplaced-machine demonstration establishes commissioning, not construction. Reconcile and correct acceptance metadata without rewriting historical traces. Execute R2–R4 in the development handoff before accepting the full demonstration: stable parameterized actions, agent-built machinery, fresh post-intervention observations, demonstrated production loss, and a matched no-action continuation.

Construct an automated plate line, inject a documented disruption, and restore sustained production.

**Acceptance:**

- No undeclared human repair or factory-building script completes the run.
- The line operates during a measurement window without manual replenishment.
- The replay shows construction, disruption, and recovery.
- Report total game time, wall time, assistance profile, and model usage.

**Phase 5 exit gate:** the demonstration is reproducible and inspectable, with a local-model baseline and optional API-model results.

---

### Phase 6 — First public release and independent reproduction

**Purpose:** make the completed work usable outside the development session.

**Dependencies:** Phases 4–5.

**Primary owners:** integration, experience, verification.

#### 6.1 — Create the installation path

Provide setup diagnostics, explicit executable configuration, example configuration, and one-command demonstration entrypoints.

**Acceptance:**

- Instructions work in a clean Windows Python environment.
- Users do not need to edit package internals.
- Diagnostics distinguish game setup, connection, dependency, and model-provider problems.

#### 6.2 — Package release artifacts

Include checkpoints, evaluation configurations, task versions, sample trajectories, and a concise limitations document.

**Acceptance:**

- Artifact hashes and version metadata are recorded.
- Evaluation does not require paid APIs.
- Large artifacts are downloadable separately from source.
- No secrets or user-specific absolute paths are embedded.

#### 6.3 — Run independent reproduction

Assign an agent that did not implement the training path to follow the public instructions.

**Acceptance:**

- Provided checkpoints evaluate successfully.
- The short training example runs.
- The agent demonstration and replay commands work.
- Every undocumented manual step becomes a release blocker.

#### 6.4 — Freeze the release

Create a release checklist and compatibility statement.

**Acceptance:**

- Required engine, contract, task, and evaluation tests pass.
- Known limitations are stated accurately.
- Deferred functionality is not advertised as available.

**Phase 6 exit gate:** publishable release containing actual learning, agent play, and debugging tools.

---

### Phase 7 — Persistent factories and intervention benchmarks

**Purpose:** connect construction, growth, repair, and adaptation.

This phase is two jobs with different dependencies, and separating them keeps the benchmark from waiting on the agent:

- **7a — benchmark construction (7.1, 7.2, 7.3, 7.5).** Persistent stages, the production curriculum, interventions, and structural holdouts. None of it depends on the agent being capable, and all of it is useful independently of who scores on it.
- **7b — agent results (7.4 and the exit gate).** Recovery evaluation and the end-to-end persistent episode. Evaluate any declared flat, authored-skill, LLM, or hybrid controller. Learned skills are a research direction, not a mandatory prerequisite.

**Dependencies:** Validated runtime and relevant action contracts for 7a; validated tasks and intervention measurements for 7b. Publication in Phase 6 and learned skills in Phase 8 are not blockers for bounded persistent-task development.

**Primary owners:** tasks, runtime, learning.

#### 7.1 — Add persistent task stages

Support goals that change without resetting the world.

**Acceptance:**

- Inventory, machines, depletion, and research persist.
- Stage transitions do not refill resources or repair the factory.
- Rewards do not duplicate earlier milestone credit.

#### 7.2 — Extend the production curriculum

Add electricity, circuits, red science, and green science in playable increments.

**Acceptance:**

- Each increment has a reference scenario, scripted validation, and agent evaluation.
- Success requires sustained output.
- Existing tasks remain compatible with their pinned versions.

#### 7.3 — Implement interventions

Add belt defects, recipe changes, power disconnection, supply reduction, resource depletion, and competing demand.

**Acceptance:**

- Intervention parameters are evaluator-only.
- The agent observes consequences through its normal interface.
- Interventions occur at recorded ticks.
- Fault generation preserves a documented feasible recovery path.

#### 7.4 — Implement paired recovery evaluation

Run an uninterrupted baseline and an intervened run from equivalent starting factories.

**Acceptance:**

- Measure production lost over time.
- Account for pre-existing buffers.
- Record repairs that damage other outputs.
- Include unresolved failures in the report.

#### 7.5 — Establish structural holdouts

Hold out factory layouts, fault combinations, and resource arrangements.

**Acceptance:**

- Test factories are not selected using agent performance.
- Publish split definitions and generator versions.
- Report ordinary-seed and structural generalization separately.

**Phase 7a exit gate:** the persistent stages, curriculum increments, interventions, and structural holdouts are built, validated by scripted solutions, and published with their generator versions.

**Phase 7b exit gate:** one persistent episode demonstrates construction, increased demand, disruption, recovery, and continued expansion.

---

### Phase 8 — Learned skills and hybrid planning

**Purpose:** make useful behavior reusable and measure the value of hierarchy.

**Dependencies:** Phases 4 and 7.

**Primary owners:** learning and experience.

#### 8.1 — Define the skill contract

Skills expose inputs, preconditions, termination conditions, timeouts, and failure reports.

**Acceptance:**

- Skills operate through the same embodied runtime.
- A failed precondition does not become a silent world mutation.
- Execution produces primitive traces and elapsed game time.

#### 8.2 — Train transferable skills

Start with navigation, delivery, furnace supply, and local repair.

**Acceptance:**

- Evaluate on unseen layouts.
- Compare learned execution with existing assisted baselines.
- Record resources, ticks, and failure frequency.

#### 8.3 — Train a manager policy

Allow a compact manager to choose among skills.

**Acceptance:**

- Skill duration is handled explicitly in manager training.
- The manager cannot exploit inconsistent time accounting.
- Compare against a fixed sequencing baseline.

#### 8.4 — Add language-model skill selection

Expose the same skill library to local and API planners.

**Acceptance:**

- Learned skills are identified separately from authored navigation.
- Planner decisions and skill failures are inspectable.
- The planner can recover from a failed skill rather than assuming success.

#### 8.5 — Run the hierarchy comparison

Compare primitive policies, assisted agents, learned-skill managers, and language-model planners using learned skills.

**Acceptance:**

- Match task and observation settings where comparisons claim equivalence.
- Report differences in assistance explicitly.
- Demonstrate a measurable benefit in success, training efficiency, execution time, or inference cost.

**Phase 8 exit gate:** at least one learned skill transfers and provides a documented advantage in a larger task.

---

### Phase 9 — Long-horizon progression toward a rocket

**Purpose:** extend the system without losing reproducibility.

**Dependencies:** Phases 7–8.

**Primary owners:** tasks and runtime, with learning and agent evaluations at each step.

#### 9.1 — Oil and fluids

Add required fluid observations, pipe interactions, refining, and chemical production tasks.

**Acceptance:**

- Tasks detect starvation, blockage, and incorrect recipe configuration.
- Policies cannot inspect fluid state outside their observation profile.
- A production chain operates over a sustained window.

#### 9.2 — Advanced intermediates

Add advanced circuits and the intermediate chains required for later science.

**Acceptance:**

- Shared-resource competition is represented.
- Agents must resolve bottlenecks without scenario resource injections.
- Reference solutions verify task feasibility.

#### 9.3 — Later science and research progression

Extend technology and production requirements incrementally.

**Acceptance:**

- Research requires actual game resources and elapsed time.
- Each milestone has a resumable evaluation scenario.
- Full progression and intermediate-save results are labeled separately.

#### 9.4 — Expansion and depletion

Add distant resource acquisition and replacement of exhausted supplies.

**Acceptance:**

- Character travel and logistics cost count.
- Previously explored regions remain subject to stale information.
- Expansion cannot depend on evaluator-provided resource coordinates.

#### 9.5 — Rocket production and launch

Implement the remaining mechanics needed for the chosen base-game rocket objective.

**Acceptance:**

- Verify completion from authoritative game state.
- Preserve the complete run manifest and trajectory.
- Record any intervention or restart.
- A full-game claim requires a start-to-finish run without undeclared assistance.

**Phase 9 exit gate:** publish the strongest verified progression result. Do not mark the full rocket objective complete until the complete run exists.

---

### Phase 10 — Research tooling and experimental branches

**Purpose:** support questions beyond task completion.

**Dependencies:** a stable persistent-factory release.

**Primary owners:** runtime, tasks, and research tooling agents.

#### 10.1 — Validated checkpoints

Support full-worker save checkpoints for experiment continuation and branching.

**Acceptance:**

- Restore engine and mod state together.
- Replayed action sequences produce matching relevant outcomes.
- Exact restoration is not conflated with reconstructed scenes.

#### 10.2 — Counterfactual prediction tasks

Ask agents to predict declared production consequences, execute the intervention in a branch, and score against the outcome.

**Acceptance:**

- Prediction targets, units, and horizons are explicit.
- Evaluator branch results are hidden until prediction submission.
- Use metric-specific errors rather than an arbitrary norm over heterogeneous state.

#### 10.3 — Observation and assistance ablations

Compare structured local state, memory, optional vision, and varying action assistance.

**Acceptance:**

- Each profile is versioned.
- Information and execution differences are disclosed.
- Results identify the system being measured, not only the model name.

#### 10.4 — Dataset export

Export demonstrations, failed trajectories, repairs, and intervention branches.

**Acceptance:**

- Include provenance and observation/action profiles.
- Keep evaluator-only annotations in separate fields or files.
- Document train/test overlap and permitted uses within the benchmark.

**Phase 10 exit gate:** at least one complete research experiment can be reproduced using released tools and artifacts.

---

### Phase 11 — Adoption, benchmark maintenance, and measured optimization

**Purpose:** make the environment dependable enough for others to build on.

**Dependencies:** Phase 6 onward; work continues alongside later gameplay phases.

**Primary owners:** integration and experience.

#### 11.1 — Task-authoring tutorial

Provide a walkthrough adding a task without modifying core runtime code.

**Acceptance:**

- A new task includes generation, reference solution, reward, and success tests.
- The example runs through the ordinary CLI and evaluation path.

#### 11.2 — Agent and policy tutorials

Provide examples for a compact policy, a language-model client, and a learned skill.

**Acceptance:**

- Examples use public APIs only.
- Each example includes a small reproducible evaluation.
- Internal implementation knowledge is unnecessary.

#### 11.3 — Benchmark submissions

Define required manifests, trajectories, checkpoints where applicable, and assistance disclosures.

**Acceptance:**

- A submitted result can be reproduced or independently inspected.
- Invalid or incomplete submissions are identifiable.
- Benchmark version changes do not overwrite old comparisons.

#### 11.4 — Performance improvement queue

Use profiles to choose optimizations: payload reduction, observation deltas, reset improvements, worker scheduling, or surface batching.

**Acceptance:**

- Every optimization has a before/after measurement.
- Semantics and outcome checks remain unchanged.
- Surface batching must pass isolation tests for research, randomness, resets, and task state before adoption.

#### 11.5 — External-use feedback

Track installation failures, custom-task friction, runtime bugs, and reproducibility failures.

**Acceptance:**

- Recurring failures become tests or documented fixes.
- New features do not displace unresolved correctness defects.
- Release notes distinguish behavior changes from performance improvements.

**Phase 11 exit gate:** the project supports independent experiments and contributions with stable, documented contracts.

## 7. Dependency map and handoff checkpoints

### Critical path

The current execution order is R0–R6 in [the development handoff](docs/DEVELOPMENT_REDIRECTION.md). Packaging can proceed alongside gameplay development; valid hard tasks do not wait for mastery.

```text
Existing Phases 0–3: preserve validated runtime and task contracts
    → R0–R1: Reconcile evidence and correct experiment handling
    → R2: Shared expressive actions for RL and LLM clients
    → R3: Actual construction and commissioning
    → R4: Measured disruption and recovery
    → R5: Controlled learning comparisons on stable interfaces
    → R6 / Phase 6: Independent reproduction and first public release

Validated contracts → Phase 7a: Persistent benchmark infrastructure
Validated recovery + declared controller → Phase 7b: Persistent agent result
Stable tasks → Phase 8: Learned skills and hybrid-control experiments
Verified production/progression contracts → Phase 9: Rocket progression
```

Phase 5 work can run alongside Phase 4 on stable contracts. Phase 6's first public release requires the amended reproducible-learning evidence and the actual construction/recovery demonstration; it does not require completion of the three-family mastery target or a favorable Phase 4b result.

Phase 4b informs controller development without blocking valid benchmark construction. Phase 7a can proceed on validated runtime/action contracts; Phase 7b requires valid recovery measurements and an evaluated controller, but does not require that controller to use learned skills.

Phase 10 begins once persistent-state checkpointing can be validated. Phase 11 begins with the first release and continues throughout development.

### Required handoff at every phase boundary

The phase owner supplies:

- Completed task IDs and remaining work.
- Verification commands and results.
- A representative successful run.
- Relevant failure examples.
- Performance measurements where applicable.
- Contract or configuration changes.
- Known limitations.
- The next phase’s prerequisites.

The integration owner records one of three statuses:

| Status | Meaning |
|---|---|
| Accepted | All required gates passed with evidence |
| Needs correction | Implementation exists, but a gate failed or evidence is missing |
| Blocked | A named external dependency prevents the remaining verification |

A phase is never accepted because its implementation is large, its demonstration looks convincing, or the next phase would be more interesting.

The first major handoff target remains concrete: **a clean Windows installation can train a small embodied policy, run an agent that builds and repairs a factory, and inspect both through the same runtime and replay tools.**
