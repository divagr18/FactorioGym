# Design

The architecture and behavioural contracts this implementation is written
against, and the evaluation requirements it is measured by.

Docstrings throughout `src/` cite this document by section -- `(DESIGN.md 2.1)`,
`DESIGN 5.3 requires ...`. Numbers below 4 are the sections here. Numbers of
the form 5.3 refer to phases of the original staged build plan, which is not
carried into this repository; the requirement each one names is quoted inline
where it is cited, so nothing depends on having it.

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
| Development model | One owner coordinating many coding agents |

Keep combat, Space Age, multiplayer agents, pixel-only RL, approximate simulators and Linux support outside the initial release sequence.

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

**And the target is not currently met, by arithmetic rather than by attempt.** the RL theory notes puts a defensible negative at ~0.4 M steps for `navigate` and ~15–27 M for the H=300 families — 55–100 h per family per seed at 75 steps/s, so “~2,000 h ≈ 3 months” for six families at three seeds. Measured end-to-end throughput is 25–43 steps/s at 8 workers, and the published 50k-step cells are 0.2–0.3% of that floor. So those numbers are a **budget-limited** miss, which PLAN section 3 already distinguishes from a finding about the task: “an underpowered baseline makes a negative result an artifact of the budget rather than a finding about the task.” The route that fits the machine is a shorter-horizon variant or teacher-seeding (R5.3), not a lower bar.
