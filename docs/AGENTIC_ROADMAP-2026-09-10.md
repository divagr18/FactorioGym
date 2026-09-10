# FactorioRL: agent-first development roadmap

Date: 2026-09-10. Status: implementation handoff; no implementation or new run is claimed here.

## 1. Decision and intended outcome

The next milestone is DeepSeek building as much of a factory as it can from a fresh world. FactorioRL should become a useful place to play, develop, evaluate, and eventually train agents. Compact-policy PPO remains a supported research track, but further optimization of introductory tasks does not gate agent development.

This document takes precedence over the development sequence in DEVELOPMENT_REDIRECTION.md where priorities differ. Preserve its experiment-integrity rules and the existing historical evidence. Read HANDOFF-2026-09-10.md for the latest reported results; this roadmap does not certify them anew. Leave Phase 4.5 open under its existing requirements. Report PPO, imitation, and hybrids separately rather than reinterpret mastery retrospectively.

### User decisions

| Decision | First-run specification |
|---|---|
| Model | DeepSeek V4.1 Flash through the official API |
| Start | Fresh map, ordinary starting items, no existing factory |
| World | Natural terrain and resources; enemies disabled; normal technology progression |
| Tools | Practical game tools and bounded action batches; no agent-written code yet |
| Clock | Continuous simulation, including during inference; game speed 1.0 |
| Budget | At most $5 accounted API cost and 30 minutes of agent-run wall time |
| Objective | Build and expand useful automated production as far as possible |
| Outcome | Preserve partial progress, final save, actions, observations, usage, and replay |

Implementation defaults: base Factorio 2.0.60 without expansion content; structured observations for the model; existing graphical client for human watching when available. No combat, injected faults, factory blueprint, scripted construction solution, or supplied fault markers in the first attempt. Explain tool semantics and public game mechanics freely: hiding recipe ingredients or machine footprints is not a useful challenge.

## 2. Current foundation and concrete gaps

Inspected for this plan: agent loop, memory, runner, adapters, live-watching tool, world initialization, worker save handling, and agent documentation.

- Reuse the provider-independent loop, observation summaries, memory, parameterized interactions, manifests, worker lifecycle, and HTML replay. Memory already exists; extend its actual plan and outcome lifecycle rather than introduce a second memory subsystem.
- Live watching already supports continuous time and speed 1.0. Extract reusable wiring rather than fork another runner. Continuous observations can become stale during inference; action validation must use current state.
- The documented demo is hardwired to plate_line and overwrites tracked evidence. Add a general agent command whose outputs are run-local.
- Existing world initialization is built around authored scenes; its comments explicitly describe disabled natural resources. A normal map must not pass through a destructive scenario reset or receive benchmark starting inventory.
- Existing worker launch accepts a saved world. Build on that for preservation, while separating fresh-world initialization from resume.
- Full crafting, exploration, recipes, research, and placement must be checked end to end through the model-visible interface. Having a Lua verb does not establish that the agent can discover and use it.

## 3. Short horizon: get DeepSeek playing

Execute A0 through A5 in order. Each phase produces its gate evidence before the next dependent phase. Infrastructure success and model competence are separate results; poor gameplay does not fail an otherwise valid first experiment.

### A0 — Freeze the run contract and remove setup ambiguity

**A0.1 Configuration.** Add `factoriorl agent` with task/world selection, model/provider settings, clock mode, seed, spend limit, wall limit, and optional client launch. Reuse configuration resolution and credential redaction. Add an `open_factory` mode without changing existing frozen task definitions or holdouts.

The intended first-run interface is:

```powershell
uv run factoriorl agent --task open_factory --base-url https://api.deepseek.com --model deepseek-flash --api-key-env DEEPSEEK_API_KEY --clock realtime --seed 20260910 --max-cost-usd 5 --max-wall-seconds 1800 --launch-client
```

This is a proposed interface, not a command that currently exists. Retain existing demo and watching entrypoints as compatible wrappers where practical; they must not silently acquire new benchmark semantics.

**A0.2 Provider.** Use the existing OpenAI-compatible transport. Official documentation retrieved on 2026-09-10 names `deepseek-flash` as DeepSeek-V4.1-Flash. Use thinking mode with a maximum of 4096 generated tokens per request, including reasoning where the API accounts for it. Follow the current official thinking/tool-call protocol; preserve required provider continuation fields internally without presenting hidden reasoning as a user-facing trace. Record requested and returned model identifiers and protocol settings. Do not substitute another model after an error.

**A0.3 Limits.** Use one outstanding model request. The $5 cap covers all live provider checks, retries, and the exploratory attempt in this work package. The 30-minute run clock starts immediately before the first gameplay observation after engine readiness. A provider request and every tool wait inherit the remaining deadline. Reserve worst-case request cost before dispatch, including maximum output and all input at the uncached peak rate. Use actual usage when available; retain the reservation for requests with uncertain billing. Stop before dispatch if it could exceed the remaining budget. If a safe input-token upper bound or current pricing cannot be established, fail the paid preflight instead of guessing. Report this as conservative accounting, not an independently verified provider invoice.

**Gate A0:** offline adapter fixtures cover valid output, malformed output, provider error, missing usage, truncated generation, retry accounting, and deadline expiry. A mock delayed response proves no additional request starts after budget exhaustion. Configuration changes reach actual execution and recorded artifacts, not just an argparse field. No paid calls until A1–A4 are ready; the first paid compatibility call comes out of the same $5 allowance.

### A1 — A real fresh-world lifecycle

**A1.1 World creation.** Introduce a separate natural-map initialization path. Use seed 20260910, standard resource/terrain settings and starting area, with enemy spawning disabled. Do not reroll after seeing an inconvenient map. Keep cliffs, trees, water, finite resource deposits, crafting requirements, and normal technology triggers. Obtain the base game's ordinary starting-item definition from the installed game's scenario source and mirror it for the controlled character; record the exact inventory and initialization version. Disable introductory cutscene/wreck bonuses so they cannot supply an accidental construction kit; declare this difference from interactive freeplay.

**A1.2 Character and observer.** The agent controls one embodied character through the existing reach/inventory rules. The viewing client must not create a second productive participant or alter the agent's inventory. Use a non-interacting observer if supported; record viewer presence and any unavoidable perturbation. Failure to launch the viewer should leave headless execution and replay available, with an explicit warning in the result.

**A1.3 Persistence.** Save the initialized world before the first action, periodically every five wall-clock minutes, and at termination. Preserve the final save before stopping the worker. If saving fails, record it and retain the latest verified checkpoint; never report a nonexistent final save. An explicit later resume creates a new run segment linked to its parent and does not reset or duplicate inventory. Do not automatically spend another budget on resume.

**Gate A1:** real-engine probe shows natural deposits and terrain, expected starting inventory, no prebuilt production assets or bonus items, normal recipe locks, and ticks advancing while the controller is idle. Reload a save and verify inventory, machines, technology, and relevant production state survive. Existing benchmark reset fixtures still pass.

### A2 — Tools sufficient to bootstrap a factory

**A2.1 Observation and knowledge.** Expose local visible terrain/resources/entities, stable handles, coordinates, machine footprints and facing, inventories, fuel, status, recipe, and local connection information. Provide queryable recipe ingredients/products, eligible crafting machines, and technology prerequisites from game prototypes. Distinguish static game knowledge from observed world state. Remembered entities retain observation timestamps and are never presented as current sightings. Do not expose unexplored resource locations or evaluator-only state.

**A2.2 Practical operations.** Support inspect, walk to a specified location/entity, mine a specified resource/entity, craft a quantity, place with explicit coordinates/facing, rotate, transfer an explicit quantity, configure a recipe, select research, and wait for a condition. Implement missing behavior through existing typed actions wherever possible. Walking and mining run bounded controllers; they do not teleport, grant resources, or decide the factory layout. Placement uses current collision/reach checks and reports actionable failure reasons. Removal must recover items according to game rules, not delete machinery administratively.

**A2.3 Batches.** Extend the decision format compatibly with a sequence of up to eight explicit tool actions. No loops, arbitrary Python/Lua, or implicit factory planner. Execute serially; validate again before every action; stop at the first failure and report completed, failed, and unexecuted actions. Do not roll back successful game actions. Bound the whole batch to 30 wall-clock seconds and the global remaining budget; an action still running is reported with its operation identity and can be inspected/cancelled. Never blindly resend a mutating action after an ambiguous transport timeout.

**A2.4 Freshness.** Refresh observations after batches and before dispatching actions whose targets may have changed. Include observation tick, execution tick, and elapsed wall time. Condition waits poll at a bounded rate and return completion or timeout. A waiting agent must not leave walking/mining controls engaged unintentionally while it thinks.

**Gate A2:** controlled engine tests demonstrate the constituent chain of gathering resources, crafting, placing, fueling, and obtaining machine-produced output through exactly the public tools. Separately test a stale target, blocked placement, insufficient ingredients, exhausted deposit, partial batch failure, cancellation, and a lost reply. Test scaffolding is not included in DeepSeek's prompt or runtime assistance.

### A3 — Persistent agency and a useful task objective

**A3.1 Objective.** Give the model: build a productive factory from the available starting items, automate gathering and processing, expand useful production, and progress toward electricity, assembly, and research as resources and time allow. State that progress is measured and it should continue until the controller stops the run. Do not prescribe coordinates, machine ordering, a build sequence, or hidden success conditions.

**A3.2 Memory.** Reuse existing memory for explored locations and action outcomes. Allow explicit updates to a compact work plan and notes, with current subgoal, next steps, and unresolved failures. Keep model assertions separate from confirmed observations. Supply current observation, active plan, relevant remembered facts, and recent outcomes on each call; compact bounded older history without dropping outstanding work or repeatedly encountered failures.

**A3.3 Stall handling.** After three identical failed action/target/argument combinations, supply a clear repeated-failure observation and ask for a revised plan. Do not silently select a successful action. Five consecutive provider/parse failures end as model_failure using the existing policy. Gameplay mistakes consume time and remain visible; do not end merely because an arbitrary production target was missed.

**Gate A3:** offline multi-turn fixtures show the plan survives compaction, stale facts remain labeled, failures enter memory, a stopped batch returns control, and evaluator data does not enter the prompt. The model can choose an ordinary tool call or an explicit batch using the same interface.

### A4 — Make continuous play measurable and inspectable

**A4.1 Artifacts.** Every run writes config, manifest, status, decisions, tool events, production samples, summary, initial save, and final/latest save beneath its own runtime directory. Extend existing replay to render batch members and their individual outcomes. Keep secrets and raw private transcripts out of export bundles. Record concise model-supplied action reasons and plan updates; do not require or publish private chain of thought.

**A4.2 Progress.** Measure resources gathered, items crafted, machine-produced outputs by item type, placed functional machines by type, researched technologies, and time to first machine production. Keep cumulative production separate from inventory transfers and handcrafting. Record output deltas in 60-second game-time windows. Call a line unattended only when machine production continues across a full window with no agent transfer, mining, crafting, or placement interventions servicing that line; otherwise report production without that label. Partial windows remain partial. No arbitrary composite score for the first attempt.

**A4.3 Timing and limits.** Report simulated ticks, wall time, model latency, calls, tokens, accounted spend, tool actions, and interventions. Sample lightweight operational metrics approximately every five seconds, including during model calls, without sharing an unsafe concurrent RCON stream; serialize game access through one owner. At the deadline stop new actions, cancel controller-held activity, pause the world for the final snapshot/save, then stop the worker. The paused finalization period is outside gameplay and recorded separately.

**Gate A4:** replay reconstructs a mixed batch with partial failure; production sampling distinguishes handcrafting from machinery; the deadline interrupts an outstanding wait/provider call and preserves artifacts. A provider outage remains distinguishable from a gameplay failure. Tests exercise observed behavior rather than merely asserting configuration fields exist.

### A5 — Run once, then diagnose

**A5.1 Preflight.** Run relevant offline tests and bounded engine probes, check live API compatibility, confirm the requested model and conservative price accounting, then execute the fresh-map attempt. Preserve existing experiments and use an isolated worker/checkout where needed. Assume the user's hardware configuration; do not investigate it again.

**A5.2 The attempt.** One seed, one model, one configuration, remaining $5 work-package allowance, and 30-minute run limit. No mid-run code/prompt edits, hand-built machinery, injected resources, or hidden assistance. Watching is allowed. If intervention is necessary, label and log it rather than presenting the remainder as autonomous.

**A5.3 Report.** Produce a short run report with the replay/save links, actual factory progress, first consequential failure, representative trace evidence, and the next three ranked fixes. Separate interface/runtime defects, inaccessible knowledge, agent decisions, and time/cost limitations. A zero-output run can still complete the diagnostic milestone if its execution and failure are explainable. Do not rerun until the first failure analysis identifies a concrete change; additional paid attempts need their own authorized allowance.

**Gate A5:** another engineer can open the replay, load the saved factory, understand why execution ended, and reconcile stated progress with production/tool records. This is an exploratory result, not a benchmark success claim.

## 4. Medium and long horizon

These are ordered directions, not authorization for unbounded implementation or paid runs. Refine their implementation plans from A5 evidence. Calendar dates depend on the fresh-map/tool gaps; phase gates, not optimistic day estimates, control advancement.

### B — Reliable factory building and operation

1. Fix the highest-impact first-run interface defects, preserving the original run. Add each reproduced defect as a behavioral regression.
2. Declare a small development set of five fresh-map seeds before subsequent comparisons. Track per-seed progress and costs; hold prompts/tool versions fixed within each comparison. Keep development seeds distinct from a later untouched evaluation set.
3. Add production milestones for smelting, powered production, assembly, and research without ending the world at each milestone. Support explicit save/resume and continued expansion.
4. Introduce a separate build-operate-recover scenario after construction works: controlled fault, observable production loss or avoided loss, repair, resumed output, then expansion. Keep it distinct from the natural fresh-map attempt.

**Gate:** publish repeated, inspectable factory attempts, including failures, and a recovery demonstration with a paired no-action control. Competence claims name the actual start state, tools, assistance, clock, and budget.

### C — Reusable agent procedures and multiple agent implementations

Add agent-written programs when traces show repeated sequences or model round trips limiting progress. Programs may call only the public game tools; enforce execution/iteration/time limits, cancellation, and per-call tracing. No raw RCON, evaluator access, unrestricted filesystem, network, or credentials. Define this runtime in a separate implementation plan before building it; do not call in-process exec a sandbox.

Keep the reference agent separate from the environment so external harnesses can use the same tools. Expose task discovery, run lifecycle, observations, actions, and artifacts through a documented Python interface first. Add other protocol adapters only for an identified integration. Compare tools/batches versus programs on the same declared tasks and budgets; label assistance differences.

**Gate:** a second agent implementation runs without modifying environment internals, and a program-based agent's entire game interaction remains inspectable and bounded.

### D — Public adoption and credible evaluation

Ship an installable package, engine setup instructions, downloadable sanitized example artifacts/checkpoints where distribution rights allow, a short first-run command, task-authoring examples, and compatibility information. Test the complete route from a clean machine/checkout rather than only the unit suite.

Separate evaluation tracks for construction, operation, diagnosis/recovery, and expansion. Declare layout, recipe, fault-type, and subgoal-count shifts explicitly. Freeze held-out evaluation only after development iteration; repeated tuning on a frozen set makes it a development set. Report success/progress alongside cost and time, across seeds and repeated stochastic runs. Continuous-time and exact-step results occupy separate profiles because provider latency affects the former.

**Gate:** an independent user can install, connect their own agent, run a task, inspect the output, and author a task through documented interfaces. Adoption and useful contributed runs are evidence toward becoming a reference environment; a single impressive replay is not.

### E — Learning that improves the agent

Retain the reproducible PPO baseline and close bounded outstanding evidence checks. Pause broad PPO tuning for introductory mastery. Prioritize future learning experiments only when they address a demonstrated agent bottleneck: repeated execution skills, recovery from learner-visited states, tool selection, or planning informed by action outcomes.

Use collected agent trajectories with provenance and access labels for imitation and later RL. Scripted solvers, LLM demonstrations, privileged labels, and reward-only training remain distinguishable. Before DAgger, verify that an expert can label valid next actions on representative learner-visited states; restarting a solver successfully on a few perturbations is insufficient. Before training a hybrid, compare against the same agent with the existing scripted executor.

**Gate:** learning improves an identified capability under a declared comparison, including end-to-end agent cost/reliability where relevant. Do not count a new optimizer or another sweep as product progress by itself.

## 5. Changes to the standing plan and handoff rules

- Insert A0–A5 as the immediate execution sequence; prioritize B–D after the first report. Phase 4.5 stays a separate unmet research target and does not block this sequence.
- Extend existing APIs additively: general agent command, open_factory lifecycle, bounded batch decisions, explicit real-time profile, budget accounting, and batch-aware replay. Existing frozen task/action contracts must not silently change.
- Do not start a runtime rewrite, replace the real game, build a new RL algorithm stack, or implement arbitrary-code execution in A0–A5.
- Do not spend another cycle relitigating whether BC counts as learning. Publish method-specific evidence, reconcile missing floors/seeds, and move on.
- Record the open-questions brief's discrepancies as unresolved until checked against versioned generator/evidence artifacts. Do not use its subgoal-count explanation as an established diagnosis.
- Before each phase, check whether another coding agent has already implemented it. Amend this plan with measured findings rather than duplicate work. Keep historical results immutable and never overwrite tracked evidence from exploratory runs.

### Reference for the provider configuration

[DeepSeek official models and pricing](https://api-docs.deepseek.com/quick_start/pricing/), retrieved 2026-09-10: `deepseek-flash` maps to V4.1 Flash, official OpenAI-format base URL is `https://api.deepseek.com`, and both thinking and non-thinking modes are supported. Listed peak rates are $0.30 per million uncached input tokens and $1.20 per million output tokens. Snapshot applicable pricing at implementation preflight; aliases and prices can change. [Thinking-mode guide](https://api-docs.deepseek.com/guides/thinking_mode) defines the provider-specific continuation behavior to verify before the live run.
