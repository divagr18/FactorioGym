# FactorioRL development redirection and coding-agent handoff

Date: 2026-09-08. Scope: development priorities, experiment correctness, and amendments to PLAN.md.

This document implements the user's request to adjust the development trajectory using the implementation review and subsequent transcript review. It is not a new audit result or a report of completed fixes. The book review was paused at the user's request; recommendations here do not claim a new synthesis of the books.

## 1. Intended outcome

Preserve the existing runtime and build toward a shared Factorio playground for compact RL policies, LLM agents, and hybrid controllers. The next flagship capability is an agent constructing a small factory from parts, operating it, and recovering from a demonstrated disruption. Reliable measurements and useful agent interfaces are first-class deliverables.

The near-term priority is not to make three introductory tasks pass by repeatedly changing their generators. A valid difficult task remains useful with a weak baseline. Keep the three-family mastery target as a separate research milestone and report misses honestly.

Do not rewrite the runtime, replace Factorio simulation, weaken embodiment, or start a new algorithm stack by default. Exact stepping, isolated workers, typed interactions, observations, adapters, replay, and recorded evidence are assets to retain.

## 2. Evidence and corrections to carry forward

Read this section before treating any old summary as current fact.

| Finding | Interpretation to use |
|---|---|
| Previous review ran 331 unit/contract tests successfully | Evidence of that checkout's tested behavior; not a new engine verification and not proof of experimental correctness |
| Existing discrete catalog restricts general construction and stable target selection | A product/interface limitation; not an explanation of every learning failure |
| Reference inspection solved the particular repair layouts with existing actions | Those repairs are expressible. Investigate their learned behavior separately from construction limitations |
| Latest transcript describes task v1.6, both faults published, richer repair training, terminal power gaps in training, and a new holdout v3 | A deliberate versioned curriculum redesign. Do not characterize it as an unauthorized silent threshold change |
| User reports the new curriculum running with nine workers and log `D:\rb_cur3.log` | Reported external run status. This handoff does not verify that process or claim a result |
| Older completed delivery results include structural rates 0.92, 0.85, 0.54; power includes 0.77, 0, 0 | Historical results tied to their actual manifests/holdouts, not results of the current curriculum. Filenames containing v3 are insufficient to identify a holdout |
| Current skills are authored controllers selected by a learned policy | Do not describe the controllers themselves as learned skills |
| Existing demonstration starts with preplaced machines | Commissioning evidence, not construction evidence |
| Some transcript diagnoses and numerical claims were retracted | Use raw artifacts and corrected limitations, not the most confident narrative |

Evidence entrypoints:

- Repository implementation: `src/factoriorl/env.py`, `skills.py`, `vecenv.py`, `learn/train.py`, `tools/demonstration.py`, and `tasks/families/plate_line.py` under `src/factoriorl/`.
- Session bundle: `D:\factoriorl-session-2026-09-08\LEDGER.md`, `LIMITATIONS.md`, `conversation\session-78997bdc-earlier.jsonl`, and `conversation\session-3db25a88-current.jsonl`.
- Independent reports: `D:\factoriorl-session-2026-09-08\agent-transcripts\report-ppo-infeasibility-audit.md` and `report-repair-belt-inspection.md`.
- Prior full critique supplied by user: `a local attachment`.

The session bundle may be newer than this checkout. Reconcile versions before editing. These local paths are review provenance, not portable release dependencies. Move only the needed, sanitized evidence into portable release artifacts; do not bundle raw conversations or credentials.

## 3. Operating rules for the next coding agent

1. Inspect current code and manifests before applying a finding; another agent may already have fixed it. Record confirmed, fixed, superseded, or unresolved.
2. Preserve the running curriculum experiment and its environment. Do not kill, restart, overwrite checkpoints, change its checkout, or launch competing jobs merely to follow this document. Offline edits and bounded verification can proceed in isolation.
3. Assume the user-provided RTX 4060 PC and Ryzen configuration. Do not rediscover hardware. Distinguish historical laptop measurements from PC measurements.
4. Preserve immutable old results. A new task version can be better engineering without validating the old experiment's claim.
5. Use small discriminating probes before training matrices. A failed learning run alone does not distinguish an execution defect, insufficient observations, exploration, representation, optimization, or transfer.
6. Never promote a plausible mechanism to a diagnosis without testing it. Do not declare PPO impossible, a representation incapable, or a required sample budget from a few failures or a worst-case theorem.
7. Runtime validity and agent competence have separate statuses. Passing a scripted solver proves feasibility under its access profile, not learnability or a neural policy's competence.

Every investigation records: observation and artifact; hypothesis; competing explanation; discriminating test; result; supported conclusion and remaining uncertainty. A conclusion without its source artifact stays provisional.

## 4. Build sequence and acceptance gates

### R0 — Establish a trustworthy starting point

**R0.1: Reconcile status.** Inventory current task, observation, action, skill, reward, and holdout versions; locate the exact manifests for each cited result. Create `docs/CURRENT_STATUS.md` as the maintained entrypoint. Keep the ledger as history and mark stale handoffs accordingly.

**Gate:** Every current capability has a status and evidence link. Unverified capabilities and running experiments are visibly pending. No older score is attached to the latest curriculum by filename inference. A new agent can find the next work package without reconstructing the entire conversation.

**R0.2: Protect experiment identity.** Record code revision plus dirty-tree identity when applicable, engine/mod versions, resolved configuration, all seed streams including run-ID-derived streams, scene hashes, checkpoint hash, action/assistance profiles, evaluation sampling mode, and exclusions. Existing fields should be reused before adding a second manifest schema.

**Gate:** Two nominally identical comparisons demonstrably score identical scenes. Repeated runs with the same numeric seed but different generated scenes are not labeled identical replications. Baseline cache identity includes the actual wrapper/skill action space and relevant task/reward configuration.

### R1 — Repair experiment correctness before drawing new conclusions

**R1.1: Infrastructure failures in PPO rollouts.** Review `env.step_payload`, vector truncation handling, and the actual MaskablePPO collector. The review found that `excluded_from_metrics=True` excluded reporting but did not prevent training on a broken transition.

Initially prefer explicit rollout failure over implementing fragile transition deletion. Abort the affected collection before its next optimizer update, mark the run failed/incomplete, and preserve the last valid checkpoint. Recovery/resumption can be added with explicit semantics later.

**Gate:** A failure-injection test traverses the real collector path and proves there is no optimizer update using the invalid rollout. A simulator failure is not silently treated as a legitimate time-limit boundary. A successful test of the logger alone is insufficient.

**R1.2: Frozen evaluation exhaustion.** A failed eligible episode can be excluded while consuming its fixed index, leaving the evaluator waiting for a count it can no longer reach.

Implement bounded retries of the same scene identity, or finish with an explicit incomplete-coverage failure. Preserve each failed attempt and prohibit silent substitution with another scene.

**Gate:** Inject a failure at the start, middle, and final allocated scene. Each evaluation terminates with either exact unique coverage or an explicit incomplete result. Completed scenes are counted once; retry limits are enforced. No failure is replaced with an agent failure score.

**R1.3: Temporal accounting.** Choose and document the objective's time unit. The existing skill aggregation uses an undiscounted internal reward sum and a single PPO discount per skill decision. That is a different time preference from the primitive objective.

To preserve a primitive-step discounted objective for a skill of duration k, use `R = sum(gamma**i * r[i], i=0..k-1)` and continuation discount `gamma**k`. Implement matching return/advantage handling; changing reward aggregation alone is insufficient. Specify whether trace decay is measured per decision or per primitive step rather than silently assuming stock GAE is duration-aware. Alternatively retain learning at primitive time with a declared hierarchical controller. Do not advertise either approach as already implemented.

**Gate:** On a fixed trajectory, grouping transitions preserves the intended discounted return, including termination and legitimate truncation handling. Verify shaping cancellation under that same convention. Report policy decisions, primitive transitions, simulated ticks, optimizer updates, and wall time. Flat/skill comparisons state which budget is matched; no claim isolates hierarchy if objective or experience also changed.

### R2 — Make construction and stable interaction expressible

**R2.1: Canonical parameterized actions.** Audit the existing addressed LLM path and reuse it. Define a common semantic contract for movement/navigation, placement position and orientation, target-specific transfer quantities, recipe selection, crafting, and cancellation. Choose exact command names from the existing protocol where possible.

Keep physical reach, collision, inventory, technology, and elapsed game-time enforcement in the runtime. No arbitrary evaluator access. Assistance remains declared; navigation cannot choose a factory design or reveal unseen obstacles.

**Gate:** A scripted conformance client can place and operate a small production chain using only policy-accessible actions, without task-specific runtime shortcuts. Target identity remains stable through movement. Stale/deleted handles, invalid placements, and insufficient inventory have typed observable outcomes. Old catalog artifacts remain runnable under their original profile.

**R2.2: RL action adapter.** Add a factorized or autoregressive policy interface over operation, visible entity target, placement candidate, orientation, item, and amount as applicable. Begin with bounded observed entities and local placement candidates; arbitrary whole-map actions are unnecessary. Parameter masks depend on previously chosen arguments and permitted observations. Handle an empty candidate set explicitly.

**Gate:** RL and LLM adapters can issue equivalent valid semantic actions and produce equivalent state changes from matched states. Tests establish argument/mask consistency and absence of privileged-information shortcuts. A bounded training smoke check verifies sampling, log probabilities, and PPO updates for the new distribution; this is not a mastery claim.

**R2.3: Useful observations.** Publish readable machine statuses, available recipes, inventories, and action failure explanations where legitimately observable. Separate the task objective from optional supplied fault/subgoal hints. Avoid replacing an unresolved multi-fault task with an undocumented automatic subgoal selector.

**Gate:** The task and available interactions are understandable from policy-visible data. Hidden diagnostic state is not smuggled through masks or errors. Both clients' different encodings and assistance are recorded.

### R3 — Construct and commission a real small factory

**R3.1: Feasible bounded construction task.** Start with declared inventory, resources, and terrain. The required functional production chain must be placed by the agent. Explicitly list any preplaced infrastructure. Use a policy-access reference builder first to validate geometry, resources, and timing; it must not complete the evaluated agent's run.

**Gate:** The reference builder completes construction under the same embodied action rules. The scene cannot already satisfy the production objective. Acceptance counts newly constructed required components and sustained output, not just final inventory or one transient plate.

**R3.2: Agent result and replay.** Run a bounded LLM baseline once contracts work; train compact policies as a separate learning track. Preserve failures and avoid embedding the reference build sequence in runtime assistance.

**Gate:** Replay shows agent-issued construction, commissioning, and the complete measurement window. Report supplied knowledge, assistance, game time, decisions, inference usage, and throughput. A failed policy remains a valid baseline result; the construction demonstration gate stays unmet until an agent actually passes.

### R4 — Demonstrate actual recovery

**R4.1: Fix intervention observation freshness.** Route interventions and external time advancement through a path that refreshes the policy observation. Review the old demonstration's direct `session.step` calls and subsequent reads of cached `env._observation`.

**Gate:** Every post-intervention prompt/observation has the correct tick and current permitted state. A regression check reproduces and prevents the previous stale-observation case.

**R4.2: Establish disruption and recovery separately.** Removing inventory fuel need not stop a burner immediately. Measure the actual production effect after accounting for stored energy and material in transit. Define outage and recovery criteria before evaluating.

**Gate:** From a validated common pre-intervention state, compare intervention-plus-agent, intervention-plus-no-action, and preferably undisturbed operation. Use validated save/replay reconstruction if state branching is unavailable; do not assume arbitrary snapshots already work. Show production loss and subsequent sustained restoration attributable to agent actions. Record unsuccessful repairs and collateral production damage.

**R4.3: Separate three tracks.** Directed repair supplies fault locations. Diagnosis-and-repair supplies production goals and ordinary observations. Persistent operation requires monitoring and continued production through unannounced disruptions.

**Gate:** Each track declares its hints and assistance, and has an independent outcome. Current v1.6 directed repair is legitimate. It does not automatically establish diagnosis. Retain one-fault-training/two-fault-test as an optional compositional challenge, not a mandatory defect or mandatory training distribution.

### R5 — Run interpretable learning experiments

**R5.1: Diagnose the current curriculum outcome.** Let its existing run produce evidence. Separate familiar-distribution execution from structural transfer. Log per-scene progress: reaching the fault, valid repair attempt, structural repair, restored flow, sustained objective, and timeout/failure reason. Instrumentation may use evaluator truth for analysis, never as undeclared policy input.

**Gate:** Reports identify where attempts fail. Neither zero success nor high training success alone is called an algorithmic explanation. Both sampled and argmax evaluations are labeled; neither is substituted after seeing a better score.

**R5.2: Controlled probes.** For a suspected terminal-gap shortcut, evaluate the same checkpoint on matched interior/terminal scenes. For multi-fault behavior, distinguish reaching, fixing, retargeting, and completing. Confirm policies' actual actions rather than inferring them solely from generator distributions. The combined v1.6 redesign can test whether the new recipe works; attributing a gain to an individual change requires a separate controlled experiment.

**R5.3: Demonstration learning.** Compare scratch PPO, behavior cloning, and BC-initialized PPO on a stable task/interface. Add DAgger only after the reference policy can supply valid actions from learner-visited states, including off-demonstration recovery states. Check observation/teacher information mismatch and keep teacher interventions out of ordinary evaluation.

**Gate:** Demonstration counts and teacher access are disclosed. Behavioral cloning is justified as a measured training route, not as the only possible solution. Training traces and evaluation scenes are disjoint. Stop after the smallest comparison that resolves the question; no broad matrix without a stated purpose.

### R6 — Release and adoption

Package a reproducible small learning recipe, a construction/recovery demonstration when it passes, replay, task-authoring instructions, and known failures. An environment alpha can ship earlier with valid hard tasks and weak baselines. Do not relabel an alpha as the flagship first public release.

**Gate:** Another user can install, evaluate a provided checkpoint, run the short learning recipe, connect an agent, and author a small task without editing internals. Report actual learning separately from mastery. The first public release still requires a reproducible learning result and real construction/recovery; it no longer requires three families to clear 80% before other development proceeds.

## 5. Exact changes to PLAN.md

These are intentional user-requested plan amendments, not retrospective acceptance of old results:

| Existing section | Amended meaning |
|---|---|
| Section 3, first learning target; 4.5 | Retain the numerical three-family structural mastery target. Separate it from runtime acceptance and first-release reproducible-learning evidence |
| Generalization | Label new instances, new compositions, and new mechanisms/structures separately. Matched difficulty is useful for isolating transfer; deliberately harder composition tests remain legitimate when labeled |
| Failed gates | Investigate validity failures and report competence failures. Do not require task modification merely because a valid learner underperforms |
| Phase 4b | Study authored temporal abstraction with correct objectives and budget accounting. Remove the claim that horizon is the only controllable factor and that one comparison separates abstraction from optimization universally |
| Phase 5 | Add stable parameterized interfaces shared by clients. Reopen construction/recovery acceptance where the old evidence establishes only commissioning or lacks a valid intervention comparison |
| Phase 6 | Keep reproducibility, learning evidence, and demonstrated gameplay requirements. Environment-alpha distribution is permitted before agent mastery |
| Phase 7a | Begin bounded task/intervention infrastructure after required runtime/action contracts, without waiting for Phase 6 publication or a favorable hierarchy result |
| Phase 7b | Accept agent evidence from declared flat, authored-skill, LLM, or hybrid systems; Phase 8 is not a mandatory prerequisite |
| Phase 8 | Reserve learned-skill claims for trained controllers; test transfer and compare controllers/managers separately |
| Dependency map | R0–R1 correctness; R2 expressiveness; R3 construction; R4 recovery; R5 controlled learning; R6 reproduction. Independent packaging and task work can proceed alongside learning |

The dependency order does not authorize overlapping engine workloads on the training machine. Maintain isolation and respect the currently running experiment.

## 6. Claims and tests to avoid

- No claim of best/fastest Factorio environment without comparable workloads, hardware, profiles, and measurements. Engine ticks, agent steps, and API operations are different units.
- No inference that changing run IDs while retaining numeric seeds reproduces identical training scenes.
- No inference that missing serialized manifest fields prove the runtime lacked those features; inspect serialization and code identity.
- No claim that a positive incomplete-task reward proves stopping is optimal. Compare discounted continuation values and actual behavior; reward sign alone is insufficient.
- No claim that pooled entity features cannot represent a relationship without a representation-specific argument. Compare actual encodings and policies.
- No claim that an unmarked fault is necessarily undiscoverable if observations contain diagnostic geometry or machine state.
- No fixed sample requirement inferred from a worst-case bound for a different setting.
- No cached primitive-action random floor reused for a skill-augmented policy.
- No passing unit suite used to substitute for engine or real-collector validation where the defect crosses that boundary.

## 7. First work package to execute

Start with R0 and R1.1–R1.2: reconcile the current checkout with the newer session bundle, verify the two collector/evaluation findings, implement only still-present defects, and add focused failure-injection checks. In an isolated lane, specify R2 by reusing existing addressed actions. Do not launch another release matrix first.

Return a handoff with changed files, exact checks and results, immutable evidence references, unresolved risks, current-run status only if actually observed, and the next bounded task. Preserve this document as direction; maintain day-to-day status in `docs/CURRENT_STATUS.md`.
