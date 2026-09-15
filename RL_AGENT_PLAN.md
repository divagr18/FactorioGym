# FactorioRL learner plan: close T3 and T4

> **PAUSED — 2026-09-15.** This track is paused while a fast early-game simulator
> is built. That work trains a small RL policy from scratch in C and keeps real
> Factorio as the held-out verifier; it continues on branch `sim-parity`. Nothing
> below has been withdrawn, but three defects were found in the T2 work before
> pausing, and anyone resuming must not inherit the claims they undermine:
>
> 1. **The log-probability "parity" check compares a tensor with itself.**
>    `training/run_gemma_group.py` computes `logprobs` as `log_softmax` of
>    `generated.scores`, and `recomputed` as `compute_transition_scores(sequence,
>    generated.scores, normalize_logits=True)`, which normalizes those same
>    scores. Both sides come from one set of processed logits, so agreement is
>    guaranteed and proves nothing. It is not the fresh forward-pass
>    recomputation T2-C.4 requires. It replaced an earlier real recomputation in
>    `453ef32`, after `4943d44` and `c74203b` tried to make that one agree.
> 2. **`construct_smelting_line` pays for hand-loading a furnace.**
>    `machine_produced` is `produced - handcrafted - mined`
>    (`mod/factoriorl/world.lua`), so plates smelted from hand-mined ore count as
>    machine output, and `run_verification` (`src/factoriorl/env.py`) has no
>    provenance check. Placing a furnace, fuelling it, hand-mining ore and
>    feeding it scores 1.0 without a drill. `docs/evidence/a4-production.json`
>    already shows hand-mined ore turning into "machine" plates. The fix is
>    scheduled on `sim-parity` because the simulator reuses this task. Decision 4
>    below ("hand-crafted output receive[s] no reward") is not true of the code
>    as it stands.
> 3. **The launchers point at a machine layout that does not exist here.**
>    `tools/launch_t2_group.ps1` and `_launch_gemma_group.ps1` hard-code
>    `D:\FactorioRL-agentic-t2` / `/mnt/d/FactorioRL-agentic-t2` (absent on this
>    PC), the WSL subnet `172.29.224.0/20`, and `/root/factoriorl-t0/.venv`, as
>    does `training/run_t2_service.sh`.
>
> Also unverified: the "four same-scene episodes … zero reward" group in section 3
> has no evidence file, and no rollout throughput has been measured. And the branch
> is not lint-clean: `uv run ruff check .` reports 31 errors, all introduced here
> (`master` passes) — 22 in `training/`, 6 in `src/factoriorl/agent/loop.py`, and
> one each in `agentic/bridge.py`, `agent/policy.py`, `tools/agentic_bridge.py`
> and a rollout test. `tests/unit` and `tests/contract` pass (1282).

**Status:** implementation handoff (paused 2026-09-15, see above)  
**Date:** 2026-09-12  
**Starting branch:** `codex/agentic-t2`  
**Implementation baseline:** `c877fc7`  
**Target machine:** Windows host `pc`, Ubuntu 24.04 under WSL, RTX 4060 8 GB, Ryzen 5 5600  
**Learner:** `unsloth/gemma-4-E2B-it-unsloth-bnb-4bit` with LoRA  
**Teacher:** DeepSeek V4.1 Flash through the official OpenAI-compatible API

## 1. Outcome

Close T3 with a reproducible base-model evaluation, a versioned DeepSeek demonstration dataset, and an SFT adapter that can take valid native Factorio actions and earn nonzero verifier reward on frozen validation scenes.

Close T4 with one honest GRPO feasibility result. The result may be positive or negative, but it must compare the base model, the selected SFT checkpoint, and every GRPO seed under the same frozen interface and evaluation protocol. It must preserve enough evidence to replay an episode and recompute the sampled-token log probabilities.

This cycle trains one useful capability in real Factorio: construct and operate a smelting line. It does not claim that the small learner can play the whole game. The learned adapter should later be callable as a skill by the open-world agent, which is how the RL and agentic tracks join.

## 2. Decisions that are already settled

1. **Use the canonical agent interface.** Extract a reusable policy-facing component from `src/factoriorl/agent/summary.py`, `parsing.py`, and the relevant parts of `loop.py`. DeepSeek, Gemma, evaluation, and training must receive the same public game facts, native action names, native handles, and validation rules.
2. **Remove the temporary alias layer.** The custom `SYSTEM`, `policy_view`, and `HANDLE_OPTIONS` implementation in `training/run_gemma_group.py` was a useful probe, not the training contract. Executable movement from that prompt is not evidence of task understanding. Do not add another synthetic identifier namespace.
3. **Constrain syntax only.** A JSON grammar or equivalent decoder constraint may guarantee a parseable envelope. It must not select an action, repair a bad handle, or hide a semantic error. Semantic mistakes remain visible learner behavior and receive the normal public error receipt.
4. **Use native verifiable reward.** Training reward comes from `construct_smelting_line` verification: normalized newly machine-produced iron plates. Parsing success, valid JSON, plausible prose, placements, or hand-crafted output receive no reward.
5. **Treat the whole episode as one GRPO sample.** The initial objective uses terminal episodic reward. All learner-generated assistant tokens in the episode contribute to the loss; system text, observations, tool receipts, and teacher text do not.
6. **Keep the task and interface frozen within an experiment.** A prompt, task, reward, catalog, parser, observation renderer, split, or generation change creates a new experiment ID. It is never silently applied midway through a run.
7. **Optimize for the 4060 without weakening evidence.** Use the verified 4-bit base model and train only LoRA parameters. Use microbatch size one and gradient accumulation over complete groups. Do not block T3 or T4 on vLLM: the currently tested vLLM path cannot load this model correctly. A vLLM path may replace generation only after adapter loading, sampling, and log-probability parity pass.
8. **Defer W&B.** Write complete local JSONL, JSON, CSV, and checkpoints. Logging must be structured so a W&B sink can be added later without changing the experiment.
9. **Use sequential groups first.** Four attempts start from the same scene and frozen policy, with a complete reset between attempts. Add concurrent workers only after correctness and a measured throughput benefit.
10. **Do not use the structural holdout.** T3 and T4 use the frozen training and validation splits. The 100-scene structural holdout remains sealed for T5.

## 3. Current state and the actual boundary of T2

The following is already implemented and should be retained:

- the `construct_smelting_line` task and action-locked verifier;
- reference success and random/no-action floor checks;
- the authenticated Windows bridge with an explicit `--game-speed 120` setting;
- append-only rollout JSONL with payload hashes;
- sequential four-attempt same-scene resets;
- frozen Gemma generation in 4-bit on the RTX 4060;
- sampled-suffix logit computation that fits in 8 GB;
- stable scene identity independent of reset-volatile entity handles;
- invalid bridge calls recorded rather than erased.

T2 is **provisionally demonstrated, not closed**. The last compact-prompt probe produced four same-scene episodes with eleven model decisions, eight bridge-executable actions, three invalid calls, one completed verification, and zero reward. It exposed the interface mismatch but did not satisfy the log-probability, loss-mask, exact-prompt, or independent-world gates.

The current artifact schema stores prompt token IDs but not the exact messages or rendered prompt. It does not pin the tokenizer, chat template, prompt version, parser, catalog, observation renderer, sampling configuration, or model revision. It also folds final verification into the last tool result. Those omissions must be fixed before demonstrations or on-policy data are collected.

## 4. Definition of done

### T3 is closed when

- the canonical policy interface is shared by the open-world agent and learner path;
- the unmodified Gemma learner has a complete 32-scene validation baseline;
- the DeepSeek compatibility preflight passes without exposing credentials;
- a versioned training-only teacher dataset contains at least 20 verified successes, or collection reaches the 200-attempt ceiling and the shortfall is published as a failed T3 gate;
- an SFT LoRA is selected using validation data only;
- the selected student has high enough action validity to interact, earns nonzero verifier reward, and produces enough same-scene reward variation to make GRPO identifiable;
- the dataset, checkpoint, configs, logs, cost, and evaluation report can be regenerated from checked-in commands.

### T4 is closed when

- stored behavior log probabilities reproduce under the training backend within a declared tolerance;
- a GRPO update changes only LoRA weights and can resume exactly from a checkpoint;
- the 100-update feasibility run completes or triggers a predeclared stop condition;
- promising feasibility is extended to 500 updates for three seeds; otherwise the negative result is closed at the failed gate;
- base, SFT, and SFT-to-GRPO checkpoints are compared on the same 32 frozen validation scenes;
- every seed, failure, restart, exclusion, and infrastructure error appears in the final report;
- a replayable successful episode is exported if any trained checkpoint succeeds.

## 5. Run contract and artifact layout

Every command accepts one checked-in YAML or TOML configuration and creates a new immutable run directory. A run must refuse to append to an existing directory.

```text
runtime/agentic-rl/<experiment-id>/<run-id>/
  config.resolved.yaml
  manifest.json
  events.jsonl
  episodes.jsonl
  metrics.jsonl
  evaluation.csv
  checkpoints/
  replays/
  stdout.log
  report.md
```

The manifest records:

- git commit and dirty-tree state;
- host, WSL distribution, GPU, CUDA, driver, Python, Torch, Transformers, TRL, Unsloth, bitsandbytes, and Factorio versions;
- base model repository and immutable revision, tokenizer revision, tokenizer hash, chat-template hash, quantization config, and LoRA config;
- task, task version, verifier version, scene split revision, observation profile, action catalog digest, parser version, prompt version, and exact static-prefix digest;
- sampler parameters, maximum turns, action budget, construction ticks, verification ticks, group size, and policy revision;
- bridge version, mod source digest, protocol version, game speed, worker count, and reset policy;
- teacher provider, requested model, returned model, thinking setting, cache usage, token usage, and cost when applicable;
- parent dataset or checkpoint digests;
- termination status and infrastructure-failure count.

`episodes.jsonl` schema v2 stores, for every model decision:

- exact role/content messages before rendering;
- exact rendered model input when the backend exposes one;
- prompt token IDs and attention mask;
- generated token IDs, decoded completion, sampled-token mask, and behavior log probability per generated token;
- sampling parameters and policy revision;
- parsed native action, parser result, domain-validation result, and executed native tool call;
- public observation and tool receipt before and after the call;
- step reward, terminal verifier result, terminal episodic reward, and termination reason;
- hashes for large repeated fields where the corresponding immutable content is stored once in the run.

Secrets, authorization headers, private evaluator state, reference-controller actions, and hidden provider reasoning are never written.

## 6. Phase T2-C: close the learner interface before T3

### T2-C.1 Extract one canonical `PolicyInterface`

**Build**

- Add a small reusable policy interface under `src/factoriorl/agent/` that constructs the stable instructions, task/objective block, static knowledge, action vocabulary, argument requirements, and dynamic observation summary already used by `AgentLoop`.
- Return typed data such as `PolicySessionPrefix`, `PolicyTurn`, `ParsedAction`, and `ValidationReceipt`; keep provider-specific chat rendering outside this component.
- Make `AgentLoop` call the extracted component so there is one production path, rather than copying its logic into training.
- Let the training bridge request the canonical public policy turn. The bridge should expose public state and prompt inputs, never evaluator internals.
- Use native catalog keys and native handles throughout. Delete `policy_view`, `HANDLE_OPTIONS`, and the custom prompt after equivalent tests pass.

**Acceptance gate**

- For a fixed saved observation, open-world and training callers produce byte-identical canonical message content and identical legal-action/domain data.
- Every rendered handle can be resolved by the shared domain validator in the same episode.
- The prompt contains no verifier output, reference action, hidden geometry, or reward.
- Existing agent-loop prompt, caching, and persistent-agency tests still pass.

**Verification**

```powershell
uv run pytest tests/unit/test_agent_loop.py tests/unit/test_transcript.py tests/unit/test_persistent_agency.py tests/unit/test_useful_observations.py
```

### T2-C.2 Freeze the action and error protocol

**Build**

- Define one model response contract using native action names and named arguments. Prefer the existing agent parser's accepted structured form over the temporary integer index format.
- Separate syntax parse failure, unknown action, missing/extra argument, out-of-domain value, stale handle, environment rejection, bridge failure, and verifier failure.
- Give model-caused failures a bounded public receipt and allow recovery while turn budget remains. Mark network loss, reset failure, process death, and malformed bridge responses as infrastructure failures.
- Record all generated attempts. If a turn permits a format-repair response, its tokens are learner output and remain in the trajectory.
- Add optional JSON grammar decoding behind a config flag. Its off/on setting is part of experiment identity.

**Acceptance gate**

- Fixture tests distinguish every failure class.
- Invalid native handles are not repaired or translated into valid handles.
- Infrastructure failures are excluded from reward statistics and retried under a logged policy; task failures remain zero-reward samples.
- One end-to-end test shows a bad call, public error receipt, corrected call, and successful execution without hidden assistance.

### T2-C.3 Upgrade rollout artifacts to schema v2

**Build**

- Add the fields in Section 5 and a reader that validates schema version, required digests, finite probabilities, complete masks, turn ordering, and terminal verification.
- Store terminal verification separately. Do not overwrite the last action receipt when `finish` is called.
- Add a group record containing group ID, scene digest, policy revision, member episode digests, and all four rewards.
- Preserve the v1 reader for historical inspection, but reject v1 artifacts for training.
- Add a command that audits a run without loading the model.

**Acceptance gate**

- Tampering with a message, token, receipt, reward, or group member fails its digest/audit.
- A training loader cannot admit a partial group, mixed policy revisions, mixed scene digests, or an unverified episode.
- Exact prompt reconstruction does not depend on the current source tree.

### T2-C.4 Prove token masks and log-probability parity

**Build**

- Implement one shared tokenization path for collection, SFT, evaluation, and GRPO.
- Build assistant-token masks from explicit message boundaries. Assert that observation, tool, system, padding, and teacher-only tokens have zero policy-loss weight.
- Compare generation-time transition scores with a fresh forward-pass recomputation for every sampled suffix in a stored group.
- Declare absolute and relative tolerances from a measured same-device run. Do not choose a tolerance large enough merely to pass.
- Add a one-token perturbation test that must fail parity.

**Acceptance gate**

- All four episodes in a fresh live group pass token alignment and recomputation.
- A model reload reproduces the stored log probabilities within the measured tolerance.
- No non-assistant token contributes to SFT or GRPO loss.

### T2-C.5 Prove reset isolation and replace the temporary launcher

**Build**

- Add a live group probe that checks stable scene digest plus fresh episode identity, inventories, placed entities, request IDs, pending actions, verifier counters, and time.
- Mutate attempt one deliberately and prove attempts two through four begin clean.
- Replace untracked `_launch_gemma_group.ps1` with checked-in CLI commands and documented Windows/WSL launch steps. The command must accept the bridge URL and token through environment variables without printing the token.
- Record generation tokens/second, decisions/minute, episode wall time, simulated ticks/second, peak VRAM, and reset time.

**Acceptance gate T2**

- T2-C.1 through T2-C.5 pass on the PC against a real Factorio instance.
- One audited four-member group is checked into `docs/evidence/` as a compact report; bulky artifacts remain under `runtime/` with hashes in the report.
- `git status` contains no temporary launcher or accidental runtime artifact.

## 7. Phase T3: baseline, demonstrations, and SFT

### T3.0 Freeze the experiment contract

**Build**

- Freeze 32 validation scenes and the training-scene generator before any SFT selection. Verify disjoint scene identities and keep the structural holdout inaccessible to training commands.
- Create one evaluation config shared by base, SFT, and GRPO checkpoints. Freeze temperature, top-p, maximum new tokens, turn budget, action budget, game speed, construction window, verification window, parser behavior, and syntax-constraint setting.
- Pin `construct_smelting_line` task/reward version and the canonical policy-interface version.
- Add config validation that rejects unknown keys and prints the resolved configuration without secrets.

**Acceptance gate**

- Re-running split generation produces identical digests.
- Training commands reject validation and structural-holdout scene IDs.
- Evaluation commands cannot silently override the frozen task, prompt, or sampling settings.

### T3.1 Measure the unmodified Gemma baseline

**Build**

- Evaluate the base 4-bit Gemma model once on all 32 frozen validation scenes using the full task budget.
- Run a short throughput calibration first to select only operational settings such as attention implementation. Do not tune behavior or prompts on validation outcomes.
- Report parse validity, domain validity, bridge acceptance, useful state-changing actions, machine-produced plates, episodic reward, binary success, turns, generated tokens, wall time, simulated time, peak VRAM, and termination classes.
- Publish every scene result and aggregate confidence intervals. Keep infrastructure failures separate and rerun only those scenes with the same model/config and a logged replacement link.

**Acceptance gate**

- Exactly one valid result exists for every validation scene.
- All episode artifacts pass the schema-v2 audit and log-probability check.
- The report labels this as a base-model measurement even if it scores zero.

### T3.2 Preflight the DeepSeek teacher

**Build**

- Use `https://api.deepseek.com/v1`, requested model `deepseek-flash`, and key variable `DEEPSEEK_API_KEY`. Confirm the returned model identifier and current provider protocol at run time; do not substitute a model after an error.
- Reuse the existing OpenAI-compatible adapter, thinking-mode continuation handling, usage accounting, and cache telemetry.
- Send the same canonical public information and native tool contract used by Gemma. Provider chat syntax may differ, but the policy-visible facts may not.
- Run one reset, one request, one parsed native action, one tool execution, and one continued request. Confirm that private reasoning is neither logged nor echoed back incorrectly.
- Set explicit wall-time, output-token, request, and dollar ceilings. Abort before collection if price accounting or cache accounting is unknown.

**Acceptance gate**

- The preflight records requested and returned model IDs, action/continuation success, token usage, cache hit/miss tokens, estimated/charged cost when available, and no secret value.
- The exact invariant prefix stays byte-stable across turns and provider cache hits appear after the initial miss.
- The teacher executes a native action through the same validator used by the learner.

### T3.3 Collect teacher attempts in bounded waves

**Build**

- Collect training-only attempts in waves of 20, 30, 50, and 100, stopping as soon as 20 verified successes have been retained or 200 total attempts have run.
- After the first 20, continue only if there is at least one verified success or the failures show measurable partial reward and a correctable interface/task cause. Otherwise pause collection and diagnose before spending more.
- Keep all attempts, including zero reward, malformed calls, recovery sequences, and infrastructure exclusions. Never present failed traces as SFT successes.
- Permit the teacher the same observation, action, and episode budgets as evaluation. Do not give it verifier state, scripted geometry, reference actions, or a stronger tool catalog.
- Preserve legitimate recovery actions in successful trajectories. Remove only provider-hidden reasoning and transport metadata that is outside the learner contract.
- Deduplicate exact trajectories, audit scene separation, and write a dataset card with success rate, reward distribution, failure taxonomy, tokens, cache rate, cost, prompt/task digests, and provenance.

**Acceptance gate**

- Preferred gate: at least 20 verified successful training trajectories with native actions and complete schema-v2 evidence.
- Hard ceiling: 200 attempts. If fewer than 20 succeed, close collection as a failed gate and diagnose teacher/task/interface behavior instead of manufacturing or silently adding scripted labels.
- No validation or structural-holdout scene occurs in the dataset.

### T3.4 Build the SFT dataset

**Build**

- Use verified successful teacher trajectories, including their genuine recovery turns. Produce one example per learner decision with the bounded preceding context exactly as it will be constructed at inference time.
- Apply assistant-token-only loss to teacher action text. Tool results and observations are context; they have zero target weight.
- Weight trajectories equally initially so long recoveries do not dominate merely because they contain more turns. Record both trajectory and token counts.
- Create deterministic train and SFT-development partitions by scene, not by turn. The frozen 32-scene RL validation set remains separate.
- Store raw, normalized, and tokenized dataset digests. A transformation change increments the dataset version.

**Acceptance gate**

- A mask audit samples first, middle, recovery, and final turns and shows targets only on assistant action tokens.
- Decoding target IDs reconstructs the teacher action exactly.
- Every target action was accepted in the source episode; every source episode has positive verifier reward.

### T3.5 Train and select the SFT LoRA

**Build**

- Load the verified 4-bit Gemma base and attach a versioned LoRA. Start with one epoch; allow at most three epochs.
- Use microbatch size one with gradient accumulation. Enable gradient checkpointing if required. Log optimizer steps, effective tokens/batch, learning rate, loss, gradient norm, throughput, peak VRAM, and checkpoint digest.
- Save resumable optimizer/RNG state and an inference-only adapter at every epoch.
- Evaluate each epoch on the fixed 32-scene validation protocol. Select by verified episodic reward and success; use tool validity as a safety diagnostic, not a proxy reward.
- Compare base and each SFT epoch scene by scene. Do not select by training loss.

**Acceptance gate T3-A**

- Training completes without updating frozen base weights.
- Resume from an epoch checkpoint produces the same next-batch IDs, loss within numerical tolerance, and optimizer step.
- The selected adapter digest and selection rule are recorded before GRPO starts.

### T3.6 Establish that GRPO has a learnable signal

**Build**

- Evaluate the selected SFT checkpoint on all 32 validation scenes.
- Collect ten no-update four-attempt groups on training scenes with frozen SFT weights.
- Measure within-group reward variance, action validity, reward density, episode lengths, and dominant failure classes.

**Acceptance gate T3**

All of the following are required to start T4:

- at least 90% of completions parse under the frozen response contract;
- at least 85% of parsed calls pass domain validation and reach the bridge;
- at least four of the 32 validation scenes obtain nonzero verifier reward;
- at least three of the ten dry-run groups contain more than one reward value;
- no artifact, split, reset-isolation, or log-probability audit failure remains open.

These are engineering promotion thresholds, not claims of statistical significance. If they fail, stay in T3 and classify the cause:

- bad syntax: improve SFT data or enable a frozen syntax grammar;
- bad handles/actions: fix canonical grounding or collect more valid demonstrations;
- valid actions but no production: add successful demonstrations or introduce an easier training-only curriculum stage;
- successful episodes but uniform groups: adjust scene/task sampling or group size only in a new experiment;
- base/SFT model too weak: compare one similarly sized learner under the same interface before increasing model size.

## 8. Phase T4: first GRPO experiment

### T4.0 Implement and verify the update path offline

**Build**

- Initialize from the selected SFT LoRA and keep base weights frozen.
- Implement group-relative advantages over terminal episodic rewards. For group rewards `r`, use `(r - mean(r)) / (std(r) + epsilon)`; define the standard-deviation convention and epsilon in config and tests.
- Apply each episode advantage to all of that episode's sampled assistant tokens. Average active-token loss within each episode, then average the four episode losses, so a long recovery trajectory does not outweigh another group member merely because it generated more tokens.
- Use the stored behavior-policy log probabilities in the clipped objective. Initial settings are group size four, learning rate `5e-6`, one optimization pass per group, symmetric clip `0.2`, no entropy bonus, and no explicit KL penalty.
- Accumulate all four group members before an optimizer step. Never split a group across policy revisions.
- Invalidate learner KV caches after every update. Any generation cache is scoped to one policy revision.

**Acceptance gate**

- Hand-calculated toy groups cover varied, identical, and partially failed rewards.
- Identical-reward groups produce zero advantages and no parameter update.
- A positive-advantage action's likelihood increases in a controlled one-step test; a negative-advantage action's likelihood decreases.
- Only LoRA tensors change. Frozen base tensors remain byte-identical.
- Saving and resuming reproduces the next group selection, sampled RNG state, and update within the declared tolerance.

### T4.1 Run a five-update live systems smoke test

**Build**

- Execute five complete sequential groups against live Factorio.
- At each update, load one frozen policy revision, collect four complete attempts, audit the group, update once, invalidate caches, and save a lightweight recovery checkpoint.
- Track collection time, update time, GPU utilization, VRAM, reset failures, bridge latency, tokens/second, rewards, advantages, ratio statistics, clip fraction, loss, and gradient norm.

**Acceptance gate**

- Twenty episodes complete with no mixed revision, scene, or prompt identity.
- An interrupted run resumes without duplicating or dropping a group.
- No NaN/Inf, unbounded ratio, memory leak, stale cache, or unclassified failure occurs.
- The measured wall-time projection for 100 updates is written before the feasibility run begins.

### T4.2 Run the 100-update feasibility block

**Build**

- Start a new immutable run from the selected SFT checkpoint using the frozen configuration.
- Validate at update 0, 25, 50, 75, and 100 on all 32 validation scenes. Checkpoint at the same points.
- Do not edit prompt, reward, task, parser, split, catalog, sampler, or optimizer settings during the run.
- Apply the predeclared stop conditions below automatically and emit a final diagnostic report when one fires.

**Stop conditions**

- nonfinite loss or gradients: stop immediately;
- three consecutive worker/reset infrastructure failures: stop;
- twenty consecutive groups with identical within-group rewards: pause and diagnose the signal;
- tool/domain validity more than ten percentage points below the SFT baseline at two consecutive validations: stop;
- no improvement over the best prior validation mean reward across four validation checks: retain the best checkpoint and close feasibility;
- run manifest, artifact audit, split isolation, or policy-revision mismatch: stop immediately and invalidate affected samples.

**Acceptance gate T4-A**

The run is promising only if:

- it reaches update 100 without an integrity failure;
- its selected checkpoint improves paired mean validation reward over SFT's point estimate;
- validation success does not decrease;
- tool/domain validity stays within ten percentage points of SFT;
- improvement is not explained solely by one scene or an evaluator anomaly.

If this gate fails, T4 closes as a negative feasibility result. Do not automatically change optimizer family or run 1,500 more updates.

### T4.3 Extend promising GRPO to three seeds

**Build**

- Run 500 updates for three declared training seeds, including the feasibility seed if its configuration is identical.
- Preserve identical validation scenes and generation settings. Choose each seed's checkpoint by its own validation results, never by structural holdout.
- Report all seeds separately and aggregate only after showing their variation.
- Export checkpoint, resolved config, training curve, evaluation table, and at least one replay for every seed that succeeds.

**Acceptance gate T4**

- Three seed runs are complete or have triggered documented stop conditions.
- The report contains base, SFT, update-100, and selected update-500 comparisons.
- Paired per-scene deltas, confidence intervals, tool-validity curves, reward curves, failure taxonomy, throughput, and compute time are published.
- The conclusion uses the evidence actually obtained: improvement, instability, no effect, or capability loss.

### T4.4 One controlled fallback, only when the result points to it

Do not implement multiple algorithms in parallel during the first GRPO run.

- If imitation works but GRPO destroys validity through large token-level ratios, compare one controlled stabilization method such as GSPO or REINFORCE++ with the same data and evaluation contract.
- If rewards are uniformly zero, return to T3 competence/curriculum work; another optimizer cannot recover a missing learning signal.
- If rollout wall time dominates and policies spend long periods waiting for sequential groups, benchmark asynchronous collection and policy staleness before considering SAO.
- If the Gemma learner remains incapable after strong SFT data, compare a similarly sized Qwen-family model through the same `PolicyInterface` before adding environment-specific hints.

Any fallback is a new experiment and is outside the T4 closure gate above.

## 9. Implementation work packages and suggested commits

Each work package should land as a reviewable commit with tests and evidence. Do not combine unrelated refactors with live experimental results.

| Order | Suggested commit | Depends on | Evidence required |
|---:|---|---|---|
| 1 | `Extract canonical policy interface` | current head | fixed-observation equivalence fixture |
| 2 | `Use native policy contract in learner rollouts` | 1 | parser/domain failure matrix |
| 3 | `Version replayable rollout artifacts` | 1-2 | schema-v2 audit and tamper tests |
| 4 | `Verify learner masks and behavior log probabilities` | 3 | fresh/reloaded parity report |
| 5 | `Close live T2 reset and rollout gate` | 1-4 | real four-episode group report |
| 6 | `Freeze agentic RL splits and evaluation config` | 5 | split/catalog/prompt digests |
| 7 | `Evaluate base Gemma validation baseline` | 6 | 32-scene report |
| 8 | `Preflight and collect DeepSeek demonstrations` | 6 | provider preflight and dataset card |
| 9 | `Build masked SFT dataset` | 8 | dataset/mask audit |
| 10 | `Train and select Gemma SFT adapter` | 9 | epoch comparisons and resume proof |
| 11 | `Measure SFT group reward variation` | 10 | ten-group dry-run report |
| 12 | `Implement verified GRPO update` | 4, 10 | offline math/update tests |
| 13 | `Run live GRPO systems smoke` | 11-12 | five-update report and time projection |
| 14 | `Run 100-update GRPO feasibility` | 13 | validation curves and gate decision |
| 15 | `Complete three-seed GRPO evaluation` | 14 pass | all-seed final report |

Work packages 7 and 8 may run independently after package 6, but they must not share a mutable Factorio worker. Package 12 may be developed against synthetic and frozen artifacts while teacher collection runs. Live learner evaluation/training on the 4060 should have exclusive GPU access.

## 10. Required commands

Expose supported commands through the project CLI rather than ad hoc scripts:

```text
factoriorl learner doctor
factoriorl learner collect --config <path>
factoriorl learner audit --run <path>
factoriorl learner evaluate --config <path> --checkpoint <base-or-adapter>
factoriorl learner collect-teacher --config <path>
factoriorl learner build-sft --config <path>
factoriorl learner train-sft --config <path>
factoriorl learner train-grpo --config <path>
factoriorl learner replay --episode <id>
factoriorl learner report --runs <paths...>
```

`learner doctor` is the only prerequisite command. It verifies imports, exact dependency versions, CUDA availability, model load, one forward/backward LoRA update, bridge reachability, reset, action, verification, disk space, and run-directory writability. It must fail fast with a specific remediation and must not make a paid teacher call.

## 11. Test and evidence policy

Run narrow tests after each work package and the full unit suite before every live milestone. Live tests use dedicated ports and unique run directories. Never delete or rewrite an earlier run to make a gate pass.

Minimum regression coverage:

- canonical prompt equivalence and evaluator-information exclusion;
- native handle freshness and stale-handle rejection;
- async action completion and lost-reply classification;
- first-item collection, machine status, placement footprints, partial batch failure, and saved-state isolation;
- reward provenance and no hand-production credit;
- same-scene group identity and independent reset state;
- assistant-only SFT/GRPO masks;
- behavior log-probability parity and perturbation failure;
- complete-group policy revisions;
- LoRA-only updates and exact checkpoint resume;
- cache invalidation after weight updates;
- structural-holdout refusal;
- declared CLI arguments are read by their commands.

For each gate, write a compact Markdown report under `docs/evidence/agentic-rl/` containing the command, resolved config digest, relevant artifact hashes, result, failures, and next decision. Large models, JSONL trajectories, and replays stay out of Git.

## 12. Immediate next action

Begin with T2-C.1. Extract the canonical policy interface and replace the temporary Gemma prompt path before collecting another model rollout. The first live run after that change is evidence for T2 closure, not another prompt experiment. Once the full T2 gate passes, freeze the experiment contract and run the 32-scene base baseline and DeepSeek preflight.
