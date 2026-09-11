# FactorioRL: train a small language-model agent with verifiable game rewards

Date: 2026-09-11. Status: T0 model-update fit and T1 task gate passed; T2's restricted bridge is implemented but has not yet had a cross-host rollout probe. See [the T0 handoff](HANDOFF-AGENTIC-T0-2026-09-11.md), [the T1 handoff](HANDOFF-AGENTIC-T1-2026-09-11.md), and [the T2 bridge handoff](HANDOFF-AGENTIC-T2-BRIDGE-2026-09-11.md). This remains the implementation plan for subsequent phases, not evidence that the agent has learned a game capability.

## 1. Direction and fixed decisions

**Purpose:** demonstrate that a pretrained language-model agent improves at factory construction through experience in the actual Factorio engine.

The first result is a reusable construction capability across unfamiliar layouts. Full-game play remains the integration goal; it is not the first training objective. This document governs the next agentic-RL implementation sequence. Preserve historical requirements, frozen tasks, and results in the master plan and earlier handoffs.

| Area | Decision |
|---|---|
| Primary learner | `google/gemma-4-E2B-it` |
| Fallback | `Qwen/Qwen3-1.7B`, only if the bounded Gemma compatibility/memory probe fails |
| Teacher | User-provided Qwen API; exact endpoint, model ID, and credential variable deferred by the user |
| Hardware | Existing RTX 4060 PC; no hardware rediscovery |
| Compute | No fixed GPU-hour cap; stop on failed validity gates, repeated infrastructure errors, or uninformative learning |
| Initial training | Verified-demonstration SFT, followed by synchronous GRPO |
| Parameter updates | LoRA adapters; text-only observations; freeze multimodal towers |
| Simulation | Real Factorio with controlled advancement during training |
| Demonstrations | Preserve continuous-time open-world play as a separate profile |
| Framework | TRL and a supported Gemma training backend; no custom optimizer implementation |
| Deferred | SAO, broad algorithm comparisons, vision training, delegation, and full-game RL |

**Teacher configuration is explicitly deferred.** Model-fit checks, verifier development, rollout integration, and local baseline evaluation proceed independently. Live teacher collection requires the API base URL, exact model ID, credential environment-variable name, and a successful compatibility check. Do not request or print the key itself, infer a model identifier from the conversational name, or substitute a paid provider. Qwen API access supplies inference, not trainable teacher weights. Unlimited access does not imply unlimited concurrency: begin with one request at a time and honor rate limits.

## 2. Phases and acceptance gates

### T0 — Isolate the training stack and verify model fit

**T0.1: Dependencies and process boundaries**

Create a separate language-model training environment and lockfile. Do not upgrade the existing SB3 environment in place.

Use Linux/WSL2 for the trainer and retain the established Windows Factorio worker runtime. Put a small run-scoped bridge between them for reset, observation, action execution, and reward retrieval. Restrict bridge operations to registered game tools; do not expose arbitrary Lua or shell execution.

**T0.2: Gemma probe**

Use the documented Unsloth Gemma E2B training route. The current Gemma 4 notebook uses
`fast_inference=False`, and current Unsloth vLLM VLM support excludes Gemma 4; therefore the
Gemma update-cycle probe runs without in-process vLLM. Keep vLLM installed and test it as a
separate rollout-serving compatibility gate after a successful adapter update. The probe must
retain enough VRAM for recomputing token log probabilities and one optimizer step. Start with:

- Text-only inputs and model-native chat formatting.
- LoRA rank 8 on supported language attention/MLP modules.
- Gradient checkpointing and microbatch size one.
- Maximum 2,048 input tokens and 256 generated tokens for the initial probe.
- One model process for the Gemma update check. Do not run a second standalone vLLM server
  alongside a trainable copy of the model on this GPU. For a later compatible rollout backend,
  vLLM must run only while optimizer activations are absent.

Use supported quantized adapter loading where the selected backend supports it. Record the exact model revision, quantization, trainable modules, and package versions.

The probe must exercise generation, recomputation of generated-token log probabilities, backward propagation, an optimizer step, adapter saving, reloading, and another generation. A successful model load or SFT step alone does not establish GRPO fit.

Attempt Gemma's supported 4-bit adapter configuration, then one lower-memory configuration with
a 1,024-token input and its language decoder only (`text_only=True`), omitting the unused vision
tower. The lower-memory attempt uses the explicit Unsloth 4-bit checkpoint and an all-GPU map;
this avoids treating a known automatic CPU/disk device-map failure as an out-of-memory result.
Run a separate vLLM adapter-serving compatibility test after a successful update; a
Gemma-specific vLLM incompatibility is a throughput gate, not evidence that the model cannot
train. If either update attempt fails, select Qwen3-1.7B and repeat. Do not spend days patching
kernels or relying on heavy CPU offload before a first result.

**Gate T0:** a complete update cycle fits with measured memory headroom; finite gradients change adapter weights; saving and resuming work. Publish the selected model and configuration. If neither model passes, stop with the measured blocker.

### T1 — Establish a scored construction task

Add a new versioned task, `construct_smelting_line`, without altering existing frozen tasks.

**Starting state:** a peaceful bounded site contains iron ore and no production machines. The character receives two burner drills, two furnaces, and sufficient coal. No iron plates, preloaded machines, or prebuilt connections are supplied.

**Goal:** construct and fuel a line that produces at least ten new iron plates during a fixed 3,600-tick verification window. During that window, the agent cannot manually mine, craft, transfer, or build. This checks machine production after construction rather than a self-reported achievement.

**Interaction limits:**

- At most 32 model turns.
- At most eight explicit actions per batch.
- At most 256 tool actions.
- At most 18,000 construction ticks before verification.
- A bounded `finish` operation starts verification early; otherwise verification follows the construction limit.

The initial task tests construction and operation. It does not claim automated fuel replenishment or end-to-end resource acquisition.

**Task splits:** declare three geometry families: open patches, offset patches, and constrained sites with obstacles. Use open and offset layouts for training, distinct instances for validation, and constrained layouts for the structural holdout.

Freeze 32 validation scenes and 100 structural test scenes before training. Keep their identities disjoint from demonstrations and training. Check feasibility with an evaluator-only reference controller; its geometry and actions never enter learner observations.

**Reward:** use normalized newly produced target output in the verification window:

`reward = min(new_machine_iron_plates / 10, 1)`

Report binary success separately. Do not reward verbosity, valid JSON, claimed plans, placements alone, or hand-produced inventory.

**Gate T1:** reference solves, no-action failures, reward accounting, reset isolation, and exploit probes pass. Moving existing items, unrelated production, repeated placement, or stopping a machine before verification cannot manufacture success.

### T2 — Integrate multi-turn rollouts correctly

Wrap the environment through TRL's stateful environment interface or its documented rollout extension.

**Group construction:** a group contains four independent attempts from an identical initial scene. Freeze learner weights while collecting the group. Initially execute attempts sequentially, resetting the worker completely between them.

**Training records:** record scene identity, observations, tool calls, tool results, generated token IDs, behavior-policy log probabilities, token masks, rewards, policy version, and termination reason.

Only learner-generated action text contributes to policy loss. Tool responses and game observations are conditioning inputs. Do not treat teacher-generated responses as on-policy learner samples.

Begin with non-thinking generation and concise structured actions. Keep provider-specific teacher formatting separate from the learner's training contract.

**Context and caching:** keep objective and tool schemas in a stable prefix. Append concise observations and receipts within a rollout. Use deterministic bounded history management with current state, active plan, and recent outcomes; apply the same transformation during collection and training.

Invalidate model-dependent KV caches after every weight update. Preserve teacher prompt caching within its sessions. Do not reuse cached learner states across policy versions.

**Gate T2:** grouped scene hashes match; independent worlds do not leak state; token masks exclude observations; log-probability recomputation agrees within the selected backend's documented numerical tolerance. Infrastructure failures are recorded separately and never silently converted into task failures.

### T3 — Establish baselines and collect demonstrations

**T3.1: Baseline**

Evaluate the selected unmodified learner on the 32 validation scenes, recording tool validity, production, success, and costs. Do not tune on the structural holdout.

**T3.2: Teacher collection**

Require the deferred teacher configuration and one compatibility check before collection. Never print credential values.

Collect up to 200 attempts on training scenes. Retain every attempt and its verification result. Select successful trajectories for SFT, retaining legitimate recovery actions rather than only ideal executions.

If fewer than 20 attempts succeed, inspect the first failure patterns and stop collection for diagnosis. Do not automatically generate thousands of unsuccessful traces.

Teacher and learner use the same game information and tools. A separate scripted demonstration source may be added only as an explicitly labeled dataset revision.

**T3.3: SFT**

Train the LoRA adapter with assistant-token loss only. Run one epoch, validate, and allow up to three epochs; select by validation success, with verifier output and tool validity as supporting metrics.

**Gate T3:** the student can produce valid actions and obtain nonzero verification rewards. If it cannot, diagnose observations, task difficulty, or imitation before starting GRPO.

### T4 — First GRPO experiment

Use the SFT checkpoint as initialization:

- Group size four.
- Learning rate `5e-6`.
- One optimization pass per collected group.
- Standard GRPO objective and group-relative advantages.
- Symmetric clipping at 0.2.
- No added entropy bonus.
- No explicit KL penalty for the initial implementation.
- Same sampling distribution for rollout and stored behavior probabilities.
- Gradient accumulation to process complete groups with microbatch size one.

Validate every 25 optimizer updates and checkpoint at each validation point.

Run an initial 100-update feasibility block. If valid and promising, extend to 500 updates per training seed, using three seeds. These are experiment boundaries, not GPU-hour limits.

**Stop conditions:**

- Nonfinite loss or gradients: stop immediately.
- Three consecutive worker/reset failures: stop the run.
- Twenty consecutive groups with identical rewards within each group: pause for diagnosis.
- Tool-validity deterioration exceeding ten percentage points from the SFT baseline at two validations: stop and investigate.
- No validation improvement across four checks: retain the best checkpoint and close the experiment rather than automatically extend it.

Do not change reward, prompts, or task versions mid-experiment. A change creates a new experiment.

**Gate T4:** report the original model, SFT model, and SFT-to-GRPO model, including unsuccessful training seeds. Checkpoint curves must show whether competence is retained or lost.

### T5 — Held-out evaluation and a visible demonstration

Select checkpoints using validation only. Evaluate once on the frozen structural set with fixed generation settings, reporting each training seed separately.

Primary measures:

- Verified construction success.
- New machine output.
- Tool failures and recovery.
- Model turns, generated tokens, simulated time, and wall time.
- Demonstration collection and training compute.

Use paired scene comparisons and confidence intervals. Report variability across training seeds rather than pooling it away.

Run the best validated checkpoint in a graphical demonstration and export the adapter, configuration, evaluation recipe, and replay.

**Scientific outcome gate:** claim improvement only if it survives held-out evaluation and is not driven by one selected seed. A reproducible negative result completes the feasibility cycle; it does not justify presenting RL as effective.

## 3. Implementation boundaries and public interfaces

Add a dedicated learner command group for model/trainer preflight, teacher trajectory collection, SFT training, GRPO training, and checkpoint evaluation/replay export.

Every command accepts a configuration file and writes into a unique run directory. Configurations identify task version, scene split, model revision, adapter settings, observation profile, clock policy, and verifier version.

Use a worker pool interface, but initially allow only one active environment and one learner generation job. Increase concurrency only after measurement shows benefit.

All model-only dependencies remain optional. Existing benchmark, replay, and agent commands continue working without the language-model training stack.

Required regression scenarios include first-item collection, truthful machine status, stale targets, asynchronous completion, partial batch failure, saved-state isolation, reward provenance, correct token masks, and checkpoint resume.

Before implementation, reconcile the existing prompt/interface review and newer fixes against the current checkout. Do not reimplement completed fixes. Preserve unrelated running jobs, local changes, historical results, and credentials.

## 4. Decisions after the first result

Do not implement every RL method now.

- **If GRPO learns:** extend to material transport and connected production chains, then recovery and more complex products.
- **If imitation works but GRPO degrades it:** study update stability; compare one controlled alternative such as GSPO or REINFORCE++.
- **If groups are uniformly unsuccessful:** improve starting competence or task progression; changing the optimizer alone is not the default response.
- **If long trajectories and waiting workers dominate cost:** evaluate asynchronous training, with SAO as a candidate and explicit policy-staleness accounting.
- **If the local model fits but lacks competence:** compare Gemma and Qwen under the same interface before expanding model size or renting hardware.
- **If a learned capability works reliably:** invoke it from the open-world agent and measure whether it improves autonomous progress and reduces teacher calls.

The deliverable is a credible first example of a language-model agent learning useful behavior in real Factorio. Full-game progression remains the destination, with each added capability justified by measured learning.

## 5. References and feasibility limits

- [Gemma E2B model card](https://huggingface.co/google/gemma-4-E2B-it): E2B is an effective parameter designation; memory fit must account for the complete model.
- [Unsloth Gemma training guide](https://unsloth.ai/docs/models/gemma-4/train): published fine-tuning and RL recipes have different memory requirements. An 8GB fine-tuning example is not proof that this multi-turn GRPO workload fits.
- [TRL GRPO trainer](https://huggingface.co/docs/trl/grpo_trainer): stateful environments and multi-turn rollout integration. Pin a verified dependency combination rather than relying on moving defaults.
- [GRPO / DeepSeekMath](https://arxiv.org/abs/2402.03300), [GSPO](https://arxiv.org/abs/2507.18071), [REINFORCE++](https://arxiv.org/abs/2501.03262), [SAO](https://arxiv.org/abs/2607.07508).

Sources were consulted during planning on 2026-09-11. T0 has since completed a measured 4-bit update-cycle probe. No teacher call, Factorio rollout, SFT run, or RL run has been performed.
