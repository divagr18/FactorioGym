# Agentic RL T0 handoff

Date: 2026-09-11. T0 is complete for the Gemma training-update path. This note
supersedes the plan's former assumption that model fit was untested.

## Decision

Use `unsloth/gemma-4-E2B-it-unsloth-bnb-4bit` for the first local
language-model update experiments. Keep the run text-only, freeze unused
multimodal components, use LoRA rank 8, and begin with a 1,024-token context.
The accepted test configuration explicitly maps the model to the single GPU.

The measured T0 cycle generated text, recomputed generated-token log
probabilities, backpropagated, updated adapter weights, saved the adapter,
reloaded it, and generated again. Its post-update memory was 7.46 GB allocated
and 7.76 GB reserved on the RTX 4060's 8.59 GB reported capacity. Those figures
are whole-process training runtime memory, not the size of the four-bit weights.

Evidence: [update-cycle report](evidence/agentic-t0-gemma-e2b-2026-09-11.json).

## Environment actually used

The isolated WSL runtime resolved to:

- PyTorch 2.10.0+cu128
- Transformers 5.6.2
- TRL 0.29.1
- vLLM 0.19.1
- Unsloth 2026.9.4 and Unsloth Zoo 2026.9.3

The project runtime definitions and reproducible probe are in
[`training/`](../training/). They are intentionally separate from the existing
SB3 environment and do not start Factorio or call a provider.

## Important compatibility finding

The successful path needed a narrow text-only metadata repair: the checkpoint's
text configuration exposed `architectures: null`, while the installed Unsloth
generation wrapper assumes an iterable architecture list. `probe.py` records
when this repair is applied. Treat it as a dependency-sensitive compatibility
point: retain the probe as a regression gate whenever the model, Transformers,
or Unsloth changes.

The direct vLLM test of that exact BitsAndBytes checkpoint did **not** pass. It
loaded the checkpoint shards, then vLLM 0.19.1 stopped on a tensor-shape
assertion in its Gemma 4 BitsAndBytes weight loader. This was not an OOM: the
report recorded 0 GB CUDA allocation before engine startup, and the failure
occurred while matching checkpoint tensors to vLLM parameters. Do not use this
checkpoint through direct vLLM serving yet, and do not describe vLLM rollouts as
working for Gemma. Evidence: [vLLM compatibility report](evidence/agentic-t0-gemma-e2b-vllm-2026-09-11.json).

## Consequence for the next phase

T1 can proceed: implement the isolated, verifiable smelting construction task
and the Windows-worker/WSL learner bridge. T2 must use a compatible non-vLLM
rollout path at first, or pass a separate vLLM test with a checkpoint format
vLLM can load. Do not run a standalone serving engine beside the trainable copy
on this GPU.

No teacher configuration, demonstrations, SFT, GRPO, or Factorio rollouts have
been performed. Keep those claims out of project status until their separate
acceptance gates pass.
