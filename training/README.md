# Agentic training runtime

This directory is a deliberately separate Linux/WSL environment. It must not be
installed into the repository's SB3 environment or added to the distributable
`factoriorl` wheel.

From WSL, create the lock and environment:

```bash
cd /mnt/d/FactorioRL/training
python3 -m pip install --user --upgrade uv
./bootstrap.sh
```

Run the actual Gemma T0 update-cycle probe. Gemma 4 currently uses Unsloth's
supported non-vLLM training route; the runtime still includes vLLM for the
separate rollout-serving compatibility check:

```bash
uv run python probe.py --load-in-4bit --output ../runtime/agentic-t0/gemma-e2b
```

If that does not fit, run the required lower-memory attempt before choosing the
fallback model:

```bash
uv run python probe.py \
  --max-seq-length 1024 \
  --model unsloth/gemma-4-E2B-it-unsloth-bnb-4bit \
  --load-in-4bit \
  --text-only \
  --force-single-gpu \
  --output ../runtime/agentic-t0/gemma-e2b-lowmem
```

The probe writes a JSON report and an adapter. It passes only after generation,
generated-token log-probability recomputation, backward propagation, an optimizer
step, adapter save/load, and post-reload generation all succeed. It never starts
Factorio or calls a paid model provider.

`vllm` is Linux-only here by design. The actual Factorio worker remains on
Windows; T0 is model fit only. A later bridge is the boundary between the two.
Current Unsloth vLLM support excludes Gemma 4, so `--fast-inference` is a
separate compatibility check and is not used for the supported Gemma update
probe.

Run that direct vLLM compatibility check after the update probe:

```bash
uv run python vllm_probe.py --output ../runtime/agentic-t0/gemma-e2b-vllm
```

The Gemma notebook explicitly installs newer Transformers and TRL packages with
`--no-deps`, beyond Unsloth's published package metadata. `bootstrap.sh` makes
that same experimental compatibility override explicit. The probe records the
exact resolved versions; an import, generation, or update failure is a T0
failure, not something to paper over.

## Windows-worker / WSL rollout bridge

T2 uses a deliberately small authenticated HTTP bridge. Run it on the Windows
Factorio host, from this repository's normal environment:

```powershell
$env:FACTORIORL_BRIDGE_TOKEN = "<long-random-value>"
uv run python tools/agentic_bridge.py --token-env FACTORIORL_BRIDGE_TOKEN
```

The WSL learner reads the same token from its environment and uses
`training/bridge_client.py`. The public operations are `reset`, `observe`,
`act`, and `finish`. `act` accepts only a current catalog index and exactly its
declared arguments. `finish` is one-shot and invokes T1's action-locked
verification window. The bridge never returns evaluator truth, RCON details,
worker paths, or a generic command endpoint.

The default listener is loopback (`127.0.0.1:8765`). If WSL localhost forwarding
is unavailable, bind with `--host 0.0.0.0` and allow the port only from the
current WSL subnet in Windows Firewall. Remove that firewall exception after the
run. Do not expose the bridge to the LAN.

Run one audited, canonical-prompt group from WSL:

```bash
FACTORIORL_BRIDGE_URL=http://<windows-host>:8765 \
FACTORIORL_BRIDGE_TOKEN="<same-long-random-value>" \
uv run python run_gemma_group.py --output ../runtime/agentic-t2/gemma-group-<run-id>

uv run python audit_rollout.py --run ../runtime/agentic-t2/gemma-group-<run-id>
```

The group command uses the same policy renderer and native action parser as the
agent loop. It records the exact messages and rendered prompt, sampled token
IDs and log probabilities, parser result, native tool traffic, terminal
verification, and a same-scene group manifest. The audit refuses altered JSONL,
mixed scenes or policy revisions, missing terminal evidence, and sampled-token
log-probability recomputation outside the declared tolerance.
