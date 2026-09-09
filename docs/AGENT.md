# Connecting a language-model agent

This is the only part of the project that can cost money. Installing,
evaluating a checkpoint and running the learning recipe all use a local engine
and a local checkpoint and contact nothing.

## Check the provider first

```
uv run factoriorl doctor-agent --base-url <url> --model <name> --api-key-env OPENAI_API_KEY
```

It makes **one real call** with the actual system prompt, so it bills one
request — that is the point, since a provider that accepts a connection and
rejects the real request is the failure worth catching. It reports the
credential as `{variable, set, empty, length}` and never the value.

Defaults describe a *local* endpoint: `--base-url http://127.0.0.1:8080/v1`,
`--model local-model`, `--api-key-env ""`. A local llama.cpp, vLLM, LM Studio
or Ollama server needs no key at all and works with the defaults.

For a hosted reasoning model you usually need two more flags:

```
--token-parameter max_completion_tokens --no-temperature
```

Those exist because providers reject `max_tokens` and `temperature` on some
models. The adapter also self-heals both cases at runtime — it retries once on
an "unsupported parameter" or "unsupported value" error, records the
substitution in `healed_parameters`, and `describe()` surfaces it. That last
part matters: dropping `temperature` means the run is no longer deterministic,
so the substitution is published rather than hidden.

The cost of not having that: a `gpt-5.6-luna` run where every call was rejected
and the loop absorbed each rejection into a wait fallback. It ended after five
decisions having never once reached the model, and the cause was in
`decisions.jsonl` and nowhere in the result.

## Credentials

Two rules, both structural rather than habitual.

**A key is read from the process environment and nowhere else.** No adapter
takes a file path. `.env` in the workspace root is supported as a convenience —
it *seeds* the environment before anything reads it, which is what a shell
`set -a; . ./.env` does. Copy `.env.example` and fill in the one variable your
endpoint needs. An already-exported variable wins over the file.

Both `doctor-agent` and `demo` load it through the same function, which they
did not always do: `doctor-agent` read `os.environ` while the demo carried its
own loader, so a key present only in `.env` made the diagnostic report "not
set" about a run that then worked.

**Redaction lives next to the secret.** The adapter holds the key privately and
every string written to a run directory — model text, provider errors, the
rendered JSON — passes through a redactor that knows the value. The loop never
receives the key, so it cannot write one out by accident. The failure this
prevents is an endpoint echoing its `Authorization` header into a 401 body and
landing verbatim in `decisions.jsonl` next to results people share.

## Run one

```
uv run factoriorl demo
```

`demo` is **hardwired to `plate_line`** and to the OpenAI-compatible adapter,
and writes to a fixed evidence path. It is a demonstration entrypoint, not a
general runner. There is no `factoriorl agent --task X` subcommand; running an
agent on another task means calling `agent.runner.run_task` from Python, or
using `tools/watch_agent.py`. `AnthropicMessagesAdapter` exists and is
reachable only from Python — no CLI command constructs it. Both gaps are
recorded in `docs/LIMITATIONS.md`.

## What the agent sees, and what it writes

The prompt is a rendered text summary — task, character, inventory, visible
entities, resources, in-flight actions, recent events, and the legal actions
with their argument domains. The reply contract is JSON:
`{"action": <index>, "reason": ...}` plus optional `target` and `arguments`.
Three attempts per decision; a malformed reply and a failed call share that
budget. Exhausting it yields a `fallback_wait`, and five consecutive fallbacks
end the run as `model_failure` rather than letting it wander.

A denylist asserts the reference build geometry never reaches the prompt, so
the agent is not quietly handed the answer.

Each run directory gets `decisions.jsonl` — one line per decision, with the
prompt as sent, the model's attempts, the parsed action, the legal actions it
was offered, latency, and the result. Plus `manifest.json`, `result.json` and
`status.json`.

## Replay

```
uv run factoriorl replay runtime/runs/<run_id>
```

Writes a self-contained HTML file beside the run: map, character, inventory,
throughput, objective markers, the prompt as sent, the chosen action with its
status and error, the model's attempts, and the legal actions offered. **No
network calls and no CDN** — "a replay can be inspected without contacting the
model provider" is a property of a file with no requests in it, and a test
asserts it. Credential fields are excluded, also asserted.

It reads `decisions.jsonl`, so it works on agent runs and **refuses training
runs** with a message saying so. A trained policy's actions are not logged
anywhere; `tools/trace_scenes.py` records them for named scenes but in an
aggregate shape replay cannot read. See `docs/LIMITATIONS.md`.

A worked example ships in the release bundle:
`watch-20260909T131905-376d37f0`, a successful `plate_line` run — 79 decisions,
30 iron plates.
