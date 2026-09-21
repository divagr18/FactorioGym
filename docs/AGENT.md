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
uv run factoriorl agent --task deliver --base-url <url> --model <name> \
    --api-key-env OPENAI_API_KEY --max-cost-usd 1 --max-wall-seconds 600
```

`--task` takes a registered task **or** an open world. `open_factory` is a
fresh natural map with freeplay's ordinary starting items and no enemies:

```
uv run factoriorl agent --task open_factory --clock realtime \
    --base-url https://api.deepseek.com --model deepseek-flash \
    --api-key-env DEEPSEEK_API_KEY --thinking \
    --max-cost-usd 5 --max-wall-seconds 1800
```

Three things this command does that no earlier entrypoint did:

**It will not dispatch a request it cannot afford.** Before each call it
reserves the worst case -- every input token priced as a cache miss, output at
its ceiling -- and refuses to send if that would breach `--max-cost-usd`. The
reservation is released when the provider reports real usage. A call whose usage
never arrives keeps its reservation as committed spend and is flagged, so the
total is never presented as exact.

**The clock starts at the first gameplay observation**, not at process start, so
a ninety-second map generation does not eat the agent's budget. Every provider
request inherits whatever is left of it.

**`--clock realtime`** runs at game speed 1.0 and does not pause the world
between decisions, so the factory keeps running while the model thinks. That
makes the run a demonstration rather than a measurement, and it is recorded as
one in the manifest. Benchmark runs use `stepped`, the default.

`--adapter anthropic` selects `AnthropicMessagesAdapter`, which was previously
reachable only from Python.

### Watching a run

Two viewers, and they work together on the same artifacts.

**Live, in a real Factorio window** — `--launch-client` starts a second Factorio
and connects it to the worker, which is already a dedicated server:

```
uv run factoriorl agent --task open_factory --clock realtime --launch-client \
    --client-warmup 30 --hold-open 60 --max-cost-usd 5 --max-wall-seconds 1800
```

The joiner is made a **spectator** with no character, on both
`on_player_created` and `on_player_joined_game`, with the intro cutscene exited
and the engine-given body destroyed — so it cannot reach, mine, build or
transfer, and an identity guard means it can never destroy the agent's own body.

`--client-warmup` exists because a join stalls the server while it transfers the
map, and a stall *inside* a step is an infrastructure failure rather than a task
outcome. `--hold-open` leaves the final state on screen after the run.

**Watching is a perturbation, and the run says so.** A joined client creates a
`LuaPlayer` no measured run has; the server settings let a connected client run
console commands and pause the world; and a client reorders RCON replies —
measured at 89 inversions across one watched run against 0 on every player-free
one. So a watched run records `measurement: demonstration` and the observed
inversion count, and is not comparable to a measured one.

**Viewer presence is observed, not assumed.** The run artifact records whether a
client was requested, whether it launched, whether it survived warmup, and
whether it was alive at teardown. If the client cannot start, or starts and dies,
the run **continues headless** and the reason is written into the result rather
than printed and lost.

**Afterwards, as a self-contained HTML page** — the replay works on any agent
run, open worlds included, and takes a run id:

```
uv run factoriorl replay <run_id>
```

It writes `replay.html` beside the run: map, character, inventory, the prompt as
sent, the chosen action with its status, the model's attempts, and the legal
actions offered. **No network calls and no CDN**, asserted by a test — a replay
can be inspected without contacting the model provider.

### Prompt caching, and why it decides whether a run is affordable

For `deepseek-flash` an input token served from the provider's prefix cache
costs $0.006 per million against $0.30 for one that is not -- **fifty times**
cheaper. Over a 300-decision run at ~2,500 volatile tokens a turn, an
append-only conversation costs about **$1.19** at a 99.3% hit rate; rebuilding
the prompt each turn costs about **$34**. Against a $5 cap that is the
difference between a run that can happen and one that cannot.

So the conversation only ever grows: a retry appends the model's rejected answer
and the reason rather than rewriting the turn already sent, and a call that
failed in transport is re-sent byte-identical. `--thinking` enables the
provider's thinking mode -- note that it accepts `temperature` and then ignores
it, so such a run is **not** reproducible from its seed, and the manifest says
so.

The older demonstration entrypoint still exists:

```
uv run factoriorl demo
```

`demo` is **hardwired to `plate_line`** and to the OpenAI-compatible adapter.
It is not a plain agent run: it drives five phases — commission, measure a
production window, inject a fuel outage, recover, measure again — which is what
DESIGN 5.7's evidence *is*, so it was not folded into `factoriorl agent`.

It used to write `docs/evidence/phase5-demonstration.json` unconditionally, with
no `--out`, so running it dirtied the working tree and destroyed committed
evidence. It now writes into the run's own directory, and overwrites the tracked
file only under `--publish-evidence`. It also runs under a spend cap now
(`--max-cost-usd`, default $5): it previously built its own adapter and had no
ceiling at all.

It is a demonstration entrypoint, not a general runner. There is no `factoriorl agent --task X` subcommand; running an
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

A worked example is included when you build a bundle with
`tools/package_release.py`: `watch-20260909T131905-376d37f0`, a successful
`plate_line` run — 79 decisions, 30 iron plates. It is **not** in the
repository, because `release/` is gitignored and nothing is published for
download; you need the bundle from whoever built it, or your own agent run.
