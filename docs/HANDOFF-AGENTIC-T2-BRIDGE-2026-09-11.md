# Agentic RL T2 bridge handoff

The first T2 boundary is implemented. It is not a rollout-training result.

`tools/agentic_bridge.py` runs beside Factorio on Windows. It creates one
isolated `FactorioEnv` and publishes an authenticated bridge with exactly four
operations: reset, observe, act, and finish. The handler serializes requests,
which is required because the worker session is single-threaded.

`src/factoriorl/agentic/bridge.py` is the policy-facing contract. It returns
only raw observation, catalog metadata, argument domains, action masks, and
transition/verification outcomes. It never returns evaluator truth,
`machine_produced` totals, Lua, RCON credentials, filesystem paths, or generic
protocol access. An action must be a current catalog index with exactly the
arguments declared by that template. `finish` calls the T1 verifier once and
ends the episode.

`training/bridge_client.py` is a dependency-free WSL client. It uses a bearer
token supplied through environment configuration; no credential is written into
the repository or transcript.

Focused bridge and T1 tests passed (7 tests), as did Ruff and compilation. The
remaining T2 work is a real cross-host launch and a rollout recorder that stores
token ids, behaviour log probabilities, tool calls/results, policy version, and
termination reason. Do not claim WSL-to-Windows throughput or multi-turn GRPO
compatibility until that end-to-end probe has run.
