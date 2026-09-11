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

Focused bridge and T1 tests passed (7 tests), as did Ruff and compilation. A live
probe on the RTX 4060 PC started the Windows service, retrieved a real
`construct_smelting_line` observation from Windows, and reached its authenticated
health endpoint from Ubuntu WSL. That PC has localhost forwarding disabled and
its WSL NAT traffic is firewall-blocked by default, so the probe used a temporary
port-8765 rule restricted to the active WSL subnet and removed it during cleanup.
The result is recorded in `docs/evidence/agentic-t2-bridge-2026-09-11.json`.

`training/rollout_artifacts.py` now provides the append-only JSONL artifact
format. It rejects an episode unless every model completion has aligned token
IDs and behavior log probabilities, the policy and scene identities are pinned,
and the terminal reason is explicit. Each record carries a SHA-256 of its
canonical episode payload. The focused writer/bridge/task tests pass (10 tests).

The remaining T2 work is to connect the WSL generation loop to that writer and
run a client-driven reset/act/finish rollout. Do not claim throughput or
multi-turn GRPO compatibility until that end-to-end rollout probe has run.
