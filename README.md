# FactorioRL

Embodied Factorio environment for training compact RL policies, developing
language-model agents, and comparing hybrid systems in the same game world.
Flagship target: an agent that builds a factory, recovers from disruption, and
continues toward a peaceful base-game rocket launch.

Status: **Phases 0–3 accepted**. Phase 4's RL baseline runs and its validity items are closing; its three-family mastery target (4.5) is **unmet** and recorded as such. Current work follows R0–R6 in [docs/DEVELOPMENT_REDIRECTION.md](docs/DEVELOPMENT_REDIRECTION.md).
See `PLAN.md` for the strategy, `docs/LEDGER.md` for gate status and evidence,
and `docs/ACTION_MATRIX.md` for the action contract.

## Requirements

- Windows, Python 3.11+, [uv](https://docs.astral.sh/uv/)
- Factorio **2.0.60 (build 83512)** — the pinned engine build
  (`src/factoriorl/engine_config.py`); other builds are refused

Engine discovery order: `FACTORIO_RL_ENGINE` env var → `.factoriorl.user.json`
(`{"engine": {"executable": "<path>"}}` in the repo root, gitignored) →
default `D:\Factorio\bin\x64\factorio.exe`.

## Setup

```
uv sync                       # runtime only (stdlib)
uv sync --extra rl            # + numpy, gymnasium: enough for FactorioEnv
uv sync --group train         # + torch/sb3-contrib, CUDA build, local only
uv run factoriorl doctor      # probe the engine and the pinned build
uv run factoriorl doctor-train  # check torch, CUDA and the GPU
```

`rl` is a PEP 621 extra so `pip install factoriorl[rl]` gives an outside
researcher a working environment; `train` is a PEP 735 dependency group so a
multi-GB CUDA torch never enters wheel metadata. `[project] dependencies` stays
empty — a unit test blocks numpy and gymnasium and imports the whole runtime
path to prove it.

## Commands

```
uv run factoriorl worker start      # launch one supervised worker
uv run factoriorl worker reap       # kill engines orphaned by a failed parent
uv run factoriorl tasks list        # registered task families
uv run factoriorl tasks validate    # fail before training, not at step 40,000
uv run factoriorl runs list|show|verify
uv run factoriorl train --task navigate --steps 40000
uv run factoriorl bench transport   # round-trip decomposition
uv run factoriorl bench speed       # game.speed changes pacing, not outcomes
uv run factoriorl action-matrix     # regenerate docs/ACTION_MATRIX.md
uv run factoriorl phase0-gate       # ... through phase4-gate
```

## Tests

```
uv run pytest tests/unit tests/contract   # engine-free, fast
uv run pytest tests/engine                # engine integration (real binary)
uv run ruff check . && uv run ruff format --check .
```

Engine tests are marked `engine`; they skip when the executable is absent. Pass
`--require-engine` (as the phase gates do) to turn that skip into a failure —
PLAN.md treats "skipped because Factorio was unavailable" as an incomplete
state, not a pass.

## Measured characteristics

| | |
|---|---|
| RCON round trip | 16.7 ms (sentinel-framed; was 267 ms) |
| Env step | 24.0 ms → **41.6 steps/s** on one worker |
| Reset | 6.2 ms |
| Observation | 1.1–5.3 KB |
| `advance(30)` at `game.speed=30` | 25.6 ms (was 819 ms) |

## Architecture

- `mod/factoriorl/` — Lua: authoritative game interaction, stepping,
  observations, entity handles, the action matrix. Talks to Python only through
  the typed protocol; arbitrary Lua is evaluator-only.
- `src/factoriorl/` — worker lifecycle, RCON transport, protocol v2, typed
  session, task engine, Gymnasium env, encoders, rewards, training.
- `src/factoriorl/pool.py` — multi-worker orchestration with guaranteed
  teardown (context manager, `atexit`, and a Windows kill-on-close job object so
  engines cannot outlive a crashed parent).
- Each worker gets isolated write-data, mod directory, save, ports and logs
  under `runtime/workers/<id>/`. Ports are claimed through cross-process
  reservations in `runtime/ports/`. The user's Factorio profile is never
  touched.

## License / provenance

Private research project; Factorio is property of Wube Software Ltd.
