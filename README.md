# FactorioRL

Embodied Factorio environment for training compact RL policies, developing
language-model agents, and comparing hybrid systems in the same game world.
Flagship target: an agent that builds a factory, recovers from disruption, and
continues toward a peaceful base-game rocket launch.

Status: **Phase 0–1** (engine feasibility, protocol v1, worker lifecycle).
See `PLAN.md` for the full strategy and phase plan, and `docs/LEDGER.md` for
gate status and evidence.

## Requirements

- Windows, Python 3.11+, [uv](https://docs.astral.sh/uv/)
- Factorio **2.0.60 (build 83512)** — the pinned engine build
  (`src/factoriorl/engine_config.py`); other builds are refused

Engine discovery order: `FACTORIO_RL_ENGINE` env var → `.factoriorl.user.json`
(`{"engine": {"executable": "<path>"}}` in the repo root, gitignored) →
default `D:\Factorio\bin\x64\factorio.exe`.

## Setup

```
uv sync
uv run factoriorl doctor
```

`doctor` probes the engine and prints the resolved/pinned build metadata.

## Commands

```
uv run factoriorl worker start    # launch one supervised worker, print handshake
uv run factoriorl worker reap     # kill engines orphaned by a failed parent run
uv run factoriorl phase0-gate     # run the Phase 0 exit gate end to end
uv run factoriorl phase1-gate     # run the Phase 1 exit gate, record its evidence
```

`phase1-gate` runs the engine suite with `--require-engine`, so it fails rather
than skips when Factorio is missing, and writes the transcript to
`docs/evidence/phase1-engine-suite.txt`.

## Tests

```
uv run pytest tests/unit tests/contract   # no Factorio required
uv run pytest tests/engine                # engine integration (real binary)
uv run ruff check . && uv run ruff format --check .
```

Engine tests are marked `engine`; they are skipped (not failed) when the
executable is absent. Pass `--require-engine` (as the phase gates do) to turn
that skip into a failure -- PLAN.md treats "skipped because Factorio was
unavailable" as an incomplete state, not a pass.

## Architecture sketch

- `mod/factoriorl/` — Lua mod: authoritative game interaction, stepping,
  observations, episode bookkeeping. Talks to Python only through the typed
  protocol; arbitrary Lua is evaluator-only.
- `src/factoriorl/` — Python: worker lifecycle (`worker.py`), RCON transport
  (`rcon.py`), protocol v1 types (`protocol.py`), typed session (`session.py`).
- `src/factoriorl/pool.py` -- multi-worker orchestration: a registry with
  guaranteed teardown (context manager, `atexit`, and a Windows kill-on-close
  job object so engines cannot outlive a crashed parent), plus orphan reaping.
- Each worker gets isolated write-data, mod directory, save, ports, and logs
  under `runtime/workers/<id>/` (gitignored). Ports are claimed through
  cross-process reservations in `runtime/ports/`, so two Python processes
  cannot hand the same pair to two engines. The user's Factorio profile and
  mods are never touched.

## License / provenance

Private research project; Factorio is property of Wube Software Ltd.
