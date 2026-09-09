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
- **Factorio 2.0.60, build 83512**, exactly. Other builds are refused.

### Getting that build

You need a Factorio licence; this project ships no game files. The exact build
matters and the latest release will not do, so:

1. Buy or sign in at [factorio.com](https://factorio.com).
2. Download **2.0.60** from the release archive at
   <https://factorio.com/download/archive/> — the headless build is enough and
   is the smaller download. A Steam install will normally be a newer version
   and cannot be pinned back, so the archive is the reliable route.
3. Point this project at it, either way round:

```
set FACTORIO_RL_ENGINE=C:\path\to\factorio\bin\x64\factorio.exe
```

or copy `.factoriorl.user.json.example` to `.factoriorl.user.json` and edit the
path. Discovery order is the environment variable, then that file, then a
built-in default; the first path that exists wins.

`uv run factoriorl doctor` reports the build and **fails** if it is not 83512.

Why so strict, and what it costs you: `docs/LIMITATIONS.md`. Short version --
the observation and action contracts are written against one build's
prototypes, so a different build is a different environment rather than a
slightly different one.

## Setup

```
uv sync --extra rl --group train   # everything, including CUDA torch
uv run factoriorl doctor           # engine present, and the pinned build
uv run factoriorl doctor-train     # torch, CUDA and the GPU
uv run pytest tests/unit tests/contract -q   # engine-free, ~30 s
```

Lighter tiers exist if you do not need to train:

```
uv sync                 # stdlib only: `doctor` works, nothing else
uv sync --extra rl      # + numpy, gymnasium: enough for FactorioEnv
```

**A source checkout is the only supported install.** `pip install factoriorl`
would give you the Python package without `mod/` or `tools/`, and neither a
worker nor `demo`/`replay` can start without those — see `docs/LIMITATIONS.md`.
An earlier version of this file claimed a unit test proved the runtime imports
cleanly without numpy and gymnasium. There is no such test; the claim is
withdrawn.

## Doing something with it

Four things, in the order a newcomer wants them. Full walkthroughs in `docs/`.

**Evaluate a provided checkpoint** — no API key, no training, no network.

```
uv run python tools/package_release.py --runs shaped-20260909T143924-c49958ea
uv run factoriorl evaluate --checkpoint shaped-20260909T143924-c49958ea --episodes 100
```

Checkpoints live in `runtime/runs/`, which is gitignored, so a fresh clone has
the *declaration* (`docs/evidence/phase4-checkpoints.json`) and not the files.
`package_release.py` assembles a `release/` bundle from runs you have; if you
have none yet, train one with the recipe below. `evaluate` reads the task and
action space from the run's own manifest and refuses a checkpoint this tree
cannot feed, with a diagnosis rather than a shape error.

**Run the short learning recipe** — ~20 minutes on one GPU. See
[docs/RECIPE.md](docs/RECIPE.md) for the expected numbers and the random floor
they must be read against.

```
uv run factoriorl train --task deliver --steps 25000 --skills \
    --holdout docs/evidence/holdout_v3.json --eval-episodes 100
```

**Connect a language-model agent** — the only part that can cost money. See
[docs/AGENT.md](docs/AGENT.md).

```
uv run factoriorl doctor-agent --base-url <url> --model <name> --api-key-env OPENAI_API_KEY
uv run factoriorl demo
uv run factoriorl replay runtime/runs/<run_id>
```

`replay` renders a self-contained HTML file with no network calls. It works on
**agent** runs; a training run has no per-decision trace and is refused.

**Author a task** — see [docs/AUTHORING_TASKS.md](docs/AUTHORING_TASKS.md).

## Commands

```
uv run factoriorl doctor             # engine + pinned build     (game setup)
uv run factoriorl doctor-train       # torch, CUDA, GPU          (dependencies)
uv run factoriorl doctor-agent       # model provider reachable  (provider)
uv run factoriorl worker start       # launch one supervised worker (connection)
uv run factoriorl worker reap        # kill engines orphaned by a failed parent
uv run factoriorl tasks list         # registered task families
uv run factoriorl tasks validate     # fail before training, not at step 40,000
uv run factoriorl runs list|show|verify
uv run factoriorl evaluate --checkpoint <run|path>
uv run factoriorl train --task deliver --steps 25000
uv run factoriorl demo               # language-model agent on plate_line
uv run factoriorl replay <run_dir>   # self-contained HTML, agent runs only
uv run factoriorl bench transport    # round-trip decomposition
uv run factoriorl bench speed        # game.speed changes pacing, not outcomes
uv run factoriorl action-matrix      # regenerate docs/ACTION_MATRIX.md
uv run factoriorl phase0-gate        # ... through phase4-gate
```

The first four are the diagnostics, and they are deliberately separate: PLAN 6.1
requires that a game-setup problem, a dependency problem, a connection problem
and a model-provider problem be distinguishable rather than all arriving as one
traceback.

## Tests

```
uv run pytest tests/unit tests/contract   # engine-free, fast
uv run pytest tests/engine                # engine integration (real binary)
uv run ruff check . && uv run ruff format --check .
```

Engine tests are marked `engine`; they skip when the executable is absent. Pass
`--require-engine` (as the phase gates do) to turn that skip into a failure --
PLAN.md treats "skipped because Factorio was unavailable" as an incomplete
state, not a pass.

## What this does and does not do

Read `docs/LIMITATIONS.md` before trusting any number here. It is long on
purpose, and `docs/RELEASE_CHECKLIST.md` lists what is deferred and not
available.

The environment is real and the tasks are hard. The learning results are
**weak, and reported separately from mastery**: one family (`deliver`) reaches
about 0.89 on its structural holdout against a measured random floor of 0.01,
which is learning. The three-family 80% mastery target is **unmet**. There is
no demonstrated construction-and-recovery run. This is an environment alpha,
not a flagship release.

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
