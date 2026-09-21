# FactorioRL

An embodied reinforcement-learning environment for **Factorio** — the real
game, driven through a Lua mod over RCON, with a typed protocol rather than
screen-scraping. An agent walks a character around, mines, places machines and
moves items, and is scored on whether the factory it built actually produces.

It is paired with [**factory-sim**](https://github.com/divagr18/factory-sim), a
pure-C11 tick-exact simulator of the same tasks that runs ~3,600× faster than
the engine. Policies train in the simulator; **the real game is the verifier**,
never the training loop. That split is the point of the project: a fast
environment is only worth having if something independent can check it.

## Results

Policies trained entirely in the simulator, then replayed in **real Factorio**
on held-out scene families they never trained on:

| task | simulator held-out | real engine held-out |
|---|---|---|
| `construct_smelting_line` | 86.8% (4 seeds) | **90.6%** |
| `build_line` | 94.9% (3 seeds) | **90.6%** |
| `plate_line` | 98.2 / 100 / 99.0% (3 seeds) | — |

Held-out means a *structurally different* scene family — an ore patch behind a
wall, a screen to walk around — not merely a different random seed.

Two findings that moved those numbers, both measured over four seeds per arm:

- **Draw an action's arguments in order.** Sampling target, item and amount
  independently made "give 20 coal to the furnace" a 1-in-3,960 conjunction
  whose reward sat at exactly 0.00 for 7M steps. Conditioning each argument on
  the previous took held-out success **60.0% → 86.8%** and cut the spread
  between seeds from 51 points to 10.
- **Gate a reverse curriculum on evidence, not a clock.** Sliding the
  demonstration window on a fixed schedule failed 3 runs in 4; advancing it
  only when the policy solves the window's *deepest* cut failed 0 in 4.

Negative results are recorded with the same care, in
[factory-sim/docs/algorithms.md](https://github.com/divagr18/factory-sim):
critic-free learners (GRPO, RLOO) and an ACCEL-style generated curriculum were
both implemented, measured, and did not beat the baseline here.

## What you can run without owning Factorio

Most of this repository needs the game. The test suite does not:

```
uv sync
uv run pytest tests/unit tests/contract -q     # 1,334 tests, ~70 s, no engine
```

That covers the protocol, the task engine, encoders, rewards, the action
contract and the frozen-holdout machinery. Engine tests are marked `engine` and
skip when the binary is absent — `--require-engine` turns that skip into a
failure, because PLAN.md treats "skipped because Factorio was unavailable" as
an incomplete state rather than a pass.

## Where to look

If you are reading this as a code sample, these are the parts worth your time:

| | |
|---|---|
| `src/factoriorl/tasks/spec.py` | task contract: predicates, reward kinds, layout families, splits |
| `src/factoriorl/tasks/families/` | the tasks themselves; `plate_line.py` documents *why* a task is shaped as it is |
| `src/factoriorl/parameterized.py` | the `MultiDiscrete` action space and its per-dimension masks |
| `src/factoriorl/encoders.py` | game state → tensors, and the ordering guarantees the policy depends on |
| `src/factoriorl/pool.py` | multi-worker orchestration with guaranteed teardown |
| `mod/factoriorl/actions.lua` | the Lua side; see the post-creation overlap guard and why it exists |
| `tools/freeze_holdout.py` | makes "held-out" a checkable object instead of a phrase |
| `tests/contract/` | the action and observation contracts, tested independently of the engine |

`tools/freeze_holdout.py` is the one I would read first. It exists because
"held-out success rate" meant nothing checkable: every run evaluated on
whatever scenes it happened to generate, and three seeds scored three different
sets. The tool enumerates the exact scenes for a fixed seed stream, hashes
them, and regenerates from source to report drift.

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

Why so strict, and what it costs you: `docs/LIMITATIONS.md`. Short version —
the observation and action contracts are written against one build's
prototypes, so a different build is a different environment rather than a
slightly different one.

## Setup

```
uv sync --extra rl --group train   # everything, including CUDA torch
uv run factoriorl doctor           # engine present, and the pinned build
uv run factoriorl doctor-train     # torch, CUDA and the GPU
```

Lighter tiers exist if you do not need to train:

```
uv sync                 # stdlib only: `doctor` works, nothing else
uv sync --extra rl      # + numpy, gymnasium: enough for FactorioEnv
```

**A source checkout is the only supported install.**

## Doing something with it

**Evaluate a checkpoint** — no API key, no network. **A fresh clone has no
checkpoint**, so run the recipe below first, or obtain a `release/` bundle from
whoever ran one. `runtime/` and `release/` are both gitignored and nothing is
published for download; `docs/evidence/phase4-checkpoints.json` declares which
checkpoints a result cites, and does not contain them.

```
uv run factoriorl evaluate --checkpoint <run_id> --episodes 100
```

It reads the task and action space from the run's own manifest — those must
match or a skill-trained policy is silently scored with a truncated catalog —
refuses a checkpoint this tree cannot feed with a diagnosis rather than a shape
error, and reports the random floor beside the rate.

**Run the short learning recipe** — ~20 minutes on one GPU. See
[docs/RECIPE.md](docs/RECIPE.md) for the expected numbers and the random floor
they must be read against.

```
uv run factoriorl train --task deliver --steps 25000 --skills \
    --holdout docs/evidence/holdout_v3.json --eval-episodes 100
```

**Connect a language-model agent** — the only part that can cost money, and
every path through it is spend-capped. See [docs/AGENT.md](docs/AGENT.md).

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
uv run factoriorl action-matrix      # regenerate docs/ACTION_MATRIX.md
```

The first four are the diagnostics, and they are deliberately separate: a
game-setup problem, a dependency problem, a connection problem and a
model-provider problem must be distinguishable rather than all arriving as one
traceback.

## What this does and does not do

Read `docs/LIMITATIONS.md` before trusting any number here. It is long on
purpose.

The environment is real and the tasks are hard. **The strong results above come
through the simulator**, with the engine used to verify transfer. Training
directly against the engine, in the older primitive action space, is much
weaker: one family (`deliver`) reaches about 0.89 on its structural holdout
against a measured random floor of 0.01, and the three-family 80% mastery
target is **unmet**. There is no demonstrated end-to-end
construction-and-recovery run. This is an environment alpha, not a flagship
release.

## Measured characteristics

| | |
|---|---|
| RCON round trip | 16.7 ms (sentinel-framed; was 267 ms) |
| Engine step | 24.0 ms → **41.6 steps/s** on one worker |
| Engine, 8 workers | 261 decisions/s |
| Simulator, 16 CPU threads | **947,000 decisions/s** (~3,600× the engine) |
| Reset | 6.2 ms |
| Observation | 1.1–5.3 KB |

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

Project documents: `PLAN.md` for strategy, `docs/LEDGER.md` for gate status and
evidence, `docs/ACTION_MATRIX.md` for the action contract, `docs/archive/` for
superseded session notes.

## License

MIT; see [LICENSE](LICENSE). Factorio is a game and trademark of Wube Software
Ltd. This project is independent and not affiliated with or endorsed by Wube.
It ships no game files, and running it needs your own copy of Factorio or
Wube's headless server. See [NOTICE](NOTICE).
