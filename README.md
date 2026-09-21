# FactorioGym

An embodied reinforcement-learning environment for **Factorio** — the real
game, driven through a Lua mod over RCON with a typed protocol rather than
screen-scraping. An agent walks a character around, mines ore, places machines
and moves items, and is scored on whether the factory it built actually
produces.

The environment is paired with a pure-C11 tick-exact simulator of the same
tasks that runs roughly 3,600× faster than the engine. Policies train in the
simulator; **the real game is the verifier, never the training loop**. That
split is the point: a fast environment is only worth having if something
independent can check what it taught.

## Results

Policies trained entirely in the simulator, then replayed in **real Factorio**
on scene families they never trained on:

| task | simulator held-out | real engine held-out |
|---|---|---|
| `construct_smelting_line` | 86.8% (4 seeds) | **90.6%** |
| `build_line` | 94.9% (3 seeds) | **90.6%** |
| `plate_line` | 98.2 / 100 / 99.0% (3 seeds) | — |

Held-out means a *structurally different* scene family — an ore patch behind a
wall, a screen to walk around — not merely a different random seed. The frozen
evaluation set is an object, not a phrase: `tools/freeze_holdout.py` enumerates
the exact scenes for a fixed seed stream, hashes them, and regenerates from
source to report drift.

Two findings that moved those numbers, each measured over four seeds per arm:

- **Draw an action's arguments in order.** Sampling target, item and amount
  independently made "give 20 coal to the furnace" a 1-in-3,960 conjunction
  whose reward component sat at exactly 0.00 for 7M steps. Conditioning each
  argument on the previous took held-out success **60.0% → 86.8%** and cut the
  spread between seeds from 51 points to 10.
- **Gate a reverse curriculum on evidence, not a clock.** Sliding the
  demonstration window on a fixed schedule failed 3 runs in 4; advancing it
  only once the policy solves the window's *deepest* cut failed 0 in 4.

A bug in the game's own placement API surfaced the same way:
`can_place_entity` reports success under fast-replace semantics that
`create_entity` does not honour, silently stacking two drills on one tile. It
reproduced 29 times in 64 episodes; `mod/factoriorl/actions.lua` carries the
guard and `tools/probe_duplicate_drill.py` the reproduction.

## What you can run without owning Factorio

Most of this repository needs the game. The test suite does not:

```
uv sync
uv run pytest tests/unit tests/contract -q     # 1,320 tests, ~60 s, no engine
```

That covers the protocol, task engine, encoders, rewards, the action contract
and the frozen-holdout machinery. Engine tests are marked `engine` and skip
when the binary is absent; `--require-engine` turns that skip into a failure,
because "skipped because Factorio was unavailable" is an incomplete state
rather than a pass.

## Where to look

If you are reading this as a code sample, these are the parts worth your time:

| | |
|---|---|
| `src/factoriorl/tasks/spec.py` | the task contract: predicates, reward kinds, layout families, splits |
| `src/factoriorl/tasks/families/plate_line.py` | a task, and an unusually explicit record of *why* it is shaped as it is |
| `src/factoriorl/parameterized.py` | the `MultiDiscrete` action space and its per-dimension masks |
| `src/factoriorl/encoders.py` | game state to tensors, and the ordering guarantees the policy relies on |
| `src/factoriorl/pool.py` | multi-worker orchestration with guaranteed teardown |
| `mod/factoriorl/actions.lua` | the Lua side, including the post-creation overlap guard |
| `tools/freeze_holdout.py` | what makes "held-out" checkable |
| `tests/contract/` | action and observation contracts, tested without the engine |

`tools/freeze_holdout.py` is the one I would read first. It exists because
"held-out success rate" meant nothing checkable: each run evaluated on whatever
scenes it happened to generate, so three seeds scored three different sets and
all three were published under the same words.

## Requirements

- Windows, Python 3.11+, [uv](https://docs.astral.sh/uv/)
- **Factorio 2.0.60, build 83512**, exactly. Other builds are refused.

You need a Factorio licence; this project ships no game files. The exact build
matters and the latest release will not do — download 2.0.60 from
<https://factorio.com/download/archive/> (the headless build is enough), then:

```
set FACTORIO_RL_ENGINE=C:\path\to\factorio\bin\x64\factorio.exe
```

or copy `.factoriorl.user.json.example` to `.factoriorl.user.json` and edit the
path. Discovery order is the environment variable, then that file, then a
built-in default; the first path that exists wins. `uv run factoriorl doctor`
reports the build and **fails** if it is not 83512.

Why so strict, and what it costs you: [docs/LIMITATIONS.md](docs/LIMITATIONS.md).
The observation and action contracts are written against one build's
prototypes, so a different build is a different environment rather than a
slightly different one.

## Setup

```
uv sync --extra rl --group train   # everything, including CUDA torch
uv run factoriorl doctor           # engine present, and the pinned build
uv run factoriorl doctor-train     # torch, CUDA and the GPU
```

Lighter tiers: `uv sync` alone is stdlib only and enough for `doctor`;
`uv sync --extra rl` adds numpy and gymnasium, enough for the environment.
A source checkout is the only supported install.

## Using it

**Evaluate a checkpoint** — no API key, no network. A fresh clone has no
checkpoint, so train one first. `runtime/` and `release/` are gitignored and
nothing is published for download.

```
uv run factoriorl evaluate --checkpoint <run_id> --episodes 100
```

It reads the task and action space from the run's own manifest — those must
match, or a skill-trained policy is silently scored with a truncated catalog —
refuses a checkpoint this tree cannot feed with a diagnosis rather than a shape
error, and reports the random floor beside the rate.

**Train** — see [docs/RECIPE.md](docs/RECIPE.md) for expected numbers and the
random floor to read them against.

```
uv run factoriorl train --task deliver --steps 25000 --skills \
    --holdout docs/evidence/holdout_v3.json --eval-episodes 100
```

**Author a task** — [docs/AUTHORING_TASKS.md](docs/AUTHORING_TASKS.md).
**Connect a language model** — [docs/AGENT.md](docs/AGENT.md); optional, the
only part that can cost money, and every path through it is spend-capped.

## Commands

```
uv run factoriorl doctor             # engine + pinned build     (game setup)
uv run factoriorl doctor-train       # torch, CUDA, GPU          (dependencies)
uv run factoriorl doctor-agent       # model provider reachable  (provider)
uv run factoriorl worker start       # launch one supervised worker (connection)
uv run factoriorl worker reap        # kill engines orphaned by a failed parent
uv run factoriorl tasks list|validate
uv run factoriorl runs list|show|verify
uv run factoriorl evaluate --checkpoint <run|path>
uv run factoriorl train --task deliver --steps 25000
uv run factoriorl replay <run_dir>   # self-contained HTML, agent runs only
uv run factoriorl bench transport    # round-trip decomposition
uv run factoriorl action-matrix      # regenerate docs/ACTION_MATRIX.md
```

The first four are the diagnostics, and they are deliberately separate: a
game-setup problem, a dependency problem, a connection problem and a
model-provider problem must be distinguishable rather than all arriving as one
traceback.

## What this does and does not do

Read [docs/LIMITATIONS.md](docs/LIMITATIONS.md) before trusting any number
here. It is long on purpose.

The environment is real and the tasks are hard. **The strong results above come
through the simulator**, with the engine verifying transfer. Training directly
against the engine in the older primitive action space is much weaker: one
family reaches about 0.89 on its structural holdout against a measured random
floor of 0.01, and a three-family 80% mastery target is **unmet**. There is no
demonstrated end-to-end construction-and-recovery run. This is an environment
alpha, not a finished benchmark.

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
  session, task engine, Gymnasium environment, encoders, rewards, training.
- `src/factoriorl/pool.py` — multi-worker orchestration with guaranteed
  teardown: a context manager, `atexit`, and a Windows kill-on-close job object
  so engines cannot outlive a crashed parent.
- Each worker gets isolated write-data, mod directory, save, ports and logs
  under `runtime/workers/<id>/`, with ports claimed through cross-process
  reservations. The user's own Factorio profile is never touched.

[docs/DESIGN.md](docs/DESIGN.md) is the contract the implementation is written
against, and docstrings cite it by section.

## License

MIT; see [LICENSE](LICENSE). Factorio is a game and trademark of Wube Software
Ltd. This project is independent and not affiliated with or endorsed by Wube.
It ships no game files, and running it needs your own copy of Factorio or
Wube's headless server. See [NOTICE](NOTICE).
