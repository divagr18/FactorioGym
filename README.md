<p align="center">
  <img src="docs/factorio-gym-banner.svg" alt="FactorioGym: reinforcement learning for factory construction" width="100%">
</p>

# FactorioGym

Reinforcement learning on Factorio construction tasks, with the real game as
the judge.

FactorioGym drives headless Factorio 2.0 from Python through a Lua mod. An agent
walks, mines by hand, places burner drills, stone furnaces, transport belts,
burner inserters and chests, and fuels them. A task is scored on what the
factory it built actually produces. Every task has held-out layout families,
and a frozen, hashed holdout set, so a result names the exact scenes it was
measured on.

The game is too slow to train in: about 260 decisions per second across eight
workers. Training happens in **[factory-sim](https://github.com/divagr18/factory-sim)**,
a C reimplementation of the same mechanics. It runs about 950,000 decisions per
second on 16 threads and is checked tick by tick against traces recorded here.
The two repositories are halves of one system:

```
                 golden traces, task scenes, frozen holdout
  FactorioGym  ─────────────────────────────────────────────►  factory-sim
  the real game                                                fast simulator
  (verification)  ◄───────────────────────────────────────────  (training)
                   trained policies, evaluated on the engine
```

You need factory-sim to *train* at a useful speed. You need FactorioGym, and a
copy of Factorio, to find out whether what you trained works in the game.

## Results: trained in the simulator, evaluated in the game

One checkpoint per task. It was trained only in factory-sim, then played on
Factorio 2.0.60 through the same `ParameterizedEnv` any policy uses. The
simulator column is the same checkpoint scored in the simulator.

| Task | Layouts | Simulator (512 episodes) | Factorio (32 episodes) |
|---|---|---|---|
| `construct_smelting_line` | training families | 78.3% | 75.0% |
| | **held out** (`obstructed_patch`) | 92.0% | **90.6%** |
| `build_line` | training families | 99.6% | 100% |
| | **held out** (`narrow_patch`) | 99.2% | **90.6%** |

Held-out families are layouts the policy never trained on, such as an ore patch
behind a wall, or a long narrow strip. With 32 episodes, 29/32 carries a 95%
interval of roughly 76–97%. The simulator overestimated held-out `build_line`
by about eight points; on `construct_smelting_line` the two agree within
1.4–3.3 points.

Every episode's scene digest and action vectors are recorded, so the engine run
can be replayed in the simulator:
[construct_smelting_line](docs/evidence/sim-transfer-m5.json),
[build_line](docs/evidence/sim-transfer-build-line.json).

These are sampled policies. Played greedily (argmax), the same checkpoints
succeed about 3% of the time: they rely on sampling to break out of repeated
actions.

## Results: programs written by language models, played in the game

factory-sim can also score short Python programs, `def build(world):`, that
act through the same action space. `tools/program_transfer.py` plays them on
the engine, checks each scene's digest against the simulator's, and replays
every episode in the simulator.

| Programs | Scenes | Factorio | Agrees with the simulator |
|---|---|---|---|
| Found by program search: from each of 4 runs, the program with the best validation score | the same 100 held-out `construct_smelting_line` scenes | 397/400 (99.3%) | 400/400 |
| Written by Qwen3.5-9B after SFT + GRPO on factorio-build: a random 16 of its holdout programs | the same 10 held-out scenes | 136/160 (85%) | 160/160 |

Evidence: [program search](docs/evidence/program-transfer.json),
[Qwen3.5-9B](docs/evidence/program-transfer-grpo-pc.json) (and its
[second half](docs/evidence/program-transfer-grpo-laptop.json)). The training
recipes, and the Evolve & Reinforce model that keeps most of the base model's
planning ability, are in factory-sim's README and its
[factorio-build](https://github.com/divagr18/factory-sim/tree/main/integrations/verifiers/factorio_build)
environment; the models are in the [models collection].

## Belts and inserters

Transport belts, burner inserters and wooden chests are measured tick by tick
on the engine: belt-line segments and their merge delays, turns, sideloads,
loops, inserter swings and fuel, and inserters picking from moving belts.
[docs/sim-logistics.md](docs/sim-logistics.md) records every measurement and
the rule factory-sim implements from it; the probes are the
`tools/probe_logistics*.py`, `tools/probe_inserter_*.py` and
`tools/probe_handmine*.py` scripts.

`belt_smelting` 1.1.0 uses them. An iron patch, a coal patch and an output
chest are too far apart for one standing spot to reach two of them. The agent
starts with 4 drills, 4 furnaces, 40 belts, 10 burner inserters and 20 coal,
and must deliver 150 iron plates into the chest during an action-locked
ten-minute window. Twenty coal runs a line for only about 79 plates, so the
line also needs coal from the coal patch. The reference solver hand-mines 40
coal first and delivers 197-198 plates in 200 of 200 engine episodes (100
train, 100 test scenes;
[evidence](docs/evidence/belt-smelting-gate.json)). factory-sim ports the
task draw for draw, and its port of the solver succeeds on 599 of the first
600 simulator scenes.

## What you need

| To | You need |
|---|---|
| Run the test suite | Python 3.11+ and [uv](https://docs.astral.sh/uv/). No game. |
| Run tasks on the engine | Factorio **2.0.60, build 83512**; the headless server is enough. Windows is the tested platform. |
| Train policies at speed | [factory-sim](https://github.com/divagr18/factory-sim) and an NVIDIA GPU |
| Train directly on the engine | The training extras below. It works, but only small tasks are practical at engine speed. |

The build is pinned because the action and observation contracts depend on
that build's prototypes; other builds are refused. Get it from the
[Factorio release archive](https://factorio.com/download/archive/). No game
files ship with this repository.

## Quick start

Tests, without the game:

```powershell
uv sync
uv run pytest tests/unit tests/contract -q
```

Point FactorioGym at the game, then check the setup:

```powershell
$env:FACTORIO_RL_ENGINE = "C:\path\to\factorio\bin\x64\factorio.exe"
uv sync --extra rl --group train
uv run factoriorl doctor         # engine path and pinned build
uv run factoriorl doctor-train   # PyTorch, CUDA, GPU
```

You can instead copy [`.factoriorl.user.json.example`](.factoriorl.user.json.example)
to `.factoriorl.user.json` and put the path there.

### Evaluate a simulator-trained policy on the game

Train in factory-sim (see its README), which writes `policy.ts` and
`final.json` to a run directory, then:

```powershell
uv run python tools/sim_transfer.py --policy ..\factory-sim\runs\<run>\policy.ts `
    --sim-report ..\factory-sim\runs\<run>\final.json --episodes 32
```

This step does not import factory-sim: the policy is a TorchScript file, and it
plays through FactorioGym's own environment.

### Train on the engine

A small task, trained on the game itself:

```powershell
uv run factoriorl train --task deliver --steps 25000 --skills --holdout docs/evidence/holdout_v3.json --eval-episodes 100
uv run factoriorl evaluate --checkpoint <run_id> --episodes 100
```

Evaluation reads the task and action space from the run's manifest, and prints
a random-policy baseline beside the success rate. Pass the frozen holdout
explicitly. A fixed training seed does not fix the evaluation scenes; the
holdout file does. [The learning recipe](docs/RECIPE.md) has expected results
and runtimes.

## How it works

- **Engine control.** Each worker is a headless Factorio process with its own
  save, mod directory, ports and logs under `runtime/workers/`. Python talks to
  it over RCON through a typed protocol. A Windows job object kills the engines
  if the parent process dies. Your own Factorio profile is never touched.
- **Actions.** `ParameterizedEnv` exposes `MultiDiscrete([22, 33, 122, 5, 15, 4])`:
  operation, target, placement tile, direction, item and amount, each with its
  own mask. Engine edge cases are guarded in the mod: for example, the game
  accepts a second drill placed on top of the first
  ([probe](tools/probe_duplicate_drill.py)). The v3 profile, which
  `belt_smelting` uses, is `MultiDiscrete([25, 97, 226, 5, 19, 4])`: the same
  22 operations plus hand-mining a resource tile, `take_fuel` and `finish`,
  with a mask per operation that follows the engine's own reach rules.
- **Observations.** A local tile grid around the character, a table of nearby
  entities, the inventory, character state and the task goal. `local-v3` adds
  each belt's lane counts and turn, each inserter's hand and its pickup and
  drop points, a drill's drop point, the task's public markers and the free
  inventory slots.
- **Tasks.** A task declares its layout families, train/test split, reward and
  success predicate ([authoring guide](docs/AUTHORING_TASKS.md)). The
  construction tasks count only plates made by machines during a verification
  minute in which the agent cannot act, so a hand-fed furnace or a single
  plate does not pass.
- **Holdout.** [`freeze_holdout.py`](tools/freeze_holdout.py) generates a
  scene set, hashes every scene, and detects drift by regenerating them from
  source ([`holdout_v3.json`](docs/evidence/holdout_v3.json)).

The contracts the implementation is written against are in
[docs/DESIGN.md](docs/DESIGN.md).

## Limits

- **Small factories.** The learned results are a drill, a furnace and fuel.
  No belts, inserters or power in any learned result yet: `belt_smelting` has
  a reference solver, measured mechanics and a simulator port, but no trained
  policy or model.
- **Engine training is weak.** The strong results above come from simulator
  training. Policies trained directly on the engine haven't met the project's
  own bar (three task families at 80% held out).
- **Small engine samples.** 32 episodes per split for the policies, 10 to 100
  scenes per program, so the intervals are wide.
- **Tested on Windows only.** Other platforms may work, but haven't been tried.
- **No policy checkpoints are distributed.** The trained language models are
  in the [models collection].

[docs/LIMITATIONS.md](docs/LIMITATIONS.md) has the details and the
measurements behind each one.

## Commands

| Command | Purpose |
|---|---|
| `uv run factoriorl tasks list` | List the tasks |
| `uv run factoriorl tasks validate` | Check task definitions |
| `uv run factoriorl train` / `evaluate` | Train or evaluate on the engine |
| `uv run factoriorl runs list` | List recorded runs |
| `uv run factoriorl worker start` / `reap` | Launch a supervised engine, or kill orphaned ones |
| `uv run factoriorl bench transport` | Measure engine round-trip latency |
| `uv run factoriorl doctor-agent` | Check a language-model provider ([agent guide](docs/AGENT.md)) |
| `uv run factoriorl replay <run_dir>` | Render an HTML replay of a language-model agent's run |
| `uv run python tools/program_transfer.py --program-db <genealogy.sqlite>` | Play `def build(world):` programs on the engine beside the simulator |
| `uv run python tools/demo_play.py --record <file>` | Film a scripted build of coal, iron and copper sites in a watching client; `tools/make_clip.py` and `tools/make_showreel.py` cut the recording |

## Citing

If you use FactorioGym in academic work, please cite it. GitHub's "Cite this
repository" button reads [CITATION.cff](CITATION.cff); in BibTeX:

```bibtex
@software{agrawal2026factoriogym,
  author  = {Agrawal, Divyansh},
  title   = {{FactorioGym}: an embodied Factorio environment for reinforcement learning},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/divagr18/FactorioGym},
  license = {Apache-2.0}
}
```

## License

[Apache-2.0](LICENSE). Factorio is a game and trademark of Wube Software Ltd.
This project is independent, and is not affiliated with or endorsed by Wube.
No game files are included. Running engine tasks needs your own copy of
Factorio or Wube's headless server. See [NOTICE](NOTICE).
