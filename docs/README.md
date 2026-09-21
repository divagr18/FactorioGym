# Documents

| | |
|---|---|
| [DESIGN.md](DESIGN.md) | the architecture and behavioural contracts the implementation is written against. Docstrings cite it by section. |
| [LIMITATIONS.md](LIMITATIONS.md) | what this environment cannot do, and what each number is worth. Read before trusting a result. |
| [ACTION_MATRIX.md](ACTION_MATRIX.md) | the action contract, generated from source by `factoriorl action-matrix` |
| [RECIPE.md](RECIPE.md) | the learning recipe, its expected numbers, and the random floor to read them against |
| [AUTHORING_TASKS.md](AUTHORING_TASKS.md) | how to add a task family |
| [AGENT.md](AGENT.md) | connecting a language model, the spend cap, and the replay viewer |

## evidence/

The artifacts results cite, rather than the results themselves.
`holdout_v3.json` is the frozen held-out evaluation set — the object that makes
"held-out success rate" checkable, produced and verified by
`tools/freeze_holdout.py`. The reward audit, solvability sweeps, engine
benchmarks and simulator parity records live beside it.

Several are loaded directly by tests, so they are inputs as well as records.
