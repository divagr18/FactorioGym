# Contributing

Issues and pull requests are welcome. Most of this project needs a Factorio
licence to exercise fully, so the sections below are ordered by how much of the
game you need: none, then some.

## Without the game

The test suite does not need Factorio, and it covers the protocol, the task
engine, encoders, rewards, the action contract and the frozen-holdout
machinery.

```
uv sync                                        # stdlib + pytest + ruff
uv run pytest tests/unit tests/contract -q     # ~1,320 tests, about a minute
uv run ruff check .
uv run ruff format --check .
```

That is enough to change task definitions, reward shaping, the action space,
the encoders or any of the tooling under `tools/`, and to know whether you
broke something.

## With the game

Requires **Factorio 2.0.60, build 83512** exactly; other builds are refused,
and `docs/LIMITATIONS.md` explains why the pin is that tight.

```
uv sync --extra rl --group train
uv run factoriorl doctor          # engine present, and the pinned build
uv run pytest tests/engine        # real engine, real ticks
```

Engine tests carry the `engine` marker and skip when the binary is absent.
Pass `--require-engine` to turn that skip into a failure: "skipped because
Factorio was unavailable" is an incomplete state, not a pass, and anything
claiming to be verified against the engine should be run that way.

## What a good change looks like

**Say what you measured.** This repository's numbers are attached to the
artifacts that produced them, under `docs/evidence/`. A change that moves a
published number should move the evidence with it, in the same commit. A
number without a recoverable scene set describes nothing — that is what
`tools/freeze_holdout.py` exists to prevent.

**Add tasks through the documented path.** `docs/AUTHORING_TASKS.md` walks
through a task family end to end. `src/factoriorl/tasks/families/plate_line.py`
is the worked example, and it records why it is shaped the way it is as well as
what it does.

**Do not weaken a check to make it pass.** Lowering a threshold, relaxing an
assertion or substituting scripted behaviour for learned behaviour turns a
failing test into a false one. If a check is wrong, change it deliberately and
say so in the commit; if it is right, the code is what needs to move.

**Keep the docstring citations honest.** Docstrings throughout `src/` cite
`docs/DESIGN.md` by section. If you change behaviour the design describes,
update the section rather than leaving the citation pointing at prose that no
longer matches.

**Regenerate, do not hand-edit, generated files.** `docs/ACTION_MATRIX.md`
comes from source:

```
uv run factoriorl action-matrix
```

## Things that will get a change rejected

- **Committing runtime state.** Worker directories, saves, engine logs and port
  reservations live under `runtime/`, which is gitignored. Evidence files under
  `docs/evidence/` are small and *are* committed, because a result whose
  artifact is not in source control cannot be re-checked later.
- **Touching a real Factorio installation or profile.** Workers are isolated by
  construction: their own write-data directory, mod directory, save, ports and
  logs. Nothing may write outside that sandbox.
- **Changing protocol v1 without a version bump.** `src/factoriorl/protocol.py`
  and `mod/factoriorl/` are a matched pair, and the fixtures under
  `tests/fixtures/protocol_v1/` are the contract. A change to the wire format
  needs a new version and new fixtures, not an edit to the old ones.
- **Machine-specific paths, hostnames or credentials in committed files.**
  `tools/package_release.py` redacts them out of run manifests and then audits
  its own output; committed source should not need the same treatment.

## Factorio's intellectual property

This project ships no game files and is not affiliated with Wube Software.
[NOTICE](NOTICE) has the full statement; the short version for contributors:

- Never copy code, prototype definitions or data files out of the game, or out
  of any project whose licence does not permit it.
- Never commit game binaries, graphics, sounds, fonts, locale files, saves or
  dumps of the game's prototype data.
- Anything that simulates a game mechanic is written from scratch, from this
  project's own measurements — which is what `docs/evidence/` records.

## Licence

Contributions are accepted under the Apache License 2.0; see [LICENSE](LICENSE).
