# Contributing to FactorioRL

One owner coordinates many coding agents. Every assignment uses the work
package template below (`docs/WORK_PACKAGE_TEMPLATE.md`); phases are gated by
evidence in `docs/LEDGER.md`.

## Setup

```
uv sync
uv run factoriorl doctor      # probes the pinned Factorio build
```

Requires Factorio **2.0.60 (build 83512)**. Configure a non-default path via
`FACTORIO_RL_ENGINE` or `.factoriorl.user.json` in the repo root (gitignored):

```json
{"engine": {"executable": "D:/Factorio/bin/x64/factorio.exe"}}
```

## Lint and tests

```
uv run ruff check .            # lint
uv run ruff format --check .   # format
uv run pytest tests/unit tests/contract   # no engine needed
uv run pytest tests/engine               # engine integration (real Factorio)
uv run factoriorl phase1-gate            # Phase 1 gate + recorded evidence
```

Engine tests are marked `engine`. They skip when the binary is absent; the
phase gates pass `--require-engine`, which turns that skip into a failure, so a
gate can never pass on skipped engine tests.

## Rules

- Bulky runtime state lives under `runtime/` (gitignored): worker dirs, saves,
  engine logs, port reservations. Never commit it. Gate *transcripts* are small
  and do get committed, under `docs/evidence/`; a gate whose evidence is not in
  source control cannot be re-checked later.
- Mark a phase Accepted in `docs/LEDGER.md` only after its gate has run and
  passed, with the transcript committed. Phase 1 was once marked Accepted
  before its gate ran, against a run that failed 11 of 27 tests; the ledger
  entry must always be downstream of the evidence, never ahead of it.
- Never touch the user's Factorio profile or the `D:/Factorio` installation.
  Workers are isolated by construction (own write-data, mod directory, ports).
- Protocol v1 (`src/factoriorl/protocol.py` + `mod/factoriorl/`) is frozen:
  changes require a version bump and updated fixtures under
  `tests/fixtures/protocol_v1/`.
- Engine-dependent acceptance criteria are verified against the real binary;
  "works with a mock" is an incomplete state (PLAN.md definition of done).
- Do not lower gate thresholds or substitute scripted behavior to pass a gate.
