# Release checklist and compatibility statement

PLAN 6.4. Run through this before calling anything a release, and read the
"deferred" list at the bottom as part of it — "deferred functionality is not
advertised as available" is an acceptance criterion, not a courtesy.

## Compatibility statement

| | |
|---|---|
| release kind | **environment alpha** — not the flagship first public release |
| engine | Factorio **2.0.60, build 83512**, exactly; refused otherwise |
| platform | Windows. Nothing is deliberately Windows-specific; nothing has been run elsewhere |
| Python | 3.11+ |
| install | source checkout via `uv sync`. The wheel does not work — see LIMITATIONS §8 |
| protocol | 2 |
| observation profile | `local-v2` |
| feature extractor | version **7** |
| goal encoding | version 4 |
| agent summary encoding | version 3 |
| frozen holdouts | `holdout_v1`, `holdout_v2`, `holdout_v3` (live: v3) |
| declared checkpoints | `docs/evidence/phase4-checkpoints.json` |

Task versions at this release:

| task | version | | task | version |
|---|---|---|---|---|
| `build_line` | 1.1.0 | | `navigate` | 1.2.0 |
| `deliver` | 1.3.0 | | `plate_line` | 1.2.0 |
| `diagnose_line` | 1.0.0 | | `repair_belt` | 1.6.0 |
| `keep_line_running` | 1.0.0 | | `restore_power` | 1.6.0 |
| `mine_smelt` | 1.2.0 | | `supply_furnace` | 1.2.0 |

Any of these moving is a compatibility break for a checkpoint or a frozen
holdout entry, not a patch. A task version bump invalidates its holdout entry
on its own, and an extractor version bump can invalidate a checkpoint —
`architecture_signature` is the thing to compare, because only one of six
extractor bumps ever changed a shape the loader could not absorb.

## Required before release

**Tests and lint**

- [ ] `uv run ruff check . && uv run ruff format --check .`
- [ ] `uv run pytest tests/unit tests/contract -q` — 972 at this writing, all green
- [ ] `uv run pytest tests/engine --require-engine` — the skip must be a failure, not a pass
- [ ] `uv run factoriorl tasks validate`
- [ ] `uv run factoriorl phase4-gate --mode reproduce` — 28 of 28

**Diagnostics distinguish the four failure classes** (PLAN 6.1)

- [ ] `factoriorl doctor` — game setup; **fails** on a build other than 83512
- [ ] `factoriorl doctor-train` — dependencies; fails loudly on a CPU-only wheel
- [ ] `factoriorl worker start` — connection
- [ ] `factoriorl doctor-agent` — model provider

**The five gate clauses, by someone who did not build it** (PLAN 6.3)

- [ ] install from the public instructions on a clean environment
- [ ] evaluate a provided checkpoint
- [ ] run the short learning recipe
- [ ] connect an agent
- [ ] author a small task

Every undocumented manual step found here is a release blocker, not a note.

**Artifacts** (PLAN 6.2)

- [ ] `uv run python tools/package_release.py --runs <ids>` exits 0
- [ ] its audit reports clean — no absolute paths, no hostname, no secret-shaped strings
- [ ] the bundle contains at least one `model.zip` and one `decisions.jsonl`
- [ ] every run in the bundle cites a holdout the bundle also contains
- [ ] `MANIFEST.json` records hashes, engine, protocol and all ten task versions
- [ ] `excluded_evidence` lists any file whose cited holdout hash resolves against nothing
- [ ] evaluation needs no paid API — true by construction; `evaluate` is local only

**Honesty**

- [ ] learning is reported separately from mastery, with the random floor beside every rate
- [ ] `docs/LIMITATIONS.md` reflects the current state and nothing in it is stale
- [ ] no claim that 4.5 or three-family mastery is met
- [ ] withdrawn results are recorded as withdrawn, not deleted

## Deferred, and not available

Do not advertise any of these.

- **Three-family 80% mastery (PLAN 4.5).** Unmet, recorded as budget-limited.
  One family (`deliver`) reaches ~0.89 structurally against a 0.01 floor.
- **Construction and recovery demonstration.** The earlier claim was withdrawn;
  the disruption it used emptied fuel inventories while production continued
  for 1,170 more ticks. Not re-demonstrated.
- **Working wheel / PyPI distribution.** See LIMITATIONS §8.
- **Replay for trained policies.** Agent runs only.
- **`factoriorl agent` subcommand.** `demo` is hardwired to `plate_line`.
- **Anthropic adapter from the CLI.** Implemented, Python-only.
- **DAgger and SMILe.** Both need a state-conditioned expert that can be queried
  from a learner-visited state; the scripted solvers are open-loop.
- **Learned skills.** The skill layer is authored, not trained.
- **Non-Windows platforms.** Untested rather than unsupported.

## Not part of this release

No git tag, no GitHub release, no uploaded assets. The bundle is built locally
and `release/` is gitignored by design — regenerate it with
`tools/package_release.py` rather than looking for it in the repository.
