# Documents

There are a lot of these. Most were written to answer a specific question at a
specific time, and they are kept because a decision without its reason is
folklore. This page says which ones are current and which are history, so you
do not have to guess.

## Start here

| | |
|---|---|
| [LIMITATIONS.md](LIMITATIONS.md) | what this environment cannot do, and what each number is worth. Read before trusting any result. |
| [ACTION_MATRIX.md](ACTION_MATRIX.md) | the action contract, generated from source by `factoriorl action-matrix` |
| [RECIPE.md](RECIPE.md) | the short learning recipe, its expected numbers, and the random floor to read them against |
| [AUTHORING_TASKS.md](AUTHORING_TASKS.md) | how to add a task family |

## Current state

| | |
|---|---|
| [LEDGER.md](LEDGER.md) | gate status and the evidence each claim rests on |
| [CURRENT_STATUS.md](CURRENT_STATUS.md) | where the project is now |
| [DEVELOPMENT_REDIRECTION.md](DEVELOPMENT_REDIRECTION.md) | why the plan changed after Phase 4, and the R0-R6 work that followed |
| [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md) | what is deferred and deliberately unavailable |

## The language-model agent path

Optional, and the only part that can cost money.

| | |
|---|---|
| [AGENT.md](AGENT.md) | connecting a provider, the spend cap, and the replay viewer |
| [AGENTIC_RL_PLAN.md](AGENTIC_RL_PLAN.md) | the plan for the agent track |

## Evidence

`evidence/` holds the artifacts results cite: frozen holdouts, engine suite
output, reward audits, benchmark records. `evidence/holdout_v3.json` is the
frozen held-out evaluation set — see `tools/freeze_holdout.py` for what makes
it checkable rather than merely named.

## History

Dated files are session handovers and superseded roadmaps, kept for their
reasoning rather than their currency: `HANDOFF-*.md`,
`AGENTIC_ROADMAP-2026-09-10.md`, and `../HANDOFF.md`. Nothing current depends
on them being read.
