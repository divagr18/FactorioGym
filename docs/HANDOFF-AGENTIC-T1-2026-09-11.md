# Agentic RL T1 handoff

Date: 2026-09-11. T1 is implemented and its task gate passed.

## What landed

`construct_smelting_line` is a new versioned production task. Every scene starts
with iron ore, two burner drills, two stone furnaces, and coal; it has no
pre-built machine, iron plate, or machine inventory. Its train layouts are open
and offset patches. Its frozen structural holdout is an obstructed, narrow ore
patch.

The agent gets at most 600 decisions and 18,000 construction ticks. Calling
`finish` is represented by `FactorioEnv.run_verification()`: it locks policy
actions, samples the next 3,600 ticks through normal worker stepping, and scores
only the delta in Factorio's `machine_produced["iron-plate"]` counter. Success
requires ten or more plates. Ordinary production, handcrafting, transfers,
placements, and output before the window cannot count.

The terminal reward is `min(machine_output / 10, 1)` for this task's binary
threshold: the successful reference receives 1.0 and a failed verifier receives
0. This intentionally differs from the compact-policy reward audit's dense
shaping rule. The audit marks it as an `action-locked` terminal verifier rather
than inventing a shaping signal that would change the stated objective.

## Acceptance evidence

- `uv run factoriorl tasks validate`: all declarations valid.
- `uv run python tools/freeze_holdout.py --verify`: the new task has 100 frozen
  structural holdout scenes. Existing task entry hashes remain unchanged.
- `uv run python tools/solvability.py --tasks construct_smelting_line`: reference
  success 1.00 on train and test; random floor 0.00 on both. The tool wrote
  [phase3-solvability-partial.json](evidence/phase3-solvability-partial.json).
- Focused task, holdout, sustained-output, and reward tests: 136 passed.

`tests/unit/test_construct_smelting_line.py` specifically checks the output
threshold, ignores a large ordinary-production counter in favour of the
machine-only delta, reserves the final minute, and rejects a second verifier
call.

## Next implementation boundary

T2 needs to expose `run_verification()` as the agent-facing `finish` tool in the
Windows-worker/WSL learner bridge, with its result as a terminal rollout record.
Do not let a model call it twice, send an action during its window, or treat a
normal `wait` action as verification. SFT, teacher calls, and GRPO remain
unstarted.
