# The short learning recipe

One command, about twenty minutes, and a number you can check against ours.

```
uv run factoriorl train --task deliver --steps 25000 --skills \
    --holdout docs/evidence/holdout_v3.json --eval-episodes 100
```

This exists because PLAN 6.1 asks for a one-command demonstration entrypoint
and R6's gate asks that a stranger be able to run it. It is deliberately the
*smallest* thing that learns rather than the best thing we have.

## What you should get

Six cells were run on 2026-09-09 across two machines — three seeds under each
of two reward settings, all scored on the same 100 frozen `holdout_v3`
episodes.

| | structural holdout (argmax) |
|---|---|
| seed 1 | 0.84 |
| seed 2 | 0.91 / 0.94 |
| seed 3 | 0.92 |
| **range** | **0.84 – 0.94** |

Training success reaches 1.00. Wall time was 1,000–1,400 s per cell on an RTX
3050 laptop and an RTX 4060 desktop.

If you land inside 0.84–0.94 the recipe reproduced. Outside it, the most likely
causes are a different worker count (throughput, not outcome), a different
`--seed`, or dropping `--skills`, which changes the action space and therefore
the task.

## The number this must be read against

**The random floor on this row is 0.01.** Every run measures its own uniform
random baseline on the same episodes and records it in
`result.json → evaluation.structures.random_baseline`; ours came back 0.01
(1 of 100). So 0.89 against 0.01 is a real result.

That comparison is not a formality here. `docs/LIMITATIONS.md` records four
occasions in one day where a family's floor moved when the action space changed
and a headline survived that should not have — `deliver`'s own primitive floor
is 0.00 and its skills floor has been measured as high as 0.11 on the training
split. **Quote a rate from this recipe only with the floor from the same run
beside it.** The number is in the file; there is no excuse.

## Learning, not mastery

R6's gate says to "report actual learning separately from mastery", so:

- **Learning:** `deliver` reaches ~0.89 on unfamiliar structures against a 0.01
  floor, reproducibly, across three seeds and two machines. That is the
  reproducible learning result R6 asks for.
- **Mastery:** PLAN 4.5 wants three families at 0.80 on the structural split
  with at least one production or repair family. That is **unmet**, and
  recorded as budget-limited rather than as a finding: the published cells are
  0.2–0.3% of the steps a defensible negative would need for the long-horizon
  families.

`deliver` is a logistics task. The repair and production families are much
harder and none of them clears the bar; `repair_belt` scored **0.00** on every
row at 25k steps. Do not read this recipe as evidence about those.

## Two things worth knowing before you interpret episode lengths

**`--skills` changes what a "step" is.** `max_decision_steps` is compared
against the environment's primitive step count, so it is a primitive budget
despite the name. One skill decision expands into several primitive steps — a
solved `deliver` episode is about 4 decisions and 18 primitive steps. Both
numbers are in `per_scene` (`steps` and `primitive_steps`); their ratio is what
lets you compare a skills run with a primitive one.

**Greedy evaluation can livelock.** A rejected action leaves the world
unchanged, so a deterministic policy re-selects it forever and burns the whole
budget. Every row therefore reports both arms, and you should quote both.
On this recipe the sampled arm scores ~0.98 where argmax scores ~0.89, and part
of that gap is scenes where argmax cannot act at all rather than scenes it
scores worse on. `docs/LIMITATIONS.md` has the traced instance.

## Then what

```
uv run factoriorl evaluate --checkpoint <the run id it printed> --episodes 100
uv run python tools/diagnose_run.py <run_id>
```

`evaluate` re-scores the checkpoint you just trained, which is also how you
score one somebody hands you. `diagnose_run.py` reads the per-scene rows and
reports where attempts ended rather than only how many succeeded, plus a paired
comparison if you give it two runs.
