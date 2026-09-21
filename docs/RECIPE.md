# The short learning recipe

One command, about twenty minutes, and a number you can check against ours.

```
uv run factoriorl train --task deliver --steps 25000 --skills \
    --holdout docs/evidence/holdout_v3.json --eval-episodes 100
```

This exists because DESIGN 6.1 asks for a one-command demonstration entrypoint
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

**Most of that is evaluation, not training, and it prints nothing while it
runs.** `--eval-episodes 100` scores three rows under two arms plus a random
baseline — around 800 episodes — which dominates the wall clock: at
`--steps 2000` training finishes in about 80 seconds and the command still
takes a quarter of an hour. It is not hung. Lower `--eval-episodes` if you only
want to see the pipeline work, but note that the frozen holdout has 100
episodes and scoring fewer means you are no longer comparable to the numbers
above.

The command also prints its whole result as JSON, including per-scene rows —
hundreds of kilobytes. Redirect it to a file and read
`evaluation.structures.success_rate` (the headline), its `stochastic`
counterpart, and `random_baseline`. Note `eval_split` at the top says `val`;
the structural number the table above discusses is the `structures` row.

If you land inside 0.84–0.94 the recipe reproduced. Outside it, the most likely
causes are a different worker count (throughput, not outcome), a different
`--seed`, or dropping `--skills`, which changes the action space and therefore
the task.

## The number this must be read against

**The random floor here is volatile, so use the one your own run measured.**
Every run measures its own uniform random baseline on its own episodes and
records it at `result.json → evaluation.<row>.random_baseline`; `factoriorl
evaluate` reports one too. Measured values for `deliver` under skills span
**0.01 to 0.11** depending on the scene draw — our six cells came back 0.01,
an independent 100-episode run on a different seed came back 0.07, and
`docs/LIMITATIONS.md` records 0.11 on the training split.

An earlier version of this file stated 0.01 as a property of the row. It is
not; it was one draw, and quoting it as the floor made the headline look
better than the spread justifies. 0.89 clears even the high end, which is the
claim that survives.

That comparison is not a formality here. `docs/LIMITATIONS.md` records four
occasions in one day where a family's floor moved when the action space changed
and a headline survived that should not have. **Quote a rate only with the
floor from the same run beside it.** Both numbers are in the file.

## Learning, not mastery

R6's gate says to "report actual learning separately from mastery", so:

- **Learning:** `deliver` reaches ~0.89 on unfamiliar structures, reproducibly,
  across three seeds and two machines, against a random floor measured between
  0.01 and 0.11. It clears the high end of that spread comfortably. That is the
  reproducible learning result R6 asks for.
- **Mastery:** DESIGN 4.5 wants three families at 0.80 on the structural split
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
