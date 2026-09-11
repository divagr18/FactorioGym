# Handoff: the open-world objective, and an agent that stops being ambitious

**Scope.** One issue only: what `open_factory` tells the agent to do, and why two
agents with very different amounts of information both narrowed rather than
expanded. Interface and runtime defects are out of scope here -- they are covered
in `docs/evidence/a5-first-run.md` and `a5-second-run.md`, and the ones that
mattered are already fixed and committed.

**Status.** Diagnosed, evidence gathered, **nothing changed**. The objective is
exactly as it was. This document exists so the decision can be made by someone
else with the evidence in front of them.

---

## 1. What the agent is told, verbatim

`src/factoriorl/worlds.py:112`, `OPEN_FACTORY_OBJECTIVE`. Rendered into the
cached static prefix once per run, never per turn.

```
YOUR OBJECTIVE
Build a productive factory from the items you start with.

  - Automate gathering and processing rather than doing them by hand.
    Handcrafting and hand-mining are how you bootstrap, not how you
    produce.
  - Expand production that is useful to you: more of what you are short
    of, and the machines that make it.
  - As resources and time allow, work toward electricity, assembly and
    research.

Your progress is measured -- what you gather, craft, place and research,
and what your machines produce without your help. There is no target
number and no hidden win condition. Keep working until the controller
stops the run; it will not stop because you did something wrong.

How to build is yours to decide. Nothing here tells you where to put a
machine or in what order to build, because nobody has decided that for
you.
```

There is no reward function. `OpenWorldEnv._succeeded()` returns `False`
unconditionally; runs end on the wall clock, the spend cap, or provider failure,
never on achievement. Nothing here can be "reward hacked" -- there is no reward.
What follows is a local optimum under an underspecified goal.

## 2. The evidence

Two runs, same seed (20260910), same model (`deepseek-flash`), same settings,
**identical objective text**. Run 8 had every interface fix from `18fa77f` and
`e72c539`; run 7 had none of them.

| | run 7 | run 8 |
|---|---|---|
| run id | `agent-20260911T003442-d24b2ccb` | `agent-20260911T121214-a5b1aa8a` |
| decisions | 253 | 112 |
| machine-made iron plate | 216 | **563** |
| machine-made coal | **0** (54 hand-mined) | **269** |
| drills / furnaces placed | 1 / 1 | **4 / 2** |
| copper plate | 16 | **0** |
| electronic circuits | 10 | **0** |
| lab | 1, built and never powered | 0 |
| belts + inserters crafted | 40 | 15 |
| belts + inserters **placed** | **0** | **0** |
| ended holding | -- | **iron-plate x485** |
| outcome | stopped by hand | ran the full 1800s, $0.65 of $2.00 |

The interface fixes worked, and worked well. Run 8 automated its own coal supply
-- a thing run 7 never attempted -- built four drills, walked to the stone patch
run 7 had declared unreachable, and reasoned explicitly about footprints and
output tiles (*"Place the stone furnace so its footprint covers the drill's
output tile"*, *"rotate it so its output no longer lands on the other drill"*).

**And it got narrower.** It produced 2.6x the iron and reached strictly less of
the game: no copper, no circuits, no research, and 485 plates in its pocket that
nothing consumed. Both runs crafted belts and inserters and placed none.

What the two runs support together: **information was the binding constraint on
execution, and is not the binding constraint on ambition.** An earlier reading of
run 7 -- "the agent wasn't unambitious, it was blind" -- was right about the
drill on the patch edge and the stone it never walked to, and wrong as a general
explanation. Run 8 could see, and narrowed anyway.

## 3. Three specific defects in the wording

### 3.1 The measurement sentence credits hand-work

> Your progress is measured -- **what you gather, craft, place and research**,
> and what your machines produce without your help.

Hand-gathering and hand-crafting are listed **first**, as progress. That
contradicts the bullet three lines above it (*"Handcrafting and hand-mining are
how you bootstrap, not how you produce"*), and both runs resolved the
contradiction the same way: run 7 hand-mined 54 coal and hand-crafted 76 gears
and 40 belts.

The root mistake is a category error in the prompt, not in the measurement. A4.2
publishes `mined_by_hand`, `handcrafted` and `machine_produced` separately **so a
reader can check the subtraction** -- it is a descriptive instrument. The
objective describes it to the agent as a scoreboard. Those are different things.

### 3.2 "As resources and time allow" is a self-assessed opt-out

It makes electricity, assembly and research *conditional*, and the agent judges
the condition. Run 7's plan at decision 216 is that clause being exercised
verbatim:

> **"Practical maximum without stone."**

It did not ignore the instruction. It satisfied it by deciding resources did not
allow.

### 3.3 "Expand production that is useful to you" is circular

Useful *for what*? With no terminal goal, "useful" resolves to "keeps the current
loop running": short of coal, so haul coal, to smelt iron, to make gears, which
nothing consumes. The bullet reads as a direction and is not one. It licenses
exactly the behaviour both runs converged on.

## 4. Hard constraints on any rewrite

Read these before touching the text. They are enforced by tests, not convention.

- **`tests/unit/test_persistent_agency.py::test_the_objective_prescribes_nothing_a3_1_forbids`**
  asserts the objective contains **no digit at all**
  (`assert not [c for c in objective if c.isdigit()]`), and no phrase from
  `FORBIDDEN_IN_PROMPT` or `FORBIDDEN_IN_OBJECTIVE`
  (`src/factoriorl/agent/loop.py:192` and `:212`). Those ban build ordering
  (`"first build"`, `"start by"`, `"in this order"`), thresholds (`"at least"`,
  `"success is"`, `"you win"`) and solver geometry (`"drop_position"`,
  `"two tiles south"`).
- **`::test_the_open_world_states_an_objective_and_a_benchmark_task_does_not`**
  pins three substrings: `"Build a productive factory"`, `"progress is
  measured"`, `"until the controller"`.
- **`::test_the_objective_is_invariant_so_it_belongs_in_the_cached_prefix`** --
  it must stay constant within a run. It lives in the cached prefix; a per-turn
  objective is paid for on every turn (measured: hoisting 1,440 invariant
  characters cut a 300-turn run's context at turn 900 from 680k to 400k tokens).

A3.1 forbids prescribing *how*. It does not forbid stating *what* more plainly.
Every fix in section 5 is a removal of contradiction or opt-out, not an addition
of prescription -- that line is the point and is worth holding.

## 5. Recommendation

Fix the three defects by **removing** them, not by adding build guidance:

1. Rewrite the measurement sentence so it does not present hand-work as
   progress. It currently contradicts the prompt's own bullet.
2. **Delete "as resources and time allow."** It is a licence to stop, not a
   priority ordering.
3. Give "expand" a referent that is not itself. Directional and
   threshold-free, e.g. *a factory is more productive when more of what it makes
   is made by machines rather than by your hands, and when it can make things it
   could not make before.*
4. Optionally, state that the technology tree is the map of what becomes
   possible. That is a fact about the game, not a build order, and the knowledge
   block already ships all 196 technologies.

**Costs to accept first.** `OPEN_FACTORY_VERSION` goes `0.3.0` -> `0.4.0` (the
constant exists for exactly this), making runs 7 and 8 non-comparable to
everything after. The cached static prefix changes, so the next run's first turn
takes a cache miss on it -- negligible, well under a cent at DeepSeek miss rates.

**What would falsify this.** If a run on the reworded objective still ends
holding hundreds of plates with belts uncrafted in its pocket, the objective was
not the binding constraint either. The next suspect would be that nothing in the
world *wants* iron: with no consumer downstream, "produce more" has nothing
pulling it toward breadth.

## 6. An adjacent finding, deliberately not folded in

Both runs crafted belts and inserters and placed none. Run 8 crafted five burner
inserters with the stated reason *"so I can move coal from a belt/chest into the
furnaces"* and never placed one. It also solved a drill jam by putting a chest on
the drill's output tile, which **disconnected** that drill from its furnace: the
chest ended holding `iron-ore x157` while both furnaces reported `no
ingredients`.

This is not obviously a prompt defect and is not claimed as one. Placing an
inserter correctly requires reasoning about *two* tiles -- the one it takes from
and the one it puts into -- which may simply be harder than anything else in the
catalog. Worth its own investigation; do not merge it into the objective
rewrite.

## 7. Where to look

- objective: `src/factoriorl/worlds.py:112` (`OPEN_FACTORY_OBJECTIVE`), version
  at `:47`
- constraints: `src/factoriorl/agent/loop.py:192` (`FORBIDDEN_IN_PROMPT`), `:212`
  (`FORBIDDEN_IN_OBJECTIVE`)
- tests: `tests/unit/test_persistent_agency.py:60-98`
- reports: `docs/evidence/a5-first-run.md`, `docs/evidence/a5-second-run.md`
- raw runs: `runtime/runs/agent-20260911T003442-d24b2ccb`,
  `runtime/runs/agent-20260911T121214-a5b1aa8a` (gitignored). `decisions.jsonl`
  carries every plan the agent stated, and the plan trace is the most
  informative artifact of both runs.
- replay: `uv run factoriorl replay <run-id>`
