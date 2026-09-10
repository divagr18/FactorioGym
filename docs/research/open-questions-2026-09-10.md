# Two questions for a researcher, 2026-09-10

Written to be handed to someone with literature access who has **not** been
working in this repository. Everything needed to answer is below; no number
here is asserted without its provenance, and where I have a hypothesis I say so
and say what would falsify it.

Context in one paragraph: FactorioRL is a reproducible RL environment and
benchmark on Factorio 2.0.60. Tasks are procedurally generated scenes; a scene
is a pure function of `(master_seed, run_id, branch, index)`, so evaluation
sets are frozen and paired. Every family has three evaluation rows — `seeds`
(training layout families), `val`, and `structures` (a held-out layout family,
the published headline). Numbers below are on a frozen 100-episode holdout,
argmax arm, primitive action space, with the random floor from the same 100
episodes where one has been measured.

---

## Question 1 — Does behaviour cloning count as "learning" for a mastery criterion?

### The situation

Our plan's item 4.5 requires **three task families at ≥0.80 success on the
held-out structural split**. Three families now clear it:

| family | structural | seeds | random floor | route |
|---|---|---|---|---|
| `deliver` | 0.89 (0.84–0.94 over 3 seeds) | 3 | 0.01–0.11 (varies by draw) | ordinary PPO |
| `restore_power` | 1.00 / 1.00 | 2 | **not yet measured** | behaviour cloning |
| `keep_line_running` | 0.87 / 0.94 | 2 | **0.00** (0/100, measured) | behaviour cloning |

Two of the three are behaviour cloning on labels produced by a **scripted
reference solver** that has full evaluator knowledge (it can read entity
positions and objective markers the policy never sees in its observation).

A separate plan item, 4.2, forbids scripted labels in ordinary PPO training.
It does not mention behaviour cloning, which did not exist in the project when
4.2 was written.

- **Strict reading:** BC trains directly on privileged scripted labels, so it
  inherits 4.2's prohibition. Only `deliver` qualifies; 4.5 is unmet with one
  family.
- **Permissive reading:** 4.2 constrains the PPO reward/label pipeline
  specifically; BC is a distinct, disclosed method. 4.5 is met with three.

### What bears on it from our own data

- **BC is not a free pass.** It fails on the families that are actually hard:
  `build_line` reaches **0.9811** expert-action accuracy and **0.00** success
  on all three rows; `repair_belt` reaches 0.13–0.17 (Q2). So admitting BC does
  not trivially manufacture passes — it appears to pass only where the horizon
  is short.
- **BC → PPO does not preserve the number.** BC-initialised PPO inherits the
  clone's competence and then loses it during optimisation. So "clone, then
  fine-tune with RL" is not currently a route that keeps the result.
- **The clone never sees privileged state.** The labels come from a solver that
  did, but the network is trained on the ordinary policy observation. What is
  privileged is the *supervision*, not the input.

### What I actually want back

1. **Norms.** In published RL benchmark papers, when a benchmark reports a
   "solved" or mastery threshold, is behaviour cloning from a scripted or
   privileged expert conventionally admitted as evidence that the *benchmark
   task* is learnable, or is it conventionally reported as a separate
   **baseline/upper-reference** row? Concrete examples either way are more
   useful than a general principle.
2. **Is the strict/permissive distinction a real one in the field**, or am I
   inventing a rule nobody else applies? Specifically: is "trained on
   privileged expert labels" treated as a different claim from "trained from
   reward"?
3. **The honest framing.** If both readings are defensible, what is the
   phrasing that a reviewer would find neither overclaiming nor falsely modest?
   I would rather publish "one family by RL, two by imitation, stated
   separately" than argue for a number.
4. **Does the `build_line` result change the answer?** 0.98 accuracy → 0.00
   success is, to me, the strongest argument that BC here measures something
   real about the task rather than about the method. Is that a recognised
   diagnostic, and does it have a name?

I am **not** asking which answer is more favourable to us. If the answer is
"4.5 is unmet with one family", that is the answer I will record.

---

## Question 2 — Why does `repair_belt` fail by every route, and what should we try next?

### The task

Restore a broken belt line so items reach an unloading chest. Primitive action
space, 300-step primitive budget, 30 ticks per decision, reference solve ≈ 22
actions. The reference solver locates the gap from the observation, walks south
of it, places a belt facing north, rotates it east, and repeats up to six times
— so it **re-scans** rather than replaying a fixed plan.

### Every route tried, and its result

| method | structural | note |
|---|---|---|
| PPO from scratch, 25k steps | **0.00** | never solved a *training* episode either |
| Behaviour cloning, 2 seeds | **0.17 / 0.13** | expert-action accuracy 0.845 / 0.897 |
| BC-initialised PPO | inherits, then **collapses** | endpoints hide it; the curves show it |
| Low learning rate (collapse fix) | prevents collapse | but transfer is seed-dependent: seed 2 → 0.88/0.98, seed 1 → 0.00/0.00 |

### The observation that reframes this, found today

`repair_belt` is the **only** family in the benchmark whose held-out split
changes the *number of subgoals*:

| family | train | val | test (published) |
|---|---|---|---|
| `deliver` | two_chest, decoy_chest | stacked_depot | screened_depot |
| `restore_power` | pole_gap, pole_gap_far | two_gaps | **gap_near_drill** (one gap) |
| `keep_line_running` | operation | operation_scattered | operation_walled |
| `build_line` | square_patch | offset_patch | narrow_patch |
| **`repair_belt`** | **gap, gap_far (one defect)** | **misrotation (one defect, different kind)** | **double_gap (two defects)** |

Every other family varies *layout* while holding the task composition fixed.
`repair_belt` alone asks a policy trained on one-defect scenes to fix two.

And the BC row breakdown matches that exactly:

| seed | seeds (1 defect, trained kind) | val (1 defect, **new kind**) | structures (**2 defects**) |
|---|---|---|---|
| 1 | 0.39 | 0.33 | **0.17** |
| 2 | 0.59 | 0.62 | **0.13** |

Changing the defect *kind* costs roughly nothing (0.39→0.33; 0.59→0.62).
Changing the defect *count* costs half to two-thirds. Note also that
`restore_power`, our BC success at 1.00, has a two-defect family (`two_gaps`)
in **val** but a one-defect family in the published **test** row — so the two
families' headline rows are not asking the same kind of generalisation at all.

**Hypothesis (untested):** `repair_belt`'s structural row is measuring
*compositional / count generalisation*, not layout generalisation, and it is
the only row in the benchmark that does. If so, "repair_belt is hard" is partly
a statement about how its splits were drawn rather than about the task.

**What would falsify it:** include `double_gap` scenes in training and see
whether the structural number moves; or evaluate on a held-out *single*-defect
layout family and see whether it lands near the `seeds` row.

### What I actually want back

1. **Is the hypothesis well-posed**, and is this a recognised failure mode with
   a literature? I am thinking of compositional generalisation and of
   systematic generalisation over subgoal count — but I do not know whether the
   right frame is compositional generalisation, curriculum design, options and
   hierarchy, or something else entirely.
2. **Is a split like this a benchmark design defect or a legitimate hard
   split?** Arguments both ways. Held-out compositional generalisation is a
   respectable thing to demand; but demanding it in exactly one of five
   families, undeclared, makes cross-family comparison meaningless. What is
   standard practice for declaring this?
3. **What should we try**, in rough order of expected value per GPU-hour? Our
   budget is small — two consumer GPUs, cells of 25k–50k steps, hours not days.
   Suggestions that require millions of steps are useful to know about but we
   cannot run them. For reference, a comparable open project (Factorion) needed
   ~2.5M timesteps on a far cheaper simulator to reach 0.78 on its tasks.
4. **The seed-dependent transfer under a low learning rate** (seed 2 →
   0.88/0.98, seed 1 → 0.00/0.00, same configuration) — is this a known
   signature of something specific, or just small-sample variance we should
   resolve by running more seeds?
5. **DAgger.** The obvious next step is DAgger-style relabelling, since our
   solver is queryable at any state and ABJKS Thm. 13.3 is exactly the
   compounding-error result our `build_line` number illustrates. Is DAgger the
   right tool for a *compositional* gap, or does it only fix the distributional
   one? My instinct is that it fixes the wrong problem here, and I would like
   that checked.

### Constraints worth knowing before answering

- Evaluation is paired and frozen; we can compare arms on identical scenes and
  we use McNemar / within-scene permutation tests.
- The reference solver is queryable at arbitrary states (it re-scans), so
  DAgger-style relabelling is mechanically available.
- Greedy evaluation can livelock: a rejected action leaves the world unchanged,
  so a deterministic policy re-selects it forever. We report both argmax and
  sampled arms for this reason.
- A measured random floor accompanies every rate, and floors move when the
  action space changes — `deliver`'s floor has been measured anywhere from 0.01
  to 0.11 depending on the scene draw.
