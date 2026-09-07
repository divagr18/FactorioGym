# Artificial Intelligence and Games — what it gives FactorioRL, and what it does not

Source: Georgios N. Yannakakis and Julian Togelius, *Artificial Intelligence and Games*, 2nd edition
(Springer, 2024). Citations below are to the **printed page numbers** of that edition. In the PDF at
`D:\Books for RL\Artifical Intelligence and Games.pdf` the PDF page is the printed page **+ 26**
(printed 231 = PDF 257).

## Framing

The user's rating of this book — "keep optional, supplementary" — is right, but it undersells one
part. Our six task generators *are* procedural content generation, our entire generalization claim
rests on how we split generated content, and this book is the standard reference for exactly that
question. Chapters 7–9 (PCG) are worth reading properly. The rest is mostly not for us.

The single most useful idea in the book for this project is **expressive range analysis**
(§9.7.3, pp. 300–301, after Smith & Whitehead 2010): measure your generator's output in a small
space of designer-chosen descriptors, plot it, and look at the shape of the cloud. Applied to
`src/factoriorl/tasks/families/`, this is not an abstraction — it is a tool we can write in an
afternoon that would have caught the "held-outs are farther away" defect automatically, and that
catches a **larger defect that is still present today** (Section "Generator diagnostics", finding
G1). Second most useful: the book's accumulated experience with game-AI competitions
(§6.3.3–6.3.5, §13.2.1, §14.1) is a direct catalogue of the mistakes Phase 6 and Phase 11 are set
up to make.

### Where this book has nothing for us — stated plainly

- **Factory / automation / logistics games do not appear.** The word "Factorio" occurs zero times.
  Chapter 6's genre survey covers board, card, arcade, racing, FPS, Minecraft, dungeon crawlers,
  puzzle, strategy, text and serious games. There is no long-horizon construction genre. **Phases
  7 and 9 get essentially nothing from the survey chapters.** The closest analogue is Minecraft
  (§6.4.3, pp. 210–212), and the useful part of that is Voyager's automated-curriculum-plus-skill-
  library structure (p. 211), which is a Phase 8 reference, not a Phase 7 one — and the book is
  candid that Voyager works partly because GPT-4 already knows Minecraft's recipes, a crutch we do
  not have for Factorio 2.0 (p. 212).
- **Nothing on fault injection, intervention, or recovery benchmarks.** Phase 7.3/7.4 is
  unsupported by this literature. The only transferable machinery is *constraint satisfaction over
  generated content* (§8.3–8.4), which we can repurpose (recommendation 9).
- **Part IV (Chapters 10–12, player modeling and affect) does not apply.** It is about modeling
  human players' experience from annotations, physiology and gaze. We have no human players, no
  experience to model, and the flagship claim is competence, not enjoyment. Experience-driven PCG
  (§7.2.3, pp. 243–246) and EDRL (§8.6, p. 278) are structurally interesting but their driving
  signal is a *player experience model*; substituting "policy success rate" changes the method
  enough that the book's guidance stops applying. Chapter 15 (ethics of affective games, loot-box
  difficulty manipulation, p. 427) is irrelevant here.
- **Mixed-initiative / AI-assisted design tools** (§7.2.3 pp. 240–243, Fig. 7.3–7.4) are about
  human designers in the loop. We have one owner and many coding agents; there is no designer-facing
  tool in the plan.
- **No statistical methodology for RL evaluation.** No confidence intervals, no seed-variance
  discussion, no discussion of how many evaluation episodes are enough. Our Wilson-interval +
  per-seed reporting in `src/factoriorl/learn/train.py` is already ahead of anything the book
  offers on this axis. Do not expect help here.
- **No code and no implementation guidance** — the authors say so themselves (Preface, p. xi).
  Chapters 1–3 are a refresher on AI methods; skip them.

---

## Generator diagnostics

This section is the deliverable. It specifies a tool that can be written directly from this
description, and reports what a manual pass over the six generators already shows.

### Why: the book's argument

Two passages do the work.

1. **Functional content breaks discontinuously.** §7.2.1, p. 235: "a game level is more like
   program code than like natural language text… one or a few items added or removed… might make it
   impossible to complete the level, however hard you try." *Necessary* content must always be
   correct; *optional* content need not be. Every blueprint we generate is necessary content. This
   is why `validate_all()` and `tools/solvability.py` exist and why they are the right shape.
2. **A generator is judged by the shape of its output space, not by its samples.** §9.7.3, p. 300:
   "It is one thing if our generator is only able to generate levels with very specific
   characteristics within a narrow space of its expressive range and another if our level generator
   is able to express a broad spectrum of level properties, yielding uniformly covered expressive
   ranges. Such information can be directly visible on the illustrated heatmaps or scatter plots."
   Fig. 9.9 (p. 301) is the canonical picture: the Ropossum generator's output plotted as a scatter
   in (linearity, density).

The second passage is the one we have never acted on. We have validated individual *episodes*
(in-box, non-overlapping, reachable, success-false-at-reset) and never characterized the
*distribution* a layout family produces. §8.2, p. 262 sharpens why that matters: a random number
seed is "a representation with no locality whatsoever", and larger representation spaces "increase
the expressive range of the generator." A layout family whose only free variable is one integer has
an expressive range of a handful of scenes, and evaluating 100 episodes on it is 100 draws from a
handful of scenes.

### What to measure: two descriptor groups, kept apart

The rebuild we already did ("holdouts must be structural at matched difficulty") is exactly the
claim that difficulty and structure are *separate* axes. Make that mechanical by computing two
disjoint descriptor groups from each `Blueprint`, engine-free.

Build once per blueprint: an occupancy grid over the scene box (`Blueprint.radius`), blocked tiles
taken from the rounded positions of every `EntitySpec` whose name is in the collision set (reuse
whatever set the Phase-3 reachability check uses; walls, chests, furnaces, drills, solar panels,
poles and inserters block, belts do not), plus a BFS from `character_position` over the free tiles.

**Group D — difficulty descriptors. These must OVERLAP between train and test.**

| Descriptor | Definition | Notes |
|---|---|---|
| `d_path` | Sum of BFS geodesic distances along the required waypoint chain (character → each marker the success predicate references, in solution order) | The honest "how far must the character walk" number. For `navigate` it is start→goal; for `deliver`, start→src→dst. |
| `d_steps_ref` | `SolveTrace.steps` from `factoriorl.tasks.reference.solve` | Already computed by `tools/solvability.py`; it is the best difficulty measure we own because it counts *catalog actions*, not tiles. Requires the engine, so make it optional. |
| `d_slack` | `1 − d_steps_ref / max_decision_steps` | Budget headroom. A holdout with less slack is harder, full stop. |
| `d_items` | Total item count the task requires moved or produced (`at_least` over success predicates) | Constant per family today; will not be once Phase 7 lands. |

**Group S — structure descriptors. At least one must SEPARATE train and test.**

| Descriptor | Definition | What it distinguishes |
|---|---|---|
| `s_detour` | `d_path / euclidean(character, goal)` | 1.0 = open field; >1 = the route must go around something. This is *the* axis that separates "farther away" from "differently shaped". `navigate/open` ≈ 1.0, `wall_corridor` slightly above, `wall_arc` clearly above. |
| `s_turns` | Number of direction changes along the BFS shortest path | Corner count, the same descriptor the book's Fig. 9.10 maze map uses ("average corridor length and number of corners", p. 301). |
| `s_occupancy` | Fraction of blocked tiles inside the bounding box of {character} ∪ {markers} | Clutter density, independent of distance. |
| `s_distractors` | Count of entities whose `marker` is referenced by *no* success predicate | `decoy_chest`, `dst_neighbour`, `furnace_b`, `fuel_chest`. Currently the only structural difference several "holdout" families have. |
| `s_fanout` | Max number of interactable entities within reach radius (10 tiles) of any waypoint | Captures `two_furnaces` / `shared_input`-style ambiguity. |

Add Phase-7 descriptors later in the same shape: `s_chain_depth` (position of the faulted machine
in the production graph), `s_blast_radius` (machines downstream of the fault), `d_repair_cost`
(items the scripted repair consumes).

### Diagnostic 1 — the expressive-range plot

`tools/expressive_range.py`. For each registered task, for each layout family, draw N = 512 seeds
via `SeedPlan(master, run_id).generator_rng(Branch.GENERATOR, i)` (this is what `Branch.GENERATOR`
is for and it is currently unused), call `task.generate(family, rng)`, compute the descriptors, and
emit:

- a scatter plot per task with **x = a D descriptor** (`d_path`), **y = an S descriptor**
  (`s_detour`), one colour per layout family, train families as filled markers and test as hollow;
- the same as a 10×10 binned heatmap per family (10 bins per measure is the book's own convention
  for MAP-Elites archives, §3.3.2, p. 89);
- a JSON companion under `docs/evidence/expressive-range.json`.

**How to read it.** The train and test clouds should be **vertically displaced and horizontally
overlapping**. Horizontal displacement (different `d_path`) means the holdout is harder or easier,
not different — that is the failure mode we already hit once. A test cloud sitting *inside* the
train cloud on both axes means the holdout is not a holdout at all. A cloud collapsed to a point
means the family has no expressive range.

### Diagnostic 2 — the split-validity check (numeric, gateable)

Turn the plot into two numbers and put them in `factoriorl.tasks.validate_all()` and the Phase-3
gate, replacing the name-only check at `src/factoriorl/gate_phase3.py:200–204` (which asserts only
that the test family's *string* is not in the train set).

For each descriptor, normalize to [0,1] over the pooled train+test samples, bin into 20 bins, and
compute the **overlapping coefficient** `OVL = Σ_bins min(p_train, p_test)` (1.0 = identical
distributions, 0.0 = disjoint).

- **Difficulty parity:** for every descriptor in group D, require `OVL ≥ 0.80`.
- **Structural separation:** for at least one descriptor in group S, require `OVL ≤ 0.50`.
- Cross-check with a 1-nearest-neighbour classifier over group S only: leave-one-out balanced
  accuracy separating train from test samples must be `≥ 0.75`. If a classifier cannot tell them
  apart from structure alone, they are not structurally different.

Both are milliseconds and engine-free, so they belong in `validate_all()` next to
`footprint_conflicts()`, not in a separate offline tool.

### Diagnostic 3 — expressive-range size (coverage, uniformity, cardinality)

Per family, over the same 512 seeds:

- `coverage` = occupied cells / 100 in the 10×10 (D, S) archive;
- `uniformity` = normalized Shannon entropy of the cell histogram (1.0 = flat, the book's
  "uniformly covered expressive range", p. 300);
- `distinct` = number of unique canonicalized blueprints (sorted tuple of
  `(name, rounded position, direction, marker)` plus character inventory) per 512 seeds.

`distinct` is the one that bites. **Require `distinct ≥ 100` for any family used in a 100-episode
evaluation**, and fail validation below it. Evaluating 100 episodes against a family that can only
produce 3 scenes reports a number about 3 scenes.

### What this measurement already shows, by hand

I computed the descriptors mentally over the six generators. Five of six families have a defective
split. This is the highest-leverage finding in this review.

**G1 — three test families have no generator branch at all, so they are generatively identical to a
train family.**

- `deliver.py`: `generate()` branches on `decoy_chest` and `stacked_depot` only. The test family
  `screened_depot` falls through to the base case, which is exactly the train family `two_chest`.
  The docstring at line 42 says "the destination sits behind a wall segment" — **no wall is ever
  placed.** Worse, the function still contains `rng.uniform(26.0, 34.0) if family.name != "far_depot"`,
  a live branch for a family that no longer exists in `FAMILIES` — the fossil of the
  farther-is-harder design we rebuilt away from.
- `supply_furnace.py`: the val family `far_ore` has no branch (identical to train `linear_row`).
  The test family `shared_input` differs only by adding a coal chest — which is inert, because the
  furnace is already created with `contents={"coal": 10}`. The test family is train plus a decoration.
- `mine_smelt.py`: the test family `screened_patch` adds *more ore tiles* next to the patch. Nothing
  screens anything; there is no obstacle entity. It is arguably *easier* than train, not different.

The Phase-3 gate passes all of these because it compares family names.

**G2 — two train families are exact duplicates of another train family.**
`repair_belt.py` branches on `misrotation` and `double_gap`; `gap_far` falls through to `gap`.
`restore_power.py` branches on `two_gaps` and `gap_near_drill`; `pole_gap_far` falls through to
`pole_gap`. The name says "far"; the generator has no distance parameter.

**G3 — two generators have an expressive range in the single digits.**
`restore_power.generate()` fixes `chain = 5` and every entity coordinate; the only random variable
is `gap = rng.randint(1, 3)`. Train content space = **3 distinct scenes**. The test family fixes
`gap = chain - 1`, so the held-out content space = **1 scene**. A 100-episode held-out evaluation
of `restore_power` evaluates one scene one hundred times.
`repair_belt.generate()` varies `length ∈ [6,9]` and `gap ∈ [2, length−2]`: 18 distinct train
scenes, with everything else deterministic (`LINE_Y = 0.0`, `start_x = 4`).
By contrast `navigate`, `deliver`, `mine_smelt` and `supply_furnace` sample a continuous bearing and
distance, so their `distinct` counts are effectively 512/512.

**G4 — the only family with four genuinely distinct layout families is `navigate`, and it is the
only family clearing the acceptance bar.** `open` / `wall_corridor` / `pillar_field` / `wall_arc`
each place different obstacle topology at the same distance range (12–30 tiles). It is also the
easiest family, so this is correlation and not proof — but it means navigate's 0.95 held-out is the
only number in the project that currently measures structural transfer at all.

**G5 — the deliver result cannot mean what it appears to mean.** `deliver` reports 0.65 train /
0.10 held-out. Under G1 the test family's content distribution is a *subset of the train family's
support* (train is a 50/50 mixture of `two_chest` and `decoy_chest`; test is plain `two_chest`).
A structural-transfer explanation is therefore unavailable. The gap is a seed-branch effect, an
evaluation-protocol effect, or small-n variance — and finding out which is more valuable than any
further training run on `deliver`.

---

## Benchmark design for Phases 6 and 11

The book's competition experience (§6.3.3–6.3.5, §13.2.1 p. 395, §13.2.2 p. 399, §14.1) is a list
of failure modes that a benchmark author walks into. Five apply directly.

**1. Plan for the benchmark being beaten; the response is a new generator, not a new threshold.**
§13.2.1, p. 395: "When algorithms are developed that 'beat' existing benchmarks, new benchmarks
need to be developed. For example, the success of an early planning agent in the first Mario AI
competition necessitated that the software be augmented with a better level generator for the next
competition." The 2009 Mario winner was pure A* in state space; the 2010 organizers did not make
levels longer, they changed the *generator* to emit **dead ends requiring backtracking**, which
defeats best-first search structurally (§6.3.3, pp. 194–195). That is precisely our
structural-holdout-at-matched-difficulty principle, validated by a benchmark that ran for years.
The Simulated Car Racing competition instead had to change games entirely (p. 395).

**2. Test on content the submitter has never seen.** GVGAI's founding motive (§6.3.5, p. 199) was
that "submissions were getting increasingly game-specific by incorporating more and more domain
knowledge", so submissions are "tested on unseen games… impossible to tailor the submissions to."
Simulated Car Racing "used freshly generated tracks, unseen by the participants, to test submitted
controllers" (p. 399).

**3. A fixed, hard-to-extend content set invites tuning to individual instances.** §6.3.4, p. 198
on ALE: "there is only a limited number of Atari 2600 games, and it is highly non-trivial to create
new games for this platform. This makes it possible to tune architectures and even agents to
individual games." Our defence is that content is generated and the generator is versioned — but
only if the generator has expressive range (finding G3).

**4. Without a common protocol, results are simply not comparable.** §6.3.6, p. 202 on Street
Fighter: "there is no commonly used benchmark and the papers use different experimental conditions,
so it is very hard to compare the performance of different approaches… it is unclear exactly how
the game state was represented." PLAN 11.3's manifest requirement is the right instinct; it needs
to extend to the *content distribution*, not just the code version.

**5. Do not report one number.** GVGAI found "clear non-transitivity in the rankings… families of
algorithms perform better on families of games" (§6.3.5, p. 200). A single FactorioRL score would
hide exactly the effect that makes the benchmark interesting. PLAN section 3 already forbids an
undocumented composite; keep it forbidden. §14.1, p. 408 gives the reporting frame instead: three
generalities — **game generality**, **task generality**, **user/designer generality** — which map
for us onto family generality (one policy across families), profile generality (same policy under
different observation/assistance profiles), and structural generality (unseen layout families).

**6. One asset the book says almost nobody has: our forward model.** §14.2, p. 414: "the real
roadblock when it comes to implementing many types of AI methods for game-playing is that existing
game engines do not provide for forward models… There is a real opportunity for building a new type
of game engine, which obeys to a Model-View-Controller framework so that the game state is clearly
separable from the presentation layer." Our exact stepping (`auto_pause=false` plus explicit
`tick_paused` toggling, per `docs/LEDGER.md`) with a headless server and an evaluator-only truth
channel *is* that separation. Advertise it in the Phase 6 release notes as a first-class benchmark
property: FactorioRL supports planning agents, not only learners. That is a differentiator against
every environment the book surveys except GVGAI's planning track.

---

## Recommendations

Each has an idea + citation, a specific repo change, a phase, expected impact, and cost/risk.

### 1. Fix the five broken layout families before running another training experiment

**Idea.** Necessary content must always be correct (§7.2.1, p. 235); the whole point of a holdout is
that the content differs. **Repo change.** In `src/factoriorl/tasks/families/`: add real generator
branches for `deliver.screened_depot` (place the wall segment the docstring promises, on the
geodesic between `src` and `dst`, keeping `dst_distance` in the training range), `mine_smelt.screened_patch`
(place obstacle entities, do not add ore), `supply_furnace.shared_input` (make the second consumer
actually compete for ore rather than adding an inert coal chest), `repair_belt.gap_far`,
`restore_power.pole_gap_far`. Delete the dead `far_depot` branch in `deliver.py`.
**Phase.** 3 (blocking Phase 4.5). **Impact.** Highest of anything here. Today, four of six
reported held-out numbers measure seed transfer while claiming structural transfer.
**Cost/risk.** A day of work. Risk: it invalidates the current `deliver` and `supply_furnace`
numbers — which is the point; they were never structure results. Bump `TaskSpec.version` on each
touched family so old runs stay comparable.

### 2. Write `tools/expressive_range.py`

**Idea.** Expressivity metrics and expressive-range visualization (§9.7.3, pp. 300–301, Fig. 9.9;
descriptor-space plotting after Smith & Whitehead). **Repo change.** New module
`src/factoriorl/tasks/descriptors.py` (occupancy grid, BFS, the D and S descriptor tables above,
importable without the `rl` extra so it stays on the `tasks` side of the import-direction test) and
`tools/expressive_range.py` writing `docs/evidence/expressive-range.json` plus per-task scatter/heatmap
PNGs. Draw seeds via the currently-unused `Branch.GENERATOR`. **Phase.** 3, maintained through 11.
**Impact.** Turns "is this holdout structural?" from a code-review judgement into a plot.
**Cost/risk.** Low; engine-free, no new dependency beyond matplotlib in the dev extra.

### 3. Replace the name-only holdout check with the split-validity check

**Idea.** Same as (2), made gateable. **Repo change.** Add `split_validity()` to
`factoriorl.tasks.validate_all()` computing the OVL parity/separation thresholds and the 1-NN
balanced accuracy from Diagnostic 2; call it from `src/factoriorl/gate_phase3.py`, replacing lines
200–204. **Phase.** 3. **Impact.** The regression test for the mistake we already made once and the
one described in finding G1. **Cost/risk.** Low. Risk: threshold bikeshedding — fix OVL ≥ 0.80 /
≤ 0.50 now, record them in the gate output, and revise with evidence rather than by feel.

### 4. Add an expressive-range floor to task validation

**Idea.** "A generator only able to generate levels with very specific characteristics within a
narrow space of its expressive range" is a defective generator (§9.7.3, p. 300); a bare seed has no
locality and buys no variety (§8.2, p. 262). **Repo change.** In `validate_all()`, add
`distinct_blueprints` over 512 sampled seeds per family and fail below 100 for any family in a
`test` or `train` split. Then fix `restore_power.py` (randomize `chain`, pole spacing, chain
bearing, generator placement — currently every coordinate is a literal) and `repair_belt.py`
(randomize `LINE_Y`, `start_x`, line bearing). **Phase.** 3. **Impact.** `restore_power`'s held-out
evaluation currently has a sample space of one. **Cost/risk.** Low. Risk: widening these generators
may expose engine placement failures the current fixed layouts never hit — that is what
`tools/solvability.py` is for, run it after.

### 5. Make difficulty an explicit, controllable generator parameter

**Idea.** Controllability: "another way is to use a set of parameters that control the content
generation along a number of dimensions… A vector of content features was used in [896] to generate
levels for Infinite Mario Bros that satisfy a set of feature specifications" (§7.2.2, p. 237).
**Repo change.** Extend `LayoutFamily` in `src/factoriorl/tasks/spec.py` with a
`difficulty: tuple[float, float] = (0.0, 1.0)` range, and change the generator signature to
`generate(family, rng, difficulty: float)`, with each family mapping difficulty monotonically onto
its D descriptors (distance range, obstacle count, chain length) and **never** onto its S descriptors.
`FactorioEnv.reset()` samples difficulty from the family's range. **Phase.** 3, prerequisite for 6.
**Impact.** Makes recommendations 3 and 6 possible; today difficulty is entangled with the family
name (`gap_far`, `far_ore`), which is exactly why those families rotted into no-ops.
**Cost/risk.** Medium — touches every generator and the task contract. Do it together with (1).

### 6. Difficulty-ramped training curriculum, with a frozen evaluation distribution

**Idea.** §5.3.1, p. 176: "One method for combating overfitting in RL is to ensure that the
training scenario varies as much as possible… through procedural content generation, where a new
level is generated for each episode. If the PCG method includes a way of controlling the difficulty
of the generated levels, you can use this to create a form of curriculum, where harder levels are
generated as the policy gets better at solving existing levels; this can further increase
generalization." Corroborated at §6.3.5, p. 199 (Justesen et al. on GVGAI: PCG levels with
difficulty rising in tandem with policy strength "significantly increased policy generalization",
though "the models never learned to be fully general even within a single game").
**Repo change.** In `src/factoriorl/learn/train.py`, a `DifficultyScheduler` that widens the
sampled difficulty window on `Branch.TRAIN` when the trailing-100-episode training success rate
exceeds a threshold. `Branch.EVAL` sampling **must** stay at the family's full declared range,
unconditionally, and the manifest must record the schedule. **Phase.** 4 (after 5).
**Impact.** The plausible route for `deliver`, `mine_smelt` and `repair_belt` to clear 80% without
touching the bar. **Cost/risk.** Medium. Three named failure modes to guard: (a) the schedule
leaking into evaluation — prevented by the branch separation, and worth a unit test alongside the
existing disjoint-seed test; (b) the policy overfitting to the schedule rather than the task — the
book is blunt that "RL always tends to overfit unless it is explicitly incentivized (or forced) not
to" (p. 200), so evaluate on the full range throughout, not only at the end; (c) ramping anything in
group S, which confounds curriculum with structural transfer. **The 80% threshold on the frozen
full-range test distribution does not move.**

### 7. Fix `supply_furnace` by raising its floor, not by relabelling it

**Idea.** A task an untrained agent solves 30–67% of the time is not measuring competence — the
book's parallel is §6.3.6, p. 202: "learning to beat the built-in NPC opponents is often not
particularly hard, as they typically have known vulnerabilities that can be exploited even by quite
simple policies. Simple policies that exploit particular weaknesses… will typically not generalize."
And §9.7.2, p. 299: functional properties should be expressed as constraints the generator must
satisfy. **Repo change.** In `supply_furnace.py`: remove the free `contents={"coal": 10}` so fuelling
is a required step, raise `TARGET_COUNT`, and make `shared_input` a real competing-demand family
(a second furnace drawing from the same chest). Then add a **random-baseline ceiling constraint** to
`validate_all()` fed from `tools/solvability.py`: a family whose recorded `random_rate` exceeds
0.10 is a task defect, reported the same way an unsolvable family is.
**Phase.** 3/4. **Impact.** Converts a family that cannot be a benchmark into one that can.
**Cost/risk.** Low, but it will drop `supply_furnace`'s measured learnability; that is a correct
measurement replacing an incorrect one.

### 8. Phase 7 persistent factories: solver-based generation, not more hand-written branches

**Idea.** §8.3, pp. 265–266: solver-based PCG searches spaces of *partial* solutions, "iteratively
pruning the search space: eliminating parts of the search space repeatedly until only viable
solutions are left"; Tanagra generates platform levels from solvability constraints (max jump
length, distance from jumps to enemies) "very fast" and provably; SAT-backed solvers admit worst-case
bounds. The caveat matters too: "It is generally complicated to include simulation-based tests or
any other call to the game engine inside a constraint satisfaction program."
**Repo change.** New `src/factoriorl/tasks/gen/solver.py` expressing factory-layout constraints as a
CP-SAT model (OR-Tools): non-overlapping footprints, belt path connectivity source→sink, inserter
adjacency and orientation, electric-pole supply-area coverage of every powered machine, drill
footprint over resource tiles, character-reachable free tile adjacent to every machine that must be
interacted with. Generators call the solver and emit a `Blueprint`; the existing structural
validators in `spec.py` become the post-check, and `tools/solvability.py` stays the *engine-side*
simulation test the solver cannot contain. **Phase.** 7. **Impact.** Hand-written per-family branches
already rotted at six families (findings G1/G2); they will not survive persistent factories.
**Cost/risk.** Medium-high: a new dependency and a real modelling effort, and the constraint model
becomes a thing that can itself be wrong. Mitigation: the solver's guarantee is structural only;
solvability remains the scripted-reference-in-engine check it is today.

### 9. Phase 7 faults: generate-and-test with a hard feasible-repair constraint

**Idea.** Constructive vs generate-and-test (§7.2.2, p. 237) and hard vs soft constraints in
quality-diversity generation (§8.4, pp. 267–268: "constraints can be either 'hard' — e.g., can a
player complete this game level or not — or 'soft'"). The feasible-infeasible two-population
algorithm (§3.3.2, p. 87) exists precisely because penalizing infeasible candidates over-penalizes
them; keep a second population that optimizes toward the feasibility boundary.
**Repo change.** Model an intervention as a typed edit over a persistent factory blueprint
(remove entity / rotate entity / drain fuel / change recipe / deplete resource / add competing
demand), and gate every candidate on a **hard constraint**: a scripted *repair* reference in
`src/factoriorl/tasks/reference.py` must complete it from the post-fault state within the declared
budget. PLAN 7.3 already requires "fault generation preserves a documented feasible recovery path"
— this makes it the generator's acceptance test rather than a review comment.
**Phase.** 7. **Impact.** Directly satisfies the 7.3 acceptance criterion; without it, a failed
recovery run is uninterpretable. **Cost/risk.** Medium; the repair reference is real work, and it
must live in `reference.py` so the import-direction test keeps it out of PPO training.

### 10. Use MAP-Elites to *construct* the Phase 7 holdout split

**Idea.** §8.4, pp. 266–268: quality-diversity partitions a behaviour space into a grid and keeps
the best individual per cell; "behavior partitioning allows designers to control and illuminate the
design space", and "PCGQD offers online expressivity analysis as a byproduct" — so illumination and
generation are the same run. §9.7.3, p. 301, Fig. 9.10 shows exactly this used as a generator
diagnostic (a maze archive over corridor length × corner count). §3.3.2, p. 89 gives the archive
mechanics and the ten-bins-per-measure convention.
**Repo change.** `src/factoriorl/tasks/gen/qd.py`: archive dimensions = (`s_chain_depth`,
`s_blast_radius`), fitness = repair-feasibility margin (budget slack of the scripted repair);
mutation = the typed fault edits from (9). Then **partition archive cells, not seeds**, into
train/val/test, and publish the cell assignment. This is the scalable replacement for hand-named
layout families, and it satisfies PLAN 7.5's "test factories are not selected using agent
performance" by construction — the archive is built before any agent runs.
**Phase.** 7. **Impact.** Gives Phase 7 a holdout split with a defensible construction story instead
of six hand-written names. **Cost/risk.** High compute, but generation is offline (§7.2.3, p. 239,
temporality) so it is amortized across the whole phase. Risk: choosing bad behaviour characterizations
— the book is explicit that these are the designer's ad-hoc choice (p. 89), so plot the archive
(Diagnostic 1) before committing to a split.

### 11. Build an evaluator-side planning oracle on the exact-stepping forward model

**Idea.** We have what §14.2, p. 414 says game engines lack. §5.2.1, p. 169 draws the distinction we
need: searching in *physical space* only moves the agent, "the rest of the game environment is not
updated", whereas searching in *state space* "steps the game engine forward between each simulated
action" and is "vastly more useful as it can be used for problem-solving beyond mere navigation".
§5.2.3, pp. 172–173 gives two techniques for our situation: rolling-horizon evolution (optimize a
10–20 action sequence, execute the first action, re-plan) and the portfolio approach of Churchill and
Buro — evolve *which script each unit runs* rather than raw actions — which is the right shape for
large catalogs. §5.2.2, p. 171 gives the necessary caveat: in fine-grained time, rollouts rarely
reach a terminal state, so cap rollout length and use a state evaluation function, and prune the
action set to search deeper.
**Repo change.** `tools/oracle.py`, evaluator-side only: rolling-horizon evolution over the
task's `catalog_subset`, using `advance` for lookahead, with the success predicate plus a
distance-to-marker heuristic as the state evaluator. Report `oracle_steps` per episode and add
`steps_used / oracle_steps` as a normalized efficiency metric alongside the success rate.
**Phase.** 8 for the full version; a cheap A*-in-state-space variant for `navigate` and `deliver` is
affordable now. **Impact.** Converts `d_path` in Diagnostic 1 from a proxy into a measurement, and
gives evaluation a competence axis beyond binary success. **Cost/risk.** High. Branching lookahead
needs the validated checkpoints of Phase 10.1 to be efficient; at 75 env steps/s a naive
implementation is too slow. **Strictly evaluator-side** — PLAN 4.2 forbids scripted solution labels
reaching PPO, and this must not become a teacher.

### 12. Procedural personas as the Phase 7.4 evaluation instrument

**Idea.** §9.7.3, p. 302: "An interesting approach to AI-based testing is the use of procedural
personas… data-driven inferred models of dissimilar play styles." §6.5, p. 213 gives the concrete
recipe from MiniDungeons: personas are "formalized… by attaching differing utilities to measures such
as the number of treasures collected, the number of monsters killed or the number of turns taken to
reach the exit."
**Repo change.** PLAN 7.4 already demands five separate recovery measurements (time to restore,
production lost, repair resources, damage to other production, final output). Define three
evaluator-side reference policies weighting those differently — *fast-repair*, *resource-conserving*,
*throughput-preserving* — and report the paired recovery benchmark against all three. A fault whose
outcome is identical under all three personas is not discriminating between recovery strategies and
should be dropped from the split.
**Phase.** 7. **Impact.** Makes the five recovery numbers interpretable relative to a reference
frontier rather than as five raw values. **Cost/risk.** Low-medium; the personas are utility
weightings over an existing scripted repair, not new agents.

### 13. Version the generator separately from the task, and pre-declare the saturation response

**Idea.** §13.2.1, p. 395 (Mario needed a new level generator once A* won; Simulated Car Racing
changed games) and §14.1.2, p. 410: the Mario AI Framework's focus on one level-generation problem
"has also led to a clear overfitting of methods."
**Repo change.** Add `generator_version` to `TaskSpec` in `spec.py`, distinct from the existing
`version`, and record it in `manifest.py`. Write the saturation policy into `docs/` for Phase 11.3:
when a family is saturated, the response is a new `generator_version` with new layout families at
matched difficulty (verified by recommendation 3), published as a new benchmark version, with prior
results retained under their generator version. **Phase.** 6, enforced through 11.
**Impact.** Makes PLAN 11.3's "benchmark version changes do not overwrite old comparisons"
mechanical. **Cost/risk.** Low.

### 14. Record the training content distribution in the submission manifest

**Idea.** §6.3.6, p. 202 (incomparable Street Fighter results from undeclared conditions) and
§4.2.1.1, p. 144 (the fairness of a benchmark interface is "a surprisingly difficult endeavor").
GVGAI's answer was to withhold the test content entirely (§6.3.5, p. 199); Simulated Car Racing's
was freshly generated unseen tracks (§13.2.2, p. 399).
**Repo change.** Extend `src/factoriorl/manifest.py` to record, per run, the set of layout families
sampled, the generator version, the difficulty schedule if any, and the **descriptor histogram** of
the training distribution from recommendation 2. The verification lane then has a mechanical leakage
check: if a submission's training descriptor histogram overlaps the test region, the split was
compromised regardless of what the config claims. **Phase.** 6 (manifest), 11 (submission rules).
**Impact.** This is the leakage check PLAN's verification lane is supposed to run and currently
cannot. **Cost/risk.** Low.

### 15. Report a three-cell generality profile per checkpoint, never a single score

**Idea.** §14.1, p. 408: game generality, task generality, user/designer generality; §6.3.5, p. 200:
rankings across content families are non-transitive, so an aggregate hides the effect.
**Repo change.** In `src/factoriorl/learn/train.py`'s evaluation output and the Phase 6 release
artifacts, report a 3×N table per checkpoint: **structural generality** (unseen layout family, the
current headline), **family generality** (the same checkpoint evaluated on the other five task
families, expected near zero today with one-model-per-family — report it anyway, it is the Phase 8
baseline), **profile generality** (same checkpoint under a second observation or assistance profile).
Each cell with its own Wilson interval. **Phase.** 6, extended in 8 and 10.3.
**Impact.** Gives Phase 8's hierarchy comparison a pre-existing baseline instead of inventing one
retroactively. **Cost/risk.** Low; it is more evaluation runs, not new machinery.

---

## Do NOT do this

1. **Do not adopt experience-driven PCG or dynamic difficulty adjustment as described.**
   §7.2.3, pp. 243–246 and §8.6, p. 278 (EDRL). The driving signal is a *player experience model*
   built from human annotations, and EDRL's reward "quantifies Koster's notion of fun" by moderating
   content diversity. We have no players, no fun, and no annotations. Recommendation 6 takes the one
   transferable sentence (§5.3.1, p. 176) and leaves the framework.
2. **Do not use PCGML — GANs, LLM level generators, n-gram/Markov models — for Factorio scenes.**
   §8.5, pp. 269–275. Their premise is a corpus of existing hand-made levels (MarioGPT fine-tunes on
   Super Mario Bros levels; the Video Game Level Corpus exists for this). We have no corpus, and
   §7.2.1, p. 235 explains why generative models fail hardest on exactly our content type: functional
   content has no graceful degradation.
3. **Do not use Wave Function Collapse for factory layouts.** §8.1.4, pp. 257–261. WFC enforces
   *local tile adjacency* constraints. Belt throughput, inserter orientation and power-network
   coverage are global properties. WFC would produce factories that look plausible and do not run.
4. **Do not make PCGRL the Phase 7 generator.** §8.6, pp. 275–278. It requires dense rewards over
   edit sequences ("dense rewards seem to be necessary for PCGRL to work well", p. 278) and would
   add a second reinforcement-learning system to debug beside the one that has not yet passed its
   own gate. Recommendations 8–10 give the same generality without a second RL loop.
5. **Do not make a holdout harder in order to make it "more of a test."** That mistake is already in
   this repo's history and its fossil is still in `deliver.py` (`far_depot`). Diagnostic 2 exists to
   stop it recurring.
6. **Do not pursue human-likeness, believability, or Turing-test tracks.** §4.2.1.3, pp. 145–146;
   §14.1.1, p. 409. These need human play traces and human judges. We have neither, and the flagship
   claim in PLAN section 1 is competence, not believability.
7. **Do not report a single FactorioRL score.** §6.3.5, p. 200 (non-transitivity across content
   families) and PLAN section 3. Recommendation 15 is the alternative.
8. **Do not build a fixed, hand-curated scenario set as the benchmark's content.** §6.3.4, p. 198
   (ALE). Ship generators and splits.
9. **Do not let the reference solver, the oracle of recommendation 11, or any persona from
   recommendation 12 near PPO training.** PLAN section 4 forbids replacing learning with scripted
   behaviour, and `src/factoriorl/tasks/reference.py` already documents the import-direction test
   that enforces it. Keep the new tools on the same side of that line.

---

## Open questions

1. **What actually causes `deliver`'s 0.65 train / 0.10 held-out gap?** Under finding G1 the test
   family's content distribution is inside the train family's support, so a structural-transfer
   explanation is unavailable. Candidates: the `decoy_chest` mixture in train inflating the train
   number, a `Branch.EVAL` sampling artifact, or n too small. Answer this before running any further
   `deliver` experiment.
2. **Which difficulty descriptor actually predicts our success rate?** Recommendation 6's curriculum
   ramps an axis we have not yet shown is the axis that matters. Fit `d_path`, `d_steps_ref` and
   `d_slack` against per-episode outcomes from an existing checkpoint first.
3. **Is a secret split compatible with PLAN 6.2 and 11.3?** GVGAI and Simulated Car Racing withheld
   test content; our plan requires publishing everything. A middle path — publish the generator code
   and the test split's *descriptor statistics* (auditable, difficulty-matched) but withhold the
   family module until a version freeze — needs a decision, not a default.
4. **Can a rolling-horizon oracle run at our throughput?** At ~75 env steps/s, a 15-action horizon
   with a population of 20 costs ~300 engine steps per decision. Does this need Phase 10.1 checkpoint
   branching to be viable at all, or is a cheap A*-in-state-space version enough for the introductory
   families?
5. **Does the Phase 7 QD archive need a surrogate model to be affordable?** The book's precedent is
   Karavolos et al.'s CNN predicting FPS level–weapon balance from simulated bot matches (§9.6,
   pp. 296–298 and Fig. 9.8) and surrogate-assisted MAP-Elites for Hearthstone decks (§6.2.2,
   p. 189) — both
   introduced precisely because per-candidate simulation was too expensive. If the repair-feasibility
   fitness requires an engine run per candidate, we will need the same trick.
6. **Is "the scripted reference solution finished" still the right feasibility oracle for persistent
   factories?** For the six introductory families it works because there is one solution shape. A
   persistent factory has many, and a single scripted solution failing would no longer prove
   infeasibility.
7. **Does one-model-per-family survive Phase 8?** §14.1, p. 408 argues single-game single-task
   results need validation in a general context, and §6.3.5, p. 200 reports that RL on GVGAI overfits
   "not only to a single game, but even a single level of a single game." Our family-generality cell
   in recommendation 15 will probably read ~0.00 at Phase 6. Decide now whether that is a reported
   limitation or a Phase 8 acceptance criterion.
