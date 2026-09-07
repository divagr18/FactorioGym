# Automated Planning and Acting (Ghallab, Nau & Traverso, CUP 2016) — whole-program design review

Reviewed against `PLAN.md` (all sections), `docs/LEDGER.md`, `docs/evidence/`,
and `src/factoriorl/` as of 2026-09-07 (Phases 0–2 Accepted, Phases 3–4 Needs
correction).

**Source.** `a local copy of Automated Planning and Acting.pdf` — the
complete Cambridge University Press text, 360 PDF pages, ISBN 978-1-107-03727-4.
Every citation below is `§section, p.N` with **N a printed page number** (PDF
page = printed page + 12). All eight chapters, both appendices and the
bibliography are present; the alphabetical index is not, which affects nothing
here.

---

## 0. This supersedes a deck-based analysis, and here is what changed

The previous version of this file was written against
`D:\Books for RL\Automated Planning and Acting.pdf`, which is not the book. It
is a ~70-slide lecture deck: Chapter 3 (refinement methods, RAE — the chapter
this project needs) got **one slide**, Chapter 7 (perceiving, monitoring, goal
reasoning, learning) was **absent entirely**, and Chapter 4 was four slides.
Every citation in that document to `§3.x`, `§4.3–4.4` or `§7.x` was marked
`[verify]` and was, in fact, unverified. That analysis is superseded in whole.

Most of its *architectural* conclusions survive contact with the real text —
several are now much better supported than they were. But the complete book
falsifies, mis-attributes or materially expands the following, and the
differences change what we should build.

### Claims the complete book shows to be **wrong**

1. **"We have a single character who can only do one thing at a time, so there
   is no concurrency to exploit"** (previous do-not list, item 4). False in the
   book's own terms. §3.3.2 introduces the `interleave` operator precisely for
   the case where *multiple tasks* are in flight but *at most one primitive
   command executes at a time* (pp. 95–96, Example 3.9, Algorithm 3.5 `IRT`).
   That is exactly our situation: start a smelt, walk to the ore patch while it
   runs, come back. A single actuator does not remove task concurrency; it
   constrains it. Rejecting interleaved refinement on single-character grounds
   was a mistake, and it would have cost us the whole of Phase 7's "build while
   the factory runs" behaviour.

2. **"The book raises descriptive/operational model drift as its own stated
   weakness (§3.5)."** It does not. §3.5.3 (p. 110) raises a different and more
   interesting open problem: refinements at different levels of abstraction need
   **different state and action representations**, and "a comprehensive approach
   for this problem has yet to be developed." Chapter 8 repeats it (p. 311):
   "Relations and mappings among such heterogeneous representations should be
   addressed systematically." The drift risk is real and the engine-in-the-loop
   consistency test is still the right answer — but it answers a problem we
   identified, not one the authors named.

3. **"The book gives an extended treatment of planning with resources
   (§4.2 / the 2004 volume, Ch.15)."** The complete book contains **no resource
   algorithm at all**. §4.1 (p. 114) supplies only the vocabulary — resources are
   *borrowed* (space, tools) or *consumed* (energy), and time is the odd one out
   because it flows whether or not anyone acts. §4.3.5 (p. 130) notes the
   Meta-CSP framing can carry "resource constraints, which are quite often part
   of planning problems" — i.e. TemPlan itself does not. §4.6.2 (p. 152) hands
   the topic to the scheduling literature (Laborie & Ghallab 1995; Laborie 2003;
   Baptiste et al. 2006). The resource recommendation survives, but it is now
   honestly labelled: the book gives us a distinction and a place to hang
   constraints, not a method.

### Claims that were **unsupported** and now have real citations

4. The combinatorial-explosion argument against our flat, fully-bound
   `catalog.py` was asserted on the code alone. Now exactly citable: §2.7.1,
   p. 60 — grounding a representation "incurs a combinatorial explosion in the
   size of the representation… if a planning operator has *p* parameters and each
   parameter has *v* possible values, then there are *v^p* ground instances."

5. Landmarks-as-subgoals was cited to §2.3 (heuristic functions). The correct
   citation for the use we make of it is §2.6.2, p. 58: "The elements of a
   compound goal… could be used as subgoals… Another possibility may be to
   compute an ordered set of landmarks and choose the earliest one as a subgoal."
   The same page carries a directly analogous **video-game precedent**:
   Killzone 2 runs a SeRPE-like planner several times per second producing four-
   or five-action plans under a Run-Concurrent-Lookahead-like loop, because "the
   current state changes quickly as the game progresses."

6. "Do not build an STN/STNU solver" was our judgement. The authors make the
   argument themselves, twice: dynamic-controllability checking is polynomial
   "but the constant factor is high," so they offer pseudo-controllability as the
   compromise (§4.4.3, p. 141); and refining durations down to command
   granularity "introduces more noise in operational models" (§4.5.1,
   pp. 142–143), with dispatch propagation costing O(n³) per cycle (p. 144).

### Claims that were **materially incomplete** — mechanisms the deck hid

7. **Retry is not backtracking.** §3.2.2, p. 86: Retry "does not go back to a
   previous computational node to pick up another option among the candidates
   that were applicable when that node was first reached. It finds another method
   among those that are **now** applicable for the current state of the world σ.
   This is essential because RAE interacts with a dynamic world." Our world keeps
   ticking through a failed method; a retry that reuses a stale candidate set is
   wrong by construction. The previous analysis described the `tried` set and
   missed the property that makes it work.

8. **A method's role can be an *event*, not only a task.** §3.1.1, p. 76;
   §3.1.2, p. 77; Example 3.3 `m-emergency`, pp. 78–79. Phase 7's interventions
   are events. This mechanism was absent from the previous architecture entirely.

9. **Goals in RAE, and per-step goal monitoring.** §3.2.3, pp. 87–89. A method
   whose role is `achieve(g)` is tested against σ *before* its body starts, *at
   every progression step*, and *when it finishes* (p. 88 gives the three
   modifications to `Progress`), and the same is generalised to tasks by an
   `expected-results` field (pp. 88–89). We invented a weaker version; the
   book's is better and cheaper.

10. **REAP simulates the remaining refinement tree after every completed
    command.** §3.4.2, Algorithm 3.11 line (i), p. 104. That `Simulate(σ, T′)`
    step is what lets the actor abandon a doomed course of action *before* it
    fails, and it is the second of the planner's two stated benefits (p. 106).
    The previous architecture had a planner and a retry and no simulate.

11. **Context-dependent plans.** §5.7.2, pp. 188–190, Definition 5.20. Policies
    are memoryless; a task like "go there and come back" needs different actions
    in the *same* state depending on which subtask is active. The book explicitly
    rejects the obvious workaround — "extending the representation of a state to
    include… the history of states visited so far… might work in theory, but its
    implementation is not practical" (p. 189) — in favour of an explicit context.
    This is the sharpest available diagnosis of our own `deliver` result and it
    was completely missing before.

12. **The partial-programming framework.** §7.3.2, p. 292 — the single most
    on-point paragraph in the book for Phase 8, and it lives in the chapter the
    deck omitted. Partial programs are hierarchical nondeterministic state
    machines with "open choice steps, where the best actions remain to be
    learned"; they constrain the learnable policy class and "yield a significant
    speed-up in the number of experiments with respect to unguided reinforcement
    learning"; and "the framework seems to be well adapted to the partial
    specification of acting methods, where operational models are further
    acquired through learning." That is a description of what Phase 8 should be.

13. **Monitoring and goal reasoning** (§7.2, pp. 280–287): three levels
    (platform, action/plan, goal), the causal-structure monitor of Algorithm 7.1
    (p. 283), the invariant formulation (Σ, s₀, g, ψ) (p. 284), the statement
    that "refinement methods introduced in Chapter 3 are adequate for expressing
    monitoring activities" (p. 286), and the whole of goal reasoning
    (pp. 286–287). None of this existed in the previous analysis.

14. **The POMDP critique** (§6.8.3, p. 270): POMDP "should be called the
    invisible state MDP model because it does not consider any part of *s* as
    being observable," and an actor should instead split state variables into
    visible and hidden (the MOMDP factoring). Together with §1.3.2's
    invisible / observable(visible | hidden) trichotomy (p. 11) this describes
    our observation profile exactly, and tells us what it is still missing.

---

## 1. What this book is for us, and what it is not

**The thesis.** Planning and acting are different activities over different
models. *Descriptive* models say which state results from an action — the
actor's "know what," used to reason about which actions achieve objectives.
*Operational* models say how to perform an action — the "know how," a program
with control flow, executed in context, able to fail and try something else
(§1.3.1, pp. 8–9; restated in Ch. 8, p. 310). Descriptive models are
deliberately coarser (p. 9). Deliberation is **hierarchical** — not primarily to
reduce search cost but "to address the heterogeneous nature of the actions about
which the actor is deliberating" (§1.2.2, p. 7) — and **continual and online**:
"the actor generally deliberates at runtime about how to carry out the tasks it
is currently performing… The cost of minor mistakes and retrials are often lower
than the cost of extensive modeling, information gathering, and thorough
deliberation" (p. 7). The book's slogan for the relationship: "planning is useful
to shed light on the road ahead, not to lay an iron rail all the way to the goal"
(§1.3.3, p. 13).

**Why it fits Factorio.** Factorio is a prerequisite game, and the prerequisite
chain is *published exactly* by the recipe and technology graph — so §2.6.2's
landmark-subgoal proposal (p. 58) is free for us in a way it is not free for
anyone else. The world is dynamic: belts move and furnaces smelt during our
30-tick interval, so we violate the classical **static-environment** assumption
(§2.1.1, p. 20, assumption 1) while satisfying the determinism assumption — and
§3.1's operational models are introduced explicitly to relax exactly that
("Dynamic environment. The environment is not necessarily static. Our operational
models deal with exogenous events," p. 74). PLAN 8.1's five required skill
properties — inputs, preconditions, termination conditions, timeouts, failure
reports — are a refinement-method signature restated (§3.1.2, p. 77; §3.2.4,
p. 89).

**Where it does not fit.** The book's *solvers* mostly assume small enumerable
symbolic state spaces: Chapter 2's A*/PSP, Chapter 5's AND/OR search and search
automata, Chapter 6's value and policy iteration. None run over a 6×65×65 float
tensor in a continuous-position world with hundreds of entity prototypes. §6.7.6
(p. 268) is candid about the ceiling: value iteration is "easily implemented and
practical" only when the state space fits in memory, "typically on the order of
few mega states." **Take the architecture and the representational commitments;
leave most of the solvers.** The book is also explicit that it is theory: "most
of the technical material is theoretical. Case studies and application-oriented
work would certainly enrich the integration of planning and acting view developed
in here" (Preface, p. xiv). Two constructions we would most want are marked
unfinished by the authors: mixing temporal planning with temporal refinement
acting (§4.5.3, p. 148), and mapping between representations at different
hierarchy levels (§3.5.3, p. 110).

**The empirical hook, verified.** `docs/evidence/phase4-feasibility-sweep.json`:
`supply_furnace` reaches **1.00 train success and 0.30 held-out against a 0.50
random baseline**; `deliver` reaches **0.65 train and 0.10 held-out against a
0.30 random baseline**. On unseen layout families both learned policies are
*worse than random*. `navigate` is the only family with signal (0.95 held-out).
A flat policy over fully-bound templates is memorising layouts, and the structure
that would transfer — *go to the source, take, go to the sink, give* — is not
represented in the learned object anywhere. §5.7.2's memoryless-policy argument
(p. 189) says why `deliver` in particular should be expected to fail this way:
the correct action at a given observed position differs according to which
subtask is active, and nothing in our observation says which one that is. This is
an architecture problem, not a hyperparameter problem.

---

## 2. Target architecture for Phases 8–9

A diagram in prose. Existing modules in `code font`; proposed new ones marked
**new**. The layering is the book's (§1.2.2 Fig. 1.2, p. 6; §3.2.1 Fig. 3.1,
p. 82; §3.4.2 REAP, pp. 103–106).

**Layer 0 — the execution platform.** `mod/factoriorl/`,
`src/factoriorl/session.py`, `protocol.py`, `rcon.py`, `worker.py`, `pool.py`,
`supervisor.py`. Unchanged. These provide the book's **commands** (§3.1.1,
p. 76): the ten `ActionType` values, each with a real duration, real
preconditions, and a structured failure code from the 24-value `ErrorCode`
vocabulary. The book requires `status(command) ∈ {running, done, failed}`
maintained by the platform (p. 76); our `ActionStatus` plus the Phase 2.2
in-flight registry already is that. Nothing above this layer may bypass it
(PLAN 8.1: "skills operate through the same embodied runtime").

**Layer 1 — the command catalog.** `src/factoriorl/catalog.py`. Unchanged as an
artifact, demoted in role: it stays an ordered, content-hashed list of
fully-bound command templates that a flat policy can index, and it stops being
the *only* action interface. It becomes the leaf of the refinement tree.

**Layer 2 — the descriptive domain.** **new**
`src/factoriorl/deliberate/domain.py`. A state-variable vocabulary (§2.1.2) plus
one `CommandModel(name, params, pre, eff, cost, duration)` per `ActionType`, with
`Range(x)` extended by the book's `unknown` symbol (§3.1.1, p. 75). This promotes
`src/factoriorl/action_matrix.py` and `docs/ACTION_MATRIX.md` from
prose-plus-tests into a machine-readable object, with the existing matrix tests
*generated from it* so the three cannot drift.

**Layer 3 — abstraction and agent state.** **new**
`src/factoriorl/deliberate/state.py`: `abstract(observation, memory) -> Sigma`.
Turns the wire observation plus remembered map into the coarse state variables
`domain.py` names. This is the book's σ: "the state of the actor's knowledge,
rather than the true state of the world" (§1.3.2, p. 11). Every variable carries
one of three statuses — *visible*, *hidden* (observable but not currently
observed; our `remembered` entries with `age`), *invisible* — per §1.3.2, p. 11
and the MOMDP factoring recommended at §6.8.3, p. 270. **Signature takes
observation and memory only; never `session.truth()`.** This object is also
PLAN §2's promised "typed objects and concise summaries for language-model
agents," so the LM agent and the planner cannot diverge on what they believe.

**Layer 4 — the method library.** **new**
`src/factoriorl/deliberate/methods.py`.
`Method(role, params, pre, expected_results, invariants, body, deadline)` where
`role` is a **task, a goal `achieve(g)`, or an event** (§3.1.2, p. 77; §3.2.3,
p. 87). Bodies are generators yielding subtasks, subgoals, assignments and
commands, with the book's control constructs plus `if t1 fails then t2`
(§5.7.1, p. 188) and `{interleave: …}` (§3.3.2, p. 96). Content-hashed and
versioned exactly as `ResolvedCatalog.digest()` hashes the catalog. Two body
kinds behind one signature: *authored* (the Phase 3.2 reference solutions,
restructured) and *learned* (**new** `src/factoriorl/skills/`, a MaskablePPO
policy over a catalog subset trained against one subgoal, wrapped so RAE cannot
tell the difference). §7.3 supports the split directly: learning is "mostly
devoted to learning operational models for acting… operational models are at a
lower level, more detailed, and often more domain-specific than descriptive
models. They are more difficult to specify by hand" (p. 287).

**Layer 5 — the acting engine.** **new** `src/factoriorl/deliberate/rae.py`. An
`Agenda` of refinement stacks; each entry is `(τ, m, i, tried, T)` (§3.2.2,
p. 83; §3.4.2, p. 103). `Progress` advances one stack by one step and checks
`status(command)`; on `failed`, `Retry` pops the entry, adds `m` to `tried`, and
**recomputes** `Instances(M, τ, σ)` against the *current* σ minus `tried`
(§3.2.2, p. 86); when no candidate remains, failure propagates to the parent.
Goal-role and `expected_results` conditions are tested before the body, at every
progression step, and at completion (§3.2.3, p. 88). Invariants are tested every
cycle (§7.2.2, p. 284). Stop/suspend/resume conditions and deadlines against a
`now` state variable are supported (§3.2.4, p. 89). **This is the module Phase 7's
recovery benchmarks actually measure.**

**Layer 6 — method choice and lookahead.** **new**
`src/factoriorl/deliberate/choose.py`, one signature
`choose(task, candidates, sigma) -> Method | None`, four implementations named in
the run manifest: `StaticOrder` (Phase 5 baseline), `RefinementTree` (a bounded
forward search over `domain.py` + `methods.py` returning a refinement tree —
SeRPE's role, §3.3.1, p. 92), `LearnedChoice` (PLAN 8.3's manager), `LMChoice`
(PLAN 8.4). Plus `simulate(sigma, tree) -> bool`, REAP's check on the unexecuted
part of the tree after each completed command (§3.4.2, p. 104). PLAN 8.5's
four-way comparison then becomes four configurations of one system rather than
four systems, which is the only way its "match task and observation settings
where comparisons claim equivalence" is achievable.

**Layer 7 — the actor loop.** **new** `src/factoriorl/deliberate/actor.py`, in
one of the book's three named forms: `Refine-Lookahead`, `Refine-Lazy-Lookahead`,
`Refine-Concurrent-Lookahead` (§3.4.1, Algorithms 3.7–3.9, pp. 100–101).
**Default: lazy.** Example 3.12 (p. 102) shows why: a method written only for the
start state makes `Refine-Lookahead` return failure at step 2 even though the
remaining actions would have worked, and lazy lookahead keeps going as long as
`Simulate` predicts success. Our replanning is Python-cheap and our env step is
~24 ms-expensive, which points the same way.

**Layer 8 — monitoring.** **new** `src/factoriorl/deliberate/monitor.py`, the
three levels of §7.2 (p. 281): platform (a dead worker is already
`InfrastructureFailure`, not a task outcome — keep it that way), action/plan (the
causal-structure monitor of Algorithm 7.1, p. 283, plus invariants, p. 284), and
goal. Per §7.2.2 p. 286, monitoring lives in refinement methods and is triggered
by RAE — so `monitor.py` supplies predicates and diagnosis and `rae.py` runs them.

**Layer 9 — goal reasoning.** **new** `src/factoriorl/deliberate/goals.py`
(Phase 9). Detect a discrepancy, explain it, generate or retire a goal, resolve
conflicts among goals under consideration — the ARTUE shape (§7.2.3, p. 286) —
with commitment management, alternative assessment and plan control from the Plan
Management Agent (pp. 286–287). Phase 7.1's "goals that change without resetting
the world" is this module's minimal version.

**Where the RL environment sits.** `src/factoriorl/env.py` becomes an *adapter*,
not the top. `FactorioEnv` is how a learned method body is trained and run: one
subgoal, one budget, one catalog subset, the existing Gymnasium contract.
`encoders.py`, `vecenv.py` and `learn/` survive essentially unchanged. The one
structural change `env.py` needs is that an episode must be able to *begin*
without the world being *reset* (R14).

**One sentence.** RAE holds refinement stacks of tasks, goals and events; each is
refined by a method chosen by `choose.py` and vetted by `simulate`; a method body
is authored Python or a learned policy stepping `FactorioEnv`; both bottom out in
catalog commands sent through `session.step`; `monitor.py` watches invariants and
causal support; `Retry` re-derives candidates against the *current* σ; and the
planner reasons over an abstract state strictly coarser than what any policy sees.

---

## 3. Recommendations

### R1 — Write the descriptive model down now, as executable code

**Book.** Descriptive models ("know what") and operational models ("know how")
are different artifacts for different consumers (§1.3.1, pp. 8–9; Ch. 8, p. 310).
A planner consumes action templates with preconditions and effects (§2.1.3). An
actor needs descriptive models *of its commands* even when it does not need their
operational models, because those are embedded in the platform (§1.3.1, p. 9).

**Change.** New `src/factoriorl/deliberate/domain.py`: a state-variable
vocabulary and one `CommandModel(name, params, pre, eff, cost, duration)` per
`protocol.ActionType`. Generate the assertions in
`src/factoriorl/action_matrix.py` and the table in `docs/ACTION_MATRIX.md` from
it, so `factoriorl action-matrix --check` checks a derivation rather than a
transcription. Extend every variable's range with `unknown` as its default
(§3.1.1, p. 75) — today `encoders.encode` writes 0.0 for both "chest is empty"
and "I have never seen inside this chest," which are different facts.

**Phase.** 3 (file + generated tests). Consumed by R6, R8 and R15 in Phases 3–4;
indispensable in 8.

**Impact.** High and early. Without it, Phase 8's "hybrid planning" has no
planner input and the phase reduces to a learned option hierarchy.

**Cost/risk.** ~1 day for ten commands. Risk is drift against Lua; R12 is the
mitigation.

**Conflict.** None. PLAN 2.1 already mandates an action matrix stating
preconditions, inventory effects, reach, time behaviour, failure outcomes and
cancellation. This is that artifact made executable.

---

### R2 — Stop growing the flat catalog; add tasks, methods and commands above it

**Book.** The refinement hierarchy has three kinds of thing: **tasks** (abstract
activities), **methods** (alternative context-dependent ways to refine a task,
with a precondition and a program body), and **commands** (executable by the
platform) (§3.1.1–3.1.2, pp. 76–77). And the reason a fully-ground action set
cannot be the whole story is stated exactly: grounding "incurs a combinatorial
explosion in the size of the representation… if a planning operator has *p*
parameters and each parameter has *v* possible values, then there are *v^p*
ground instances" (§2.7.1, p. 60).

**Change.** `catalog.py` is 100% commands with every argument bound at authoring
time, and `LEDGER.md`'s Phase 3 correction is the *v^p* explosion arriving on
schedule: `place_*` was welded to one offset (`$ahead`), so a belt gap at x=7
could only be repaired from exactly x=5; the fix enumerated 4 offsets × 3 items,
plus 3 movement strides × 4 directions, plus tile-floor arithmetic in
`env._context()`. That works at three placeable items. Phase 9 has pipes,
refineries, assemblers, chemical plants, ~60 placeable prototypes and arbitrary
positions. Add `deliberate/methods.py` with the `Method` dataclass and leave
`catalog.py` alone — it is the correct leaf layer and its content-hash discipline
is one of the better things in this repo.

**Phase.** Types in 3–4; library populated in 5 and 8.

**Impact.** Turns PLAN 8.1's skill contract into a 30-line dataclass instead of a
subsystem invented under deadline.

**Cost/risk.** Low now (a dataclass and a registry). High later (retrofit).

**Conflict.** None. Worth stating in PLAN that 8.1's five skill properties *are* a
method signature, so the connection is not rediscovered.

---

### R3 — Restructure the Phase 3.2 reference solutions as refinement methods

**Book.** A method body is a program with control constructs, subtasks and
commands (§3.1.2, p. 77; Examples 3.2 and 3.4, pp. 78–80). Failure of a step is
handled by trying a *different method for the same task*, chosen against the
current state (§3.2.2, Algorithm 3.3, p. 85).

**Change.** `src/factoriorl/tasks/reference.py` already contains method bodies
with the structure thrown away: `Driver.walk_to` is a three-stride greedy axis
walk that already detects being blocked, and `solve_deliver` is
`goto(src); take; goto(dst); give` written as straight-line Python whose failure
collapses to a single `stuck_reason` at the top. Re-express as: task
`deliver(item, n, src, dst)` with method `m-deliver-direct`; task `goto(target)`
with methods `m-walk-axis` (today's walker) and, from Phase 5.1,
`m-navigate-path`. Keep `SolveTrace` — it becomes the refinement-stack trace and
the Phase 5.5 replay record.

**Phase.** 3. `LEDGER.md` downgraded Phase 3 from Accepted for the absence of
these solvers; restructuring them now is a refactor of new code at its cheapest
moment, before Phase 5.1 replaces `walk_to` and Phase 8.2 picks skill targets.

**Impact.** Closes the open Phase 3 gate; each method written is simultaneously a
Phase 8.2 skill target, a Phase 5 assisted-agent primitive, and (with R15) a
solvability witness.

**Cost/risk.** No extra cost — the work is already owed. The risk of *not* doing
it is that these get rewritten in Phase 8.

**Conflict.** None. PLAN 3.2 requires "a scripted solution"; it does not require a
straight-line script.

---

### R4 — Implement Retry against the *current* state, and make method choice the single learning seam

**Book.** §3.2.2, p. 86, is emphatic: Retry "does not go back to a previous
computational node to pick up another option among the candidates that were
applicable when that node was first reached. It finds another method among those
that are now applicable for the current state of the world σ. This is essential
because RAE interacts with a dynamic world." RAE does not retry instances already
in `tried`, because deciding that a failure cause has cleared "would require a
complex analysis" (p. 86). The choice among applicable methods is the seam where
heuristics, a planner, or learning plug in (§3.2.4, p. 90; §3.4.2, p. 103).

**Change.** Today failure is *observed and discarded*: `env.step` returns
`info["action_status"]` and `info["action_error"]` and the episode continues as if
nothing happened; `reference.Driver.do` returns `False` and unwinds to the top.
Implement `deliberate/rae.py` with the stack, the `tried` set, and candidate
re-derivation. Our 24-value `ErrorCode` enum is already the failure vocabulary a
retry policy needs: `OUT_OF_REACH` → reposition and retry; `NO_ITEMS` → refine an
acquire subtask first; `COLLISION` → a different placement method;
`TECH_NOT_SELECTABLE` (which Phase 2.1 already correctly distinguishes from
`TECH_LOCKED`, because the 2.0 tech tree is trigger-gated at its root) → refine
into a crafting task rather than waiting forever. Add `deliberate/choose.py` with
one signature and four implementations; the learned one reuses our masking
machinery verbatim, because the mask over candidate methods *is* the applicability
test over `Method.pre`, exactly as `env.action_masks()` builds a mask from
`ActionTemplate.requires` today.

**Phase.** Interface in 5; the engine in 7 — this is what makes intervention and
recovery measurable at all.

**Impact.** Highest-value single item for Phase 7. PLAN 7.3/7.4 require an agent
that observes an intervention's consequences and recovers; there is currently no
component whose job is to notice. It also collapses PLAN 8.3, 8.4 and 8.5 into
configurations of one system.

**Cost/risk.** ~300 lines plus tests. The risk is over-generalising early; build
it against exactly the six families first.

**Conflict.** None.

---

### R5 — Add event-role methods, and make Phase 7 interventions events

**Book.** A refinement method's role is a task **or an event** — "an occurrence
detected by the execution platform… corresponding to exogenous changes in the
environment to which the actor may have to react" (§3.1.1, p. 76; §3.1.2, p. 77).
Example 3.3's `m-emergency` (pp. 78–79) suspends what the robot is doing, puts
down its load, moves to the origin, and posts an `address-emergency` task, with a
computable state variable recording that it is engaged. §3.2.4 (p. 89) adds
stop/suspend/resume conditions tested at every `Progress` call, for the whole
stack and not just its top; and p. 90 suggests the ordering heuristic "react to
events first and then address new tasks, before progressing on the old ones."

**Change.** PLAN 7.3's interventions — belt defects, recipe changes, power
disconnection, supply reduction, resource depletion, competing demand — are
exogenous events by construction, and PLAN 7.3 already requires that "the agent
observes consequences through its normal interface" with intervention parameters
evaluator-only. So the agent must not *receive* the event; it must *derive* one.
Add an event vocabulary to `deliberate/monitor.py`
(`production_stopped(marker)`, `entity_missing(handle)`, `power_lost(network)`,
`input_starved(marker)`) derived from the abstract state, and let `methods.py`
carry event-role methods that RAE dispatches ahead of task progress.

**Phase.** Event vocabulary in 5 alongside monitoring; used in 7.3–7.4.

**Impact.** Makes "detect" a named, timeable step, which is what lets Phase 7.4
report *time to detect* separately from *time to repair*. PLAN §3 already forbids
merging recovery metrics; without this they are merged by default.

**Cost/risk.** Low. Risk: an event vocabulary derived from truth would be
evaluator leakage — derive it from `abstract(observation, memory)` and test that.

**Conflict.** None.

---

### R6 — Represent Factorio prerequisites as landmarks, and use them for the goal vector and the shaping

**Book.** §2.6.2, p. 58: subgoals for online planning may come from the elements
of a compound goal in a reasonable order, or from "an ordered set of landmarks"
with "the earliest one as a subgoal." Same page: the Killzone 2 precedent —
short-term objectives such as "get to shelter," a SeRPE-like planner producing
four- or five-action plans several times per second, in a game where the state
changes quickly.

**Change.** In Factorio landmarks are free and exact: the recipe and technology
graph gives them. Add `deliberate/landmarks.py::landmarks(goal, domain) ->
tuple[Predicate, ...]` in prerequisite order, over the existing closed
`PredicateKind` vocabulary. Two consumers:

1. **The goal vector.** `encoders.GOAL_FEATURES = 12`; `env._goal_vector()`
   spends slot 0 on elapsed-budget fraction and slots 1..10 on the *success*
   predicates, which for all six families are all-false until the episode is
   essentially over. Replace with landmark achievement: dense, monotone,
   non-gameable progress. This is also the cheap half of the context fix in R7.
2. **Shaping.** Generate each `TaskSpec`'s `RewardComponent`s from landmarks
   rather than hand-authoring them per family (`mine_smelt` hand-picks
   `plates_produced` and `ore_mined`; those *are* its last two landmarks).

**Phase.** 4 for both consumers. Indispensable in 9, where a rocket plan has
hundreds of landmarks and no other tractable decomposition.

**Impact.** The direct answer to credit assignment over horizons far longer than
600 decisions, and the proposal here most likely to move the Phase 4.5 numbers.

**Cost/risk.** Moderate; needs R1. **Risk: landmark shaping is not
potential-based and can be farmed.** Mitigation: express each landmark as the
existing `RewardKind.HIGH_WATER`, which pays for the increase of a maximum and
never the level — `rewards.py` already makes that structurally safe. Do not
invent a level-valued landmark reward.

**Conflict — real, with a deadline.** Changing the goal vector changes the
observation profile: bump `local-v1 → local-v2` and `EXTRACTOR_VERSION`, which
invalidates every checkpoint trained against the old one. PLAN 6.2 requires
published checkpoints with recorded hashes. **This must land before the Phase 4.5
three-seed release result, not after.** Success predicates are untouched, so
PLAN 3.5's shaping-off parity still holds and the 80% bar is unchanged.

---

### R7 — Give the policy an explicit *context*, because a memoryless policy provably cannot do `deliver`

**Book.** §5.7.2, pp. 188–190. "Policies as defined so far are stationary or
memoryless… they always perform the same action in the same state, independently
of the actions that have been previously performed" (p. 188). For a sequence
`t1; t2` "we might need to perform different actions… in the same state depending
on whether the actor is trying to achieve the first goal… or the second. As a
simple example, consider the case in which a robot has to move to a given
location and has to come back afterward" (p. 189). The book rejects
history-in-the-state — "might work in theory, but its implementation is not
practical" — and introduces an explicit **context** naming which subtask is
active; a context-dependent plan is `(C, c₀, act, ctxt)` with `act : S × C → A`
(Definition 5.20, p. 189). §6.8.4 (p. 271) gives the pragmatic version:
"extending the state representation (with variables representing the context) is
often easier than handling general nonstationary stochastic models."

**Change.** Our `deliver` policy is memoryless over a Dict observation, and
PLAN §3 rightly forbids recurrent policies before the baseline works. The book's
alternative is exactly what PLAN permits: explicit context features. Concretely,
extend `env._goal_vector()` (and `encoders.GOAL_FEATURES`) with a small one-hot
**phase** vector derived from the landmark prefix achieved so far (R6) — for
`deliver`, "not yet holding" vs "holding, en route to sink." This is a
non-recurrent, declared, versioned feature, and it is the minimal thing that
makes the task representable at all.

**Phase.** 4, in the same `local-v2` bump as R6 — do not spend two profile
versions on one change.

**Impact.** `deliver` at 0.10 held-out against a 0.30 random baseline is the
signature of an unrepresentable task, not a badly tuned one. This is the cheapest
credible fix and it is testable: if the phase feature does not move `deliver`,
the diagnosis was wrong and we learn that before Phase 8.

**Cost/risk.** Low. Risk: the phase vector must be derived from the observation
and from landmark predicates the policy can already evaluate — never from
`session.truth()`, or the result is an evaluator-information result.

**Conflict.** Observation-profile bump; same deadline as R6.

---

### R8 — Model borrowed vs consumed resources ourselves, and check resource profiles at generation time

**Book.** §4.1, p. 114: "different kinds of resources may need to be **borrowed**
(e.g., space, tools) or **consumed** (e.g., energy). Time is a resource required
by every action, but it differs from other types of resources." §4.3.5 (p. 130)
notes the CSP/Meta-CSP framing can carry resource constraints, which TemPlan
itself does not; §4.6.2 (p. 152) hands resource scheduling to the scheduling
literature. **The book gives us the distinction and a place to hang constraints,
not an algorithm — and we should say so rather than cite an algorithm that is not
there.**

**Change.** Today the only resource declarations are
`Blueprint.character_inventory` and `EntitySpec.contents`, and the engine-free
checks in `tasks/spec.py` are purely geometric (`out_of_box`,
`footprint_conflicts`). Add a consumable-resource check to `tasks.validate_all()`:
for each sampled blueprint, the reference method's declared consumable
requirement must be ≤ what the scene provides. Two live examples: `repair_belt`
hands the character 5 transport belts while the **test** family `double_gap`
needs 2 — that margin is an accident of the generator, not a checked property;
and `mine_smelt` gives coal in varying amounts across families with nothing
checking the amount suffices to smelt the required plates. Mark each resource
`borrowed` or `consumed` in `domain.py`; only consumed ones need a profile check
now.

**Phase.** 3 (the checker); 9, where scarcity is the actual task (PLAN 9.2
"shared-resource competition is represented"; 9.4 depletion).

**Impact.** Turns PLAN 3.2's "randomness alone does not establish task quality"
into a mechanical check over the 24,000 generated scenes `validate_all` already
samples, with no engine.

**Cost/risk.** Low; engine-free.

**Conflict.** None.

---

### R9 — Deadlines and durations, but no STN solver

**Book.** Two arguments, both from the authors. First, refining durations to
command granularity is actively harmful: "it may be useful to account for the
time needed to open a door, which can be assessed from statistics. However,
breaking this duration into how long it takes to reach for the handle and how
long to turn the handle introduces more noise in operational models" (§4.5.1,
pp. 142–143). Second, dispatch propagation is O(n³) per cycle (p. 144) and
dynamic-controllability checking is "demanding of computational resources… the
complexity growth is polynomial, but the constant factor is high" (§4.4.3,
p. 141), which is why the book itself offers pseudo-controllability as the
compromise. Deadline failure has two repairs: stop the delayed action and replan
from the current state, or let it finish late and repair the remainder — the
second is preferable when the violation is absorbable, "however, if this
navigation is taking longer than expected because robot r1 broke down, a better
option is to seek another robot" (§4.5.1, p. 145). Local repair means removing the
failed action and its assertions and re-running the planner on the resulting flaws
(p. 145).

**Change.** We already have contingent durations: `ActionStatus.RUNNING` plus
`session._settle()`'s poll loop covers mining, crafting, smelting and research —
durations the engine owns. Add `duration: (lb, ub)` to `CommandModel` (R1), read
from `D:/Factorio/doc-html/runtime-api.json`, which `LEDGER.md` already names as
the authority for this build. Add `deadline` to `Method`, tested by RAE every
cycle against a `now` state variable (§3.2.4, p. 89). Implement both repair forms
and record which one fired. **Do not build an STN or STNU.**

**Phase.** Durations recorded in 3 alongside R1; deadlines used in 8.1; both
repair forms in 7.4.

**Impact.** Gives PLAN 8.1's "timeouts" a definition, and gives PLAN 7.4's
recovery report a distinction it currently lacks (absorbed delay vs abandoned
method).

**Cost/risk.** Moderate. The temptation to grow this into a temporal network is
the risk; the do-not list names it.

**Conflict.** None.

---

### R10 — Monitor causal support and invariants, using the book's two mechanisms

**Book.** Monitoring is "(i) detecting discrepancies between predictions and
observations, (ii) diagnosing their possible causes, and (iii) taking first
recovery actions" (§7.2, p. 281), across three levels: platform, action/plan,
goal. Two concrete mechanisms:

- **Causal structure.** For a plan π = ⟨a₁…a_k⟩ achieving g, regress to get
  intermediate goals G = ⟨g₀…g_{k+1}⟩; `Progress-Plan` (Algorithm 7.1, p. 283)
  finds the largest *i* whose gᵢ the current state supports and performs aᵢ — so
  it *skips* steps already satisfied and *repeats* steps whose effects did not
  take. Example 7.1 (p. 283): the robot finds the door already open and skips two
  actions; later it observes an empty gripper and repeats the pickup. That is a
  Factorio repair loop in miniature.
- **Invariants.** An extended planning problem (Σ, s₀, g, ψ) where ψ must hold in
  every state along the plan; violation "means a failure of the plan. It allows
  quite early detection of infeasible goals or actions, even if the following
  actions in the plan appear to be applicable and produce their expected effects"
  (§7.2.2, p. 284). With the caution that domain-entailed invariants are largely
  syntactic and insufficient — "an actor has to monitor specific conditions…
  Such conditions cannot be deduced from the specification of (Σ, s₀, g); they
  have to be expressed specifically as monitoring rules" (p. 284).

And the integration point: "Refinement methods introduced in Chapter 3 are
adequate for expressing monitoring activities; RAE procedure can be used for
triggering observation and commands required for monitoring" (p. 286).

**Change.** `TaskSpec.failure` is already a tuple of predicates evaluated every
step by `env._failed()` — a monitor with the wrong name and one consumer (episode
termination). Add `invariants: tuple[Predicate, ...]` to `TaskSpec` and `Method`,
evaluated by RAE, whose violation aborts the **current method** rather than the
episode: `restore_power` should abandon a "wire up this pole" method the moment
the boiler stops working, not 200 decisions later at truncation. Separately,
implement `Progress-Plan` over the landmark sequence from R6 as
`monitor.progress_index(sigma, landmarks)` — a dozen lines over
`Predicate.evaluate`, giving skip-and-repeat behaviour for free.

**Phase.** Spec fields in 3; consumed in 7.3–7.4.

**Impact.** Gives Phase 7.4 a time-to-detect measurement, and gives every method a
cheap way to notice it has been invalidated.

**Cost/risk.** Low; reuses `Predicate.evaluate` unchanged.

**Conflict.** None.

---

### R11 — Keep the abstract state coarser than the observation, and classify every variable visible / hidden / invisible

**Book.** "Predicted states are in general less detailed than the observed one"
(§1.3.2, p. 10). σ "is the state of the actor's knowledge, rather than the true
state of the world," and "information in a dynamic environment is ephemeral: some
of the values in σ may be out-of-date" (p. 11). The design of an actor should
distinguish **invisible** variables (only estimable), **observable** ones, and
among the latter **visible** (currently known) from **hidden** (requiring an
observation action) (p. 11). §3.1.4 (p. 81) adds the erosion rule: "outside of
room1, the robot cannot trust such a fact indefinitely. At some point, it has to
consider that the location of that person is unknown unless it can sense it
again." §6.8.3 (p. 270) is the sharpest version: POMDP "should be called the
invisible state MDP model because it does not consider any part of *s* as being
observable"; the MOMDP factoring into visible × hidden is the recommended shape.

**Change.** We got the hard part right already and should say so: Phase 2.3's
sensor region plus `remembered` entries carrying `age`, kept in a separate array
so distant machine state can never be presented as current, is precisely the
book's visible/hidden split with erosion. Two gaps remain. (a) There is no
abstract state at all — any planner built today would have to plan over 6×65×65
float planes; add `deliberate/state.py` with `abstract(observation, memory)`,
versioned like an observation profile, **never taking `truth`**, and extend the
existing "mask uses no privileged information" test to cover it. (b) `unknown` is
not represented: `encoders.encode` writes 0.0 for a container we have never looked
inside and 0.0 for one we know to be empty. Add a per-feature known-flag or the
`unknown` sentinel of §3.1.1 (p. 75).

**Phase.** 5.4 (persistent agent memory is the same object as `memory` here);
consumed from 8.

**Impact.** The single artifact without which no planner is possible. It doubles
as PLAN §2's "typed objects and concise summaries for language-model agents."

**Cost/risk.** Moderate. Risk: too coarse makes plans unexecutable, too fine makes
search intractable. Start at marker granularity.

**Conflict.** PLAN §2 says one canonical observation supplies three views; this
adds a fourth. Argue it as a *derived* view — make it structural by having it take
the observation as input — so the canonical-observation contract holds.

---

### R12 — Generate one model from the other, and adjudicate the pair against the engine

**Book.** §3.5.3 (p. 110) names the open problem: refinements at different levels
need different state and action representations, and "such a generalization will
require formal definitions of the relationships among tasks, states and actions at
different levels, translation algorithms based on these definitions… A
comprehensive approach for this problem has yet to be developed." Chapter 8
(p. 311) repeats it as a research priority. §7.3.3 (p. 293) adds that knowledge
engineering "for the integration of operational models and acting methods remains
to be developed."

**Change.** We are unusually well placed on this, because we have a real engine
that can adjudicate. Make `deliberate/domain.py` the single source and generate
from it (a) the `action_matrix.py` assertions and the generated
`docs/ACTION_MATRIX.md`, and (b) the precondition fixtures the Lua side is checked
against. Then add a **model-consistency test** in `tests/engine/`: for N sampled
states, executing command *c* in the real engine must produce the effects
`domain.py` predicts, or the test fails. Budget it as a sampled check — at ~24 ms
per step it cannot be exhaustive.

**Phase.** 3–4 for the generation and the test; the payoff is in 8.

**Impact.** Prevents the worst Phase 8 outcome — building a planner and then
discovering its model of Factorio is wrong. It also turns an open problem the
authors name into a passing test, which is a publishable observation in its own
right.

**Cost/risk.** Engine-dependent, so `tests/engine/` and `--require-engine`, per
PLAN §4's rule that "tests skipped because Factorio was unavailable" is an
incomplete state.

**Conflict.** None.

---

### R13 — Measure dead-ends and safe-solution structure, not just success rate

**Book.** §5.2.3, pp. 162–163, Definitions 5.7–5.11: a **solution** may reach a
goal; a **safe solution** is one where from every reachable state a goal is still
reachable; **unsafe** solutions "may achieve the goal but are not guaranteed to do
so… the agent may end up at a nongoal state or end up in a 'bad cycle' where it is
not possible to go out and reach the goal"; **safe cyclic** solutions are
legitimate — a retry loop is a solution. §5.6 (p. 183) adds the acting-time
warning: "not all domains are safely explorable, and not all actions are
reversible… the actor may not easily recognize that it is trapped in a dead end,
for example, a navigation robot can enter an area where it is possible to navigate
but impossible to get out of that area."

**Change.** `learn/train.py::evaluate` reports success rate, Wilson interval, mean
reward and mean length. It cannot distinguish "ran out of budget while making
progress" from "walked into a state from which the goal is unreachable." Add a
**dead-end rate**: for each truncated episode, run the reference method (R3) from
the truncated state, evaluator-side; if it cannot finish, that episode ended in a
dead end. `deliver` at 0.10 held-out against a 0.30 random baseline has a dead-end
signature — a random agent rarely does anything irreversible and ours evidently
does.

**Phase.** 4 (the metric); a requirement in 8.

**Impact.** Directly serves PLAN §4's failed-gate procedure step 2 ("identify
whether the failure is a runtime defect, task defect, learning limitation, or
performance limitation"). Success rate alone cannot make that call; dead-end rate
can.

**Cost/risk.** Reuses R3; costs evaluation wall-clock, so run it on the evaluation
set only.

**Conflict.** **This is an additional metric, not a replacement.** The 80%
held-out bar in PLAN §3 and 4.5 stands untouched.

---

### R14 — Split `FactorioEnv.reset` into scenario-install and episode-begin

**Book.** A method's internals may be authored or learned without the acting
layer's interface changing (§3.1.1, p. 76; §3.4.2, p. 103, where REAP pushes a new
stack entry for a subtask on the live σ). For that to hold, a learned body must be
able to run from an arbitrary current state, not only a fresh one. §4.5.1 (p. 145)
makes the same demand of repair: "plan repair in case of a failure has to be
performed with respect to the current state."

**Change.** `FactorioEnv.reset()` today does five things at once: pick a layout
family, generate a blueprint, install it, reset the world, initialise
goal/accountant/counters. Split into `reset_scenario()` (the first four — today's
behaviour, what the Phase 3 504-reset leakage run tests) and
`begin_episode(goal, budget, catalog_subset)` (the fifth, **with no world
mutation**). Then a Phase 8.2 skill is trained by `begin_episode(subgoal)` on a
live world; a Phase 7.1 persistent stage transition is the same call; a Phase 8.3
manager's decision is `begin_episode` on a child.

**Phase.** **Do the split in Phase 4.** Use it in 7 and 8.

**Impact.** The highest rewrite-risk item on this list. If `reset` stays fused,
Phase 7.1 and Phase 8.2 each need surgery on `env.py`, on `vecenv.py` (whose
auto-reset in `step_wait` calls `env.reset()` directly), and on every gate that
resets — `gate_phase3`, `gate_phase4`, `tools/solvability.py`,
`tasks/reference.py`.

**Cost/risk.** ~1 day now against roughly a week later plus a checkpoint
invalidation, because the split changes when the goal vector is set and therefore
what the policy sees on step 0.

**Conflict.** None; PLAN 7.1 demands it implicitly ("stage transitions do not
refill resources or repair the factory").

---

### R15 — Prove solvability by plan existence over the descriptive model, keeping the scripted solver as the baseline

**Book.** Refinement planning constrains search to what the methods can produce,
which "can convey a substantial efficiency advantage" but fails when the library
does not cover the situation (§3.5.2, p. 109; Example 3.12, p. 102). The book's
own remedy is a **generative fallback**: replace SeRPE's failure line with
`if τ is a goal achieve(g) then return find-plan(Σ, s, g)` (§3.3.1, p. 91), and it
surveys three published ways to combine refinement with classical planning
(§3.5.2, pp. 109–110).

**Change.** Once `domain.py` exists (R1), extend `tasks.validate_all()` and
`tools/solvability.py` to run a bounded forward search over the *abstract* state of
each sampled blueprint and assert a plan exists. The abstract state space at marker
granularity is tiny; this runs over the 24,000 generated scenes `validate_all`
already samples **without launching an engine**, before any worker starts. Adopt
the book's fallback rule literally: when a task has no applicable method, fall back
to generative search over the domain rather than reporting failure.

**Phase.** 3.

**Impact.** Catches the whole `place_*`-welded-to-one-offset class of defect at
validation time rather than after a 1,470-second sweep across three workers.
`LEDGER.md` is explicit that this defect cost Phase 3 its acceptance and five of
six families their learnability.

**Cost/risk.** Small search. **Risk: a plan exists in the model but not in the
engine** — R12's consistency test is the guard, which is why R12 is not optional.

**Conflict.** None. PLAN 3.2 asks for "a solvability argument backed by generation
constraints and a reference solution"; this strengthens the argument and keeps the
reference solution as the required scripted baseline.

---

### R16 — Write methods with holes and learn the holes; discount the manager over elapsed ticks

**Book.** §7.3 opens by arguing that learning belongs mainly to *operational*
models: "operational models are at a lower level, more detailed, and often more
domain-specific than descriptive models. They are more difficult to specify by
hand" (p. 287). It then names the framework: partial programs are "hierarchical
nondeterministic finite-state machines… specified by the teacher as partial
programs, with the usual programming constructs augmented with **open choice
steps, where the best actions remain to be learned**. These specifications
constrain the class of policies that can be learned using an extended SMDP
Q-learning technique. The partial programming framework can yield a significant
speed-up in the number of experiments with respect to unguided reinforcement
learning… The framework seems to be well adapted to the partial specification of
acting methods, where operational models are further acquired through learning"
(§7.3.2, p. 292). SMDPs are named as the foundation of exactly this family of RL
approaches (§6.8.4, p. 271). §6.5.2 (p. 254) gives the complementary shape: a
method body *is* an SSP over the command space, solved by online lookahead.
Q-learning's practical caveat is stated plainly: convergence "is very slow in the
number of trials… simulated experiments can be a critical component" (p. 289).

**Change.** This is the concrete design for Phase 8. Do not train free-floating
skills and hope a manager composes them. Write `m-deliver`, `m-supply-furnace`,
`m-repair-belt` as authored method bodies whose *choice points* are open — "which
approach position," "which of these belts to replace first," "walk or wait" — and
learn only those, with MaskablePPO over the applicable-option mask, which is the
machinery we already have. Two supporting changes: (a) add `RewardKind.LANDMARK`
(monotone, paid once on first achievement) to `rewards.py`; (b) for the Phase 8.3
manager, discount over *method* transitions as
`gamma ** (elapsed_ticks / decision_ticks)` — the standard SMDP discount — rather
than once per method invocation. Record both gammas in the manifest.

**Phase.** `RewardKind.LANDMARK` in 4; partial-program method bodies in 8.2; the
SMDP discount in 8.3.

**Impact.** Our throughput is the binding constraint on the whole program (~24 ms
per env step; 41.6 steps/s single-worker, 202 at eight). A framework whose stated
benefit is a large reduction in the *number of experiments* targets the exact
resource we are short of. And without the duration-aware discount the manager
treats a 3-tick `wait` and a 3,000-tick smelt as equally costly — which is
precisely the "inconsistent time accounting" PLAN 8.3 says the manager must not be
able to exploit. This is the mechanism that makes 8.3's criterion satisfiable
rather than aspirational.

**Cost/risk.** Moderate. Risk: authored choice points that are too coarse leave
nothing to learn, and PLAN §4 forbids "replacing learning with scripted behaviour
to pass a gate." Mitigation: report, per method, the fraction of decisions taken at
open choice points, and treat a method with none as an authored baseline rather
than a learned skill.

**Conflict.** None, provided `success` predicates are untouched so PLAN 3.5's
shaping-off parity holds and the 80% bar is measured on the same thing.

---

## 4. Decisions to make now, to avoid a Phase 8 rewrite

Cheap today, expensive-to-impossible after the Phase 4.5 release result and the
Phase 6 freeze. Ordered by deadline.

1. **Fix the observation profile once: landmark goal vector plus explicit phase
   context, `local-v1 → local-v2`, `EXTRACTOR_VERSION` bump (R6, R7).**
   Deadline: **before the Phase 4.5 three-seed run.** After it the checkpoints are
   published artifacts with recorded hashes (PLAN 6.2). Do both changes in one
   version bump; two bumps is one wasted invalidation.
2. **Split `FactorioEnv.reset` into `reset_scenario` + `begin_episode` (R14).**
   Deadline: before Phase 7 starts. Everything hierarchical needs an episode that
   begins without a world reset.
3. **Add `preconditions`, `provides` and `invariants` to `TaskSpec` (R8, R10).**
   Deadline: before task versions freeze at PLAN 6.4 — adding a spec field later
   bumps every task version and breaks published comparisons. `provides ⊇` the
   next stage's `preconditions` also lets `validate_all()` prove a whole Phase 7.1
   stage chain composes without launching an engine.
4. **Add `profiles.deliberation` to the run manifest** (`actor_loop`,
   `method_library_version`, `method_library_digest`, `choice`, `planner`,
   `abstraction_version`), even if it reads `{"actor_loop": "none", "choice":
   "flat-policy"}` for all of Phase 4. Deadline: before the first published
   manifest schema. Costs nothing now and it is what makes Phase 4 results
   comparable to Phase 8 results at all. This is an upstream contract change under
   PLAN §4 and goes through the integration owner, not a unilateral edit.
5. **Fix the state-variable vocabulary and write `domain.py` (R1).** Deadline:
   before the landmark work (R6) and the solvability check (R15), both of which
   consume it. Everything downstream inherits this vocabulary; changing it later
   changes the planner, the abstraction, the landmarks and the shaping at once.
6. **Restructure the Phase 3.2 reference solutions into methods (R3).** Deadline:
   before Phase 5.1 replaces `Driver.walk_to` with real navigation and before
   Phase 8.2 picks skill targets. Cheapest the week they landed.
7. **Establish and test the rule that `abstract()`, `Method.pre`, `choose()` and
   the event vocabulary read observation + memory, never `session.truth()`
   (R5, R11).** Deadline: before the first line of `state.py`. Free now; unpicking
   a `truth` dependency later invalidates every hybrid result produced up to that
   point. Extend the existing mask test rather than writing a new one.
8. **Decide that the method library is a versioned, content-hashed artifact (R2).**
   Deadline: with the first method. Retrofitting provenance onto a library is how
   checkpoints become opaque blobs — the argument `learn/policy.py` already makes
   for `EXTRACTOR_VERSION`.
9. **Decide the manager's decision frequency: method boundaries, not every 30
   ticks.** Deadline: before Phase 8 design freezes, measured in the Phase 4.3
   profiling harness. See open question 4.

---

## 5. Do NOT do this

1. **Do not build a domain-independent HTN planner or a PDDL/ANML front end.** We
   will have on the order of twenty tasks. A hand-written method library with a
   bounded forward search over it delivers the same behaviour in a week rather
   than a quarter. **But do implement the book's one-line generative fallback**
   (§3.3.1, p. 91) — that is where the coverage failure of Example 3.12 (p. 102)
   actually bites, and it costs almost nothing.
2. **Do not implement STN/STNU dynamic-controllability checking.** The authors
   supply both reasons: the constant factor is high (§4.4.3, p. 141) and refining
   durations to command granularity "introduces more noise in operational models"
   (§4.5.1, pp. 142–143). Every duration in Factorio is a published constant in
   the engine's own prototype data. Per-method deadlines and per-command min/max
   durations (R9) capture the value.
3. **Do not replace MaskablePPO with value or policy iteration.** They assume an
   enumerable state space; §6.7.6 (p. 268) puts the practical ceiling at "a few
   mega states." Take Chapter 5's safe-solution property (R13) and Chapter 6's
   SMDP discount (R16), not the solvers.
4. **Do not adopt plan-space / partial-order planning for factory layout** (§2.5).
   The book notes PSP implementations "tend to run much more slowly than the
   fastest state-space planners" and that most researchers abandoned it for
   forward search (p. 54). Our constraints are almost entirely
   resource-prerequisite, which landmark ordering captures. **Note the corrected
   rationale:** the reason is *not* "we have one character so there is no
   concurrency" — §3.3.2's `interleave` (pp. 95–96) shows task concurrency under a
   single-command constraint is exactly our case, and R4/R5 should support it.
5. **Do not feed the planner's plan into the policy's observation as a hint.**
   That is the assisted profile leaking into the primitive profile. It breaks
   PLAN 10.3's ablation claims and makes every Phase 4 number non-comparable. The
   plan is a choice made *above* the policy, not a feature fed to it.
6. **Do not let `state.py`, `Method.pre`, `choose.py` or the event vocabulary read
   `session.truth()`.** Convenient and fatal: it turns every hybrid result into an
   evaluator-information result. `env.action_masks()` holds this line today with a
   test that reconstructs the mask from a policy-visible observation; extend that
   test, never weaken it.
7. **Do not use the hierarchy to shorten episodes in order to reach the 80% bar.**
   If a family only passes with a method library, that is a *different
   deliberation profile* and belongs in its own row of the PLAN 8.5 comparison.
   PLAN §4 forbids lowering thresholds; it equally forbids quietly changing what
   the threshold is measured on.
8. **Do not let `factoriorl.tasks` import `factoriorl.deliberate`.** The
   import-direction test in `tests/unit/test_task_isolation.py` keeps tasks free of
   worker management for a good reason. Extend the same discipline: tasks declare
   predicates; `deliberate` consumes them; never the reverse.
9. **Do not defer `domain.py` to Phase 8 on the grounds that it is a Phase 8
   artifact.** It is the input to a Phase 3 solvability check (R15) and a Phase 4
   shaping generator (R6). Deferring it defers those.
10. **Do not rewrite `catalog.py` into a parameterised action space.** The flat,
    fully-bound, content-hashed catalog is *correct as the command layer*;
    §2.7.1's *v^p* argument (p. 60) is about not making it the *only* layer, not
    about redesigning it. The fix for its combinatorics is above it.
11. **Do not model our environment as stochastic.** We violate the classical
    *static-environment* assumption, not the determinism assumption (§2.1.1,
    p. 20). §6.7.1 (p. 261) warns specifically against the reflex of modelling
    navigation as probabilistically wandering, and §6.7.5 (pp. 266–267) notes that
    estimating costs and probabilities "adds a significant burden to the modeling
    step" and may "hide qualitative preferences and constraints through arbitrary
    quantitative measures." Randomised *initial states* are not stochastic
    transitions. Treat exogenous factory dynamics with operational models and
    events (R5), not with a transition distribution.
12. **Do not build a belief-state or POMDP layer.** §6.8.3 (p. 270): the belief
    space is O(2^|S|) over an already-exponential |S|, and the POMDP observation
    model "is quite restrictive and often unrealistic." The visible/hidden
    factoring of R11 is the recommended alternative and it is what we already
    half-have.

---

## 6. Open questions

1. **Does the descriptive model need spatial reasoning?** Can `place(item,
   marker)` abstract away tile geometry, with an `m-place-adjacent` method doing
   the positioning? If yes, the planner's state space stays tiny and R15's
   plan-existence check closes. If no, it does not. And if markers are the
   abstraction, *who invents markers* during Phase 9 expansion, when there is no
   generator to declare them? §3.5.3 (p. 110) says the general answer to this class
   of question does not exist yet, so we will be inventing ours.
2. **What granularity does `connected(pole, network)` need?** Does the planner need
   the electric-network graph, and can it be derived from a 32-tile sensor with
   remembered observations, or does it need a memory structure that accumulates
   topology across an episode? §7.1.3's anchoring problem (pp. 278–279) is the
   right frame: an electric network is an object with no perceptual footprint of
   its own.
3. **We have no observation action.** §6.8.3 (p. 270) is explicit that "one does
   not observe at every step all observable variables… These observation actions
   have a cost and need to be planned for," and §7.1.2 (p. 278) treats planning to
   perceive as a first-class problem. Today our only sensing action is *moving*.
   Does Phase 9's expansion need an explicit `scan`/`survey` task with a cost, and
   if so does it belong in the command catalog or only in the method library?
4. **Deliberation cost against a 24 ms env step.** At 41.6 env steps/s
   single-worker, a 20 ms lookahead per decision halves throughput, and throughput
   is the binding constraint on the entire program. Does the manager decide only at
   method boundaries (cheap, coarse) or every 30 ticks (expensive, reactive)? The
   Killzone 2 precedent (§2.6.2, p. 58) ran a planner several times per second on
   four-action plans — but that game's actor was not also the training bottleneck.
   Measure in the Phase 4.3 harness before Phase 8 design is fixed.
5. **Is a learned skill a method *body* or a method *choice function*?** The book
   permits both (§6.5.1–6.5.2, pp. 253–254), PLAN 8.2 vs 8.3 implies both, and
   §7.3.2's partial-program framing (p. 292) suggests a third answer — the same
   method containing both authored structure and learned choice points. R16 picks
   the third. If that is wrong we find out in 8.2, and the fallback is 8.2 as
   written. But PLAN should say which it means.
6. **Where does the LM agent sit?** At the choice function (R4), or above RAE as a
   goal generator (§7.2.3, pp. 286–287)? Different failure modes, different token
   costs, different claims about what was learned. PLAN 8.4 does not distinguish
   them and should.
7. **Are the six introductory families tasks, or methods?** `navigate` and
   `deliver` look like methods for larger tasks; `mine_smelt` and `restore_power`
   look like tasks. If four of six are really methods, PLAN 4.5's acceptance bar is
   measuring *skill quality*, not *task quality* — defensible, arguably the point,
   but it should be stated rather than discovered in Phase 8.
8. **Does `deliver` fail from dead ends, from exploration, or from missing
   context?** R7 says missing context (§5.7.2, p. 189); R13 gives the metric that
   distinguishes it from dead ends. Run R13 before committing to R7's profile bump
   — a two-hour experiment that decides a checkpoint-invalidating change.
9. **Can R12's model-consistency test run at a useful rate?** It is
   engine-dependent and the engine is the bottleneck. How many sampled
   state/command pairs per command give useful coverage inside a gate's time
   budget?
10. **What is the recovery *target* under Phase 7.4's paired evaluation?** §4.5.1
    (p. 145) offers two repairs — abandon and replan, or absorb the delay and
    repair the remainder — and says which is better depends on *why* the deadline
    was missed. Our intervention taxonomy does not currently distinguish absorbable
    from structural faults. It should, before the paired-recovery metric is frozen.

---

## 7. One gap in PLAN.md worth raising with the integration owner

PLAN Phase 8 is titled "Learned skills and hybrid planning," and four of its five
sub-items (8.1 skill contract, 8.2 train skills, 8.3 manager, 8.4 LM selection,
8.5 comparison) are about skills. **There is no line item anywhere in PLAN for the
descriptive domain model, the abstract state, or the planner** — the three things
without which the word "planning" in the phase title has no referent. As written,
Phase 8 delivers a learned option hierarchy and calls it hybrid planning.

The ordering PLAN gives matches the book's layering exactly and needs no change.
The recommendation is to add **8.0 — descriptive domain model, abstract state, and
refinement engine** as a prerequisite, with its artifacts (R1, R4, R11) actually
built in Phases 3–5 where they already pay for themselves, so that 8.0 is an
acceptance step rather than a build step.
