# Automated Planning and Acting (Ghallab, Nau, Traverso) — whole-program design review

Reviewed against `PLAN.md` (all sections), `docs/LEDGER.md`, and
`src/factoriorl/` as of 2026-09-07 (Phases 0–2 Accepted, 3–4 Needs correction).

---

## Citation provenance — read this first

The file at `D:\Books for RL\Automated Planning and Acting.pdf` is **not** the
Cambridge University Press book. It is a 70-page Beamer lecture deck (produced
2018-11-26, 870 KB, 604 lines of extractable text) that *summarises* the book.
Its coverage is badly skewed for our purposes:

| Book chapter | Deck coverage |
|---|---|
| 1 Introduction | slides 1–9, adequate |
| 2 Deterministic models | slides 10–26, thorough |
| **3 Refinement methods** | **slide 34 only — six bullet points** |
| 4 Temporal models | slides 35–39, thin |
| 5 Nondeterministic models | slides 40–50, adequate |
| 6 Probabilistic models | slides 51–59, adequate |
| **7 Other deliberation functions** | **absent** |

Chapter 3 is the chapter this project needs, and the local artifact contains one
slide of it. Chapter 7 (perceiving, monitoring, goal reasoning, learning) is
absent entirely.

So citations below are given as **`GNT §x.y`** referring to the CUP text, with
**`[deck s.N]`** appended wherever the local artifact corroborates the claim.
Claims cited to `GNT §3.x`, `§4.3–4.4` and `§7.x` are from the book itself and
have **no local corroborating page**; treat them as verifiable-but-unverified
here and check them against a copy of the book before they are quoted in a
publication. Every recommendation that depends on such a citation is marked
`[verify]`. Nothing in the concrete-change column depends on the citation being
exact — the changes stand on the code.

**Action item independent of this review:** acquire the actual book (or the
authors' companion papers on RAE — Ghallab/Nau/Traverso 2016 Ch.3, and
Patra et al. on RAE+UPOM) before Phase 8 design is frozen.

---

## Framing: what this book argues, and where it applies to us

The book's central stance is that **planning and acting are different activities
over different models, and conflating them is the standard mistake**. An actor
needs *operational* models — how to perform an activity, as a program with
control flow, in the current context, with the ability to fail and try
something else — and a planner needs *descriptive* models — what state results
from an action, abstract enough to search over (`GNT §1.3.1` [deck s.8]).
Descriptive states are deliberately *coarser* than what the actor observes
(`GNT §1.3.2` [deck s.9]). Deliberation is **hierarchical** (some things the
actor wants to do do not map to a platform command) and **continual online**
(the actor refines, monitors, reacts, repairs and replans throughout acting)
(`GNT §1.2` [deck s.5]). The book's answer is the refinement-method stack:
*tasks* are refined by *methods* into subtasks and *commands*, and the Refinement
Acting Engine (RAE) runs that stack, retrying alternative methods when one fails
(`GNT §3.1–3.2`) `[verify]`.

**Where it applies to FactorioRL, hard.** Factorio is a prerequisite game. Every
goal decomposes the same way the book's tasks do: to launch a rocket you must
have blue science, which requires advanced circuits, which requires plastic,
which requires oil, which requires a refinery, which requires... — a landmark
chain 200 deep. Our Phase 8 is literally named "learned skills and hybrid
planning" and its 8.1 acceptance criteria ("skills expose inputs, preconditions,
termination conditions, timeouts, and failure reports") are a *verbatim
description of a refinement method signature*. Phase 7's intervention benchmarks
require an actor that can notice a failure mid-plan and repair — that is
`GNT §3.2`'s Retry plus `GNT §7.2`'s monitoring. Phase 9's rocket progression is
a landmark-ordered plan. The book is the right book.

**Where it does not apply.** The book's *algorithms* mostly assume a small,
enumerable, symbolic state space. Chapter 2's A\*/DFBB, Chapter 5's AND/OR
search and search automata [deck s.43–47], and Chapter 6's value/policy
iteration [deck s.55–57] are all unusable directly on a 65×65×6 sensor tensor
over a continuous-position world with ~200 entity prototypes. Chapter 4's
STNU dynamic-controllability machinery is overkill for a world where every
duration is a known constant published in the engine's own prototype data.
**Take the book's architecture and its representational commitments; leave most
of its solvers.** The one exception worth taking wholesale is the landmark idea
from `GNT §2.3` [deck s.26], because in Factorio landmarks are free and exact.

**The empirical hook.** `docs/evidence/phase4-feasibility-sweep.json` shows
`supply_furnace` at **1.00 train success, 0.30 held-out**, and `deliver` at
**0.65 train, 0.10 held-out against a 0.30 random baseline** — the learned
policy is *worse than random* on an unseen layout family. A flat policy over
fully-bound templates is memorising the layout. The structure that would
transfer — *go to the source, take, go to the sink, give* — is not represented
in the learned object anywhere. That is the book's thesis showing up in our own
evidence file, and it is the reason this is an architecture problem rather than a
hyperparameter problem.

---

## Target architecture for Phase 8–9

A diagram in prose. Existing modules in `code font`; proposed new ones marked
**new**.

**Layer 0 — the platform.** `mod/factoriorl/` and `src/factoriorl/session.py`,
`protocol.py`, `rcon.py`, `worker.py`. Unchanged. In the book's terms these
provide **commands**: `move`, `mine`, `craft`, `place`, `rotate`, `transfer`,
`set_recipe`, `research`, `wait`, `cancel` — the ten `ActionType` values, each
with a real duration, real preconditions, and a structured failure code from the
24-value `ErrorCode` vocabulary. This is already correct and already the right
granularity. Nothing above it may bypass it (PLAN §2: "skills operate through
the same embodied runtime").

**Layer 1 — the command catalog.** `src/factoriorl/catalog.py`. Stays exactly as
it is: an ordered, content-hashed list of fully-bound command templates that a
flat policy can index into. It stops being the *only* action interface and
becomes the *leaf* interface.

**Layer 2 — the descriptive domain.** **new** `src/factoriorl/deliberate/domain.py`.
State-variable vocabulary (`GNT §2.1.2` [deck s.12]) plus a `CommandModel` per
command: parameters, preconditions, effects, cost, and duration bounds. This is
`docs/ACTION_MATRIX.md` and `src/factoriorl/action_matrix.py` promoted from prose
+ tests into a machine-readable object, with the existing matrix tests
*generated from it* so the two cannot drift.

**Layer 3 — abstraction.** **new** `src/factoriorl/deliberate/abstract.py`:
`abstract(observation, memory) -> State`. Turns the wire observation plus the
agent's remembered map into the coarse state variables `domain.py` talks about
(`holds(item)=n`, `at(marker)`, `working(handle)`, `gap_at(x,y)`,
`powered(handle)`). **Reads the policy-visible observation and agent memory only
— never `session.truth()`.** This is also the LM agent's "concise summary" that
PLAN §2 asks the canonical observation to supply.

**Layer 4 — the method library.** **new** `src/factoriorl/deliberate/methods.py`.
`Method(task, params, pre, invariants, body, deadline)`, where `body` is a
generator yielding subtasks and commands. Several methods per task; applicability
decided by `pre` against the abstract state. Content-hashed and versioned exactly
like `ResolvedCatalog.digest()`. Two kinds of body coexist behind the same
signature:

- *authored* bodies — the Phase 3.2 reference solutions, rewritten (see R3);
- *learned* bodies — **new** `src/factoriorl/skills/`, a MaskablePPO policy over
  the primitive catalog trained against a subgoal, wrapped so that from RAE's
  point of view it is indistinguishable from an authored method.

**Layer 5 — the acting engine.** **new** `src/factoriorl/deliberate/rae.py`.
An agenda of **refinement stacks**; each stack entry is `(task, method, step,
tried)`. `Progress` advances one stack by one step; `Retry` pops a failed entry,
adds the method to `tried`, and selects an untried applicable method; when none
remain, failure propagates to the parent (`GNT §3.2`) `[verify]`. Monitors
invariants each cycle (`GNT §7.2`) `[verify]`. **This is the module Phase 7's
recovery benchmarks measure.**

**Layer 6 — method choice.** **new** `src/factoriorl/deliberate/choose.py`, one
interface, three implementations that the manifest names:
`StaticOrder` (Phase 5 baseline) | `LookaheadPlanner` (a forward search over
`domain.py` + `methods.py`, `GNT §3.3`'s SeRPE role) | `LearnedChoice` (PPO over
the applicable-method mask, PLAN 8.3's manager) | `LMChoice` (PLAN 8.4). This
single interface is where PLAN 8.5's four-way hierarchy comparison becomes four
configurations of one system rather than four systems.

**Layer 7 — the actor loop.** **new** `src/factoriorl/deliberate/actor.py`, in
one of the book's three named forms — `Run-Lookahead`, `Run-Lazy-Lookahead`,
`Run-Concurrent-Lookahead` (`GNT §2.6.1`) `[verify]` [deck s.6 on receding
horizon]. Recommended default: lazy lookahead (plan, execute until an invariant
or command fails, replan), because our replanning cost is Python-cheap and our
env step is 24 ms-expensive.

**Layer 8 — goal reasoning.** **new** `src/factoriorl/deliberate/goals.py`
(Phase 9). Given a discrepancy — depletion, a broken supply chain, a new
technology unlocked — generates the next goal for RAE (`GNT §7.3`) `[verify]`.
Phase 7.1's "goals that change without resetting the world" is this module's
minimal version.

**Where the RL env sits in all of this.** `src/factoriorl/env.py` becomes an
*adapter*, not the top. `FactorioEnv` is how a *learned method body* is trained:
it presents one subgoal, one budget, one command catalog subset, and the existing
Gymnasium contract. The critical structural change is that `reset()` today always
installs a blueprint and rebuilds the world; a skill must be able to begin from
whatever state its parent method left behind (see R15). Everything else in
`env.py`, `encoders.py`, `vecenv.py`, `learn/` survives unchanged.

**Reading the stack in one sentence:** RAE holds a refinement stack of tasks;
each task is refined by a method chosen by `choose.py`; a method body is either
authored Python or a learned policy stepping `FactorioEnv`; both bottom out in
catalog commands sent through `session.step`; `monitor` watches invariants and
`Retry` handles failure; the planner is consulted only at method-choice points,
over an abstract state that is strictly coarser than what the policy sees.

---

## Recommendations

### R1 — Write the descriptive model down now, as code

**Book.** Descriptive models say what state results from an action; operational
models say how to perform it. They are different artifacts for different
consumers (`GNT §1.3.1` [deck s.8]). Planning needs the descriptive one
(`GNT §2.1.3`, action templates [deck s.13]).

**Change.** New `src/factoriorl/deliberate/domain.py`. A state-variable
vocabulary and one `CommandModel(name, params, pre, eff, cost, duration)` per
`protocol.ActionType`. Derive the existing precondition assertions in
`src/factoriorl/action_matrix.py` and `docs/ACTION_MATRIX.md` from it, so the
prose matrix, the tests, and the planner's model are one object.

**Phase.** 3 (the file and the generated tests). Payoff in 3 (R17), 4 (R6), 8.

**Impact.** High and early: it is the input to a plan-existence solvability check
(R17) and to landmark shaping (R6), both of which are Phase 3/4 deliverables.
Without it Phase 8 has no planner input and the "hybrid planning" half of Phase 8
cannot be built at all.

**Cost/risk.** ~1 day for the ten commands we have. Risk is drift against Lua;
R13 is the mitigation.

**Conflict.** None. PLAN 2.1 already requires an action matrix stating
preconditions, inventory effects, reach, time behaviour and failure outcomes.
This is that artifact made executable.

---

### R2 — Adopt tasks / methods / commands as the vocabulary, and stop growing the flat catalog

**Book.** The refinement hierarchy has exactly three kinds of thing: *tasks*
(abstract activities), *methods* (alternative ways to refine a task, with
preconditions and a program body), and *commands* (executable by the platform)
(`GNT §3.1`) `[verify]` [deck s.34 names hierarchy as the reason for operational
models].

**Change.** `catalog.py` today is 100% commands with every argument bound at
authoring time. The Phase 3 correction in `LEDGER.md` is the failure mode: the
`place_*` templates were bound to a single fixed offset, so a belt gap could only
be repaired from exactly one standing position, and the fix was to enumerate four
offsets × three items plus two movement strides × four directions. That works at
three placeable items. Phase 9 has oil, pipes, refineries, assemblers, ~60
placeable prototypes and arbitrary positions; enumeration is not available. Add
`deliberate/methods.py` with the `Method` dataclass. Leave `catalog.py` exactly
as it is — it is the correct leaf layer.

**Phase.** Types in 3–4; library populated in 5 and 8.

**Impact.** This is what makes PLAN 8.1's skill contract a 30-line dataclass
instead of a new subsystem invented under deadline.

**Cost/risk.** Low now (the dataclass), high later (retrofitting).

**Conflict.** None — PLAN 8.1's five required skill properties *are* a method
signature. Worth saying so explicitly in PLAN so the connection is not
rediscovered.

---

### R3 — Rewrite the Phase 3.2 reference solutions as refinement methods, not scripts

**Book.** A method body is a program with control constructs, calls to subtasks,
and calls to commands; failure of a step is handled by trying another method for
the same task (`GNT §3.1.2`, `§3.2`) `[verify]`.

**Change.** `src/factoriorl/tasks/reference.py` already contains method bodies
with the structure discarded. `solve_deliver` is
`goto(src); take(item,n); goto(dst); give(item,n)` written as straight-line
Python over a `Driver`, with failure collapsing to a `stuck_reason` string at the
top. Re-express as: task `deliver(item, n, src, dst)` with method
`m-deliver-direct`; task `goto(target)` with methods `m-walk-axis` (today's
greedy axis walk, which `Driver.walk_to` already implements and which already
detects being blocked) and, from Phase 5.1, `m-navigate-path`. Keep `SolveTrace`
— it becomes the refinement-stack trace and the Phase 5.5 replay record.

**Phase.** 3. These now exist (commit `42071d2`, "Phase 3.2: reference solutions,
and the seven defects they found") after `LEDGER.md` downgraded Phase 3 for their
absence — so this is a *refactor of new code*, at its cheapest possible moment,
before Phase 5 and Phase 8 both build on the current shape.

**Impact.** Closes the open Phase 3 gate; every method written here is
simultaneously a Phase 8.2 skill target, a Phase 5 assisted-agent primitive, and
(with R17) a solvability witness.

**Cost/risk.** No extra cost — the work is already owed. Risk of writing them as
scripts instead: they get rewritten in Phase 8.

**Conflict.** None. PLAN 3.2 requires "a scripted solution"; it does not require
it be a straight-line script.

---

### R4 — Make method choice the single hook where learning plugs in

**Book.** RAE must choose among applicable method instances; that choice can be
arbitrary, heuristic, or delegated to a planner (`GNT §3.3–3.4`) `[verify]`. The
choice point is the seam.

**Change.** `deliberate/choose.py`, one function signature
`choose(task, candidates, state) -> Method | None`, four implementations
(static / planner / learned / LM). The learned one reuses our masking machinery
verbatim: the mask over candidate methods *is* the applicability test over
`Method.pre`, exactly as `env.action_masks()` builds a mask from
`ActionTemplate.requires` today.

**Phase.** Interface in 4 (50 lines); implementations in 8.3 and 8.4.

**Impact.** PLAN 8.3 (manager policy), 8.4 (LM skill selection) and 8.5
(hierarchy comparison) collapse from three subsystems into three configurations
of one, which is the only way 8.5's "match task and observation settings where
comparisons claim equivalence" is achievable.

**Cost/risk.** Trivial now.

**Conflict.** None.

---

### R5 — Put a named actor loop in the repo, and record which one ran

**Book.** Three actor loops: `Run-Lookahead` (replan every step, execute the
first action), `Run-Lazy-Lookahead` (execute the plan until it is invalidated,
then replan), `Run-Concurrent-Lookahead` (plan and act in parallel)
(`GNT §2.6.1`) `[verify]`; the receding-horizon rationale is that first steps are
more reliable than later ones [deck s.6].

**Change.** There is no actor loop in the repo today: `env.step()` is called
directly by `model.learn()` and by `reference.Driver`. PLAN 5.3 ("a
provider-independent agent loop") and PLAN 8.3 both need one and would otherwise
each invent one. Add `deliberate/actor.py` with `run_lazy_lookahead(...)`.

**Phase.** 5 for the LM agent; 8 for the hybrid comparison.

**Impact.** Gives PLAN 8.5 a *named axis* to compare along. Right now the axis
has no name, and an unnamed axis is how a comparison becomes the undocumented
composite score PLAN §3 forbids.

**Cost/risk.** Small.

**Conflict.** It adds a run-manifest field (see R16). PLAN §2 enumerates the
manifest fields and PLAN §4 forbids an agent redefining an upstream contract for
its own convenience — so this must go through the integration owner as an
explicit contract change, not be slipped in.

---

### R6 — Represent Factorio prerequisites as landmarks; use them for the goal vector and for shaping

**Book.** The best-known way to build a heuristic is relaxation; landmark
heuristics identify facts that must hold in *every* solution (`GNT §2.3`
[deck s.26 lists max-cost, additive, delete-relaxation and landmark heuristics]).

**Change.** In Factorio, landmarks are free and exact because the recipe graph
gives them: to produce a transport belt you must first hold iron plates and iron
gears; to smelt you must have ore and fuel co-located in a furnace. Add
`deliberate/landmarks.py::landmarks(goal, domain) -> tuple[Predicate, ...]` in
prerequisite order. Two consumers:

1. **The goal vector.** `encoders.py` `GOAL_FEATURES = 12`; `env._goal_vector()`
   spends slot 0 on elapsed-budget fraction and slots 1..10 on the *success*
   predicates — which for every one of our six families are all-false until the
   episode is essentially over. Replace with the landmark-achievement vector: a
   dense, monotone, non-gameable progress signal.
2. **Shaping.** Generate the `RewardComponent`s in each `TaskSpec` from
   landmarks instead of hand-authoring them per family (`mine_smelt` hand-picks
   `plates_produced` and `ore_mined`; those *are* its last two landmarks).

**Phase.** 4 for both consumers; 9 is where it becomes indispensable (a rocket
plan has hundreds of landmarks and no other tractable decomposition exists).

**Impact.** This is the direct answer to credit assignment over horizons longer
than 600 decisions. It is also the only proposal here that plausibly moves the
Phase 4.5 numbers.

**Cost/risk.** Moderate; needs R1 first. **Risk: landmark shaping is not
potential-based and can be farmed.** Mitigation: express each landmark as the
existing `RewardKind.HIGH_WATER` (monotone, pays for the *increase* of a maximum,
never the level), never as a level — `rewards.py` already makes that
structurally safe.

**Conflict — real, and it has a deadline.** Changing the goal vector changes the
observation profile. That requires bumping `local-v1 → local-v2` and
`EXTRACTOR_VERSION`, which invalidates every checkpoint trained against it. PLAN
6.2 requires published checkpoints with recorded hashes. **So this must land
before the Phase 4.5 three-seed release result, not after.** Success predicates
are untouched, so PLAN 3.5's "success remains unchanged when shaping is disabled"
still holds.

---

### R7 — Add `preconditions` and `provides` to `TaskSpec`, separate from `success`

**Book.** A planning problem is `P = (Σ, s0, g)` [deck s.14]; a method carries a
`pre:` (`GNT §3.1.2`) `[verify]`. Goals and preconditions are the same kind of
object — a constraint on state variables — which is what lets one activity's
precondition be another's goal.

**Change.** `tasks/spec.py::TaskSpec` has `success`, `failure`, `rewards`,
budgets and catalog, but nothing that states *what must be true to begin*. Add
`preconditions: tuple[Predicate, ...]` and `provides: tuple[Predicate, ...]`,
both drawn from the existing closed `PredicateKind` vocabulary so
`Predicate.evaluate` handles them unchanged.

**Phase.** Field in 3; used in 7.1.

**Impact.** PLAN 7.1's "goals that change without resetting the world" needs
exactly this and essentially nothing else: stage N+1's `preconditions` are
checked against the live world instead of a reset, and `tasks.validate_all()`
can prove a whole stage chain composes (`provides` ⊇ next `preconditions`)
*without launching an engine*, in the same pass that already checks splits and
budgets.

**Cost/risk.** Two dataclass fields plus one validation rule. **Do it before
task versions freeze for the public release (PLAN 6.4)** — adding a spec field
later bumps the version of every task and breaks published comparisons.

**Conflict.** None.

---

### R8 — Model reusable vs consumable resources, and check resource profiles at generation time

**Book.** A principal motivation for explicit time is "modelling the effects,
conditions, and **resources borrowed or consumed** by an action at various
moments along its duration" [deck s.35] — borrowed = reusable (returned after
use), consumed = depleted. A plan is resource-consistent only if each resource's
profile stays within capacity over the relevant interval (`GNT §4.2`, and the
extended treatment in the authors' earlier volume, *Automated Planning: Theory
and Practice* 2004, Ch.15) `[verify]`.

**Change.** Today the only resource declarations are
`Blueprint.character_inventory` and `EntitySpec.contents`, and the engine-free
solvability checks in `spec.py` are purely geometric (`out_of_box`,
`footprint_conflicts`). Add a consumable-resource check to
`tasks.validate_all()`: for each sampled blueprint, the reference method's
declared consumable requirement must be ≤ what the scene provides. Concretely:
`repair_belt` hands the character 5 transport belts and the **test** family
`double_gap` needs 2 — that margin is currently an accident of the generator, not
a checked property. `mine_smelt` gives `{"coal": 10}` except in `fuelled_furnace`
where it gives 4 plus 6 in the furnace; nothing checks that 4 is enough to smelt
3 plates.

**Phase.** 3 (the checker); 9 (where scarcity is the actual task).

**Impact.** Turns PLAN 3.2's "randomness alone does not establish task quality"
from a principle into a mechanical check over the 24,000 generated scenes
`validate_all` already samples.

**Cost/risk.** Low; no engine needed. No risk.

**Conflict.** None.

---

### R9 — Treat ongoing actions as contingent links with deadlines

**Book.** In an STNU, *contingent* links have durations the actor does not
control; execution is by **dispatching** — propagating time windows as events
occur — and the failure mode to detect is a **deadline failure**
(`GNT §4.3.2`, `§4.4.2–4.4.3`) `[verify]`.

**Change.** We already have contingent durations: `ActionStatus.RUNNING` plus
`session._settle()`'s poll loop covers mining, crafting, smelting and research —
durations the engine owns and the actor cannot shorten. Today `env.step` just
polls until settled and exposes `self[8] = inflight` to the policy. Add to
`CommandModel` (R1) a `duration: (lb, ub)` read from the engine's own prototype
data (mining time, recipe energy — `D:/Factorio/doc-html/runtime-api.json` is the
authority per `LEDGER.md`), and to `Method` a `deadline`. RAE fails a method when
its deadline passes rather than waiting indefinitely.

**Phase.** Duration bounds recorded in Phase 3 alongside R1; deadlines used in
8.1.

**Impact.** PLAN 8.1's "timeouts" gets a definition. More importantly, PLAN 8.3's
"the manager cannot exploit inconsistent time accounting" is precisely the book's
dispatching-consistency requirement, and STNU framing is what makes it checkable
rather than aspirational.

**Cost/risk.** Moderate. **Explicitly do not build an STN solver** (see the
do-not list) — per-method deadlines plus min/max command durations capture
almost all the value at a fraction of the cost.

**Conflict.** None.

---

### R10 — Make failure, retry and method-level replanning first-class

**Book.** RAE's `Retry`: on failure of a step, pop the refinement-stack entry,
add the failed method to the entry's `tried` set, and select an untried
applicable method for the same task; when none remain, propagate the failure to
the parent entry (`GNT §3.2`) `[verify]`. The stack entry is `(τ, m, i, tried)`.
Interleaving acting with planning is even more strongly motivated once outcomes
are non-deterministic [deck s.40, s.43].

**Change.** Today failure is *observed and ignored*: `env.step` returns
`info["action_status"]` and `info["action_error"]` and then continues the episode
as though nothing happened, and `reference.Driver.do` returns `False` and unwinds
all the way to the top with a `stuck_reason`. Neither retries at the right level.
Implement `deliberate/rae.py` with the refinement stack and the `tried` set. Our
24-value `ErrorCode` enum is already the failure vocabulary a retry policy needs:
`OUT_OF_REACH` → reposition and retry the same method; `NO_ITEMS` → refine an
acquire subtask first; `COLLISION` → try a different placement method;
`TECH_NOT_SELECTABLE` (the trigger-gated 2.0 root, already correctly
distinguished from `TECH_LOCKED`) → refine into a crafting task instead of
waiting.

**Phase.** Interface in 5; the engine in 7 — this is what makes intervention and
recovery *measurable*.

**Impact.** Highest-value single item for Phase 7. PLAN 7.3/7.4 require an agent
that observes an intervention's consequences and recovers; there is currently no
component whose job is to notice.

**Cost/risk.** RAE proper is ~300 lines plus tests. The risk is building it too
early and over-generally; build it against exactly the six families first.

**Conflict.** None.

---

### R11 — Add monitoring invariants to the task spec, evaluated by the actor

**Book.** Monitoring is a distinct deliberation function: platform, action, plan
and goal monitoring, detecting discrepancies between the predicted and the
observed (`GNT §7.2`) `[verify]`. Chapter 7 has no local deck coverage.

**Change.** `TaskSpec.failure` is already a tuple of predicates evaluated every
step by `env._failed()` — a monitor with the wrong name and exactly one consumer
(episode termination). Add `invariants: tuple[Predicate, ...]`, evaluated by RAE
rather than by the env, whose violation aborts the *current method* rather than
the episode. Example: `restore_power` should abandon a "wire up this pole" method
the moment the boiler stops working, not 200 decisions later at truncation.

**Phase.** Spec field in 3; consumed in 7.3.

**Impact.** Gives Phase 7.4 a **time-to-detect** measurement, which "time to
restore production" currently conflates with time-to-repair. Those are different
capabilities and PLAN §3 already insists recovery metrics not be merged.

**Cost/risk.** Low — reuses `Predicate.evaluate` unchanged.

**Conflict.** None.

---

### R12 — Keep the planner's state strictly coarser than the observation, and never derive it from `truth`

**Book.** "Predicted states are in general less detailed than the observed one"
(`GNT §1.3.2` [deck s.9]). Predicted states are what the actor reasons *about*;
observed states are what it reasons *from* while performing.

**Change.** FactorioRL currently has exactly two state representations:
`encoders.encode()`'s tensors (policy input) and `session.truth()` (evaluator
ground truth). There is no abstract state, so any planner built today would have
to plan over 6×65×65 float planes. Add `deliberate/abstract.py`, versioned like
an observation profile. **Its input signature must be
`abstract(observation, memory)` — never `truth`.** `env.action_masks()` already
gets this exactly right ("built from the observation alone... which is what makes
'no privileged information in masks' checkable"); extend the same discipline and
the same test to the abstraction function.

**Phase.** 5.4 (persistent agent memory is the same object as `memory` here);
used from 8.

**Impact.** The single artifact without which no planner is possible. It doubles
as PLAN §2's promised "typed objects and concise summaries for language-model
agents" — that summary should *be* the abstract state, not a prose rendering of
the tensor, or the LM agent and the planner will diverge in what they believe.

**Cost/risk.** Moderate. Risk: abstraction that is too coarse makes plans
unexecutable, too fine makes search intractable. Start at marker granularity.

**Conflict.** PLAN §2 says "one canonical structured observation supplies" three
views. This adds a fourth derived view. Argue it as a *derived* view of the same
canonical observation, and make that structural (it takes the observation as
input), so the canonical-observation contract is preserved rather than forked.

---

### R13 — Do not maintain two hand-written models; generate one from the other and test the pair against the engine

**Book.** The acknowledged cost of the descriptive/operational split is that two
models are written separately and drift apart; the book raises this against its
own architecture (`GNT §3.5` discussion) `[verify]`.

**Change.** We are unusually well placed to answer this objection, because we
have a real engine in the loop. Make `deliberate/domain.py` the single source and
generate from it (a) the Python action-matrix test cases in
`tests/` that `action_matrix.py` backs and (b) the precondition fixtures the Lua
side is checked against. Then add a **model-consistency test** in
`tests/engine/`: for N sampled states, executing command `c` in the real engine
must produce the effects `domain.py` predicts, or the test fails.

**Phase.** 3–4 (the generation and the test); the payoff is in 8.

**Impact.** Prevents the worst Phase 8 outcome: building a planner and then
discovering its model of Factorio is wrong. Turns the book's own stated weakness
into a passing test.

**Cost/risk.** The consistency test is engine-dependent, so it belongs in
`tests/engine` and runs at ~24 ms/step — budget it as a sampled check, not
exhaustive.

**Conflict.** None; it strengthens PLAN §4's "works with a mock is an incomplete
state".

---

### R14 — Measure dead-ends, not just success — the "safe solution" property

**Book.** Solutions are classified as unsafe / safe-acyclic / safe-cyclic: a
*safe* solution is one where from **every** reachable state a goal is still
reachable [deck s.42, s.54]; an unsafe policy "may or may not terminate; if it
does, it may reach either a goal or a state with no applicable action"
[deck s.54]. Safe cyclic solutions are legitimate — retry loops are solutions.

**Change.** `learn/train.py::evaluate` reports `success_rate`, Wilson interval,
mean reward and mean length. It cannot distinguish "ran out of budget while
making progress" from "walked into a state from which the goal is unreachable".
Add a **dead-end rate**: for each truncated episode, run the reference method
(R3) from the truncated state, evaluator-side; if it cannot finish, the episode
ended in a dead end. `deliver`'s 0.10 held-out against a 0.30 random baseline is
almost certainly a dead-end signature — a random agent rarely does anything
irreversible, and our agent does.

**Phase.** 4 (the metric); a requirement in 8.

**Impact.** Directly serves PLAN §4's failed-gate procedure step 2 ("identify
whether the failure is a runtime defect, task defect, learning limitation, or
performance limitation"). Success rate alone cannot make that call; dead-end rate
can.

**Cost/risk.** Reuses R3. Costs evaluation wall-clock, so run it on the
evaluation set only.

**Conflict.** None — and to be explicit: **this is an additional metric, not a
replacement.** The 80% held-out success bar in PLAN §3 and 4.5 stands untouched.

---

### R15 — Split `FactorioEnv.reset` into scenario-install and episode-begin

**Book.** A method's internals may be authored or learned; the acting layer's
interface does not change when they do (`GNT §3.1.1`, `§3.4`) `[verify]`. For
that to be true, a learned body must be able to run from an arbitrary current
state, not only from a fresh one.

**Change.** `FactorioEnv.reset()` today does five things at once: pick a layout
family, generate a blueprint, install it, reset the world, and initialise
goal/accountant/counters. Split into `reset_scenario()` (the first four —
today's behaviour, what Phase 3's 504-reset leakage run tests) and
`begin_episode(goal, budget, catalog_subset)` (the fifth — **no world
mutation**). Then:

- a Phase 8.2 skill is trained by `begin_episode(subgoal)` on a live world;
- a Phase 7.1 persistent stage transition is the same call;
- a Phase 8.3 manager's decision is `begin_episode` on a child.

**Phase.** **Do the split in Phase 4.** Use it in 7 and 8.

**Impact.** This is the highest rewrite-risk item on the list. If `reset` stays
fused, Phase 7.1 and Phase 8.2 each require surgery on `env.py`, `vecenv.py`
(whose auto-reset in `step_wait` calls `env.reset()` directly), and every gate
that resets — `gate_phase3`, `gate_phase4`, `tools/solvability.py`,
`tasks/reference.py`.

**Cost/risk.** One day now against roughly a week later plus a checkpoint
invalidation, because the split changes when the goal vector is set and therefore
what the policy sees on step 0.

**Conflict.** None; PLAN 7.1 demands it implicitly ("stage transitions do not
refill resources or repair the factory").

---

### R16 — Version the deliberation stack in the run manifest, as profiles already are

**Book.** An actor is characterised by which deliberation functions it has and
how they are organised (`GNT §1.2`, `§1.4` [deck s.4–5]); comparing two actors
without stating those is comparing two unnamed things.

**Change.** `manifest.py` records `profiles.{observation, action, assistance,
catalog, catalog_digest, resolved_catalog}`. Add
`profiles.deliberation = {actor_loop, method_library_version,
method_library_digest, choice, planner, abstraction_version}`. Content-hash the
method library exactly as `ResolvedCatalog.digest()` hashes the catalog.

**Phase.** Field in 4; populated from 8.

**Impact.** PLAN 8.5 requires "report differences in assistance explicitly" and
PLAN 10.3 requires "results identify the system being measured, not only the
model name". A four-way hierarchy comparison without this field is exactly the
undocumented composite score PLAN §3 forbids.

**Cost/risk.** Trivial now; a schema migration later.

**Conflict.** Same as R5: a manifest schema change is an upstream contract change
under PLAN §4 and needs the integration owner, not a unilateral edit.

---

### R17 — Replace the *solvability proof* with plan existence; keep the scripted solver as the *baseline*

**Book.** Forward state-space search with a relaxation heuristic (`GNT §2.2–2.3`
[deck s.15–26]). Plan existence over a declared domain is a stronger statement
than one script succeeding on one seed.

**Change.** Once `domain.py` exists (R1), extend `tasks.validate_all()` and
`tools/solvability.py` to run a bounded forward search (greedy best-first with an
additive-cost relaxation — [deck s.23, s.26]) over the abstract state of each
sampled blueprint, and assert a plan exists. The abstract state space at marker
granularity is tiny; this runs over 24,000 scenes **without launching an
engine**, in the same validation pass that already runs before any worker starts.

**Phase.** 3.

**Impact.** Catches the whole `place_*`-bound-to-a-fixed-offset class of defect
at validation time rather than after a 1,470-second sweep across three concurrent
workers. `LEDGER.md` is explicit that this defect cost Phase 3 its acceptance and
five of six families their learnability.

**Cost/risk.** Small search. **Risk: a plan exists in the model but not in the
engine** — R13's consistency test is the guard, and this pairing is why R13 is
not optional.

**Conflict.** None. PLAN 3.2 asks for "a solvability argument backed by
generation constraints and a reference solution"; this strengthens the argument
and keeps the reference solution as the scripted baseline PLAN also requires.

---

### R18 — Shape at method boundaries and discount by elapsed ticks (SMDP), once methods exist

**Book.** RAE reports success or failure per *task*, not per command; a method's
termination condition is its own success test (`GNT §3.1.2`, `§3.2`) `[verify]`.
Chapter 6's value functions accumulate cost over the actions a policy performs
[deck s.55], which for variable-duration actions means the discount must track
duration, not step count.

**Change.** `rewards.py` computes a component per 30-tick decision. Add
`RewardKind.LANDMARK` (monotone, paid once on first achievement of a landmark
from R6). For the Phase 8.3 manager, discount over *method* transitions as
`gamma ** (elapsed_ticks / decision_ticks)` — the standard SMDP discount — rather
than per method invocation.

**Phase.** `RewardKind.LANDMARK` in 4; the SMDP discount in 8.3.

**Impact.** Without the duration-aware discount the manager treats a 3-tick
`wait` and a 3,000-tick smelt as equally costly, which is precisely the
"inconsistent time accounting" PLAN 8.3 says the manager must not be able to
exploit. This is the mechanism that makes that criterion satisfiable.

**Cost/risk.** Low. Risk: `gamma` becomes two numbers (per-decision and
per-method); record both in the manifest.

**Conflict.** None, provided `success` predicates are untouched so PLAN 3.5's
shaping-off parity still holds.

---

## Decisions to make now, to avoid a Phase 8 rewrite

These are cheap today and expensive-to-impossible after the Phase 4.5 release
result and the Phase 6 freeze. Ordered by deadline.

1. **Split `FactorioEnv.reset` into `reset_scenario` + `begin_episode` (R15).**
   Deadline: before Phase 7 starts. Everything hierarchical needs an episode that
   begins without a world reset.
2. **Fix the goal vector to landmarks and bump `local-v1 → local-v2` (R6).**
   Deadline: **before the Phase 4.5 three-seed run**, because after it the
   checkpoints are published artifacts with recorded hashes (PLAN 6.2).
3. **Add `preconditions`, `provides` and `invariants` to `TaskSpec` (R7, R11).**
   Deadline: before task versions freeze at PLAN 6.4. Adding a spec field later
   bumps every task version and breaks published comparisons.
4. **Add `profiles.deliberation` to the run manifest (R16), even if it reads
   `{"actor_loop": "none", "choice": "flat-policy"}` for all of Phase 4.**
   Deadline: before the first published manifest schema. It costs nothing now and
   it is what makes Phase 4 results comparable to Phase 8 results at all.
5. **Fix the state-variable vocabulary and write `domain.py` (R1).** Deadline:
   before the landmark work (R6) and the solvability check (R17), both of which
   consume it. Everything downstream inherits this vocabulary; changing it later
   changes the planner, the abstraction, the landmarks and the shaping at once.
6. **Refactor the just-landed Phase 3.2 reference solutions into methods (R3).**
   Deadline: before Phase 5.1 replaces `Driver.walk_to` with real navigation and
   before Phase 8.2 picks skill targets — both of which will otherwise harden the
   current straight-line shape. Cheapest the day after they landed, and it gets
   more expensive every week.
7. **Establish the rule that abstraction and method preconditions read
   observation + memory, never `truth` (R12), and add the test that enforces
   it.** Deadline: before the first line of `abstract.py`. Free now; unpicking a
   `truth` dependency later invalidates every hybrid result produced up to that
   point.
8. **Decide that the method library is a versioned, content-hashed artifact
   (R16).** Deadline: with the first method. Retrofitting provenance onto a
   library is how checkpoints become opaque blobs — the same argument
   `learn/policy.py` already makes for `EXTRACTOR_VERSION`.

---

## Do NOT do this

1. **Do not build a general HTN planner or a PDDL front end.** The book's own
   Ch.3 discussion notes HTN planning needs descriptive models of every method as
   well as operational ones (`GNT §3.5`) `[verify]`. We will have on the order of
   twenty tasks. A hand-written method library with a bounded forward search over
   it delivers the same behaviour in a week rather than a quarter, and does not
   drag in a domain language nobody else on the project will edit.
2. **Do not implement STN/STNU dynamic-controllability checking** (`GNT §4.3.2`).
   Every duration in Factorio is a published constant in the engine's own
   prototype data. Per-method deadlines and per-command min/max durations (R9)
   capture the value; the controllability algebra buys nothing here.
3. **Do not replace MaskablePPO with value or policy iteration** [deck s.55–57].
   Those assume an enumerable state space or a good initial `V0`. Take Ch.6's
   *concepts* — the safe-policy property (R14) and the duration-aware discount
   (R18) — not its solvers.
4. **Do not adopt plan-space / partial-order planning** (`GNT §2.5` [deck s.28])
   for factory layout. Its payoff is ordering flexibility; our constraints are
   almost entirely resource-prerequisite, which landmark ordering already
   captures, and we have a single character who can only do one thing at a time,
   so there is no concurrency to exploit.
5. **Do not feed the planner's plan into the policy's observation as a hint.**
   That is the assisted profile leaking into the primitive profile. It would
   break PLAN 10.3's ablation claims and make every Phase 4 number
   non-comparable. The plan is a *choice made above* the policy, not a feature
   fed to it.
6. **Do not let `abstract.py`, `Method.pre` or `choose.py` read
   `session.truth()`.** It is convenient and it is fatal: it turns every hybrid
   result into an evaluator-information result. `env.action_masks()` holds this
   line today with a test that reconstructs the mask from a policy-visible
   observation; extend that test, do not weaken it.
7. **Do not use the hierarchy to shorten episodes in order to reach the 80%
   bar.** If a family only passes with a method library, that is a *different
   deliberation profile* and belongs in a separate row of the PLAN 8.5
   comparison, reported as such. PLAN §4 forbids lowering thresholds; it equally
   forbids quietly changing what the threshold is measured on.
8. **Do not let `factoriorl.tasks` import `factoriorl.deliberate`.** The
   import-direction test in `tests/unit/test_task_isolation.py` keeps tasks free
   of worker management for a good reason. Extend the same discipline: tasks
   declare predicates; `deliberate` consumes them; never the reverse.
9. **Do not defer `domain.py` to Phase 8 on the grounds that it is a Phase 8
   artifact.** It is the input to a Phase 3 solvability check (R17) and a Phase 4
   shaping generator (R6). Deferring it defers those too.
10. **Do not rewrite `catalog.py` into a parameterised action space.** The flat,
    fully-bound, content-hashed catalog is *correct as the command layer* and its
    reproducibility discipline is one of the better things in this repo. The fix
    for its combinatorics is a layer above it, not a redesign of it.

---

## Open questions

1. **Does the descriptive model need spatial reasoning?** Can `place(item,
   marker)` abstract away tile geometry, with a `m-place-adjacent` method doing
   the positioning? If yes, the planner's state space stays tiny. If no, R17's
   plan-existence check does not close. And if markers are the abstraction, *who
   invents markers* during Phase 9 expansion, when there is no generator to
   declare them?
2. **What granularity does `connected(pole, network)` need?** Does the planner
   need the electric-network graph, and can that be derived from a 32-tile sensor
   with remembered observations, or does it need a memory structure that
   accumulates network topology across an episode?
3. **Is a learned skill a method *body* or a method *choice function*?** The book
   permits both (`GNT §3.4`) `[verify]`, and PLAN 8.2 vs 8.3 implies both, but
   PLAN never says whether they share an observation encoder. If they do,
   `encoders.py` must serve two very different decision frequencies.
4. **Deliberation cost against a 24 ms env step.** RAE plus a lookahead adds
   Python-side cost per decision. At 41.6 env steps/s single-worker, a 20 ms
   lookahead halves throughput — and throughput is the binding constraint on the
   whole program. Does the manager decide only at method boundaries (cheap,
   coarse) or every 30 ticks (expensive, reactive)? This needs measuring before
   Phase 8 design is fixed, and the Phase 4.3 profiling harness is the place to
   measure it.
5. **Where does the LM agent sit?** At the choice function (R4), or above RAE as
   a goal generator (`GNT §7.3` goal reasoning) `[verify]`? The two have
   different failure modes, different token costs, and different claims about
   what was learned. PLAN 8.4 does not distinguish them and should.
6. **Are the six introductory families tasks, or methods?** `navigate` and
   `deliver` look like methods for larger tasks; `mine_smelt` and `restore_power`
   look like tasks. If four of six are really methods, then PLAN 4.5's acceptance
   bar is measuring *skill quality*, not *task quality* — which is defensible and
   arguably the point, but it should be stated rather than discovered in Phase 8.
7. **Can R13's model-consistency test run at a useful rate?** It is
   engine-dependent and the engine is the bottleneck. How many sampled
   state/command pairs per command give useful coverage within a gate's time
   budget?
8. **Does `deliver` fail from dead ends or from exploration?** R14's metric
   answers this, and the answer determines whether the fix is a task change, a
   catalog change, or a hierarchy. Worth running before committing to any of the
   above.

---

## One gap in PLAN.md worth raising with the integration owner

PLAN Phase 8 is titled "Learned skills and hybrid planning" and its five
sub-items are 8.1 skill contract, 8.2 train skills, 8.3 manager policy, 8.4 LM
skill selection, 8.5 hierarchy comparison. Four of the five are about *skills*.
**There is no line item anywhere in PLAN for the descriptive domain model, the
abstract state, or the planner** — the three things without which the word
"planning" in the phase title has no referent. As written, Phase 8 delivers a
learned option hierarchy and calls it hybrid planning.

The ordering PLAN gives (contract → skills → manager → LM → comparison) matches
the book's layering exactly and needs no change. The recommendation is to add
**8.0 — descriptive domain model and abstract state** as a prerequisite, with its
artifacts (R1, R12) actually built in Phases 3–4 where they already pay for
themselves, so that 8.0 is an acceptance step rather than a build step.
