# Algorithms for Decision Making — design review of FactorioRL

Kochenderfer, Wheeler & Wray, *Algorithms for Decision Making* (MIT Press, 2022).
Citations are `§section, p.N` using the book's **printed** page numbers.

Reviewed against: `PLAN.md`, `docs/LEDGER.md`, and `src/factoriorl/`
(`encoders.py`, `env.py`, `catalog.py`, `rewards.py`, `vecenv.py`,
`learn/policy.py`, `learn/train.py`, `tasks/`) at the state recorded in the
ledger — Phases 0–2 Accepted, Phase 3 and Phase 4 **Needs correction**.

---

## What this book offers FactorioRL, and where it does not apply

This book is a *formulation* book before it is an algorithms book. Its four
useful contributions here, in order of value:

1. **It names the object we are actually building.** FactorioRL is not a
   generic deep-RL environment; it is a **deterministic POMDP with a randomized
   initial-state distribution**, solved online, on a simulator we can pause and
   step exactly. The book has an entire part (IV, chs. 19–23) on state
   uncertainty and an entire chapter (9) on online planning that assumes exactly
   the generative model we own. Almost nothing else in the RL literature assumes
   you can *stop the world*.
2. **It settles the search question with cost formulas, not vibes.** §9.3–9.9
   and §22.2–22.7 give per-decision complexity for forward search, branch and
   bound, sparse sampling, MCTS and open-loop planning in the exact form we need
   to multiply by our measured 24.03 ms/step and 6.21 ms/reset.
3. **It is the best single source for our evaluation methodology.** Chapter 14
   (Policy Validation) is short, and it endorses — with reasons — three things
   PLAN.md already asserts on instinct: report uncertainty (§14.1), do not
   collapse heterogeneous metrics into one score (§14.4), and stress-test
   against models you did not optimize against (§14.3). It also supplies two
   tools PLAN.md does not yet have: importance sampling for rare outcomes
   (§14.2) and most-likely-failure search (§14.5).
4. **It gives a principled, cheap alternative to recurrence.** §20.1 (the
   belief-state MDP), §21.1 (QMDP), and §23.1 (finite-state controllers) between
   them describe three ways to have memory without an LSTM.

**Where it does not apply.** The book is a *tabular and small-parametric* book.
Its algorithms are written for state spaces you can enumerate: alpha vectors
(§20.3), point-based value iteration (§21.4), exact belief filters (§19.2),
QMDP's `O(|A|²|S|²)` iteration (§21.1, p.428) — none of these run on a Factorio
surface. Chapter 5 (structure learning), Part V (multiagent), the Kalman-filter
family (§19.3–19.5, our state is discrete and combinatorial, not
linear-Gaussian), and inverse RL (§18.4–18.6, we *have* a reward function and
PLAN.md requires it be declared) are all out of scope. The deep-RL machinery we
actually use is confined to one appendix (D) and about fifteen pages
(§12.5 PPO clamping p.257, §13.2 GAE p.269) — this book will not help tune
MaskablePPO. Use it for *what problem to pose*; use Sutton & Barto for *how to
fit the network*.

One caution about method-transfer: the book's algorithms assume `T(s'|s,a)` is
cheap. Ours costs 24 ms and lives in another OS process. Every recommendation
below carries an explicit engine-step budget for that reason.

---

## How we should formulate the problem

The book's answer, stated as choices we can make in this repo.

### 1. The problem class is a *deterministic* POMDP, not a stochastic MDP

Factorio's simulation is deterministic; our ledger proves it (Phase 0.5: ten
reset→act→observe cycles producing byte-identical records; Phase T's
`bench speed`: "5 cycles at speed 1 and 30, compared field by field, zero
mismatches, `identical: true`"). Within an episode, `T(s'|s,a)` is a **point
mass**. All the randomness in FactorioRL lives in `SeedPlan.generator_rng`,
sampled once at `reset` — i.e. in the initial-state distribution `b`, which the
book writes as `P(τ) = b(s₁)∏ T(s_{t+1}|s_t,a_t)` (§14.1, eq. 14.2, p.282).

This matters enormously and is currently unexploited:

- **Sparse sampling (§9.5, p.187) is pointless for us.** Its whole purpose is to
  cap the branching on next *states*. Our branching on next states is 1. Use
  `m = 1` and the algorithm degenerates to forward search (§9.3) — which for a
  deterministic problem is just depth-first search (Appendix E.3, p.599).
- **Open-loop planning's known failure mode does not bite from transition
  stochasticity.** Example 9.9 (p.203) shows open-loop plans losing to
  closed-loop plans *because* the agent could have branched on an observed
  stochastic outcome. Our transitions are deterministic, so within a known
  region an open-loop plan is exactly as good as a closed-loop one. The residual
  gap comes only from **state uncertainty** — walking somewhere and *seeing*
  something new.
- Therefore the right object to sample over is not "which successor state" but
  **"which completion of the unobserved world"**. That is precisely
  *multiforecast model predictive control* / hindsight optimization (§9.9.3,
  eq. 9.7, p.208), and its determinized-belief cousin DESPOT (§22.6, p.459).

Record this in `PLAN.md`'s architecture section. It changes which algorithms are
even candidates.

### 2. We already maintain a belief. It is a *filter*, and it is truncated

PLAN.md §3 forbids recurrent policies before the baseline works, and the review
brief frames our design as "remembered entities plus an age feature fed to a
feedforward policy," implying memorylessness. That framing is wrong, and the
correct framing is better news.

§20.1 (p.407) — "Any POMDP can be viewed as an MDP that uses beliefs as states"
— says a policy is legitimate if its input is a *sufficient statistic of the
history*. Phase 2.3 already builds one: the Lua-side remembered-entity store
with `age`, updated by a rule (entities inside the sensor region but absent from
the sweep are deleted, "because the agent looked and they were not there"). That
is a **discrete state filter** (§19.2, p.380) with a degenerate, point-mass
belief per entity. `encoders.py` is a *belief encoder*, not an observation
encoder.

So the design is principled. Its two real defects are:

- **The belief is truncated to the 32 nearest entities** (`MAX_ENTITIES = 32`,
  `encoders.py:57`), sorted by distance (`encoders.py:150–156`). At Phase 4
  scale (5–15 entities per scene) this is free. At Phase 7 scale a persistent
  factory has thousands, and the 32 slots will be consumed entirely by the
  belts under the agent's feet. The one entity that matters — the machine that
  stopped 40 tiles away — is exactly the one distance-sorting evicts.
- **`age` is the only uncertainty statistic.** For a static wall that is
  sufficient. For a *machine's contents* it is not: what we need is
  `P(the furnace is still running)`, which decays with age at a rate that
  depends on the machine's type and last-seen fill level. The book's answer is
  either an explicit filter over the small number of *state variables that
  matter* (§19.2), or particles over them (§19.6, p.390).

### 3. Factor the observation by *relevance*, not by proximity

The book's core representational argument (§2.4, p.25; §2.5, p.33; §2.6, p.35)
is that a joint distribution over `n` binary variables needs `2ⁿ − 1`
parameters, but conditional independence structure reduces the running example
from 31 parameters to 10. The lesson transfers directly: **do not scale the
representation with the number of entities; scale it with the number of
decision-relevant factors, which is constant.**

Concretely, for Phase 7, the entity list should stop being "32 nearest
anything" and become a fixed budget *per role*: k slots for the task-relevant
machine type, k for power, k for logistics, k for the disrupted subsystem, plus
one coarse **aggregate plane** (per-type counts and status summed into e.g.
8×8 super-cells) that is `O(1)` in entity count and `O(area)` in map size. That
is a factored representation with a fixed parameter budget per factor.

### 4. Objectives are multi-attribute and must stay a *vector*

Chapter 6 establishes that rational preferences imply a scalar utility (§6.2,
p.112) — but §14.4 (p.289) is the operative section for us: when a system has
competing objectives, the analysis object is a **Pareto frontier**, and policies
dominated on every metric are eliminated while the rest are reported as a
frontier, not ranked. PLAN.md §3's ban on "one undocumented composite score" for
Phase 7 recovery metrics is not squeamishness; it is what §14.4 prescribes.
Where PLAN.md is *under*-specified is that it does not say what to do instead —
§14.4 does: sweep a policy parameter, plot the trade-off curve, report the
frontier (fig. 14.5, p.289; example 14.5, p.292).

Note the split the book insists on and we should adopt verbatim: the **reward
function used for optimization** may be scalar (it must be, for PPO), while the
**performance metrics used for validation** are a vector and are never summed.
`rewards.py` is the former; the Phase 7 recovery report is the latter. They
should not share a configuration object.

### 5. Action formulation: bounded, ordered, and mask-legal is right; the
### catalog is the *planning model*

`catalog.py`'s fully-bound templates give `|A| ∈ [9, ~40]`. Every complexity
formula in ch. 9 is exponential in `|A|`, so this is not merely tidy — it is the
single design decision that makes search affordable at all. Keep it.

§14.3's advice (p.288) — "we typically want our planning model … to be
relatively simple to prevent overfitting … our evaluation model can be as
complex as we can justify" — argues for exactly the structure we have: a coarse,
bounded catalog and a full-fidelity engine. Do not be tempted to enrich the
catalog to make search stronger; a 14-action catalog searched to depth 4 is
`14⁴ = 38k` leaves, and a 40-action one is `2.6M`.

---

## Recommendations

Each: idea + citation; the repo change; the phase; expected impact; cost/risk;
conflict with PLAN.md.

### 1. Stop average-pooling the spatial grid to a single vector

**Idea.** §8.1 (p.161) and Appendix D.4 (p.587): a parametric value/policy
representation must retain the features the decision depends on. A convolutional
stack is a spatial encoder; global average pooling deletes the spatial signal.

**Repo change.** `learn/policy.py:37–46`: three stride-2 convs take the
`6×65×65` grid to `64×9×9`, and then `nn.AdaptiveAvgPool2d(1)` collapses it to
64 numbers. **All positional information in the grid is destroyed.** Walls and
buildings survive this because they *also* appear in `entities` with relative
`(dx, dy)` features (`encoders.py:161–163`). **Resources do not** — ore tiles
enter only through `resources.tiles` → the grid planes (`encoders.py:125–132`)
and never through the entity list. So today, on `mine_smelt`, the policy can see
*how much* ore is nearby and cannot see *which way it is*. Replace the pooling
with a spatially-preserving head (flatten the `9×9`, or an egocentric
multi-scale crop: `16×16` at 1 tile/cell plus `16×16` at 4 tiles/cell), **and**
add an explicit relative-vector feature for the nearest resource of each type in
`RESOURCES` to the `self` or `goal` block.

**Phase.** 4, immediately — this is corrective work under the existing
"Needs correction" status, alongside the catalog defect the ledger records.

**Impact.** Potentially decisive for `mine_smelt` and `supply_furnace`. The
ledger says "five of six families cannot be learned as designed" and attributes
it to the `$ahead` placement binding; that binding is now fixed
(`catalog.py:72`, `PLACE_OFFSETS`) but the resource-direction blindness is a
second, independent cause that the same investigation would not have surfaced.

**Cost/risk.** Bump `EXTRACTOR_VERSION` (`learn/policy.py:20`) — this
invalidates every existing checkpoint, which is exactly what that constant is
for. Parameter count rises from 266,070 to roughly 600k–900k if you flatten
`64×9×9`; use a `1×1` conv to 16 channels before flattening to stay near 400k.
Larger input to the head means more GPU memory, on a 4.3 GB card.

**PLAN conflict.** None. PLAN 4.1 asks for "a compact encoder for local spatial
state"; a global-average-pooled encoder is not one.

### 2. Give the shaping components their policy-invariance status explicitly

**Idea.** §17.5, eq. 17.15, p.343: a shaped reward leaves the optimal policy
unchanged **iff** `F(s,a,s') = γβ(s') − β(s)` for some potential `β`.

**Repo change.** `rewards.py` implements exactly this for `RewardKind.POTENTIAL`
(`rewards.py:99`) — cite Ng, Harada & Russell via §17.5 in the module docstring,
because it is currently justified only by the loop-sum argument. But
`HIGH_WATER` (`rewards.py:91–95`) and `STEP_COST` (`rewards.py:86–88`) are **not**
potential-based and *can* change what is optimal. Add a `policy_invariant: bool`
field to `RewardComponent` in `tasks/spec.py:183`, set `True` only for
`POTENTIAL`, and surface it in the run manifest's `reward.components` block
(`learn/train.py:277–280`).

**Phase.** 4.4 (shaping dependence), where this is the whole question.

**Impact.** PLAN 4.4 asks the report to "distinguish faster learning from changed
task difficulty." With this field the distinction is mechanical: only
non-invariant components can have changed the task. It also tells you where to
look when shaping-on and shaping-off success rates diverge.

**Cost/risk.** An afternoon. No behaviour change.

**PLAN conflict.** None; it strengthens 3.5 and 4.4.

### 3. Report relative standard error alongside the Wilson interval

**Idea.** §14.1, p.285: `SE = σ̂/√n` (eq. 14.5) and the **relative** standard
error `σ̂/(µ̂√n)` (eq. 14.6). The book's worked case (exercise 14.2, p.296) shows
an absolute error of `3×10⁻³` with a *relative* error of `0.316` — the absolute
number looks fine and the estimate is nearly worthless.

**Repo change.** `learn/train.py:165` (`_wilson`) is correct for a proportion
and should stay. Add relative standard error to the dict returned by
`evaluate()` (`learn/train.py:155–162`) and to `random_baseline()`. It is the
number that matters for the *random baseline* and for early training, where
`success_rate` is near zero: the Phase 4 pilot's baseline `0.00 (0/10)` with
Wilson `[0.0, 0.2775]` is honest but uninformative, and the relative error makes
"this measurement cannot distinguish 0 from 0.2" explicit.

**Phase.** 4.5 and 6.2 (published artifacts).

**Impact.** Prevents publishing a headline number whose sample size cannot
support it. At the acceptance threshold — 100 episodes at 80% — Wilson is
`[0.71, 0.87]`; a reader deserves to see that the interval half-width is 8
points before comparing two seeds.

**Cost/risk.** Trivial. Do **not** let this become a reason to change the
threshold; it changes only what is reported next to it.

**PLAN conflict.** None; PLAN §3 asks for "aggregate uncertainty" and this is
more of it.

### 4. Adopt lookahead-with-rollouts as the first search policy

**Idea.** §9.2, p.183, alg. 9.1: act greedily with respect to values estimated
by simulating a rollout policy `π` to depth `d`. The book states its status
plainly: "This approach often results in better behavior than that of the
original rollout policy … It can be viewed as an approximate form of policy
improvement used in the policy iteration algorithm (§7.4)." Complexity
`O(m|A||S|d)`; for a deterministic problem with one successor, `m|A|d` engine
steps per decision.

**Repo change.** New module `src/factoriorl/search.py`, owning its own
`WorkerPool` (see the search subsection below). Use the trained MaskablePPO
policy as the rollout policy `π`, masked. For `repair_belt` (`|A| = 14`,
`catalog_subset` at `tasks/families/repair_belt.py:50–65`) with `d = 8` and
`m = 1`: **112 engine steps per decision**, 2.7 s on one worker, 0.55 s on
eight.

**Phase.** 8 (hybrid planning) as a first-class agent; usable in Phase 5 as a
strong non-learned baseline for the demonstration.

**Impact.** The cheapest thing in the book that provably improves on the policy
we already have. It is also the correct control for PLAN 8.5's hierarchy
comparison: "policy + one-step lookahead" is the baseline that any learned
manager must beat.

**Cost/risk.** Requires the snapshot/restore primitive (rec. 6). At 0.55 s per
decision and 300 decisions, an episode costs ~3 minutes; 100 evaluation episodes
cost 5 hours. Budget it as an evaluation configuration, not a training loop.

**PLAN conflict.** None, but it must be declared as an **assistance profile**
under PLAN 2.5 / 10.3 — a search policy is not the same system as a compact
policy and results must not be compared as though they were.

### 5. Use branch and bound, not plain forward search, if you go deeper than 2

**Idea.** §9.4, p.185, alg. 9.3: prune an action subtree when its `Q` upper
bound falls below the best lower bound found so far; identical result to forward
search, often a third of the visits (example 9.2, p.187). §9.11 exercise 9.2
(p.209): two admissible heuristics combine as `min(h₁, h₂)`.

**Repo change.** In `search.py`. The bounds are available cheaply and honestly:
- **Lower bound `U`**: the value head of the trained MaskablePPO critic,
  or a rollout of the policy (§22.7, p.464 notes a rollout is not a *guaranteed*
  lower bound and converges to one).
- **Upper bound `Q̄`**: the best-action-best-state bound (§21.1, eq. 21.2,
  p.429), `max_{s,a} R(s,a)/(1−γ)`. Our reward is bounded by construction —
  `SPARSE_SUCCESS` weight 1.0 plus the potential telescoping term — so this is
  computable from `tasks/spec.py`'s declared `RewardComponent` weights without
  touching the engine. A tighter admissible bound for the navigation-flavoured
  families: remaining potential achievable at maximum walking speed.

**Phase.** 8.

**Impact.** Turns depth 3 from `14³ = 2744` steps (66 s/decision, 1 worker) into
roughly 900 (22 s), per the book's ~3× figure. Not enough to make depth 3 an
online policy; enough to make it a teacher (see below).

**Cost/risk.** Bounds derived from a *learned* critic are not admissible, so
optimality is lost (§9.7, p.197 notes an inadmissible heuristic "may still
converge to a policy that performs well"). Say so in the profile.

**PLAN conflict.** The bounds must be built from declared reward weights and the
policy's own critic — **never** from `truth()` or the blueprint. This is the
same constraint `env.action_masks()` already satisfies (`env.py:124–139`) and it
should be enforced the same way: a test that constructs the bound from a
policy-visible observation and asserts equality.

### 6. Add `snapshot` / `restore` to the protocol — the missing primitive

**Idea.** Every online-planning algorithm in ch. 9 and ch. 22 requires a
*generative model* it can re-invoke from an arbitrary state (`randstep`,
alg. 9.1, p.184). We do not have one.

**Repo change.** `protocol.py:47` (`RequestType`) has `RESET`, which restores a
**blueprint** — an episode *start*, in 6.21 ms. There is no way to return to a
mid-episode state. Add `SNAPSHOT` and `RESTORE` to `RequestType` and
`mod/factoriorl/runtime.lua`, serialising into an extension of the existing
blueprint schema (`tasks/spec.py:70`, `Blueprint.to_dict`). The blueprint schema
covers entities, resources, character position and inventory, markers and
unlocked recipes. It does **not** cover the state that makes Factorio Factorio:

  - belt line contents, per-lane, with sub-tile positions;
  - assembling-machine / furnace crafting progress and fluid boxes;
  - burner fuel remaining and current burn progress;
  - inserter hand contents and swing angle;
  - character mining and crafting-queue progress (`encoders.py:186–191` reads
    both, so they are policy-visible and must round-trip);
  - force production statistics (`PredicateKind.PRODUCED`, `spec.py:160–163`,
    reads these — a snapshot that loses them silently changes success);
  - research progress;
  - the entity handle table and the remembered-observation store (Phase 2.4/2.3),
    without which restored handles resolve `unknown_handle`;
  - `RewardAccountant._high_water` and `._potential` (`rewards.py:63–64`) —
    Python-side, so restore must include a Python-side hook.

**Verification.** `world_digest` already returns *sorted lines, not a hash*
(ledger, Phase 3.6) precisely so a mismatch says *what* leaked. Reuse it: a test
that runs `snapshot → k steps → digest_A`, then `restore → same k steps →
digest_B`, and asserts `digest_A == digest_B` line for line, across all six
families and `k ∈ {1, 30, 300}`. Anything not captured makes search operate on a
different MDP than the one being scored, which is the worst possible failure
mode — silently optimistic search.

**Phase.** 10.1 already calls for "full-worker save checkpoints for experiment
continuation and branching" with the acceptance criterion "exact restoration is
not conflated with reconstructed scenes." That is this work, and everything in
ch. 9 depends on it, so it should be **pulled forward to Phase 8**.

**Impact.** Unlocks recommendations 4, 5, 7, 8, 9 and 16. Without it none of
them exist.

**Cost/risk.** This is the largest single item in this review — a week of
runtime work plus the digest test suite. The honest risk is that some engine
state is not reachable from the Lua API at all (fluid-box internals, belt
sub-tile offsets); if so, **document the gap in the action/observation profile**
and treat the snapshot as an approximation with a stated error, per PLAN §2
("Any necessary approximation must be documented in the action profile and
covered by tests"). Do not ship a snapshot whose fidelity is unmeasured.
Factorio's own save/load is exact but requires a process restart — too slow for
per-decision use, but it is the correct **reference** against which to measure
the fast snapshot's fidelity, exactly as Phase 3.3 compares fast reset against
freshly loaded reference workers.

### 7. Frame the planner as hindsight optimization over sampled map completions

**Idea.** §9.9.3, eq. 9.7, p.208 (multiforecast MPC / hindsight optimization):
optimize `m` deterministic forecast scenarios in parallel, each free to choose
its own action sequence, **subject to the constraint that the first action
agrees across scenarios**. §22.6 (p.459) is the POMDP form: determinize the
belief into `m` particle scenarios, reducing the tree from `O(|A|^d|O|^d)` to
`O(|A|^d m)`.

**Repo change.** In `search.py`. Our belief's uncertainty is entirely about the
**unobserved map**, so a scenario is a *completion*: sample `m` plausible worlds
consistent with the current remembered store, plan a deterministic sequence in
each, and take the agreed first action. Because tasks are generated from
declared `Blueprint`s by `RegisteredTask.generate` (`env.py:151`), we can sample
completions from the *same generator* conditioned on what has been observed —
which is the honest, non-privileged version of a belief prior.

**Phase.** 8, and it becomes essential at 9.4 ("previously explored regions
remain subject to stale information") and 9.5.

**Impact.** This is the algorithm the project's architecture was accidentally
built for, and it is the one that makes a search agent's exploration behaviour
non-degenerate: with `m = 1` the planner assumes the unknown map is whatever it
guessed and never plans to look.

**Cost/risk.** Real leakage hazard. Sampling completions from the task generator
means the planner knows the generator's *distribution*. That is defensible — a
human player knows Factorio's world generation too — but it is **assistance**
and must be declared in the profile (PLAN 2.5), disclosed in 8.5's comparison,
and it must sample from the *distribution*, never be handed the realized seed.
Add a test in the spirit of `phase2-gate`'s `privileged_calls_by_agent: 0`: the
planner's RCON client is wrapped so `truth()` and `world_digest` are unreachable
from it.

### 8. Use search as a DAgger teacher, not as the shipped policy

**Idea.** §18.1 (p.355) behavioural cloning maximizes `∏ πθ(a|s)` over expert
pairs but suffers **cascading errors** — the expert never visits the states the
learner ends up in (example 18.2, p.358: the race driver who never drifts onto
the grass). §18.2 (p.358, alg. 18.2) DAgger fixes this by rolling out the
*learner*, labelling those states with the expert, and aggregating.

**Repo change.** New `learn/dagger.py`. Iteration: roll out the current
MaskablePPO policy on the 8-worker `FactorioVecEnv`; sample states from those
rollouts; for each, snapshot, run bounded search (rec. 4/5) on a *separate*
search pool, record `(observation, search_action)`; aggregate; fit by
cross-entropy. This is precisely §13.4's actor-critic-with-MCTS loop
(eq. 13.29–13.33, p.276–277) with the search statistics as the target.

**Phase.** 8.2 ("train transferable skills") — DAgger-from-search is the most
plausible route to a *transferable* skill, because the teacher is available on
unseen layouts whereas hand-scripted solvers are not.

**Impact.** The right use of our compute. Budget: overnight = 8 h = 28,800 s at
202 steps/s = **5.8M engine steps**. At 112 steps per label (rec. 4), that is
~50,000 labels per night; at 900 steps per label (depth-3 branch and bound),
~6,400. 50k labels is enough to correct a PPO-initialised 266k-parameter policy;
it is *not* enough to clone one from scratch, which is why the DAgger form
(correcting a policy) and not the BC form (learning from scratch) is the right
one.

**Cost/risk.** Two hazards.
1. `tasks/reference.py`'s solvers read `env._truth` (`reference.py:96–105`).
   If those ever become the DAgger expert, the learned policy has distilled
   evaluator knowledge. **The search teacher must be observation-only.**
   The existing import-direction test (training never imports
   `tasks.reference`) is exactly the right mechanism; extend it to
   `learn/dagger.py`.
2. §18.2's own caveat: DAgger "is not guaranteed to converge." §18.3 (SMILe,
   p.358–362, alg. 18.3) gives the convergent variant by mixing in the expert
   with decaying weight `β`. If DAgger oscillates, that is the fix.

**PLAN conflict.** PLAN 4.2 forbids scripted solution labels in **ordinary PPO
training**, and this is not ordinary PPO training — it is a separate, declared
mode. But the Phase 4 release result must be produced *without* it, and the
DAgger agent must appear in 8.5's comparison as its own system with its
assistance disclosed. Do not let a DAgger checkpoint be published as "the
compact RL baseline."

### 9. Reuse the search tree across decisions

**Idea.** §22.5, p.459: "The algorithm does not need to be reinitialized with
each decision. The history tree and associated counts and value estimates can be
maintained between calls. The observation node associated with the selected
action and actual observation becomes the root node at the next time step."

**Repo change.** In `search.py`, keep `N` and `Q` keyed by *history* across
`step` calls; on committing action `a` and receiving observation `o`, reroot at
the `(a, o)` child rather than discarding. Only valid because our world is
deterministic and paused — the child node's state *is* the new state.

**Phase.** 8.

**Impact.** For a receding-horizon planner at depth `d`, rerooting recovers
roughly a `1/|A|` fraction of prior work for free. At `|A| = 14` that is a ~7%
saving on its own; the real gain is that value estimates accumulate over the
episode instead of restarting from a rollout each decision.

**Cost/risk.** History keys grow without bound over a 300-decision episode.
Prune subtrees not descended from the new root. Memory is not the constraint;
the bookkeeping bug where a stale subtree survives a reroot is.

### 10. Move to an explicit belief over machine state, before recurrence

**Idea.** §19.2 (p.380) discrete state filter; §21.1 eq. 21.3 (p.429) — the
QMDP idea generalized: `U(b) = max_a ∫ Q(s,a) b(s) ds`, "approximated through
sampling," which is exactly what a neural `Q` plus sampled particles gives you
without any tabular iteration. §23.1 (p.471) finite-state controllers: a policy
with its own finite internal node set, "more computationally tractable than
belief-point methods."

**Repo change.** Three options, in increasing cost, all cheaper than an LSTM:
1. **Enrich the filter's sufficient statistics** (cheapest, do this first).
   `encoders.py:174–175` gives a remembered entity exactly two uncertainty
   features: `is_remembered` and `log(age)`. Add: last-seen `status`,
   last-seen fill fraction, **and a decayed-confidence scalar** whose decay
   rate is per entity type (a wall's belief never decays; a furnace's decays
   fast). This is a hand-specified `O(o|a,s')` and it is honest — it uses only
   public game mechanics, which PLAN §3 permits for masks and should equally
   permit for features.
2. **Particles over the small uncertain part** (§19.6, p.390, alg. 19.6):
   maintain `m` samples of "is the remembered furnace still fed", resampled by
   observation weight. Watch §19.5's particle deprivation (p.393) and use
   adaptive injection (§19.7, alg. 19.9, p.395) with the paper's recommended
   `α_fast = 0.1, α_slow = 0.001, ν = 2`.
3. **A finite-state controller head** (§23.1, alg. 23.1, p.472): a small
   discrete node set `X` with learned `ψ(a|x)` and `η(x'|x,a,o)`. With `|X| = 8`
   this is a few hundred parameters of *memory* rather than a 256-unit LSTM.

**Phase.** 1 in Phase 7 (persistent factories, where staleness first bites);
2 and 3 in Phase 8.

**Impact.** Option 1 is where the value is. Phase 7 is the first time an agent
must decide whether a remembered fact is still true; `log(age)/log(3601)` is not
enough information to make that decision, and the policy has no way to learn the
per-type decay because the type index is a single scalar
(`encoders.py:165`, `type_index / 11`) rather than a one-hot.

**Cost/risk.** Option 1: bump `EXTRACTOR_VERSION`, invalidate checkpoints.
Options 2–3 are research, not engineering.

**PLAN conflict.** PLAN §3 bans recurrent policies "before the basic baseline
works." Option 3 *is* a recurrent policy in substance and must wait for Phase 4
acceptance. Options 1 and 2 are not — a filter maintained outside the network
and fed forward is a belief-MDP policy (§20.1), which is precisely the
distinction the ban should be read against. Record that reading in PLAN.md so
the question is settled once.

### 11. Replace `ent_coef` with a count-based exploration bonus for the sparse
### families

**Idea.** §16.3 (p.323): R-MAX assigns `r_max` to state-action pairs with fewer
than `m` visits (eq. 16.3) and self-loops them (eq. 16.4), so
under-explored pairs are worth `r_max/(1−γ)` — "this exploration incentive
relieves us of needing a separate exploration mechanism." §15.4 (p.305–308):
UCB1's `ρ_a + c√(log N / N(a))` (eq. 15.3), quantile exploration, and
Thompson/posterior sampling, which "does not require careful parameter tuning."

**Repo change.** `learn/train.py:50` sets `ent_coef = 0.01`, an undirected
strategy (§15.3, p.303) that "maintains a constant amount of exploration,
despite there being far more uncertainty earlier." Two families give it nothing
to work with. `repair_belt` (`repair_belt.py:38–47`) pays sparse success plus a
high-water counter that is pinned at zero until the repair is already done, plus
a step cost — so the reward is *negative and constant* over the entire
exploration phase. `restore_power` (`restore_power.py:41–45`) is worse still:
sparse success and step cost, **no shaping component at all**. Undirected
exploration cannot find either goal. Add an optional
`RewardKind.NOVELTY` component in `tasks/spec.py:172` computed in `rewards.py`
from a visit count over a **coarse discretization of the agent's own
observation** (tile-quantized position × a hash of the nearby entity multiset),
with `bonus = c/√N(s̃)`.

**Phase.** 4.4 / 4.5, as one arm of the shaping comparison.

**Impact.** This is the most likely single unblocker for the two repair
families, which are also the families PLAN 4.5 needs ("at least one qualifying
family must involve production or repair").

**Cost/risk.** A novelty bonus is **not** potential-based, so it changes what is
optimal (rec. 2) and must be flagged `policy_invariant=False` and disabled at
evaluation. Also, PLAN 3.5's exploit criteria apply: a novelty bonus paid for
*state* novelty invites the agent to wander; decay `c` on a schedule and prove
in a test that a dismantle/rebuild loop cannot farm it (the count is
monotone-increasing, so it cannot, but assert it).

**PLAN conflict.** None, provided it is logged as its own reward component
(PLAN §3: "Log each reward component individually" — `rewards.py` already
guarantees this by construction).

### 12. Report Phase 7 recovery as a Pareto frontier, with a swept parameter

**Idea.** §14.4, p.289: vary a design parameter, plot the two metrics against
each other, highlight the non-dominated set; example 14.5 (p.292) does exactly
this for safety vs. operational efficiency, and shows the optimized family
*dominating* the threshold-policy families. Exercise 14.5 (p.296) gives the
dominance test for three metrics.

**Repo change.** PLAN 7.4 requires five recovery metrics (time to restore,
production lost, repair resources, damage to other production, final sustained
output). Build the report as a `pandas`-free table plus a frontier computation
in a new `factoriorl/report.py`: compute the non-dominated set over the metric
vector across (policy, seed, intervention) and publish it. Sweep one declared
parameter — the natural one is the **repair-urgency weight** in the reward
config, since `RewardComponent.weight` is already in the manifest
(`learn/train.py:278`).

**Phase.** 7.4, 7.5.

**Impact.** Converts PLAN's prohibition ("Do not combine these into one
undocumented composite score") from a rule into a deliverable. A frontier is
also the only honest way to compare a fast-but-destructive repair against a
slow-but-clean one.

**Cost/risk.** A sweep multiplies evaluation cost by the number of parameter
settings. Five settings × 100 episodes × 3 seeds × 3 minutes/episode is 75
hours. Budget it, or sweep at reduced episode count and report the frontier with
its (wider) intervals.

**PLAN conflict.** None. It implements PLAN §3 and 7.4.

### 13. Add most-likely-failure search to the intervention benchmark

**Idea.** §14.5, p.291–294: convert the problem into an adversarial one where
the adversary picks outcomes to minimize our return while maximizing trajectory
likelihood (eq. 14.15), or, for a defined failure set, maximize likelihood
subject to reaching failure (eq. 14.16). "We can play back the most likely
failure trajectory and gauge whether that trajectory merits concern."

**Repo change.** Phase 7.3 injects interventions from a declared, evaluator-only
schedule. Add an **adversarial intervention search**: over the declared,
feasible intervention space (belt defect location, power disconnect point,
supply reduction magnitude), search for the intervention the current policy
handles worst. Implement in `search.py` over `session.define_scenario` +
snapshot, scoring by the recovery metric vector.

**Phase.** 7.3 / 7.5, and it is the natural engine for 7.5's structural holdout
generation — *except* see the risk.

**Impact.** Directly produces PLAN 4.5's and 7.4's "failure examples," and turns
"unresolved failures" in the report from an accident into a measurement.

**Cost/risk.** **The trap PLAN 7.5 already names**: "Test factories are not
selected using agent performance." An adversarially-selected intervention *is*
selected using agent performance, so it must live in a **separate diagnostic
suite**, never in the test split. Also §14.5's fault-generation constraint maps
to PLAN 7.3's "fault generation preserves a documented feasible recovery path" —
the adversary must be constrained to the feasible set, or it will find
unrecoverable faults and report them as policy failures.

**PLAN conflict.** Only if misused as above. Say so in the ledger entry.

### 14. Use importance sampling for the rare-failure and rare-success regimes

**Idea.** §14.2, p.287, eq. 14.12: sample from a proposal `P'` and weight by
`P(τ)/P'(τ)`; eq. 14.13 gives the optimal proposal. The book's demonstration
(fig. 14.3, p.288): importance sampling converged within `10⁴` samples where
direct sampling needed `5×10⁴`, generating 939 collisions vs. 246.

**Repo change.** In `learn/train.py`'s evaluation path. Two applications:
1. **Early training**, when success is rare: bias the *initial state
   distribution* — the layout family and generator parameters drawn in
   `env.reset` (`env.py:148–151`) — toward configurations the policy is closer
   to solving, and reweight. `SeedPlan.generator_rng` already makes the
   proposal reproducible.
2. **Phase 7 recovery**, where a catastrophic outcome is rare by design.

**Phase.** 7.4 for the rare-failure case; optionally 4.2 for training diagnosis.

**Impact.** Cuts evaluation cost for rare-outcome metrics by roughly the factor
the book demonstrates (~5×), on a 24 ms/step budget where that is hours.

**Cost/risk.** Serious. Importance sampling for a *headline success rate* would
mean the published number is no longer a plain count over a frozen split, and
PLAN's acceptance criterion ("100 held-out episodes per family per training
seed") is deliberately a plain count. **Use importance sampling only for
supplementary rare-event metrics; never for the acceptance measurement.**
Eq. 14.12's footnote 6 also requires `P'` to assign nonzero likelihood
everywhere `P` does — a proposal that zeroes out a layout family silently biases
the estimate.

### 15. Make the "planning model vs. evaluation model" split explicit and
### stress-test across it

**Idea.** §14.3, p.288: "We typically want our planning model … to be relatively
simple to prevent overfitting to potentially erroneous modeling assumptions …
our evaluation model can be as complex as we can justify. … A simpler planning
model is often more robust to perturbations in the evaluation model." Fig. 14.4
(p.289) plots performance as the evaluation model deviates. Eq. 14.14 gives
robust DP: `U_{k+1}(s) = max_a min_i (R_i + γ Σ T_i U_k)`.

**Repo change.** We *have* this split and have not named it. The planning model
is the coarse one the policy sees: 30-tick decisions (`spec.py:219`), a 14–40
action catalog, a 32-entity belief. The evaluation model is Factorio at 60 UPS.
Name them in the run manifest (`manifest.py`) as distinct `planning_profile` and
`evaluation_profile` entries, and add a **robustness sweep** to the Phase 6
release artifacts: hold the policy fixed and vary `decision_ticks` (15/30/60)
and the sensor radius (16/32/48) *at evaluation only*, publishing the resulting
curve. That is fig. 14.4 for FactorioRL.

**Phase.** 6.2 (release artifacts), 10.3 (observation ablations) — 10.3 already
asks for this and §14.3 says why it belongs in the *release*, not a later
research branch.

**Impact.** A policy that collapses when `decision_ticks` moves from 30 to 31 has
overfit the discretization, and no other check in the plan would find it. This is
cheap insurance for a benchmark that other people will vary.

**Cost/risk.** Three settings × two axes × 100 episodes is 600 extra episodes per
family. At 41.6 steps/s and ~150 steps/episode, ~36 minutes per family per axis.
Affordable.

**PLAN conflict.** None; it makes 10.3 partly a release gate rather than
deferred work.

### 16. Prefer a heuristic evaluation function over rollouts at low simulation
### counts

**Idea.** §9.6 example 9.8, p.196: on 2048, heuristic board evaluations
"tend to be efficient and can more effectively guide the search when run counts
are low," and "lookahead rollout evaluations take about 18 times longer than
heuristic evaluations." §9.7 (p.197) formalizes the heuristic-guided version.

**Repo change.** In `search.py`, make the leaf evaluator pluggable and default to
**the MaskablePPO value head**, not a rollout. A forward pass costs microseconds
on the GPU; a depth-8 rollout costs 8 × 24 ms = 192 ms. This single choice
changes the per-decision budget for depth-`d` search by a factor of `d`.

**Phase.** 8.

**Impact.** Multiplies the affordable simulation count `m` by roughly the rollout
depth. Combined with rec. 4 this is what makes an online search policy viable at
all on this hardware.

**Cost/risk.** The critic is only as good as its training; §9.7's guarantee needs
an *upper* bound and a learned critic is neither an upper nor a lower bound. Use
it as the forward-search leaf value `U` (§9.3, alg. 9.2, where `U` is just "an
estimate"), not as the branch-and-bound `Q̄`.

---

## Using the pausable simulator for search and teacher data

This is the section that matters most, because the capability is unusual and we
are not using it.

### What we have, measured

| Primitive | Cost | Source |
|---|---|---|
| One fused `step` (act + 30 ticks + observe) | **24.03 ms** (p95 26.55) | ledger, Phase 3 |
| Steps/s, one worker | 41.6 | ledger, Phase 3 |
| Steps/s, eight workers | 202 | `vecenv.py:13` |
| Reset to a blueprint | **6.21 ms** (p95 7.69) | ledger, Phase 3 |
| Snapshot / restore to arbitrary state | **does not exist** | `protocol.py:47` |

A 6.21 ms reset is the striking number. It exists because Phase 3.2 chose
programmatic blueprints over map generation. The same mechanism —
`session.define_scenario(payload, digest)` then `session.reset(blueprint_hash)`
(`env.py:86–92`, `env.py:156`) — **is already a state-restore**, for the one
state that happens to be an episode start. Recommendation 6 is the observation
that generalizing it is the highest-leverage runtime work in the project.

### The algorithm: masked, determinized, receding-horizon search

Concretely, for a decision at state `s_t`:

```
plan(s_t, d, m, budget):
  snap = worker.snapshot()                     # ~6 ms, target
  root = reroot_or_new(history)                # §22.5, p.459
  while engine_steps_used < budget:
      worker.restore(snap)                     # ~6 ms
      simulate(root, depth=d)                  # d x 24 ms
  return argmax_a Q(root, a)                   # or pi_MCTS ∝ N(root,a)^η, §13.4 eq 13.28
```

- **Selection** inside the tree: probabilistic UCB (§13.4, eq. 13.27, p.276),
  `Q(s,a) + c·πθ(a|s)·√N(s) / (1 + N(s,a))`, with the prior supplied by our
  MaskablePPO actor. Restricted to the legal set from `env.action_masks()`
  (`env.py:124`) — masking is a `|A|` reduction applied at *every* node, which
  in a depth-`d` tree is a `(|A|/|A_legal|)^d` saving. On `repair_belt` from a
  position where no placement precondition is met, `|A_legal|` drops from 14 to
  9; at depth 4 that is a 5.8× smaller tree, for free.
- **Expansion**: one node per simulation, per alg. 9.6 (p.190).
- **Evaluation** at the leaf: the critic head (rec. 16), not a rollout.
- **Determinization** of the unobserved map: `m` scenario completions sampled
  from `RegisteredTask.generate`'s distribution conditioned on the remembered
  store (§22.6, p.459; §9.9.3, eq. 9.7, p.208), with the first action
  constrained to agree across scenarios.
- **No observation branching.** Our transitions are deterministic, so
  `|O| = 1` per scenario and §22.2's `O(|A|^d|O|^d)` (p.454) collapses to
  `O(|A|^d)` per scenario, `O(|A|^d m)` overall — exactly DESPOT's bound.

### Per-decision cost, on this machine

`|A| = 14` (`repair_belt`), one worker at 24.03 ms/step, restore assumed 6 ms.

| Configuration | Engine steps / decision | 1 worker | 8 workers |
|---|---|---|---|
| 1-step lookahead + policy rollout, `d=8`, `m=1` (§9.2) | 112 | 2.7 s | 0.55 s |
| 1-step lookahead + critic leaf, `d=1` | 14 | 0.34 s | 0.07 s |
| Forward search `d=2` (§9.3) | 196 | 4.7 s | 0.95 s |
| Forward search `d=3` | 2,744 | 66 s | 13 s |
| Branch and bound `d=3`, ~3× pruning (§9.4) | ~900 | 22 s | 4.3 s |
| MCTS `m=100`, `d=10`, critic leaf (§9.6) | 1,000 | 24 s | 4.8 s |
| MCTS `m=400`, `d=10`, `m_scenarios=4` (§22.6) | 16,000 | 385 s | 77 s |

Read against a 300-decision `repair_belt` episode and a 100-episode evaluation:

- **Only the first two rows are online policies.** Everything at or below
  "forward search `d=3`" costs more than an hour per *episode* on eight workers.
- **Everything else is a teacher.** An 8-hour overnight run buys 5.8M engine
  steps: **51,000** labels at 112 steps each, **6,400** at 900, **5,800** at
  1,000. That is a DAgger correction budget, not a cloning budget — which is why
  recommendation 8 corrects a PPO-initialised policy rather than cloning from
  scratch.
- **Search cost scales with `|A|`, so it scales with the task.** `navigate`'s
  9-action catalog searches `(9/14)^d` cheaper than `repair_belt`'s; the full
  `primitive-v1` catalog (~40 templates, `catalog.py:119`) is unsearchable past
  depth 2. Any future catalog growth should be costed against this table.

### How it integrates with `vecenv.py` — and why it must not

`FactorioVecEnv` (`vecenv.py:38`) implements SB3's `VecEnv`, whose contract has
no notion of speculative stepping or undo. Its 8 workers are the *rollout*
workers and their episode streams are deliberately disjoint
(`vecenv.py:66`: `env._episode_index = index * 100_000 - 1`). Putting search
inside `step_wait` would corrupt on-policy data collection — PPO's importance
ratio (§12.5, p.257) assumes the behaviour policy is `πθ_old`, and speculative
steps executed on the same worker would silently change the state the returned
transition came from.

Therefore:

- **Do not modify `vecenv.py`.** Add `src/factoriorl/search.py` with its own
  `WorkerPool` (`pool.py`) — a second pool of `k` search workers, sized from the
  same profiling PLAN 4.3 requires. On this machine, 8 rollout + 4 search
  workers is the natural split; the ledger already notes the commit-limit
  ceiling and the launch-storm stagger (`vecenv.py:69`).
- **Search parallelizes across `m` scenarios, not across tree nodes.** Assign
  one scenario per search worker; the "first action must agree" constraint
  (eq. 9.7) is enforced in Python after joining. This avoids tree-parallel
  MCTS's virtual-loss machinery entirely.
- **The search worker must be observation-only.** Wrap its `RCONClient` exactly
  as `gate_phase2` does, so `truth()`, `world_digest` and raw Lua are
  unreachable and the count of privileged calls is a *measurement*, not an
  assertion.
- **Deterministic replay is the fallback if snapshot fidelity is imperfect.**
  A search worker can be driven to `s_t` by `reset(blueprint) + replay(a_1..a_t)`
  at `t × 24 ms` — 3.6 s at `t = 150`. Too slow per decision, fine for
  generating a few thousand teacher labels overnight, and it is the *reference*
  against which snapshot fidelity is measured.

---

## Do NOT do this

1. **Do not build a tabular belief, alpha vectors, or point-based value
   iteration.** §20.3 (p.411), §21.4 (p.431) and §21.5 assume you can enumerate
   `S`. A Factorio surface has ~`4225` tiles in the sensor region alone. These
   chapters are for understanding, not implementation.
2. **Do not use a Kalman/EKF/UKF filter** (§19.3–19.5, p.383–390). Our state is
   discrete and combinatorial; the linear-Gaussian assumption is not
   approximately true and the filter would be a well-formed answer to a
   different question.
3. **Do not use sparse sampling with `m > 1`** (§9.5, p.187). Our transitions
   are deterministic; every extra sample is an identical simulation at 24 ms.
4. **Do not adopt QMDP as the agent.** §21.1 (p.429) is explicit: QMDP "assumes
   perfect observability after the first step" and therefore "can poorly
   approximate the value of information-gathering actions." Factorio's whole
   partial-observability story *is* information gathering — walking over to look
   at a machine. QMDP is useful only as an upper bound for branch and bound.
5. **Do not use inverse reinforcement learning** (§18.4–18.6, p.361–369) to
   recover a reward from demonstrations. We declare our reward (`tasks/spec.py`)
   and PLAN §3 requires that declaration be logged per component. Learning it
   back would make the benchmark unfalsifiable.
6. **Do not let a search or DAgger agent's number be published as the compact-RL
   baseline.** PLAN 4.5's threshold is for the learned policy. A hybrid agent is
   a different system and belongs in 8.5's comparison with its assistance
   disclosed.
7. **Do not let the search planner read `truth()`.** Its bounds, priors and
   scenario completions must come from declared reward weights, the policy's own
   critic, and the *distribution* of the generator — never the realized seed,
   the blueprint, or the fault schedule.
8. **Do not use importance sampling for the acceptance measurement.** §14.2 is
   for supplementary rare-event metrics only. The 80%-over-100-episodes gate
   stays a plain count on a frozen split.
9. **Do not adversarially select the held-out test split.** §14.5 finds
   most-likely failures; PLAN 7.5 forbids selecting test factories by agent
   performance. Both are right; keep them in separate suites.
10. **Do not add actions to make search stronger.** Every ch. 9 cost is
    exponential in `|A|`. The catalog's boundedness is a load-bearing design
    decision, not a limitation to be relaxed.
11. **Do not treat a rollout as a lower bound.** §22.7 (p.464): "A rollout is
    not guaranteed to produce a lower bound … because it is based on a single
    trial up to a fixed depth." If branch-and-bound pruning uses one, the search
    is no longer equivalent to forward search and the profile must say so.
12. **Do not ship a snapshot whose fidelity is unmeasured.** A restore that
    silently loses belt contents or crafting progress makes search optimistic in
    a way no downstream test would catch.

---

## Open questions

1. **Can Factorio's Lua API reproduce belt sub-tile item positions, fluid-box
   internals, and inserter swing angle well enough for an exact snapshot?** If
   not, what is the measured divergence over 30/300/3000 ticks, and does it
   change any success predicate? This determines whether rec. 6 yields exact
   search or approximate search, and the answer must be measured before Phase 8
   is planned. (Method: `snapshot → k steps → world_digest` vs.
   `restore → k steps → world_digest`, line-diffed.)
2. **What is the real per-scenario belief prior?** Rec. 7 proposes sampling map
   completions from `RegisteredTask.generate`'s distribution. Is that assistance
   we are willing to declare, or should the planner use a weaker,
   task-agnostic prior (uniform over unexplored tiles)? The weaker prior is more
   honest and much worse; PLAN 10.3's ablation framework is where this gets
   settled.
3. **Does the entity-list truncation actually bind at Phase 7 scale, and at what
   entity count?** Measure before redesigning: instrument `encoders.encode` to
   log how often the 32-slot budget is saturated and what fraction of
   task-relevant entities are evicted, on a Phase 7 persistent scene.
4. **Is a finite-state controller (§23.1) competitive with an LSTM at a tenth
   the parameters on our tasks?** This is a genuinely open research question in
   our setting and a good Phase 8 experiment, but it must not be attempted
   before Phase 4 acceptance.
5. **How much of the Phase 4 shortfall is the pooled-CNN defect (rec. 1) versus
   the reward gradient?** Cheapest discriminating experiment: retrain
   `mine_smelt` with only the relative-nearest-resource feature added and no
   architecture change. If success moves, rec. 1 is the cause.
6. **What is the right `decision_ticks`?** §9.1 (p.181, example 9.1) makes the
   depth/horizon trade-off concrete. 30 ticks is inherited, not derived.
   Rec. 15's sweep would answer it — and a longer interval both shortens the
   episode (fewer decisions to credit) and makes each search node cover more
   game time.
7. **Does tree reuse across decisions (rec. 9) survive our `reset` semantics?**
   Rerooting assumes the committed action's child node's state is the new state.
   That holds only if snapshot/restore is exact (question 1) — the two are
   coupled.
8. **Is there a defensible upper bound `Q̄` tighter than best-action-best-state
   (§21.1, eq. 21.2)?** For the navigation-flavoured families a
   max-walking-speed bound is obviously tighter; for `repair_belt` and
   `restore_power` it is not obvious that anything admissible and non-trivial
   exists. Without one, branch and bound (rec. 5) degenerates to forward search.
9. **What parameter should the Phase 7 Pareto sweep vary?** Rec. 12 suggests a
   repair-urgency reward weight, but §14.4's examples sweep *policy* parameters
   (thresholds), not reward weights. Sweeping the reward weight means retraining
   per point; sweeping a decision threshold at inference does not. Is there an
   inference-time knob in our architecture that trades restore-time against
   collateral damage?
