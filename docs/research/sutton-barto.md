# Sutton & Barto against FactorioRL: a whole-program design review

This is a design review of the FactorioRL experiment — the environment, the objective, the
rollout architecture, the curriculum and the twelve-phase arc — read against *Reinforcement
Learning: An Introduction*, 2nd edition. It is not a book summary and it is not a bug report
on a current failure. It is an argument about which parts of the program stand on solid
theoretical ground, which do not, and what to build now so that Phases 7–9 are additions
rather than rewrites.

## Which copy, and how to read the citations

**This document supersedes an earlier version of this file that was written against an
incomplete draft** (`D:\Books for RL\SuttonBartoIPRLBook2ndEd.pdf`, 352 pages, fifteen
chapters, Chapter 10 an empty stub). That copy is missing roughly a quarter of the finished
book, including every chapter this review most needs. The earlier version said so honestly
and cited primary sources where the draft failed — but it also carried section numbers, page
numbers and at least one whole citation forward from the draft's numbering, and several of
those do not survive contact with the complete text. The corrections are listed below and
should be treated as errata against anything already built on that document.

All citations here are to **`bookdraft2017nov5.pdf`** — the complete
second-edition draft of 5 November 2017: 445 PDF pages, seventeen chapters, References at
printed p. 395. Citations give the **printed page**; **PDF page = printed page + 18**
(printed p. 215 = PDF page 233). Where a claim rests on a result the book cites rather than
proves (Ng–Harada–Russell; Wiewiora), that is said explicitly.

Repository files referenced: `src/factoriorl/rewards.py`, `env.py`, `encoders.py`,
`catalog.py`, `vecenv.py`, `learn/train.py`, `learn/policy.py`, `tasks/spec.py`,
`tasks/families/*.py`, `tests/unit/test_tasks_and_rewards.py`, `docs/LEDGER.md`, `PLAN.md`.

### Previous claims that are wrong or unsupported by the complete text

1. **"Potential-based reward shaping is not in this book."** False. §17.4 *Designing Reward
   Signals*, p. 386, names it: "Wiewiora (2003) showed that this initialization is equivalent
   to the more complex 'potential-based shaping' technique for changing rewards described by
   Ng, Harada, and Russell (1999)." §3.2 footnote 6, p. 43, forward-references §17.4 for
   exactly this. The theorem is still not proved in the book, but the technique is named,
   endorsed with a caveat, and given a strictly better alternative (see R8).
2. **"Options and temporal abstraction are absent."** False. §17.2 *Temporal Abstraction via
   Options*, pp. 380–382, gives the full modern formulation, the option model equations
   (17.2) and (17.3), the hierarchical Bellman equation (17.4), option value iteration
   (17.5), and how to learn option models as GVFs.
3. **The book's option is not (initiation set, policy, termination condition).** The previous
   version asserted that triple. §17.2, p. 381: "We define a pair of these as a generalized
   notion of action termed an *option*… ω = ⟨π_ω, γ_ω⟩" — an internal policy and a
   **state-dependent termination function**, with no initiation set. Availability appears
   separately as Ω(s), "the set of options available in state s" (eq. 17.4, p. 382). That is
   an action mask, and it changes what Phase 8 should build (R14).
4. **"§14.4, p. 288 (Elevator Dispatching), the SMDP formulation with e^{−β(t₂−t₁)},
   Bradtke & Duff 1995."** This citation does not exist in the complete text. The strings
   "semi-Markov", "SMDP", "Bradtke and Duff" and the elevator case study appear nowhere in
   the body; Chapter 14 is *Psychology* and Chapter 16's case studies are TD-Gammon, Samuel,
   Watson, memory control, Atari, Go, web services and thermal soaring. The elevator and
   job-shop studies were dropped in the second edition. Anything in the previous document
   that leaned on that citation — the SMDP discounting recommendation and the manager-policy
   recommendation — was resting on the incomplete draft's chapter numbering. The correct
   in-book authority is §17.2 eqs. (17.2)–(17.3), p. 381–382, and §12.8, p. 253. Semi-MDPs
   are acknowledged only in a bibliographic note, §17.2 notes, p. 392 ("classical work on
   Semi-MDPs (e.g., see Puterman, 1994)").
5. **"§11.3, p. 260: 'When the policy is approximated, we generally have to abandon the
   discounted-reward setting…'"** Not in this text at that location or in that form. The
   material is §10.4 *Deprecating the Discounted Setting*, pp. 205–206, and its actual
   argument is sharper and narrower: in a **continuing** problem the average of discounted
   returns is exactly r(π)/(1−γ), so "The discount rate γ thus has no effect on the problem
   formulation. It could in fact be zero and the ranking would be unchanged" (p. 205). The
   sentence that matters for FactorioRL is the one the previous version did not have:
   "**The discounted case is still pertinent, or at least possible, for the episodic case**"
   (p. 206). The previous recommendation over-reached for Phase 7.
6. **"§11.3, pp. 261–262: R-learning assumes ergodicity and no episodes."** Right in
   substance, wrong in location. Average reward and differential semi-gradient Sarsa are
   §10.3, pp. 202–204; ergodicity is stated at p. 202; R-learning is named in the
   bibliographic note at p. 207.
7. **"§9.6, p. 249: mean-squared error ≤ (1−γλ)/(1−γ) × minimum possible error."** In this
   text §9.6 is *Nonlinear Function Approximation: Artificial Neural Networks*, p. 182. The
   bound is **Eq. (9.14), p. 169**, and it is `VE(w_TD) ≤ 1/(1−γ) · min_w VE(w)`, proved for
   the **continuing** case with linear function approximation; p. 170 notes only that "For
   episodic tasks, there is a slightly different but related bound (see Bertsekas and
   Tsitsiklis, 1996)." The λ-dependent form belongs to Chapter 12, not Chapter 9.
8. **"§9.5, Fig. 9.12, p. 248: 'in all cases, performance became much worse as λ approached
   1.'"** This sentence is not in the complete text. The correct, and stronger, replacement is
   §12.13 *Conclusions*, p. 261 — quoted in full in R6 — which says the opposite is true for
   long episodes and only warns against λ ≈ 1.
9. **"§9.1, p. 227: 'the flexibility of the function approximator is thus a scarce
   resource.'"** Not in this text. The only occurrence of "scarce" is p. 261, about data. The
   argument the previous version wanted is §9.2, p. 162: "By assumption we have far more
   states than weights, so making one state's estimate more accurate invariably means making
   others' less accurate. We are obligated then to say which states we care most about."
10. **Chapter 2 and Chapter 8 section numbers were all off.** Optimistic initial values is
    §2.6, pp. 26–27 (not §2.5, p. 41); UCB is §2.7, pp. 27–28 (not §2.6, p. 42), and its
    verdict is "In these more advanced settings the idea of UCB action selection is usually
    not practical" (p. 28), not "there is currently no known practical way". Dyna-Q+'s κ√τ
    bonus is §8.3, p. 138 (not p. 205); trajectory sampling is §8.6, pp. 144–146 (not
    pp. 213–214); heuristic search is §8.9, pp. 150–151 (§8.7 is RTDP, §8.8 is *Planning at
    Decision Time*). **§8.10 Rollout Algorithms and §8.11 Monte Carlo Tree Search, pp.
    152–155, do not exist in the draft at all** and change the model-based recommendation
    materially (R16).
11. **"§15.5, p. 309: 'shaping' in this book means curriculum."** Half right, and the wrong
    half is load-bearing. The Skinnerian/curriculum sense is §14.3, pp. 299–300 and §14.7,
    p. 307; the bibliographic note at p. 309 explicitly flags that "Ng, Harada and Russell
    (1999) used the term shaping in a sense somewhat different from Skinner's." Both senses
    are present, and the reward-engineering sense is the one at §17.4, p. 386.
12. **"§7.2, pp. 172–174 defines the λ-return; Fig. 7.4 is the weighting picture."** The
    λ-return is **§12.1, p. 236**; §7.2 is *n-step Sarsa*, p. 119. n-step TD is §7.1,
    pp. 115–118, with Figure 7.2's intermediate-n result at p. 118.
13. **Chapter 3 pages were ~15 off throughout.** §3.2 *Goals and Rewards* is pp. 42–43,
    §3.3 *Returns and Episodes* pp. 43–44, §3.4 *Unified Notation* pp. 45–46. The content
    the previous version quoted is correct; the page numbers were not.

The one previously-unsourced claim that the complete text **confirms** is the substance of
the deadly triad, §11.3, p. 215 — and it confirms it in a form that changes the verdict for
this project (§3 below).

---

## 1. Shaping audit

### 1.1 Verdict on the current implementation: correct, and correct for the right reason

The terminal-state fix in `rewards.py` and `env.py` is **right, and its truncation branch is
right too** — but the truncation branch is right only because of a coupling in `vecenv.py`
that nothing currently tests. That coupling is the one thing in this section that needs work.

**Termination.** `phi_next = 0.0 if terminated else value` is exactly what the book requires,
and there are three independent statements of it:

* §3.4, pp. 45–46: episode termination is "the entering of a special absorbing state that
  transitions only to itself and that generates only rewards of zero."
* §3.5, p. 46, immediately after eq. (3.12): "**Note that the value of the terminal state, if
  any, is always zero.**" A potential function stands in for a value estimate; Φ(terminal)
  must be 0.
* §12.8, p. 253, which is the cleanest form: under state-dependent discounting
  γ: S → [0,1], "A terminal state just becomes a state at which γ(s) = 0 and which
  transitions to the start state." Setting γ(s_T) = 0 in `γΦ(s') − Φ(s)` *is* the fix.

`env.py` computes `terminated = succeeded or self._failed()` **before** calling
`accountant.step`, so the flag describes s′ and not the previous transition. That ordering is
required and is present.

**Truncation.** Not zeroing Φ on truncation is correct, and the reason is worth writing down
because it is not obvious and because a future refactor could break it silently. Under
potential-based shaping the shaped optimal value function is V′(s) = V(s) − wΦ(s) — the book
records this fact through Wiewiora's equivalence at §17.4, p. 386 (see R8). SB3's PPO
bootstraps a time-limit truncation by adding `γ · V̂(s_T)` to the final reward, and the
critic it queries is the *shaped* critic. So over a truncated episode:

```
  Σ_t γ^t · w(γΦ(s_{t+1}) − Φ(s_t))  +  γ^T · V′(s_T)
= w(γ^T Φ(s_T) − Φ(s_0))             +  γ^T V(s_T) − w γ^T Φ(s_T)
= γ^T V(s_T) − w Φ(s_0)
```

The path-dependent `γ^T Φ(s_T)` cancels **exactly** against the shaped bootstrap. Invariance
survives truncation. Had the code zeroed Φ on truncation instead, the total would carry a
surviving `− w γ^T Φ(s_T)`, which *is* path-dependent — so the asymmetry in the current code
is not a stylistic choice, it is the only correct one. §3.3, p. 43, is the licence for the
distinction: an episode "ends in a special state called the terminal state, followed by a
reset"; a budget cut is not that state.

**The untested coupling.** The cancellation above depends entirely on the bootstrap
happening. `FactorioVecEnv.step_wait` sets `info["TimeLimit.truncated"] = truncated and not
terminated` and preserves `terminal_observation`; SB3's `DummyVecEnv` does the same on the
single-worker path. Nothing asserts either. If a later change drops that key — a custom
wrapper, a `SubprocVecEnv`, a Monitor ordering change — the shaping silently becomes
policy-dependent again, rewarding *truncating in a high-potential state*, and no test fails.
This is the same failure mode as the original bug, one layer out.

**Reset-time seeding.** Correct. `FactorioEnv.reset` calls `accountant.reset(self._observation,
self._truth)` after the scene is installed, observed, and truth refreshed, so `_potential[name]`
is Φ(s₀) of the *new* episode; `_high_water[name]` likewise. `FactorioVecEnv.step_wait` and
`DummyVecEnv` both re-enter that path on `done`. There is no cross-episode carry and no stale
`previous` on the first step. One residual: if `truth["markers"]` is momentarily absent at
reset, `_measure` returns 0.0 and the episode's shaping constant is silently 0 rather than
−wΦ(s₀). Not a correctness bug, but it degrades without a signal.

**The existing test.** `test_potential_shaping_telescopes_to_a_policy_independent_constant`
runs two Φ paths of equal length from one s₀ and asserts both sum to −w·Φ(s₀). That is the
right test. It could be strengthened cheaply: the two paths currently have the *same* length
(three steps), so the test does not exercise the T-dependence that the bug produced. Use
different lengths.

### 1.2 Remaining reward defects

**(a) `rewards.GAMMA` is a module constant, `TrainConfig.gamma` is a config field, and
nothing ties them.** They both read 0.99 today by coincidence. `train()` passes
`gamma=config.gamma` to `MaskablePPO` and never touches the accountant. The manifest's
`reward` block records `shaping_enabled`, a component digest, and per-component weights — it
does not record the shaping γ at all, so a run where the two disagree is not even detectable
after the fact. Section 2 below argues γ must move per task; the first such run breaks
invariance silently. Minimal fix in R1.

**(b) `RewardComponent.shaping` is never read.** `RewardAccountant.step` branches on
`component.kind` and `self.shaping_enabled` only. The flag *is* written into the run manifest
in two places (`train.py`'s `reward.components`, and `TaskSpec.to_dict()`), so a manifest can
assert a component is not shaping while the accountant treats it as shaping. `deliver`'s
`delivered` component carries `shaping=False`, which is true; every high-water component
carries the default `True`, which is also true; `step_cost` carries the default `True`, which
is **false** — see (c). The field is one edit away from being either meaningful or gone.

**(c) `STEP_COST` is evaluated after the `shaping_enabled` gate, so PLAN 4.4's ablation moves
two variables.** §3.2, p. 42, is unambiguous that a per-step cost is the task, not a hint:
"In making a robot learn how to escape from a maze, the reward is often −1 for every time step
that passes prior to escape; this encourages the agent to escape as quickly as possible."
That is the same paragraph that introduces the reward hypothesis. Under `shaping=False` today
the agent has no time pressure at all, so the "sparse" arm of 4.4 is a *different task*, and
4.4's own acceptance criterion — "The report distinguishes faster learning from changed task
difficulty" — cannot be met by the comparison it asks for.

**(d) High-water components are monotone but unbounded, and on two families the subgoal
outweighs the goal.** `HIGH_WATER` pays `weight × (max-ever − initial)` of a raw counter with
no cap anywhere in `RewardComponent`. Monotone is not bounded. Measured against a sparse
success weight of 1.0 in every family:

| Task | Component | Weight | What bounds it | Payable **without** success |
|---|---|---|---|---|
| `mine_smelt` | `ore_mined` = inventory iron-ore | 0.05 | nothing; patch holds 9 × 1500 = 13 500 ore, character inventory ≈ 4 000 | **tens to ~200** |
| `supply_furnace` | `ore_carried` = inventory iron-ore | 0.02 | source chest holds 50 | **exactly 1.0** |
| `deliver` | `carried` = inventory iron-plate | 0.02 | source chest holds 40 | **0.8** |

`supply_furnace` is the cleanest case: the maximum shaping obtainable by picking up ore and
never smelting anything is *numerically equal to* the reward for completing the task. On
`mine_smelt` the dominant term in the objective is "hold as much ore as you have ever held,"
and standing on the patch repeating `mine_nearest_5` for 400 decisions pays many times what a
plate pays. This is §3.2, p. 43, almost verbatim: "a chess-playing agent should be rewarded
only for actually winning, not for achieving subgoals such as taking its opponent's pieces
… If achieving these sorts of subgoals were rewarded, then the agent might find a way to
achieve them without achieving the real goal." §17.4, p. 386, restates it as the practical
warning: "augmenting the reward signal with well-intentioned supplemental rewards may lead
the agent to behave very differently from what is intended; the agent may end up not achieving
the overall goal at all."

**(e) The opposite defect on the two families that never succeed.** `repair_belt`'s only
shaping is `CONTAINER_HOLDS(sink, iron-plate)` high-water at 0.1, and its success predicate is
`CONTAINER_HOLDS(sink, iron-plate, at_least=1)` — the shaping fires on the same transition as
success, on an episode that then terminates. `restore_power` declares no shaping at all beyond
the step cost. These are not shaped tasks; they are sparse tasks with a shaping field filled
in. `docs/LEDGER.md` records that three families never succeed even during training, and
attributes it to the (now fixed) `$ahead` placement binding. That was real. This is a second,
independent cause on the same two families, and R8 is the book's own preferred answer to it.

**(f) `deliver`'s module docstring claims a failure predicate that does not exist.** "the
failure predicate catches the degenerate case where the items stop existing at all" —
`SPEC` has no `failure=` argument, and neither does any of the other five families.
`env._failed()` is therefore always `False` on every task, which means `terminated ==
succeeded` everywhere and the terminal-Φ branch has so far only ever been exercised on
success. Phase 7.3's interventions are where failure terminals first appear, and that is the
first time entering a terminal state will *cost* the agent Φ. That is what the theory
prescribes; it should be a decision, not a surprise (open question 1).

**(g) Infrastructure-failure transitions enter the PPO buffer.** `env.step`'s
`InfrastructureFailure` path returns `(obs, 0.0, False, True, {"excluded_from_metrics": True})`
and the module docstring says "the trainer must drop it." Nothing drops it. The vec env sets
`TimeLimit.truncated` on it like any other truncation, so SB3 bootstraps `γ·V̂(s_T)` onto a
reward of zero for an episode that was never an episode. §3.3, p. 43, defines an episode as
ending in a terminal state followed by a reset; a crashed worker produces neither, so the
transition is not a sample from the MDP. `CurveLogger` records `excluded`, but
`final_train_success_rate` averages `curve.rows[-20:]` without filtering it.

### 1.3 Minimal fixes

```python
# rewards.py
@dataclass
class RewardAccountant:
    components: tuple[RewardComponent, ...]
    shaping_enabled: bool = True
    gamma: float = 0.99  # (a) owned here; module GAMMA deleted

    def step(self, observation, truth, succeeded, terminated=False):
        ...
        if component.kind is RewardKind.STEP_COST:  # (c) above the gate: the step
            out[component.name] = -abs(component.weight)  #     cost is the task (S&B 3.2 p.42)
            continue
        if not self.shaping_enabled:
            out[component.name] = 0.0
            continue
        ...
        if component.kind is RewardKind.HIGH_WATER:
            previous = self._high_water.get(component.name, value)
            ceiling = previous if component.cap is None else min(value, previous + ...)
            gain = max(0.0, min(value, component.cap or value) - previous)  # (d)
```

Plus: add `cap: float | None = None` to `RewardComponent` and set it on
`mine_smelt.ore_mined`, `supply_furnace.ore_carried`, `deliver.carried`; surface `cap` in
`TaskSpec.to_dict()`; thread `TrainConfig.gamma → FactorioEnv → RewardAccountant.gamma` and
assert equality once in `train()`; record the shaping γ in the manifest's `reward` block;
either honour `RewardComponent.shaping` or delete it from the dataclass and both manifests.

### 1.4 Tests worth adding (all engine-free, the standard that file already sets)

1. `test_truncation_bootstrap_key_is_present` — step a real `FactorioEnv` through a
   `DummyVecEnv` and a `FactorioVecEnv` to truncation with a fake session, and assert both
   `info["TimeLimit.truncated"] is True` and `info["terminal_observation"]` exists. This is
   the test that protects §1.1's cancellation; without it the truncation branch is correct by
   accident.
2. Strengthen the existing telescoping test to use **two paths of different length**, so it
   exercises the `γ^T` dependence the bug produced.
3. `test_reward_gamma_matches_training_gamma` — build the env `train()` builds with a
   non-default `TrainConfig.gamma` and assert `env.accountant.gamma == model.gamma`.
4. `test_shaping_budget_is_below_the_sparse_weight` — over every registered task, sum
   `weight × cap` across high-water components plus `weight × sup Φ` across potential
   components, and assert the total is strictly less than the sparse-success weight. This is
   the regression test PLAN 4.4 demands ("Any shaping exploitation becomes a task or reward
   defect with a regression test") and it makes `cap` mandatory rather than advisory.
5. `test_step_cost_survives_disabling_shaping`.

---

## 2. Objective and horizon: the return formulation for each phase

The three settings are episodic (§3.3, p. 43), discounted (§3.3, p. 44), and average reward
(§10.3, p. 202). The decision is *not* "discounted or not" in the abstract. Two things
matter here: γ = 0.99 is currently a single global constant applied to budgets that differ by
a factor of 2.7, and Phases 7 and 9 are continuing tasks wearing episodic clothes — but less
so than the previous version of this document claimed.

### Phases 3–6, the six introductory families: episodic and discounted, with γ per task

These terminate, reset in 6.21 ms, and have a natural final time step. §3.3, p. 43, is
exactly this case, and §10.4, p. 206, explicitly protects it: "The discounted case is still
pertinent, or at least possible, for the episodic case." Nothing here needs to change.

γ = 0.99 does. Its effective horizon is 1/(1−γ) = 100 decisions. `mine_smelt`'s budget is
400 decisions with a terminal-only sparse reward. Do the arithmetic the current configuration
implies: success at decision 250 is worth 0.99²⁵⁰ = **0.081**, while the discounted step cost
accrued getting there is 0.001·(1 − 0.99²⁵⁰)/0.01 = **0.092**. The discounted return of a
*successful* long episode is negative. The whole advantage of succeeding at step 250 over
truncating at 400 is 0.087 — against an uncapped `ore_mined` term worth tens. §3.3, p. 44:
"As γ approaches 1, the return objective takes future rewards into account more strongly;
the agent becomes more farsighted." Size γ to the budget (R5).

The cost of doing so is real and citable, which is why R5 says *sweep*, not *assume*.
Eq. (9.14), p. 169: `VE(w_TD) ≤ 1/(1−γ) · min_w VE(w)` — "Because γ is often near one, this
expansion factor can be quite large, so there is substantial potential loss in asymptotic
performance with the TD method." Moving `mine_smelt` from γ = 0.99 to γ = 0.99875 multiplies
that bound's expansion factor from 100 to 800. (p. 170 notes the episodic bound is "slightly
different but related"; the direction of the trade-off is the same.)

### Phase 7, persistent factories: still episodic for scoring; add a rate channel

PLAN 7.1's stage transitions keep the world alive across goal changes, but the episode still
ends and the world still resets, so §10.4's futility argument does **not** apply — its premise
is "an infinite sequence of returns with no beginning or end" (p. 205). Do not switch the
learner to differential Sarsa; §10.3, p. 202, requires ergodicity and no termination, and
FactorioRL has both terminals and resets.

What *is* an average-reward quantity is PLAN 7.4's measurement list: "time to restore
production", "production lost during disruption", "final sustained output" are all rates —
r(π) in the sense of eq. (10.6), p. 202. The reward vocabulary in `tasks/spec.py` cannot
express one: `PRODUCED` is a level and `HIGH_WATER` is a level's maximum. Decide now that a
`RATE` predicate kind exists (items per N ticks over a window), because retrofitting it after
Phase 7 tasks are authored means re-versioning every task (R17).

### Phase 8, manager over skills: use the option model, not a scalar γ

§17.2, eqs. (17.2)–(17.3), pp. 381–382, is the arithmetic. The reward part of an option model
is the discounted sum *along the option's execution*,
`r(s,ω) = E[R₁ + γR₂ + … + γ^{τ−1}R_τ]`, and the transition part folds the elapsed time into
the model itself: `p(s′|s,ω) = Σ_k Pr{S_k = s′, τ = k} γ^k`. The consequence, stated on
p. 382, is that the hierarchical Bellman equation (17.4) and the option value iteration (17.5)
carry **no explicit γ at all** — it lives inside p. And p. 381 flags the split that PLAN 8.3's
acceptance criterion is really about: "discounting is according to γ, but termination of the
option is according to γ_ω." A manager transition therefore needs two numbers the environment
does not currently produce: the discounted reward accrued *inside* the option, and Δt. R14.

### Phase 9, rocket progression: discounted return is an algorithm, not a score

Over a full progression γ^T underflows for any γ < 1, so a discounted return is meaningless as
a published number. PLAN already says the right thing ("milestones reached under declared
game-time and compute budgets"). Add: discounted return is never the Phase 9 benchmark
figure, and each resumable milestone scenario (PLAN 9.3) carries its own γ sized to its own
budget. **If** Phase 9 ever runs a single uninterrupted continuing episode with no reset —
which 9.5's "start-to-finish run" implies — then §10.4, pp. 205–206, applies in full and the
objective must be average reward. That is the one place in the arc where the setting genuinely
changes, and the book itself poses the combination as an open exercise (Exercise 17.1,
p. 382: "This section has presented options for the discounted case, but discounting is
arguably inappropriate for control when using function approximation (Section 10.4). What is
the natural Bellman equation for a hierarchical policy … for the average reward setting?").

---

## 3. The deadly triad, and what it says about the structural holdout

§11.3, p. 215, names the triad — function approximation, bootstrapping, off-policy training —
and then supplies the operational fact the incomplete draft could not: "**If any two elements
of the deadly triad are present, but not all three, then instability can be avoided.**" Also,
critically: "note that the danger is *not* due to control, or to generalized policy iteration
… The danger is also *not* due to learning or to uncertainties about the environment, because
it occurs just as strongly in planning methods, such as dynamic programming, in which the
environment is completely known." A perfect simulator buys nothing here.

**Phase 4 has two of three, and the missing one is off-policy training.** MaskablePPO is
on-policy: the rollout is collected under the current policy and discarded after
`n_epochs` passes. Divergence in Baird's sense (§11.2, pp. 212–213) is not a FactorioRL
risk today, and the project should stop worrying about it. §11.3, p. 216: "Finally, there is
off-policy learning; can we give that up? On-policy methods are often adequate… Off-policy
methods free behavior from the target policy. This could be considered an appealing
convenience but not a necessity."

**PLAN Phase 8 is exactly where the third leg arrives, and the book says so.** Two sentences,
both worth pinning to Phase 8's plan:

* §11.3, p. 216, on why off-policy cannot simply be abandoned: "the agent learns not just a
  single value function and single policy, but large numbers of them in parallel." That is
  GVFs (§17.1) and option models (§17.2) — PLAN 8.2 and 8.3.
* §17.2 bibliographic notes, p. 393: "An important limitation of these early works is that
  they treated either the tabular case or the on-policy case. The general case of
  **intra-option learning involves off-policy learning, which could not be done reliably with
  function approximation** at that time. Although now we have a variety of stable off-policy
  learning methods using function approximation, their combination with option ideas had not
  been significantly explored at the time of publication of this book."

The practical consequence for Phase 8 is in R15: use the simple option update — §17.2, p. 381,
"In the simplest case, the learning process 'jumps' from option initiation to option
termination, with an update only occurring when an option terminates" — and do **not** build
intra-option learning. That keeps Phase 8 at two legs.

**The structural holdout is not a triad question at all.** It is a distribution question, and
the authority is §9.2, pp. 162–163. The prediction objective is `VE(w) = Σ_s μ(s)[v_π(s) −
v̂(s,w)]²`, weighted by a distribution μ you are "obligated … to say" — and in an episodic
task, the box on p. 163 is explicit: "the on-policy distribution is a little different in that
it **depends on how the initial states of episodes are chosen**. Let h(s) denote the
probability that an episode begins in each state s." A held-out layout family is a different
h, therefore a different μ. p. 170: "Critical to these convergence results is that states are
updated according to the on-policy distribution. For other update distributions, bootstrapping
methods using function approximation may diverge." Every guarantee the training run enjoys is
stated on the training μ and none of them transfer to `wall_arc`, `screened_depot`,
`screened_patch`, `double_gap`, `gap_near_drill` or `shared_input`.

Two consequences, both cheap: measure the critic's transfer separately from the actor's
(R12), and recognise that h(s) is a **design knob you own** and almost nobody else does
(R9).

One more triad-adjacent finding that bears directly on the encoder. §11.3, p. 215:
"bootstrapping can impair learning on problems where the state representation is poor and
causes poor generalization … A poor state representation can also result in bias; this is the
reason for the poorer bound on the asymptotic approximation quality of bootstrapping methods
(Equation 9.14)." FactorioRL's observation is partial by construction (32-tile sensor,
remembered entities with an age feature, 32 padded rows). That is an argument for *less*
bootstrapping — a higher λ — which is R6.

---

## 4. Recommendations

Each gives the idea with a citation to the complete text, the specific repository change, the
phase, the expected impact, the cost and risk, and any conflict with PLAN.md.

---

**R1 — One owner for γ, asserted and manifested.**
*Idea.* The telescoping identity `Σ γ^t w(γΦ(s_{t+1}) − Φ(s_t)) = w(γ^TΦ(s_T) − Φ(s_0))` uses
the *same* γ the learner discounts with; §3.3, p. 44, eq. (3.8), defines exactly one discount
rate for a task.
*Change.* Delete `rewards.GAMMA`; add `gamma: float` to `RewardAccountant`; thread
`TrainConfig.gamma → FactorioEnv.__init__ → accountant`; assert equality once in `train()`;
add the shaping γ to the manifest's `reward` block. Test 1.4(3). Note the existing test module
imports `GAMMA`, so that import moves.
*Phase.* 4.1, before 4.5.
*Impact.* Prevents a silent invariance break on the first run that changes γ — which R5 asks
for immediately.
*Cost/risk.* Trivial.
*Conflict.* None.

---

**R2 — Cap high-water shaping and enforce a shaping budget below the sparse weight.**
*Idea.* §3.2, p. 43 (subgoal rewards let the agent "achieve them without achieving the real
goal"); §17.4, p. 386 ("the agent may end up not achieving the overall goal at all").
*Change.* `cap: float | None` on `RewardComponent`, clamped in the `HIGH_WATER` branch; caps
on `mine_smelt.ore_mined`, `supply_furnace.ore_carried`, `deliver.carried`; `cap` into
`TaskSpec.to_dict()`; test 1.4(4) over every registered task.
*Phase.* 3.5 feeding 4.4.
*Impact.* Largest single expected effect on `mine_smelt`, where the subgoal currently
dominates the objective, and on `supply_furnace`, where it exactly ties it.
*Cost/risk.* Small. Changes the reward configuration, so pinned task versions must be bumped
and the manifest digest will change.
*Conflict.* None; it is PLAN 4.4 executed ("Any shaping exploitation becomes a task or reward
defect with a regression test").

---

**R3 — Move `STEP_COST` outside the shaping switch.**
*Idea.* §3.2, p. 42: "the reward is often −1 for every time step that passes prior to escape;
this encourages the agent to escape as quickly as possible" — a task definition, in the
paragraph that states the reward hypothesis.
*Change.* Evaluate `STEP_COST` before the `shaping_enabled` gate. Test 1.4(5). Record in the
ledger that the shaping ablation now varies exactly one thing.
*Phase.* 4.4.
*Impact.* Makes PLAN 4.4's required distinction ("faster learning" vs "changed task
difficulty") answerable at all.
*Cost/risk.* Two lines; changes what "sparse" means in any future run, so it must be recorded.
*Conflict.* None; arguably required by 4.4 as written.

---

**R4 — Honour `RewardComponent.shaping`, or delete it.**
*Idea.* PLAN §3 requires runs to record the reward configuration and to "Separate sparse task
outcomes, optional shaping, evaluation metrics." A field recorded in two manifests and read by
nothing is a false record.
*Change.* Either drive the `shaping_enabled` gate off `component.shaping` (which also
implements R3 for free, since `step_cost` would carry `shaping=False`), or remove the field
from `RewardComponent`, `TaskSpec.to_dict()` and `train.py`'s reward block.
*Phase.* 3.5.
*Impact.* Removes a provenance defect from every manifest already written.
*Cost/risk.* Trivial. Prefer the first option: it makes R3 a data change rather than a code
change, and it makes future non-shaping components (a `RATE` outcome, R17) declarable.
*Conflict.* None.

---

**R5 — Size γ per task from the decision budget, and sweep it.**
*Idea.* §3.3, p. 44: "As γ approaches 1, the return objective takes future rewards into
account more strongly; the agent becomes more farsighted." Against that, Eq. (9.14), p. 169:
`VE(w_TD) ≤ 1/(1−γ) · min_w VE(w)`, "this expansion factor can be quite large."
*Change.* Add `gamma: float` to `TaskSpec`, defaulting to `1 − 1/(2·max_decision_steps)`
(0.99667 for `navigate`, 0.99875 for `mine_smelt`); use it in `train()` and, via R1, in the
accountant; record per-task γ in the manifest; run a 3-point sweep on one short and one long
family rather than adopting it blind.
*Phase.* 4.2 / 4.3.
*Impact.* On `mine_smelt` this moves the discounted value of a success at decision 250 from
0.081 to 0.73, against a step-cost accrual that stays at ~0.1. Likely the second-largest
effect in this document after R9.
*Cost/risk.* Low code cost. The real risk is the 1/(1−γ) bound above and higher return
variance, both of which land on a critic that has not yet learned anything — which is why this
is a sweep, and why R6 is its companion.
*Conflict.* PLAN 4.1 pins hyperparameters into resolved configs; this makes γ a *task*
property rather than a run property. Record that contract change in the ledger.

---

**R6 — Sweep `gae_lambda`; do not inherit SB3's 0.95.**
*Idea.* GAE is the λ-return in forward view (§12.1, p. 236: a valid update target is "any
average of n-step returns"). Two results point the same way for this project. §12.13, p. 261:
"On tasks with many steps per episode, or many steps within the half-life of discounting, it
appears significantly better to use eligibility traces than not to … On the other hand, if the
traces are so long as to produce a pure Monte Carlo method, or nearly so, then performance
degrades sharply. An intermediate mixture appears to be the best choice." And, for a partially
observed environment: "Monte Carlo methods may have advantages in non-Markov tasks because
they do not bootstrap. Because eligibility traces make TD methods more like Monte Carlo
methods, they also can have advantages in these cases … Eligibility traces are the first line
of defense against both long-delayed rewards and non-Markov tasks" (p. 261). §11.3, p. 215,
adds the bias argument: bootstrapping "can impair learning on problems where the state
representation is poor."
*Change.* Add `gae_lambda` to `TrainConfig` (currently unset, so 0.95 applies silently); sweep
{0.9, 0.95, 0.98, 0.99} on `navigate` and one production family; record it in the manifest.
*Phase.* 4.2 / 4.3.
*Impact.* FactorioRL is *exactly* the case p. 261 describes: 150–400 decisions per episode,
terminal-only reward on four of six families, and a deliberately partial observation. Cheap,
and the optimum is task-dependent (§7.1, p. 118, Fig. 7.2: "an intermediate value of n worked
best").
*Cost/risk.* Four training runs. At ~42 env steps/s, run the sweep on `navigate` (150-step
budget) first.
*Conflict.* None.

---

**R7 — Measure rollout length against episode length, and log the ratio.**
*Idea.* §7.1, p. 117: "If t+n ≥ T (if the n-step return extends to or beyond termination),
then all the missing terms are taken as zero, and the n-step return defined to be equal to the
ordinary full return." A rollout shorter than an episode contains no complete return, so all
credit assignment routes through a critic whose only guarantee (Eq. 9.14, p. 169) is on the
training distribution.
*Change.* `train.py` computes `steps_per_env = max(MIN_STEPS_PER_ENV, n_steps // workers)` —
with the defaults of 512 and 8 workers, that is 64 against budgets of 150–400. The manifest
already has a `rollout` block; add `steps_per_env / max_decision_steps` and the fraction of
rollout windows containing a terminal. Then treat `MIN_STEPS_PER_ENV` as a swept parameter in
4.3's worker-count profiling rather than a constant.
*Phase.* 4.3.
*Impact.* On `mine_smelt`, roughly one rollout window in six can contain the sparse reward at
all. The `train.py` comment already records that eight workers collapsed training success from
1.00 to 0.05 at an unchanged step count; the current mitigation fixes the *update count* and
not the *horizon per rollout*.
*Cost/risk.* Configuration and logging only. Longer rollouts mean fewer updates per wall-clock
hour — a genuine trade-off, and 4.3 is the place to measure it.
*Conflict.* None; it is PLAN 4.3 with two more measured quantities.

---

**R8 — Give `repair_belt` and `restore_power` a gradient by initialising the value function,
not by adding reward.**
*Idea.* This is the single most useful thing the complete text adds. §17.4, p. 386: "It is
tempting to address the sparse reward problem by rewarding the agent for achieving subgoals
that the designer thinks are important way stations to the overall goal. But augmenting the
reward signal with well-intentioned supplemental rewards may lead the agent to behave very
differently from what is intended … **A better way to provide such guidance is to leave the
reward signal alone and instead augment the value-function approximation with an initial guess
of what it should ultimately be**", i.e. `v̂(s,w) ≐ wᵀx(s) + v₀(s)` (eq. 17.12) — "This
initialization works for arbitrary nonlinear approximators and arbitrary forms of v₀.
Wiewiora (2003) showed that this initialization is equivalent to the more complex
'potential-based shaping' technique." §3.2 footnote 5, p. 43, says the same thing in one line:
"Better places for imparting this kind of prior knowledge are the initial policy or initial
value function."
*Change.* Two options, provably equivalent, and the second is better engineering:
 (a) add a capped `POTENTIAL` component on `CHARACTER_WITHIN(marker="gap")` for both families;
 (b) add an optional `v0` term to the value head in `learn/policy.py` — a small fixed function
 of the goal/self features, added after the value MLP — declared per task and recorded in the
 manifest. Under (b) the reward signal stays exactly the sparse predicate plus the step cost,
 the 4.4 ablation stays clean, and there is no shaping budget to police.
Either way both families need markers the blueprint may not currently carry
(`restore_power` declares only `drill`; `repair_belt` only `sink`).
*Phase.* 3.2 / 3.5, feeding 4.2.
*Impact.* These are two of the three families `docs/LEDGER.md` records as never succeeding
even in training. With the `$ahead` catalog defect fixed, the remaining obstacle on both is
that there is no gradient at all before the terminal event.
*Cost/risk.* Moderate — generator changes for the markers. Route (b) is a change to the
policy, so it bumps `EXTRACTOR_VERSION` and invalidates existing checkpoints: do it before
4.5, not after. Note that a potential built from evaluator `truth` is only safe *because* the
invariance theorem holds; §1.1 confirms it now does, so the design is licensed.
*Conflict.* None. It sits inside PLAN 3.5's "configurable shaping". Record in the ledger that
reward may read `truth` while masks may not, which `env.action_masks()` already enforces.

---

**R9 — Use the 6.21 ms reset to own the start-state distribution. Training split only.**
*Idea.* This is the project's largest unexploited advantage and the book points at it three
times. §5.3, p. 79, defines exploring starts, then immediately concedes the thing FactorioRL
does not have to concede: "The assumption of exploring starts is sometimes useful, but of
course it cannot be relied upon in general, **particularly when learning directly from actual
interaction with an environment**. In that case the starting conditions are unlikely to be so
helpful." §5.10, p. 94: "Such exploring starts can sometimes be arranged in applications with
simulated episodes, but are unlikely in learning from real experience." §9.2, p. 163: in
episodic tasks the on-policy distribution μ "depends on how the initial states of episodes are
chosen. Let h(s) denote the probability that an episode begins in each state s" — and μ is
what the VE objective (eq. 9.1, p. 162) is weighted by. §8.6, p. 145: on-policy trajectory
sampling starts "in a start state (or according to the starting-state distribution)", and its
advantage "is large and long-lasting" for problems with many states and small branching
factors (p. 145–146, Fig. 8.9).
*Change.* Add an optional `start_state_curriculum` to `TaskSpec`: a declared fraction of
**training** episodes reset with the character placed near the task-critical location (the
belt gap, the pole gap, the ore patch) rather than at the origin. Blueprint generation already
takes an `rng` and already carries markers, so this is a generator change, not a runtime
change. Manifest it. **The `val` and `test` splits keep the canonical start, always**, enforced
by a test.
*Phase.* 3.2 / 3.3, feeding 4.2.
*Impact.* Plausibly the highest-leverage change in this document for the three families that
never succeed. It does not make the task easier at evaluation; it makes the terminal reward
reachable during learning.
*Cost/risk.* Moderate implementation. The risk is serious and one-directional: leakage into
the held-out split would be exactly the contamination PLAN §3 forbids. Assert it.
*Conflict.* None with PLAN, but it changes the training distribution, so the held-out result
becomes the only number comparable across configurations — which PLAN already requires.

---

**R10 — Add auxiliary prediction heads to the shared extractor.**
*Idea.* §17.1, pp. 379–380. "One simple way in which auxiliary tasks can help on the main task
is that they may require some of the same representations as are needed on the main task. Some
of the auxiliary tasks may be easier, with less delay and a clearer connection between actions
and outcomes. If good features can be found early on easy auxiliary tasks, then those features
may significantly speed learning on the main task." The architecture is described concretely
on p. 380: a network whose last layer splits into multiple *heads*, all propagating error into
a shared body, "which would then try to form representations, in its next-to-last layer, to
support all the heads … In many cases this approach has been shown to greatly accelerate
learning on the main task (Jaderberg et al., 2017)." And these predictions are formally GVFs,
eq. (17.1), p. 379 — the same objects §17.2, p. 382, uses to learn option models.
*Change.* `FactorioExtractor` already has a shared body and a 256-d head. Add 2–3 auxiliary
heads with cheap, already-available cumulants: distance to the nearest goal marker at the next
step, whether the last action succeeded (`_last_action_ok` is computed and thrown away), and
inventory delta. Weight their losses small; log them; disable them at evaluation (they change
nothing at inference — they only shape the body during training).
*Phase.* 4.1, before 4.5.
*Impact.* Directly attacks the "no gradient before the terminal event" problem on
`repair_belt` and `restore_power` without touching the reward signal at all, and it builds
exactly the multi-head machinery Phase 8's option models need (§17.2, p. 382: "One natural way
to learn an option model is to formulate it as a collection of GVFs").
*Cost/risk.* Moderate; it is a policy change, so `EXTRACTOR_VERSION` bumps and checkpoints
invalidate — do it before 4.5. It also adds loss weights to tune. Note the book's own caveat,
p. 380: "There is no necessary reason why this has to be true, but in many cases it seems
plausible."
*Conflict.* None. Auxiliary heads are not a custom RL algorithm and not a recurrent policy;
PPO's objective is unchanged.

---

**R11 — Add a k-th order observation/action history to the `self` vector. Not a recurrent
policy.**
*Idea.* §9.1, p. 161, states the limitation precisely: "extending reinforcement learning to
function approximation also makes it applicable to partially observable problems … **What
function approximation can't do, however, is augment the state representation with memories of
past observations.**" §17.3, p. 385, gives the cheap fix: "Perhaps the simplest example of an
approximate state is just the latest observation, S_t ≐ O_t. Of course this approach cannot
handle any hidden state information. **Better is to use the last k observations and actions**,
S_t ≐ O_t, A_{t−1}, O_{t−1}, …, O_{t−k}, for some k ≥ 1, which can be achieved by a
state-update function that just shifts the new data in and the oldest data out. This kth-order
history approach is still very simple, but can greatly increase the agent's capabilities."
*Change.* `encoders.SELF_FEATURES` is 12 with slots 9–11 unused. Add the last k = 4 action
indices (one-hot or embedded) and a 2-vector of position delta per step. This is a shift
register, not a recurrent unit.
*Phase.* 4.1, alongside R10 (same version bump).
*Impact.* The environment is partially observable by design (32-tile sensor; remembered
entities with an `age` feature). The current observation gives the policy no way to know it
just tried `place_transport_belt_east` and it failed — `_last_action_ok` is computed in
`env.step` and never encoded. On `repair_belt` and `restore_power`, where the failure mode is
repeated illegal placement, that is likely to matter.
*Cost/risk.* Small. Bumps `EXTRACTOR_VERSION`.
*Conflict.* PLAN §3 forbids recurrent policies before the baseline works. A fixed-length
history buffer in the observation is not a recurrent policy: no hidden state crosses the
network, and the observation space stays a fixed-shape `Dict`. Say so explicitly in the ledger
so a reviewer does not read it as a violation.

---

**R12 — Measure critic transfer separately from actor transfer on the structural holdout.**
*Idea.* §9.2, pp. 162–163, and p. 170. The VE objective is weighted by μ, which in an episodic
task depends on the start-state distribution h(s); "Critical to these convergence results is
that states are updated according to the on-policy distribution" (p. 170). A held-out layout
family is a different μ, and nothing bounds the critic's error there.
*Change.* In `evaluate()`, additionally record `V̂(s₀)` from the policy's value head against
the realised discounted return of each held-out episode, and report the mean signed error and
its spread. One extra forward pass per episode.
*Phase.* 4.5 and 7.5.
*Impact.* Distinguishes "the actor did not transfer" from "the critic did not transfer" — two
different fixes. Gives PLAN 7.5's "Report ordinary-seed and structural generalization
separately" a quantity beyond success rate.
*Cost/risk.* Negligible.
*Conflict.* None. **This adds a diagnostic alongside the 80% threshold; it does not touch it.**

---

**R13 — A recency exploration bonus, training-only, declared and logged.**
*Idea.* §8.3, p. 138: "if the modeled reward for a transition is r, and the transition has not
been tried in τ time steps, then planning updates are done as if that transition produced a
reward of r + κ√τ, for some small κ. This encourages the agent to keep testing all accessible
state transitions and even to find long sequences of actions in order to carry out such
tests." Figure 8.6, p. 138, shows it is the only Dyna variant that finds the shortcut. §17.4,
p. 388, generalises it: "An example is the 'bonus reward' described in Section 8.3. Instead of
being tied to a specific task, this reward signal encourages exploration in general, which
benefits the learning of many specific tasks." The two obvious alternatives come with the
book's own refusals: §2.6, p. 27, on optimistic initial values — "it is not well suited to
nonstationary problems because its drive for exploration is inherently temporary … The
beginning of time occurs only once, and thus we should not focus on it too much"; §2.7, p. 28,
on UCB — "Another difficulty is dealing with large state spaces, particularly when using
function approximation … In these more advanced settings the idea of UCB action selection is
usually not practical."
*Change.* A training-only `RewardKind.NOVELTY` component: `κ√τ` where τ is decisions since
this (catalog index, coarse state bucket) was last taken, κ small enough that R2's budget test
still passes. Logged as its own component; in the manifest; **hard-disabled in `evaluate()`
and `random_baseline()`**. Note Exercise 8.4, p. 139, proposes the cleaner variant — the bonus
"used not in updates, but solely in action selection" — which for PPO would mean a logit bias
rather than a reward term, and which would leave the reward function untouched. Prefer that if
it can be made to work without perturbing the policy gradient; otherwise take the reward route
and treat it exactly like shaping.
*Phase.* 4.2 for the families that never succeed; carried into 7.3, where interventions make
the environment genuinely nonstationary and this bonus is at its most valuable.
*Impact.* The concrete failure mode is a policy that never tries `place_*` because it has
never paid off. A recency bonus is the cheapest correct answer and needs no value machinery.
*Cost/risk.* **This is not potential-based and it does change the optimal policy.** Declared,
weighted, per-component logged, and off for every reported evaluation. If it is ever on during
evaluation the result is invalid.
*Conflict.* None with PLAN, but it adds a third thing the 4.4 ablation must hold fixed — see
R3.

---

**R14 — Build five small things now so options are addable in Phase 8 without a rewrite.**
*Idea.* An option is ω = ⟨π_ω, γ_ω⟩ (§17.2, p. 381) — an internal policy and a
state-dependent termination function. Availability is Ω(s), "the set of options available in
state s" (eq. 17.4, p. 382), which is an action mask. The option model has two parts: the
discounted reward accrued *during* execution, `r(s,ω) = E[R₁ + γR₂ + … + γ^{τ−1}R_τ]` (eq.
17.2, p. 381), and a transition part with γ^k folded in, `p(s′|s,ω) = Σ_k Pr{S_k = s′, τ = k}
γ^k` (eq. 17.3, p. 382). "Options are designed so that they are interchangeable with
(primitive) actions" and "are a strict generalization of low-level actions" (p. 381).
PLAN 8.1's skill contract is already this, term for term. The plumbing is not.
*Change.* Five items, all cheap now and expensive later:
 1. `env.step` must return elapsed ticks in `info["ticks"]`. It does not; `spec_.decision_ticks`
    is assumed constant everywhere, and the moment a skill spans a variable number of ticks
    that assumption is silently wrong.
 2. `RewardAccountant.step` must be able to return the **interval-integrated discounted**
    reward, not an endpoint sample. High-water is safe (monotone) and potential is safe
    (telescoping), but any future `RATE` kind (R17) is not, and eq. (17.2) requires the
    integral. Decide the contract now: a skill reports the discounted sum of the primitive
    rewards along its execution.
 3. `_goal_vector[0]` normalises elapsed time by `max_decision_steps`. Normalise by ticks
    against `max_game_ticks` instead — under variable-duration decisions, "fraction of
    decisions used" is no longer the quantity that makes truncation Markov. (See open
    question 3: `max_game_ticks` is currently exactly 60 × `max_decision_steps` on all six
    families and can never bind.)
 4. `ActionTemplate` gains an optional `duration_ticks` and an optional termination
    descriptor, so a skill is just another catalog entry and the ordered-index /
    content-hash reproducibility discipline in `catalog.py` survives intact.
 5. Record Δt per transition in the rollout even though SB3 cannot use it. A learner that
    can (R15 route b) will need the history.
*Phase.* 3.4 / 4.1 for the plumbing; 8.1 for the skills.
*Impact.* Phase 8 becomes an addition rather than a simultaneous refactor of the observation
encoding, the reward accountant and the catalog.
*Cost/risk.* Low now. Item 3 changes the observation, so bump `EXTRACTOR_VERSION` before 4.5,
not after — batch it with R10 and R11.
*Conflict.* None. It is PLAN 8.1's contract given somewhere to live. Note that PLAN 8.1's
"preconditions" map onto Ω(s), i.e. onto the mask — and the mask must therefore stay
computable from policy-visible information only, which `env.action_masks()` already
guarantees for primitives.

---

**R15 — First manager: update only at option termination, with fixed-duration skills.**
*Idea.* §17.2, p. 381: "many of the algorithms in this book can be generalized to learn
approximate option-value functions and hierarchical policies. **In the simplest case, the
learning process 'jumps' from option initiation to option termination, with an update only
occurring when an option terminates.** More subtly, updates can be made on each time step,
using intra-option learning algorithms, which in general require off-policy learning." Read
that against §11.3, p. 215 (the third leg of the triad) and the bibliographic note at p. 393
("their combination with option ideas had not been significantly explored at the time of
publication of this book"). Per-transition discounting is available in principle — §12.8,
p. 253, generalises γ to γ: S → [0,1] with `G_t = Σ_k R_{k+1} Π_i γ_i` (eq. 12.24) — but SB3
applies one scalar γ, so it is a custom learner.
*Change.* Make PLAN 8.3's "Skill duration is handled explicitly in manager training" concrete.
Route (a), legal today: constrain every skill to consume exactly k decision intervals (padded
with `wait`), which makes the manager a standard MDP on a coarser clock with a constant γ^k
and needs no algorithm change. Route (b), post-Phase-4: a manager learner with per-transition
γ^Δt per eq. (12.24). Do **not** build intra-option learning for the first manager.
*Phase.* 8.3.
*Impact.* Lets Phase 8 start without a custom algorithm, and keeps it at two legs of the
triad. Deciding this now stops Phase 8 from being blocked on a policy question.
*Cost/risk.* Fixed durations waste engine time on padding and bias skill selection toward
whatever k suits. Both are measurable and must be reported.
*Conflict.* PLAN §3's ban on custom RL algorithms applies **before the baseline works**. Route
(a) is legal today; route (b) is explicitly post-Phase-4 and should be labelled as such in the
work package.

---

**R16 — No Dyna. Decision-time planning at evaluation only, and only after checkpoint
branching exists.**
*Idea.* §8.2, p. 134, gives the case for indirect RL: "Indirect methods often make fuller use
of a limited amount of experience and thus achieve a better policy with fewer environmental
interactions." That premise requires simulated experience to be cheaper than real experience.
Here the model *is* the environment, so a learned Factorio model would be slower to query than
Factorio and wrong besides (§8.3, pp. 137–139, on the cost of a wrong model). Dyna buys
nothing. Three things in the same chapter, all absent from the incomplete draft, *do* apply:
 * §8.6, pp. 144–146, trajectory sampling — the on-policy distribution is where backups
   should be spent, and it is generated by starting "in a start state (or according to the
   starting-state distribution)". That is R9, and it is the part of Chapter 8 FactorioRL
   should actually act on.
 * §8.9, pp. 150–151, heuristic search: "if one has a perfect model and an imperfect
   action-value function, then in fact deeper search will usually yield better policies …
   On the other hand, the deeper the search, the more computation is required, usually
   resulting in a slower response time." §8.8, p. 150: "Decision-time planning is most useful
   in applications in which fast responses are not required." FactorioRL pauses between
   decisions, so search costs wall clock and not game time — an unusually favourable case.
 * §8.10, p. 152, rollout algorithms: "the aim of a rollout algorithm is to improve upon the
   default policy; not to find an optimal policy … In some applications, a rollout algorithm
   can produce good performance even if the rollout policy is completely random." §8.11,
   pp. 153–155, MCTS; and §16.6.2, p. 370, notes that AlphaGo Zero used MCTS *during* self-play
   as a policy-improvement operator, not only at play time.
*Change.* No model learning, ever. Instead: (i) treat PLAN 10.1's validated checkpoints as the
enabling primitive, because the current 6.21 ms reset installs a *blueprint* and cannot branch
from an arbitrary mid-episode state — state that dependency explicitly; (ii) once branching
exists, a depth-1 or depth-2 lookahead over the masked catalog scored by the critic, run at
**evaluation only**, is exactly §8.9's construction and needs no new learner. At ~40 catalog
entries and ~42 steps/s, depth 2 costs on the order of 1 600 engine steps per decision.
*Phase.* 10.1 supplies the checkpoint; the search itself is a 10.3 ablation.
*Impact.* Converts "we have a perfect cheap simulator" from a slogan into two concrete
mechanisms and closes off the one that would waste months.
*Cost/risk.* R9 is cheap; the lookahead is expensive and must be disclosed as assistance under
PLAN 10.3 ("Information and execution differences are disclosed").
*Conflict.* A search-augmented policy is a *different system* from the compact policy. PLAN
8.5's "Match task and observation settings where comparisons claim equivalence" applies; it
must never be reported against the 80% acceptance threshold as if it were the baseline.

---

**R17 — Add a `RATE` predicate kind before Phase 7 tasks are authored.**
*Idea.* §10.3, p. 202, eq. (10.6): the quality of a policy in a continuing setting is
`r(π) = lim_{h→∞} (1/h) Σ E[R_t]` — a rate. PLAN 7.4's required measurements ("time to restore
production", "production lost over time", "final sustained output") and PLAN 7.2's "Success
requires sustained output" are all rates. `PredicateKind` has `PRODUCED` (a level) and
`RewardKind` has `HIGH_WATER` (a level's maximum). Neither can express one.
*Change.* Add `PredicateKind.RATE` (item, window in ticks, at_least per window) and let it
drive both a success predicate and a reward component. It needs the interval-integrated
counter contract from R14 item 2.
*Phase.* 3.1 / 3.5, before any Phase 7 task is authored.
*Impact.* Retrofitting after Phase 7 tasks exist means re-versioning every task and
invalidating every pinned comparison. PLAN 7.2 requires "Existing tasks remain compatible with
their pinned versions", which is precisely what a late retrofit would break.
*Cost/risk.* Small now, large later. The measurement-window semantics need care: a rate
computed over a window is not Markov in the endpoint state unless the window's contents are
observable, so the window counter must appear in the observation.
*Conflict.* None; it is PLAN 3.1's "closed vocabulary" extended once, deliberately.

---

**R18 — Drop infrastructure-failure transitions from the learner, not just from the metrics.**
*Idea.* §3.3, p. 43: an episode "ends in a special state called the terminal state, followed by
a reset to a standard starting state or to a sample from a standard distribution of starting
states." A crashed worker produces neither, so the transition is not a sample from the MDP and
must not enter a return estimate. `env.py`'s own docstring says the trainer must drop it;
nothing does, and SB3 bootstraps `γ·V̂(s_T)` onto a reward of zero.
*Change.* Either (a) raise on infrastructure failure and let PLAN §2's recovery path handle it,
preserving the run directory as evidence, or (b) keep returning the sentinel but have a
callback zero the advantage for that transition and abort the run above a configured rate.
Also filter `excluded` rows out of `final_train_success_rate`. Record the infrastructure-
failure count in `result.json` either way.
*Phase.* 4.1 / 4.3.
*Impact.* A worker that crashes intermittently currently injects zero-reward bootstrapped
transitions into training and reports the episodes as failures in the curve. Neither is
visible in the published numbers.
*Cost/risk.* Small. Option (a) is more honest and matches PLAN §2's "Worker crashes are
infrastructure failures, not ordinary task failures"; option (b) keeps long runs alive.
*Conflict.* None; it is PLAN §2 made mechanical.

---

## 5. Do NOT do this

* **Do not add eligibility traces to PPO.** GAE is the λ-return in forward view (§12.1,
  p. 236). §12.2 onwards is the *backward* view — the same algorithm arranged for online
  incremental updating, which a batch policy-gradient learner does not do. The lever is
  `gae_lambda` (R6). Adding traces would buy nothing and would count as a custom algorithm
  under PLAN §3.
* **Do not learn a transition model or build a Dyna loop.** §8.2, p. 134 — the case for
  indirect RL is that it makes "fuller use of a limited amount of experience." Your model
  would be a slower, wronger copy of a simulator you already own. See R16.
* **Do not replay old rollouts with full-trajectory importance ratios.** §5.5, p. 86: "the
  variance of the ordinary importance-sampling estimator is in general unbounded because the
  variance of the ratios can be unbounded." Example 5.5 and Figure 5.4, pp. 87–88: ordinary IS
  fails to converge after ten million episodes on a one-state MDP, and the failure occurs
  "whenever … trajectories contain loops." §5.8, p. 92, on the extraneous factors: "they do not
  change the expected update, but they add enormously to its variance. In some cases they could
  even make the variance infinite." Your episodes are 150–400 decisions of a walking agent —
  nothing but loops. PPO's clipped surrogate over `n_epochs` is already bounded, per-decision
  reuse; if more reuse is wanted, raise `n_epochs` and measure. Note the honest counterweight
  at §17.5, p. 388 — "Techniques such as 'replay buffers' are often used to retain and replay
  old data so that its benefits are not permanently lost. An honest assessment has to be that
  current deep learning methods just don't learn well online" — and note also that DQN's
  replay works *because* Q-learning is off-policy (§16.5, p. 364: "Since Q-learning is an
  off-policy algorithm, it does not need to be applied along connected trajectories"). PPO is
  not.
* **Do not switch the Phase 4 or Phase 7 families to average reward or differential Sarsa.**
  §10.3, p. 202: the average-reward setting "applies to continuing problems, problems for
  which the interaction between agent and environment goes on and on forever without
  termination or start states", and requires ergodicity. Your tasks terminate and reset.
  §10.4, p. 206: "The discounted case is still pertinent, or at least possible, for the
  episodic case." The framing applies to Phase 9's uninterrupted run; the algorithm applies
  nowhere yet.
* **Do not push γ toward 1.0 as a blanket fix for sparse reward.** Eq. (9.14), p. 169: the TD
  fixed point's error bound is `1/(1−γ)` times the minimum, and "this expansion factor can be
  quite large." R5 says *size γ to the budget and sweep it*, which is not the same
  instruction.
* **Do not use UCB.** §2.7, p. 28: "Another difficulty is dealing with large state spaces,
  particularly when using function approximation … In these more advanced settings the idea of
  UCB action selection is usually not practical."
* **Do not rely on optimistic initialisation for exploration.** §2.6, p. 27: its drive to
  explore is "inherently temporary … The beginning of time occurs only once, and thus we
  should not focus on it too much." A curriculum (R9) and Phase 7's interventions make your
  problem nonstationary by construction. Note this is *not* an argument against R8: R8
  initialises the *value function* with task knowledge, which §17.4, p. 386, explicitly
  endorses; §2.6 is about initialising action values *optimistically* to force exploration.
* **Do not build intra-option learning for the first manager.** §17.2, p. 381: intra-option
  algorithms "in general require off-policy learning", which is the third leg of the triad
  (§11.3, p. 215), and p. 393 records that its combination with options and function
  approximation "had not been significantly explored." R15 route (a) instead.
* **Do not let the R13 novelty bonus run during evaluation, and do not let R9's start-state
  curriculum touch `val` or `test`.** Both are training-only by definition; both are invisible
  in a success rate; both would invalidate a published result.
* **Do not introduce recurrent policies or a custom RL algorithm before the baseline works.**
  PLAN §3 forbids it and this document respects it. R11 is a fixed-length history in the
  observation, not a recurrent policy — §17.3, p. 385, treats the two as different things.
  R15 route (b) and R16's lookahead are labelled post-Phase-4.
* **Do not fold recovery and production measurements into a composite score.** PLAN §3 already
  forbids it. §10.3's differential values, p. 202, are the right *shape* for a rate but they
  are a value function, not a benchmark number.
* **Do not lower the 80% held-out threshold, and do not reinterpret an existing run to meet
  it.** Several recommendations here (R2, R3, R5, R8, R10, R11) change the task, the reward or
  the encoder, so the `navigate` pilot's 0.40 is not comparable across them and must be re-run
  rather than reinterpreted.

---

## 6. Open questions

1. **Failure terminals and the potential.** No family declares a failure predicate today, so
   `terminated == succeeded` everywhere and the terminal-Φ branch has only ever fired on
   success. After Phase 7.3 adds interventions, entering a *failure* terminal will cost the
   agent −wΦ(s), so failing near the goal is punished more than failing far from it. §3.5,
   p. 46, prescribes it ("the value of the terminal state, if any, is always zero"). Is it the
   intended incentive, or does `restore_power` want a failure predicate at all?
2. **Which route for R8?** Adding a `POTENTIAL` component and adding a `v₀` term to the value
   head are provably equivalent (§17.4, p. 386, citing Wiewiora 2003), but they differ in
   auditability: the reward route is visible in the per-component log and must be policed by
   the R2 budget test; the value route leaves the reward signal pristine but hides the prior in
   a checkpoint. Which does the project want to be able to inspect?
3. **`max_game_ticks` is unreachable on every family.** It is exactly 60 × `max_decision_steps`
   in all six specs, while every decision consumes exactly 30 ticks — so `max_decision_steps`
   always binds first and the tick budget is dead configuration. Is it reserved for Phase 7's
   ongoing actions (in which case R14 item 3 depends on the answer), or should it be removed?
4. **Who owns a skill's initiation set?** §17.2, eq. (17.4), p. 382, puts availability in Ω(s),
   the manager's action set — i.e. the mask, which must be built from policy-visible
   information only. PLAN 8.1 puts preconditions on the skill. Skill-side or task-side, and
   how does a skill's precondition become a mask entry without reading `truth`?
5. **Is the critic expected to transfer across layout families, or only the actor?** R12
   measures it; nobody has said what the answer should be, and it changes whether a held-out
   failure is a representation problem or an exploration problem.
6. **Is `ent_coef = 0.01` doing what it is assumed to do at this batch size?** With 8 workers ×
   64 steps and a 9–18 entry masked catalog whose support changes every step, the entropy term
   is applied to a distribution over a varying support. Worth one ablation before R13 adds a
   second exploration mechanism on top of it.
7. **Does Phase 9 ever run a genuinely continuing episode?** PLAN 9.5 requires "a
   start-to-finish run without undeclared assistance." If that run has no reset, §10.4,
   pp. 205–206, says the discounted formulation is futile and the objective must be average
   reward — and Exercise 17.1, p. 382, asks the combined options-plus-average-reward question
   the book leaves open. If instead 9.5 is scored as a sequence of resumable milestones (9.3),
   the episodic setting survives and this never arises. Which is it?
8. **What is the correct evaluation-time treatment of a search-augmented policy (R16)?** It is
   a different system under PLAN 8.5. Does it get its own benchmark row, or is it excluded
   from the compact-policy comparison entirely?
9. **Can checkpoint branching (PLAN 10.1) reach an arbitrary mid-episode state?** The current
   reset installs a blueprint in 6.21 ms; a rollout algorithm (§8.10, p. 152) needs to branch
   *from where the agent is*, which is a full-worker save. R16 depends on the answer, and so
   does PLAN 10.2's counterfactual prediction task.
