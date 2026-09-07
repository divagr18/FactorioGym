# Theory sanity check: FactorioRL against Agarwal, Brantley, Jiang, Kakade, Sun

**Source.** *Reinforcement Learning: Theory and Algorithms* (Agarwal, Brantley, Jiang,
Kakade, Sun), working draft dated 2026-06-27, at `D:\Books for RL\rltheorybook_ABJKS.pdf`.
Cited below as **ABJKS**, by printed page and numbered result. The PDF page is the
printed page **+ 12** (printed p. 74 is PDF page 86).

**Scope of this document.** A whole-program review of FactorioRL's RL objectives,
data-collection design, evaluation methodology and phase arc, judged against what this
book actually proves. It is not a bug report and not a book summary.

---

## 0. What theory can and cannot settle here

ABJKS is a book about *lower bounds and guarantees*. It can settle four kinds of question
for us decisively, and it can settle nothing else.

It settles: (a) whether a measurement design is capable in principle of supporting the
conclusion we draw from it; (b) whether a sample budget is within a factor of ten, or a
factor of a million, of any regime where any algorithm has a guarantee; (c) whether an
exploration mechanism can reach a reward at all, which is a probability computation, not
an empirical question; and (d) which of our claims are *impossible* rather than merely
unproven — the impossibility results (Prop. 4.6 p. 57, §4.2 p. 58, Thm. 3.10 p. 45,
§6.2 p. 74, §12.2.2 p. 148) are the sharpest instruments in the book for our purposes.

It cannot settle: whether a 266k-parameter CNN generalizes from `open`/`wall_corridor`
to `wall_arc`. Every guarantee in the book is stated for a class $\mathcal{F}$ or
$\Pi$ under realizability or Bellman-completeness assumptions we have never measured,
and the book is explicit that these assumptions are load-bearing rather than technical
(§3.4, p. 45: linear realizability of $Q^\star$ alone permits $2^{\Omega(d)}$ sample
complexity *even with a generative model*). Neural generalization across layout families
is outside the book entirely. So is hierarchy, options, and anything Phase 8 proposes.
Where the book is silent I say so rather than dressing an analogy as a citation.

One structural mismatch applies to everything below and is stated once. **Every theorem
in ABJKS presumes a Markov state.** Our observation is a 32-tile sensor region plus a
`remembered` store carrying observation age (LEDGER §2.3, §3.4) — the agent's input is
not the state of the Factorio world, and our policy is feedforward. We are running
memoryless policies on a POMDP. This is a defensible engineering order (PLAN §3 defers
recurrence deliberately), but it means every bound below is an *optimistic* bound: it
describes a problem strictly easier than the one we are solving. That caveat belongs in
the release limitations document, not only in this file.

---

## 1. Claims audit

| # | Our claim / method | Verdict | Citation | Corrected wording or method |
|---|---|---|---|---|
| A1 | The feasibility sweep's signal test: "a family shows signal when its held-out Wilson lower bound clears the random baseline's Wilson upper bound" (`tools/feasibility_sweep.py`) | **Unsound** | Union-bound practice throughout ABJKS: Thm. 2.6 p. 26 ($\log c\|S\|\|A\|/\delta$), UCBVI $L:=\ln(2SAHK/\delta)$ p. 75, Lem. 4.5 p. 56 ($\delta \to \delta/(\|\mathcal F\|K)$), Cor. 8.7 p. 100 ($\delta/(TH)$) | Four separate defects — see R1. As executed at $n_{\text{eval}}{=}20$ / $n_{\text{rand}}{=}10$, `supply_furnace` needed **19/20 = 95 % held-out** to be declared "signal", and `deliver` needed **17/20 = 85 %**. Report as: "the sweep had power only against near-perfect held-out performance; it does not distinguish a hard task from an underpowered probe." |
| A2 | "Five of six families cannot be learned as designed, and three never succeed even during training" (LEDGER, Phase 3 correction note) | **Half sound, half overstated; and not reproducible from committed evidence** | §6.2 p. 74; §4.2 p. 58 | The **"never succeed during training"** half is sound and is the real finding: `runtime/runs/sweep-*` shows `mine_smelt`, `repair_belt` and `restore_power` at train 0.00 / held-out 0.00 / **random 0.00**. The **"cannot be learned as designed"** half over-reaches, because §6.2 p. 74 predicts *exactly* this observation for a task that is perfectly learnable but not reachable by undirected exploration — a zero is consistent with "unsolvable through the catalog" and with "$A^{-H}$ unreachable", and the sweep cannot separate them. The catalog-defect diagnosis, not the sweep, is what distinguishes them. Separately: the committed artifact `docs/evidence/phase4-feasibility-sweep.json` contains **3 runs** (`deliver`, `supply_furnace`, `navigate`); the other three live only under gitignored `runtime/runs/`, so the LEDGER's central Phase-3 correction cites evidence a reader cannot reproduce — the same failure mode as the Phase 1 correction note. |
| A3 | 25,000 steps establishes that a family is or is not learnable | **Unsound for negatives, sound for positives** | Thm. 6.1 p. 74 (UCBVI beats the trivial bound only once $K \gtrsim H^2S^2A$); Thm. 12.3 p. 145 (CPI terminates in $\le 8/(\epsilon^2(1-\gamma)^2)$ iterations) | 25k steps is **~97 episodes** for `supply_furnace` (257 steps/ep) and **~106** for `deliver`. With $H{=}300$, $H^2 = 90{,}000$ episodes is the floor at which an *optimal tabular optimistic* algorithm's regret bound first beats "do nothing". A positive result at 97 episodes is real evidence; a negative one is no evidence. Wording for a negative: "no measurable signal at 25k steps; this budget is ~3 orders of magnitude below any regime with a guarantee and cannot distinguish an unlearnable task from an unexplored one." |
| A4 | The 80 % bar on a **structurally held-out layout family**, over 100 episodes × 3 seeds | **Bar sound; attainability claim unsupportable** | Assum. 4.1 p. 54 (concentrability $C$); Prop. 4.6 p. 57 (no coverage ⇒ impossible); Thm. 10.4 p. 122 ($\lVert d^{\pi^\star}_\rho/\mu\rVert_\infty$); Thm. 12.4 p. 145 ($\lVert d^{\pi^\star}/\mu\rVert_\infty$); Thm. 11.6 p. 136 ($\epsilon_{\text{approx}}$ transferred by $d^\star/\sigma$) | Do **not** lower the bar. Do change how the number is framed. Training never visits `wall_arc`, so the density ratio $d^{\pi}(s,a)/\mu(s,a)$ is $+\infty$ there and *every* performance bound in the book that applies to our algorithm family becomes vacuous. The 80 % number is therefore a claim about neural generalization, not about RL. Report it as **"out-of-support structural transfer"** with an explicit line: "no guarantee in the policy-optimization literature covers this measurement; it is an empirical generalization result." |
| A5 | PPO's entropy bonus is our exploration mechanism, on `restore_power` (sparse only, $\lvert A\rvert{=}17$, $H{=}300$) and `repair_belt` ($\lvert A\rvert{=}18$, $H{=}300$) | **Unsound** | §6.2 p. 74 (uniform action noise reaches the goal w.p. $A^{-H}$; expected episodes to first reward $\sim A^H$); §4.2 p. 58 (needle-in-a-haystack $\Omega(\lvert A\rvert^H)$ lower bound even with a generative model); Prop. 10.1 p. 120 (vanishing gradients: $\lVert\nabla^k V\rVert \le (1/3)^{H/4}$ at suboptimal parameters) | `restore_power` has **no shaping component at all** (only `step_cost`), so it is exactly the chain MDP of Fig. 6.1/10.1. Undirected exploration will not find the reward at any budget this project can afford, and Prop. 10.1 says the gradient is exponentially small *even computed exactly*. This is a proof, not a hypothesis. Either give it a dense reachable signal, or seed it from a teacher (R6), or drop it from the Phase 4.5 candidate set with this citation as the reason. |
| A6 | "Potential-based shaping is $\gamma\phi(s')-\phi(s)$, so any closed loop in state space sums to zero" (`src/factoriorl/rewards.py` module docstring) | **Unsound as written** | Lem. 1.16 p. 19 (PDL) | LEDGER §3.5 already corrected this to $(\gamma-1)\sum\phi < 0$ and the test asserts the right property; the code docstring still says "sums to zero" and is the version a reader meets first. Separately: policy invariance of potential shaping additionally requires $\phi \equiv 0$ at terminal/absorbing states, which `RewardAccountant` never enforces — the terminal transition pays $\gamma\phi(s_T)-\phi(s_{T-1})$ with $\phi(s_T) \ne 0$. Our shaping is therefore **not** invariant, only close to it. |
| A7 | "High-water shaping makes the anti-exploit criteria structurally true" (`rewards.py`, LEDGER §3.5) | **Overstated** | Lem. 1.16 p. 19; Thm. 4.3 p. 55 (the objective you optimize is the objective you get) | High-water shaping is monotone but **not** potential-based, so it changes the optimal policy of the shaped MDP. `supply_furnace` pays 0.15 per plate high-water against a sparse success of 1.0: **7 plates out-earn success**. `mine_smelt` pays 0.2 per plate: **5 plates out-earn success**. A policy that farms the shaping term and never satisfies the predicate is optimal for the reward we hand PPO. Correct wording: "high-water shaping prevents *double* payment for the same progress; it does not preserve the sparse optimum, and the weights are not currently checked against the sparse term." |
| A8 | $\gamma = 0.99$ with `max_decision_steps` of 150–300 | **Unsound parameterization** | Thm. 2.8 p. 28 ("one rough correspondence is $H \approx 1/(1-\gamma)$"); p. 27 on horizon normalization | $1/(1-\gamma) = 100$ against $H = 250$–$300$. Terminal success is discounted to $0.99^{250} = 0.081$ and $0.99^{300} = 0.049$, while `step_cost` at 0.001/step is paid essentially undiscounted over the same window (≈0.22 total). For `deliver`, `supply_furnace`, `repair_belt` and `restore_power` the **discounted optimum can prefer failing early to succeeding late**. This is a reward-specification defect, not a tuning knob. |
| A9 | All Phase 4 evaluation runs on `split="test"`; the `val` split exists per family and is used nowhere | **Unsound — leakage** | ABJKS is silent on protocol, but PLAN §3 states the rule; the sweep then selected candidates on those numbers | `train.py:309,313` hardcode `split="test"`. `tools/feasibility_sweep.py` used those test-split numbers to choose which families get the long Phase 4.5 runs. That is model/task selection performed on the frozen test split, which PLAN §3 forbids ("Test results use frozen layouts and seeds excluded from tuning"). Every family already touched (`navigate`, `deliver`, `supply_furnace`) has a contaminated test split for selection purposes. See R2. |
| A10 | The random baseline, at `min(eval_episodes, 10)` episodes | **Unsound** | Lem. A.1 p. 189 (Hoeffding: half-width $\propto \sqrt{\log(2/\delta)/2n}$) | 10 episodes gives a Wilson upper bound of 0.2775 at 0/10 and **0.7634 at 5/10**. This single choice is what made the sweep's signal test nearly unpassable. The run directories show the noise directly: two `supply_furnace` runs at the same nominal seed produced random baselines of **0.30 and 0.50**, and two `deliver` runs produced held-out **0.00 and 0.10**. Baselines are cheap (no gradient step); run them at the same $n$ as the policy, minimum 100. Also note `supply_furnace`'s random baseline of **5/10** — a task a uniform-random masked policy solves half the time is weak evidence for a learning claim regardless of the bar. |
| A11 | The random baseline is compared to the policy as two independent Wilson intervals | **Unsound method (the data are paired)** | Lem. A.1 p. 189; general union-bound practice | Both envs are constructed with the same `SeedPlan`, `Branch.EVAL` and `split="test"`, and both start at `_episode_index = 0`. The baseline therefore runs on **exactly the same scenes** as the policy's first 10 eval episodes. Paired data analysed as unpaired throws away most of the power. Use McNemar / a paired bootstrap on the per-episode indicator differences. |
| A12 | Action masking applied every step, built from the observation alone | **Sound as an MDP restriction; voids two named guarantees** | Thm. 9.4 p. 112 (PG theorem); Thm. 10.3 p. 121 (softmax global convergence requires $\mu$ full support); §10.3 p. 121–122 (log-barrier exists precisely to stop probabilities collapsing) | The mask is a deterministic function of the observation (`env.action_masks()` uses `_context()` only), so the policy-gradient identities of Thm. 9.4 still hold on the restricted MDP with state-dependent $A(s)$ — this is the right design and the "masks use no privileged information" test is genuinely valuable. But masking sets $\pi(a\mid s) = 0$ exactly, which is the failure mode Thm. 10.3's full-support condition and §10.3's log barrier are built to prevent, and it makes the entropy bonus a bonus over $\lvert A(s)\rvert$, not $\lvert A\rvert$ — a varying, uncontrolled quantity. State it as a limitation; do not remove masking. |
| A13 | PPO gives monotonic policy improvement / trust-region behaviour | **Unsound if claimed** | §12.2.2 p. 148, verbatim: "The practical version of TRPO and PPO, in general, cannot guarantee monotonic improvement because they relax the trust region to a constraint averaged over the state distribution" — with an explicit counterexample | The counterexample is *exactly* our situation: the new policy deviates arbitrarily at states the old policy never visits, i.e. the held-out layout family. Never write "PPO improves monotonically" or "the trust region keeps the policy near the previous one" in a report. |
| A14 | On-policy PPO discards each transition after one update | **Sound but wasteful; a measured opportunity** | Thm. 4.3 p. 55 (FQI works offline given coverage $C$ and $\epsilon_{\text{approx},\nu}$); §4.2 p. 58 (naive importance-sampled policy selection costs $\lvert A\rvert^H$) | The book supports both halves: reusing data via *importance-weighted policy search* is exponentially bad in $H$ (§4.2 p. 58), but reusing it via *fitted regression* (FQI, Thm. 4.3) is fine provided the logging mixture covers the target occupancy. At 24 ms/env-step, every discarded transition is real money. See R7. |
| A15 | Function-approximation error is never measured | **Unsound omission** | Assum. 4.2 p. 54 ($\epsilon_{\text{approx},\nu}$); Thm. 4.3 p. 55 ("even with infinite data, lack of Bellman closure creates an error floor"); Thm. 11.6 p. 136 (approximation term transferred by $d^\star/\sigma$) | We have a 266k-parameter encoder and no measurement of whether it can represent $Q^\star$ or is closed under Bellman backups. Both the offline (Thm. 4.3) and the NPG (Thm. 11.6) bounds contain an *additive, irreducible* term for this that no amount of data removes. The residual is cheap to measure — see R8. |
| A16 | Evaluation uses `deterministic=True` (argmax) while training optimises the stochastic policy | **Overstated equivalence** | Thm. 9.4 p. 112, Thm. 10.3 p. 121, Thm. 11.6 p. 136 all bound $V^{\pi_\theta}$, never $V^{\arg\max \pi_\theta}$ | Nothing guarantees the greedy policy is at least as good as the sampled one; on a POMDP it can be strictly worse (deterministic memoryless policies are a strictly weaker class). Report both, or state that the reported number is for the greedy policy and is not the quantity PPO optimised. |
| A17 | Phase 9: multi-hour horizons "toward a rocket", with policies at the 30-tick primitive granularity | **Unsupportable as an RL claim** | Thm. 2.8 p. 28 ($H^3\lvert S\rvert\lvert A\rvert/\epsilon^2$ *with a generative model*); Cor. 8.8 p. 101 ($m = \tilde O(d^2H^5/\epsilon^2)$, $T=\tilde O(Hd)$, total $\tilde O(d^3H^7/\epsilon^2)$ trajectories); §4.2 p. 58 ($\lvert A\rvert^H$) | One game hour at 30 ticks/decision is $\approx$ 7,200 decisions; a rocket run is $10^5$–$10^6$. Every horizon exponent in the book — $H^3$ at best with a *simulator oracle*, $H^5$–$H^7$ under Bellman rank, $\lvert A\rvert^H$ agnostically — makes primitive-granularity learning at that horizon impossible by orders of magnitude no engineering can close. Phase 9 must be framed as **planning + scripted/learned skills with RL confined to short-horizon sub-problems**, which is what Phase 8 builds. Say "documented progression achieved by a hybrid system", never "learned". |
| A18 | Phase 8: learned skills and hybrid planning will show "a measurable benefit" | **Not covered by the book; the strongest *available* argument is horizon reduction** | Cor. 8.8 p. 101 ($H^5$ in $m$); §8.4.2 p. 102 (Q-state abstraction: complexity in $\lvert Z\rvert$ not $\lvert S\rvert$); §8.4.3 p. 102 (low-occupancy structure) | ABJKS covers no hierarchy, options, or semi-MDPs. What it does say is that sample complexity is polynomial with a *high* exponent in $H$ and in an abstraction's dimension, not the raw state count. Cutting $H$ from 300 primitive steps to ~10 skill decisions is a $30^5 \approx 2.4\times10^7$ reduction in the leading $H^5$ term. That reframes Phase 8 from "an interesting comparison" to **the only route by which Phase 9 is reachable at all**, and it justifies moving skill work earlier rather than treating it as optional. |
| A19 | Per-seed Wilson 95 % CIs plus a pooled aggregate; composite scores forbidden | **Sound**, and better than common practice | Lem. A.1 p. 189; §2.5 / Ch. 2 on reporting $\epsilon,\delta$ | Keep exactly as is, with the multiplicity correction of R1. PLAN §3's ban on composite scores is directly supported: Thm. 2.6/2.8 and Thm. 4.3 all state $\epsilon$ against a *named* quantity; a composite has no such quantity and no interval means anything under it. |
| A20 | Three training seeds | **Sound as a minimum, insufficient for a variance claim** | Thm. 9.7 p. 115 (the objective is non-concave; distinct local optima exist); Lem. 9.9 p. 116 (SGD error floor $\sigma^2$ that does not vanish with $T$) | Three seeds supports "at least one seed reached X" and a crude spread. It does **not** support "the method achieves X" or any seed-variance interval. Publish per-seed rates and the min, and never report a mean-over-seeds without the per-seed values beside it. PLAN §3 already requires this. |
| A21 | The scripted reference solutions exist only as a solvability checker (`src/factoriorl/tasks/reference.py`) | **A sound tool, badly underused** | Thm. 13.3 p. 162 (BC: $\frac{\gamma}{(1-\gamma)^2}\sqrt{\ln(\lvert\Pi\rvert/\delta)/M}$); Thm. 13.4 p. 163 (DM-ST: **linear** in $1/(1-\gamma)$); Thm. 13.6 p. 169 (AggreVaTe) | This is the single largest untapped sample-complexity lever in the repository. See R6. Ch. 13's whole point is that a queryable expert converts an exploration problem into a supervised-learning problem with no $\lvert A\rvert^H$ term anywhere. |
| A22 | "~75 env steps/s across 8 engines" (planning figure) | **Inconsistent with the recorded evidence** | — | `docs/evidence/phase4-worker-profile.json`: 8 workers = **202.18 steps/s** env-only. `docs/evidence/phase4-feasibility-sweep.json`: end-to-end training measured **17.3–21.7 steps/s** at 1 worker. 75 is neither. Pin one number per purpose (env-only vs end-to-end, per worker count) before any budget in §2 is quoted in a report. |

---

## 2. Sample budget

Three rates are used, all from our own evidence, because they differ by 10×:

* **202 steps/s** — env-only, 8 workers (`phase4-worker-profile.json`).
* **75 steps/s** — the current planning figure (provenance unclear, see A22).
* **20 steps/s** — end-to-end training, 1 worker, as actually measured in the sweep.

| Budget | @202 /s | @75 /s | @20 /s |
|---|---|---|---|
| 25,000 | 2 min | 6 min | 21 min |
| 300,000 | 0.4 h | 1.1 h | 4.2 h |
| 1,000,000 | 1.4 h | 3.7 h | 13.9 h |
| 3,000,000 | 4.1 h | 11.1 h | 41.7 h |
| 10,000,000 | 13.8 h | 37 h | 5.8 d |
| 100,000,000 | 5.7 d | 15.4 d | 58 d |

### What the theory asks for

There is no theorem that gives "the budget for MaskablePPO on a Factorio POMDP". What
exists are three anchors, each of which is a *lower* bound on what an idealised method
needs, and therefore a floor for us.

**Anchor 1 — episodes, not steps, are the unit (Thm. 6.1 p. 74).** UCBVI's regret bound
falls below the trivial bound $KH$ only once $K \gtrsim H^2 S^2 A$. Dropping the $S^2$
term (which is astronomical for a $6\times65\times65$ grid, so this is generous beyond
absurdity), $H^2$ alone is **22,500 episodes for `navigate`** ($H{=}150$) and **90,000
episodes for the $H{=}300$ families**. At the measured episode lengths that is:

| Family | $H$ | mean ep. steps (measured) | $H^2$ episodes | steps | @75 /s |
|---|---|---|---|---|---|
| `navigate` | 150 | 16.1 | 22,500 | 0.36 M | 1.3 h |
| `deliver` | 250 | 235.7 | 62,500 | 14.7 M | 55 h |
| `supply_furnace` | 300 | 257.0 | 90,000 | 23.1 M | 86 h |
| `repair_belt` | 300 | ~300 (never succeeds) | 90,000 | 27 M | 100 h |
| `restore_power` | 300 | ~300 (never succeeds) | 90,000 | 27 M | 100 h |
| `mine_smelt` | 400 | ~400 (never succeeds) | 160,000 | 64 M | 237 h |

This is where an *optimal, strategically-exploring, tabular* method's guarantee first
becomes non-vacuous. We are running a non-optimal, non-strategic, function-approximating
method on a POMDP. Treat these as a floor with no ceiling attached.

**Anchor 2 — function-approximation PAC (Cor. 8.8 p. 101).** Under bounded Bellman rank
$d$ with linear Bellman completeness, $m = \tilde O(d^2H^5/\epsilon^2)$ per stage and
$T = \tilde O(Hd)$ iterations, so $\tilde O(d^3H^7/\epsilon^2)$ trajectories. Even at a
fictitiously small $d = 10$ and $\epsilon = 0.2$, $H = 300$ gives $H^7 = 2\times10^{17}$.
The number is not the point; the **exponent** is. It says the only variable we can move
is $H$, and moving it requires temporal abstraction (A18).

**Anchor 3 — imitation (Thm. 13.3 p. 162, Thm. 13.4 p. 163).** BC needs
$M \gtrsim \ln(\lvert\Pi\rvert/\delta)\gamma^2/((1-\gamma)^4\epsilon^2)$ expert
state-action pairs, with **no exploration term at all** and no dependence on $\lvert A\rvert^H$.
With a known/queryable simulator the dependence drops from $1/(1-\gamma)^2$ to
$1/(1-\gamma)$ (Thm. 13.4). The scripted solvers we already have make this the cheapest
budget in the whole program — a few thousand demonstration episodes, hours not weeks.

### Per-phase budget

**Phase 4 (current).**

* *Feasibility probes:* 25k steps is fine to detect a strong positive and worthless as a
  negative (A3). Keep them, relabel their output, and run them on **`val`** (R2).
* *A defensible negative result* needs the Anchor-1 floor: **~0.4 M steps for `navigate`,
  ~15–27 M for the $H{=}300$ families**. At 75 steps/s the latter is 55–100 h per family
  per seed. Six families × 3 seeds at that budget is **~2,000 h ≈ 3 months of wall clock**.
  This is the arithmetic that PLAN 4.5's "overnight introductory training recipe" has to
  survive, and at present it does not.
* *The honest Phase 4.5 plan* that fits the machine: pick the families whose measured
  episode length is short. `navigate` at 16 steps/episode is 1.3 h/seed to the Anchor-1
  floor. The $H{=}300$ families are not overnight-trainable at primitive granularity, and
  the correct response is R6 (teacher-seeded) or a shorter-horizon task variant with its
  own version — **not** a lower bar.
* *Evaluation:* 100 held-out episodes × 3 seeds × 6 families = 1,800 episodes. At ~250
  steps/episode that is 450k steps ≈ **1.7 h at 75 /s**. Evaluation is cheap; the random
  and scripted baselines at the same $n$ add the same again. There is no reason for the
  10-episode baseline (A10).

**Phase 8 (skills and hierarchy).** Budget per skill at the Phase-4 rate but with the
skill's own short $H$ — a navigate-class skill at $H\approx 30$ has $H^2 = 900$ episodes
≈ 15k steps ≈ **3 min at 75 /s**. That is the entire argument for hierarchy: four skills
cost less than one flat $H{=}300$ family. The manager then runs at $H \approx 10$–20 skill
decisions; its own Anchor-1 floor is ~400 episodes. Budget **~2 M steps total for the
skill library plus manager**, i.e. ~7 h — versus ~100 h for one flat family. Phase 8 is
the cheap phase, not the expensive one, and the plan should say so.

**Phase 9 (rocket progression).** No RL budget is defensible at primitive granularity
(A17). Budget it as *planning wall-clock and game time*, with RL appearing only as
Phase-8 skills, each with its own Phase-4-style budget. Any figure of the form "N steps
to a rocket" should not be published.

---

## 3. Recommendations

Each is: citation → repo change → phase → expected impact → cost.

**R1. Fix the four defects in the signal test.**
*Citation:* union-bound practice, Thm. 2.6 p. 26, Lem. 4.5 p. 56, Cor. 8.7 p. 100; Lem. A.1 p. 189.
*Change:* in `tools/feasibility_sweep.py` and `learn/train.py::_wilson`,
(i) take `z` as a parameter and pass the Bonferroni value for the number of intervals in
the report — $z = 2.394$ for 3, $2.638$ for 6, **$2.991$ for 6 families × 3 seeds**;
(ii) run the random baseline at the same $n$ as the policy (delete the `min(..., 10)`);
(iii) replace the non-overlap heuristic with a paired McNemar test on the per-episode
indicators, since the two envs already run identical scenes (A11);
(iv) record in the report the *smallest held-out success rate the test could have
detected*, so a null result is legible.
*Phase:* 4 (now). *Impact:* the difference between "five families cannot be learned" and
"we could not have detected it". *Cost:* under a day; baselines add ~1 h of wall clock.

**R2. Move all model and task selection to the `val` split.**
*Citation:* PLAN §3; the leakage is a protocol failure the book cannot bless (A9).
*Change:* add a `split` parameter to `TrainConfig`, default `"val"`; `train.py:309,313`
stop hardcoding `"test"`; the sweep and `shaping_comparison.py` run on `val`; only the
Phase 4.5 release run touches `test`, once, after the configuration is frozen. Record in
`docs/LEDGER.md` that the sweep already consumed the test split for selection on
`navigate`/`deliver`/`supply_furnace`.
*Phase:* 4 (now, before 4.5). *Impact:* makes the release number mean what PLAN §3 says
it means. *Cost:* hours. The `val` families already exist and are unused.

**R3. Report the structural holdout as an out-of-support transfer measurement.**
*Citation:* Assum. 4.1 p. 54; Prop. 4.6 p. 57; Thm. 10.4 p. 122; Thm. 12.4 p. 145; §12.2.2 p. 148.
*Change:* every result table gains **three** rows per family, not one — train-family
in-distribution, `val` (unseen seeds, seen structure), `test` (unseen structure). PLAN §3
already demands "unfamiliar seeds and unfamiliar structures reported separately" and the
code currently reports neither separately. Add a standing sentence to the results
document: "the training occupancy assigns zero mass to the held-out family, so the
concentrability coefficient of Assum. 4.1 (ABJKS p. 54) is unbounded and no policy-
optimization guarantee applies; this row measures neural generalization."
*Phase:* 4, permanent. *Impact:* the single highest-value honesty change in the audit; it
also makes the gap between the seed-holdout and structure-holdout rows the most
informative number in the report. *Cost:* one extra eval env per family, ~1 h wall clock
per release run. **Do not lower the 80 % bar** — report the seed-holdout row alongside it
so that a structural failure is diagnosable rather than merely disqualifying.

**R4. Fix the discount/horizon mismatch, or state it.**
*Citation:* Thm. 2.8 p. 28 ($H \approx 1/(1-\gamma)$); p. 27 on normalization.
*Change:* set $\gamma$ per task as $1 - 1/H$ (0.9933 at $H{=}150$, 0.9967 at $H{=}300$)
and record it in the manifest; **and** couple `rewards.GAMMA` to `TrainConfig.gamma`
rather than a module constant, since potential shaping's near-invariance depends on the
two matching. Alternatively keep $\gamma{=}0.99$ and record explicitly that the objective
discounts terminal success to 0.05–0.08 and is therefore not the success rate we report.
*Phase:* 4. *Impact:* on the $H{=}300$ families this may be the whole difference between a
policy that tries and one that gives up. *Cost:* a config change plus a re-run.

**R5. Audit shaping weights against the sparse term, and zero the terminal potential.**
*Citation:* Lem. 1.16 p. 19; Thm. 4.3 p. 55.
*Change:* add a validation check in `tasks/__init__.py::validate` asserting that the
maximum achievable sum of shaping components is strictly less than the sparse success
weight — `supply_furnace` (0.15 × plates) and `mine_smelt` (0.2 × plates) currently fail
this. Add `phi(terminal) = 0` to `RewardAccountant.step` so potential shaping is actually
invariant. Correct the `rewards.py` docstring's "sums to zero" (A6).
*Phase:* 4 (validation runs before any worker launches, so this is free to check).
*Impact:* removes a class of exploit PLAN 3.5 claims is already excluded. *Cost:* hours.

**R6. Promote the scripted solvers from checkers to teachers.**
*Citation:* Thm. 13.3 p. 162 (BC), Thm. 13.4 p. 163 (linear-in-horizon with known
transitions — and our transitions *are* queryable: we own the simulator), Thm. 13.6 p. 169
(AggreVaTe), §13.5 p. 168.
*Change:* a new `factoriorl.learn.imitate` module that (a) rolls the reference solver to
produce $(s,a)$ pairs on the **train** families, (b) behaviour-clones the same encoder,
(c) optionally continues with DAgger/AggreVaTe-style relabelling on states the learner
visits. PLAN 4.2's rule — "the agent receives no scripted solution labels during ordinary
PPO training" — is preserved by making this a *separate, declared* training mode with its
own manifest field and its own results column, never mixed into the PPO baseline.
*Phase:* 4 for `deliver`/`supply_furnace`; prerequisite for 8 and 9.
*Impact:* the largest in this document. It replaces an $\lvert A\rvert^H$ exploration
problem with an MLE problem whose sample complexity is $\ln\lvert\Pi\rvert/M$ and no
horizon exponentiation. It is also the only mechanism in the book that can make
`restore_power` learnable. *Cost:* ~1 week; a few thousand demonstration episodes is hours
of wall clock. Note the honest caveat Thm. 13.3 carries: BC's error compounds as
$1/(1-\gamma)^2$ under its own state distribution, which is why (c) matters.

**R7. Keep every transition; add an offline-evaluation path over the logged data.**
*Citation:* Thm. 4.3 p. 55 (FQI under coverage $C$ and $\epsilon_{\text{approx},\nu}$);
§4.2 p. 58 (do **not** do this with importance-weighted policy search).
*Change:* write rollout transitions to a compressed on-disk buffer alongside `curve.csv`
(they are already produced and thrown away at 24 ms each). Use it for FQI-style offline
policy evaluation and for the $\epsilon_{\text{approx}}$ measurement of R8. Do **not** use
it for off-policy PPO with importance weights.
*Phase:* 4 to build, 7/8 to exploit (persistent-factory data is far more expensive to
regenerate). *Impact:* turns a 100-hour training run into a reusable dataset; PLAN 10.4
already wants dataset export and this is its substrate. *Cost:* days; disk.

**R8. Measure the inherent Bellman error of the encoder.**
*Citation:* Assum. 4.2 p. 54; Thm. 4.3 p. 55; Thm. 11.6 p. 136.
*Change:* a `tools/bellman_residual.py` that, on held-out logged transitions, computes
$\lVert \hat f - \mathcal{T}\hat f\rVert_{2,\nu}$ for the trained critic. Report it beside
every success rate.
*Phase:* 4. *Impact:* distinguishes "not enough data" from "this network cannot represent
the value function", which is currently indistinguishable and is an *additive irreducible
floor* in both Thm. 4.3 and Thm. 11.6. *Cost:* ~2 days, no extra engine time (uses R7's
buffer).

**R9. Give `restore_power` a reachable signal, or remove it from the Phase 4.5 candidate
set with the citation.**
*Citation:* §6.2 p. 74; §4.2 p. 58; Prop. 10.1 p. 120.
*Change:* it is currently sparse-only at $H{=}300$, $\lvert A\rvert{=}17$ — the chain MDP.
Either add a potential term over distance-to-gap (bounded, terminal-zeroed per R5), or
seed it by R6, or record in the LEDGER: "excluded from the Phase 4.5 candidate set;
undirected exploration reaches the reward with probability $\approx 17^{-300}$ (ABJKS §6.2
p. 74), and the gradient is exponentially small even computed exactly (Prop. 10.1 p. 120)."
*Phase:* 4. *Impact:* stops a guaranteed-failing run from consuming 100 h of the budget.
*Cost:* an hour to write down; a day to shape.

**R10. Pre-register the Phase 4.5 protocol before the release run.**
*Citation:* the union-bound discipline of Cor. 8.7 p. 100 / Lem. 4.5 p. 56 — the
correction depends on how many comparisons you *could* have made, which is only knowable
in advance.
*Change:* a committed `docs/evidence/phase4-5-protocol.md` fixing, before any test-split
episode runs: the families, the seeds, the number of eval episodes, $z$, the multiplicity
count, and the success criterion. Then run once.
*Phase:* 4.5. *Impact:* makes the release result a confirmatory measurement rather than
the maximum over an unrecorded search. *Cost:* an afternoon.

**R11. Report the stochastic-policy success rate alongside the greedy one.**
*Citation:* Thm. 9.4 p. 112, Thm. 10.3 p. 121, Thm. 11.6 p. 136 — all bound $V^{\pi_\theta}$.
*Change:* `evaluate()` gains a second pass with `deterministic=False`; both go in the
results table.
*Phase:* 4. *Impact:* small but removes an unstated substitution between the trained and
the reported object; a large gap between the two is itself a diagnostic. *Cost:* doubles
evaluation wall clock (~1 h per release run).

**R12. State the POMDP caveat once, structurally.**
*Citation:* Ch. 1 p. 3–9 (every result is for an MDP); §13.2 p. 162 on train/test
distribution mismatch.
*Change:* one paragraph in the release limitations document (PLAN 6.2) and one in
`docs/LEDGER.md`: the observation profile is partial, the policy is memoryless, and
therefore no cited guarantee applies without an additional assumption we have not tested.
*Phase:* 4/6. *Impact:* prevents a whole family of overclaims later. *Cost:* an hour.

**R13. Reframe Phase 8 as horizon reduction and move it earlier where possible.**
*Citation:* Cor. 8.8 p. 101 ($H^5$); §8.4.2 p. 102; §8.4.3 p. 102; Thm. 2.8 p. 28 ($H^3$).
*Change:* PLAN 8.5's acceptance ("demonstrate a measurable benefit in success, training
efficiency, execution time, or inference cost") should name **training sample complexity
at fixed success** as the primary metric, with the $H$-reduction stated as the mechanism.
The measurement is: steps-to-80 % for a flat policy on a composite task vs. steps-to-80 %
for skills + manager, counting skill-training steps in the hierarchical total.
*Phase:* 8. *Impact:* converts a vague comparison into a falsifiable claim with a
predicted direction. *Cost:* planning only.

**R14. Stop quoting a single throughput number.**
*Citation:* n/a — this is PLAN 4.3 ("more workers are not described as faster unless
measurements show it") and PLAN 11.4 applied to our own planning figures.
*Change:* the manifest already records worker count; add an `end_to_end_steps_per_second`
field distinct from the env-only figure, and quote the pair. Resolve the 202 / 75 / 20
discrepancy (A22) in the LEDGER.
*Phase:* 4. *Impact:* every budget in §2 above changes by 10× depending on which number is
meant. *Cost:* an hour.

**R15. Record `supply_furnace`'s random baseline as a task-design finding.**
*Citation:* Thm. 2.7 p. 27 / the general principle that a bound is only as meaningful as
the gap it certifies.
*Change:* a uniform-random masked policy solved `supply_furnace` **5/10** on the held-out
split. An 80 % bar over a 50 % floor certifies far less than an 80 % bar over a 0 % floor.
Either tighten the generator (more distant ore, larger arena) with a version bump, or
report the random baseline immediately adjacent to every success rate so a reader can see
the gap. PLAN §3 already requires publishing random baselines; make adjacency mandatory.
*Phase:* 4. *Impact:* prevents the release's headline family from being the weakest
evidence in it. *Cost:* hours to report; a task version bump if tightened.

---

## 4. Do NOT do this

1. **Do not lower the 80 % bar, the 100-episode count, or the 3-seed count.** PLAN §4
   forbids it and nothing in the book argues for it. If the bar is unreachable on a
   structural holdout, the correct action is R3 (report the seed-holdout row beside it),
   not a smaller number.
2. **Do not claim PPO has a trust region or monotonic improvement.** §12.2.2 p. 148 gives
   an explicit counterexample, and its mechanism — deviation at unvisited states — is
   precisely our held-out family.
3. **Do not "fix" exploration by raising `ent_coef`.** §6.2 p. 74 shows the failure is
   $A^{-H}$ reachability, not insufficient noise; more entropy under a mask makes the
   policy uniform over $\lvert A(s)\rvert$ and does not change the exponent.
4. **Do not reuse old rollouts via importance-weighted policy search.** §4.2 p. 58: the
   variance is $\lvert A\rvert^H$. Reuse them via fitted regression (R7) instead.
5. **Do not remove action masking to "restore" the softmax convergence theorem.** The
   guarantee (Thm. 10.3 p. 121) is asymptotic and, per the book, "believed to be extremely
   slow". Masking is the better trade; document the caveat (A12) and keep it.
6. **Do not report a single pooled success rate without per-seed values.** Thm. 9.7 p. 115
   (non-concavity) means seeds can land in genuinely different optima; a mean hides that.
7. **Do not select which families to run long on the `test` split.** That is A9; it is
   already partially done and needs recording, not repeating.
8. **Do not write "N steps to a rocket" or any Phase 9 sample-complexity figure at
   primitive granularity.** A17.
9. **Do not treat the scripted solver as a baseline only.** PLAN 4.2's exclusion is about
   ordinary *PPO* training; forgoing imitation entirely discards the one lever with no
   horizon exponent (Ch. 13).
10. **Do not add a composite score, even "for convenience".** PLAN §3 forbids it and every
    $\epsilon$ in the book is stated against a named quantity.
11. **Do not describe a 25k-step null result as a task-difficulty finding.** A3.
12. **Do not tune anything against the structural holdout.** It is the one measurement
    whose value comes entirely from never having been optimized against; a single
    hyperparameter chosen on it destroys the only distribution-shift evidence the project
    has.

---

## 5. Open questions

1. **What is our actual $\epsilon_{\text{approx}}$?** (Assum. 4.2 p. 54.) Unmeasured, and
   it is an irreducible additive floor in Thm. 4.3 and Thm. 11.6. R8 answers it.
2. **Is the observation Markov *enough*?** The `remembered` store with age is an attempt at
   sufficiency. Nothing tests it. A cheap probe: train an identical policy with and without
   the `remembered` block and compare — if there is no difference, either memory is
   unnecessary or the encoder is not using it, and both are worth knowing.
3. **Does the structural holdout preserve the optimal-policy class?** If `wall_arc` requires
   an action sequence structurally absent from `open`/`wall_corridor` optima, no amount of
   generalization helps and the measurement is testing the wrong thing. The scripted solver
   can answer this: compare its action histograms across families.
4. **Is `deliver`'s 0.65 train / 0.10 held-out gap distribution shift or overfitting to
   layout?** These have different fixes. The `val` split (unseen seeds, seen structure)
   separates them in one run and is currently never used (R2).
5. **What is the concentrability coefficient $C$ (Assum. 4.1) between our training families
   and our `val` families?** For the `test` family it is $+\infty$ by construction, but for
   `val` it is finite and estimable from R7's buffer. That number would be the first
   quantitative handle we have on how much shift we are asking for.
6. **Does the `_context()` late binding break the fixed action semantics?** Action index $k$
   binds to "nearest entity" / "nearest resource", so the same index means different things
   in different states. It remains a valid MDP (the binding is a function of the observation),
   but it makes $\lvert A\rvert$ a poor proxy for the true branching factor, and the
   catalog-digest reproducibility argument covers the index, not the semantics.
7. **What is the right multiplicity count for the release?** Bonferroni over 18 ($z=2.99$)
   is defensible if 6 families × 3 seeds are all reported. If the sweep already narrowed to
   3 families using the test split, the honest count is larger, not smaller. R10 makes this
   answerable in advance rather than arguable afterwards.
8. **Is there a shorter-horizon variant of the $H{=}300$ families** that keeps the mechanic
   and lands inside the overnight budget? PLAN 4.5's overnight target and the Anchor-1 floor
   are currently incompatible (§2), and a versioned easier variant with an honest name is a
   legitimate answer where lowering the bar is not.
9. **Where does Phase 7's persistent-factory episode fit any of this?** Persistent episodes
   have no reset, so the episodic sampling model of §1.4 p. 18 does not apply, and neither
   does anything downstream of it. The book has no framework for this and the plan should
   say what measurement replaces success rate.
10. **Why does the committed sweep artifact hold only 3 of the 6 families?**
    `docs/evidence/phase4-feasibility-sweep.json` records `deliver`, `supply_furnace` and
    `navigate`. The runs for `mine_smelt`, `repair_belt` and `restore_power` exist under
    gitignored `runtime/runs/sweep-*` (all at train 0.00 / held-out 0.00 / random 0.00) but
    were never rolled into the artifact, so the LEDGER's "five of six families" finding is
    not reproducible from the repository. This is the Phase-1 correction note's failure mode
    recurring: a ledger entry ahead of its committed evidence. Fixing it is a
    `CONTRIBUTING.md` process question, not a theory one, but it gates every claim above
    that depends on the sweep.
11. **Do the three zero-success families fail for the same reason?** `mine_smelt`
    ($H{=}400$), `repair_belt` and `restore_power` all read 0.00 across train, held-out and
    random. The catalog `$ahead` defect explains the two placement families; `mine_smelt`
    places nothing, so it needs its own diagnosis. The scripted solver (R6) answers this
    mechanically and is the intended instrument (PLAN 3.2).
