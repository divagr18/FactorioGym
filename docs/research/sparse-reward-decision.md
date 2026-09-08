# What to do about `repair_belt` and `restore_power`

A decision note, written against the five books already parsed into this
directory, after the PLAN 4.5 matrix measured `repair_belt` at 0.00 on every row
and every seed.

The measurement, first, because everything below is a response to it:

```
repair_belt, 50,000 steps, seeds 1 and 2
  190 and 199 episodes logged
  0 successes, ever
  mean episode reward -0.300, flat from the first segment to the last
  step cost over a full 300-decision episode: 300 x 0.001 = 0.300
```

The reward earned over the entire run was exactly the step cost. Nothing else
ever paid.

---

## 1. The notes predicted this, and I did not act on it

`algorithms-for-decision-making.md` §11, written before any of these runs:

> `repair_belt` pays sparse success plus a high-water counter that is pinned at
> zero until the repair is already done, plus a step cost — so the reward is
> *negative and constant* over the entire exploration phase. `restore_power` is
> worse still: sparse success and step cost, **no shaping component at all**.
> Undirected exploration cannot find either goal.

"Negative and constant over the entire exploration phase" is the measurement,
predicted in advance. The recommendation was filed under Phase 4.4/4.5 and the
matrix was launched without it.

---

## 2. My proposed fix is the one Sutton and Barto warn against

My instinct after the measurement was to add intermediate rewards -- pay for a
contiguous belt line, for a pole placed, for progress toward the gap. That is
precisely the move §17.4 (p. 386) cautions against:

> It is tempting to address the sparse reward problem by rewarding the agent for
> achieving subgoals that the designer thinks are important way stations to the
> overall goal. But augmenting the reward signal with well-intentioned
> supplemental rewards may lead the agent to behave very differently from what
> is intended … **A better way to provide such guidance is to leave the reward
> signal alone and instead augment the value-function approximation with an
> initial guess of what it should ultimately be.**

This repo has already been bitten by exactly the failure that warning describes:
`deliver` seed 3 converged on `take_iron-plate_20` and never delivered, because
a well-intentioned `carried` term paid for holding plates. Adding two more
hand-designed subgoal rewards to families whose geometry I understand less well
than `deliver`'s is not a good bet.

---

## 3. Three options the books actually support

### A. Initialise the value function, do not touch the reward
*Sutton & Barto §17.4 p. 386, eq. 17.12; Wiewiora (2003).*

`v̂(s,w) ≐ wᵀx(s) + v₀(s)` -- a fixed prior added after the value MLP, a
function of the goal and self features, declared per task and recorded in the
manifest. Wiewiora showed this is *equivalent* to potential-based shaping, so it
inherits policy invariance, and the reward signal stays exactly the sparse
predicate plus the step cost.

**Why it is the best-engineered option:** the 4.4 shaping ablation stays clean,
there is no shaping budget to police, and `tools/reward_audit.py`'s plateau
invariant remains trivially satisfied because no reward component is added.

**Cost:** both families need a marker they do not currently carry --
`repair_belt` declares only `sink`, `restore_power` only `drill`. The gap is the
thing a value prior should be a function of. That is a generator change, a task
version bump, and a re-freeze.

### B. A count-based novelty bonus
*Algorithms for Decision Making §16.3 (R-MAX, eq. 16.3–16.4), §15.4 (UCB1,
eq. 15.3).*

`bonus = c/√N(s̃)` over a coarse discretisation of the agent's own observation.
The notes call it "the most likely single unblocker for the two repair
families".

**The catch, which the notes state:** a novelty bonus is *not* potential-based,
so it changes what is optimal. It must be declared `policy_invariant=False`,
disabled at evaluation, decayed on a schedule, and proven unfarmable by a
dismantle/rebuild loop.

`learn/train.py` currently sets `ent_coef = 0.01` -- undirected exploration,
which §15.3 (p. 303) notes "maintains a constant amount of exploration, despite
there being far more uncertainty earlier". Two families give it nothing to work
with.

### C. Behaviour cloning from the scripted solvers
*RL theory (ABJKS) Thm. 13.3 p. 162, Thm. 13.4 p. 163.*

BC needs `M ≳ ln(|Π|/δ)γ²/((1−γ)⁴ε²)` expert state-action pairs, **with no
exploration term at all** and no `|A|^H` dependence. The notes are blunt about
it:

> The scripted solvers we already have make this the cheapest budget in the
> whole program — a few thousand demonstration episodes, hours not weeks. … It
> is also the only mechanism in the book that can make `restore_power`
> learnable.

We have reference solutions that solve both families at 1.00 on both splits.
This converts an `|A|^H` exploration problem into a maximum-likelihood problem.

**The constraint:** PLAN 4.2 forbids scripted-solution labels during ordinary
PPO training. The notes handle it -- make BC a *separate, declared* training
mode with its own manifest field and its own results column, never mixed into
the PPO baseline. Thm. 13.3's honest caveat is that BC error compounds as
`1/(1−γ)²` under its own state distribution, which is why DAgger-style
relabelling on learner-visited states is the follow-on.

---

## 4. The number that reframes the whole result

*RL theory, Anchor 1.*

A **defensible negative** at `H = 300` needs roughly **15–27 M steps** -- at 75
steps/s, 55–100 hours per family per seed.

The matrix ran **50,000**. That is about 0.3% of the budget a negative result
would require.

So the honest reading of `repair_belt` at 0.00 is not "this family cannot be
learned". It is:

* the reward provably has no gradient (measured: reward = step cost exactly), so
  the budget is not the binding constraint *for this configuration*; **and**
* even with a gradient, 50,000 steps could not license a negative claim at this
  horizon.

Both clauses matter. The first says more steps alone would not have helped. The
second says the run cannot be quoted as evidence that the family is unlearnable.
`docs/LIMITATIONS.md` and the ledger should say the second, not the first.

---

## 5. Recommendation

**A and C, in that order; not B, and not hand-designed subgoal rewards.**

1. **A first**, because it is the only option that leaves the reward signal
   untouched -- which keeps the 4.4 ablation, the plateau audit and the sparse
   optimum all intact -- and because Wiewiora's equivalence means it is
   potential-based shaping with none of the policing.
2. **C next**, as a separately declared training mode, because it is the
   cheapest thing in the program by a wide margin and the theory says it is the
   only mechanism that makes `restore_power` learnable at all.
3. **Not B for now.** It is the most likely single unblocker, but it is the only
   one of the three that changes what is optimal, and this repo has already had
   one reward component quietly redefine a task's optimum this week.
4. **Not more steps at the current reward.** The measurement rules it out
   directly.

The immediate, cheap step that does not need any of the above: stop quoting the
0.00 as a property of the families. It is a property of a reward with no
gradient, run for 0.3% of the steps a negative would need.
