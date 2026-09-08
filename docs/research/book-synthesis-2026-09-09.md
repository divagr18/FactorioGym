# Book review: decisions and experiments for FactorioRL

Date: 2026-09-09. Read alongside `docs/DEVELOPMENT_REDIRECTION.md`.

## Scope and source provenance

This is a focused source-text review across five substantive books, not a claim to have read all 2,607 pages of the six PDFs cover to cover. I read selected sections directly through the page-level extracted text, checked the source editions, verified that all six PDF SHA-256 hashes match the extraction index, and visually inspected the options reward equation, GAE derivation, and generator-coverage figures. Earlier agent research summaries are not the basis of this synthesis. No training or engine experiment was run for this review.

All page numbers below are **one-based PDF pages**, not printed page numbers. The files are in `D:\Books for RL`.

| Source key | Actual local source | Principal sections read |
|---|---|---|
| SB | `RL Sutton Barto.pdf`, Sutton and Barto, complete second-edition draft, November 5, 2017, 445 pages | Afterstates: 130–131; Dyna: 151–153; changing models and prioritized sweeping: 155–157; average reward: 220–223; stochastic policies: 284–285; auxiliary predictions, options, state estimation, reward design: 397–405 |
| ADM | `Algorithms for Decision Making.pdf`, Kochenderfer, Wheeler, Wray, 2022, 700 pages | Value of information: 141–143; GAE: 291–293; policy validation: selected pages 303–313; imitation: 377–380; finite-state controllers: 493–496 |
| APA | `Automated Planning and Acting.pdf`, Ghallab, Nau, Traverso, 2016, 359 pages | Operational state and refinement: 87–88, 93–98; temporal execution: 153–155; monitoring and goal reasoning: 292–298 |
| AIG | `Artifical Intelligence and Games.pdf` (filename spelling retained), Yannakakis and Togelius, second-edition text dated August 5, 2024, 524 pages | Hybrid players and representations: 204–206; constraint generation and quality diversity: 292–293; generator evaluation and figures: 326–327 |
| TH | `rltheorybook_ABJKS.pdf`, Agarwal, Brantley, Jiang, Kakade, Sun, working draft June 27, 2026, 227 pages | Sampling models: 29–30; strategic exploration: 86; coverage and vanishing gradients: 131–132; average trust-region limitation: 160; imitation assumptions and distribution matching: 173–176 |

`SuttonBartoIPRLBook2ndEd.pdf` is a separate **unfinished 2014/2015 draft**, 352 pages. I inspected its front matter and contents to identify it; it is not a second independent authority corroborating SB. APA is the actual book, not the lecture deck mentioned in an earlier agent's research history. Some tool outputs cut long batches; the source references above identify the reviewed sections, not an assertion that every intervening page or proof received exhaustive review.

The source texts include drafts and occasional notation inconsistencies. Use equation context, assumptions, and boundary checks; do not copy pseudocode mechanically. In particular, SB's options discussion has inconsistent wording around termination/continuation probability. The duration-discounted reward and transition equations are the relevant basis for our accounting, not that ambiguous sentence.

## Main recommendation

Keep the existing redirection: correctness, expressive shared actions, construction, recovery, then controlled learning comparisons. The books add a more specific architectural direction: **learn compact predictions and reusable execution, and make planning accountable to observed effects**. This gives compact policies and LLMs useful complementary roles without assuming either must solve the entire game alone.

The ideas below are proposals for evaluation, not newly accepted plan gates or implementation claims. Start with the bounded experiments at the end rather than implementing every chapter at once.

## 1. Separate four different horizons before tuning rollouts

**Source:** ADM §13.2, PDF 291–293; SB options, PDF 399–400.

The episode budget, reward discount, GAE trace decay, and collector fragment length are different quantities. A long episode and gamma close to one do not imply that a late reward strongly affects every earlier action in an individual PPO update.

For ordinary fixed-duration transitions, GAE combines TD residuals with weights `(gamma * lambda)^k`. Illustratively, with gamma 0.999 and lambda 0.95, a residual 60 decisions later has direct weight about 0.0434; 100 decisions later, about 0.00536. Raising gamma from 0.99 to 0.999 changes the 60-step weight from about 0.0252 to 0.0434. It does not remove trace decay. These are arithmetic illustrations, not measurements of a particular run's lambda.

This is **not** a hard learning horizon: a learned critic can transmit information through bootstrapping over subsequent updates. Conversely, a long fragment with a poor critic is not automatically a good estimate. At fragment boundaries, later outcomes enter through the bootstrap rather than through directly observed residuals in that fragment.

**Factorio experiment:** instrument the lag from a meaningful action to structural effect, first output, and task success. On ordinary on-policy trajectories, retain the raw reward components, critic values, TD residuals, fragment boundaries, and resulting advantages. Check whether successful repair attempts receive useful advantages and how that changes as the critic learns. Do not insert reference-solver trajectories into stock on-policy PPO to make this diagnostic.

If lag and estimation are implicated, compare a small number of per-worker fragment lengths and lambda choices on fixed validation scenes. Record rollout size and optimizer-update count: increasing workers while keeping aggregate buffer size fixed shortens each worker's contiguous experience.

**Gate:** a report explains whether lack of visitation, delayed credit, or critic error is the observed bottleneck. A long episode alone is not grounds to increase the training budget. For options, first resolve duration-aware return/advantage conventions from R1.3; this fixed-step formula cannot silently be transferred to variable-duration actions.

## 2. Use simulator resets to study coverage, with the privilege declared

**Source:** TH §1.4 and §10.1, PDF 29–30, 131–132; strategic exploration, PDF 86.

The theory distinguishes ordinary episodic interaction from a simulator oracle that can query any state/action. Factorio being a simulator does not mean the current environment provides the latter. Validated mid-episode restoration would nevertheless let us conduct much better local experiments.

The useful implication of the chain examples is that learning can lack signal because it rarely reaches informative states. The examples do not prove an exponential sample requirement for our tasks or an impossibility result for PPO.

**Factorio experiment:** create a small bank of validated training-only states: normal start, within interaction range, just before a valid repair, after one of two repairs, and after structural repair but before output. Measure conditional completion from each state. If near-fault starts succeed and ordinary starts fail, reaching the useful region becomes a concrete suspect. If valid repair states are reached but the action is rarely chosen, investigate action selection and representation instead.

Use these resets as an explicitly declared training curriculum if the probe supports it. Preserve canonical evaluation starts and report reset assistance and restoration cost. This is inspired by coverage analysis; it is not an implementation of UCBVI or a theorem-backed neural curriculum.

**Gate:** snapshots include machines, in-flight actions, queues, timers, inventories, memory, and other relevant state, or are reconstructed by validated replay. Do not evaluate on freshly fabricated states that violate game mechanics. Never use hidden test scenes as a curriculum bank.

## 3. Learn predictions before adding more reward shaping

**Source:** SB §17.1, PDF 397–398, and §17.3, PDF 401–403; AIG §5.6, PDF 205–206.

Auxiliary tasks can teach representations through signals other than the task reward. The book explicitly presents their usefulness as contingent, not guaranteed. This offers a third option between sparse-reward PPO and increasingly elaborate hand-shaped rewards.

**First predictions to try:** inventory change after transfer; action completion/failure; whether observed output will increase within a declared number of ticks; whether a currently observed machine will remain productive over that window. Start with one easy, observable prediction and one delayed operational prediction. Do not begin by reconstructing the entire world.

Predict future observable outcomes conditioned on the actual following policy, unless a separately declared experiment supplies counterfactual actions. A forecast that means "output under current behavior" is not the same as "output if we wait". Include policy identity when it changes the target distribution.

Train a small extra head on the existing encoder; compare PPO with and without the head using the same external reward and data budget. Use future observations as labels with masks for unobserved or censored outcomes. Never encode unobserved truth as a negative label. Training on privileged labels is a possible separate profile, not an implicit default.

**Gate:** predictions are calibrated or accurate on held-out scenes, and the gameplay comparison determines whether they help. If prediction improves but policy performance does not, retain it as a diagnostic result rather than claiming improved intelligence. Shared gradients can interfere with control; auxiliary losses need their own ablation.

## 4. Memory should preserve action consequences and uncertainty

**Source:** SB §17.3, PDF 401–403; ADM §23.1, PDF 493–496; APA §3.1.4, PDF 93.

A list of previously seen entities is not necessarily a sufficient state representation. The same immediate view can require different actions depending on what was tried, what happened, and how long ago it happened. A compact internal state can carry that information without retaining an ever-growing transcript.

**Factorio design:** distinguish observed facts, dated memories, predicted effects, and unresolved hypotheses. Record stable target identity, last attempted interaction, returned status, observed consequence, and the condition that would make retry worthwhile. "Issued fuel transfer" must not become "machine is fuelled" without a successful result; "inventory was full" must not become "inventory is full forever".

Compare the current observation-only compact policy with a small action/history stack or recurrent policy. Compare LLM runs with the existing memory against structured action-outcome memory at a matched prompt budget. Inspect the current implementation first: memory already exists and should be extended rather than duplicated.

Construct diagnostic scene pairs whose current observations are identical or deliberately similar but whose relevant histories differ. Check that the test actually requires memory; if every useful distinction is in the current view, recurrence need not help.

**Gate:** fewer repeated unproductive actions and better completion on history-dependent scenes, without hidden-state input or cross-episode evaluation contamination. A sampled-vs-argmax performance gap is a reason to investigate aliasing, not proof of it: SB's short-corridor example (PDF 284–285) shows why stochastic policies can help under restricted representations, but several other mechanisms could explain our gap.

## 5. An operational skill needs an observable effect contract

**Source:** APA §§3.2 and 7.2, PDF 94–98, 292–298; temporal acting, PDF 153–155.

The planning text separates executing a command from observing its intended effect. It also distinguishes retries in a changed world from computational backtracking. This is directly relevant to repeated transfers, partially completed factories, and repairs that do not restore flow.

Extend skill metadata with: initiation evidence, commanded action, expected observable effect, elapsed-time bounds, completion evidence, interruption conditions, and failure reason. Keep predicted and observed effects separate in the trace. Re-evaluate alternative methods against current state after failure.

For example, a successful transfer means items moved; it does not establish resumed production. A completed belt placement means an entity was placed; it does not establish orientation, connected flow, or downstream output. The manager should verify the consequence appropriate to its goal.

**Cheap implementation:** enrich existing action/skill records and add a small controller-side monitor. A complete hierarchical planning language is unnecessary. The runtime reports facts and executes actions; a declared controller makes task-specific choices. Hardcoded monitors that solve diagnosis must be labeled assistance rather than quietly shared with an allegedly unassisted agent.

**Gate:** after a partial failure, the agent continues from real current state and does not repeat the same failed method without a relevant change. Inject a delay and an actual action failure separately: the monitor waits for a plausible delayed effect but eventually reports failure under its declared deadline.

## 6. Test information gathering as gameplay

**Source:** ADM §6.6, PDF 141–143; APA monitoring, PDF 293–297.

Information is valuable when it can improve a decision, with sensing cost considered separately. More observation is not automatically better use of time or tokens.

Create diagnosis tasks in which several faults cause the same initial symptom: no fuel, blocked output, absent material, or broken connectivity. Ensure some observable inspection can distinguish them. An agent must choose where to look and then act on what it learns; it must not receive an evaluator-generated diagnosis.

Inspection costs should follow the declared profile: physical travel consumes game time, and additional tool calls consume inference resources. Avoid inventing arbitrary penalties solely to force the desired behavior. In a paused-world profile, LLM deliberation does not consume simulated production time; report wall-clock and game-clock costs separately.

**Gate:** report inspections, unnecessary replacements, resources, and time to restored output. Include matched ambiguous cases requiring different repairs and unambiguous cases where inspection is unnecessary. Do not reward information-gain proxies by default: the task's operational result should establish whether inspection was worthwhile.

## 7. Build a small model of skill outcomes before a full world model

**Source:** SB options and GVFs, PDF 397–400; AIG hybrids, PDF 204–205; APA temporal execution, PDF 153–155.

An option model concerns cumulative reward and the state at termination, including elapsed time. That suggests learning compact operational forecasts: success probability, duration, resource consumption, and observable effects of a parameterized skill.

For Factorio, a controller might predict whether navigation will finish before a machine's fuel buffer is exhausted, or whether a planned transfer will supply enough material for the measurement window. The first model can be empirical tables or a small regressor over collected skill traces. A neural simulation of all Factorio mechanics is not a prerequisite.

**Experiment:** compare a hand-authored selector with an outcome-model-assisted selector over the same skill library. Keep execution frozen while assessing the selector. Later replace one authored execution skill with a learned controller and measure that change separately.

**Gate:** prediction quality and controller improvement are independently measured. All imagined transitions are labeled. Do not feed model-generated experience into stock on-policy PPO as if it were fresh real rollouts. A model's benefit must exceed its inference and data-collection costs.

## 8. Use afterstate reasoning selectively for placement

**Source:** SB §6.8, PDF 130–131.

The immediate result of an action can sometimes be known even when its later consequences are not. Learning a value over that result can share knowledge across different action/state pairs leading to the same result.

For construction, inspect whether placement candidates can be described by their immediate resulting local structure: position, orientation, inventory expenditure, and observable connectivity. A shared candidate scorer may generalize better than unrelated logits for each placement template.

This is a design hypothesis, not proof that all locally identical structures have equal value. Travel time, spent resources, hidden surroundings, remaining budgets, and in-flight production can distinguish them. Factorio also advances while acting; exact instantaneous afterstates require a justified boundary. Start with validated local consequences and avoid predicting hidden collision validity from truth.

**Gate:** compare candidate-scoring representations on translated/rotated valid layouts while holding the candidate set, information, and budget fixed. Treat symmetry as a tested property: not every Factorio mechanic is rotation/reflection invariant. Defer a new value-learning algorithm until the simpler representation test is informative.

## 9. Evaluate the generator's coverage, not just its seed count

**Source:** AIG §8.4 and generator evaluation, PDF 292–293, 326–327, especially Figures 9.9–9.10.

The game-AI text treats a generator as an object to evaluate: what parts of its intended content space can it actually produce? This gives a concrete way to prevent another terminal-gap surprise or a nominally large holdout containing very few distinct scenes.

Create a coverage table over fault count, fault type, interior/terminal location, detour requirement, distractor count, sensor visibility, reference-solution length, and budget slack. Start with descriptors implicated by the current failures, not an enormous Cartesian product. Store scene examples and hashes for each occupied cell.

A quality-diversity search such as MAP-Elites is optional future work. Initially, enumerate existing generators, bin their outputs, and inspect holes and unintended correlations. Diverse-looking scenes can still share the same solution shortcut.

**Gate:** each split publishes content coverage and overlap. Hold out selected combinations deliberately while retaining separate familiar-instance tests. Preserve training distribution weights separately from stress-test coverage weights. Do not assume a uniform grid across descriptors represents the intended game distribution.

## 10. Stress tests and benchmark estimates are different products

**Source:** ADM chapter 14, selected PDF 303–313.

Searching for failures is useful even when it deliberately samples unusual scenes. But the resulting failure fraction is not an unbiased estimate of normal performance. The book's rare-event treatment uses distribution accounting for precisely this reason.

Maintain a frozen benchmark distribution for scores and a growing diagnostic suite for discovered counterexamples. Store simplified failures such as a terminal pole gap, changed item-flow orientation, one misleading nearby chest, or a late inventory depletion. A diagnostic that motivates a fix becomes a regression case, not a fresh unseen test.

For release comparisons, retain per-training-seed variation as well as within-checkpoint episode uncertainty. Re-evaluating one checkpoint on many scenes does not estimate training instability. Report a cost/success trade-off for compact PPO, LLMs, and hybrids rather than a single score hiding different compute budgets.

**Gate:** reports identify which samples estimate the target distribution and which search for failure. Adversarially discovered cases are not silently pooled into a representative success percentage. Exact-scene pairing and environment-version identity remain required.

## 11. Test adaptation to improvements as well as failures

**Source:** SB §8.3, PDF 155–156, blocking and shortcut maze examples.

This is an especially useful new benchmark idea. A broken route forces an agent to notice that its model is wrong. A newly available better route can remain undiscovered because the old route still works.

Factorio equivalent: after a factory is operating, make a cheaper supply source accessible, open a shorter path, or offer an optional higher-throughput configuration. The previous plan remains feasible. Measure whether the agent discovers and uses the improvement, and how much useful production it sacrifices to inspect alternatives.

For a controlled test, interventions are declared evaluation mechanics. Later, agent-unlocked technology or exploration can create the opportunity naturally. Keep opportunity awareness explicit: announced upgrades test replanning; unannounced upgrades also test exploration.

**Gate:** compare output/cost with a no-adaptation continuation. Distinguish adaptation within an episode from parameter learning across episodes. Do not implement Dyna-Q+ just because its illustrative maze solved this issue; first establish the Factorio behavior we want to measure.

## 12. Keep factory construction and factory operation objectives distinct

**Source:** SB §10.3–10.4, PDF 220–223; ADM validation, PDF 303–304.

Persistent production naturally invites rate-based measurements, whereas constructing a factory involves finite investment, resource use, and time to first output. An asymptotic average-reward objective can ignore a finite setup cost; finite resources and irreversible construction also undermine simple steady-state assumptions.

Keep near-term evaluation finite and interpretable: time to first sustained output, cumulative delivered production over a fixed tick budget, production loss during disruption, repair cost, and final output rate. Compute rates per simulated time, not per skill decision. Continue using the established PPO baseline while measuring these separately; replacing gamma with one does not implement an average-reward learner.

**Gate:** a policy that produces a small burst then stops cannot pass a sustained-operation objective. A policy that builds indefinitely without useful output cannot win by collecting intermediate construction rewards. If average-reward learning becomes a research branch, specify recurrence/resource assumptions and initial-investment evaluation explicitly.

## Proposed bounded experiments, in order

| ID | Question | Smallest useful experiment | Decision rule |
|---|---|---|---|
| B1 | Where does useful experience disappear? | Phase visitation and repair-to-output lag audit on existing compatible traces; new instrumented pilot only if needed | Select the next intervention from evidence: reachability/coverage, representation, credit, or transfer |
| B2 | Does the controller need action-outcome history? | Existing memory vs structured consequences for LLM; observation vs short history for compact policy on diagnostic pairs | Keep the extension if it improves the intended history-dependent behavior at declared cost |
| B3 | Can the encoder learn operational consequences? | One or two auxiliary heads with unchanged external reward | Check prediction quality first, then control benefit; do not conflate the two |
| B4 | Do demonstrations omit recovery states? | BC rollout, expert relabeling of learner-visited states, aggregate retraining | Proceed only if the teacher can act from those states; measure closed-loop success, not just label accuracy |
| B5 | What content does the generator actually cover? | Offline descriptor table and representative scenes; bounded engine solvability checks | Fix unintended holes or correlations in a new version, retain intentional OOD cells |
| B6 | Can agents adapt while the old strategy still works? | One opportunity intervention with a matched continuation | Add to persistent benchmarks if it distinguishes adaptation from simple fault reaction |

B1 and B5 are mostly analysis/tooling. B2 follows the shared interface work. B3 and B4 should be separate experiments, not stacked changes. B6 belongs after valid persistent production and interventions exist. None requires a broad new matrix or stopping an existing training run.

## Suggested additions to the development plan

These are recommendations, not automatic edits to PLAN.md:

- R0/R5: add phase-conditional visitation, effect lag, and rollout/advantage diagnostics (B1).
- R2/R4: give memory and skill outcomes observed/predicted/unknown semantics; define expected effects and verification windows (B2).
- R5: test auxiliary prediction and learner-state imitation as separate routes (B3/B4).
- R0/R5 and Phase 7a: add generator-coverage artifacts and separate stress regressions from score estimation (B5).
- Phase 7: add opportunity adaptation alongside directed repair and diagnosis (B6).
- Phase 8: learn skill outcome/duration models before considering a full neural world model; compare execution and selection independently.

Do not add all of these to the critical path. Correctness and actual construction/recovery remain the immediate development commitments. The main research question worth building toward is: **which combination of compact prediction, reusable execution, memory, and deliberation yields reliable factory behavior under a declared compute budget?**
