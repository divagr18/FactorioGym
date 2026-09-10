# Review of the agent's prompts, observations, and action interface

Reviewed 2026-09-10 against HEAD `bd41dd3` plus the working tree. Several world/agent files already had uncommitted changes during inspection; findings describe the inspected code, not exclusively that commit. No runtime code changed and no engine or paid model runs were started for this review.

Evidence: current summary, loop, memory, transcript, catalog, environment, knowledge and Lua sensor code; all 235 decision records from `runtime/runs/agent-20260910T230218-0f2a3ffd`; selected rendered observations at the beginning, middle, and end; offline reproductions of domain validation and rendering. The logged `prompt` field is not the complete request: see finding 9. Causal claims about model behavior are therefore limited.

## Verdict

The interface has enough machinery to support useful play, but it still misstates game facts and restricts basic actions. Fix those before adding a more elaborate planner, delegation, vision, or automatic factory connection. The problem is not primarily that the model needs a longer system prompt.

The best existing decisions are structured output, explicit observations, engine-derived recipe data, stable handles, failure records, persistent plans, and bounded execution. The weaknesses are independently maintained descriptions of the same state, RL-shaped argument domains reused for open-world play, and memory/cache policies treated as goals rather than aids to decision-making.

## Findings in priority order

### 1. P1 — The map gives incorrect machine diagnoses

Source: `src/factoriorl/agent/summary.py`, `_STATUS_NAMES` around line 588 and `_map_lines` around line 820; `mod/factoriorl/sensor.lua`, grid legend construction.

The Python table calls status 18 “waiting for space in destination” and 53 “no ingredients.” The real run's engine-sourced entity records identify 18 as `no_ingredients`, 53 as `no_fuel`, and 34 as `waiting_for_space_in_destination`. These are not hypothetical mismatches: the last recorded prompt describes furnace h1026 as waiting for destination space in its map legend and `no_ingredients` in NEARBY ENTITIES.

Consequently the handoff's final-furnace diagnosis is not established by that map text. It does not prove the furnace had a full output inventory. The output counters are still evidence of production; the explanation of the stall needs correction using authoritative state.

Fix: publish engine-resolved status names in the grid legend too. Render one shared entity status representation across map, nearby entities, and factory summary. Unknown enum values should remain unknown, not be guessed from a second table.

Acceptance: captured records for statuses 18, 53, 34 and 27 render consistent names in every view. Revisit the completed-run report before blaming its failure to expand on the model alone.

### 2. P1 — `take_from` cannot request an item absent from the character's inventory

Source: `src/factoriorl/env.py:207`, `argument_domains`; `src/factoriorl/agent/parsing.py:355`, `_validate_arguments`; shared item argument binding in `catalog.py`.

The shared `items` domain contains only the character's inventory. That is a reasonable input restriction for giving an item, but the same domain validates taking from another entity. An offline reproduction with a furnace holding ten iron plates and a character holding only coal rejects `take_from(item="iron-plate")` before the game sees it.

This is a genuine capability gap: collecting the first unit of a new product can be impossible through the advertised tool. It also gives an arbitrary advantage to items the character happened to retain.

Fix: action-specific domains. Giving draws from the character; taking draws from observed source inventories, including output inventories; placement draws from placeable held items. Keep final quantity/capacity checks at execution time. Do not solve this by giving the agent one of every item.

Acceptance: collect the first plate with zero plates held; give the last coal; reject an item not in the selected source; show the model the same action-specific possibilities the validator accepts.

### 3. P1 — Fuel summaries manufacture false blockers

Source: `summary.py:877`, `_factory_lines`; `summary.py:811`, `_map_lines`; `sensor.lua:290` onwards.

`_factory_lines` interprets any absent fuel inventory as “NO FUEL.” An offline reproduction renders both an iron chest and a working electric mining drill with that warning. Nearby-entity rendering already contains the correct caution: absent fuel does not establish an empty fuel-consuming machine.

Separately, the map labels an empty burner fuel slot as definitive proof the machine cannot run. The earlier burner-decay evidence demonstrated stored energy after inventory depletion. These representations can disagree with actual activity even for a burner.

Fix: distinguish energy source, fuel inventory, active energy/status, and unknown data. A useful message is “fuel inventory empty; currently working,” when supported. Do not turn this into a complex custom energy simulator; use game state and avoid unjustified conclusions.

Acceptance: chest, electric machine, genuinely unfueled burner, and burner consuming its last inserted fuel all receive accurate descriptions. Uppercase should mean observed activity, not merely absence from a partial list of stopped states.

### 4. P1 — The charted map and navigation domain are disconnected

Source: `summary.py:333`, `survey_block`; `open_world.py:150`; `env.py:268`, `_destinations`.

The current working tree adds a static resource survey and increases the chart radius to 192. That supersedes the handoff's claim that no charted map is exposed. However, `_destinations` only reads local resource records, entity records, and remembered entities. It does not read the survey. It also selects the nearest capped list of individual tiles, so one nearby deposit can crowd out more useful destinations.

Offline inspection/reproduction confirms adding an 80-tile survey location does not add a navigation destination. The model can be told to walk to mapped coal while its intended navigation command refuses that coordinate. Directional movement remains possible, but this defeats the practical tool.

Fix: open-world navigation should accept explicit bounded coordinates without promising hidden path knowledge, or at minimum include known survey locations, remembered factory locations, and exploration waypoints. Unknown space may be a movement goal without revealing what is there; obstacle discovery can happen during movement. Group destinations by useful location instead of enumerating twelve adjacent ore tiles.

Acceptance: walk toward a charted distant deposit while standing on a nearer deposit; return to an out-of-sensor factory; explore a chosen direction with a bounded waypoint. Record increased chart radius as a changed starting-information profile, not a neutral bug fix.

### 5. P1 — Failure memory teaches permanent prohibitions from temporary conditions

Source: `memory.py`, `Attempt.failed`, `repeated_failures`, `compact`, and `render` around lines 145–376.

Three historical identical failures produce “Sending it again will fail again.” Success does not reset that count. The offline reproduction fails crafting a furnace three times for missing ingredients, then records a successful identical craft; the rendered memory still says it will fail again.

`Attempt.failed` also treats `running` as failure. The loop records accepted ongoing actions into this memory, so normal asynchronous activity can be mislabeled as refused work. Preserving all failures indefinitely and re-rendering them makes this worse over a long run.

Fix: separate accepted/running/completed/rejected/failed. Detect repeated failures since the last success or relevant state change, and phrase reminders conditionally. Bound old failure summaries; preserve the lesson and its prerequisites rather than every event. Do not infer that a delivered-to container is permanently ruled out in an operating factory.

Acceptance: accepted mining is not failure; new ingredients or a successful retry clear the stale warning; repeated identical rejection under unchanged conditions still triggers a useful reminder.

### 6. P1 — The system prompt describes the wrong clock and a stronger legality guarantee than exists

Source: `loop.py:109`, SYSTEM_PROMPT; `summary.py:928` onwards.

It says a decision advances the world by a fixed interval and actions require several decisions to finish. The selected mode is continuous time: the factory changes while the model thinks. It also says the agent cannot see outside the sensor despite the new charted survey and remote factory telemetry.

“The exact list of actions the environment will currently accept” confuses a valid verb with an executable verb/argument combination. Current recipe rendering correctly distinguishes x0 affordability, while the system-level promise suggests every listed action will work. Placement candidates are a center-based approximation, not item-specific footprint checks.

Fix: render a short mode-specific contract. State continuous time, observation timestamps, active operations, and execution-time validation. Label verbs “available commands” and placement coordinates “candidate positions.” Explain local sight, last-seen memory, initial map survey, and remote telemetry as distinct information sources.

Expose remaining wall time and spend allowance. “Decision 530 of 100000” is not useful planning guidance in a run about to hit its 30-minute limit. Report model turns separately from primitive/tool actions.

Acceptance: exact-step and real-time prompts contain the correct clock rule; a delayed call produces an explicit age; the agent sees actual remaining time and can plan a short final action.

### 7. P2 — The batch contract promises receipts that are logged but not clearly returned to the model

Source: `loop.py:650`, `_execute_sequence`; `loop.py:746`, `_compose`; `summary.py`, event rendering; `catalog.py`, `mine_at` and WAIT_FOR.

The executor records completed, failed, and unexecuted members. The next model turn is a summary plus memory, not a structured receipt of the preceding batch. Six recent engine events can omit or obscure an eight-action sequence, and some events render a Python-style dictionary as an action name. A human can reconstruct tool logs more easily than the model can.

In the completed run, recorded runtime errors include 25 `busy` and 16 `no_items` outcomes. Current mine instructions say one per reply and that interleaving waits does not help. That is an awkward workaround for an asynchronous operation API, not a durable batch design.

Fix: append a compact previous-batch receipt containing each member's execution status, actual result/quantity, operation identity if still running, and the reason remaining actions were skipped. Make `wait_for` able to target completion of a named operation; distinguish starting an operation from finishing it. Mining a specified quantity should be a bounded practical operation rather than repeated single-item model calls.

Acceptance: mine-start → wait-for-that-operation → transfer works; insufficient wait returns running/timeout without false success; failed member three clearly marks members four through eight unexecuted in the next model request.

### 8. P2 — Technology knowledge explains that a trigger exists but omits how to satisfy it

Source: `mod/factoriorl/knowledge.lua`, `knowledge.technologies` around line 193; `knowledge.py:254`, `_research_cost`.

Lua retains only `research_trigger.type`; Python prints `trigger:<type>`. The triggering item/entity and quantity are discarded. Removing trigger technologies from the selectable frontier fixes invalid research calls, but does not tell the model what action unlocks them.

Fix: retain structured trigger requirements and render their subject/count. Surface current research and completion changes in the live view. Static recipe data should clearly separate handcrafting from machine categories. Use exact game data, not another prose table maintained by hand.

Acceptance: for each trigger technology in the installed game, the rendered knowledge names the actionable trigger requirements, and a completion event updates available capabilities.

### 9. P2 — The artifact called `prompt` is not the prompt sent

Source: `loop.py:459`, Decision.to_dict; `loop.py:746`, `_compose`; `_new_transcript` and `_ask`.

The decision log stores `summary.render()`. The actual request additionally includes memory, static knowledge, action instructions, objective, survey, conversation history, and retry corrections. ObservationSummary.to_dict also omits grid, built, and craftable fields that affect rendering. Rebuilding from current code is especially unsafe while that code is changing.

This limits this review: exact stored observations and model actions were inspected, but the completed run's complete request stream is not available in the `prompt` field. All 235 stored prompts lack YOUR FACTORY, despite the handoff describing that feature. Treat the completed run and current working-tree additions separately.

Fix: persist the redacted static message once and append-only sent message events, including exact composed memory and retry corrections. Store request boundary offsets and a canonical digest. Do not copy the full history into each decision record. Include all renderer input fields in the observation artifact or explicitly make the message log authoritative.

Acceptance: reconstruct every request byte-for-byte after model-facing serialization/redaction without importing current prompt code; compare its digest with what was sent. Keep provider private reasoning out of the ordinary replay.

### 10. P2 — Prefix caching is useful, but unlimited history is not a memory strategy

Source: `transcript.py`; `memory.py:302`, `compact`; completed-run usage fields.

The first request contained 17,721 input tokens; the last contained 557,218. Every turn appends another largely repeated world view, then repeats remembered failures and plans inside that history. Cache hits lower charges but do not establish that long stale histories improve decisions or latency. No causal claim about this run's competence follows just from its length.

The transcript explicitly never compacts automatically. The memory compactor preserves every failure, every remembered entity, and the historical plan list in storage. This needs a bounded long-run policy before extending sessions or adding child agents.

Fix: retain the useful static reference and append-only windows, but set a context budget with explicit compaction at a safe boundary. Keep current facts, active plan, relevant failures, and references to older events. Measure bounded-history versus current-history behavior at matched tasks; do not adopt RAG or a vector database without demonstrated need. Record cache resets and their cost.

Acceptance: a synthetic extended session stays within its declared context limit and retains outstanding work, while resolved failures disappear from the active brief. Measure request size and latency as well as cache hit rate.

### 11. P2 — Spatial information is duplicated without one consistent interpretation

Source: map, nearby entity, factory, and argument renderers in `summary.py`; `_placement_candidates` in `env.py`; static knowledge footprint rendering.

The agent translates between relative entity offsets, absolute half-tile arguments, map cell coordinates, actual snapped machine centers, and decimal output positions. The rule “output tile is the other's tile” is too imprecise for multi-tile footprints and different entity connection mechanisms. The run shows a requested placement and the resulting center need not be represented identically.

Fix: use absolute coordinates consistently for actionable positions; keep distance/bearing as optional convenience. Explain requested position versus actual snapped center. Report output/pickup positions and the observed receiving entity or ground, plus footprint bounds. A read-only placement preview can answer validity/occupied cells without deciding a layout. Do not jump straight to an automatic `connect_entities` planner to compensate for unclear geometry.

Acceptance: placing a 2x2 machine, rotating a drill, and orienting an inserter give consistent position and connection information across all views. Connection diagnostics should be specific to the entity mechanism rather than one universal adjacency slogan.

## Overengineering, underengineering, and what to keep

**Overengineering:** numeric RL catalogs plus globally enumerated argument domains are shaping the LLM interface too strongly. Twelve movement variants dominate a compact command list even when a practical navigation tool exists. Keyword denylists such as `drop tile` and `first build` confuse public mechanics with leaked solutions; changing a word bypasses them, while legitimate explanations can be prohibited. Independently authored status interpretations and multiple renderers create more disagreement than clarity. Cache-hit maximization has displaced a bounded context design.

**Underengineering:** the model needs authoritative machine state, action-specific domains, proper operation completion, real batch receipts, actionable research triggers, time remaining, and faithful request logging. These are basic interface contracts. They should precede a sophisticated planner or multi-agent scheduler.

**Keep:** a bounded structured response, explicit plan/notes, the distinction between observations and model assertions, declared navigation assistance, local map plus readable entity data, automatic cache usage, exact replayable tool results, and immutable old runs. Ordinary recipe and footprint information is legitimate instruction; do not hide it to make the benchmark seem harder.

Nearest-first truncation is acceptable for an early small factory but needs an escape hatch: queries or pagination for omitted entities, factories, and recipes. Otherwise the prompt advertises that more exists without letting the agent inspect the relevant subset. Remote factory telemetry should be labeled and bounded; do not silently treat it as local sensing.

## Recommended revision order

1. Correct status/fuel semantics, take-item domains, and failure memory. These can make an otherwise correct agent choose or be forced into wrong behavior.
2. Connect survey information to usable navigation; supply real-time/time-remaining context, batch receipts, and operation completion.
3. Complete research/geometry explanations and persist exact sent messages.
4. Consolidate the prompt into: objective and clock; practical tool reference; current time/operations; previous-batch receipt; current world and factory state; active plan/notes. Put detailed static game knowledge in a stable reference or queryable section. Preserve low-level actions for precision without leading every turn with twelve movement choices.
5. Run offline renderer/domain regressions and small engine conformance probes, then one separately budgeted same-seed model comparison. Do not expand the agent architecture until those traces separate interface failures from planning failures.

Suggested core instruction, after the interface supports it:

> Build and expand automated production from the starting inventory. The world keeps running while you think. Use the latest observation timestamp and remaining time when planning. Available commands may still fail because their target, resources, or geometry changed. After a batch, use its receipt to distinguish completed work from running or unexecuted work. Check observed machine status and input/output before assuming production is working. Keep a short active plan and revise it when observations contradict it. Choose your own layout and production priorities.

This is a replacement for misleading boilerplate, not a supplied factory solution. Retain the precise JSON response schema separately. No new paid run or runtime fix was performed as part of this review.
