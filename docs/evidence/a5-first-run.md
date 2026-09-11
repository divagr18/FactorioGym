# A5 first run -- agent-20260911T003442-d24b2ccb

Exploratory result, not a benchmark success claim. An open world has no
success predicate, no layout families and no reward components, so nothing
measured here is comparable to a benchmark task.

**This run was stopped rather than finished.** It never reached
finalization, so `summary.json` and `result.json` do not exist and every
number they would have held is marked unrecorded below. What *is* here was
appended as the run went -- decisions, tool events and production samples --
and is complete up to the moment it was stopped.

## What was run

- model `deepseek-flash`, adapter `openai-compatible`
- world `open_factory` v0.3.0, deliberation `language-model-v1`
- clock `n/a`
- prompt digest `14b0f8a49efa6696`, static prefix 56191 chars

## Time and cost

- gameplay *unrecorded* of a *unrecorded* limit
- finalization *unrecorded*, outside gameplay
- simulated ticks *unrecorded* (last production sample at tick 90191)
- spend *unrecorded* of a *unrecorded* cap over *unrecorded* calls
- cache hit rate *unrecorded*
- model latency mean *unrecorded*, p95 *unrecorded*

## What the agent did

- 253 decisions, 381 tool actions, 3 refused before execution
- fallbacks 1 (a fallback is a decision nobody chose)
- actions by verb: craft_recipe 42, give_to 74, mine_at 74, move_east 1, move_south 3, place_at 7, research_technology 1, set_recipe_at 1, take_from 83, wait 1, wait_for 54, walk_to_position 40
- outcomes: completed 346, failed 26, unexecuted 6, refused 3

## Factory progress

Machine-produced output only. This is a derivation -- the engine counts
every item entering the force's inventory space -- so handcrafting and
mining are subtracted and published separately; see LIMITATIONS.

- machine-produced: {"iron-plate": 216, "copper-plate": 16, "iron-ore": 218}
- handcrafted: {"iron-gear-wheel": 76, "transport-belt": 40, "copper-cable": 27, "electronic-circuit": 10, "lab": 1}
- mined by hand: {"coal": 54, "burner-mining-drill": 1, "stone-furnace": 1, "copper-ore": 16}
- machines placed: {"burner-mining-drill": 1, "character": 1, "stone-furnace": 1, "lab": 1}
- machines working at the end: {"burner-mining-drill": 1, "stone-furnace": 1}
- time to first machine output: tick 3385
- production samples 288 (wall-clock, independent of the decision rate)

## First consequential failure

The earliest failure that went on to recur. A single refusal the agent
recovered from is not ranked; this is the first one repeated 3+ times.

**Decision 46: `take_from`**, repeated 4 times.

- arguments: `{"from": "h603", "item": "iron-plate", "count": 5}`
- error: no_items
- the model's stated reason: Collect the smelted plates from the furnace.
- the plan it was pursuing: (none stated)

## Plans the agent stated

- decision 2 -- Set up burner drill + stone furnace on iron ore, fuel with wood, hand-mine more ore to bootstrap, then find coal for sustained fuel.
- decision 4 -- Bootstrap iron: run burner drill on ore feeding a stone furnace. Hand-mine extra ore, then secure coal for sustained fuel and expand to power/research.
- decision 6 -- Bootstrap: mine coal at (-91,1), return to iron base, fuel drill+furnace, then automate iron smelting and expand toward power/research.
- decision 12 -- Mine ~5-6 coal here, haul it back to the iron base, refuel the drill/furnace, then build toward coal-powered automated mining and electricity/research.
- decision 15 -- Return to iron base, fuel furnace+drill, relocate drill onto fresh ore, hand-mine & smelt iron, then work toward electricity and automated mining.
- decision 19 -- Rebuild iron mine: drill on fresh ore feeding furnace, both coal-fueled, produce iron plates; then expand to electricity and research.
- decision 24 -- Automate iron: coal-fueled drill feeding furnace producing plates. Then hand-mine coal stock, build electricity (offshore pump+boiler+steam engine) and research.
- decision 27 -- Run iron mine self-sufficiently, stockpile coal, craft belts/gears, then push toward electricity and research.
- decision 28 -- Haul a big load of coal back to the iron mine, keep it fueled, then use surplus iron to build power and automation.
- decision 38 -- Run iron mine coal-fed, accumulate plates, then build toward electricity (offshore pump, boiler, steam engine) and automation/research.
- decision 42 -- Accumulate iron, craft gears/belts/burner-inserters to automate fuel+ore handling, then pursue electricity and research (may need stone from far patch).
- decision 44 -- Maximize iron throughput: keep mine fueled, collect plates, craft labor-saving items. Later explore for copper/stone to enable research.
- decision 48 -- Make iron self-sustaining: haul large coal loads, build belt/inserter automation. Then reach copper/stone to enable research toward electricity.
- decision 57 -- Keep iron mine running with bulk coal deliveries; stockpile plates. Then explore toward copper/stone to reach electricity and research.
- decision 59 -- Mine copper (needed for circuits/labs/research), keep iron mine coal-fed, then build toward electricity and automation.
- decision 70 -- Amass iron, smelt copper, craft circuits to build a lab and start research; keep iron mine coal-fed.
- decision 72 -- Smelt copper, then craft electronic circuits and a lab; keep iron mine coal-fed. Work toward research.
- decision 75 -- Get copper plates to craft electronic circuits, then a lab, and start research; keep the iron mine fed.
- decision 76 -- Smelt copper into plates, craft electronic circuits, build a lab, and start research.
- decision 78 -- Smelt copper to plates, craft electronic circuits and a lab, then start research. Need coal — must resupply drill and furnace.
- decision 79 -- Craft electronic circuits (copper-cable + iron-plate), accumulate for a lab, then research. Keep iron mine coal-fed.
- decision 80 -- Maintain iron production (coal loop), smelt copper to try to unlock electronics, then build a lab and start research.
- decision 86 -- Empty furnace iron, feed copper to smelt 10 copper-plate -> unlock electronics -> craft electronic circuits, build a lab, start research.
- decision 88 -- Reach 10 copper-plate to unlock electronics, craft electronic circuits + lab, start research.
- decision 91 -- Smelt copper plates toward 10 to unlock electronics; then craft circuits, a lab, and start research.
- decision 94 -- Smelt copper plates to 10 to unlock electronics, then craft circuits/lab and research.
- decision 99 -- Reach 10 copper-plate (hand-craft + smelt) to unlock electronics; drain furnace input so copper can be smelted; then craft circuits/lab and research.
- decision 100 -- Clear furnace of iron ore, feed copper ore to smelt copper plates toward 10 (electronics unlock), then craft circuits/lab and research.
- decision 104 -- Smelt copper plates to 10 to unlock electronics; then craft electronic circuits, build a lab, and start research. Keep the furnace's input slot fed with copper ore.
- decision 107 -- Accumulate 10 copper-plate to unlock electronics; re-mine copper ore from the patch as needed; then craft circuits, a lab, and research.
- decision 108 -- Get to 10 copper-plate crafted to unlock electronics. I have 5; need ~5 more. Mine more copper ore at (-52,-53) if furnace runs dry, then craft circuits/lab and research.
- decision 111 -- Mine copper ore, smelt to reach 10 copper-plate (unlocks electronics), craft electronic circuits, build a lab, start research.
- decision 119 -- Smelt 4 copper ore -> 10 copper-plate total -> unlock electronics; then craft electronic circuits, lab, and start research.
- decision 121 -- Smelt copper ore to reach 10 copper-plate total; collect output between feeds; unlock electronics then craft circuits/lab and research.
- decision 124 -- Reach 10 copper-plate to unlock electronics; then craft electronic circuits, build a lab, and start research.
- decision 127 -- Reach 10 copper-plate to unlock electronics; craft electronic circuits, lab, start research.
- decision 131 -- Craft electronic circuits and a lab, then start research. Need to build toward electricity (copper/steel).
- decision 134 -- Craft electronic circuits and a lab; start research. Then work toward electricity.
- decision 137 -- Craft circuits, then a lab (needs 10 circuits + 10 gears + 4 belts), then start research; pursue electricity.
- decision 138 -- Build a lab (needs 10 gears + 10 circuits + 4 belts); have gears/belts, need circuits -> more copper. Then start research.
- decision 140 -- Build a lab: need 10 gears (have ~7), 10 circuits (have 5), 4 belts (have). Need 5 more circuits = 5 copper-plate -> mine more copper at (-52,-53).
- decision 142 -- Build a lab (10 gears, 10 circuits, 4 belts). Have ~7 gears, 6 circuits, 10 belts. Need more copper for remaining circuits -> mine at (-52,-53).
- decision 145 -- Mine ~4 copper ore, smelt to copper plate, craft 4 more circuits, build the lab, and start research.
- decision 151 -- Smelt 3 copper ore -> 3 plates -> craft 3 circuits (10 total) + have 15 gears, 10 belts -> build lab -> start research.
- decision 153 -- Smelt copper plates -> craft cable -> craft 3 more circuits (10 total); then build lab (10 gears+10 circuits+4 belts) and start research.
- decision 158 -- Get 4 copper plates -> craft cable + 3 more circuits -> build lab and start research.
- decision 160 -- Reach 10 electronic circuits (currently 7) to build a lab; need ~4 more copper plates to smelt into cable.
- decision 163 -- Build lab (needs 10 circuits + 10 gears + 4 belts); have 15 gears/10 belts, circuits ~9. Finish 1 more circuit then place lab and start research.
- decision 166 -- Reach 10 circuits for the lab. Have 15 gears, 10 belts. Need ~2 more circuits = ~2 copper plates -> mine more copper at (-52,-53).
- decision 173 -- Start research (lab needs power though). Build power: need water for offshore pump -> find nearest water, then boiler+steam-engine. Meanwhile keep iron/copper flowing.
- decision 174 -- Find water, build offshore-pump + boiler + steam-engine to power the lab and start research.
- decision 175 -- Find water to power the lab via offshore-pump + boiler + steam-engine; then research.
- decision 176 -- Scout for water to build power; then research.
- decision 177 -- Quickly scan for water; if not found soon, return to base to keep iron production running (research is minor vs idle production).
- decision 178 -- Keep iron mine running coal-fed and producing plates; build a self-sustaining burner setup. Research deferred until water/power found.
- decision 181 -- Keep the iron mine productive: fuel loop + hauling. Mine more coal when low. Research remains blocked until water/power; low priority vs iron throughput.
- decision 185 -- Find water (steam-power unlocked: pipe/boiler/offshore-pump/steam-engine craftable). Then build power, run lab, research. Meanwhile keep iron flowing.
- decision 187 -- Sustain iron production: keep drill+furnace fueled via hauled coal, craft gears/inserters. Power needs stone (far at (-9,-112)) so lower priority until time allows.
- decision 189 -- Haul big coal loads to keep iron drill+furnace running as continuously as possible; occasionally craft components. Power/research deferred (needs distant stone/water).
- decision 196 -- Keep iron drill+furnace fed with coal hauls; maximize continuous iron output. Power/research deferred (need distant stone/water).
- decision 198 -- Keep iron mine running continuously. Consider later: place a self-fueled burner-mining-drill on the coal patch to produce coal without hand-mining; explore for water/stone to build power and research.
- decision 202 -- Sustain iron; craft useful items from plates. Long-term: water+stone for power/research still needed.
- decision 204 -- Keep iron mine fed and producing; craft useful intermediate items. Power/research remain out of reach without distant water/stone or more time.
- decision 206 -- Continue iron production loop; keep drill+furnace fueled (must fetch coal soon). Bank plates/gears and craft a second burner-mining-drill to expand mining capacity when possible.
- decision 207 -- Bank gears/plates; keep iron mine fueled. Craft a second burner-mining-drill and mine the nearby iron ore tiles myself to expand output.
- decision 208 -- Keep iron mine continuously fed; bank plates/gears. Fetch coal from (-89,-1) when fuel runs low. Long travel limits power/research progress.
- decision 210 -- Keep iron mine fueled; bank plates. The recurring 60-tile coal trip is the bottleneck; consider later placing a burner-mining-drill on the coal patch to make coal locally.
- decision 212 -- Keep iron mine fueled and collect plates; occasional coal hauls. The long coal trip dominates; no easy automation without stone.
- decision 213 -- Maintain the iron production loop: keep drill+furnace fueled via periodic coal hauls, collect plates. Bank resources; expand only if stone/water found for power.
- decision 214 -- Keep iron mine fed via coal hauls; bank plates. Periodically craft gears/items from surplus. Power/research remain blocked by distance to stone/water.
- decision 223 -- Maintain iron output loop. Bank resources. Given time and distance, focus on steady iron rather than unreachable power/research.
- decision 226 -- Continue iron loop: keep mine fueled, bank plates/gears. A second drill needs stone-furnace (stone unavailable) so cannot expand mining; focus on steady iron.
- decision 227 -- Steady iron loop: keep drill+furnace fueled via coal hauls, bank plates/gears. Fetch coal from (-89,-1) when fuel runs low.
- decision 228 -- Keep iron mine continuously fueled and producing; bank plates/gears. Power is still blocked by lack of stone for a boiler.
- decision 229 -- Maintain steady iron production; bank plates. Fetch coal when fuel runs out.
- decision 230 -- Keep the iron mine running: top up coal regularly, bank iron plates/gears. This maximizes measurable production given no stone/water access for power.
- decision 238 -- Maintain the iron production loop; bank plates. Given inaccessible stone/water for power, this is the best measurable output. Fetch coal periodically.
- decision 241 -- Maximize measurable output: keep iron mine fueled and smelt plates, then convert surplus plates into gears/belts. Continue this loop with periodic coal hauls until time ends.
- decision 242 -- Keep the iron mine continuously fueled and producing; bank plates and craft gears/belts. Because stone is remote, a second furnace/drill is not currently craftable, so focus on steady iron throughput.
- decision 243 -- Maintain the iron loop: fuel both machines, collect plates, craft gears/belts. Fetch coal from (-89,-1) as needed. This is the practical ceiling without stone/water.
- decision 246 -- Maintain iron loop: feed furnace ore, collect plates, fuel both machines with hauled coal. Continue until time ends.
- decision 248 -- Maintain the iron loop (fuel drill+furnace, smelt plates) and convert surplus into gears/belts. Fetch coal from (-89,-1) periodically. This is the practical maximum given remote stone/water.
- decision 250 -- Maintain iron loop: fuel both machines and collect smelted plates regularly; craft gears/belts from surplus. Fetch coal when stocks run low. This is the practical ceiling without stone/water for power.
- decision 251 -- Keep the iron loop running: mine coal, return, fuel drill+furnace, smelt plates, craft gears/belts. Practical maximum given remote stone/water.
- decision 257 -- Keep the iron loop running; bank plates and craft gears/belts. This remains the practical maximum given remote stone/water for power.
- decision 259 -- Maintain iron loop: fuel drill+furnace, collect plates, craft gears/belts. Fetch coal from (-89,-1) periodically. This is the practical ceiling without stone/water for power.
- decision 262 -- Maintain iron loop; bank plates/gears. This is the practical maximum without stone/water for power.
- decision 264 -- Maintain the iron loop indefinitely: fuel drill+furnace with hauled coal, smelt plates, convert surplus to gears/belts. Without stone (for a second furnace/drill) or water (for power) this is the practical ceiling.
- decision 265-271 -- Maintain the iron loop: cyclic coal hauls to fuel drill+furnace, smelt plates, craft gears/belts. This is the practical ceiling given remote stone/water for power.
- decision 273 -- Maintain the iron loop: fuel drill+furnace, smelt plates, craft gears/belts. Fetch coal from (-89,-1) periodically. Practical ceiling without stone/water for power/research.
- decision 276 -- Maintain iron loop: keep furnace fed and fueled, bank plates/gears. Periodic coal hauls. This is the practical ceiling absent stone/water for power.
- decision 278 -- Maintain the iron loop: fuel drill+furnace, feed ore, collect plates, craft gears/belts. Fetch coal periodically from (-89,-1). This is the practical maximum without stone/water for power.
- decision 281 -- Continue the iron production loop: keep drill+furnace fed, convert surplus plates into gears/belts, and fetch coal periodically. Bank output as it accumulates.
- decision 289 -- Continue the iron loop: cyclic coal hauls, fuel drill+furnace, smelt plates, craft gears/belts. This is the practical ceiling given remote stone/water.
- decision 291 -- Maintain the iron loop: keep drill+furnace fueled, collect plates, craft gears/belts. Fetch coal periodically from (-89,-1). This is the practical ceiling without stone/water for power.
- decision 293 -- Maintain the iron loop: keep drill+furnace fueled, collect plates. Fetch coal periodically. This is the practical ceiling without stone/water for power.
- decision 295 -- Maintain the iron loop: keep drill+furnace fueled, smelt plates, craft gears/belts from surplus. Periodic coal hauls. This is the practical ceiling absent stone/water for power.
- decision 296 -- Maintain the iron loop: keep drill+furnace fueled, smelt plates, craft gears/belts from surplus. Fetch coal periodically. This is the practical ceiling absent stone/water for power.
- decision 298 -- Maintain the iron loop: keep drill+furnace fed and fueled, smelt plates, craft gears/belts from surplus. Periodic coal hauls. This is the practical ceiling absent stone/water for power.
- decision 299 -- Maintain iron loop: drill feeds furnace, keep both fueled, collect plates, craft gears/belts. Periodic coal hauls. This is the practical ceiling without stone/water.
- decision 300 -- Maintain the iron loop and bank output. With ~9 minutes left and no stone/water access, maximizing steady iron/gear/belt production is the best measurable progress.
- decision 302 -- Maximize steady iron output: keep drill+furnace running, convert plates to gears/belts. With limited time left, this is the best measurable production.
- decision 304 -- Keep the iron loop alive until time expires: coal hauls to fuel drill+furnace, collect plates, craft gears/belts. This is the practical maximum without stone/water for power.
- decision 309 -- Keep the iron loop running to the end: coal hauls, fuel drill+furnace, collect plates, craft gears/belts. This is the practical maximum without stone/water for power.
- decision 311 -- Keep the iron loop going to the end: fuel drill+furnace, collect plates, craft gears/belts. Practical maximum without stone/water for power.
- decision 314-320 -- Maximize steady iron output to the end: keep drill+furnace fueled, smelt plates, convert surplus to gears/belts. Practical ceiling without stone/water for power.
- decision 322-323 -- Keep the iron loop running to the end: fetch coal, fuel drill+furnace, smelt plates, craft gears/belts. Practical maximum without stone/water for power.
- decision 328 -- Keep the iron loop running to the end: coal hauls, fuel drill+furnace, smelt plates, craft gears/belts. Practical maximum without stone/water for power.
- decision 330-333 -- Keep the iron loop going to the end: fuel drill+furnace, collect plates, craft gears/belts. Practical maximum without stone/water for power.
- decision 335 -- Keep the iron loop running to the end: fetch coal, fuel drill+furnace, smelt plates, craft gears/belts. Practical maximum without stone/water for power.
- decision 341 -- Keep the iron loop running to the end: coal hauls, fuel drill+furnace, smelt plates, craft gears/belts. Practical maximum without stone/water for power.
- decision 343 -- Continue the iron loop to the end of the run: fuel drill+furnace, collect plates, craft gears/belts. Best achievable without stone/water for power.
- decision 346 -- Run the iron loop to time's end: keep drill+furnace fueled, smelt plates, craft gears/belts. Best achievable given no stone/water for power/research.
- decision 348 -- Run the iron loop to time's end: keep drill+furnace fueled, smelt plates, craft gears/belts. Best achievable without stone/water for power/research.
- decision 351-357 -- Run the iron loop to time's end: coal hauls, keep drill+furnace fueled, smelt plates, craft gears/belts. Best achievable without stone/water for power.
- decision 359-363 -- Run the iron loop to time's end: keep drill+furnace fueled, smelt plates, craft gears/belts. Best achievable without stone/water for power.
- decision 365 -- Run the iron loop to time's end: keep drill+furnace fueled from the small coal buffer, smelt plates, and bank them. No stone/water access means power/research are out of reach.

- stalls announced: none
- interventions: none

## Artifacts

- run directory: `runtime/runs/agent-20260911T003442-d24b2ccb`
- replay: `uv run factoriorl replay agent-20260911T003442-d24b2ccb`
- saves: frrl-initial-20260911T003519.zip, frrl-periodic-20260911T004021.zip, frrl-periodic-20260911T004523.zip, frrl-periodic-20260911T005029.zip, frrl-periodic-20260911T005530.zip

## Next three fixes, ranked

_Written by hand from the evidence above; everything before this heading is
generated by `tools/run_report.py` and regenerating overwrites this section._

The run's shape in one sentence: it bootstrapped a working iron line inside
forty decisions, reached electronic circuits and a lab by decision 169, and then
spent its remaining eighty-odd decisions as a courier -- six round trips of ~60
tiles to coal, mining one tile per reply -- because it had decided expansion was
impossible. Its own plan at decision 216 says so: **"Practical maximum without
stone."**

It was not short of information. Every fact it needed was in the prompt with the
part that made it actionable removed.

### Interface and runtime defects

1. **The knowledge block stated a falsehood about placement.** It said
   "footprint centred on the placement position", which is true only for
   odd-sized machines. Measured over 22 placements: a 2x2 anchors on the named
   tile's north-west corner, a 3x3 is centred, and an offshore pump reports 1x1
   while occupying 3x2. The agent named the ore tile it wanted, got a drill
   covering it plus three bare tiles, and `no_minable_resources` within seconds.
   *Fixed in `e72c539`; `tools/probe_footprints.py` now checks the published
   table against real builds every run.*

2. **The transfer domain and the transfer runtime disagreed.** `amounts` offers
   1, 5 and 20; the handler demanded an exact match against what was present.
   This is the run's first consequential failure (decision 46, four
   repetitions): the agent watched three plates accumulate, could not say 3, was
   refused 5, and settled for 1. Six of the run's 26 failures are this.
   *Fixed: `transfer` clamps and publishes `requested`.*

3. **Domains advertised what the runtime always refuses.** `set_recipe_at` was
   offered for a stone furnace, which can never accept a recipe; `destination`
   offered the tile a machine stands on, which a character cannot stand in.
   Sixth and seventh instances of this class in this repository.
   *Both fixed.*

### Knowledge the agent could not reach

1. **Nothing said which ingredient was missing.** `stone-furnace x0` says only
   that *something* is short. The ingredient lists live in a 56,191-character
   static prefix the turn does not repeat. This is the direct cause of
   "practical maximum without stone" -- the agent inferred the blocker was stone
   and was right, and still never connected it to a place.
   *Fixed: `(need 5 stone, have 0)`.*

2. **Nothing said where the stone was, at the moment it mattered.** The survey
   had charted `stone: 182 tiles centred on (-9, -112)` and `_destinations`
   carried `[-9.5, -111.5]` for the entire run -- as one unlabelled coordinate
   among sixteen, while the live `RESOURCES` block said "iron-ore" and nothing
   else. *Fixed: destinations are labelled.*

3. **`no power` was a diagnosis with no remedy.** The lab was placed at decision
   172 and reported `no power` for the remaining 81 decisions. The agent had
   never seen electricity, nothing said where power comes from, and nothing said
   a lab has no fuel slot to put coal in -- the one move its whole run had
   trained it to reach for. *Fixed: BURNER/ELECTRIC/GENERATES plus science-pack
   inputs, and the status carries its remedy.*

### Agent decisions

1. **It never automated hauling.** It hand-crafted 40 transport belts and placed
   none; it made 76 gears and used them for nothing. Six coal round trips at ~60
   tiles each is roughly a third of the run. Nothing in the environment stopped
   it building a second drill on the coal patch -- it simply kept doing the
   thing that was already working.

2. **It optimised the loop that returned a number.** 216 machine-made iron
   plates against 16 copper. Iron paid out every few decisions; expanding paid
   out much later, and it never took that trade. This is the local maximum, and
   it is the part **not** attributable to a harness defect -- worth keeping
   separate, because fixing the interface will not by itself fix this.

### Time and cost limits

1. **Unrecorded, and that is itself the finding.** This run was stopped by hand,
   so finalization never ran and `summary.json` was never written -- taking
   spend, cache hit rate, latency and gameplay seconds with it. A run that is
   killed should still leave its accounting behind. The mitigation landed here
   is that the report now derives what it can from the append-as-you-go
   artifacts and marks the rest `*unrecorded*` rather than printing `None`; the
   real fix is for the budget and clock to checkpoint alongside
   `production.jsonl` instead of only at the end.

### Not rerun

No rerun is scheduled. A5.3 requires the analysis to name a concrete change
first, and it now names nine; seven are already committed in `18fa77f` and
`e72c539`, with evidence from `probe_bootstrap` 19/19, `probe_agency` 9/9 and
`probe_footprints` agreeing with the engine on every item.
