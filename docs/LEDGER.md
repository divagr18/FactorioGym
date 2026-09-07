# FactorioRL progress ledger

Phase and gate status per PLAN.md §7. Statuses: **Accepted** (all gates passed
with evidence), **Needs correction** (implementation exists but a gate failed
or evidence is missing), **Blocked** (named external dependency prevents
verification).

A phase is marked Accepted only *after* its gate has run and passed with the
transcript committed under `docs/evidence/`. The ledger entry is downstream of
the evidence, never ahead of it — see the correction note at the end of Phase 1.

Engine pin: Factorio **2.0.60 (build 83512, win64)** - see
`src/factoriorl/engine_config.py`. Verified on this workstation
(AMD Ryzen 7 5800H, **NVIDIA RTX 3050 Laptop GPU 4.3 GB**, Windows 11).

> **Hardware correction (2026-09-07).** Every document in this repository -
> PLAN.md, this ledger, and HANDOFF.md - recorded the training GPU as an
> **RTX 4060**. It is not. The machine has an **NVIDIA GeForce RTX 3050 Laptop
> GPU, 4.3 GB VRAM, sm_86**, alongside a Ryzen 7 5800H (the CPU claim was
> correct). This was discovered the first time torch was asked rather than
> assumed, and confirmed against `torch.cuda`, `Win32_VideoController` and
> `Win32_Processor`.
>
> Consequences: PLAN 4.5's "overnight introductory training recipe on the 4060"
> refers to hardware that is not present, and any figure published against "the
> 4060" would be a false provenance record. Run manifests now capture the GPU
> name, VRAM, capability and torch version at runtime, so this cannot drift
> again. In practice the environment is the bottleneck at ~24 ms per step, not
> the GPU, so the correction matters for honesty more than for feasibility.


## Phase 0 — Repository foundation and engine feasibility

**Status: Accepted** (2026-09-07)

### 0.1 — Project skeleton

Evidence: `uv sync` installs from clean state; `uv run pytest tests/unit
tests/contract` passes without Factorio (47 passed, 1 intentional skip);
importing `factoriorl` never launches Factorio; `.gitignore` excludes
`runtime/` (saves, logs, port reservations), credentials
(`.factoriorl.user.json`, `*.credentials*`, `*.token*`), checkpoints and
trajectories.

### 0.2 — Worker launch and shutdown

Evidence: `uv run factoriorl worker start` launches one isolated worker
(`runtime/workers/cli-demo/`: own write-data, mod directory with only
`factoriorl_0.1.0.zip`, save, ports, logs), handshakes the mod over RCON,
prints status + observation + liveness, `/quit` shutdown releases the process
and returns its ports to the pool. Startup failure classification is exercised
by engine tests: missing executable
(`test_startup_failure_classification_missing_executable`), invalid save
(`..._invalid_save`), unavailable port (`test_port_unavailable_detection`,
which now asserts the specific `port_unavailable` kind), engine error (log tail
captured in every `StartupFailure`).

### 0.3 — Exact stepping

Evidence: `runtime/gate-phase0/report.json` → advances of 1/30/120 ticks
exact; mixed sequence `(30,1,120,30,30)` drift 0; idle drift 0 over 0.5s;
observations only after the interval completes (advance blocks until settled).

### 0.4 — Physical actions

Evidence (gate report + engine tests): south move displaces to y≈4.45 in 30
ticks; 600-tick north walk stops at y≈-5.0 against the wall row at y=-6
(collision); overdraw transfer rejected with `no_items` and no mutation;
valid transfer moves 10/50 iron plates, source 40, character 10, total
conserved at 50 (no action creates items); out-of-reach transfer rejected
after walking away (`test_out_of_reach_transfer_fails`).

### 0.5 — Reset repeatability

Evidence: gate report → 10 consecutive cycles of
`reset → observe → move(30t south) → advance(30) → observe → transfer(10) →
observe` produce identical records for episode tick, advance result, character
position, character inventory, src/dst contents, and task counters, all
through the RCON transport the environment will use.

### Phase 0 exit gate

One command: `uv run factoriorl phase0-gate` → `passed: true`, report written
to `runtime/gate-phase0/report.json`.

**Latency correction (2026-09-07).** The figure previously recorded here —
"mean ≈305 ms/cycle for the 6-request sequence" — was wrong twice over. The gate
averaged the *seven per-request* latencies and stored the result under a key
named `mean_per_cycle_ms`, so 305 ms was the cost of one **request**; a cycle
actually cost ~2423 ms. The metric is now reported as both
`mean_per_request_ms` and a real `mean_per_cycle_ms`, with `requests_per_cycle`.

After the Phase T transport work the same gate measures **90.61 ms per request
and 634.25 ms per cycle** at `game.speed = 1`, still `passed: true` with mixed
drift 0 and idle drift 0. See `docs/evidence/transport-roundtrip.md`.

## Phase T — Transport throughput

**Status: Accepted** (2026-09-07)

Evidence: `docs/evidence/transport-roundtrip.md`.

Three defects, each measured before and after as PLAN.md 11.4 requires:

- **A 250 ms timeout on every call.** `RCONClient.command` waited out
  `CONTINUATION_TIMEOUT` looking for an empty terminator chunk Factorio never
  sends. Isolated on one connection, the stall was **250.03 ms** of a 266.69 ms
  request. Replaced with **sentinel framing**: each call takes a fresh id pair
  and follows the real command with one known to print nothing; in-order replies
  make the sentinel's arrival proof of completion. Printing commands now cost
  the same as empty ones — **266.69 ms → 16.66 ms**.
- **`game.speed` was never set**, so a 30-tick interval cost a hard 500 ms at the
  default 60 UPS. `advance(30)` went **818.57 ms → 25.56 ms (32×)**.
  `uv run factoriorl bench speed` proves pacing-only: 5 cycles at speed 1 and 30,
  compared field by field, **15.64× faster, zero mismatches, `identical: true`**.
- **The published latency metric measured per-request cost under a per-cycle
  name.** Corrected above; `WorkerSession._settle` now accounts for every probe
  and sleep instead of only its first dispatch and last probe.

Also fixed: per-call request ids, so a single timeout no longer desynchronizes
the connection permanently (every call reused one id, and the late reply poisoned
every subsequent call); and `TCP_NODELAY`, since each request is now two small
back-to-back writes.

**A latent bug this exposed.** `handle_reset` cleared the in-flight advance
bookkeeping but never re-paused the world, so a reset landing mid-advance left
the engine ticking. The 250 ms stall had masked it — an advance always settled
and re-paused itself before the next observation. At 16 ms per request it does
not, and `test_unknown_action_cannot_reach_arbitrary_execution` began failing
with `assert 1 == 4`. `begin_episode` now sets `game.tick_paused = true`.

Coverage: `tests/unit/test_rcon_framing.py`, six engine-free tests against a fake
RCON server reproducing a missing terminator, packet splits, and a late reply
from an abandoned call. Engine suite 33/33; Phase 0 gate `passed: true`.

## Phase 1 — Protocol and reliable worker lifecycle

**Status: Accepted** (2026-09-07, after one round of correction)

Gate evidence: `docs/evidence/phase1-engine-suite.txt` — **33 passed in
195.87s, exit code 0**, produced by `uv run factoriorl phase1-gate` against
real Factorio workers.

### 1.1 — Protocol v1 frozen

Typed requests/responses/errors with request and episode ids and action
execution status (`src/factoriorl/protocol.py`, `mod/factoriorl/runtime.lua`).
Fixtures under `tests/fixtures/protocol_v1/` cover success, invalid action,
ongoing operation, reset, and observation; engine test
`test_protocol_fixtures.py::test_fixture_agreement` sends each of the 10
fixtures through the typed transport and asserts agreement with the live Lua
runtime (all 10 pass). Unknown protocol version fails with `bad_protocol`;
unknown request/action types are rejected without executing anything
(`test_unknown_action_cannot_reach_arbitrary_execution` verifies no tick or
entity change); missing fields yield `missing_field` errors.

*Ongoing operations:* `advance` is the one operation that spans decision
intervals, and `test_advance_settles_from_running_to_completed` now asserts
both halves of its lifecycle — the `running` answer and the settled
`completed` response in the ledger, matched against the fixture's `settled`
block, which nothing previously checked. Ongoing **actions** (mining, crafting)
do not exist yet and are Phase 2.2 work; the runtime holds one in-flight
operation at a time (`pending_advance_request_id`), which 2.2 will need to
generalize.

*Vocabulary agreement:* Python enums and Lua string literals declare the wire
vocabulary independently, so `tests/contract/test_protocol_vocabulary.py`
(engine-free) asserts every code Lua emits is parseable by Python, that every
`RequestType` has a Lua handler, and that the mutating-type set matches. Two
`ErrorCode` values are not emitted yet and are listed as explicit reservations
(`collision` → Phase 2.1; `duplicate_request` → duplicates use
`ResultCode.DUPLICATE` today), so an unused code is a deliberate choice rather
than something that quietly rotted.

*Typed action status:* `TimedResponse.action` returns a real `ActionResult`
for `act` responses and `None` for every other request type — `status`
answers carry `result.status == "ready"`, which is a worker state, not an
action outcome (`test_act_response_carries_a_typed_action_result`).

### 1.2 — Duplicate and stale-request handling

`test_duplicate_transfer_applies_once`: identical request id applied once;
resend returns code `duplicate` and inventory/task counters do not double.
`stale_episode.json` fixture: foreign episode id rejected as `stale_episode`
(session raises `StaleEpisodeError`); the episode check runs before dispatch
for every mutating type (`advance`, `act`, `reset`).
`test_request_status_resolves_uncertain_advance`: applied advances resolve
`applied=true, settled=true, state=completed`; never-sent ids resolve
`observed=false, applied=false, state=unknown`.

Two defects found during Phase 1 correction, both now covered by regression
tests:

- **Request ids collided across sessions.** Ids were a bare per-session
  counter, so a client reconnecting to a live worker restarted at `act-1` and
  collided with ids the worker had already recorded — the worker answered
  `duplicate` with the *old* stored result and silently never applied the new
  mutation, in exactly the reconnect path `request_status` exists to make safe.
  Ids now carry a per-session nonce
  (`test_reconnected_session_ids_do_not_collide`).
- **Only successful mutations entered the ledger.** A retry after a timeout on
  a *rejected* action executed it a second time, and `request_status` could not
  distinguish "never arrived" from "arrived and was refused". Every mutating
  outcome is now recorded, and `request_status` reports
  `observed`/`applied`/`settled`/`state`
  (`test_rejected_mutation_is_recorded_and_not_reexecuted`).

Ledger bounds are unchanged: 4096 entries per episode, FIFO eviction, cleared
on reset, held in `storage` so it survives save/load.

### 1.3 — Worker supervision

`src/factoriorl/supervisor.py`: `SupervisionPolicy` with request deadline,
settle deadline, liveness interval, liveness attempts, liveness backoff, and
restart prefix — **every field is now read by code**. `check_liveness` retries
`liveness_attempts` times with backoff (a single timeout is a hiccup, not
proof of death) and short-circuits on a dead process; `ensure_alive` re-probes
only when the last answer is older than `liveness_interval_seconds`, so callers
can invoke it per step. `settle_deadline_seconds` reaches
`WorkerSession.settle_timeout`, replacing a hardcoded 30s. Captured engine
output is attached to failures via `_log_tail` (console log +
`write-data/factorio-current.log`).

`restart()` now genuinely restarts **from a known state**: the failed worker's
own `save.zip` is copied into the replacement's directory
(`WorkerManager.launch(save_source=...)`), rather than regenerating a map from
the same seed. `test_pool_restart_resumes_from_the_failed_workers_save`
asserts the replacement's save is byte-identical to the failed worker's. Note
the honest scope: this resumes the worker's reference state, not the exact tick
at which it died — Factorio writes no save on a forced kill.
Restart also records failure evidence first and releases the failed worker's
process, connection, and ports, so repeated restarts cannot exhaust the pool.

`test_kill_is_classified_infrastructure_failure`: forced kill →
`InfrastructureFailure` with evidence file.
`test_restart_does_not_overwrite_failed_evidence`: replacement starts under a
distinct worker id; the failed run's `failure-evidence.json` survives.
Socket timeouts now surface as typed `ProtocolError` rather than bare
`OSError`, so callers can resolve them with `request_status`.

### 1.4 — Multi-worker orchestration

`src/factoriorl/pool.py`: `WorkerPool` is a real orchestrator — a registry of
supervised workers with `start`/`get`/`restart`/`shutdown`/`close`, a
`heartbeat()` that honors the policy's probe interval, and context-manager
semantics. `factoriorl worker start` goes through it, so the supervised path is
the user-facing one rather than untested infrastructure.

`test_two_workers_independent`: two pooled workers on distinct ports and
directories; w1 walks east while w2 transfers 25 plates — states diverge;
resetting w1 does not affect w2; leaving the pool closes both and returns their
port reservations.

**"Both close cleanly after a failed parent run"** is implemented in three
layers and tested at the hardest one:

1. A Windows **kill-on-close job object** owns every engine, so the OS reaps
   them even on a hard kill where no Python code of ours runs.
2. `atexit` covers an unhandled exception or `sys.exit`.
3. `reap_orphans()` / `factoriorl worker reap` finds engines whose recorded
   parent pid is dead and terminates them — the recovery path for anything
   that escaped 1 and 2.

`test_workers_close_when_the_parent_run_fails` starts a pooled worker in a
child process that then exits via `os._exit(1)` — no `finally`, no `atexit` —
and asserts the engine is gone. Layer 1 is the only thing that can pass it.
This was not a hypothetical: an orphaned `t-port-busy` engine from the failed
first gate attempt was still running hours later, holding port 23510 and its
`write-data/.lock`.

*Port allocation* (PLAN.md 1.3, "ports are not accidentally shared") is now
correct across processes, not just within one. Each port is claimed by an
exclusive reservation file under `runtime/ports/`, taken **before** the bind
probe and held until release, so a second process (another pytest run, a CLI
worker, a second trainer) skips ports this one holds instead of racing it to
`bind`. Claims whose owning process is gone are reaped, so a crash cannot leak
the pool; the starting offset is randomized per process to avoid lockstep
contention. Ten engine-free tests in `tests/unit/test_ports.py` cover
allocation, reservation, release, stale and malformed claim reaping, pid
liveness, and bind-failure classification — port correctness no longer requires
launching an engine to verify.

### Phase 1 exit gate

`uv run factoriorl phase1-gate` → runs `pytest tests/engine -v --require-engine`
against real workers and writes the transcript to
`docs/evidence/phase1-engine-suite.txt`. Result: **33 passed, 0 failed,
0 skipped, exit code 0, 195.87s**.

The gate passes `--require-engine` so a missing Factorio binary *fails* the
gate instead of skipping it — PLAN.md §4 treats "tests skipped because Factorio
was unavailable" as an incomplete state, and two engine-marked tests take no
worker fixture and would otherwise let a bare machine report a green run.

### Correction note

Phase 1 was previously marked **Accepted** in this ledger on the strength of a
gate run that had not finished. When that run did finish it reported **11
failed, 16 passed**: all nine `test_fixture_agreement` cases, the
arbitrary-execution guard, and the port test. The test files were then edited
and never re-run, so the fixed code had never executed against Factorio at all.
The ledger also cited `docs/evidence/phase1-engine-suite.txt`, which did not
exist, in a repository with zero commits.

What this cost: 1.1's four acceptance criteria had no passing evidence; two
real defects (the request-id collision and the ledger's success-only recording)
were sitting behind the failing tests; a supervisor whose liveness knobs did
nothing and whose `restart()` docstring was false; and 1.4 with no orchestrator
at all. The process fix is in `CONTRIBUTING.md`: the ledger entry is written
after the gate passes, with the transcript committed.

## Phase 2 - Embodied actions and partial observations

**Status: Accepted** (2026-09-07)

Gate evidence: `docs/evidence/phase2-gate.json` - **33/33 criteria, passed: true,
`privileged_calls_by_agent: 0`**, produced by `uv run factoriorl phase2-gate`.
Capability evidence: `docs/evidence/phase2-capability-spike.md`.

### 2.0 - Capability spike

`LuaControl` behaviour on a bare character was Phase 2's largest unknown, so it
was measured before the matrix was designed. **Mining and crafting are both
native** - `mining_state` drives a player-less character and `begin_crafting`
queues and delivers - so no fallback driver and no documented approximation.
`build_from_cursor` is `LuaPlayer`-only, settled from the engine's own API dump,
so placement is assembled from `can_place_entity` + `create_entity` + debit and
`docs/ACTION_MATRIX.md` states what that excludes.

### 2.1 - Protocol v2 and the action matrix

`PROTOCOL_VERSION = 2`, declared once on each side (`protocol.py`,
`protocol.lua`) with a contract test asserting they match - they could
previously drift with only a live worker to notice. All ten actions are
implemented. `mod/factoriorl/matrix.lua` is the authority the dispatcher
*reads*: payload presence, types, ranges and defaults are validated once from
it, which deletes per-handler boilerplate and the drift that comes with it.
`src/factoriorl/action_matrix.py` mirrors it, `docs/ACTION_MATRIX.md` is
generated from that (`factoriorl action-matrix --check` fails if stale), and 12
engine-free tests compare names, order, ongoing/cancellable/reach flags and
required fields across the two.

Reach is prototype-derived: build 10, entity 10, resource 2.7 - three distinct
distances replacing one wrong constant of 6, with `can_reach_entity` applying
the engine's bounding-box rule.

**Research needed three refusals, not one.** Factorio 2.0 gates the root of the
tech tree behind triggers rather than selection: 7 of 196 technologies complete
by crafting or mining, and *every* zero-prerequisite technology is one of them,
so at a fresh start nothing is selectable at all. `tech_not_selectable` is
therefore distinct from `tech_locked` - telling an agent to wait for
prerequisites that will never arrive would be wrong; it needs to craft.

### 2.2 - Ongoing actions

One in-flight registry keyed by request id replaces both single-purpose hacks:
the advance's dedicated fields and `storage.frrl_move_deadline`, a bare global
scalar that a second move silently overwrote. Progress survives step calls, a
target becoming unavailable is a recorded failure, and completion is reported
exactly once - an invariant, not a convention, because `ledger_settle` refuses
to settle an entry that is already terminal.

Cancellation stops at the paused tick the cancel is processed, which is exact
and observable precisely because the world is paused during request handling.

**Moving interrupts mining**, found by the gate: an un-cancelled mine
re-asserted `mining_state` every tick and silently pinned the character, so a
move was accepted and then did nothing. The two are physically exclusive for a
character, as they are in the game.

### 2.3 - Local structured observations

32-tile sensor region built from three engine queries - measured at 0.011 ms
(entities), ~0 ms (resources) and 0.018 ms (a full 65x65 obstacle-only tile
query). Terrain is never built by iterating 4225 tiles. Remembered observations
carry `age`, and one update rule covers the whole of "moving into and out of
visibility updates records correctly": entities inside the region but absent
from the sweep are deleted, because the agent looked and they were not there.
Remembered contents live only in the `remembered` block, never in `entities`,
so PLAN section 2's "must not present distant machine state as current" is
enforced by the array it lives in.

Observation payloads measured **1117-5276 bytes** (mean 2659) against an 8 KB
target.

### 2.4 - Entity identity and lifecycle

Opaque episode-scoped handles over two descriptor forms, because the spike
confirmed **resources carry no `unit_number`**: a `unit` form keyed on
`unit_number` (never reused, which makes "rebuilt entities do not inherit stale
identity" true by construction) and a `tile` form with a generation counter for
resources. Destruction is detected per-entity via
`register_on_object_destroyed`, which fires regardless of `raise_destroy` and
avoids registering world-wide lifecycle events that would cost per-tick
throughput at Phase 7 scale. Reset clears every table, so prior handles resolve
`unknown_handle` - distinct from `target_missing`, because "I never knew that"
and "it is gone" are different failures for an agent.

### 2.5 - Profiles

`local-v1` and `primitive-v1`, each name plus version, echoed in every
observation and `describe`. The observation profile declares its allowed key
list and `observations.snapshot` filters through it, so adding a field is a
visible, reviewable act - the mechanism, not a convention, that keeps
evaluator-only information out of policy input. `assisted-v1` is declared with
`available = false` so Phase 5's contract is visible now and cannot be
retro-fitted into `primitive-v1` by accident.

### Inherited defects fixed

- **No `pcall` crash boundary**: a Lua error escaped to the engine. Every
  handler now runs guarded, and the boundary proved itself immediately - a 2.0
  API change (`max_health` moved to `LuaEntity`) surfaced as a structured
  `engine` error instead of killing the worker.
- **`handle_reset` never restored the character** - it does now, which matters
  once mining can destroy things.
- **`clear_scene` cleared only +-16**, so anything built outside survived a
  reset. Player-force entities are now swept surface-wide.
- **`act_move` did not validate `payload.ticks`** - the matrix validates it.

### Phase 2 exit gate

`uv run factoriorl phase2-gate` runs a scripted embodied agent that walks east
until ore enters the sensor region, mines it, mines stone, hand-crafts a
furnace, places it, fuels it, feeds it ore, waits out the smelt, and collects
the plate - then exercises rotation, recipe selection, research, handle
destruction and the reset identity boundary.

**"Without privileged mutations" is enforced, not asserted**: the agent's RCON
client is wrapped so any Lua that is not the typed dispatch template raises,
making `bridge.run` unreachable. Evaluator setup happens before the agent starts
and every such call is recorded in the report, so `privileged_calls_by_agent: 0`
is a measurement.

Result: **33/33 criteria, 0 privileged calls, a plate produced by the world
simulating natively**, 11820 game ticks in ~11 s wall.

One gate defect worth recording: the rotation and recipe checks were originally
guarded by `if target is not None:` and **silently vanished** when the agent
could not reach them - a gate quietly dropping criteria is the same failure the
Phase 1 correction note is about. They are unconditional now, which immediately
exposed the real problem: greedy movement has no pathfinding (PLAN defers it to
5.1) and the character was blocked by the 3x3 assembling machine, so those
checks now run while the targets are in view from spawn, and `_walk_towards`
detects being stuck instead of spinning out its budget.

## Phase 3 - Task engine, reset correctness, RL spaces

**Status: Accepted** (2026-09-07)

Gate evidence: `docs/evidence/phase3-gate.json` - **65/65 checks, passed: true**,
produced by `uv run factoriorl phase3-gate`.

| Measurement | Planned | Measured |
|---|---|---|
| Step | 110 ms | **24.03 ms** (p95 26.55) |
| Steps/s, one worker | 9.1 | **41.6** |
| Reset | under 150 ms | **6.21 ms** (p95 7.69) |
| Consecutive resets | 500 | **504** |

The step figure matters for Phase 4: at 41.6 steps/s a 300k-step run is ~2 hours
on a *single* worker, so training no longer depends on running enough concurrent
engines to hit this machine's commit ceiling.

### 3.1 - Task specifications

Declarative Python: frozen dataclasses plus a closed predicate and reward
vocabulary, with a discovery-based registry. `factoriorl tasks validate` runs at
construction, before any worker is launched, and reports every problem at once -
schema, budgets, split coverage, exactly one sparse-success component, and a
structural pass over sampled blueprints.

**Deviation from the approved plan, recorded deliberately:** the plan specified
TOML as the source of truth with Python only for generators. This ships the
declaration in Python dataclasses instead. Every PLAN 3.1 acceptance criterion
is still met and still provable - a TOML layer would add parsing without
changing what can be checked - but external task authoring (PLAN 11.1) will
want the file format, so this is a deferral rather than a rejection.

"A task can be added without editing worker management" is enforced as an
**import-direction test**: `factoriorl.tasks` may not import `worker`,
`worker_config`, `pool`, `supervisor`, `rcon` or `session`. If it could reach
them, adding a task could require changing them.

### 3.2 - Scenes are blueprints, not map generation

Decided on reset cost and on where scenario selection lives. Map-gen settings
are written at worker launch and consumed by `--create`, so a map-gen-driven
task would need an engine relaunch to switch or randomise - putting scenario
selection squarely inside worker management. A blueprint is also something a
solvability check can reason about: "a 4x4 iron patch 18 tiles out" is a
question you can ask a declaration and cannot ask Perlin noise.

The measurement settles it: **6.21 ms per reset**. A save-reload reset would be
two orders of magnitude slower, and PLAN 3.3's 500 consecutive resets would not
be affordable.

Blueprints are installed once and referenced by hash thereafter, so a
steady-state reset carries a hash rather than the whole scene.

### 3.3 - Six introductory families

`navigate`, `deliver`, `mine_smelt`, `supply_furnace`, `repair_belt`,
`restore_power`. Each declares four layout families across train/val/test, and
**the holdout is structural, not a different seed** - PLAN section 3 asks for
unfamiliar seeds and unfamiliar structures to be reported separately, so the
split lives on the structure. A test asserts no training family is reused as a
holdout.

Two deliberate design choices worth recording:

- `mine_smelt` measures success from **force production statistics**, not an
  inventory count, so plates that already existed at reset can never be counted
  as produced.
- Hand-crafting is off the critical path: the furnace is pre-placed, so the
  production family does not depend on crafting-queue behaviour.
- `repair_belt` needs no separate instrumentation: throughput before the repair
  is exactly zero, so the unloading chest's contents *are* the proof, and the
  success predicate cannot be satisfied by the initial state.

### 3.4 - Gymnasium spaces

Fixed-shape `Dict` space: a 6x65x65 grid, 32 padded entity rows with a mask, and
self/inventory/goal vectors. Entity rows carry the **is-remembered flag and
observation age**, so a policy can tell "I can see this" from "I saw this 500
ticks ago" - which is the point of Phase 2.3's remembered observations.

Action catalogs are ordered lists of fully-bound templates where file order *is*
the index. A task's subset is re-indexed by **catalog** order, never by the
order the task listed them, and the resolved catalog is content-hashed - a task
listing its actions differently must not change what an action index means.

Masks are built from the observation alone, never from the blueprint or the task
truth, and `wait` is unconditionally legal so a no-op fallback always exists.
Termination and truncation are exclusive, and an infrastructure failure is
**neither**: a dead worker is not a task outcome (PLAN section 2), so it is
reported with `excluded_from_metrics` for the trainer to drop.

### 3.5 - Reward accounting

Computed in Python over counters Lua maintains. `total` is
`sum(components.values())` **by construction**, so "components sum to the
returned reward" is not something a test has to catch.

Two component kinds make the anti-exploit criteria structurally true:

- **high-water** shaping rewards the increase of a maximum, never the level, so
  moving items out of the goal and back cannot pay twice;
- **potential-based** shaping is `gamma*phi(s') - phi(s)`.

One test correction worth recording: the potential-loop test initially asserted
a closed loop sums to *zero*. It does not - it sums to `(gamma - 1) * sum(phi)`,
which is **negative**. The loop costs a little rather than paying, and the
property that actually matters is that it is never profitable and that the loss
does not compound into a payout over more laps. The test asserts that instead.

Evaluator truth is a **separate `truth` request** from `observe`, so the
boundary between policy input and evaluator knowledge is structural rather than
a naming convention.

### 3.6 - Reset correctness

`world_digest` returns **sorted lines, not a hash**: a bare hash mismatch says
something leaked but not what, and PLAN 3.3 wants item, technology, timer and
reward leakage distinguished. Python hashes it for the fast comparison and diffs
the lines when they disagree.

The 504-reset run asserts digest equality per task, flat growth proxies (handle
table, remembered store, ledger, events) as the unbounded-state canary a digest
comparison alone would miss, and - the check that makes it meaningful - **a
long-running worker matches a freshly launched one** on the same scene.

Reset now also clears cumulative production statistics, which nothing did
before and which is exactly where cross-episode leakage hides.

### Bugs the gate found

- **Factorio 2.0 has 16 compass directions, not 8.** Encoding `direction / 8.0`
  produced 1.875 and pushed the feature outside its declared Box, which the
  in-space check caught on every family after the first step.
- **`500 // 6 * 6` is 498.** The reset target was missed by two until the
  division became a ceiling - a gate that quietly under-runs its own target is
  worth catching.

### Phase 3 exit gate

`uv run factoriorl phase3-gate`: config validation, the Gymnasium space and mask
invariants per family, reward-component sums, shaping-off parity, held-out split
reachability, the 504-reset leakage run with a fresh-worker comparison, and the
throughput numbers Phase 4.3 builds on. **65/65, 16 seconds.**

## Phase 4 - Compact RL baseline

**Status: Needs correction** (2026-09-07) - the pipeline is built and a pilot
ran end to end; the release learning result (4.5) is not yet produced.

### 4.1 - Baseline policy

MaskablePPO with a custom SB3 features extractor: a small CNN over the 6x65x65
grid, a **masked** per-entity MLP, and an MLP over the self/inventory/goal
vectors. Masked pooling matters: padding rows must not drag the mean toward zero
or win the max, or a scene with three entities would encode differently from the
same scene padded to thirty-two.

`266070` parameters, extractor version
`1`, recorded in every manifest - a silent
encoder change invalidates every checkpoint trained on it.

`factoriorl doctor-train` fails loudly on a CPU-only wheel, the commonest
Windows setup failure.

**The environment never imports the training stack.** A subprocess test imports
`factoriorl.env`, `encoders`, `tasks` and `cli`, discovers every task, and
asserts torch, stable_baselines3 and sb3_contrib are absent from `sys.modules`.
That test *is* PLAN 4.1's "evaluation runs without training dependencies
changing environment behaviour", made mechanical.

### 4.2 - Pilot: one family, one seed

Run `pilot-20260907T061820-e3cdf3b6`, task `navigate`, 40,000 steps,
1851 s wall, **21.6 steps/s**
end to end (policy forward, PPO updates, evaluation and the random baseline all
included; the environment alone measures 41.6 steps/s).

| | success rate | Wilson 95% |
|---|---|---|
| Held-out (`test` split, EVAL seed branch) | **0.40** (12/30) | [0.2459, 0.5768] |
| Random baseline, same split | 0.00 (0/10) | [0.0, 0.2775] |

Evaluation uses the **held-out layout family** and a **disjoint seed branch**, so
an evaluation episode can never be one the policy trained on - asserted by a
unit test over 2000 indices per branch.

### 4.3 / 4.4 / 4.5 - not yet done

Worker-count profiling, the shaping-dependence comparison and the three-seed
release result remain. The throughput needed for them is in hand (a 300k-step
run is roughly two hours on a single worker), and the memory ceiling that
motivated multi-worker scaling is much less pressing now that one worker
sustains 41.6 env steps/s.

### Phase 4 exit gate

`uv run factoriorl phase4-gate` checks **mechanics, never a stochastic training
outcome** - a gate that depends on a policy reaching a success rate is a flaky
gate. It verifies the training stack resolves with real CUDA, that published
checkpoints load *and their manifests still verify against current code*, that
the manifest carries every field PLAN section 2 requires including the measured
GPU, that train and evaluation seed branches are disjoint, and that a short
training run completes unattended.

### Defects found while building Phase 4

- **Validation was seeded with Python's `hash()`**, which is randomised per
  process, so `tasks validate` sampled different blueprints on every run and the
  suite passed or failed at random - roughly one run in eight. Worse than a
  failing check, because it hid one: with stable seeding, `navigate/pillar_field`
  turned out to place overlapping pillars in 3 of 400 seeds. Both fixed; 24,000
  generated scenes now come back clean.
- **The manifest could fail to be written at all.** `host_info()` imports torch
  to describe the GPU, and on a machine near its commit limit that raises
  `OSError: the paging file is too small`, not `ImportError`. Losing the record
  of a run is far worse than recording an unknown GPU, so the probe now degrades
  to a reason string.

## Engine facts worth remembering

- Factorio 2.0 Lua: `global` → `storage`, `game.create_player` removed
  (zero-player servers use a standalone `character` entity),
  `game.{json_to_table,table_to_json}` → `helpers.*`, `script.on_tick` →
  `script.on_event(defines.events.on_tick, ...)`.
- RCON: Source framing `length|id|type|body`, body terminated by TWO NULs;
  auth type 3 (success echoes the request id, failure id -1); commands run as
  console lines (Lua needs the `/c` prefix); values returned via
  `rcon.print`.
- `--port` and `--bind host:port` conflict; use `--bind` only.
  `allow_commands` in server-settings must be the string `"true"`.
- Headless stepping: `auto_pause=false` + explicit `game.tick_paused` toggling
  in `advance`.
- The game port is **UDP**, RCON is TCP; a TCP-only probe never detects a
  collision on the game port.
- An orphaned engine holds `write-data/.lock` indefinitely, so worker ids must
  be unique per run — a fresh worker reusing a stale id fails on the lock and
  misreports the cause.
- 2.0 moved `max_health` onto `LuaEntity` (with `get_health_ratio()`);
  `LuaEntityPrototype` exposes `get_max_health()` as a method.
- `LuaEntity`'s parent is `LuaControl`, so a bare character has `mining_state`,
  `begin_crafting`, `cursor_stack` and the reach properties - but **not**
  `build_from_cursor`, which is `LuaPlayer`-only.
- Resources are `ResourceEntityPrototype` and carry **no `unit_number`**.
- The 2.0 base tech tree is trigger-gated at its root: 7 technologies complete
  by crafting or mining and cannot be queued, and every zero-prerequisite
  technology is one of them.
- The engine ships a machine-readable API at
  `D:/Factorio/doc-html/runtime-api.json` (`api_version 6`), which is the
  authority for what exists on this build.
