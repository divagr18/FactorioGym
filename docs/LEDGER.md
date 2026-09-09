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

> **Measurement correction (2026-09-08). Every training-side number published
> for `repair_belt` and `restore_power` before commit `8b68296` is withdrawn.**
> `CurveLogger` kept one reward accumulator and one step counter for the whole
> worker vector: it summed `rewards` across all eight workers, counted vector
> steps as episode steps, and read the success flag from `infos[0]` only. A
> success in any worker but the first was therefore recorded as a zero, and
> `final_train_success_rate` -- the mean over the last twenty rows only --
> reported 24 real successes as `0.0`.
>
> Both families were declared unlearnable off curves that contain successes
> (24 and 19 on `restore_power`, 1 on `repair_belt`). On the fixed
> instrumentation, `restore_power` scores **0.77** on the frozen structural
> split against a **0.02** random floor, with 1.00 on training layouts --
> against **0.00** previously recorded for the same family and holdout id.
> Fixed in `8b68296` with `tests/unit/test_curve_logger.py`; result in
> `docs/evidence/phase4-release-v3-restore_power.json`.
>
> Three further defects found in the same investigation:
>
> 1. **Evaluation was greedy-only.** Nothing guarantees the argmax policy is at
>    least as good as the policy that trained; on a partially observed task the
>    deterministic memoryless policies are a strictly weaker class. The
>    structural row now also scores stochastically on identical episodes. On
>    `restore_power` the two arms agreed exactly (0.77 both), so this was not
>    the cause -- the arm's value is that it settles the question.
>
> 2. **`holdout_v2` was re-frozen five times** (`333bff22`, `1bd2358c`,
>    `b41e4f0c`, `2fa1d92b`, `df72fe29`). The `phase4-release-v2-*` evidence
>    cites `b41e4f0c`; the two repair families' episode sets changed twice
>    after that, so those numbers describe scenes the current holdout no longer
>    contains. Those two files are now marked `superseded`. `deliver`'s frozen
>    entry is byte-identical across the last four freezes
>    (`95dcef5ff8192eb9`), so its 0.92 stands. The integrity check hashed the
>    whole file, so re-freezing one family invalidated the hash cited by all
>    seven; holdouts now carry a per-task `entry_hash` and runs cite that.
>
> 3. **The `15-27 M` step exploration bound was cited as an impossibility
>    proof.** `docs/research/rl-theory.md` is explicit that it is "a floor with
>    no ceiling attached" -- a budget below which no guarantee exists, which
>    argues that a 50,000-step run *cannot distinguish an unlearnable task from
>    an unexplored one*. It is not evidence a task cannot be learned.
>
> The figures that prompted the re-check (`-0.3378`, and a `gapfix-...` run
> directory) do not exist in this repository and were not read off disk.
>
> **Correction to this correction (same day).** The paragraph here first said
> the observability fix "had never been trained", citing
> `extra_public_markers` being absent from the resolved spec of all twelve
> earlier repair/restore runs. `TaskSpec.to_dict` never recorded that field,
> so its absence was evidence of nothing; the claim happened to hold for
> pre-v1.5.0 runs by version number alone. Both fields are now in the resolved
> spec and therefore in the config digest.
>
> With both families run at v1.5.0 under identical budget and instrumentation,
> the marker is published in both and the outcomes diverge completely:
> `restore_power` 0.77 held-out, 1.00 on training layouts, 2394 successes in
> 2549 episodes; `repair_belt` 0.00 held-out, 0.00 on training layouts,
> **1 success in 198 episodes**, mean episode length 247.8 of a 300-step
> budget. So publishing the marker is decisive for one repair family and does
> nothing for the other, and the horizon does not separate them (250 vs 300).
> What does differ: `repair_belt` declares no resources and no water, so all
> six CNN planes are identically zero, whereas `restore_power` has iron ore
> under the drill and two planes carry signal. And `repair_belt`'s target is
> the *absence* of an entity -- the tile where a belt should be and is not --
> which a permutation-invariant pool over an entity set cannot represent.
> Those are the two open hypotheses; neither is yet tested.

> **Holdout design correction (2026-09-08). Both repair families' evaluated
> splits were testing regimes their training never showed.** Measured
> engine-free over 400-600 generated scenes per layout family
> (`docs/evidence/holdout-distribution-audit.json`):
>
> | family | split | fault count | gap position |
> |---|---|---|---|
> | `repair_belt` / `gap`, `gap_far` | train | 1 hole | interior |
> | `repair_belt` / `misrotation` | val | 1 hole **+ a twist** | interior |
> | `repair_belt` / `double_gap` | test | **2 holes** | interior |
> | `restore_power` / `pole_gap`, `pole_gap_far` | train | 1 pole | **600/600 flanked** |
> | `restore_power` / `two_gaps` | val | **2 poles** | mixed |
> | `restore_power` / `gap_near_drill` | test | 1 pole | **600/600 terminal** |
>
> `restore_power.py` fixed the held-out fault at `chain - 1` while training
> sampled `randint(1, chain - 2)`, which never reaches it: the missing pole was
> flanked by poles on both sides in every training scene and in none of the
> held-out ones, where it is always the chain's last position with the drill
> beyond. Two policies then fit training equally well -- "fill the hole between
> two poles", which scores exactly 0.00 on that holdout by construction, and
> "walk to the published gap and place", which transfers. Which one a run
> learned was a seed lottery, and that is the bimodal 0.77 / 0.00 / 0.00, not
> transfer variance. I had called it variance and proposed more seeds, which
> would have measured the lottery more precisely and explained nothing.
>
> `repair_belt` had the same defect on the fault-*count* axis, and published
> only `min(gaps)`, so on a two-hole holdout the second hole had no coordinate
> anywhere the policy could read.
>
> The parity audit cannot see either: its four difficulty descriptors are all
> route geometry to the goal marker, and neither sliding a gap along a chain
> nor adding a second one moves them.
>
> Fixed at v1.6.0 of both families:
>
> * Only `focus_marker` gets geometry (3 goal slots), so publishing a second
>   marker alone would have reached nothing. `env._focus_target` now names the
>   nearest **unrepaired** declared fault, read from the observation's own
>   entity list, so the vector advances as faults close and leaks no truth.
> * Both faults are published (`gap`, `gap2`) on both families.
> * Training spans both regimes: a third of `repair_belt`'s training scenes
>   carry a second hole and a third a misrotation; `restore_power`'s training
>   gap sampling now includes the terminal position (28% and 14% by family).
>   The evaluated families keep their identities -- `double_gap` is still always
>   two holes, `gap_near_drill` still always terminal -- so each still names a
>   structural regime, but one training has shown instances of.
> * `repair_belt`'s `delivered` component is removed. It was HIGH_WATER on the
>   same predicate as `success` with the env terminating on success, so it could
>   only pay on the terminal transition and never approach its cap; the reward
>   audit already exempted it. Declared shaping with no shaping effect made the
>   manifest read as two gradients where there was one.
>
> **`holdout_v3` is now the live holdout** (`4227eb56...`, seed plan
> `holdout-v3`, indices 3000-3099, candidates `deliver`, `repair_belt`,
> `restore_power`). Bumped rather than re-frozen in place, on its own seed
> stream. `holdout_v1` and `holdout_v2` stay in the tree as the record of what
> earlier results were measured against, and are closed. Every published number
> for these two families predates the fix and is superseded.


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

### Correction: there is no 1,800 UPS plateau

An earlier note here and in `vecenv.py` claimed the engine "plateaus near 1,800
UPS no matter what `game.speed` is set to". That was wrong, and wrong in the
direction that costs throughput: **1,800 UPS is simply what speed 30 asks
for.** Measured on a small scene, the engine tracks the multiplier faithfully
and saturates near **5,480 UPS**, reached around speed 90-100 and flat
thereafter at 120, 150, 200 and 1000.

Raising the speed alone bought nothing, and the reason was on our side.
`_collect` sleeps for the predicted advance and then polls. At speed 60 the
prediction comfortably covers the engine, so the first poll lands. At speed 90
the prediction (5.56 ms) sits a hair under the engine's actual 5.60 ms, the
first poll misses, and a 5 ms poll interval hands back exactly what the faster
ticks bought. A 0.5 ms interval makes a wrong prediction cheap:

| game.speed | step (end to end) | steps/s |
|---|---|---|
| 60 | 11.16 ms | 89.6 |
| **90** | **8.82 ms** | **113.4** |
| 120 | 9.73 ms | 102.8 |
| 200 | 10.34 ms | 96.7 |

Past saturation the prediction is optimistic again and the polls return, so 90
is the operating point rather than "as high as possible".

**A dead end worth recording, because it looks obviously right.** Bounding the
prediction by an exponential mean of the *observed* tick rate made throughput
five times worse. The only rate observable from the client is
`ticks / settle time`, which includes the sleep being predicted: a long sleep
produces a low estimate, which lengthens the next sleep. The engine's true tick
rate is not visible from this side of the socket, and a fine poll interval
handles a wrong prediction better than a feedback loop does.

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

**Status: Accepted** (2026-09-07, restored after correction)

> **Restored.** The correction below stands as the record of why this entry was
> downgraded. It is now satisfied: `tools/solvability.py` runs the reference
> solution and a random baseline for every family on both the training and
> held-out splits, and the gate carries a `reference:` section that fails if any
> scripted solution does not complete its task. **Reference solutions pass 1.00
> on train and test for all six families; the gate reports 71/71 with no
> skipped criteria, exit 0** (`docs/evidence/phase3-gate.json`,
> `docs/evidence/phase3-solvability.json`).
>
> Writing those solvers found seven environment defects that no amount of
> training would have diagnosed, which is exactly the argument PLAN 3.2 makes
> for requiring them:
>
> * `place_*` bound position to a single fixed offset, so a gap was fillable
>   from exactly one standing tile. Now one template per facing.
> * The shortest stride was 1.039 tiles against a 0.4 positioning tolerance, so
>   the walker oscillated on a ~1-tile lattice and could never stop on a chosen
>   tile -- while placement binds to `floor(position)`. Added a 2-tick `nudge_*`
>   stride of 0.297 tiles.
> * An inserter's `direction` names its PICKUP tile and it drops 1.2 tiles
>   behind itself. `repair_belt` built both inserters facing east, so the line
>   ran backwards and a correctly repaired belt delivered nothing.
> * `restore_power` connected one of three solar panels, leaving 60 kW against
>   a 90 kW drill, and placed the drill with its near edge exactly on the
>   boundary of the last pole's supply area.
> * `mine_smelt` sited the furnace by scaling the patch centre, which put a 2x2
>   furnace on the spawn tile for some bearings.
> * `deliver.screened_depot` had no generator branch at all and silently
>   generated the base *training* layout, so the previously reported 0.65 train
>   / 0.10 held-out gap for `deliver` measured nothing and is withdrawn.
> * `repair_belt.gap_far` was byte-identical to `gap`, and
>   `restore_power.pole_gap_far` to `pole_gap`.
>
> **Correction (retained).** This was marked Accepted on the strength of a 65/65 gate. The
> gate does not test what PLAN 3.2 actually requires. 3.2 says each family must
> provide "a scripted solution", a random baseline, and a declared solvability
> suite. **The scripted reference solutions were never implemented**, and the
> gate's own docstring quotes "reference solutions" while no gate section runs
> one. The random baselines exist only inside training runs, not in the gate.
>
> This is the same class of error as the Phase 1 premature acceptance: a gate
> that passes on the checks it happens to contain, for a phase whose
> requirements it does not cover.
>
> The cost was concrete. Phase 4's feasibility sweep found that **five of six
> families cannot be learned as designed**, and three never succeed even during
> training. The cause is a catalog defect: all three `place_*` templates bind
> their position to `$ahead` -- the character's tile plus two east -- so
> repairing a belt gap at x=7 requires standing at exactly x=5, and the only
> reward gradient before the repair is a high-water counter that stays at zero
> until the repair is already done. A scripted solver could not have been
> written without noticing this, which is precisely why PLAN asks for one.
>
> Everything else in this entry stands and was genuinely measured. 3.2 was
> incomplete; the sections below marked 3.1 and 3.3 through 3.6 were not.

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

### 3.3 - Six introductory families (**incomplete**: no scripted solutions)

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

**A leak the 504-reset run could not see.** Reset restored the character's
position, inventory, walking state, mining state and crafting queue, and never
its **facing**. An episode therefore began pointing wherever the previous
episode's last move had left the character, and facing is not cosmetic: the
encoder puts `direction / 16` into the self vector, so the first observation of
every episode carried a trace of the previous one. Episodes were not
independent.

The digest is why it survived so long. It recorded position, health, inventory
and crafting queue but not direction, so a run comparing 504 digests could
never have disagreed about the one field that differed. Direction is now in the
digest, which is the part of this fix that stops it recurring.

Assigning `ch.direction` does not work here, and the reason is worth recording:
the engine applies the write on the *following* tick, and the world is paused
between decisions, so the reset observation would still report the stale value.
The character is recreated at reset instead - a character created this tick
reads direction 0 immediately, and reset already invalidates every outstanding
handle, so nothing else is holding the old entity across the boundary.

Found by a reconstruction test written for an unrelated payload change, which
is the argument for writing reconstruction tests: it was comparing two
observation profiles and the only field that disagreed belonged to neither.

**A second leak, in the force rather than the character.** Factorio 2.0 has
trigger-based technologies: several early ones are researched by mining or
crafting a particular item rather than by consuming science. Ordinary task work
therefore researches things, and reset never un-researched them.

The digest had been reporting technologies since it was written, so the check
was in place the whole time and simply had nothing to catch: no reference
solution did enough mining to trip a trigger. Raising `mine_smelt` to twelve
plates and `supply_furnace` to twenty changed that, and the 504-reset run
immediately reported `tech|steam-power` present on a long-running worker and
absent on a freshly launched one.

`LuaForce.reset_technologies` is the wrong tool despite its name -- it reloads
prototype definitions while explicitly *preserving* research state. `force.reset`
would work but also unchartes the map that `on_init` deliberately charts.
Un-researching directly, clearing the research queue and reapplying technology
effects is the surgical version, and it runs before the scene is built so a
task's declared recipe unlocks land after the wipe rather than being undone by
it.

The order in which these two leaks surfaced is the point: both were invisible
until the environment got fast enough and the tasks hard enough to exercise
them. A leakage check that never fires is not evidence of a clean reset.

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
reachability, **the scripted reference solution for every family**, the 504-reset
leakage run with a fresh-worker comparison, and the throughput numbers Phase 4.3
builds on. **71/71, no skipped criteria, exit 0.** 11.95 ms mean step, 4.19 ms
mean reset, 83.7 steps/s single worker - down from 12.62 ms and up from 79.2
steps/s after the observation payload work below.

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

### 4.3 - Worker-count profiling

Measured on the optimised pipeline, `navigate`, 200 steps per worker:

| workers | steps/s | per-worker | efficiency | step ms |
|---|---|---|---|---|
| 1 | 76.0 | 76.0 | 100% | 13.0 |
| 4 | 170.2 | 42.5 | 56% | 19.8 |
| **8** | **261.1** | 32.6 | 43% | 25.5 |
| 12 | 279.4 | 23.3 | 31% | 39.7 |
| 16 | 272.3 | 17.0 | 22% | 56.4 |

Twelve workers buy 7.0% over eight. **Sixteen is slower than twelve** - a
regression, not merely a diminishing return - while step latency more than
doubles from eight to sixteen. The machine has eight physical cores and a
Factorio tick loop is latency-bound single-threaded work, which is close to the
worst case for SMT. The default stays at eight.

For scale, the same profile before the payload work: 41.8 steps/s at one worker
on a 32,750-byte observation, against 76.0 steps/s on 1,078 bytes now.

The selector has a quirk worth recording: it compares each candidate against
the last *accepted* configuration rather than the previous one, so eight was
rejected for being 14.95% better than four (threshold 15%) while twelve was
accepted for being 22.9% better than four. The two were never compared
directly. The rule is defensible but the jump is surprising, and any future
reading of this table should note that 12-vs-8 is a 7% question.

### 4b - Temporal abstraction ablation

PLAN 4b asks whether the gap between familiar and unfamiliar structures is an
abstraction problem or a tuning problem. On `deliver`, matched on seed, budget
and evaluation:

| arm | wall | seeds | val | structures |
|---|---|---|---|---|
| flat | 483s | 0/25 = 0.00 | 0.00 | 0.00 |
| skills | 545s | 25/25 = 1.00 | 0.60 | 0.00 |
| skills + obstacle sidestep | 956s | 1.00 | 0.72 | 0.00 |
| skills + stable addressing | 566s | 1.00 | 0.68 | **25/25 = 1.00** |

A flat policy cannot learn `deliver` at this budget at all; temporally extended
actions solve it outright, for 13% more wall clock.

**The diagnostic value came from the three-row split, not the headline.** On
the structural holdout the flat arm and the first skill arm both read 0.00. A
single number would have said "skills do not help", which is the opposite of
what was happening: the seeds row separates *did not learn* from *learned and
did not transfer*, and only the second is fixable by better addressing.

**The obvious explanation was the wrong one, and the record should say so.**
Structural transfer sitting at 0.00 while seeds sat at 1.00 looks like a
pathfinding failure, since the holdout puts a wall across the direct line.
Giving the skills the reference solver's slide-along-the-face behaviour
improved validation, cost 76% more wall clock, and moved structural transfer
not at all.

The cause was addressing. Skills named their target by distance rank, and rank
is a stable name only if the ranked set is stable:

    train (two_chest):      rank 0 = chest   rank 1 = chest
    test  (screened_depot): rank 0 = chest   rank 1 = wall   ranks 2-3 = wall

The policy had learned "approach 0, then approach 1", and five stone walls
pushed the destination chest past the addressable range. **No action named the
target**, so the task was unrepresentable rather than unlearned - which is why
no amount of training, shaping or walking skill could have fixed it. Excluding
obstacle types from addressing makes ranks mean the same thing on both splits,
and transfer follows immediately.

This is a stopgap scoped to obstacles. The general fix is to address by
declared entity *type* as well as rank, so that a scene dense in belts or pipes
- Phase 7 - cannot shadow the machine the agent needs.

Extending the budget settled the one family that looked like a counterexample:

    mine_smelt @12 plates, skills, structural split
        25k steps   17/25 = 0.68   random 0.24
        50k steps   25/25 = 1.00   random 0.24

At 25k this read as evidence that skills were not enough for a production task.
It was evidence of nothing but the budget.

Three families now clear the 80% structural bar, with a production qualifier
among them - `navigate` 1.00, `deliver` 1.00, `mine_smelt` 1.00.

> **Correction (2026-09-08): the random baselines quoted for the skill arms
> were wrong, and wrong in the flattering direction.** The baseline cache key
> contained the resolved *catalog* digest, but skills are added by a wrapper
> around the environment rather than by the catalog, so a flat run and a
> skill-augmented run of the same task resolved to the same key. In the
> ablation the flat arm ran first and cached a floor of **0.12** measured over
> primitive actions; all three skill arms then read that entry and reported it
> as their own. A later run under a different seed missed the cache, recomputed
> honestly, and returned **0.80**.
>
> So `deliver` with skills is 1.00 against a floor near 0.80, not against 0.12.
> The 4b conclusion survives -- flat 0.00 and skills 1.00 were measured on the
> same evaluation episodes, and that gap is real -- but "far above chance" does
> not, and the margins for `navigate` and `mine_smelt` are unverified for the
> same reason.
>
> This is the third time the same fact has bitten: **a random floor is a
> property of the action space, not of the task.** `mine_smelt`'s floor went
> 0.00 to 0.80 when skills were added, `supply_furnace`'s to 0.88, and here a
> cache built to save recomputation is what hid it. The key now carries the
> action space and a test asserts the two spaces cannot share an entry.

**This does not meet PLAN 4.5**, which
requires three training seeds per family; every number above is a single seed,
on a benchmark where two runs at the same seed and budget scored 1.00 and 0.72.
The bar is cleared on one sample and the claim needs nine runs.

Full per-run detail, including the two failed addressing variants and the
disqualified `supply_furnace`, is in `docs/evidence/phase4b-ablation.json`.
That file exists because the alternative was a ledger claim checkable only
against gitignored `runtime/runs/` - the Phase 1 failure mode.

### 4.4 / 4.5 - not yet done

The shaping-dependence comparison and the three-seed release result remain.

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

### The `deliver` seed-3 collapse, and the shaping term that caused it

The release matrix put `deliver` at 0.67, 0.38 and 0.00 across three training
seeds. The third is not noise around a mean; it is a different policy, and the
mechanism is legible.

**It is not the data.** All three seeds drew the same two layout families and
recorded zero excluded episodes. Seed 3's last success was at timestep 15,872,
with nothing in the final 9,000 steps.

**It is not entropy collapse.** Seed 3's policy entropy is 2.24 against a
uniform 2.94 -- *higher* than seed 2's 1.93. The policies did not stop
exploring; they explored toward different answers.

**They converged on different actions.** On one fixed batch of observations,
seeds 1 and 2 put their greedy mass on `give_iron-plate_20` and seed 3 puts it
on `take_iron-plate_20`, on every input. Under a deterministic evaluation that
is a policy that picks plates up and never delivers them, which is 0.00 on all
three rows.

The evaluation arithmetic confirms it on real episodes rather than sampled
observations:

| seed | structural | mean reward | mean steps (budget 120) |
|---|---|---|---|
| 1 | 0.67 | +0.890 | 27.9 |
| 2 | 0.38 | +0.540 | 65.9 |
| 3 | 0.00 | +0.062 | 95.9 |

Seed 3 runs to truncation and earns +0.06. A take-and-hold policy earns the
`carried` high-water cap of 0.15 and pays the step cost, ~0.11 over a full
episode: about +0.04, plus whatever partial credit luck supplies. Seeds 1 and 2
terminate early because they finish.

**The defect.** `carried` pays 0.02 per plate to a cap of 0.15 for having plates
in inventory. Taking is unambiguous and always available: the source is the one
container with contents, and the reward arrives immediately. Delivering pays
only at the destination -- and the destination is not identifiable from the
observation, as the `gpt-5.6-luna` run showed in the same session. So the
gradient sees a reliable positive for `take` and a gamble for `give`, and a seed
that finds the reliable term first has no pressure to leave it.

That makes the seed variance a property of the reward rather than of the
optimiser: whether a run escapes the local optimum is decided by whether it
stumbles into enough correct deliveries early. It also means the two findings of
this session are one finding. The unobservable destination is what turns `give`
into a gamble, and `carried` is what makes standing still with full pockets pay.

**What this does not establish.** One seed collapsed, so this is a mechanism
with a worked example, not a measured rate. The paired shaped-versus-sparse
comparison that would measure it was withdrawn the same day for being unpaired,
and re-running it under the frozen holdout is the test.

**Why it is not fixed here.** Reward weights are part of the task spec, so
changing them bumps `deliver`'s version, and the frozen holdout is validated
against v1.1.0 -- the change requires a re-freeze and a new candidate
declaration, which is a forward commitment rather than a patch.

## Phase 5 - Assisted agent interface

### 5.3 - Model adapters, first live results

`gpt-5.6-luna` drove the loop on two families through the provider-independent
adapter. Both runs are recorded under `runtime/runs/llm-*` with the exact prompt
sent at every decision.

| family | episodes | solved | decisions | model calls | fallbacks | malformed | tokens | mean latency |
|---|---|---|---|---|---|---|---|---|
| `navigate` v1.0.0 | 1 | 1 | 6 | 6 | 0 | 0 | 4,948 | 4.13 s |
| `deliver` v1.1.0 (skills) | 5 | 2 | 143 | 143 | 0 | 0 | 132,784 | 3.28 s |

The mechanism works: 149 decisions, every one a valid JSON object naming a legal
index, no retry consumed and no fallback taken. `SkillEnv` needed no change to
be driven by a model -- `action_vocabulary` already describes whatever a wrapper
appends -- and the run records which action space produced it, because a
`deliver` number means different things against a primitive floor (0.00) and a
skill floor (0.04).

**The success rate proves nothing yet, and is published as such.** 2 of 5 is
Wilson [0.118, 0.769]; the 0.04 skill floor is [0.007, 0.195]. Those overlap, so
this run does not establish that the model beats chance on `deliver`. It is an
interface result with a diagnosis attached, not a benchmark result.

### The diagnosis: `deliver`'s destination is not in the observation

The outcome is bimodal rather than graded. Both successes finished in 11 and 12
decisions; all three failures ran to the 40-decision cap. Nothing in between --
which is the signature of a discrete find-or-never, not of a partially learned
skill.

The recorded prompt says why. `goal` is `null`, and every container renders
identically once the source is emptied:

    NEARBY ENTITIES (4 shown)
      wooden-chest [h1] at offset (-1.0, -1.0), 1.5 tiles northwest; holds iron-plate x20; status 2
      wooden-chest [h3] at offset (+1.0, +19.0), 19.0 tiles southeast; status 2
      wooden-chest [h2] at offset (+3.0, +19.0), 19.2 tiles southeast; status 2
      wooden-chest [h4] at offset (+22.0, +2.0), 22.0 tiles southeast; status 2

The task brief says "carry items from a source container to a destination
container" while no field says *which*. The `dst` marker exists engine-side and
is what success is scored on, but markers are evaluator state: `local-v1`'s key
list does not carry them, and the RL goal vector carries budget fraction plus
success and landmark booleans, not the destination's identity.

Episode 3 is the clean demonstration -- 17 `take_iron-plate_20` alternating with
16 `give_iron-plate_20`, almost all consecutive pairs:

    take | give | take | give | take | give | take | give | ...

That is the correct strategy given what the model can see: give, observe no
reward, take the plates back, try again. It repeats on the *same* chest because
standing still leaves the ranks unchanged and nothing in the interface records
"h3 has already been ruled out". PLAN 5.4's persistent agent memory is the piece
that would end that loop, and this is the first concrete argument for it.

**Both of these are true, and neither cancels the other.** Putting four
identical containers in every scene is what dropped `deliver`'s skill-space
random floor from 0.80 to 0.04 and made the family able to carry an 80% claim at
all. It is also what put the family out of reach of an agent that sees one
observation. An RL policy still learns the generator's placement regularity from
reward over thousands of episodes; a zero-shot agent has no such channel.

`navigate` is not a counterexample. Its goal is not published either -- the model
solved it because the target was the only non-wall entity in the scene, so the
answer was available by elimination. The gap was there before `deliver` exposed
it.

### The second defect: rank addressing is unstable under movement

`approach_entity_k` names the k-th *nearest* entity, and every move re-ranks the
field. The model reasons about a chest and issues an action naming a slot whose
referent has changed by the time it executes. Episode 0 spent 26 consecutive
decisions alternating `approach_entity_{0,1}` with `move_north` without
progress.

This is the same defect that made a trained policy oscillate between two chests
for an entire episode. Excluding entities already in range fixed the case where
arriving at an entity demoted it to rank 0; it did not fix the general problem,
because any movement reshuffles the addressing. A stable handle (`h3`) is
visible in the observation and is the obvious candidate for what an approach
should name.

### Not fixed here, deliberately

`deliver` was under measurement by the release matrix when this was found.
Editing the family would invalidate those runs and change the blueprints behind
`holdout_v1.json`'s content hash, which is exactly the drift the freeze exists
to catch. Recorded, not patched.

## R4 - Fresh observations, a measured outage, three declared tracks

### The recovery claim in the tree did not hold up, and its own numbers said so

`docs/evidence/phase5-demonstration.json` recorded
`sustained_production_restored: true` and `all_clauses_met: true`. Both are now
withdrawn, in a `withdrawal` block appended to that file with every other byte
unchanged. What the file already contained:

- the disruption emptied both fuel *inventories* at tick 4050 with
  `get_fuel_inventory():clear()`, which never touches `entity.energy`;
- `state_after` records `drill_status: 1` -- **working** -- with `drill_fuel: 0`;
- `phase_4_recovery` took **3 decisions**, which is 90 ticks.

`tools/burner_decay.py` then measured what nothing had measured: after the fuel
inventories are emptied, production continues for **1,170 ticks** and makes 5
more plates. Cross-checked against the machine's own numbers -- 2,999,833 J of
`remaining_burning_fuel` against a 150 kW drill is 1,199 ticks -- so measurement
and arithmetic agree to within one decision interval. The demonstration
refuelled a line that was still running at full rate.

### Three arms, five runs, and a distribution rather than a number

`tools/recovery_arms.py`, from a digest-validated common state at tick 3600:

| arm | plates after the intervention point | outage | recovery |
|---|---|---|---|
| no_action (control) | 5 | **confirmed** at tick 5970 | refused |
| agent, three runs | 5, 39, 43 | refused, refused, confirmed | -- |
| undisturbed (ceiling) | 43 | refused | -- |

Median attributable to the agent: **34 plates**, range 0 to 38, ceiling 43.

**The agent arm is run three times because one run is not a measurement.**
`give_coal_N` fuels the *nearest* entity and the drill and furnace are two tiles
apart, so which machine gets refuelled turns on where the character happens to
be standing. Two runs of the same model on the same scene differed by 39 plates
on that alone.

**A confirmed outage in the control arm does not make the agent's arm a
recovery.** Two of three agent runs never stopped producing for the declared
duration, so there was no outage in *their* arm to recover from: that is loss
**averted**, and the report says `loss_averted` rather than
`recovery_demonstrated`. The third run's own line stopped and never came back --
`agent_failed`. The first version of that verdict printed "the agent's arm never
stopped" over an arm whose outage read `confirmed`, because it tested the
recovery clause and not the outage one.

The extended digest was equal across all five runs at tick 3600, including
`energy=2667` and `burning=2999833` -- the two fields the old digest was blind
to, and exactly the quantity the arms differ on.

### The prompt could not say why a machine had stopped

`local-v2` was bumped to v4 to publish four per-entity fields, and its own note
says why: *"per-entity status name, working flag, fuel and output contents --
what a stopped machine's cause actually is."* `sensor.entity_record` builds all
four. `summary._entity_row` copied **none** of them, so every language-model
prompt this repo has ever rendered showed `status 18` -- the exact string
`sensor.lua`'s comment calls the thing *"neither client could act on"*.

Found by reading R4.2's first agent arm: 25 decisions on a line whose fuel had
just been emptied, with the drill's record carrying `status` and nothing else.
The agent refuelled twice, blind. `SUMMARY_ENCODING_VERSION` is 3, so R3.2's
baseline and that arm do not silently compare across the change.

There is no "fuel: empty" line when the field is absent, deliberately:
`sensor.inventory_contents` returns nil both for an empty fuel inventory and for
an entity that has none, so an absent field cannot tell a drained drill from a
transport belt. The engine's own status name carries that diagnosis -- a burner
out of fuel reports `no_fuel`.

### A declared scene has to be the installed scene

`diagnose_line` declared a hundred-plate output jam in every one of its scenes.
The engine accepted **none** of them: `LuaEntity.insert` picks an inventory by
what an item is *for*, and a furnace's own product belongs to neither its source
nor its fuel slot. So the second fault the family was built around did not
exist.

It survived four solvability runs and a unit test that asserted the
*declaration* rather than the installation. `world.build_blueprint` now falls
back to `get_output_inventory()` and returns an `undelivered` list, and
`FactorioEnv.begin_episode` raises on a non-empty one: a scene that is not the
declared scene produces numbers that describe no task.

### What a task declares about itself

- **`TaskSpec.track`** replaces `tools/release_matrix.CATEGORY`, a dict
  hard-coded in a tool whose own comment said *"the task specs carry no category
  field"*. It listed six of eight tasks; `plate_line` and `build_line` were
  absent, so `CATEGORY.get(task, "unknown")` filed both under `unknown` and PLAN
  4.5's qualification filter could not see the two most production-like families
  in the repo. The default track is `"unclassified"`, which is **not** a valid
  track, so an undeclared one fails `tasks validate` instead of being filed by
  accident.
- **`fault-location:published`** is now a declared assistance. Fault *selection*
  was named (`goal-focus:nearest_unrepaired`) and fault *localisation* was not,
  though publishing the tile is the larger hint: with it, "repair a broken line"
  is "walk to a published coordinate and place one item". Both repair families
  had been doing it undeclared since v1.6.0. Three comments in the tree claimed
  `gap` was evaluator-only and "never reaches an observation" -- false since
  v1.6.0, and in both family modules the sentence sat 170 lines below the
  `extra_public_markers` that falsifies it.
- **`landmarks` is in `to_dict()`**, because each one occupies a slot in the goal
  vector the encoder receives. `extra_public_markers` and `focus_marker` went in
  for that reason and this was left out.
- **`TaskSpec.disruptions`** and a typed `disrupt` request. Every disruption
  before now was a driver-side `bridge.run` string fired at a hard-coded point
  in a script: the tick lived in the driver, the targets lived in the string, and
  no trace could state either. The env applies a declared disruption from the one
  place every observation refresh goes through and `resync`s immediately, so the
  first post-disruption observation is fresh by construction -- the defect that
  made the old demonstration's next prompt say "game tick 450" while the world
  was at 4050.

### A benchmark's floor is a property of its deadline, not only of its scene

`diagnose_line` had to be measured three times before it was worth anything, and
each refusal taught something transferable.

1. **One fault, 5..9 tiles, 500 decisions**: reference 0.60, random **0.40**, and
   on the held-out split random **0.60 beat the reference's 0.40**. Every fix in
   this catalog is a cheap, frequently sampled button press, and `give_coal_N`
   and `take_iron-plate_5` address the *nearest* entity -- so with two machines a
   few tiles away there is no target to get wrong, and a policy sampling sixteen
   actions for hundreds of decisions is nearly a competent one.
2. **Distance.** Moved to 24..30 tiles, the regime `navigate` measures a 0.00
   random floor in (path length 24.7), still inside the 32-tile sensor radius so
   the task stays a diagnosis rather than becoming a search. Random fell to 0.20
   on training scenes and 0.00 on the holdout.
3. **The deadline.** With a *trailing* window and a long budget, acceptance can
   be reached from a fix made almost anywhere in the episode, and a
   skill-equipped random policy scored **1.00**. `plate_line`'s floor in the same
   action space is 0.00 -- not because its scene is harder but because it asks
   for 30 cumulative plates, which only a line running for most of the episode
   produces. Cutting the budget to just past the earliest possible acceptance
   (200 decisions, reference at 120) is what makes the number mean anything.

**A structural descriptor cannot see an empty fuel slot.** The split audit
reported `diagnose_line`'s holdout as "the same shape of scene under a different
name", correctly: its four families place the identical drill, furnace and ore
patch and differ only in machine *contents*. `content_tile_count` sees where a
scene's contents are and nothing saw what was inside them, so
`declared_item_count` was added -- the same class of blind spot `goal_isolation`
was added for, and it changed no other family's verdict.

### Engine facts

- **`LuaEntity.insert` cannot fill a furnace's result slot.** It selects an
  inventory by the item's role, and a furnace's product has no role among its
  inputs. `get_output_inventory():insert(...)` does it.
- **`clear_statistics` does not reset a research trigger's counter.**
  `steam-power`'s trigger is `craft-item iron-plate count=50`; the counter is
  cumulative on the force and *consumed* rather than cleared when it fires -- 67
  plates fired it and left 17, then 33 more fired it again. Statistics read 0
  after the reset and the trigger still re-fired, `disable_research()` has no
  effect, and `research_enabled` is read-only. **No Lua accessor resets it**, so
  sequential episodes on one worker cannot reach the same technology state,
  whatever reset does. Every arm of a state-comparison experiment therefore needs
  its own worker.
- **A connected multiplayer client breaks the RCON transport, and neither the
  pause nor the payload size is why.** With a client attached, requests start
  returning an **empty body**, which `RCONClient.lua` raises as
  `non-json rcon response: ''` -- not an `OSError`, so a poll catching only
  that dies too. Three hypotheses were tested and all three are ruled out:
  - *the pause.* Factorio services RCON inside its tick loop, so a
    `tick_paused` server plausibly never runs the command.
    `configure(free_running=True)` was added and **verified on the engine** --
    paused, the tick held at 0 across two seconds; free-running, 2 to 123 --
    and the failure was unchanged.
  - *payload size.* `scenario_define` carries the whole blueprint, ~10 KB for
    `plate_line`, and the server broadcasts every command to connected players
    (it fills their screen). Pre-installing every scene before the join moved
    the failure to `session.reset`, a small request.
  - *transience.* Six retries a second apart got no answer at all.

  Since an RCON reply *is* the command's printed output, produced inside the
  tick loop, a connected client appears to break the association between a
  command and its reply on the same read. That is a property of the transport;
  watching a live run would need a different channel between Python and the
  mod. `tools/replay.py` is the path that works and perturbs nothing.

  Two lesser facts from the same attempt: the engine gives a joining player a
  character and starts the freeplay intro cutscene, so the mod now puts any
  joiner into `defines.controllers.spectator` with no character (checked by
  identity against `storage.frrl_character`, or it would delete the agent's own
  body); and `FactorioEnv.reset` **increments** `_episode_index`, so
  pre-resetting to show a scene installs the *next* episode's scene rather than
  the one the loop will use.
- **OpenAI's `gpt-5.6` family rejects two parameters this repo sent.**
  `max_tokens` is refused outright ("Use 'max_completion_tokens' instead") and
  `temperature: 0.0` is refused as an *unsupported value* -- "Only the default
  (1) value is supported". Both arrive as HTTP 400, the agent loop's `wait`
  fallback absorbed them, and a `gpt-5.6-luna` run ended after five decisions
  having never once reached the model; the cause was in `decisions.jsonl` and
  nowhere in the result. `OpenAICompatibleAdapter` now retries a rejected
  parameter by name and records the substitution in `healed_parameters`, because
  dropping `temperature` means the run is no longer deterministic and a manifest
  has to be able to say so.

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

### Placement and production, measured during R3 (2026-09-09)

- A 2x2 entity snaps to an **integer** centre. `place_at` offers tile centres
  (`x.5`), so a requested `t + 0.5` lands at `t + 1`: a drill requested at
  (0.5, 0.5) is created at (1, 1). A 2x2 at centre `(cx, cy)` then occupies
  tiles `{cx-1, cx} x {cy-1, cy}` -- **not symmetric about its own centre**,
  which is why rotating a valid layout is not obviously still valid.
- A burner mining drill's `can_place_entity` is **false with no ore beneath
  it**. So a drill's position is chosen from the ore patch rather than free,
  and a placement probe run on bare ground reports every candidate refused.
- `surface.create_entity` **does not feed**
  `force.get_entity_build_count_statistics`. The `place` handler builds that
  way, so a `BUILT` predicate reading those statistics saw `{}` beside a
  running line and 49 plates produced. The handler counts its own successful
  placements instead, which is also the better channel: it counts what the
  agent placed through the action interface, and the scene install path does
  not go through that handler.
- `create_entity` **ignores collision**; `can_place_entity` is the only guard.
  A furnace at (1, 2) beside a drill at (1, 1) catches the drop and produces
  100 plates, and is unreachable through the legal action path because
  `can_place_entity` refuses it.
- A south-facing burner drill at (1, 1) drops at (1.5, 2.2969), i.e. into tile
  (1, 2). Drop offsets per facing are a clean rotation:
  north (-0.5, -1.2969), east (1.2969, -0.5), south (0.5, 1.2969),
  west (-1.2969, 0.5). The **valid furnace centres** are the two whose 2x2 tile
  footprint covers the drop tile without overlapping the drill: for that drill,
  (1, 3) and (2, 3). (0, 3) produces nothing and the drill stalls.
- A correct drill+furnace pair makes **14 plates in its first 3600 ticks and 15
  per 3600 thereafter**, and stalls at **100** when nothing empties the
  furnace's output slot -- about 24,000 ticks.
- A **healthy** line reads status `working` on only **7.5%** of sampled ticks
  (102,451 samples): one burner drill outpaces one stone furnace, so the drill
  sits in `waiting_for_space_in_destination` 87% of the time and the furnace in
  `full_output` once its output backs up. `working` is therefore useless as an
  acceptance criterion, though `working_counts > 0` over a window is fine.
- Entity aliases are markers at runtime: `world.truth()` merges
  `scene.aliases` positions into `markers`, so anything reading only
  `Blueprint.markers` sees a smaller set than the environment does.
- **Placement geometry is rotation invariant; it is not reflection
  invariant.** Measured over every integer furnace offset within 4 tiles of a drill,
  for all four facings: each facing admits exactly two productive centres, all
  three rotations match exactly, and reflecting south's `{(0,2), (1,2)}` in y
  gives `{(0,-2), (1,-2)}` where north's measured set is `{(-1,-2), (0,-2)}`.
  The cause is the 2x2 footprint's parity -- it extends one tile in the negative
  direction and none in the positive, which survives a 90-degree rotation and
  not a flip. **Translation is invariant**, checked at all four parities of
  drill centre -- (5,0), (0,5), (3,-7), (-6,-6) -- so the parity is relative to
  the entity, not to the world lattice. The symmetry group is therefore C4 plus
  translation, not D4, and any shared placement scorer may tie parameters across
  rotations and translations but not across reflections.
