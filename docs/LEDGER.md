# FactorioRL progress ledger

Phase and gate status per PLAN.md §7. Statuses: **Accepted** (all gates passed
with evidence), **Needs correction** (implementation exists but a gate failed
or evidence is missing), **Blocked** (named external dependency prevents
verification).

A phase is marked Accepted only *after* its gate has run and passed with the
transcript committed under `docs/evidence/`. The ledger entry is downstream of
the evidence, never ahead of it — see the correction note at the end of Phase 1.

Engine pin: Factorio **2.0.60 (build 83512, win64)** — see
`src/factoriorl/engine_config.py`. Verified on this workstation
(AMD Ryzen 7 5800H, RTX 4060, Windows 11).

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
to `runtime/gate-phase0/report.json` with latency measurements (mean
≈305 ms/cycle for the 6-request sequence).

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
