# FactorioRL handoff — Phase 1 closed, Phase 2 next

**Date:** 2026-09-07
**Machine:** this laptop (Ryzen 7 5800H, RTX 4060, Windows 11) — NOT the 4060 desktop.
**Engine:** Factorio **2.0.60 (build 83512, win64)** at `D:\Factorio\bin\x64\factorio.exe`,
pinned in `src/factoriorl/engine_config.py` (other builds refused at launch).

---

## Where we are

Phases 0 and 1 are **Accepted** with committed evidence. Phase 2 (the full
action matrix) has not started.

| Item | Status | Evidence |
|---|---|---|
| 0.1 repo skeleton | ✅ | `uv sync` from clean; `uv run pytest tests/unit tests/contract` = 47 passed, 1 intentional skip, no Factorio needed; `ruff check` + `ruff format --check` clean |
| 0.2 worker launch/shutdown | ✅ | `uv run factoriorl worker start` end-to-end; failure classification tests (missing exe / invalid save / busy port / engine error) |
| 0.3 exact stepping | ✅ | gate report: 1/30/120 exact, mixed-sequence drift 0, idle drift 0 |
| 0.4 physical actions | ✅ | movement 4.45 tiles/30t; wall collision stops at y≈-5.0; transfer conservation (50→40+10); overdraw rejected `no_items`; out-of-reach rejected |
| 0.5 reset repeatability | ✅ | 10 identical cycles over the RCON transport |
| **Phase 0 exit gate** | ✅ `passed: true` | `uv run factoriorl phase0-gate` → `runtime/gate-phase0/report.json`, mean ≈305 ms per 6-request cycle |
| 1.1 protocol v1 frozen | ✅ | 10/10 fixtures agree with the live Lua runtime; vocabulary-agreement tests; typed `ActionResult` |
| 1.2 duplicate/stale handling | ✅ | per-episode ledger (4096, FIFO, `storage`-backed); session-nonced request ids; every mutating outcome recorded; 4-state `request_status` |
| 1.3 supervision | ✅ | every `SupervisionPolicy` knob consulted; restart from the failed worker's own save; kill → `InfrastructureFailure`; evidence preserved |
| 1.4 multi-worker | ✅ | `WorkerPool` orchestrator; cross-process port reservations; engines die with a crashed parent (job object) |
| **Phase 1 exit gate** | ✅ **33 passed, exit 0** | `uv run factoriorl phase1-gate` → `docs/evidence/phase1-engine-suite.txt`, 195.87s |

### What changed while closing Phase 1

Phase 1 had been marked Accepted before its gate finished; the run that
followed reported **11 failed / 16 passed**, and the fixes made afterwards
were never re-run. Closing it properly surfaced four real defects, not just a
stale transcript:

1. **Request ids collided across sessions.** Ids were a per-session counter, so
   a reconnecting client restarted at `act-1`, hit the dedup ledger, and had
   its genuinely-new mutation answered with an old stored result — a silent
   lost write in exactly the recovery path `request_status` exists to protect.
   Fixed with a per-session nonce; regression test
   `test_reconnected_session_ids_do_not_collide`.
2. **Only successful mutations entered the ledger.** A retry after a timeout on
   a rejected action ran it twice, and `request_status` could not tell "never
   arrived" from "arrived and was refused". Every mutating outcome is recorded
   now; `request_status` returns `observed`/`applied`/`settled`/`state`.
3. **`restart()` did not restart from a known state.** It regenerated a map
   from the same seed and never opened the failed worker's save, despite the
   docstring. It now copies that save into the replacement.
4. **Port allocation was process-local.** A module global plus a probe-then-bind
   window; a second Python process restarting at the same base would hand out
   ports already in use. Now claimed through exclusive reservation files under
   `runtime/ports/`, taken before the probe and reaped when the owner dies.

Plus two things that were claimed but absent: `SupervisionPolicy`'s liveness
knobs were read by nothing, and 1.4 had no orchestrator — only a test that
happened to construct two workers.

### The port issue, concretely

The failing `test_port_unavailable_detection` was not a flaky bind. An
orphaned `t-port-busy` engine from an earlier failed run was **still alive
hours later**, holding port 23510 — the port the test hardcoded — and its
`write-data/.lock`. Three fixes, in order of what they prevent:

- Engines can no longer outlive their parent: a Windows **kill-on-close job
  object** reaps them even on a hard kill (`test_workers_close_when_the_parent_run_fails`
  proves it by killing a child with `os._exit`). `atexit` and
  `factoriorl worker reap` cover the softer cases.
- Ports are reserved **across processes**, not just within one, and stale
  claims from dead owners are reaped.
- Test worker ids are unique per run, so a leftover directory or a live orphan
  can never collide with a fresh worker and misreport the cause.

Bind failures now classify as `port_unavailable` instead of a generic
`engine_error`, so the failure names the problem (PLAN.md 0.2).

---

## Architecture decisions (and why)

1. **One headless server per worker, versioned JSON over RCON.** Matches PLAN.md;
   no transport experiments yet. RCON round-trip ≈50–60 ms.
2. **Total isolation per worker** under `runtime/workers/<id>/`: own `write-data`
   (via `--config` with `[path] write-data=`), own `--mod-directory`, own save,
   ports, logs. The user's profile (heavy modpacks!) and the game install are
   never touched.
3. **Base game enforced.** The user profile enables Space Age feature flags;
   workers' `mod-list.json` explicitly disables `space-age`, `quality`,
   `elevated-rails`. Peaceful map, no biters, no pollution.
4. **Exact stepping = `auto_pause=false` + explicit `game.tick_paused`.** The
   advance handler records `(request_id, start_tick, target)` in the ledger and
   unpauses; `on_tick` re-pauses and settles the stored response at exactly
   tick N. `advance()` polls `request_status` until settled, so callers only
   observe completed intervals.
5. **Agent body = standalone `character` entity in `storage`.** Factorio 2.0 has
   no LuaPlayer on zero-player servers.
6. **Closed typed dispatch.** The only wire surface is `remote.call("frrl_bridge",
   "dispatch", json)` with a fixed handler table. Arbitrary Lua exists only in
   `bridge.run` for the evaluator; the typed protocol cannot reach it.
7. **Dedup ledger per episode** (request_id → stored response, capped 4096, FIFO).
   Resends return `code: duplicate` with the stored result — including stored
   *rejections*; `reset` starts a new episode and clears the ledger; foreign
   episode ids → `stale_episode`.
8. **Cross-process port reservations** under `runtime/ports/`, claimed before
   the bind probe, released on shutdown, reaped when the owning pid is gone.
9. **Kill-on-close job object** owns every engine, so a crashed parent cannot
   leave orphans holding locks and ports.
10. **Module-scoped workers in tests.** Per-test workers exhausted the machine's
    commit limit. Destructive tests (kill/restart/two-worker/parent-crash) still
    get dedicated workers.

---

## Engine facts learned the hard way (Factorio 2.0.60)

### API changes vs 1.x (all hit at runtime, all in the mod)

| Old (1.x) | New (2.0) | Symptom if wrong |
|---|---|---|
| `global.x` | `storage.x` | `attempt to index global 'global' (a nil value)` in `on_init` |
| `script.on_tick(fn)` | `script.on_event(defines.events.on_tick, fn)` | `LuaBootstrap doesn't contain key on_tick` at load |
| `game.create_player{...}` | removed | no replacement — use a standalone `character` entity |
| `game.json_to_table` / `table_to_json` | `helpers.*` | empty RCON response is the silent sign |

### RCON wire protocol (empirically confirmed)

- Source framing `int32 length | int32 id | int32 type | body`, `length` = id+type+body.
- **Body must end with TWO NUL bytes.** Single-NUL packets → `RCON received an
  invalid message type <ascii of your body>` and nothing executes.
- Auth: type 3, body = password. **Success: id = request id echoed** (not 2),
  empty body. Failure: id = -1.
- Commands: type 2, run as a **console line** — Lua needs `/c `, and **only
  `rcon.print(...)` produces output** (`/c return 1+1` → empty).
- Multi-packet responses: type-0 chunks carrying the request id, terminated by
  an empty chunk; short-response paths may omit the terminator (250 ms
  continuation timeout).
- RCON Lua errors do NOT kill the server session — wrap in pcall and return
  `{error = ...}` JSON.

### CLI / config quirks

- `--port N` **conflicts** with `--bind host:port` → use `--bind` only.
- `server-settings.json`: `allow_commands` must be the **string** `"true"`.
- `map-settings.json` must contain **every** prototype key; exact list in
  `worker_config.py::MAP_SETTINGS`, cross-checked against
  `D:/Factorio/data/base/prototypes/map-settings.lua`.
- `write-data/.lock` prevents a second instance on the same data dir; an
  orphaned engine holds it indefinitely.
- The **game port is UDP** (RCON is TCP); a TCP-only probe never collides.
- `--create` exit-code + missing save file → classified `invalid_save`.

---

## Key files

```
mod/factoriorl/            Lua runtime: bridge, dispatcher, stepping, actions, scene
src/factoriorl/
  engine_config.py         engine discovery + 2.0.60/83512 pin
  paths.py                 workspace/runtime layout (incl. runtime/ports, docs/evidence)
  worker_config.py         per-worker configs (server/map-gen/map-settings, pid file)
  modpack.py               mod zip + mod-list (DLCs disabled), rebuilt every launch
  worker.py                launch/shutdown, port reservations, job object, classification
  rcon.py                  Source RCON client (two-NUL, /c, rcon.print)
  protocol.py              protocol v1 types (frozen)
  session.py               typed session, nonced ids, settle loop, ActionResult
  supervisor.py            policy, liveness, restart-from-save, evidence
  pool.py                  WorkerPool orchestrator, atexit, orphan reaping
  gate_phase0.py           one-command Phase 0 gate
tests/unit/                engine-free protocol/config/port tests
tests/contract/            fixture structure + Python↔Lua vocabulary agreement
tests/engine/              fixture agreement + lifecycle/supervision/pool fault tests
tests/fixtures/protocol_v1/  10 frozen wire fixtures
docs/LEDGER.md             phase status + evidence index
docs/evidence/             committed gate transcripts
```

## Commands

```
uv sync
uv run factoriorl doctor            # engine probe
uv run factoriorl worker start      # single worker via the pool
uv run factoriorl worker reap       # kill engines orphaned by a dead parent
uv run factoriorl phase0-gate       # Phase 0 exit gate
uv run factoriorl phase1-gate       # Phase 1 exit gate + evidence
uv run pytest tests/unit tests/contract
uv run ruff check . && uv run ruff format --check .
```

## Prerequisites and risks for Phase 2

Phase 2 owns the action matrix, ongoing actions, local observations, entity
identity, and profiles. What it inherits:

- **Ongoing actions start from a single-slot design.** `advance` is the only
  multi-interval operation and the runtime tracks exactly one
  (`pending_advance_request_id`), refusing a second. PLAN 2.2 needs progress
  across `step` calls, cancellation at a documented boundary, and
  completion reported once — that needs a general in-flight table, and
  `ActionStatus.RUNNING/FAILED/CANCELLED` need their first real use.
- **Observations are a fixed reference scene**, not the 32-tile sensor region
  PLAN 2.3 requires: `observations.snapshot` reports the character plus two
  named chests. There is no visibility filtering, no remembered-observation
  age, and no entity identity/lifecycle (2.4).
- **Ledger persistence across autosave/reload is assumed, not asserted.**
  `storage.frrl_state` is restored in `on_load` and the load path runs on every
  launch, but no test forces a save/reload mid-episode. Longer Phase 2/3
  episodes will actually hit autosaves.
- **`bridge.run` shares the RCON port with the typed protocol.** The typed
  dispatch cannot reach it (verified), but anything holding the RCON password
  holds arbitrary Lua. There is no agent-facing surface yet; this becomes a
  real separation requirement in Phase 5.
- **Non-Windows has no job object.** Layer 1 of parent-crash cleanup is
  Windows-only; `atexit` and `reap_orphans` still apply. Linux is out of scope
  for the initial release, but the gap is real and the test skips there.

## Open questions / notes for the next phase

- `advance` rejects a second advance while one runs (`precondition`) — fine for
  the single-client contract; vectorized envs serialize per worker anyway.
- Movement is `walking_state` + deadline; no pathfinding yet (Phase 5.1).
- `until-tick` was never evaluated as an alternative to `tick_paused`. Revisit
  only with profiling evidence (PLAN.md 11.4).
- RCON is single-threaded server-side; if latency matters, batch requests per
  round trip before changing transport.
