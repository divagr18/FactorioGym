# Transport round-trip: before and after

PLAN.md 11.4 requires every optimization to carry a before/after measurement and
to leave semantics unchanged. This records both for the Phase T transport work.

Engine: Factorio 2.0.60 (build 83512, win64). Host: Ryzen 7 5800H, RTX 4060 laptop,
Windows 11. Commands: `uv run factoriorl bench transport`,
`uv run factoriorl bench speed`, `uv run factoriorl phase0-gate`.

## What was wrong

Three independent problems, none of which any test could see.

**1. The published latency number measured something other than its name.**
`gate_phase0.py` averaged the *seven per-request* latencies of a cycle and stored
the result under `mean_per_cycle_ms`. The committed 305 ms was therefore the cost
of one **request**; a cycle actually cost ~2423 ms. `docs/LEDGER.md` and
`HANDOFF.md` both repeated the misreading, and both also said "6 requests" where
the code timed 7.

**2. Every call paid a 250 ms timeout.** `RCONClient.command` entered a
continuation loop whenever a response body was non-empty, ending it only on an
empty terminator chunk — which Factorio never sends for short responses. So the
loop always ran out the full `CONTINUATION_TIMEOUT`.

**3. `game.speed` was never set.** A headless server targets 60 UPS, so a 30-tick
decision interval cost a hard 500 ms of wall clock. This was invisible because
`WorkerSession._settle` reported only its first dispatch plus its last probe,
hiding every intermediate probe and sleep.

## Measured, on one connection

The stall is isolated by timing two commands that differ only in whether they
print: an empty response short-circuits the continuation loop, a non-empty one
runs it out. The difference *is* the stall.

| Measurement | Before | After |
|---|---|---|
| Zero-output command (`/c local _ = 1`) | 16.66 ms | 16.65 ms |
| Printing command (`/c rcon.print("x")`) | 266.69 ms | **16.66 ms** |
| **Continuation stall** | **250.03 ms** | **0.00 ms** |
| `advance(30)` wall clock, `game.speed = 1` | 818.57 ms | 533.22 ms |
| `advance(30)` wall clock, `game.speed = 30` | 525.21 ms | **25.56 ms** |

`advance(30)` improved **32×**. The stall was exactly `CONTINUATION_TIMEOUT`, to
within 0.03 ms.

Phase 0 gate, same 7-request cycle, `game.speed = 1`:

| | Before | After |
|---|---|---|
| Mean per request | 267 ms (published as 305 ms "per cycle") | **90.61 ms** |
| Mean per cycle | ~2423 ms (never published) | **634.25 ms** |

The per-request figure is still 90 ms at speed 1 only because the cycle contains
one `advance(30)`, which spends 500 ms waiting for the engine to tick. At
`game.speed = 30` that term nearly vanishes.

## The fixes

**Sentinel framing** replaces the timeout with a fact. Each call now takes a
fresh id pair: the real command is sent with id *N*, followed immediately by a
command known to print nothing with id *N+1*. A Source RCON server answers in
order on one connection, so the arrival of the sentinel's reply proves the real
response is complete. Exact, and it costs one extra packet instead of 250 ms.

A zero-output command was confirmed to answer promptly with an empty body
(16.66 ms), which is the precondition the design rests on. Five candidate
sentinel bodies were probed for game-state side effects; **all five advance
`game.tick` by exactly 0.00 per command**, so the sentinel cannot perturb the
simulation.

Per-call ids also fix a latent correctness bug. Every call previously reused one
id, so a single timeout desynchronized the connection *permanently*: the late
reply arrived during the next call and every call thereafter raised "unexpected
rcon packet id". Replies bearing an id older than the current call are now
drained as the stale packets they are.

**`TCP_NODELAY`** is set on the client socket. Each request is one small write,
and the sentinel makes it two back to back — precisely the shape Nagle plus
delayed ACK turns into a ~40 ms stall.

**`game.speed`** is settable through the evaluator channel (Phase 2 gives it a
typed `configure` request).

## Semantics unchanged

`uv run factoriorl bench speed` runs the same reset/move/advance/transfer cycle
at `game.speed` 1 and 30 on one worker and compares the episode records field by
field — episode tick, advance result, character position, character inventory,
both chest contents, and task counters.

```
speeds:            [1.0, 30.0]
cycles per speed:  5
wall_ms:           speed_1 = 3166.32,  speed_30 = 202.45
speedup:           speed_30 = 15.64x
mismatches:        []
identical:         true
```

**15.6× faster, byte-identical outcomes.**

The Phase 0 gate passes unchanged (`passed: true`, no failures, mixed drift 0,
idle drift 0), and the full engine suite is 33/33.

## A latent bug the speedup exposed

`test_unknown_action_cannot_reach_arbitrary_execution` began failing with
`assert 1 == 4`: the game tick advanced while the world was supposed to be
paused. The sentinel was not the cause — the probe above shows every candidate
is tick-free.

The real defect was in `runtime.lua`: `handle_reset` cleared the in-flight
advance bookkeeping but never re-paused the world. A reset landing while an
advance was still running left the engine ticking with nothing to stop it. The
250 ms stall had been masking it — at 267 ms per request, an advance always
settled and re-paused itself before the next observation. At 16 ms per request
it does not.

`begin_episode` now sets `game.tick_paused = true` explicitly. PLAN.md section 2:
the world pauses between decisions.

This is worth stating plainly: making the transport 16× faster turned a dormant
correctness bug into a visible test failure. The speedup did not cause it; it
removed the accident that hid it.

## Coverage

`tests/unit/test_rcon_framing.py` drives a fake RCON server that reproduces
what a live Factorio cannot be asked for on demand: a response with no
terminator chunk, a response split across packets, and a late reply from a call
that already timed out. Six engine-free tests, run on every commit.
