"""Operational sampling on a wall clock, not on the agent's turn (A4.3).

Roadmap A4.3: "Sample lightweight operational metrics approximately every five
seconds, including during model calls, without sharing an unsafe concurrent RCON
stream; serialize game access through one owner."

**Why a thread is the only shape that works here.** Production was sampled once
per environment step, inside `env._adopt`. In an exactly-stepped run that is
every 30 ticks and perfectly regular. In the realtime run A5 will make, the gap
between two samples is *the model's latency* -- tens of seconds, during which
the world keeps running -- and `ProductionMetrics` refuses a window it did not
observe densely enough (`max_sample_gap`, 30 ticks). So without this, A4.2's
60-second windows would be rejected as unsampled almost every time, and the run
report would have production numbers with nothing behind them.

**Why it is safe.** `RCONClient` holds a lock across a whole request/response
exchange, so the socket is the single owner A4.3 asks for. The sampler shares
the session rather than opening a second connection: a second connection would
be a second owner of game access, which is the thing the clause rules out. The
contention that remains is benign and lands where it helps -- during a model
call the loop holds nothing, and a model call is where the long gaps are.

**What it does not do.** It never acts. It reads `world.truth` and nothing else,
so it cannot change the run it is measuring, and it cannot advance a tick.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: How often to sample, in wall-clock seconds. A4.3 says "approximately every
#: five seconds"; the cadence is approximate by construction because a sample
#: may wait on the connection the agent is using.
DEFAULT_INTERVAL_SECONDS = 5.0


@dataclass
class Sampler:
    """Reads the world on a wall clock and appends one JSON line per sample."""

    session: Any
    destination: Path
    interval_seconds: float = DEFAULT_INTERVAL_SECONDS
    #: Counted rather than raised. A sampler that killed a thirty-minute run
    #: because one read timed out would be trading the thing being measured for
    #: the measurement.
    samples: int = 0
    failures: int = 0
    _thread: threading.Thread | None = None
    _stop: threading.Event = field(default_factory=threading.Event)
    _started_at: float = 0.0
    _errors: list[str] = field(default_factory=list)

    def start(self) -> Sampler:
        if self._thread is not None:
            return self
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self._started_at = time.perf_counter()
        self._thread = threading.Thread(target=self._run, name="frrl-sampler", daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout: float = 10.0) -> Sampler:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        return self

    # ------------------------------------------------------------- internals

    def _run(self) -> None:
        # `wait` rather than `sleep`, so stopping is immediate rather than up to
        # one interval away -- at the deadline the run is already over and every
        # further second is finalization time nobody wants to pay.
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def _sample(self) -> None:
        at = time.perf_counter()
        try:
            truth = self.session.truth().response.result or {}
        except Exception as failure:  # noqa: BLE001 - recorded, never fatal
            self.failures += 1
            if len(self._errors) < 8:
                self._errors.append(f"{type(failure).__name__}: {failure}")
            return
        row = {
            "wall_seconds": round(at - self._started_at, 3),
            # How long the read itself waited, which is mostly how long the
            # agent's own exchange held the connection. Recorded because a
            # cadence that drifts is a fact about the run, not a detail.
            "read_ms": round((time.perf_counter() - at) * 1000, 1),
            "tick": truth.get("tick"),
            "produced": truth.get("produced") or {},
            "machine_produced": truth.get("machine_produced") or {},
            "handcrafted": truth.get("handcrafted") or {},
            "mined_by_hand": truth.get("mined_by_hand") or {},
            "placed_counts": truth.get("placed_counts") or {},
            "working_counts": truth.get("working_counts") or {},
        }
        with self.destination.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, default=str) + "\n")
        self.samples += 1

    def to_dict(self) -> dict:
        return {
            "interval_seconds": self.interval_seconds,
            "samples": self.samples,
            "failures": self.failures,
            "errors": list(self._errors),
            "path": str(self.destination),
            "measures": (
                "world truth read on a wall clock, independent of the agent's "
                "decision rate; never acts and never advances a tick"
            ),
        }
