"""A4.3: sampling on a wall clock, and stopping a run properly.

Two things this section fixes, both of which were quietly wrong rather than
missing:

**Sampling followed the agent's turn, not the clock.** Production was recorded
once per environment step. In an exactly-stepped run that is every 30 ticks. In
the realtime run A5 makes, the gap between samples is the *model's latency* --
tens of seconds -- and `ProductionMetrics._sampled()` refuses a window it did
not observe densely enough. A4.2's 60-second windows would have been rejected
as unsampled almost every time.

**`RunClock` published a claim that was false.** `to_dict()` says "gameplay
only; worker launch and final snapshot excluded", and `clock.stop()` ran after
the final save, the viewer teardown and `--hold-open`.
"""

from __future__ import annotations

import json
import threading
import time

from factoriorl.agent.runner import finalize
from factoriorl.agent.sampler import Sampler


class FakeResponse:
    def __init__(self, result):
        self.result = result


class FakeTimed:
    def __init__(self, result):
        self.response = FakeResponse(result)


class FakeSession:
    """Counts what was asked of it, and can be made slow or made to fail."""

    def __init__(self, *, delay: float = 0.0, fail_truth: bool = False):
        self.delay = delay
        self.fail_truth = fail_truth
        self.calls: list[str] = []
        self.tick = 0
        self._lock = threading.Lock()

    def truth(self):
        if self.fail_truth:
            raise OSError("the worker stopped answering")
        with self._lock:
            self.calls.append("truth")
            self.tick += 60
            tick = self.tick
        time.sleep(self.delay)
        return FakeTimed({"tick": tick, "machine_produced": {"iron-plate": tick / 60}})

    def act(self, action, **payload):
        self.calls.append(f"act:{action}:{payload.get('target_request_id')}")
        return FakeTimed({"status": "completed"})

    def configure(self, speed=None, *, free_running=None):
        self.calls.append(f"configure:free_running={free_running}")
        return FakeTimed({"applied": {"free_running": free_running}})


class FakeCheckpointer:
    def __init__(self, fail: bool = False):
        self.saved: list[str] = []
        self.fail = fail

    def save(self, label):
        if self.fail:
            raise RuntimeError("the save never materialised")
        self.saved.append(label)
        return {"label": label, "verified": True}


# ------------------------------------------------------------------ sampler


def test_the_sampler_writes_on_its_own_clock(tmp_path):
    session = FakeSession()
    destination = tmp_path / "production.jsonl"
    sampler = Sampler(session=session, destination=destination, interval_seconds=0.02).start()
    time.sleep(0.25)
    sampler.stop()

    rows = [json.loads(line) for line in destination.read_text().splitlines()]
    assert len(rows) >= 4, rows
    assert sampler.samples == len(rows)
    # Ticks advance across samples, so these are distinct reads of the world
    # rather than one read written repeatedly.
    assert [row["tick"] for row in rows] == sorted(row["tick"] for row in rows)
    assert rows[0]["machine_produced"]


def test_a_sampler_failure_is_counted_and_never_raised(tmp_path):
    """Killing a thirty-minute run because one read timed out would trade the
    thing being measured for the measurement."""
    session = FakeSession(fail_truth=True)
    destination = tmp_path / "production.jsonl"
    sampler = Sampler(session=session, destination=destination, interval_seconds=0.02).start()
    time.sleep(0.12)
    sampler.stop()

    assert sampler.samples == 0
    assert sampler.failures > 0
    assert sampler.to_dict()["errors"]
    assert not destination.exists() or destination.read_text() == ""


def test_stopping_does_not_wait_out_the_interval(tmp_path):
    """At the deadline the run is over and every extra second is finalization
    time nobody wants to pay, so the wait is interruptible."""
    session = FakeSession()
    sampler = Sampler(
        session=session, destination=tmp_path / "p.jsonl", interval_seconds=30.0
    ).start()
    at = time.perf_counter()
    sampler.stop()

    assert time.perf_counter() - at < 5.0


def test_the_sampler_only_ever_reads(tmp_path):
    session = FakeSession()
    sampler = Sampler(session=session, destination=tmp_path / "p.jsonl", interval_seconds=0.02)
    sampler.start()
    time.sleep(0.1)
    sampler.stop()

    # It cannot change the run it is measuring, and it cannot advance a tick.
    assert set(session.calls) == {"truth"}


def test_the_sampler_feeds_the_windowing_metrics(tmp_path):
    """The reason the sampler exists, not a side effect of it.

    `ProductionMetrics` refuses a window it did not observe densely enough, and
    in a realtime run the environment's own cadence is the model's latency. A
    file full of samples that the windowing code never sees would leave every
    60-second window thrown out as unsampled -- which is the state this fixes.
    """
    from factoriorl import worlds
    from factoriorl.production import ProductionMetrics

    metrics = ProductionMetrics.for_world(worlds.get("open_factory"))
    session = FakeSession()
    sampler = Sampler(
        session=session,
        destination=tmp_path / "production.jsonl",
        metrics=metrics,
        interval_seconds=0.02,
    ).start()
    time.sleep(0.2)
    sampler.stop()

    report = metrics.report()
    assert report["samples"] >= 4
    assert metrics.items == ("iron-plate",)
    assert report["counts"] == "machine_produced"
    assert sampler.to_dict()["feeds_metrics"] is True


def test_metrics_survive_being_recorded_from_two_threads(tmp_path):
    """The environment's own thread records too. Two threads appending to one
    list is a race whose outcome is silent: a dropped sample is indistinguishable
    from a window nobody observed."""
    from factoriorl import worlds
    from factoriorl.production import ProductionMetrics

    metrics = ProductionMetrics.for_world(worlds.get("open_factory"))
    stop = threading.Event()

    def busy():
        tick = 1
        while not stop.is_set():
            metrics.record({"tick": tick}, {"machine_produced": {"copper-plate": tick}})
            tick += 60

    other = threading.Thread(target=busy, daemon=True)
    other.start()
    sampler = Sampler(
        session=FakeSession(),
        destination=tmp_path / "p.jsonl",
        metrics=metrics,
        interval_seconds=0.01,
    ).start()
    time.sleep(0.3)
    sampler.stop()
    stop.set()
    other.join(timeout=5)

    assert set(metrics.items) == {"iron-plate", "copper-plate"}
    assert metrics.report()["samples"] > 0


# ------------------------------------------------------------- finalization


def test_finalization_cancels_then_pauses_then_saves():
    """A4.3 names the order, and the order is the point: a walk still in flight
    would keep moving the character while the snapshot is taken."""
    session = FakeSession()
    checkpointer = FakeCheckpointer()
    observation = {"inflight": [{"request_id": "step-7:act", "action": "navigate"}]}

    record = finalize(session, checkpointer, observation=observation)

    # The two `truth` reads are the tick either side of the save.
    assert session.calls == [
        "act:cancel:step-7:act",
        "configure:free_running=False",
        "truth",
        "truth",
    ]
    assert checkpointer.saved == ["final"]
    assert [entry["step"] for entry in record["steps"]] == [
        "cancel:step-7:act",
        "pause",
        "save",
    ]
    assert record["cancelled"] == ["step-7:act"]
    assert not record["failed_steps"]


def test_the_world_is_shown_to_be_still_across_the_save():
    """ "Paused" is otherwise a claim about a knob rather than about the world.

    The fake advances its tick on every read, so this is the case where the
    pause did *not* take -- and the record has to say so rather than reporting
    a clean finalization.
    """
    record = finalize(FakeSession(), FakeCheckpointer(), observation={})

    # The fake advances 60 ticks per read, so this is the case where the pause
    # did not take. One tick would have been fine -- that is the engine's floor,
    # since `tick_paused` set inside a tick lets that tick finish -- and sixty
    # is not.
    assert record["ticks_across_the_save"] == 60
    assert record["world_was_still_for_the_save"] is False


def test_one_tick_across_the_save_is_the_floor_not_a_failure():
    class BarelyMoving(FakeSession):
        def truth(self):
            self.calls.append("truth")
            self.tick += 1
            return FakeTimed({"tick": self.tick})

    record = finalize(BarelyMoving(), FakeCheckpointer(), observation={})

    assert record["ticks_across_the_save"] == 1
    assert record["world_was_still_for_the_save"] is True


def test_nothing_in_flight_still_pauses_and_saves():
    session = FakeSession()
    checkpointer = FakeCheckpointer()

    record = finalize(session, checkpointer, observation={"inflight": []})

    assert record["cancelled"] == []
    assert checkpointer.saved == ["final"]


def test_a_failed_step_does_not_cost_the_run_its_save():
    """A finalization that half-worked is the case a reader most needs told."""

    class Refusing(FakeSession):
        def act(self, action, **payload):
            raise RuntimeError("the request had already settled")

    session = Refusing()
    checkpointer = FakeCheckpointer()
    observation = {"inflight": [{"request_id": "step-3:act"}]}

    record = finalize(session, checkpointer, observation=observation)

    assert record["failed_steps"] == ["cancel:step-3:act"]
    # Pause and save still ran.
    assert checkpointer.saved == ["final"]
    assert "the request had already settled" in record["steps"][0]["error"]


def test_a_failed_save_is_recorded_rather_than_raised():
    record = finalize(FakeSession(), FakeCheckpointer(fail=True), observation={})

    assert record["failed_steps"] == ["save"]
    assert "never materialised" in record["steps"][-1]["error"]


def test_finalization_is_timed_separately_from_gameplay():
    """`RunClock.to_dict()` claims the final snapshot is excluded. It is only
    true if finalization is measured on its own, which is what this is."""
    record = finalize(FakeSession(), FakeCheckpointer(), observation={})

    assert record["seconds"] >= 0.0
    assert "outside gameplay" in record["measures"]
    assert all("seconds" in entry for entry in record["steps"])
