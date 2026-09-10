"""Saves that are verified before they are reported.

A1.3's sharpest clause is "never report a nonexistent final save", so most of
these are about the ways a save can appear to have worked and not have.
"""

from __future__ import annotations

import subprocess
import sys as _sys
import threading
import time
from pathlib import Path

import pytest

from factoriorl.agent.checkpoint import Checkpointer, newest_checkpoint

ROOT = Path(__file__).resolve().parents[2]


class FakeSession:
    """A session that writes the save file itself, on demand.

    Standing in for the engine lets these tests exercise the exact thing that
    cannot be faked at the mod layer: the gap between "issued" and "on disk".
    """

    def __init__(self, saves_dir: Path, *, paused: bool = False, writes: bool = True) -> None:
        self.saves_dir = saves_dir
        self.paused = paused
        self.writes = writes
        self.saved: list[str] = []
        self.advanced = 0
        #: Bytes written per chunk, and how many chunks. More than one chunk
        #: rehearses a file that exists before it is finished.
        self.chunks = 1
        self.chunk_delay = 0.0

    def save(self, name: str):
        self.saved.append(name)
        if self.writes:
            threading.Thread(target=self._write, args=(name,), daemon=True).start()
        return _Timed({"issued": True, "name": name, "tick": 7, "paused": self.paused})

    def _write(self, name: str) -> None:
        self.saves_dir.mkdir(parents=True, exist_ok=True)
        target = self.saves_dir / f"{name}.zip"
        for _ in range(self.chunks):
            time.sleep(self.chunk_delay)
            with target.open("ab") as handle:
                handle.write(b"factorio-save-bytes")

    def advance(self, ticks: int) -> None:
        self.advanced += ticks


class _Timed:
    def __init__(self, result: dict) -> None:
        self.response = type("R", (), {"result": result})()


def _checkpointer(tmp_path: Path, **kwargs) -> tuple[Checkpointer, FakeSession]:
    write_data = tmp_path / "write-data"
    session = FakeSession(
        write_data / "saves", **{k: v for k, v in kwargs.items() if k in {"paused", "writes"}}
    )
    keeper = Checkpointer(
        session=session,
        write_data=write_data,
        run_dir=tmp_path / "run",
        interval_seconds=kwargs.get("interval_seconds", 300.0),
        timeout_seconds=kwargs.get("timeout_seconds", 5.0),
    )
    return keeper, session


# --- a save that works -------------------------------------------------------


def test_a_verified_save_is_copied_out_of_the_worker_directory(tmp_path) -> None:
    """`WorkerManager.cleanup` rmtree's the worker directory at teardown, and the
    engine writes saves inside it. A checkpoint left there is deleted with it."""
    keeper, session = _checkpointer(tmp_path)
    record = keeper.save("initial")

    assert record["verified"] is True
    kept = Path(record["path"])
    assert kept.is_file()
    assert kept.parent == tmp_path / "run" / "saves"
    assert record["bytes"] > 0
    assert len(record["sha256"]) == 64
    assert keeper.latest is record


def test_every_save_gets_a_unique_name(tmp_path) -> None:
    """A fixed name makes a stale file indistinguishable from a fresh one, and
    resuming the wrong world is the failure that matters. `tools/capture.py`
    learned the same lesson for screenshots."""
    keeper, session = _checkpointer(tmp_path)
    keeper.save("initial")
    time.sleep(1.05)  # the stamp has second resolution
    keeper.save("periodic")
    assert len(set(session.saved)) == 2


def test_a_file_that_is_still_growing_is_waited_for(tmp_path) -> None:
    """ "Exists" is not "finished": the engine writes the zip incrementally, so a
    single size read can catch it mid-write and copy out a truncated archive."""
    keeper, session = _checkpointer(tmp_path)
    session.chunks, session.chunk_delay = 4, 0.1
    record = keeper.save("initial")
    assert record["verified"] is True
    assert record["bytes"] == 4 * len(b"factorio-save-bytes")


# --- a save that does not ----------------------------------------------------


def test_a_save_that_never_appears_is_a_failure_not_a_checkpoint(tmp_path) -> None:
    """The clause this module exists for: never report a nonexistent save."""
    keeper, session = _checkpointer(tmp_path, writes=False, timeout_seconds=0.6)
    record = keeper.save("final")

    assert record["verified"] is False
    assert "error" in record
    assert "never settled" in record["error"]
    assert keeper.latest is None
    assert keeper.to_dict()["failed"] == 1


def test_a_later_failure_does_not_discard_the_last_verified_checkpoint(tmp_path) -> None:
    """A1.3: retain the latest verified checkpoint."""
    keeper, session = _checkpointer(tmp_path, timeout_seconds=2.0)
    good = keeper.save("initial")
    session.writes = False
    keeper._last_attempt = None
    bad = keeper.save("final")

    assert good["verified"] and not bad["verified"]
    assert keeper.latest is good
    assert keeper.to_dict()["latest_verified"]["label"] == "initial"


def test_saving_never_raises_into_the_run(tmp_path) -> None:
    """A run that cannot save should carry on and say so, not die at minute
    twenty-nine holding a world it was about to lose anyway."""

    class Broken(FakeSession):
        def save(self, name: str):
            raise RuntimeError("rcon went away")

    keeper = Checkpointer(
        session=Broken(tmp_path / "write-data" / "saves"),
        write_data=tmp_path / "write-data",
        run_dir=tmp_path / "run",
        timeout_seconds=0.5,
    )
    record = keeper.save("periodic")
    assert record["verified"] is False
    assert "rcon went away" in record["error"]


# --- the paused-world trap ---------------------------------------------------


def test_a_paused_world_is_advanced_so_the_deferred_save_can_run(tmp_path) -> None:
    """`game.server_save` executes at end of tick, and under exact stepping the
    world is paused between decisions. Without a tick the save is issued and
    silently never happens."""
    keeper, session = _checkpointer(tmp_path, paused=True)
    record = keeper.save("initial")
    assert session.advanced == 1
    assert record["advanced_a_tick"] is True
    assert record["world_was_paused"] is True


def test_a_running_world_is_not_advanced(tmp_path) -> None:
    keeper, session = _checkpointer(tmp_path, paused=False)
    keeper.save("initial")
    assert session.advanced == 0


# --- the cadence -------------------------------------------------------------


def test_maybe_save_does_nothing_before_the_interval_elapses(tmp_path) -> None:
    keeper, session = _checkpointer(tmp_path, interval_seconds=300.0)
    assert keeper.maybe_save() is not None
    assert keeper.maybe_save() is None
    assert keeper.maybe_save() is None
    assert len(session.saved) == 1


def test_maybe_save_fires_once_the_interval_has_passed(tmp_path) -> None:
    keeper, session = _checkpointer(tmp_path, interval_seconds=0.05)
    keeper.maybe_save()
    time.sleep(0.06)
    assert keeper.maybe_save() is not None
    assert len(session.saved) == 2


def test_a_failed_save_still_resets_the_interval(tmp_path) -> None:
    """Otherwise a broken save is retried on every decision, and a run whose
    provider is fine spends its wall clock hammering a disk that is not."""
    keeper, session = _checkpointer(
        tmp_path, writes=False, interval_seconds=300.0, timeout_seconds=0.4
    )
    keeper.maybe_save()
    assert keeper.maybe_save() is None


# --- finding one again -------------------------------------------------------


def test_the_newest_checkpoint_is_found_by_run_directory(tmp_path) -> None:
    """By modification time, not by name.

    Names are `frrl-<label>-<stamp>.zip`, so a lexicographic sort orders by
    label: `frrl-final-...` sorts *before* `frrl-initial-...`. That would resume
    the world as it was before the run rather than after it -- which is why the
    labels here are deliberately the wrong way round alphabetically.
    """
    keeper, session = _checkpointer(tmp_path)
    first = keeper.save("initial")
    time.sleep(1.05)
    second = keeper.save("final")
    assert Path(second["path"]).name < Path(first["path"]).name, (
        "this test is only meaningful while the labels sort against the order"
    )
    assert newest_checkpoint(tmp_path / "run") == Path(second["path"])


def test_a_run_with_no_saves_reports_none(tmp_path) -> None:
    assert newest_checkpoint(tmp_path / "empty") is None


# --- the resume rule ---------------------------------------------------------


def _cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [_sys.executable, "-m", "factoriorl.cli", "agent", *args],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )


def test_a_resume_will_not_silently_take_a_fresh_allowance() -> None:
    """A1.3: "Do not automatically spend another budget on resume." Inheriting
    the parent's cap and falling back to the default both do exactly that."""
    result = _cli("--task", "open_factory", "--resume-from", "some-run")
    assert result.returncode == 2
    assert "explicit --max-cost-usd" in result.stdout + result.stderr


def test_a_resume_from_a_missing_checkpoint_fails_before_launching_a_worker() -> None:
    result = _cli("--task", "open_factory", "--resume-from", "no-such-run", "--max-cost-usd", "1")
    assert result.returncode == 2
    assert "no verified checkpoint" in result.stdout + result.stderr


def test_a_benchmark_task_cannot_be_resumed() -> None:
    """Its scene is rebuilt from a declared blueprint on every reset, so there is
    nothing in a save for it to resume."""
    save = ROOT / "pyproject.toml"  # any existing file; the check is on the task
    result = _cli(
        "--task",
        "deliver",
        "--resume-from",
        str(save),
        "--max-cost-usd",
        "1",
        "--model",
        "local-model",
    )
    assert result.returncode == 2
    assert "open worlds only" in result.stdout + result.stderr


@pytest.mark.parametrize("label", ["initial", "periodic", "final"])
def test_labels_survive_into_the_record(tmp_path, label: str) -> None:
    keeper, _ = _checkpointer(tmp_path)
    assert keeper.save(label)["label"] == label
