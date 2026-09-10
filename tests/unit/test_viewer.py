"""Watching a run live, and reporting honestly what watching cost.

The mod side is already correct -- `control.lua` makes a joiner a spectator with
no character -- so these are about the Python half, and specifically about the
two things roadmap A1.2 asks for that `tools/watch_agent.py` does not do:
observe viewer presence rather than assert it, and leave a failed viewer as a
recorded warning on a run that still finishes.
"""

from __future__ import annotations

from pathlib import Path

from factoriorl.agent.viewer import Viewer


class FakeProcess:
    """A client that is alive until it is not."""

    def __init__(self, exit_code: int | None = None) -> None:
        self.returncode = exit_code
        self.terminated = False

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -1


def _viewer(**kwargs) -> Viewer:
    return Viewer(requested=True, address="127.0.0.1:23400", warmup_seconds=0.0, **kwargs)


# --- not asking for one ------------------------------------------------------


def test_a_headless_run_starts_no_client() -> None:
    viewer = Viewer.start(
        requested=False,
        executable=Path("nowhere/factorio.exe"),
        address="127.0.0.1:1",
        mod_directory=Path("nowhere"),
    )
    assert viewer.process is None
    described = viewer.to_dict()
    assert described["requested"] is False
    assert described["launched"] is False
    assert described["measurement"] == "headless"


# --- asking for one that cannot start ----------------------------------------


def test_a_client_that_cannot_start_leaves_the_run_alive_and_says_so() -> None:
    """A1.2: a failed viewer must leave headless execution and replay available,
    with an explicit warning. `watch_agent.py` prints one to stdout and records
    nothing, so a run artifact could not tell you the viewer never appeared."""
    viewer = Viewer.start(
        requested=True,
        executable=Path("D:/definitely/not/here/factorio.exe"),
        address="127.0.0.1:23400",
        mod_directory=Path("."),
    )
    assert viewer.process is None
    assert viewer.joined is False
    assert viewer.warnings, "a failed launch must be recorded, not just printed"
    assert "continues headless" in viewer.warnings[0]
    assert "23400" in viewer.warnings[0], "tell the user where to join manually"


# --- asking for one that starts and then dies --------------------------------


def test_a_client_that_dies_during_warmup_is_noticed() -> None:
    """`watch_agent.py` catches only an `OSError` at spawn and consults
    `poll()` once, in its teardown. A version mismatch or a refused join looks
    exactly like this, and used to leave a run reporting itself as watched with
    nobody watching."""
    viewer = _viewer(process=FakeProcess(exit_code=1))
    viewer.wait_for_join()

    assert viewer.joined is False
    assert viewer.warnings and "exited during warmup" in viewer.warnings[0]
    assert viewer.to_dict()["joined"] is False


def test_a_client_that_survives_warmup_is_recorded_as_joined() -> None:
    viewer = _viewer(process=FakeProcess())
    viewer.wait_for_join()
    assert viewer.joined is True
    assert viewer.alive is True


# --- what the artifact says --------------------------------------------------


def test_viewer_presence_is_observed_rather_than_asserted() -> None:
    """The difference this module exists for.

    `tools/watch_agent.py` writes a literal `"watched": True` into its
    provenance whether or not a client ever started.
    """
    requested_but_failed = Viewer(requested=True, address="a", warmup_seconds=0.0)
    assert requested_but_failed.to_dict()["launched"] is False

    actually_running = _viewer(process=FakeProcess())
    assert actually_running.to_dict()["launched"] is True


def test_a_watched_run_declares_itself_a_demonstration() -> None:
    """A joined client creates a LuaPlayer no measured run has, and the server
    settings let it run console commands and pause the world."""
    described = _viewer(process=FakeProcess()).to_dict()
    assert "demonstration" in described["measurement"]
    assert "LuaPlayer" in described["measurement"]


def test_the_rcon_reply_inversions_a_client_causes_are_persisted() -> None:
    """Measured at 89 across one watched run and 0 on every player-free run.
    The transport already counts them; they were printed and then forgotten."""
    described = _viewer(process=FakeProcess()).to_dict(reordered_replies=89)
    assert described["rcon_reordered_replies"] == 89

    # Absent, not null, when the count is not yet known. The manifest is written
    # before the first decision, and a `null` under this name reads as "no
    # inversions" rather than "not measured yet".
    assert "rcon_reordered_replies" not in _viewer(process=FakeProcess()).to_dict()


# --- teardown ----------------------------------------------------------------


def test_hold_open_keeps_the_final_state_on_screen_then_closes() -> None:
    process = FakeProcess()
    viewer = _viewer(process=process)
    viewer.stop(hold_open_seconds=0.0)
    assert process.terminated is True


def test_stopping_a_client_that_already_exited_is_harmless() -> None:
    process = FakeProcess(exit_code=0)
    viewer = _viewer(process=process)
    viewer.stop()
    assert process.terminated is False


def test_stopping_when_there_is_no_client_is_harmless() -> None:
    Viewer(requested=False, address="a").stop(hold_open_seconds=5.0)
