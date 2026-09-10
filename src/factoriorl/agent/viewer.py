"""Watching a run in a real Factorio client, and saying honestly what that cost.

Every worker is a dedicated server -- `worker.py` starts the engine with
`--start-server` -- so a second Factorio can join and render the world the agent
is driving. `mod/factoriorl/control.lua` already makes any joining player a
spectator on both `on_player_created` and `on_player_joined_game`: it exits the
intro cutscene and destroys the engine-given character, with an identity guard so
it can never destroy `storage.frrl_character`. A spectator has no body, so it
cannot reach, mine, build or transfer.

What this module adds is the Python half, and one property the existing
live-watching tool does not have: **viewer presence is observed rather than
asserted.** `tools/watch_agent.py` records a hardcoded `"watched": True` whether
or not the client launched, and catches only an `OSError` at spawn -- a client
that starts and then dies is never noticed. Roadmap A1.2 asks for the opposite:
record viewer presence and any unavoidable perturbation, and leave headless
execution and replay available with an explicit warning if the viewer fails.

The perturbation is real and is recorded, not fixed
---------------------------------------------------
A joined client creates a `LuaPlayer` that no measured run has. It also reorders
RCON replies: measured at **89 inversions across one watched run and 0 on every
player-free run** (`tests/unit/test_rcon_framing.py`). And the server settings
permit a connected client to run console commands and to pause the world
(`allow_commands`, `only_admins_can_pause_the_game: False`). Bind is
`127.0.0.1`, so only a same-machine viewer can join at all.

None of that is repaired here. A watched run is a demonstration rather than a
measurement, and the artifact says so.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

#: Seconds to let the client finish joining before the first decision. A join
#: makes the server transfer the map and stall, and the first watched run ever
#: attempted hit that stall *inside* a step: the request raised an
#: infrastructure failure and the episode was truncated at decision three.
DEFAULT_WARMUP_SECONDS = 30.0


@dataclass
class Viewer:
    """A live Factorio client attached to a run, and what actually happened."""

    requested: bool
    address: str
    warmup_seconds: float = DEFAULT_WARMUP_SECONDS
    process: subprocess.Popen | None = None
    #: Everything that went wrong, in order. A viewer failure must never end a
    #: run -- headless execution and the HTML replay are unaffected -- but it
    #: must not be silent either.
    warnings: list[str] = field(default_factory=list)
    joined: bool | None = None

    @classmethod
    def start(
        cls,
        *,
        requested: bool,
        executable: Path,
        address: str,
        mod_directory: Path,
        warmup_seconds: float = DEFAULT_WARMUP_SECONDS,
    ) -> Viewer:
        viewer = cls(requested=requested, address=address, warmup_seconds=warmup_seconds)
        if not requested:
            return viewer
        # `--mod-directory` is the whole trick and was found the hard way: a
        # client on the player's own mod set is greeted with "Your active mods
        # don't match the server's. Do you want to synchronize?", and
        # synchronising is wrong twice over -- `factoriorl` is a local mod that
        # is not on the mod portal, so a sync has nowhere to fetch it from, and a
        # sync rewrites the player's real active-mod list to match a throwaway
        # server. Pointing the client at the worker's own mod directory makes the
        # two sets identical by construction.
        #
        # `--config` is deliberately left alone, so the client keeps the player's
        # graphics settings. The worker gets an isolated config precisely so a
        # *measured* run cannot depend on those; a watched run is not one.
        command = [str(executable), "--mp-connect", address, "--mod-directory", str(mod_directory)]
        try:
            viewer.process = subprocess.Popen(command)
        except OSError as failure:
            viewer.warnings.append(
                f"could not start the client ({failure}); the run continues headless "
                f"and its replay is unaffected. Join manually at {address}"
            )
            viewer.joined = False
        return viewer

    def wait_for_join(self) -> None:
        """Give the client time to join, then check it is still there.

        A plain wait rather than an RCON poll. The poll this replaces printed its
        query into the *server console*, which the watcher reads across the
        middle of their own screen, and never returned a positive count even with
        a player plainly standing in the game. A tool for watching something
        should not scribble on the thing being watched.

        What is new is the check afterwards. A client that spawns and then dies
        -- a version mismatch, a refused join -- used to leave a run reporting
        itself as watched with nobody watching.
        """
        if self.process is None:
            return
        time.sleep(self.warmup_seconds)
        if self.process.poll() is None:
            self.joined = True
            return
        self.joined = False
        self.warnings.append(
            f"the client exited during warmup (code {self.process.returncode}); the run "
            f"continues headless. A version mismatch or a refused join looks like this"
        )

    @property
    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def stop(self, hold_open_seconds: float = 0.0) -> None:
        """Leave the final state on screen, then close the client."""
        if self.process is None:
            return
        if hold_open_seconds > 0 and self.alive:
            time.sleep(hold_open_seconds)
        if self.alive:
            self.process.terminate()

    def to_dict(self, reordered_replies: int | None = None) -> dict:
        described = {
            "requested": self.requested,
            # Observed, not assumed. `tools/watch_agent.py` records a literal
            # `True` here whether or not a client ever started.
            "launched": self.process is not None,
            "joined": self.joined,
            "alive_at_teardown": self.alive,
            "address": self.address,
            "warnings": list(self.warnings),
            "measurement": (
                "demonstration: a joined client creates a LuaPlayer no measured "
                "run has, and can run console commands and pause the world. "
                "Nothing measured here is comparable to a player-free run"
                if self.process is not None
                else "headless"
            ),
        }
        if reordered_replies is not None:
            # Reply inversions, which a connected client causes and a player-free
            # run does not: 89 across one watched run against 0 on every other.
            # Counted by the transport already; recorded here rather than printed
            # and forgotten.
            #
            # Omitted rather than set to null when the count is not yet known --
            # the manifest is written before the first decision, and a `null`
            # sitting under this name reads as "no inversions" rather than "not
            # measured yet". The run result carries the real figure.
            described["rcon_reordered_replies"] = reordered_replies
        return described
