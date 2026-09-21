"""What the base game actually puts in a new player's inventory.

the agent roadmap A1.1: "Obtain the base game's ordinary
starting-item definition from the installed game's scenario source and mirror it
for the controlled character; record the exact inventory and initialization
version."

The alternative was to copy the six items into a Python literal. That is exactly
the kind of assertion this repository keeps getting burned by: the list would be
right on the day it was written and silently wrong after any engine update, with
nothing to catch it. Reading the installed game's own source makes the claim
"these are freeplay's starting items" checkable rather than remembered, and the
file's digest goes into the run manifest so a later reader can tell whether the
world they are looking at was built from the same definition.

What is deliberately *not* mirrored
-----------------------------------
Freeplay also creates a crashed ship and its debris near the spawn, containing
more items, and plays an intro cutscene. The roadmap disables both: "Disable
introductory cutscene/wreck bonuses so they cannot supply an accidental
construction kit; declare this difference from interactive freeplay." A run that
began by looting a wreck would not be demonstrating what it appears to.

The starting items are mirrored **verbatim**, including the pistol and the
magazines, even though enemies are disabled and the action matrix has no combat
verb. Curating them would be a quiet difficulty change; carrying two dead items
is honest and costs nothing but two inventory slots.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

#: Relative to the engine's install root -- the directory two levels above
#: `bin/x64/factorio.exe`.
FREEPLAY_SOURCE = Path("data") / "base" / "script" / "freeplay" / "freeplay.lua"

#: `["iron-plate"] = 8`, the only shape the table uses.
_ENTRY = re.compile(r'\["([a-z0-9\-]+)"\]\s*=\s*(\d+)')

#: The function whose body is the starting inventory.
_BLOCK = re.compile(r"local\s+created_items\s*=\s*function\s*\(\)(.*?)\bend\b", re.S)


class FreeplayUnreadable(RuntimeError):
    """The installed game's starting items could not be read.

    Raised rather than defaulted. A0.3's rule for pricing applies here for the
    same reason: a fabricated starting inventory would make the run look like
    freeplay while not being freeplay, and no later measurement could detect it.
    """


def source_path(executable: Path) -> Path:
    """Where freeplay's source lives, relative to the engine binary."""
    # bin/x64/factorio.exe -> the install root is two directories up.
    return executable.resolve().parents[2] / FREEPLAY_SOURCE


def starting_inventory(executable: Path) -> dict:
    """Freeplay's `created_items`, read from the installed game.

    Returns the item counts plus the provenance needed to defend them: the
    source path, its sha256, and its size. Raises `FreeplayUnreadable` if the
    file is missing or its table cannot be parsed.
    """
    path = source_path(executable)
    if not path.is_file():
        raise FreeplayUnreadable(
            f"freeplay's source is not at {path}. The starting inventory is read "
            f"from the installed game rather than hardcoded, so a run cannot be "
            f"started without it. Check that the engine at {executable} is a full "
            f"install rather than a headless-only unpack"
        )
    text = path.read_text(encoding="utf-8", errors="replace")
    block = _BLOCK.search(text)
    if block is None:
        raise FreeplayUnreadable(
            f"{path} has no `local created_items = function()` block. The engine's "
            f"freeplay scenario has changed shape, so the starting inventory this "
            f"project mirrors can no longer be read from it. Fix the parser rather "
            f"than substituting a remembered list"
        )
    items = {name: int(count) for name, count in _ENTRY.findall(block.group(1))}
    if not items:
        raise FreeplayUnreadable(
            f"{path} defines `created_items` but no items were parsed from it. An "
            f"empty starting inventory is almost certainly a parser failure rather "
            f"than the game's intent, so it is refused"
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "items": items,
        "source": str(path),
        "sha256": digest,
        "bytes": path.stat().st_size,
        "mirrors": "base freeplay created_items, verbatim",
        "excludes": (
            "the crashed ship, its debris and the intro cutscene, so the run "
            "cannot begin by looting a wreck. This is a declared difference from "
            "interactive freeplay"
        ),
    }
