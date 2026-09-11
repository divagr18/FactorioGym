"""Say what the agent is doing, while it is doing it.

A thirty-minute run printed nothing at all. Everything it did was recoverable
afterwards from `decisions.jsonl`, and that is the wrong time to find out it has
been hauling coal in a circle for twenty minutes: the two runs before this were
both diagnosed by reading a transcript after the fact, and one of them was
killed on a hunch that turned out to be right for the wrong reason.

Two audiences, one record:

**The console**, for working. One line per decision -- what it chose, why, and
what the world said back -- plus a marked line whenever the plan changes, which
is the single most informative event in a run and was previously invisible until
the transcript was read.

**The game window**, for watching. The reason and the current plan are drawn on
screen next to the character, which the spectator camera already follows. A
timelapse of a run is otherwise a character walking about with no indication of
what it believes it is doing.

The overlay is written with `rendering`, not with entities or chat, so it is
**invisible to the agent**: the sensor sweeps entities and resources, and a
render object is neither. Nothing here can be observed, collided with, or
mined, and no decision can be changed by whether anyone is watching. It is
issued between decisions rather than inside a step, and costs one RCON round
trip of a few milliseconds -- worth knowing for a realtime run, and the reason
it can be switched off.
"""

from __future__ import annotations

import shutil
from typing import Any

from factoriorl.rcon import LuaError, RCONError, lua_string

#: How much of a reason or plan to show. The console wraps to the terminal; the
#: overlay is bounded because a paragraph drawn over the factory hides the thing
#: it is describing.
OVERLAY_CHARS = 220

#: Drawn just above the character rather than on it, in tiles.
OVERLAY_OFFSET = -3.0

#: Set the on-screen text, creating the render object once and updating it
#: afterwards. Keyed in `storage` so a mod reload does not orphan it.
#:
#: `target` is the character entity, so the text follows the agent without this
#: having to re-issue a position every decision.
_OVERLAY_LUA = """
local text = {text}
local character = storage.frrl_character
if not (character and character.valid) then return "no character" end
local existing = storage.frrl_overlay
if existing and existing.valid then
  existing.text = text
  return "updated"
end
storage.frrl_overlay = rendering.draw_text({
  text = text,
  surface = character.surface,
  target = { entity = character, offset = { 0, {offset} } },
  color = { r = 1, g = 1, b = 1 },
  scale = 1.4,
  alignment = "center",
  scale_with_zoom = false,
  only_in_alt_mode = false,
})
return "created"
"""


def _clip(value: object, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


class Narrator:
    """Prints each decision, and optionally draws it in the game window."""

    def __init__(self, session: Any = None, *, console: bool = True, overlay: bool = True) -> None:
        self.session = session
        self.console = console
        self.overlay = overlay and session is not None
        self._plan: str | None = None
        #: Counted rather than raised on. The overlay is a convenience, and a
        #: run must not die because a cosmetic draw failed -- but silence would
        #: leave "the overlay never appeared" indistinguishable from "the agent
        #: never decided anything", so the count is reported at the end.
        self.overlay_errors = 0

    # ----------------------------------------------------------------- console

    def _width(self) -> int:
        return max(60, shutil.get_terminal_size((100, 24)).columns)

    def _say(self, line: str) -> None:
        if self.console:
            print(line[: self._width()], flush=True)

    # -------------------------------------------------------------- the record

    def decision(self, record: Any) -> None:
        """One finished decision: what it chose, why, and what happened."""
        step = getattr(record, "step", "?")
        reason = _clip(getattr(record, "reason", ""), 400)
        plan = _clip(getattr(record, "plan", ""), 400)

        # A new plan first and on its own line. It is the moment the agent
        # changes its mind, it happens a few dozen times in a long run, and it
        # is what a person scanning the console is actually looking for.
        if plan and plan != self._plan:
            self._plan = plan
            self._say(f"\n  PLAN  {plan}\n")

        resolution = getattr(record, "resolution", "")
        mark = " " if resolution == "model" else "!"
        self._say(f"{mark}{step:>4}  {reason}")

        for outcome in getattr(record, "outcomes", None) or []:
            status = outcome.get("status")
            detail = outcome.get("action_error") or outcome.get("error") or ""
            arguments = outcome.get("arguments") or {}
            shown = ", ".join(f"{k}={v}" for k, v in arguments.items())
            flag = "   " if status == "completed" else "  x"
            self._say(f"{flag}     {outcome.get('key')}({_clip(shown, 90)}) -> {status} {detail}")

        self._draw(reason=reason, plan=self._plan or "")

    def _draw(self, *, reason: str, plan: str) -> None:
        if not self.overlay:
            return
        body = _clip(reason, OVERLAY_CHARS)
        if plan:
            body += "\n" + _clip(plan, OVERLAY_CHARS)
        code = _OVERLAY_LUA.replace("{text}", lua_string(body)).replace(
            "{offset}", str(OVERLAY_OFFSET)
        )
        try:
            self.session._client.lua(code)  # noqa: SLF001 - cosmetic, see module docstring
        except (RCONError, LuaError, TimeoutError, OSError):
            # Never fatal. A watcher losing its subtitles is not a reason to end
            # a paid run, and the count says so at the end rather than a stack
            # trace saying so in the middle.
            self.overlay_errors += 1
            if self.overlay_errors >= 5:
                self.overlay = False
                self._say("  (overlay disabled after 5 failures; the run continues)")

    def to_dict(self) -> dict:
        return {
            "console": self.console,
            "overlay": self.overlay,
            "overlay_errors": self.overlay_errors,
        }
