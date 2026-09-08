"""Optional screenshot capture (PLAN.md 5.6).

    Capture selected frames for agent requests and replay checkpoints.

    Frames are associated with a verified game tick. The viewport respects the
    observation profile. Screenshot capture is disabled in ordinary RL
    training. Unsupported capture configurations fail clearly rather than
    returning stale images.

The last clause is the whole difficulty on this build, and it is not
hypothetical. A headless Factorio has no renderer, but ``game.take_screenshot``
is still defined, still accepts a well-formed request, and still returns
success:

    take_screenshot exists: true
    take_screenshot call:   true|nil
    script-output:          created, empty

So the engine's answer to "did you capture a frame" is yes, and no frame exists.
A caller that trusted the return value would attach a missing or, worse, a
previously written file to a decision and call it evidence of that tick.

Capture is therefore verified from outside the engine: the frame is requested
with a path nothing else writes, the tick is read back from the engine rather
than assumed, and the file must appear -- with a fresh mtime and a non-zero size
-- before the capture is reported as having happened. Anything else raises
:class:`CaptureUnsupported` naming what was missing.

Not part of the protocol
------------------------
Capture is deliberately not an action in ``matrix.lua``. An action would be
reachable from a policy's catalog, and PLAN 5.6 requires capture to be off in
ordinary RL training; keeping it a separate client-side facility means a
training run cannot request a frame even by accident, and the action-mask purity
tests keep meaning what they say.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from factoriorl.errors import FactorioRLError
from factoriorl.rcon import RCONClient

#: How long to wait for a frame to appear before declaring capture unsupported.
#: The engine writes asynchronously, so zero would be a race; a headless build
#: never writes at all, so this is the cost of finding that out.
APPEAR_TIMEOUT_SECONDS = 3.0
POLL_SECONDS = 0.1


class CaptureUnsupported(FactorioRLError):
    """The engine accepted the request and produced no frame."""


@dataclass(frozen=True)
class Frame:
    """One captured frame and the tick it belongs to."""

    path: Path
    #: Read back from the engine after the request, not predicted before it.
    tick: int
    resolution: tuple[int, int]
    zoom: float
    #: Radius, in tiles, the viewport was sized to -- the observation profile's,
    #: so a frame shows what the agent could see and not more.
    radius: int

    def to_dict(self) -> dict:
        return {
            "path": str(self.path),
            "tick": self.tick,
            "resolution": list(self.resolution),
            "zoom": self.zoom,
            "sensor_radius": self.radius,
        }


def viewport_zoom(radius: int, resolution: tuple[int, int]) -> float:
    """Zoom that makes the frame cover exactly the sensor's square.

    PLAN 5.6 asks that the viewport respect the observation profile. A frame
    wider than the sensor would show a reader terrain the agent could not see
    and invite conclusions the agent had no basis for; a narrower one would hide
    what it acted on.
    """
    tiles = max(2 * radius + 1, 1)
    return round(min(resolution) / (tiles * 32.0), 4)


def capture(
    endpoint,
    write_data: Path,
    *,
    position: tuple[float, float],
    radius: int,
    resolution: tuple[int, int] = (512, 512),
    name: str | None = None,
    timeout: float = APPEAR_TIMEOUT_SECONDS,
) -> Frame:
    """Request one frame and return it only if it actually materialised."""
    zoom = viewport_zoom(radius, resolution)
    stamp = name or f"frrl-{int(time.time() * 1000)}"
    relative = f"{stamp}.png"
    destination = Path(write_data) / "script-output" / relative

    # A stale file under this name would be indistinguishable from a fresh
    # capture, and the point of the exercise is not to attach one to a tick.
    if destination.exists():
        destination.unlink()

    with RCONClient(endpoint, timeout=30.0) as client:
        tick = int(
            client.lua(
                "game.take_screenshot{"
                f"position={{{position[0]}, {position[1]}}}, "
                f"resolution={{{resolution[0]}, {resolution[1]}}}, "
                f"zoom={zoom}, path='{relative}', show_gui=false, show_entity_info=false"
                "} "
                # The tick the engine was on when it took the request, read back
                # rather than assumed: associating a frame with a tick the client
                # guessed is the same defect as trusting the return value.
                "return game.tick"
            )
        )

    deadline = time.time() + timeout
    while time.time() < deadline:
        if destination.exists() and destination.stat().st_size > 0:
            return Frame(
                path=destination,
                tick=tick,
                resolution=resolution,
                zoom=zoom,
                radius=radius,
            )
        time.sleep(POLL_SECONDS)

    raise CaptureUnsupported(
        f"the engine accepted a screenshot request at tick {tick} and wrote no frame to "
        f"{destination}. A headless Factorio has no renderer: `game.take_screenshot` is "
        "defined, accepts the call and returns success without producing an image, so a "
        "caller that trusted the return value would attach a missing or stale file to "
        "this tick. Capture needs a build with graphics."
    )
