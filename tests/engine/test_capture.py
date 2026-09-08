"""Optional screenshot capture (PLAN.md 5.6), against a real engine."""

from __future__ import annotations

import pytest

from factoriorl.capture import CaptureUnsupported, capture, viewport_zoom

pytestmark = pytest.mark.engine


def test_capture_fails_clearly_on_a_build_that_cannot_render(module_worker):
    """The clause is 'unsupported capture configurations fail clearly rather
    than returning stale images', and on a headless build every configuration
    is unsupported in a way the engine does not admit to.

    Measured: `game.take_screenshot` is defined, accepts a well-formed request,
    returns success, and writes nothing. A caller trusting that return value
    would attach a missing file to a tick and call it evidence.
    """
    with pytest.raises(CaptureUnsupported) as raised:
        capture(
            module_worker.spec.rcon_endpoint,
            module_worker.spec.write_data,
            position=(0.0, 0.0),
            radius=32,
            timeout=2.0,
        )
    message = str(raised.value)
    # The failure has to say what was missing and why, or the next reader
    # repeats the investigation.
    assert "wrote no frame" in message
    assert "headless" in message
    assert "tick" in message


def test_the_viewport_is_sized_from_the_observation_profile():
    """A frame wider than the sensor shows a reader terrain the agent could not
    see; a narrower one hides what it acted on."""
    from factoriorl.encoders import LOCAL_V1

    zoom = viewport_zoom(LOCAL_V1.radius, (512, 512))
    tiles_shown = 512 / (zoom * 32.0)
    assert tiles_shown == pytest.approx(2 * LOCAL_V1.radius + 1, abs=1.0)
    # A tighter sensor must produce a tighter frame, not the same one.
    assert viewport_zoom(8, (512, 512)) > zoom
