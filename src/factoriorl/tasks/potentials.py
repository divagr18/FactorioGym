"""How far a scene is towards a working smelting line, as one number.

The definition factory-sim trains `construct_smelting_line` against
(`fsim_rl_potential`). It is **not** part of any task's reward here: the task's
reward stays the action-locked verification score, which the agentic bridge
and the frozen holdout depend on. The simulator uses it as a training-time
shaping signal, and this module is the reference that signal must reproduce --
factory-sim's `tools/sync_golden.py` evaluates it on every recorded decision of
the golden traces and its tests compare the C value bit for bit, which is why
every expression below is written in the order the C one is.

It reads only the published observation and a public marker, never evaluator
truth the policy cannot see.
"""

from __future__ import annotations

import math

#: Where a burner drill drops its output, relative to its centre, by facing
#: (`LuaEntity.direction`, 16-way). Measured on the engine
#: (`docs/evidence/sim-mechanics-m3.json`) in 1/256 tiles: N (-128, -332),
#: E (332, -128), S (128, 332), W (-332, 128).
DROP_OFFSETS = {
    0: (-128 / 256, -332 / 256),
    4: (332 / 256, -128 / 256),
    8: (128 / 256, 332 / 256),
    12: (-332 / 256, 128 / 256),
}

DRILL = "burner-mining-drill"
FURNACE = "stone-furnace"


def _holds(record: dict, key: str) -> bool:
    return sum((record.get(key) or {}).values()) > 0


def line_potential(observation: dict, truth: dict, marker: str | None) -> float:
    """phi(s) in [0, 0.9] for "a fuelled drill feeding a fuelled furnace".

    0.1 * approach   max(0, 1 - |character - marker| / 64)
    0.2 * drill      a burner drill is visible
    0.3 * line       a visible furnace's 2x2 footprint holds a drill's drop point
    0.1 * each of    the best such line's drill has fuel, its furnace has fuel,
                     its furnace holds ore or plates

    Built from `observation["entities"]`, the sweep the policy is shown, so a
    line the character has walked away from stops counting -- as it stops
    being visible.
    """
    phi = 0.0
    target = (truth.get("markers") or {}).get(marker) if marker else None
    position = (observation.get("character") or {}).get("position")
    if target is not None and position:
        dx = position[0] - target[0]
        dy = position[1] - target[1]
        approach = 1.0 - math.sqrt(dx * dx + dy * dy) / 64.0
        phi += 0.1 * (approach if approach > 0.0 else 0.0)
    entities = [e for e in (observation.get("entities") or []) if e.get("p")]
    drill = line = best = 0
    for i, d in enumerate(entities):
        if d.get("name") != DRILL:
            continue
        drill = 1
        ox, oy = DROP_OFFSETS.get(d.get("d", 0), DROP_OFFSETS[12])
        drop_x, drop_y = d["p"][0] + ox, d["p"][1] + oy
        for j, f in enumerate(entities):
            # Only a furnace makes a line: a burner drill's fuel slot takes no
            # ore, so one drill dropping into another is a jam, and counting it
            # paid a policy for placing its two drills side by side.
            if j == i or f.get("name") != FURNACE:
                continue
            fx, fy = f["p"]
            if not (fx - 1 <= drop_x < fx + 1 and fy - 1 <= drop_y < fy + 1):
                continue
            line = 1
            score = int(_holds(d, "fuel")) + int(_holds(f, "fuel"))
            score += int(_holds(f, "contents") or _holds(f, "output"))
            best = max(best, score)
    return phi + 0.2 * drill + 0.3 * line + 0.1 * best
