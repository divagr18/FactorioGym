"""Is placement geometry rotation- and reflection-invariant? (synthesis section 8)

Section 8 proposes scoring placement candidates by their resulting local
structure, on the hypothesis that a shared scorer generalises better than
unrelated logits per template. Its gate leads with the part that has to come
first: *"Treat symmetry as a tested property: not every Factorio mechanic is
rotation/reflection invariant"*, and *"Defer a new value-learning algorithm
until the simpler representation test is informative."*

So this measures the symmetry rather than assuming it. If a rotated layout is
not equivalent, a rotation-equivariant scorer would be actively wrong, and
knowing that is worth more than the scorer.

The specific reason to doubt it, before measuring: a 2x2 entity snaps to an
*integer* centre and occupies tiles `{cx-1, cx} x {cy-1, cy}`. That footprint
is not symmetric about its own centre -- it extends one tile in the negative
direction and none in the positive -- so the tile lattice a placement lands on
has a parity the entity's centre does not share. A 90-degree rotation about the
drill's centre need not map valid furnace centres to valid furnace centres.

What is held fixed, per the gate: the candidate set (every integer centre within
4 tiles of the drill), the information (each candidate judged only by
`can_place_entity` plus measured production), and the budget (7200 ticks after
fuelling, identical for every arm).

Run: uv run python tools/symmetry_probe.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from factoriorl.paths import evidence_dir  # noqa: E402
from factoriorl.rcon import RCONClient  # noqa: E402
from factoriorl.worker import WorkerManager  # noqa: E402

#: Ticks of production measured per candidate, after both machines are fuelled.
#: One measurement window plus its warm-up, so a productive pair is clearly
#: separated from an unproductive one (measured: 14 plates in the first 3600).
MEASURE_TICKS = 3600

#: Candidate furnace centres, as integer offsets from the drill's centre.
RADIUS = 4

FACINGS = ("north", "east", "south", "west")

#: Drill centres the south-facing sweep is repeated at. Odd and even offsets
#: both: a 2x2 footprint occupies `{cx-1, cx}`, so it has a parity, and an
#: even-only test could not see a lattice effect if there were one.
TRANSLATIONS = ((5, 0), (0, 5), (3, -7), (-6, -6))

SETUP = """
local s = game.surfaces[1]
s.always_day = true
for _, e in pairs(s.find_entities_filtered({force="player"})) do e.destroy() end
for _, e in pairs(s.find_entities_filtered({type="resource"})) do e.destroy() end
for dx = -12, 12 do for dy = -12, 12 do
  s.create_entity({name="iron-ore", position={dx, dy}, amount=100000})
end end
game.tick_paused = false
return "ready"
"""

#: One candidate: build drill + furnace, fuel both, and report production after
#: a fixed number of ticks. `can_place_entity` is asked first, so "refused" and
#: "placed but unproductive" stay distinct outcomes.
TRIAL = """
local s = game.surfaces[1]
for _, e in pairs(s.find_entities_filtered({force="player"})) do e.destroy() end
game.forces.player.get_item_production_statistics(s).clear()
local ox, oy = %d, %d
local drill = s.create_entity({name="burner-mining-drill", position={ox, oy},
  direction=defines.direction.%s, force="player"})
if not drill then return "drill_failed" end
local can = s.can_place_entity({name="stone-furnace", position={ox + %d, oy + %d},
  force="player", build_check_type=defines.build_check_type.manual})
if not can then
  drill.destroy()
  return "refused"
end
local f = s.create_entity({name="stone-furnace", position={ox + %d, oy + %d}, force="player"})
if not f then drill.destroy() return "create_failed" end
drill.insert({name="coal", count=50})
f.insert({name="coal", count=50})
storage.probe_start = game.tick
return "placed|" .. game.tick
"""

READ = """
local s = game.surfaces[1]
local stats = game.forces.player.get_item_production_statistics(s)
return (game.tick - (storage.probe_start or game.tick)) .. "|"
  .. stats.get_input_count("iron-plate")
"""


def rotate(offset: tuple[int, int], quarters: int) -> tuple[int, int]:
    """Rotate a tile offset a quarter turn clockwise `quarters` times."""
    x, y = offset
    for _ in range(quarters % 4):
        x, y = -y, x
    return (x, y)


def reflect(offset: tuple[int, int], axis: str) -> tuple[int, int]:
    x, y = offset
    return (-x, y) if axis == "x" else (x, -y)


def run_facing(
    rcon: RCONClient, facing: str, origin: tuple[int, int] = (0, 0)
) -> dict[tuple[int, int], dict]:
    """Which furnace offsets are placeable, and which of those produce.

    `origin` is the drill's own centre, so the same sweep can be repeated on a
    translated layout -- the other half of the gate's "translated/rotated".
    """
    results: dict[tuple[int, int], dict] = {}
    for dx in range(-RADIUS, RADIUS + 1):
        for dy in range(-RADIUS, RADIUS + 1):
            outcome = str(rcon.lua(TRIAL % (origin[0], origin[1], facing, dx, dy, dx, dy)))
            if not outcome.startswith("placed"):
                results[(dx, dy)] = {"placeable": False, "plates": 0}
                continue
            # Wait out the measurement window in game time, not wall time.
            while True:
                elapsed, plates = (int(v) for v in str(rcon.lua(READ)).split("|"))
                if elapsed >= MEASURE_TICKS:
                    break
                time.sleep(0.2)
            results[(dx, dy)] = {"placeable": True, "plates": plates}
    return results


def productive(results: dict[tuple[int, int], dict]) -> set[tuple[int, int]]:
    return {offset for offset, row in results.items() if row["plates"] > 0}


def main() -> int:
    report: dict = {
        "experiment": "synthesis section 8, symmetry as a tested property",
        "held_fixed": {
            "candidate_set": f"every integer offset within {RADIUS} tiles of the drill centre",
            "information": "can_place_entity plus measured production; no evaluator truth",
            "budget_ticks": MEASURE_TICKS,
        },
        "facings": {},
        "translations": {},
        "checks": [],
    }

    manager = WorkerManager()
    handle = manager.launch("symmetry-probe")
    try:
        with RCONClient(handle.spec.rcon_endpoint, timeout=120.0) as rcon:
            rcon.lua("game.speed = 100 return 1")
            print(rcon.lua(SETUP), flush=True)
            per_facing = {}
            for facing in FACINGS:
                print(f"measuring {facing}...", flush=True)
                results = run_facing(rcon, facing)
                per_facing[facing] = results
                report["facings"][facing] = {
                    "placeable": sorted(o for o, r in results.items() if r["placeable"]),
                    "productive": sorted(productive(results)),
                    "plates": {f"{o[0]},{o[1]}": r["plates"] for o, r in results.items()},
                }
                print(
                    f"  {facing}: {len(report['facings'][facing]['placeable'])} placeable, "
                    f"{len(report['facings'][facing]['productive'])} productive "
                    f"{report['facings'][facing]['productive']}",
                    flush=True,
                )
            # Translation: the same sweep with the drill moved. Odd and even
            # offsets both, because a 2x2 footprint has a parity and an
            # even-only test could not see a lattice effect if there were one.
            per_origin = {}
            for origin in TRANSLATIONS:
                print(f"measuring south at origin {origin}...", flush=True)
                per_origin[origin] = run_facing(rcon, "south", origin)
                found = sorted(productive(per_origin[origin]))
                report["translations"][f"{origin[0]},{origin[1]}"] = {"productive": found}
                print(f"  {origin}: {found}", flush=True)
    finally:
        manager.cleanup(handle)

    def check(label: str, ok: bool, **detail) -> None:
        report["checks"].append({"check": label, "holds": bool(ok), **detail})
        print(f"  [{'yes' if ok else 'NO '}] {label}  {detail}", flush=True)

    print("\nsymmetry:", flush=True)
    base = productive(per_facing["south"])
    # Rotation: is the productive set for each facing the rotation of south's?
    for quarters, facing in ((1, "west"), (2, "north"), (3, "east")):
        expected = {rotate(o, quarters) for o in base}
        actual = productive(per_facing[facing])
        check(
            f"rotating south's productive set {quarters * 90} degrees gives {facing}'s",
            expected == actual,
            expected=sorted(expected),
            actual=sorted(actual),
            missing=sorted(expected - actual),
            extra=sorted(actual - expected),
        )
    # Reflection: south and north differ by a flip in y, if the mechanic is
    # reflection invariant.
    check(
        "reflecting south's productive set in y gives north's",
        {reflect(o, "y") for o in base} == productive(per_facing["north"]),
        reflected=sorted(reflect(o, "y") for o in base),
        actual=sorted(productive(per_facing["north"])),
    )
    check(
        "every facing admits the same number of productive placements",
        len({len(productive(r)) for r in per_facing.values()}) == 1,
        counts={f: len(productive(r)) for f, r in per_facing.items()},
    )
    for origin, results in per_origin.items():
        check(
            f"translating the drill to {origin} leaves the productive offsets unchanged",
            productive(results) == base,
            expected=sorted(base),
            actual=sorted(productive(results)),
        )

    report["symmetric"] = all(c["holds"] for c in report["checks"])
    path = evidence_dir() / "section8-symmetry.json"
    path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(
        f"\n{'SYMMETRIC' if report['symmetric'] else 'NOT SYMMETRIC'}: wrote {path}",
        flush=True,
    )
    # Deliberately exit 0 either way: this is an experiment, and "the mechanic
    # is not rotation invariant" is a *result*, not a failure.
    return 0


if __name__ == "__main__":
    sys.exit(main())
