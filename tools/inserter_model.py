"""Reference model of a burner inserter taking items off a belt, and its checker.

The algorithm is specified in docs/sim-logistics.md ("Inserter belt pickup");
this file is the executable version of that section. `replay` runs one rig
recorded by tools/probe_inserter_chase.py (or one of the two moving-belt rigs
of tools/probe_logistics.py) through the model and compares it with the engine
on every tick:

- the hand's x offset from the inserter (the engine's `held_stack_position`,
  truncated toward zero to 1/256 tile; y is not compared, see below),
- whether the hand holds an item,
- which belt item was picked up, and on which tick,
- the burner energy spent that tick (to 0.05 J).

Belt items are read from the engine record every tick (belt motion itself is
specified and checked in docs/sim-logistics.md); everything the inserter does
is computed.

The model, in the inserter's frame (pickup tile straight ahead, "north"; world
frames are the four rotations of it):

1. Arm state: orientation `a` in turns, anticlockwise from the pickup
   direction (0.25 = the inserter's left), and length `L` in 1/256 tile. The
   hand is at (-L sin 2pi a, -L cos 2pi a) from the inserter's centre.
2. Each tick the arm moves toward a target (a_t, L_t):
   - extension: |L_t - L| < 0.001 tile: set L = L_t, free. <= EXT: set, costs
     the distance. Else move EXT; if less than EXT is left after that, set
     L = L_t (costing EXT).
   - rotation, the short way (an exact half turn: see `turn_toward`):
     |d| <= ROT: set, costs |d|. Else move ROT; if the extension reached its
     target this tick and less than ROT is left, set a = a_t (costing ROT).
   - arrived = both axes on target.
   - energy = 50 kJ x (turns rotated + tiles extended), as charged above. If
     the burner buffer holds less, the extension gets it first, the rotation
     what is left, each moving in proportion; an extension then less than EXT
     from its target is set to it, as after a full step.
3. Holding an item: target the drop point (0.5, 1.2 tiles); on arrival the item
   goes into the chest that tick.
4. Empty: the belt item being chased, if it is still on the pickup belt;
   otherwise a new one, chosen among the items on the pickup belt (after this
   tick's belt move): the lane nearer the inserter first (for a belt along the
   arm, lane 1, the left of travel), then the item furthest upstream. With no
   item, target the pickup point (0, 1 tile ahead). On arrival at an item it is
   picked up that tick.
5. If the chased item left the pickup belt this tick while the hand is over
   the pickup belt's tile, the inserter does nothing this tick (no move, no
   energy) and chooses again next tick; otherwise it chooses again at once.
6. Burner buffer (2560 J when full): refilled after every tick the inserter is
   awake. It falls asleep, keeping the buffer as its last move left it, when
   it ends a tick at the pickup point with nothing to chase and no item
   anywhere on its pickup belt's line. It wakes, and refills, when an item is
   added to that line; an item arriving on the pickup belt wakes it into a
   move paid from the stale buffer. Belts younger than ~300 ticks wake their
   inserters late, at a tick this model cannot predict (`transient`).
7. Two inserters on one belt tile: the one built later updates first in a
   tick, so an item it takes is already gone for the other; an item the
   earlier one takes is still seen by the later one that tick.

Not modelled: the hand's reported y includes a lift during swings (drawing
only; it never feeds back into the logic).

Run:
  uv run python tools/inserter_model.py                  # every family, summary
  uv run python tools/inserter_model.py --family stream  # one family, failures listed
  uv run python tools/inserter_model.py --family single --rig 17 -v
"""

from __future__ import annotations

import argparse
import json
import lzma
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs" / "evidence"

ROT = 0.013  # turns per tick
EXT = 0.035 * 256  # 1/256 tile per tick
DEADZONE = 0.001 * 256  # an extension this small is made without cost
REST = (0.0, 256.0)  # the pickup point: straight ahead, 1 tile
DROP = (0.5, 1.2 * 256)  # the drop point: straight behind, 1.2 tiles
ENERGY_PER_TURN = 50_000.0  # J per turn of rotation (burner inserter energy_per_rotation)
ENERGY_PER_TILE = 50_000.0  # J per tile of extension (energy_per_movement)
BUFFER = 2560.0  # J, a full burner inserter buffer
YOUNG_BELT_TICKS = 300  # belts younger than this wake their inserters late
LANE_OFFSET = 60  # 1/256 tile either side of the belt centre line
FACING_O = {"N": 0.0, "E": 0.25, "S": 0.5, "W": 0.75}
DIRS = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}  # local direction -> unit vector
FAMILIES = ("single", "stream", "wake", "loss", "pair", "m4")


@dataclass
class Params:
    #: "engine": each tick's buffer is the engine's (isolates the movement rules);
    #: "sleep": computed by rule 6.
    budget: str = "engine"
    #: with budget="sleep": "engine" takes the late wake-ups of young belts from
    #: the engine record, "none" assumes belts wake their inserters at once.
    transient: str = "engine"
    #: `pair` rigs: the other inserter updates after this one in a tick, so an
    #: item it takes this tick is still there when this one looks
    partner_after: bool = True
    partner_order: str = "built"  # "built": the later-built one first; "fixed": partner_after


def to_local(facing: str, x: float, y: float) -> tuple[float, float]:
    """World offset -> inserter frame (inverse of probe_inserter_chase.rotate)."""
    return {"N": (x, y), "E": (y, -x), "S": (-x, -y), "W": (-y, x)}[facing]


def to_world(facing: str, x: float, y: float) -> tuple[float, float]:
    return {"N": (x, y), "E": (-y, x), "S": (-x, -y), "W": (y, -x)}[facing]


def polar(x: float, y: float) -> tuple[float, float]:
    return (math.atan2(-x, -y) / (2 * math.pi)) % 1.0, math.hypot(x, y)


def cart(a: float, length: float) -> tuple[float, float]:
    return -length * math.sin(2 * math.pi * a), -length * math.cos(2 * math.pi * a)


def trunc(v: float) -> int:
    """The engine reports a hand offset truncated toward zero, in 1/256 tile."""
    r = round(v)
    return int(r) if abs(v - r) < 1e-6 else int(v)


def turn_toward(a: float, at: float, facing: str) -> float:
    """Signed local rotation (turns) from `a` to `at`, the short way round.

    The engine decides in world orientation (clockwise from north, in [0, 1)).
    An exact half turn goes against the sign of the difference there: +0.5
    turns anticlockwise, -0.5 clockwise. So a north-facing inserter swings
    from pickup to drop and back through the west both times, a south-facing
    one out through the east and back through the west, an east- or
    west-facing one back through the north.
    """
    f = FACING_O[facing]
    diff = (f - at) % 1.0 - (f - a) % 1.0
    if diff >= 0.5:
        diff -= 1.0
    elif diff <= -0.5:
        diff += 1.0
    return -diff  # local orientation runs the other way


@dataclass
class Arm:
    a: float
    length: float
    facing: str = "N"
    held: bool = False
    target: int | None = None  # unique id of the belt item being chased
    buffer: float = BUFFER
    asleep: bool = False

    def step(self, at: float, lt: float, budget: float = math.inf) -> tuple[bool, float]:
        """Move toward (at, lt) for one tick: (arrived, energy spent). Rule 2."""
        d = turn_toward(self.a, at, self.facing)
        dl = lt - self.length
        if abs(dl) < DEADZONE:
            ext, l_new, rl = 0.0, lt, True
        elif abs(dl) <= EXT:
            ext, l_new, rl = abs(dl), lt, True
        else:
            ext, l_new, rl = EXT, self.length + math.copysign(EXT, dl), False
            if abs(lt - l_new) < EXT:
                l_new, rl = lt, True
        if abs(d) <= ROT:
            rot, a_new, ra = abs(d), at % 1.0, True
        else:
            rot, a_new, ra = ROT, (self.a + math.copysign(ROT, d)) % 1.0, False
            if rl and abs(d) - ROT < ROT:
                a_new, ra = at % 1.0, True
        e_ext = ENERGY_PER_TILE * ext / 256
        e_rot = ENERGY_PER_TURN * rot
        if e_ext + e_rot > budget + 1e-9:
            # short of energy: the extension is paid first
            if e_ext >= budget:
                frac = budget / e_ext
                l_new, rl = self.length + math.copysign(ext * frac, dl), False
                # less than a step then left: on the target, as after a full
                # step (probe 2's seg_*_8, seg_b_7; probe 5)
                if abs(lt - l_new) < EXT:
                    l_new = lt
                a_new, ra, e_rot, e_ext = self.a, d == 0, 0.0, budget
            else:
                frac = (budget - e_ext) / e_rot
                a_new, ra = (self.a + math.copysign(rot * frac, d)) % 1.0, False
                e_rot = budget - e_ext
        self.a, self.length = a_new, l_new
        return ra and rl, e_ext + e_rot


@dataclass
class Item:
    uid: int
    lane: int
    pos: int  # 1/256 tile from the pickup belt's downstream edge
    x: float  # inserter frame, 1/256 tile
    y: float


def rig_geometry(rig: dict):
    """(belt offset from the pickup belt, lane, pos) -> inserter-frame (x, y)."""
    tiles = rig["layout"]["tiles"]
    pick = rig["layout"]["pick"] - 1
    ux, uy = DIRS[rig["layout"]["dir"]]
    lx, ly = uy, -ux  # left of travel

    def where(belt: int, lane: int, pos: int) -> tuple[float, float]:
        cx, cy = tiles[pick + belt]
        along = 128 - pos
        side = LANE_OFFSET if lane == 1 else -LANE_OFFSET
        return cx * 256 + ux * along + lx * side, cy * 256 + uy * along + ly * side

    return where


def choose(items: list[Item], rig: dict) -> Item:
    """Rule 4: nearer lane first (lane 1 on a tie), then furthest upstream."""
    across = rig["layout"]["dir"] in (1, 3)  # belt crosses in front of the inserter

    def lane_distance(i: Item) -> int:
        return round(abs(i.y) if across else abs(i.x))

    return min(items, key=lambda i: (lane_distance(i), i.lane, -i.pos))


def hand_over_pickup_tile(arm: Arm) -> bool:
    hx, hy = cart(arm.a, arm.length)
    return -128 <= hx <= 128 and -384 <= hy <= -128


@dataclass
class Result:
    ticks: int = 0
    hand_bad: int = 0
    held_bad: int = 0
    energy_bad: int = 0
    max_energy_err: float = 0.0
    picks_engine: list = field(default_factory=list)
    picks_model: list = field(default_factory=list)
    first_bad: int | None = None
    log: list = field(default_factory=list)

    @property
    def exact(self) -> bool:
        return (
            self.first_bad is None
            and self.energy_bad == 0
            and self.picks_engine == self.picks_model
        )


def add_schedule(rig: dict, horizon: int) -> list[int]:
    """Ticks at which the rig's script adds an item to the belt line."""
    out = []
    for inj in rig.get("inject", []):
        every = inj.get("every") or 0
        for j in range(inj.get("count", 1)):
            tt = inj["t"] + j * every
            if tt > horizon:
                break
            out.append(tt)
            if not every:
                break
    return out


def engine_wakes(records: list, ri: int) -> set[int]:
    """Ticks the engine refilled a sleeping inserter's buffer without a move."""
    out = set()
    for k in range(1, len(records)):
        r0, r1 = records[k - 1][1][ri], records[k][1][ri]
        if (
            len(r1) > 6
            and float(r0[6]) < BUFFER - 1e-6
            and abs(float(r1[6]) - BUFFER) < 1e-6
            and abs(float(r0[3]) - float(r1[3])) < 1e-6
            and (r1[0], r1[1]) == (r0[0], r0[1])
        ):
            out.add(records[k][0])
    return out


def partner_rig(doc: dict, ri: int) -> dict | None:
    pj = doc["rigs"][ri].get("partner")
    return None if pj is None else doc["rigs"][pj]


def partner_takes(doc: dict, ri: int) -> dict[int, int]:
    """For a `pair` rig: tick -> unique id of the item the other inserter took."""
    rig = doc["rigs"][ri]
    pj = rig.get("partner")
    if pj is None:
        return {}
    records = doc["ticks"]
    me = [float(v) * 256 for v in doc["setup"]["rigs"][ri]["pivot"]]
    other = [float(v) * 256 for v in doc["setup"]["rigs"][pj]["pivot"]]
    where = rig_geometry(rig)
    out = {}
    for k in range(1, len(records)):
        o0, o1 = records[k - 1][1][pj], records[k][1][pj]
        if not (o1[2] > 0 and o0[2] == 0):
            continue
        hx, hy = other[0] + o1[0], other[1] + o1[1]
        now = {it[2] for it in records[k][1][ri][5]}
        best = None
        for it in records[k - 1][1][ri][5]:
            if it[0] not in (-1, 0) or it[2] in now:
                continue
            lx, ly = where(0, it[1], it[3])
            wx, wy = to_world(rig["facing"], lx, ly)
            d = math.hypot(me[0] + wx - hx, me[1] + wy - hy)
            if best is None or d < best[0]:
                best = (d, it[2])
        if best is not None:
            out[records[k][0]] = best[1]
    return out


def replay(
    rig: dict,
    records: list,
    ri: int,
    p: Params,
    start: int = 1,
    verbose: bool = False,
    stop_after: int = 5,
    taken: dict | None = None,
    line_rig: dict | None = None,
) -> Result:
    """`taken`: tick -> id of an item another inserter took from this pickup belt;
    `line_rig`: the rig whose script feeds the belt line, if not this one."""
    facing = rig["facing"]
    where = rig_geometry(rig)
    res = Result()
    taken = taken or {}
    # which of two inserters on one belt tile updates first in a tick: the one
    # built later (see "Two inserters on one belt tile" in docs/sim-logistics.md)
    partner_after = (
        p.partner_after
        if p.partner_order == "fixed"
        else (rig.get("partner") is not None and rig["partner"] < ri)
    )
    adds = add_schedule(rig, records[-1][0])
    if line_rig is not None and line_rig is not rig:
        adds = sorted(adds + add_schedule(line_rig, records[-1][0]))
    add_set = set(adds)
    wakes = engine_wakes(records, ri) if p.transient == "engine" else set()

    def items_at(k: int) -> dict[int, list]:
        return {it[2]: it for it in records[k][1][ri][5]}

    row0 = records[start][1][ri]
    lx, ly = to_local(facing, row0[0], row0[1])
    a0, l0 = polar(lx, ly) if (lx or ly) else REST
    if (lx, ly) == (0, -179):
        l0 = 0.7 * 256  # as built: starting_distance 0.7
    arm = Arm(a0, l0, facing=facing, held=row0[2] > 0)
    prev_energy = float(row0[3])
    bad_streak = 0
    for k in range(start + 1, len(records)):
        t, rows = records[k]
        row = rows[ri]
        now, before = items_at(k), items_at(k - 1)
        before2 = items_at(k - 2) if k >= 2 else {}
        # Items on the pickup belt after this tick's belt move. One the engine
        # picked up this tick is gone from the record: move it as it moved the
        # tick before.
        cand = [
            Item(uid, it[1], it[3], *where(0, it[1], it[3]))
            for uid, it in now.items()
            if it[0] == 0
        ]
        for uid, it in before.items():
            if uid in now or it[0] not in (-1, 0) or (taken.get(t) == uid and not partner_after):
                continue
            v = 8
            if uid in before2 and before2[uid][0] == it[0]:
                v = before2[uid][3] - it[3]
            pos = it[3] - v + (256 if it[0] == -1 else 0)  # -1: entering the pickup belt
            if 0 <= pos < 256:
                cand.append(Item(uid, it[1], pos, *where(0, it[1], pos)))
        note = ""
        if rig.get("hold_at") is not None and t == rig["hold_at"]:
            arm.held, arm.target = True, None  # the probe put an item in the hand
            res.picks_model.append((t, None))
        if p.budget == "engine":
            budget = float(records[k - 1][1][ri][6])
        else:
            budget = arm.buffer
        energy = 0.0
        if arm.held:
            arrived, energy = arm.step(*DROP, budget)
            if arrived:
                arm.held, note = False, "drop"
        else:
            tgt = next((i for i in cand if i.uid == arm.target), None)
            if arm.target is not None and tgt is None and hand_over_pickup_tile(arm):
                arm.target, note = None, "lost"  # rule 5: this tick is spent
            else:
                if tgt is None and cand:
                    tgt = choose(cand, rig)
                if tgt is not None:
                    arm.target = tgt.uid
                    arrived, energy = arm.step(*polar(tgt.x, tgt.y), budget)
                    if arrived:
                        arm.held, arm.target, note = True, None, f"pick {tgt.uid}"
                        res.picks_model.append((t, tgt.uid))
                else:
                    arm.target = None
                    arrived, energy = arm.step(*REST, budget)
        # rule 6: the buffer
        if p.budget == "sleep":
            on_line = sum(1 for tt in adds if tt <= t) > sum(
                1 for _, u in res.picks_model if u is not None
            ) + sum(1 for tt in taken if tt <= t)
            young = p.transient == "engine" and t < YOUNG_BELT_TICKS
            woke = (t in wakes) if young else (t in add_set)
            at_rest = (
                not arm.held and arm.target is None and not cand and (arm.a, arm.length) == REST
            )
            if arm.asleep and woke and energy == 0:
                arm.asleep = False
            if cand or arm.held or arm.target is not None:
                arm.asleep = False
            if not arm.asleep and at_rest and not on_line:
                arm.asleep, arm.buffer = True, budget - energy
            elif not arm.asleep:
                arm.buffer = BUFFER
        # compare with the engine
        gone = [
            uid
            for uid, it in before.items()
            if uid not in now and it[0] in (-1, 0) and taken.get(t) != uid
        ]
        e_held = row[2] > 0
        if e_held and not records[k - 1][1][ri][2] > 0:
            res.picks_engine.append((t, gone[0] if gone else None))
        mx, my = to_world(facing, *cart(arm.a, arm.length))
        ok_hand = row[0] in (trunc(mx - 1e-3), trunc(mx + 1e-3))
        e_energy = prev_energy - float(row[3])
        prev_energy = float(row[3])
        refuel = abs(e_energy) > 50_000  # a new fuel item was loaded
        ok_energy = refuel or abs(e_energy - energy) < 0.05
        if not refuel:
            res.max_energy_err = max(res.max_energy_err, abs(e_energy - energy))
        ok = ok_hand and e_held == arm.held
        res.ticks += 1
        res.hand_bad += not ok_hand
        res.held_bad += e_held != arm.held
        res.energy_bad += not ok_energy
        if not ok and res.first_bad is None:
            res.first_bad = t
        line = (
            f"t={t} a={arm.a:.4f} L={arm.length:.2f} m=({mx:.2f},{my:.2f}) "
            f"e=({row[0]},{row[1]}) held m={arm.held} e={e_held} "
            f"E m={energy:.2f} e={e_energy:.2f} {note} "
            f"{'' if ok else '<<<'}{'' if ok_energy else ' E<<'}"
        )
        if verbose:
            print(line)
        if not ok:
            bad_streak += 1
            res.log.append(line)
            if bad_streak >= stop_after:
                break
    return res


def load_m4() -> dict:
    """The two moving-belt rigs of tools/probe_logistics.py, in this file's format.

    `flow` (near lane) and `flow2` (far lane): a 16-belt east line fed every 20
    ticks, a north-facing inserter below belt 9. Only belts 6-11 were recorded;
    the pickup belt is the fourth of them.
    """
    sys.path.insert(0, str(ROOT / "tools"))
    from probe_logistics import replay as undelta

    doc = json.loads((EVIDENCE / "sim-mechanics-m4-logistics.json").read_text(encoding="utf-8"))
    samples = undelta(doc["samples"])
    specs = [("flow", 2), ("flow2", 1)]
    rigs = [
        {
            "family": "m4",
            "facing": "N",
            "belt": "east",
            "which": key,
            "phase": 0,
            "lane": lane,
            "inject": [{"t": 0, "every": 20, "count": 200}],
            "layout": {"tiles": [[x, -1] for x in range(-8, 8)], "dir": 1, "pick": 9},
        }
        for key, lane in specs
    ]
    ticks = []
    for s in samples:
        rows = []
        for key, _lane in specs:
            ins = s[key]
            at = doc["setup"]["inserters"][key + "_ins"]["at"]
            px, py = float(at[0]) * 256, float(at[1]) * 256
            items = []
            for bi, belt in enumerate(s[key + "_belt"]):
                if bi - 3 not in (-1, 0, 1):
                    continue
                lanes = belt if isinstance(belt, list) else [belt.get("1", []), belt.get("2", [])]
                for ln, content in enumerate(lanes, start=1):
                    for _name, pos, uid in content:
                        items.append([bi - 3, ln, uid, pos])
            total = float(ins["remaining"]) + float(ins["energy"])
            rows.append(
                [
                    ins["hand"][0] - px,
                    ins["hand"][1] - py,
                    1 if ins.get("held") else 0,
                    repr(total),
                    0,
                    items,
                    ins["energy"],
                ]
            )
        ticks.append([s["t"], rows])
    return {"rigs": rigs, "ticks": ticks}


def load(family: str) -> dict:
    if family == "m4":
        return load_m4()
    with lzma.open(EVIDENCE / f"inserter-chase-{family}.json.xz", "rt", encoding="utf-8") as fh:
        return json.load(fh)


def summarize(family: str, p: Params) -> dict:
    doc = load(family)
    results = [
        replay(
            rig,
            doc["ticks"],
            ri,
            p,
            stop_after=10**9,
            taken=partner_takes(doc, ri),
            line_rig=partner_rig(doc, ri),
        )
        for ri, rig in enumerate(doc["rigs"])
    ]
    picks = sum(len(r.picks_engine) for r in results)
    same = sum(len(set(r.picks_engine) & set(r.picks_model)) for r in results)
    return {
        "rigs": len(results),
        "exact": sum(r.exact for r in results),
        "rig_ticks": sum(r.ticks for r in results),
        "picks": picks,
        "picks_same": same,
        "max_energy_err": max(r.max_energy_err for r in results),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default=None, choices=FAMILIES)
    ap.add_argument("--rig", type=int, default=None)
    ap.add_argument("-v", action="store_true")
    ap.add_argument("--budget", default="engine", choices=["engine", "sleep"])
    ap.add_argument("--transient", default="engine", choices=["engine", "none"])
    ap.add_argument("--partner-first", action="store_true")
    args = ap.parse_args()
    p = Params(budget=args.budget, transient=args.transient, partner_after=not args.partner_first)
    if args.family is None:
        modes = [
            ("movement (engine buffer)", Params()),
            ("self-contained, young-belt wakes from the engine", Params("sleep", "engine")),
            ("self-contained, young-belt delay ignored", Params("sleep", "none")),
        ]
        for name, mp in modes:
            print(name)
            for fam in FAMILIES:
                s = summarize(fam, mp)
                print(
                    f"  {fam:>6}: {s['exact']}/{s['rigs']} rigs exact, "
                    f"{s['picks_same']}/{s['picks']} pickups on the engine's tick, "
                    f"{s['rig_ticks']} rig-ticks, max energy error "
                    f"{s['max_energy_err']:.3f} J"
                )
        return 0
    doc = load(args.family)
    exact = 0
    for ri, rig in enumerate(doc["rigs"]):
        if args.rig is not None and ri != args.rig:
            continue
        res = replay(
            rig,
            doc["ticks"],
            ri,
            p,
            verbose=args.v,
            taken=partner_takes(doc, ri),
            line_rig=partner_rig(doc, ri),
        )
        exact += res.exact
        if not res.exact:
            desc = {k: rig[k] for k in ("facing", "belt", "which", "phase", "case") if k in rig}
            print(
                ri,
                desc,
                "first_bad",
                res.first_bad,
                "picks e",
                res.picks_engine[:4],
                "m",
                res.picks_model[:4],
                "energy_bad",
                res.energy_bad,
            )
            if args.rig is not None:
                for line in res.log[:8]:
                    print("   ", line)
    total = 1 if args.rig is not None else len(doc["rigs"])
    print(f"exact {exact}/{total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
