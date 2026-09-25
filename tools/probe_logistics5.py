"""Measure belt-line segments as belts change, on loops, at attachments (M4, logistics 5).

The fourth logistics probe (tools/probe_logistics4.py) left factory-sim exact
on every rig measured, with a few rules chosen rather than measured. This probe
measures them. docs/sim-logistics.md, "Fifth probe", states the findings.

A rig is data: a base tile and a list of timed operations -- build a belt, a
wooden chest, a burner inserter or a burner mining drill (on iron ore laid
under it), put an item on a belt lane
(`LuaTransportLine.insert_at`), rotate an entity (`LuaEntity.rotate`) or
destroy it -- run by one Lua interpreter here and by factory-sim's
tests/logistics_rigs5.py there. Operations at t=0 run in the build command,
later ones in the on_tick hook, before that tick's reading. Every tick it
changes, each rig's reading is logged:

- `b`: per belt (build order), "gone" or its two lanes, each a list of
  [position (1/256), unique_id, item name];
- `s`: per belt and lane, the index (belt * 2 + lane, lane 0 or 1) of the
  first belt lane of the rig whose line is the same object
  (`LuaTransportLine.line_equals`), compared across both lanes; -1 for a belt
  that is gone;
- `i`: per inserter, the item in its hand or "";
- `c`: per chest, its non-empty slots as [slot, name, count].

Families (`--family`), one evidence file each
(`docs/evidence/logistics5-<family>.json.xz`):

- `order`: where the pieces of a segment go in the activation order when it
  splits at a boundary, loses a belt or has one turned. Two long feeds
  sideload onto one old, empty main lane; an item on each reaches it on the
  same tick, and the one whose segment moves first goes 8/256 further (the
  third probe's catch-up). The change happens while one item rides the piece.
  The timings (ORDER_TIMES) were found with factory-sim.
- `change`: turning and removing belts of young and of merged lines: segment
  membership and merge and split timers.
- `feedchg`: belts of a feed near its sideload, and of the main around the
  target, removed and rebuilt or turned.
- `loop`: closed loops, young and old, sparse and compressed, fed from a
  sideload, with an inserter on them.
- `loop2`: loops built from different belts, either way round; a belt of a
  merged loop removed and rebuilt, or turned in place.
- `bound`: the boundaries attachments mark, in other geometries: inserters
  picking and dropping from either side, at the ends of a line, on and next to
  turns, several on one line, some never working; sideloads from either side,
  at the ends, before a turn, onto a west line.
- `dist`: a drop, a pickup or a feed at one belt followed by patterns of
  straights and turns: where the boundary lies along the lane.
- `drill`: the same for a burner mining drill's output, over patterns whose
  lane sums pass through every value from 468 to 768 a lane can reach.
- `loop3`: inserters taking from and dropping onto 2 x 2 loops, whose six
  inner lanes (636) lie inside the inserter's bracket: where the search for
  the boundary meets the loop's front.
- `trig`, `trig2`, `trig3`: what sets a boundary off, which boundaries a
  split uses, the delay a second split counts, boundaries close together.

Run (on the laptop, 25 to 35 s a world of up to 30 rigs):
  uv run python tools/probe_logistics5.py --family order
  (and change, feedchg, loop, loop2, loop3, bound, dist, drill, trig, trig2,
  trig3)
"""

from __future__ import annotations

import argparse
import json
import lzma
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EVIDENCE = ROOT / "docs" / "evidence"

N, E, S, W = 0, 4, 8, 12

# ---------------------------------------------------------------- rigs as data


class Rig:
    """A rig: `base` tile, `ticks` to record, timed operations."""

    def __init__(self, name: str, base: tuple[int, int], ticks: int) -> None:
        self.name, self.base, self.ticks = name, base, ticks
        self.ops: list[list] = []

    def op(self, t: int, *args) -> Rig:
        self.ops.append([t, *args])
        return self

    def belt(self, label: str, dx: int, dy: int, d: int, t: int = 0) -> Rig:
        return self.op(t, "belt", label, dx, dy, d)

    def chest(self, label: str, dx: int, dy: int, item: str | None = None, count: int = 0,
              t: int = 0) -> Rig:  # fmt: skip
        return self.op(t, "chest", label, dx, dy, item, count)

    def ins(self, label: str, dx: int, dy: int, d: int, coal: int = 2, t: int = 0) -> Rig:
        return self.op(t, "ins", label, dx, dy, d, coal)

    def drill(self, label: str, dx: int, dy: int, d: int, coal: int = 5, t: int = 0) -> Rig:
        """A burner mining drill at tile corner (dx, dy), on four tiles of iron
        ore (5,000 each) that replace whatever resource was there."""
        return self.op(t, "drill", label, dx, dy, d, coal)

    def put(self, t: int, label: str, lane: int, pos: int, item: str) -> Rig:
        return self.op(t, "put", label, lane, pos, item)

    def rotate(self, t: int, label: str, reverse: bool = False) -> Rig:
        return self.op(t, "rotate", label, 1 if reverse else 0)

    def destroy(self, t: int, label: str) -> Rig:
        return self.op(t, "destroy", label)

    def to_dict(self) -> dict:
        ops = sorted(self.ops, key=lambda o: o[0])  # stable: same-tick ops keep their order
        return {"name": self.name, "base": list(self.base), "ticks": self.ticks, "ops": ops}


def line(r: Rig, n: int, d: int = E, x0: int = 0, y0: int = 0, prefix: str = "m",
         t: int = 0) -> Rig:  # fmt: skip
    """`n` belts facing `d`, labelled prefix0.. along the flow from (x0, y0)."""
    ux, uy = {N: (0, -1), E: (1, 0), S: (0, 1), W: (-1, 0)}[d]
    for k in range(n):
        r.belt(f"{prefix}{k}", x0 + ux * k, y0 + uy * k, d, t)
    return r


def cells(start: tuple[int, int] = (104, 104), step: int = 24, per_row: int = 9):
    """Rig bases, `step` tiles apart, in the area the delay table covers."""
    k = 0
    while True:
        yield (start[0] + (k % per_row) * step, start[1] + (k // per_row) * step)
        k += 1


# ------------------------------------------------------------- family: bound

#: Attachments on an old line of 12 east belts (m0..m11 at y=0, x=0..11),
#: built at t=610 when every lane has merged. Pickups take plates put on m0
#: at t=620; drops come from a chest of two plates; feeds carry plates put on
#: their tail at t=620.
BOUND_AT = 610


def _pick(r: Rig, x: int, side: str, t: int = BOUND_AT) -> None:
    """An inserter at (x, -1) or (x, 1) taking from (x, 0) into a chest."""
    if side == "s":
        r.chest(f"pc{x}", x, 2, t=t).ins(f"pi{x}", x, 1, N, t=t)
    else:
        r.chest(f"pc{x}", x, -2, t=t).ins(f"pi{x}", x, -1, S, t=t)


def _drop(r: Rig, x: int, side: str, count: int = 2, t: int = BOUND_AT) -> None:
    """An inserter at (x, 1) or (x, -1) dropping onto (x, 0) from a chest.
    A burner inserter drops on the far lane: from the south, lane 1."""
    item = "iron-plate" if count else None
    if side == "s":
        r.chest(f"dc{x}", x, 2, item, count, t=t).ins(f"di{x}", x, 1, S, t=t)
    else:
        r.chest(f"dc{x}", x, -2, item, count, t=t).ins(f"di{x}", x, -1, N, t=t)


def _side(r: Rig, x: int, side: str, n: int = 2, t: int = BOUND_AT, put_at: int = 620) -> None:
    """A feed of `n` belts into (x, 0) from the south (onto lane 2 of an east
    line) or the north (lane 1), plates on both lanes of its tail."""
    if side == "s":
        for j in range(n, 0, -1):
            r.belt(f"f{side}{x}_{j}", x, j, N, t)
        tail = f"f{side}{x}_{n}"
    else:
        for j in range(n, 0, -1):
            r.belt(f"f{side}{x}_{j}", x, -j, S, t)
        tail = f"f{side}{x}_{n}"
    r.put(put_at, tail, 1, 128, "copper-plate").put(put_at, tail, 2, 128, "copper-plate")


def _feed_m0(r: Rig, t: int = 620) -> None:
    for lane in (1, 2):
        r.put(t, "m0", lane, 128, "iron-plate")
        r.put(t, "m0", lane, 0, "copper-plate")


def bound_rigs() -> list[Rig]:
    at = cells()
    rigs: list[Rig] = []

    def new(name: str, n: int = 12, ticks: int = 1500) -> Rig:
        r = line(Rig(name, next(at), ticks), n)
        rigs.append(r)
        return r

    for side in ("s", "n"):
        for x in (0, 4, 9, 10, 11):
            r = new(f"pick_{side}_k{x}")
            _pick(r, x, side)
            _feed_m0(r)
        for x in (0, 4, 8, 9, 10, 11):
            _drop(new(f"drop_{side}_k{x}"), x, side)
        for x in (0, 4, 9, 10, 11):
            _side(new(f"side_{side}_k{x}"), x, side)
    r = new("side_both_k4")
    _side(r, 4, "s")
    _side(r, 4, "n")
    r = new("side_s1_k4")
    _side(r, 4, "s", n=1)
    r = new("drop_both_k4")
    _drop(r, 4, "s")
    _drop(r, 4, "n")
    # the line runs into the inserter / away from it
    r = new("pick_end")
    r.chest("pc", 13, 0, t=BOUND_AT).ins("pi", 12, 0, W, t=BOUND_AT)
    _feed_m0(r)
    r = new("pick_start")
    r.chest("pc", -2, 0, t=BOUND_AT).ins("pi", -1, 0, E, t=BOUND_AT)
    r.put(620, "m0", 1, 200, "iron-plate").put(620, "m0", 2, 200, "iron-plate")
    r = new("drop_end")
    r.chest("dc", 13, 0, "iron-plate", 2, t=BOUND_AT).ins("di", 12, 0, E, t=BOUND_AT)
    r = new("drop_start")
    r.chest("dc", -2, 0, "iron-plate", 2, t=BOUND_AT).ins("di", -1, 0, W, t=BOUND_AT)

    # several on one line; some never working
    r = new("multi_drop_k2_k6")
    _drop(r, 2, "s")
    _drop(r, 6, "s")
    r = new("multi_drop_k3_pick_k4")
    _drop(r, 3, "s")
    _pick(r, 4, "n")
    r = new("multi_pick_k1_k4_k7")
    for x in (1, 4, 7):
        _pick(r, x, "s")
    _feed_m0(r)
    r = new("idle_drop_k7_drop_k2")
    _drop(r, 2, "s")
    _drop(r, 7, "s", count=0)
    r = new("idle_pick_k0_drop_k4")
    _pick(r, 0, "s")
    _drop(r, 4, "s")
    r = new("multi_side_k2_drop_k6")
    _side(r, 2, "s")
    _drop(r, 6, "n")

    # turns: m0..m5 east, m6 a turn south at (6, 0), m7..m11 south at (6, 1..5)
    def turned(name: str) -> Rig:
        r = Rig(name, next(at), 1500)
        for k in range(6):
            r.belt(f"m{k}", k, 0, E)
        r.belt("m6", 6, 0, S)
        for k in range(7, 12):
            r.belt(f"m{k}", 6, k - 6, S)
        rigs.append(r)
        return r

    r = turned("turn_pick_at")
    r.chest("pc", 8, 0, t=BOUND_AT).ins("pi", 7, 0, W, t=BOUND_AT)
    _feed_m0(r)
    r = turned("turn_drop_at")
    r.chest("dc", 8, 0, "iron-plate", 2, t=BOUND_AT).ins("di", 7, 0, E, t=BOUND_AT)
    r = turned("turn_drop_before")
    _drop(r, 4, "s")
    r = turned("turn_side_before")
    _side(r, 4, "s")
    r = turned("turn_side_before_n")
    _side(r, 5, "n")
    r = turned("turn_drop_after")
    r.chest("dc", 8, 1, "iron-plate", 2, t=BOUND_AT).ins("di", 7, 1, E, t=BOUND_AT)
    r = turned("turn_pick_before")
    _pick(r, 5, "s")
    _feed_m0(r)

    # a west line (m0 at x=11 ... m11 at x=0), fed from the south: its lane 1
    r = Rig("west_side_k4", next(at), 1500)
    line(r, 12, W, 11, 0)
    for j in (2, 1):
        r.belt(f"fs_{j}", 7, j, N)
    r.put(620, "fs_2", 1, 128, "copper-plate").put(620, "fs_2", 2, 128, "copper-plate")
    rigs.append(r)
    r = Rig("west_drop_k4", next(at), 1500)
    line(r, 12, W, 11, 0)
    r.chest("dc", 7, 2, "iron-plate", 2, t=BOUND_AT).ins("di", 7, 1, S, t=BOUND_AT)
    rigs.append(r)
    return rigs


# ------------------------------------------------------------- family: change


def change_rigs() -> list[Rig]:
    at = cells()
    rigs: list[Rig] = []

    def new(name: str, ticks: int, n: int = 8) -> Rig:
        r = line(Rig(name, next(at), ticks), n)
        rigs.append(r)
        return r

    for k in (0, 3, 7):
        r = new(f"rot_k{k}_old", 1950)
        r.rotate(650, f"m{k}").rotate(1300, f"m{k}", reverse=True)
        r = new(f"rot_k{k}_old4", 1300)
        for _ in range(4):
            r.rotate(650, f"m{k}")
        r = new(f"rot_k{k}_young4", 700)
        for _ in range(4):
            r.rotate(5, f"m{k}")
        r = new(f"rot_k{k}_young90", 700)
        r.rotate(5, f"m{k}").rotate(8, f"m{k}", reverse=True)
        r = new(f"rm_k{k}_old", 1700)
        r.destroy(650, f"m{k}").belt(f"m{k}b", k, 0, E, 1000)
        r = new(f"rm_k{k}_young", 700)
        r.destroy(5, f"m{k}").belt(f"m{k}b", k, 0, E, 12)
    # a reversed belt in a merged line, and turned back
    r = new("rot_k3_old180", 1950)
    r.rotate(650, "m3").rotate(650, "m3").rotate(1300, "m3").rotate(1300, "m3")
    # a line with a turn; the turn turned
    r = Rig("rot_turn_old", next(at), 1950)
    for k in range(4):
        r.belt(f"m{k}", k, 0, E)
    r.belt("m4", 4, 0, S)
    for k in range(5, 8):
        r.belt(f"m{k}", 4, k - 4, S)
    r.rotate(650, "m4", reverse=True).rotate(1300, "m4")
    rigs.append(r)
    # a split pending (pickup at m1: boundary m3|m4) while a belt is turned in
    # place (four quarter turns) or turned and back, downstream and upstream
    for where in (5, 2):
        for how in ("4x", "90"):
            r = new(f"split_rot_k{where}_{how}", 1500)
            _pick(r, 1, "s")
            _feed_m0(r)
            # the first pickup is about t=700 (plates put at 620 reach m1)
            if how == "4x":
                for _ in range(4):
                    r.rotate(720, f"m{where}")
            else:
                r.rotate(720, f"m{where}").rotate(723, f"m{where}", reverse=True)
    # merge timers running when a belt is turned or removed next to them:
    # two lines of 8, the belt between built at t=300 (merge timer from 300)
    for how in ("rot4", "rm"):
        r = Rig(f"rearm_{how}", next(at), 1300)
        for k in range(8):
            r.belt(f"m{k}", k, 0, E, t=0 if k != 4 else 300)
        if how == "rot4":
            for _ in range(4):
                r.rotate(310, "m6")
        else:
            r.destroy(310, "m7").belt("m7b", 7, 0, E, 312)
        rigs.append(r)
    return rigs


# ------------------------------------------------------------- family: loop


def _rect_loop(r: Rig, w: int = 4, h: int = 3, t: int = 0, start: int = 0,
               reverse: bool = False) -> list[str]:  # fmt: skip
    """A clockwise loop around a w x h rectangle: top row east, right column
    south, bottom row west, left column north, labelled l0.. in flow order
    from the top left corner, built from l<start> on, along the flow or
    against it. Returns the labels in flow order."""
    path = [(x, 0, E) for x in range(w - 1)]
    path += [(w - 1, y, S) for y in range(h - 1)]
    path += [(x, h - 1, W) for x in range(w - 1, 0, -1)]
    path += [(0, y, N) for y in range(h - 1, 0, -1)]
    n = len(path)
    for j in range(n):
        k = (start - j if reverse else start + j) % n
        x, y, d = path[k]
        r.belt(f"l{k}", x, y, d, t)
    return [f"l{k}" for k in range(n)]


#: Lane lengths on a clockwise loop: corners are right turns, lane 1 (left)
#: the outer 295, lane 2 the inner 106.
def _loop_lengths(w: int, h: int, lane: int) -> list[int]:
    n = 2 * (w - 1) + 2 * (h - 1)
    # the first belt of each run turns (fed from the side)
    corners = {0, w - 1, (w - 1) + (h - 1), 2 * (w - 1) + (h - 1)}
    out = []
    for k in range(n):
        if k in corners:
            out.append(295 if lane == 1 else 106)
        else:
            out.append(256)
    return out


def _fill_loop(r: Rig, t: int, labels: list[str], lengths: list[int], lane: int,
               spare: int) -> None:  # fmt: skip
    """Items 64 apart round a loop lane, as many as fit less `spare`, put from
    the front of the first belt upstream (each 64 behind the last put)."""
    total = sum(lengths)
    count = total // 64 - spare
    # flow coordinate g: belt k covers (start_k, start_k + L_k], position
    # p = start_k + L_k - g; walking upstream from g = L_0 - 8 (p = 8 on l0)
    starts, acc = [], 0
    for length in lengths:
        starts.append(acc)
        acc += length
    names = ("iron-plate", "copper-plate")
    g = lengths[0] - 8
    for i in range(count):
        k = max(j for j in range(len(lengths)) if starts[j] < g or (j == 0 and g > 0))
        r.put(t, labels[k], lane, starts[k] + lengths[k] - g, names[i % 2])
        g = (g - 64) % total


def loop_rigs() -> list[Rig]:
    at = cells()
    rigs: list[Rig] = []
    for age, t in (("young", 0), ("old", 650)):
        ticks = t + 500
        r = Rig(f"sq_{age}", next(at), ticks)
        r.belt("l0", 0, 0, E).belt("l1", 1, 0, S).belt("l2", 1, 1, W).belt("l3", 0, 1, N)
        r.put(t, "l0", 1, 0, "iron-plate").put(t, "l0", 1, 150, "copper-plate")
        r.put(t, "l2", 2, 50, "iron-ore")
        rigs.append(r)
        r = Rig(f"rect_{age}", next(at), ticks)
        _rect_loop(r)
        r.put(t, "l0", 1, 100, "iron-plate").put(t, "l4", 1, 30, "copper-plate")
        r.put(t, "l1", 2, 200, "iron-ore").put(t, "l6", 2, 10, "copper-ore")
        rigs.append(r)
        # both lanes as full as they go (`spare` 0), or one item short: items
        # 64 apart, the slack in one gap
        for spare in (0, 1):
            r = Rig(f"full_{age}_s{spare}", next(at), ticks)
            labels = _rect_loop(r)
            for lane in (1, 2):
                _fill_loop(r, t, labels, _loop_lengths(4, 3, lane), lane, spare)
            rigs.append(r)
        # a feed sideloading onto the loop's bottom row (lane 1): items on both
        # of its lanes reach an empty loop lane on the same tick
        r = Rig(f"feed_{age}", next(at), ticks)
        _rect_loop(r)
        r.belt("f2", 2, 4, N).belt("f1", 2, 3, N)
        r.put(t, "f2", 1, 128, "copper-plate").put(t, "f2", 2, 128, "iron-plate")
        rigs.append(r)
        r = Rig(f"feed_busy_{age}", next(at), ticks)
        _rect_loop(r)
        r.belt("f2", 2, 4, N).belt("f1", 2, 3, N)
        r.put(t, "l0", 1, 128, "iron-ore").put(t, "l3", 1, 200, "copper-ore")
        r.put(t + 20, "f2", 1, 128, "copper-plate").put(t + 20, "f2", 2, 128, "iron-plate")
        rigs.append(r)
    # an inserter dropping onto the loop and one taking from it
    r = Rig("ins_old", next(at), 1600)
    _rect_loop(r)
    r.chest("dc", 1, -2, "iron-plate", 3, t=610).ins("di", 1, -1, N, t=610)
    r.chest("pc", 2, 4, t=610).ins("pi", 2, 3, N, t=610)
    r.put(620, "l0", 1, 100, "copper-plate").put(620, "l0", 2, 100, "copper-plate")
    rigs.append(r)
    r = Rig("drop_old", next(at), 1600)
    _rect_loop(r)
    r.chest("dc", 1, -2, "iron-plate", 2, t=610).ins("di", 1, -1, N, t=610)
    rigs.append(r)
    # a larger loop: 6 x 4
    r = Rig("big_old", next(at), 1200)
    _rect_loop(r, 6, 4)
    for k, lane in ((0, 1), (3, 2), (7, 1), (9, 2)):
        r.put(650, f"l{k}", lane, 128, "iron-plate")
    rigs.append(r)
    return rigs


# ------------------------------------------------------------- family: order

#: Main: five old east belts m0..m4 at y=0. Feed A: twelve north belts a1
#: (tail, at (1, 12)) .. a12 (front, at (1, 1)) sideloading onto m1; feed B the
#: same at x=3 onto m3. Both land on the main's lane 2. An item x (copper) on
#: feed A and y (iron) on feed B reach the main on the same tick: the one whose
#: segment moves first goes 8/256 further. What happens to feed A in between,
#: with x on it, is the rig. The per-rig timings are in ORDER_TIMES.
#: Found with factory-sim so that x and y land on the same tick; y is put
#: before x (S1), after x and before the change (S2), or after it (S3).
ORDER_TIMES: dict[str, dict] = {
    "split_S1": {
        "base": (104, 104),
        "change": "split",
        "change_at": 610,
        "x_at": (640, "a5", 2, 128),
        "y_at": (630, "b5", 1, 208),
    },
    "split_S2": {
        "base": (272, 104),
        "change": "split",
        "change_at": 610,
        "x_at": (640, "a5", 2, 128),
        "y_at": (650, "b5", 1, 48),
    },
    "split_S3": {
        "base": (296, 104),
        "change": "split",
        "change_at": 610,
        "x_at": (640, "a5", 2, 128),
        "y_at": (852, "b12", 1, 224),
    },
    "rm_S1": {
        "base": (128, 104),
        "change": "rm",
        "change_at": 700,
        "x_at": (650, "a4", 2, 128),
        "y_at": (640, "b4", 1, 208),
    },
    "rm_S2": {
        "base": (152, 104),
        "change": "rm",
        "change_at": 700,
        "x_at": (650, "a4", 2, 128),
        "y_at": (660, "b4", 1, 48),
    },
    "rm_S3": {
        "base": (176, 104),
        "change": "rm",
        "change_at": 700,
        "x_at": (650, "a4", 2, 128),
        "y_at": (702, "b6", 1, 224),
    },
    "rot_S1": {
        "base": (200, 104),
        "change": "rot",
        "change_at": 700,
        "x_at": (650, "a4", 2, 128),
        "y_at": (640, "b4", 1, 208),
    },
    "rot_S2": {
        "base": (224, 104),
        "change": "rot",
        "change_at": 700,
        "x_at": (650, "a4", 2, 128),
        "y_at": (660, "b4", 1, 48),
    },
    "rot_S3": {
        "base": (248, 104),
        "change": "rot",
        "change_at": 700,
        "x_at": (650, "a4", 2, 128),
        "y_at": (702, "b6", 1, 224),
    },
    "rm11_S1": {
        "base": (320, 104),
        "change": "rm11",
        "change_at": 700,
        "x_at": (680, "a12", 2, 250),
        "y_at": (670, "b11", 1, 72),
    },
    "rm11_S2": {
        "base": (104, 128),
        "change": "rm11",
        "change_at": 700,
        "x_at": (680, "a12", 2, 250),
        "y_at": (690, "b12", 1, 168),
    },
    "rm11_S3": {
        "base": (128, 128),
        "change": "rm11",
        "change_at": 700,
        "x_at": (680, "a12", 2, 250),
        "y_at": (702, "b12", 1, 72),
    },
    "none_S2": {
        "base": (152, 128),
        "change": "none",
        "change_at": 700,
        "x_at": (650, "a4", 2, 128),
        "y_at": (660, "b4", 1, 48),
    },
}


Put = tuple[int, str, int, int]


def order_rig(name: str, base: tuple[int, int], change: str, x_at: Put, y_at: Put,
              change_at: int = 0, ticks: int = 1500) -> Rig:  # fmt: skip
    r = Rig(name, base, ticks)
    for k in range(5):
        r.belt(f"m{k}", k, 0, E)
    for j in range(1, 13):
        r.belt(f"a{j}", 1, 13 - j, N)
    for j in range(1, 13):
        r.belt(f"b{j}", 3, 13 - j, N)
    if change == "split":
        # an inserter west of a1 drops one stone on a1's lane 2 (its far
        # lane): the boundary is a3|a4
        r.chest("dc", -1, 12, "stone", 1, t=change_at).ins("di", 0, 12, W, t=change_at)
    elif change == "rm":
        r.destroy(change_at, "a2")
    elif change == "rot":
        r.rotate(change_at, "a2")
    elif change == "rm11":
        r.destroy(change_at, "a11")
    elif change == "none":
        pass
    else:
        raise ValueError(change)
    t, belt, lane, pos = x_at
    r.put(t, belt, lane, pos, "copper-plate")
    t, belt, lane, pos = y_at
    r.put(t, belt, lane, pos, "iron-plate")
    return r


def order_rigs() -> list[Rig]:
    return [order_rig(name, **spec) for name, spec in ORDER_TIMES.items()]


# ------------------------------------------------------------- family: trig


def trig_rigs() -> list[Rig]:
    """What starts the split timer at an attachment's boundary: an old line of
    12 east belts, one attachment built at t=610, items put by script."""
    at = cells()
    rigs: list[Rig] = []

    def new(name: str, ticks: int = 1800) -> Rig:
        r = line(Rig(name, next(at), ticks), 12)
        rigs.append(r)
        return r

    new("idle_pick").chest("pc", 4, 2, t=610).ins("pi", 4, 1, N, t=610)
    for lane in (1, 2):
        r = new(f"pick_up_l{lane}")
        _pick(r, 4, "s")
        r.put(700, "m0", lane, 128, "iron-plate")
        r = new(f"pick_on_l{lane}")
        _pick(r, 4, "s")
        r.put(700, "m4", lane, 200, "iron-plate")
        r = new(f"pick_down_l{lane}")
        _pick(r, 4, "s")
        r.put(700, "m9", lane, 128, "iron-plate")
        r = new(f"idle_drop_put_l{lane}")
        _drop(r, 4, "s", count=0)
        r.put(700, "m0", lane, 128, "iron-plate")
        r = new(f"side_put_l{lane}")
        r.belt("f2", 4, 2, N, 610).belt("f1", 4, 1, N, 610)
        r.put(700, "m0", lane, 128, "iron-plate")
        r = new(f"nothing_put_l{lane}")
        r.put(700, "m0", lane, 128, "iron-plate")
        r = new(f"pick_k0_l{lane}")
        _pick(r, 0, "s")
        r.put(700, "m0", lane, 128, "iron-plate")
    r = new("pick_full")
    r.op(610, "chest", "pc", 4, 2, "stone", 800).ins("pi", 4, 1, N, t=610)
    r.put(700, "m0", 1, 128, "iron-plate")
    r = new("pick_twice")
    _pick(r, 4, "s")
    r.put(700, "m9", 1, 128, "iron-plate").put(800, "m0", 1, 128, "copper-plate")
    r = new("pick_late")
    _pick(r, 4, "s")
    r.put(1100, "m0", 1, 128, "iron-plate")
    r = new("drop_after_put")
    r.put(700, "m9", 1, 128, "copper-plate")
    _drop(r, 4, "s", t=800)
    r = new("pick_after_put")
    r.put(650, "m9", 1, 128, "copper-plate")
    _pick(r, 4, "s", t=700)
    r = new("pick_n_l1")
    _pick(r, 4, "n")
    r.put(700, "m0", 1, 128, "iron-plate")
    r = new("pick_k0_both")
    _pick(r, 0, "s")
    r.put(700, "m0", 1, 128, "iron-plate").put(700, "m0", 2, 128, "copper-plate")
    return rigs


# ------------------------------------------------------------- family: feedchg


def feedchg_rigs() -> list[Rig]:
    """A feed changed next to its sideload: an old main of 8 east belts
    m0..m7, a feed of 5 belts f1 (tail) .. f5 (front) sideloading onto m3
    from the south (or the north); one belt removed at t=650 and built again
    at t=1000, or turned and turned back, or turned four times."""
    at = cells()
    rigs: list[Rig] = []

    def new(name: str, side: str = "s") -> Rig:
        r = line(Rig(name, next(at), 1700), 8)
        for j in range(1, 6):
            if side == "s":
                r.belt(f"f{j}", 3, 6 - j, N)
            else:
                r.belt(f"f{j}", 3, j - 6, S)
        rigs.append(r)
        return r

    def where(r: Rig, label: str) -> tuple[int, int, int]:
        return next((o[3], o[4], o[5]) for o in r.ops if o[1] == "belt" and o[2] == label)

    for b in ("f5", "f4", "f3", "f2", "m2", "m3", "m4", "m5"):
        r = new(f"rm_{b}")
        r.destroy(650, b).belt(f"{b}b", *where(r, b), 1000)
    for b in ("f5", "f4", "m3"):
        r = new(f"rot_{b}")
        r.rotate(650, b).rotate(1000, b, reverse=True)
        r = new(f"rot4_{b}")
        for _ in range(4):
            r.rotate(650, b)
    for b in ("f5", "f4"):
        r = new(f"rm_{b}_n", side="n")
        r.destroy(650, b).belt(f"{b}b", *where(r, b), 1000)
    return rigs


# ------------------------------------------------------------- family: loop2


def _loop_items(r: Rig, t: int) -> None:
    r.put(t, "l2", 1, 100, "iron-plate").put(t, "l7", 1, 30, "copper-plate")
    r.put(t, "l4", 2, 150, "iron-ore").put(t, "l9", 2, 60, "copper-ore")


def loop2_rigs() -> list[Rig]:
    """Where a loop's seam is: 4 x 3 loops built from different belts, along
    the flow or against it, items on both lanes put at t=650 (old) or t=0;
    merged loops with a belt removed and built again, or turned in place."""
    at = cells()
    rigs: list[Rig] = []
    for start in (0, 1, 3, 5, 8):
        for rev in (False, True):
            for age, t in (("old", 650), ("young", 0)):
                name = f"b{start}{'r' if rev else ''}_{age}"
                r = Rig(name, next(at), t + 600)
                _rect_loop(r, start=start, reverse=rev)
                _loop_items(r, t)
                rigs.append(r)
    for k in (0, 2, 6):
        r = Rig(f"rm_l{k}", next(at), 1900)
        _rect_loop(r)
        x, y, d = next((o[3], o[4], o[5]) for o in r.ops if o[1] == "belt" and o[2] == f"l{k}")
        r.destroy(650, f"l{k}").belt(f"l{k}", x, y, d, 700)
        _loop_items(r, 1300)
        rigs.append(r)
        r = Rig(f"rot4_l{k}", next(at), 1900)
        _rect_loop(r)
        for _ in range(4):
            r.rotate(650, f"l{k}")
        _loop_items(r, 1300)
        rigs.append(r)
    r = Rig("young_busy", next(at), 900)
    labels = _rect_loop(r)
    for label in labels:
        r.put(0, label, 1, 128, "iron-plate")
    rigs.append(r)
    return rigs


# ------------------------------------------------------------- family: trig2

#: Bases for `seq_*`, found with factory-sim's delay table: lane 1 of m11 has
#: a short delay and m9's a very different one (the first split is at the
#: first drop, t=656, plus d(m11)). See TRIG2_TIMES.
TRIG2_TIMES: dict[str, dict] = {
    "seq_a": {"base": (140, 320), "t2": 740},  # d(m11) 57, d(m9) 269
    "seq_b": {"base": (209, 320), "t2": 700},  # d(m11) 23, d(m9) 139
    "seq_c": {"base": (227, 296), "t2": 870},  # d(m11) 193, d(m9) 64
}


def trig3_rigs() -> list[Rig]:
    """Boundaries close together: drops (one plate each) onto lane 1 of an old
    line of 12 east belts, one, two or three belts apart, together or the
    second built after the first split (bases where d(m11) is short)."""
    at = cells()
    rigs: list[Rig] = []

    def new(name: str, base=None) -> Rig:
        r = line(Rig(name, base or next(at), 1800), 12)
        rigs.append(r)
        return r

    for xs in ((2, 4), (2, 5), (2, 3, 4), (3, 4), (4, 5, 7), (2, 3, 5)):
        r = new("drops_" + "_".join(f"k{x}" for x in xs))
        for x in xs:
            _drop(r, x, "s", count=1)
    # first at 610 (split at 656 + d(m11)), the other built 20 ticks later
    r = new("late_up", (122, 272))  # d(m11) 41
    _drop(r, 4, "s", count=1)
    _drop(r, 3, "s", count=1, t=720)
    r = new("late_down", (178, 272))  # d(m11) 36
    _drop(r, 3, "s", count=1)
    _drop(r, 4, "s", count=1, t=710)
    return rigs


def trig2_rigs() -> list[Rig]:
    """Which attachments' boundaries a split uses, and the delay a second
    split counts: an old line of 12 east belts, attachments at t=610."""
    at = cells()
    rigs: list[Rig] = []

    def new(name: str, base=None, ticks: int = 1800) -> Rig:
        r = line(Rig(name, base or next(at), ticks), 12)
        rigs.append(r)
        return r

    for name, spec in TRIG2_TIMES.items():
        r = new(name, tuple(spec["base"]), 2200)
        _drop(r, 7, "s", count=1)
        t2 = spec["t2"]
        r.put(t2, "m6", 1, 128, "copper-plate")
        _pick(r, 1, "s", t=t2)
    r = new("act_full")
    _drop(r, 3, "s", count=1)
    r.op(610, "chest", "pc4", 4, -2, "stone", 800).ins("pi4", 4, -1, S, t=610)
    r = new("act_far")
    _drop(r, 2, "s", count=1)
    _pick(r, 7, "n")
    r = new("act_adj_drops")
    _drop(r, 3, "s", count=1)
    _drop(r, 4, "s", count=1)
    r = new("act_flow")
    _pick(r, 0, "n")
    r.op(610, "chest", "dc4", 4, 2, "iron-plate", 20).ins("di4", 4, 1, S, t=610)
    r = new("act_pick_after")
    _drop(r, 2, "s", count=1)
    _pick(r, 8, "n")
    r = new("act_idle_drop_pick")
    _drop(r, 7, "s", count=0)
    _pick(r, 4, "n")
    r.put(700, "m0", 1, 128, "iron-plate")
    return rigs


# ------------------------------------------------------------- family: dist

TURN = {"S": 0, "R": 4, "L": 12}
PATTERNS = ("RSSSS", "SRSSS", "SSRSS", "LSSSS", "SLSSS", "SSLSS", "RLSSS", "LRSSS", "RSRSS",
            "LSLSS", "SRLSS")  # fmt: skip


def _path(pattern: str) -> list[tuple[int, int, int]]:
    """Belts m0..m2 east from (0, 0), then one belt per letter: S straight on,
    R a right turn, L a left turn (the belt faces the new way)."""
    vec = {N: (0, -1), E: (1, 0), S: (0, 1), W: (-1, 0)}
    cells_ = [(0, 0, E), (1, 0, E), (2, 0, E)]
    x, y, d = 2, 0, E
    for ch in pattern:
        dx, dy = vec[d]
        x, y = x + dx, y + dy
        d = (d + TURN[ch]) % 16
        cells_.append((x, y, d))
    return cells_


def dist_rigs() -> list[Rig]:
    """Where the boundary lies when the belts after the attachment turn: an
    old line m0..m2 east then a pattern of straights and turns; at m2 a drop
    from the south (lane 1) or the north (lane 2), a pickup from the south or
    the north (plates put on m0, both lanes), or a two-belt feed sideloading
    from the south (lane 2) or the north (lane 1)."""
    at = cells()
    rigs: list[Rig] = []
    for pattern in PATTERNS:
        path = _path(pattern)
        taken = {(x, y) for x, y, _ in path}
        for kind in ("drop_s", "drop_n", "pick_s", "pick_n", "side_s", "side_n"):
            side = kind[-1]
            dy = 1 if side == "s" else -1
            need = {(2, dy), (2, 2 * dy)}
            if need & taken:
                continue
            r = Rig(f"{kind}_{pattern}", next(at), 1600)
            for k, (x, y, d) in enumerate(path):
                r.belt(f"m{k}", x, y, d)
            if kind.startswith("drop"):
                _drop(r, 2, side)
            elif kind.startswith("pick"):
                _pick(r, 2, side)
                _feed_m0(r, 700)
            else:
                _side(r, 2, side)
            rigs.append(r)
    return rigs


# ------------------------------------------------------------- family: drill

#: Patterns after the drill's belt m2 whose lane sums, from m2's downstream
#: edge, pass through each threshold of 468..768 on the lane the drill drops
#: on (lane 1 from the north, lane 2 from the south), two where the drill's
#: footprint allows (found by enumeration; turn lanes are 295 and 106 long).
DRILL_PATTERNS = {
    1: ("SLLS", "RLLS", "LRLS", "SRS", "RSS", "RRS", "RRSS", "LRLLS", "LRLLRS", "SSLS",
        "SLSS", "LSSS", "SRLS", "SLRS", "RSLS", "RLSS", "RRLS", "RLRS", "SSLLS", "SLSLS",
        "SSSS"),
    2: ("LRRS", "SLS", "LSS", "LLS", "LLSS", "SSRS", "SRSS", "SRLS", "SLRS", "LRLS",
        "LLRS", "SSRRS", "SSSS"),
}  # fmt: skip


def drill_rigs() -> list[Rig]:
    """A drill's boundary on its output lane, where the belts after its own
    turn: an old line m0..m2 east (built at t=0) then a pattern (`_path`); at
    t=610 a burner mining drill with 5 coal drops onto m2, from the north
    (facing south, corner (2, -1): lane 1) or the south (facing north, corner
    (3, 2): lane 2). Then a drill dropping onto a turn: m0..m1 east, m2 a right
    turn south at (2, 0), then a pattern; the drill from the north (the turn's
    outer lane: a drill's drop point near a turn's inner corner lies where the
    belts are)."""
    at = cells()
    rigs: list[Rig] = []
    for lane, patterns in DRILL_PATTERNS.items():
        for pattern in patterns:
            r = Rig(f"d{lane}_{pattern}", next(at), 1800)
            for k, (x, y, d) in enumerate(_path(pattern)):
                r.belt(f"m{k}", x, y, d)
            if lane == 1:
                r.drill("dr", 2, -1, S, t=610)
            else:
                r.drill("dr", 3, 2, N, t=610)
            rigs.append(r)
    # onto a turn: m2 at (2, 0) facing south is a right turn (fed from the west)
    for pattern in ("SLSS", "SRLS"):
        r = Rig(f"dt_{pattern}", next(at), 1800)
        cells_ = [(0, 0, E), (1, 0, E), (2, 0, S)]
        x, y, d = 2, 0, S
        vec = {N: (0, -1), E: (1, 0), S: (0, 1), W: (-1, 0)}
        for ch in pattern:
            dx, dy = vec[d]
            x, y = x + dx, y + dy
            d = (d + TURN[ch]) % 16
            cells_.append((x, y, d))
        for k, (bx, by, bd) in enumerate(cells_):
            r.belt(f"m{k}", bx, by, bd)
        r.drill("dr", 2, -1, S, t=610)
        rigs.append(r)
    return rigs


# ------------------------------------------------------------- family: loop3

#: A 2 x 2 loop, clockwise (all right turns, lane 2 inner, 106 long) or
#: anticlockwise (all left turns, lane 1 inner): six inner lanes are 636,
#: inside the interval the inserter's reach was bracketed to (618..657).
LOOP_CW = [(0, 0, E), (1, 0, S), (1, 1, W), (0, 1, N)]
LOOP_CCW = [(0, 0, S), (0, 1, E), (1, 1, N), (1, 0, W)]
#: The eight tiles outside a 2 x 2 loop, with the loop tile each is next to.
OUTSIDE = [((0, -1), (0, 0)), ((-1, 0), (0, 0)), ((1, -1), (1, 0)), ((2, 0), (1, 0)),
           ((2, 1), (1, 1)), ((1, 2), (1, 1)), ((0, 2), (0, 1)), ((-1, 1), (0, 1))]  # fmt: skip


def _dir_to(dx: int, dy: int) -> int:
    return {(0, -1): N, (1, 0): E, (0, 1): S, (-1, 0): W}[(dx, dy)]


def loop3_rigs() -> list[Rig]:
    """A 2 x 2 loop built at t=0 (from l0, or from l2), and at t=610 one burner
    inserter outside it: taking from the loop into a chest, with one plate put
    at t=650 on the inner or the outer lane of the belt two ahead of its own;
    or dropping two plates from a chest onto the loop."""
    at = cells()
    rigs: list[Rig] = []
    for orient, loop in (("cw", LOOP_CW), ("ccw", LOOP_CCW)):
        inner = 2 if orient == "cw" else 1
        tiles = {(x, y): k for k, (x, y, _) in enumerate(loop)}
        for start in (0, 2):
            for pos, (ins_xy, belt_xy) in enumerate(OUTSIDE):
                k = tiles[belt_xy]
                ix, iy = ins_xy
                dx, dy = belt_xy[0] - ix, belt_xy[1] - iy
                kinds = ("pin", "pout", "drop") if start == 0 else ("pin", "drop")
                for kind in kinds:
                    r = Rig(f"{orient}_b{start}_{kind}_p{pos}", next(at), 1500)
                    for j in range(4):
                        x, y, d = loop[(start + j) % 4]
                        r.belt(f"l{(start + j) % 4}", x, y, d)
                    if kind == "drop":
                        r.chest("dc", ix - dx, iy - dy, "iron-plate", 2, t=610)
                        r.ins("di", ix, iy, _dir_to(-dx, -dy), t=610)
                    else:
                        r.chest("pc", ix - dx, iy - dy, t=610)
                        r.ins("pi", ix, iy, _dir_to(dx, dy), t=610)
                        lane = inner if kind == "pin" else 3 - inner
                        at_pos = 50 if kind == "pin" else 128  # an inner lane is 106 long
                        r.put(650, f"l{(k + 2) % 4}", lane, at_pos, "iron-plate")
                    rigs.append(r)
    return rigs


FAMILIES = {
    "drill": drill_rigs,
    "loop3": loop3_rigs,
    "trig2": trig2_rigs,
    "trig3": trig3_rigs,
    "dist": dist_rigs,
    "bound": bound_rigs,
    "change": change_rigs,
    "loop": loop_rigs,
    "order": order_rigs,
    "trig": trig_rigs,
    "feedchg": feedchg_rigs,
    "loop2": loop2_rigs,
}

# ---------------------------------------------------------------- the engine side

PRE = r"""
local s = game.surfaces[1]
storage.p5 = {log = {}}
local P = storage.p5
local function prepare(x0, y0, x1, y1)
  local cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
  local r = math.ceil(math.max(x1 - x0, y1 - y0) / 64) + 2
  s.request_to_generate_chunks({cx, cy}, r)
  s.force_generate_chunk_requests()
  for _, c in pairs(s.find_entities_filtered({type = "character"})) do
    c.teleport({x0 - 40, y1 + 40})
  end
  for _, e in pairs(s.find_entities_filtered({area = {{x0, y0}, {x1, y1}}})) do
    if e.valid and e.type ~= "character" and e.type ~= "resource" then e.destroy() end
  end
  local tiles = {}
  for x = x0, x1 - 1 do
    for y = y0, y1 - 1 do tiles[#tiles + 1] = {name = "grass-1", position = {x, y}} end
  end
  s.set_tiles(tiles)
end
"""

#: The interpreter. RIGS (from Python): {name, base = {x, y}, ops = {{t, op, ...}}}.
RUN = r"""
local ST = {}
for k, rig in ipairs(RIGS) do
  ST[k] = {E = {}, belts = {}, ins = {}, chests = {}, last = nil, byt = {}}
  for _, op in ipairs(rig.ops) do
    local t = op[1]
    ST[k].byt[t] = ST[k].byt[t] or {}
    table.insert(ST[k].byt[t], op)
  end
end
local function mk(spec)
  spec.force = "player"
  local e = s.create_entity(spec)
  assert(e, "create failed " .. spec.name .. " at " .. spec.position[1] .. "," .. spec.position[2])
  return e
end
local function apply(k, op)
  local st, rig = ST[k], RIGS[k]
  local kind, label = op[2], op[3]
  local bx, by = rig.base[1], rig.base[2]
  if kind == "belt" then
    local e = mk({name = "transport-belt", position = {bx + op[4] + 0.5, by + op[5] + 0.5},
                  direction = op[6]})
    st.E[label] = e
    st.belts[#st.belts + 1] = e
  elseif kind == "chest" then
    local e = mk({name = "wooden-chest", position = {bx + op[4] + 0.5, by + op[5] + 0.5}})
    if op[6] and op[6] ~= "" and op[7] > 0 then
      assert(e.insert({name = op[6], count = op[7]}) == op[7])
    end
    st.E[label] = e
    st.chests[#st.chests + 1] = e
  elseif kind == "ins" then
    local e = mk({name = "burner-inserter", position = {bx + op[4] + 0.5, by + op[5] + 0.5},
                  direction = op[6]})
    if op[7] > 0 then e.insert({name = "coal", count = op[7]}) end
    st.E[label] = e
    st.ins[#st.ins + 1] = e
  elseif kind == "drill" then
    local cx, cy = bx + op[4], by + op[5]
    for _, r in pairs(s.find_entities_filtered({area = {{cx - 0.95, cy - 0.95},
                                                        {cx + 0.95, cy + 0.95}},
                                                type = "resource"})) do
      r.destroy()
    end
    for dx = -1, 0 do
      for dy = -1, 0 do
        s.create_entity({name = "iron-ore", position = {cx + dx + 0.5, cy + dy + 0.5},
                         amount = 5000})
      end
    end
    local e = mk({name = "burner-mining-drill", position = {cx, cy}, direction = op[6]})
    if op[7] > 0 then e.insert({name = "coal", count = op[7]}) end
    st.E[label] = e
  elseif kind == "put" then
    -- a refused insert shows in the readings
    st.E[label].get_transport_line(op[4]).insert_at(op[5] / 256, {name = op[6], count = 1})
  elseif kind == "rotate" then
    assert(st.E[label].rotate({reverse = op[4] == 1}), "rotate failed " .. label)
  elseif kind == "destroy" then
    st.E[label].destroy()
  else
    error("op " .. kind)
  end
end
local function items(line)
  local o = {}
  for _, it in pairs(line.get_detailed_contents()) do
    o[#o + 1] = {math.floor(it.position * 256 + 0.5), it.unique_id, it.stack.name}
  end
  table.sort(o, function(a, b) return a[1] < b[1] or (a[1] == b[1] and a[2] < b[2]) end)
  return o
end
local function reading(st)
  local b, lines, cls = {}, {}, {}
  for i, e in ipairs(st.belts) do
    if e.valid then
      local l1, l2 = e.get_transport_line(1), e.get_transport_line(2)
      b[i] = {items(l1), items(l2)}
      lines[2 * i - 1], lines[2 * i] = l1, l2
    else
      b[i] = "gone"
    end
  end
  local reps = {}
  for j = 1, 2 * #st.belts do
    local l = lines[j]
    if l == nil then
      cls[j] = -1
    else
      local found = nil
      for _, r in ipairs(reps) do
        if l.line_equals(lines[r]) then found = r break end
      end
      if not found then reps[#reps + 1] = j found = j end
      cls[j] = found - 1
    end
  end
  local ins = {}
  for i, e in ipairs(st.ins) do
    ins[i] = (e.valid and e.held_stack.valid_for_read) and e.held_stack.name or ""
  end
  local ch = {}
  for i, c in ipairs(st.chests) do
    local o = {}
    local inv = c.get_inventory(defines.inventory.chest)
    for q = 1, #inv do
      if inv[q].valid_for_read then o[#o + 1] = {q, inv[q].name, inv[q].count} end
    end
    ch[i] = o
  end
  return {b = b, s = cls, i = ins, c = ch}
end
local function step(t)
  for k, rig in ipairs(RIGS) do
    if t <= rig.ticks then
      local st = ST[k]
      for _, op in ipairs(st.byt[t] or {}) do apply(k, op) end
      local r = reading(st)
      local key = helpers.table_to_json(r)
      if key ~= st.last then
        st.last = key
        P.log[#P.log + 1] = {k, t, key}
      end
    end
  end
end
P.t0 = game.tick
for k, _ in ipairs(RIGS) do
  for _, op in ipairs(ST[k].byt[0] or {}) do apply(k, op) end
  ST[k].byt[0] = nil
end
local old = script.get_event_handler(defines.events.on_tick)
script.on_event(defines.events.on_tick, function(e)
  if old then old(e) end
  if P.last == game.tick then return end
  P.last = game.tick
  step(game.tick - P.t0)
end)
return #RIGS
"""


def _lua_value(v) -> str:
    if v is None:
        return '""'
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, (list, tuple)):
        return "{" + ", ".join(_lua_value(x) for x in v) + "}"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k} = {_lua_value(x)}" for k, x in v.items()) + "}"
    raise TypeError(v)


def _bounds(rigs: list[dict]) -> tuple[int, int, int, int]:
    xs, ys = [], []
    for r in rigs:
        for op in r["ops"]:
            if op[1] in ("belt", "chest", "ins", "drill"):
                xs.append(r["base"][0] + op[3])
                ys.append(r["base"][1] + op[4])
    return min(xs) - 3, min(ys) - 3, max(xs) + 4, max(ys) + 4


def run_world(rigs: list[dict], label: str) -> dict:
    """Build and run `rigs` in one fresh world; the per-rig logs."""
    ticks = max(r["ticks"] for r in rigs)
    x0, y0, x1, y1 = _bounds(rigs)
    body = (PRE + f"prepare({x0}, {y0}, {x1}, {y1})\n"
            + "local RIGS = " + _lua_value(rigs) + "\n" + RUN)  # fmt: skip
    sys.path.insert(0, str(ROOT / "src"))
    from factoriorl.engine_config import resolve_game_speed
    from factoriorl.env import FactorioEnv
    from factoriorl.rcon import RCONClient
    from factoriorl.seeding import Branch, SeedPlan
    from factoriorl.session import WorkerSession
    from factoriorl.tasks import get
    from factoriorl.worker import WorkerManager

    manager = WorkerManager()
    handle = manager.launch(label)
    try:
        with RCONClient(handle.spec.rcon_endpoint, timeout=3600.0) as client:
            client.lua(f"game.speed = {resolve_game_speed()} return game.speed")
            session = WorkerSession(handle, timeout=3600.0)
            session.status()
            env = FactorioEnv(get("construct_smelting_line"), session,
                              SeedPlan(master=41, run_id=label), branch=Branch.TRAIN,
                              split="train")  # fmt: skip
            env.reset(options={"scene_index": 0})
            client.lua(body)
            env.advance(ticks + 2)
            n = client.lua("return #storage.p5.log")
            rows = []
            for a in range(1, n + 1, 100):
                rows += client.lua(
                    f"local o = {{}} for i = {a}, math.min({a + 99}, #storage.p5.log) do "
                    "o[#o + 1] = storage.p5.log[i] end return o"
                )
            session.close()
    finally:
        manager.cleanup(handle)
    logs: dict = {r["name"]: [] for r in rigs}
    for k, t, key in rows:
        logs[rigs[k - 1]["name"]].append([t, json.loads(key)])
    return logs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--family", choices=sorted(FAMILIES), required=True)
    ap.add_argument("--per-world", type=int, default=30)
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--only", default="", help="comma-separated rig names")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    started = time.perf_counter()
    rigs = [r.to_dict() for r in FAMILIES[args.family]()]
    if args.only:
        keep = set(args.only.split(","))
        rigs = [r for r in rigs if r["name"] in keep]
    groups = [rigs[i : i + args.per_world] for i in range(0, len(rigs), args.per_world)]
    logs: dict = {}

    def one(k: int) -> None:
        t0 = time.perf_counter()
        logs.update(run_world(groups[k], f"probe5-{args.family}-{k:02d}"))
        print(f"  world {k}: {len(groups[k])} rigs, {time.perf_counter() - t0:.0f} s", flush=True)

    with ThreadPoolExecutor(args.parallel) as pool:
        list(pool.map(one, range(len(groups))))
    from factoriorl.worker import WorkerManager

    doc = {"rigs": {r["name"]: r for r in rigs}, "log": logs,
           "engine": WorkerManager().engine.to_dict(),
           "wall_seconds": round(time.perf_counter() - started, 1)}  # fmt: skip
    out = Path(args.out) if args.out else EVIDENCE / f"logistics5-{args.family}.json.xz"
    out.write_bytes(lzma.compress(json.dumps(doc, sort_keys=True).encode()))
    print(f"wrote {out} ({out.stat().st_size // 1024} KiB), {doc['wall_seconds']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
