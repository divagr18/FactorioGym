"""Check the third logistics probe's rules against the stored evidence (no engine).

docs/sim-logistics.md, "Third probe", states rules for belt-line segments and
the order the engine moves them in. This replays them on the evidence in
`docs/evidence/` and prints, case by case, what they predict and what the
engine did:

- **sideload**: which of two items sideloading onto one empty lane in the same
  tick goes 8/256 further, from the order the two feed lanes were given their
  items, the ticks the items cross onto the next belt, and the merge delays;
- **young wake**: when a sleeping inserter on young belts wakes for an item
  added four belts upstream (`inserter-wake.json`, group `young`);
- **splits**: when a drill or an inserter splits a merged lane.

Run:
  uv run python tools/check_logistics3.py
"""

from __future__ import annotations

import json
import lzma
import re
import sys
from pathlib import Path

EVIDENCE = Path(__file__).resolve().parents[1] / "docs" / "evidence"


def _xz(name: str):
    return json.loads(lzma.decompress((EVIDENCE / name).read_bytes()))


def delays() -> dict[tuple[int, int, int], int]:
    out = {}
    for path in sorted(EVIDENCE.glob("logistics3-delay*.json.xz")):
        for key, t in json.loads(lzma.decompress(path.read_bytes()))["delay"].items():
            out[tuple(int(v) for v in key.split(","))] = t
    return out


D = delays()


def chain_merge(tiles: list[tuple[int, int]], lane: int) -> int | None:
    """Tick a feed chain's lane merges: the first of its belts' delays."""
    return min(D[(x, y, lane)] for x, y in tiles) if len(tiles) > 1 else None


def leading_lane(placed_first: int, placed_at: int, crossings: list[int], arrival: int,
                 merge: dict[int, int | None]) -> int:  # fmt: skip
    """The feed lane whose item goes further: the lane moved first on arrival.

    Last activated moves first; a crossing onto a belt that is still its own
    segment activates it; a lane that merges while carrying items keeps its
    place if its item is on the chain's last belt, else goes last."""
    order = [3 - placed_first, placed_first]
    merged = {lane for lane, m in merge.items() if m is not None and m <= placed_at}
    events = [(t, 1, None) for t in crossings]
    events += [
        (m, 0, lane) for lane, m in merge.items() if m is not None and placed_at < m < arrival
    ]
    for t, _, lane in sorted(events):
        if lane is not None:
            merged.add(lane)
            if crossings and t <= crossings[-1]:
                order.remove(lane)
                order.append(lane)
            continue
        for x in [x for x in order if x not in merged]:
            order.remove(x)
            order.insert(0, x)
    return order[0]


def _extra_lane(positions: list[int], far: int, k: int = 0) -> int | None:
    """From the first arrivals' positions: the feed lane whose item went 8
    further than the single-item rule (180 + k from the far lane, 59 + k)."""
    for pos in positions:
        if pos == 172 + k:
            return far
        if pos == 51 + k:
            return 3 - far
    return None


def sideload_cases():
    """(name, predicted, observed) for every two-item sideload in the evidence."""
    out = []
    # second probe: sim_* rigs and side_both (tools/probe_logistics2.py)
    p2 = _xz("sim-mechanics-m4-logistics2.json.xz")
    bases = p2["setup"]["bases"]
    rigs = {  # name: (feed belts, side, belt the items start on, lane placed first, position)
        "sim_l1first": (1, "S", 1, 1, 128), "sim_l2first": (1, "S", 1, 2, 128),
        "sim_f4_l1first": (4, "S", 4, 1, 128), "sim_f4_l2first": (4, "S", 4, 2, 128),
        "sim_feedfirst_l1": (1, "S", 1, 1, 128), "sim_feedfirst_l2": (1, "S", 1, 2, 128),
        "sim_active_l1": (1, "S", 1, 1, 128), "sim_active_l2": (1, "S", 1, 2, 128),
        "sim_other_l1": (1, "S", 1, 1, 128), "sim_mainfirst_l1": (1, "S", 1, 1, 128),
        "sim_n_l1first": (1, "N", 1, 1, 128), "sim_n_l2first": (1, "N", 1, 2, 128),
        "sim_k3_l1first": (1, "S", 1, 1, 131), "sim_k3_l2first": (1, "S", 1, 2, 131),
        "sim_three_l1first": (4, "S", 4, 1, 248), "sim_three_l2first": (4, "S", 4, 2, 248),
        "sim_pairs_l1": (2, "S", 1, 1, 128), "sim_pairs_l2": (2, "S", 1, 2, 128),
    }  # fmt: skip
    for name, (feed, side, where, first, start) in rigs.items():
        bx, by = bases[name]
        x0, y0 = 200 + bx, 200 + by
        tiles = [(x0 + 1, y0 + k) if side == "S" else (x0 + 1, y0 + feed - k)
                 for k in range(1, feed + 1)]  # fmt: skip
        merge = {lane: chain_merge(tiles, lane) for lane in (1, 2)}
        cross0 = 32 if start >= 248 else 16
        crossings = [cross0 + 32 * j for j in range(where - 1)]
        pred = leading_lane(first, 0, crossings, cross0 + 32 * (where - 1), merge)
        target = 2 if "feedfirst" in name else 1
        seen, first_in = {}, []
        for t, rec in p2["series"][name]:
            for bi, b in enumerate(rec["b"]):
                for li, lane_items in enumerate(b["l"] if b != "gone" else []):
                    for _, pos, uid in lane_items if isinstance(lane_items, list) else []:
                        if seen.get(uid) != bi:
                            seen[uid] = bi
                            if bi == target and t > 0:
                                first_in.append((t, pos, li))
        t0 = first_in[0][0]
        arrived = [p for t, p, _ in first_in if t == t0]
        obs = _extra_lane(arrived, 1 if side == "S" else 2, start % 8 if start < 248 else 0)
        out.append((name, pred, obs))
    both = {"north": [(262, 106), (262, 105)], "south": [(262, 108), (262, 109)]}
    for label, tiles in both.items():
        merge = {lane: chain_merge(tiles, lane) for lane in (1, 2)}
        out.append((f"side_both {label} feed", leading_lane(1, 0, [32], 64, merge), 1))
    # first probe: the `side` rig, feed x=132 y=150..147, fed at the back, lane 1 first
    merge = {lane: chain_merge([(132, y) for y in range(147, 151)], lane) for lane in (1, 2)}
    out.append(("probe 1 side", leading_lane(1, 0, [32, 64, 96], 128, merge), 1))
    # third probe
    p3 = _xz("logistics3-sideload.json.xz")
    for name, rec in sorted(p3.items()):
        if not isinstance(rec, dict) or "log" not in rec:
            continue
        seen, first_in = {}, []
        for t, row in rec["log"]:
            for bi, b in enumerate(row["a"]):
                for lane_items in b if b != "gone" else []:
                    for pos, uid in lane_items if isinstance(lane_items, list) else []:
                        if seen.get(uid) != bi:
                            seen[uid] = bi
                            if bi == 1:
                                first_in.append((t, pos))
        t0 = first_in[0][0]
        obs = _extra_lane([p for t, p in first_in if t == t0], 1)
        if name.startswith("ins"):
            feed, order = int(name[5]), name[-2:]
            tiles = [(201, 200 + k) for k in range(1, feed + 1)]
            merge = {lane: chain_merge(tiles, lane) for lane in (1, 2)}
            # the west inserter drops on lane 1; the first built drops first (t=47)
            first = 1 if order == "WE" else 2
            pred = leading_lane(first, 47, [63 + 32 * j for j in range(feed - 1)],
                                63 + 32 * (feed - 1), merge)  # fmt: skip
        else:
            feed, where = (int(v) for v in re.search(r"f(\d)w(\d)", name).groups())
            first = int(name[-2])
            at = re.search(r"at(\d+)", name)
            ox, oy = (int(at.group(1)), 113) if at else (300, 300)
            if name.startswith("old"):
                pred = 3 - first  # merged long before: no crossing activates anything
            else:
                tiles = [(ox + 1, oy + k) for k in range(1, feed + 1)]
                if (ox, oy) == (183, 113):
                    merge = {1: 85, 2: 42}  # logistics3-adhoc.json.xz, merge_timeline_expL "same"
                else:
                    merge = {lane: chain_merge(tiles, lane) for lane in (1, 2)}
                pred = leading_lane(first, 0, [16 + 32 * j for j in range(where - 1)],
                                    16 + 32 * (where - 1), merge)  # fmt: skip
        out.append((name, pred, obs))
    # the parity trace: drops at t=47 on the feed's third belt, west inserter built first
    tiles = [(11, y) for y in (-7, -6, -5, -4)]
    merge = {lane: chain_merge(tiles, lane) for lane in (1, 2)}
    with lzma.open(EVIDENCE / "sim-parity" / "logistics_sideload_merge.ticks.jsonl.xz", "rt") as f:
        f.readline()
        for line in f:
            r = json.loads(line)
            if r["tick"] == 127:
                belt = [e for e in r["entities"] if e["position"] == [2944, -1920]]
                lane2 = next(e["lanes"][1] for e in belt if e["name"] == "transport-belt")
                break
    obs = _extra_lane([p for _, p, _ in lane2], 1)
    out.append(("logistics_sideload_merge t=127", leading_lane(1, 47, [63, 95], 127, merge), obs))
    return out


def young_wake_cases():
    """(rig, predicted, observed) wake records of inserter-wake.json's `young` group."""
    wake = json.loads((EVIDENCE / "inserter-wake.json").read_text(encoding="utf-8"))
    young = [r for r in wake["rigs"] if r["kind"] == "young"]
    out = []
    for rep, r in enumerate(young):
        x0, y = 300 + (rep % 3) * 12, 300 + (32 + rep // 3) * 5
        merge = min(D[(x0 + k, y, 2)] for k in range(1, 9))
        # Added at t=40 on belt 1; it reaches the pickup belt 5 in record 151.
        # A merge measured at tick m is in record m - 1 of that file.
        pred = max(40, min(merge - 1, 151))
        obs = next(c[0] for c in r["changes"] if c[0] >= 40 and c[1] == 2560)
        out.append((f"young {rep}", pred, obs))
    return out


def _split_tick(log: list, lane: int) -> int | None:
    """The first tick after the lane merged whole that it is no longer whole."""
    whole = None
    for t, (belts, _) in log:
        classes = {b[lane - 1] for b in belts}
        if whole is None:
            whole = next((c for c in classes if c.startswith("1,2,3,4,5,6")), None)
        elif whole not in classes:
            return t
    return None


def split_cases():
    """(case, predicted, observed) split ticks: first interaction + delay of the last belt."""
    seg = _xz("logistics3-segments.json.xz")
    out = []
    # the drill's first output is at t=242 (its ore patch loses one ore then)
    chain = seg["logistics_smelting_chain"]["log"]
    out.append(("smelting chain lane 1", 242 + D[(12, -4, 1)], _split_tick(chain, 1)))
    # both lanes of the pickup line split at their own delay after one event
    pick = seg["logistics_belt_pickup"]["log"]
    s2, s1 = _split_tick(pick, 2), _split_tick(pick, 1)
    out.append(("belt pickup lane 1 from lane 2", s2 - D[(30, -17, 2)] + D[(30, -17, 1)], s1))
    return out


def main() -> int:
    bad = 0
    for title, cases in (("sideload", sideload_cases()), ("young wake", young_wake_cases()),
                         ("splits", split_cases())):  # fmt: skip
        ok = sum(p == o for _, p, o in cases)
        print(f"{title}: {ok} of {len(cases)} as predicted")
        for name, p, o in cases:
            if p != o:
                bad += 1
                print(f"  {name}: predicted {p}, engine {o}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
