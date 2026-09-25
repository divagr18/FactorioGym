"""Measure the hand-mining cases the first probe left open (v3 decision 1, follow-up).

`tools/probe_handmine.py` measured hand-mining and reach and left four
questions (docs/sim-logistics.md, "Hand-mining and reach"). This probe
measures them, one evidence file per family
(`docs/evidence/handmine2-<family>.json.xz`), in the same style: raw Lua,
rigs run one after another in one world, each on freshly painted grass, the
character driven the way the mod drives it, and every tick whose reading
changed logged.

Families:

- `pile`: taking up a ground pile with only partial room: how much of it
  comes, what stays, and whether mining goes on.
- `entity`: mining a chest, furnace, drill, belt or inserter whose contents
  do not all fit: what comes, what is left or dropped where, and whether the
  entity is mined at all.
- `carry`: mining while a belt carries the character: whether it is carried,
  whether mining goes on, and what happens once it is out of reach.
- `cover`: whether two entities can cover one tile centre, which is what
  selection would have to choose between. A pile left by a full inventory
  lies exactly on the tile centre, and building on the tile afterwards is
  allowed; spills next to piles already on the ground, drops from drills and
  inserters next to a centred pile, and two piles by script close together.
- `spill`: where dropped items land -- the ring order around the drop point,
  the second ring, what blocks a position -- and contents that only partly
  fit in the inventory.
- `beltpick`: a belt built over items on the ground takes them onto its lanes.
- `select`: which of two piles over one tile centre is mined first.
- `reenter`: carried out of reach while mining, and back.
- `reach`: how far away an entity or a pile is still mined, not carried.
- `select2`: two piles over one tile centre, nearer by straight line or per
  axis.
- `extra`: what `entity` left open -- a furnace mid-craft, a jammed drill's
  pending ore, a wall's own item, two piles close on one lane of a new belt.
- `beltpick2`: several piles on one tile under a new belt: the order they go
  on, and how one pushes another along its lane.
- `beltpick3`: two piles on one lane of a new belt: the spacing at which the
  second is still taken, and past the lane's end.

Ops, at rig-relative ticks (0 builds the rig, before its first reading):

- `mine` x y: the mod's `POLLS.mine` for a resource tile -- every tick select
  at the tile centre and set `mining_state` there;
- `mine_entity` label: the same for an entity, and, as the poller does, stop
  (`mining_state` false) on the first tick the entity is gone;
- `stop`, `walk` direction, `halt`: as in the first probe;
- `place` label name x y direction: the mod's `place` -- `can_place_entity`
  with `build_check_type.manual`, then `create_entity`, then one of the item
  taken from the main inventory (a refusal is logged in `notes`);
- `slot` index name count: set (count 0: clear) one main-inventory slot,
  standing in for a `give_to` or `take_from` elsewhere;
- `rotate` label: `LuaEntity.rotate`, as the mod's `rotate` does.

Every reading has: `p` position, `m` mining, `g` mining progress, `w`
walking, `inv` counts of `TRACKED`, `e` empty main slots, `a` resource
amounts (-1 gone), `sel` [name, x, y] of the selection (1/256, from the rig
base), `piles` [x, y, item, count], `ents` per labelled entity [name, x, y]
or "gone", `lanes` per labelled belt its two lanes as [position, item],
`inv_of` per labelled entity its non-empty inventories.

Run (about a minute each), for each family above:
  uv run python tools/probe_handmine2.py --family pile
"""

from __future__ import annotations

import argparse
import json
import lzma
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs" / "evidence"
sys.path.insert(0, str(ROOT / "tools"))

from probe_handmine import PRE, _lua_value, _world  # noqa: E402

TRACKED = (
    "iron-ore",
    "copper-ore",
    "coal",
    "stone",
    "wood",
    "iron-plate",
    "burner-mining-drill",
    "stone-furnace",
    "transport-belt",
    "burner-inserter",
    "wooden-chest",
    "stone-wall",
)

N, E, S, W = 0, 4, 8, 12
WOOD = ["wood", 100]


class Rig:
    """`base` tile; resources `(dx, dy, name, amount)`; entities with contents;
    the character's start and main-inventory slots; timed ops."""

    def __init__(self, name, base, ticks, start=(0.5, 0.5), slots=()):
        self.name, self.base, self.ticks, self.start = name, base, ticks, start
        self.slots = [list(s) for s in slots]
        self.resources: list[list] = []
        self.entities: list[dict] = []
        self.ops: list[list] = []

    def res(self, dx, dy, name="iron-ore", amount=50):
        self.resources.append([dx, dy, name, amount])
        return self

    def ent(self, label, name, x, y, d=0, **contents):
        self.entities.append({"label": label, "name": name, "x": x, "y": y, "d": d,
                              "contents": contents})  # fmt: skip
        return self

    def op(self, t, *args):
        self.ops.append([t, *args])
        return self

    def to_dict(self):
        return {"name": self.name, "base": list(self.base), "ticks": self.ticks,
                "start": list(self.start), "slots": self.slots, "resources": self.resources,
                "entities": self.entities, "ops": self.ops}  # fmt: skip


def wood(n):
    return [WOOD] * n


def _grid(k, step=40, per_row=8, origin=(0, 0)):
    return (origin[0] + step * (k % per_row), origin[1] + step * (k // per_row))


# ---------------------------------------------------------------- rigs


def pile_rigs() -> list[Rig]:
    """Partial room: a pile bigger than the room for its item."""
    specs = [
        # name, slots, pile item, pile count
        ("room3_of_10", [*wood(79), ["iron-plate", 97]], "iron-plate", 10),
        ("room10_of_10", [*wood(79), ["iron-plate", 90]], "iron-plate", 10),
        ("room0_of_10", [*wood(79), ["iron-plate", 100]], "iron-plate", 10),
        ("new_slot_of_10", [*wood(78), ["iron-plate", 97]], "iron-plate", 10),
        ("room10_of_30_ore", [*wood(79), ["iron-ore", 40]], "iron-ore", 30),
        ("room1_of_1_ore", [*wood(79), ["iron-ore", 49]], "iron-ore", 1),
        ("room0_of_1_ore", [*wood(79), ["iron-ore", 50]], "iron-ore", 1),
    ]
    rigs = []
    for k, (name, slots, item, count) in enumerate(specs):
        r = Rig(f"pile_{name}", _grid(k), 120, slots=slots)
        r.ent("pile", "item-on-ground", 2.5, 0.5, stack=[item, count])
        r.op(1, "mine_entity", "pile").op(100, "stop")
        rigs.append(r)
    # Room made while the partial pile is being mined: 3 of 10 first, then the
    # plate stack emptied at t=40.
    r = Rig("pile_room_made_midway", _grid(len(specs)), 160,
            slots=[*wood(79), ["iron-plate", 97]])  # fmt: skip
    r.ent("pile", "item-on-ground", 2.5, 0.5, stack=["iron-plate", 10])
    r.op(1, "mine_entity", "pile").op(40, "slot", 80, "iron-plate", 50).op(140, "stop")
    rigs.append(r)
    # A partial pile over an ore tile, mined through the tile (the mod's
    # resource poller): room 3, then the pile's rest.
    r = Rig("pile_partial_over_ore", _grid(len(specs) + 1), 400,
            slots=[*wood(79), ["iron-plate", 97]])  # fmt: skip
    r.res(2, 0)
    r.ent("pile", "item-on-ground", 2.5, 0.5, stack=["iron-plate", 10])
    r.op(1, "mine", 2, 0).op(60, "slot", 80, "", 0).op(380, "stop")
    rigs.append(r)
    # The largest pile a script can make: 60 ore (stack size 50).
    r = Rig("pile_over_stack_size", _grid(len(specs) + 2), 30, slots=[])
    r.ent("pile", "item-on-ground", 2.5, 0.5, stack=["iron-ore", 60])
    rigs.append(r)
    return rigs


def entity_rigs() -> list[Rig]:
    """An entity whose contents do not all fit in the main inventory."""
    one_free = wood(79)
    chest3 = [["iron-plate", 10], ["coal", 10], ["stone", 10]]
    specs = [
        ("chest_20_one_free", one_free, ("wooden-chest", 2.5, 0.5, 0),
         {"chest": [["iron-plate", 20]]}),
        ("chest_3kinds_one_free", one_free, ("wooden-chest", 2.5, 0.5, 0), {"chest": chest3}),
        ("chest_3kinds_room", wood(76), ("wooden-chest", 2.5, 0.5, 0), {"chest": chest3}),
        ("chest_20_stack_and_one_free", [*wood(78), ["iron-plate", 95]],
         ("wooden-chest", 2.5, 0.5, 0), {"chest": [["iron-plate", 20]]}),
        ("chest_20_full", wood(80), ("wooden-chest", 2.5, 0.5, 0),
         {"chest": [["iron-plate", 20]]}),
        ("chest_empty_full", wood(80), ("wooden-chest", 2.5, 0.5, 0), {}),
        ("chest_20_room5_no_free", [*wood(79), ["iron-plate", 95]],
         ("wooden-chest", 2.5, 0.5, 0), {"chest": [["iron-plate", 20]]}),
        ("furnace_one_free", one_free, ("stone-furnace", 3, 1, 0),
         {"fuel": [["coal", 5]], "source": [["iron-ore", 5]], "result": [["iron-plate", 5]]}),
        ("furnace_room", wood(76), ("stone-furnace", 3, 1, 0),
         {"fuel": [["coal", 5]], "source": [["iron-ore", 5]], "result": [["iron-plate", 5]]}),
        ("drill_one_free", one_free, ("burner-mining-drill", 3, 1, S), {"fuel": [["coal", 5]]}),
        ("drill_stack_and_one_free", [*wood(78), ["burner-mining-drill", 1]],
         ("burner-mining-drill", 3, 1, S), {"fuel": [["coal", 5]]}),
        ("belt_one_free", one_free, ("transport-belt", 2.5, 0.5, E),
         {"lanes": [[1, 64, "iron-plate"], [1, 192, "iron-plate"], [2, 64, "coal"],
                    [2, 192, "coal"]]}),
        ("inserter_one_free", one_free, ("burner-inserter", 2.5, 0.5, N),
         {"fuel": [["coal", 2]], "held": ["iron-plate", 1]}),
        ("inserter_room", wood(76), ("burner-inserter", 2.5, 0.5, N),
         {"fuel": [["coal", 2]], "held": ["iron-plate", 1]}),
    ]  # fmt: skip
    rigs = []
    for k, (name, slots, (ename, x, y, d), contents) in enumerate(specs):
        r = Rig(f"entity_{name}", _grid(k), 120, slots=slots)
        if ename in ("stone-furnace", "burner-mining-drill"):
            for dx in (2, 3):
                for dy in (0, 1):
                    r.res(dx, dy)
        r.ent("target", ename, x, y, d, **contents)
        r.op(1, "mine_entity", "target").op(100, "stop")
        rigs.append(r)
    return rigs


def carry_rigs() -> list[Rig]:
    """Mining while a belt carries the character."""
    rigs = []
    # Carried east along y=0, the ore two tiles north of the start: in reach
    # for about 58 ticks, then not.
    r = Rig("carry_away_ore", _grid(0), 400, start=(0.5, 0.5))
    for x in range(-1, 12):
        r.ent(f"b{x}", "transport-belt", x + 0.5, 0.5, E)
    r.res(0, -2)
    r.op(1, "mine", 0, -2).op(380, "stop")
    rigs.append(r)
    # The same with no mining: the carriage alone.
    r = Rig("carry_alone", _grid(1), 200, start=(0.5, 0.5))
    for x in range(-1, 12):
        r.ent(f"b{x}", "transport-belt", x + 0.5, 0.5, E)
    rigs.append(r)
    # Carried west past the ore: nearer, then away.
    r = Rig("carry_past_ore", _grid(2), 400, start=(6.5, 0.5))
    for x in range(-3, 9):
        r.ent(f"b{x}", "transport-belt", x + 0.5, 0.5, W)
    r.res(4, -2)
    r.op(1, "mine", 4, -2).op(380, "stop")
    rigs.append(r)
    # Mining an entity while carried: a drill (0.3 s) north of the belt.
    r = Rig("carry_mine_drill", _grid(3), 200, start=(0.5, 0.5))
    for x in range(-1, 12):
        r.ent(f"b{x}", "transport-belt", x + 0.5, 0.5, E)
    r.ent("drill", "burner-mining-drill", 2, -2, S)
    r.op(1, "mine_entity", "drill").op(180, "stop")
    rigs.append(r)
    # Mining from a standstill, then a belt built under the character mid-mine.
    r = Rig("belt_built_under_mid_mine", _grid(4), 400, start=(0.5, 0.5),
            slots=[["transport-belt", 5]])  # fmt: skip
    r.res(0, -2)
    r.op(1, "mine", 0, -2).op(30, "place", "belt", "transport-belt", 0.5, 0.5, E)
    r.op(380, "stop")
    rigs.append(r)
    return rigs


def cover_rigs() -> list[Rig]:
    """Two entities over one tile centre: how, and which is mined."""
    rigs = []
    full = [*wood(79), ["iron-ore", 50]]
    ore4 = [(2, 0), (3, 0), (2, 1), (3, 1)]
    builds = [
        ("wooden-chest", 2.5, 0.5, 0),
        ("transport-belt", 2.5, 0.5, E),
        ("burner-inserter", 2.5, 0.5, N),
        ("stone-wall", 2.5, 0.5, 0),
        ("stone-furnace", 3, 1, 0),
        ("burner-mining-drill", 3, 1, S),
    ]
    # A full inventory spills an ore at the tile centre; then a slot is freed,
    # the entity is built on the tile over the pile, and the tile is mined
    # three times, stopping in between.
    for k, (name, x, y, d) in enumerate(builds):
        r = Rig(f"spill_then_build_{name}", _grid(k), 460, slots=full)
        for dx, dy in ore4:
            r.res(dx, dy)
        r.op(1, "mine", 2, 0).op(130, "stop")
        r.op(131, "slot", 79, name, 1).op(132, "place", "built", name, x, y, d)
        r.op(140, "mine", 2, 0).op(240, "stop").op(241, "mine", 2, 0).op(340, "stop")
        r.op(341, "mine", 2, 0).op(440, "stop")
        rigs.append(r)
    n = len(builds)
    # Two piles by script 0.28 apart, both within 0.168 of the tile centre.
    r = Rig("two_piles_script", _grid(n), 360)
    r.res(2, 0)
    r.ent("p1", "item-on-ground", 2.36, 0.5, stack=["iron-plate", 1])
    r.ent("p2", "item-on-ground", 2.64, 0.5, stack=["coal", 1])
    r.op(1, "mine", 2, 0).op(100, "stop").op(101, "mine", 2, 0).op(200, "stop")
    r.op(201, "mine", 2, 0).op(340, "stop")
    rigs.append(r)
    # A full inventory mining a tile next to a pile already on the ground,
    # where a drill (0.203 off the centre) or an inserter (0.199) drops it.
    for k, (label, px, py) in enumerate((("drill_drop", 2.5, 0.296875),
                                         ("inserter_drop", 2.69921875, 0.5))):  # fmt: skip
        r = Rig(f"spill_next_to_{label}", _grid(n + 1 + k), 700, slots=full)
        r.res(2, 0)
        r.ent("pile", "item-on-ground", px, py, stack=["iron-plate", 1])
        r.op(1, "mine", 2, 0).op(680, "stop")
        rigs.append(r)
    # An inserter and a drill dropping onto the ground at a tile whose centre
    # already holds a pile.
    r = Rig("inserter_drop_by_centred_pile", _grid(n + 3), 300)
    r.ent("src", "wooden-chest", -0.5, 0.5, 0, chest=[["iron-plate", 10]])
    r.ent("ins", "burner-inserter", 0.5, 0.5, W, fuel=[["coal", 2]])
    r.ent("pile", "item-on-ground", 1.5, 0.5, stack=["coal", 1])
    rigs.append(r)
    r = Rig("drill_drop_by_centred_pile", _grid(n + 4), 600)
    for dx, dy in ((-1, -1), (0, -1), (-1, 0), (0, 0)):
        r.res(dx, dy)
    r.ent("drill", "burner-mining-drill", 0, 0, E, fuel=[["coal", 5]])
    r.ent("pile", "item-on-ground", 1.5, -0.5, stack=["coal", 1])
    rigs.append(r)
    # And with no pile there, for the drop points themselves.
    r = Rig("inserter_drop_bare", _grid(n + 5), 300)
    r.ent("src", "wooden-chest", -0.5, 0.5, 0, chest=[["iron-plate", 10]])
    r.ent("ins", "burner-inserter", 0.5, 0.5, W, fuel=[["coal", 2]])
    rigs.append(r)
    r = Rig("drill_drop_bare", _grid(n + 6), 600)
    for dx, dy in ((-1, -1), (0, -1), (-1, 0), (0, 0)):
        r.res(dx, dy)
    r.ent("drill", "burner-mining-drill", 0, 0, E, fuel=[["coal", 5]])
    rigs.append(r)
    return rigs


RING1 = [(-88, -88), (0, -88), (88, -88), (88, 0), (88, 88), (0, 88), (-88, 88), (-88, 0)]


def spill_rigs() -> list[Rig]:
    """Where dropped items land: the ring order, the second ring, and what
    blocks a position (other piles, a wall, a chest, a belt)."""
    rigs = []
    full = [*wood(79), ["iron-ore", 50]]

    def ring2(name, k, extra=()):
        # A pile 52/256 north of the centre keeps the centre free for mining
        # (it does not cover it) but blocks a drop there; ring 1 is filled by
        # script, so every ore mined lands in ring 2 or beyond.
        r = Rig(name, _grid(k), 2500, slots=full)
        r.res(2, 0)
        r.ent("blocker", "item-on-ground", 2.5, 0.5 - 52 / 256, stack=["iron-plate", 1])
        for j, (dx, dy) in enumerate(RING1):
            if (dx, dy) == (0, -88):
                continue  # overlaps the blocker
            r.ent(f"r{j}", "item-on-ground", 2.5 + dx / 256, 0.5 + dy / 256,
                  stack=["iron-plate", 1])  # fmt: skip
        for label, name_, x, y, d in extra:
            r.ent(label, name_, x, y, d)
        r.op(1, "mine", 2, 0).op(2480, "stop")
        return r

    rigs.append(ring2("ring2_bare", 0))
    rigs.append(ring2("ring2_wall_east", 1, [("wall", "stone-wall", 3.5, 0.5, 0)]))
    rigs.append(ring2("ring2_chest_north", 2, [("chest", "wooden-chest", 2.5, -0.5, 0)]))
    rigs.append(ring2("ring2_belt_west", 3, [("belt", "transport-belt", 1.5, 0.5, N)]))
    # A belt with four items a lane mined with no room at all: nine drops
    # around the belt's own position.
    r = Rig("belt_nine_drops", _grid(4), 60, slots=wood(80))
    lanes = [[lane, pos, "iron-plate"] for lane in (1, 2) for pos in (32, 96, 160, 224)]
    r.ent("target", "transport-belt", 2.5, 0.5, E, lanes=lanes)
    r.op(1, "mine_entity", "target").op(50, "stop")
    rigs.append(r)
    # The same belt, with one free slot: which lane item goes in, which drop.
    r = Rig("belt_one_free_mixed", _grid(5), 60, slots=wood(79))
    lanes = [[1, 32, "coal"], [1, 160, "iron-plate"], [2, 96, "stone"], [2, 224, "coal"]]
    r.ent("target", "transport-belt", 2.5, 0.5, E, lanes=lanes)
    r.op(1, "mine_entity", "target").op(50, "stop")
    rigs.append(r)
    # A furnace's own item dropped: at its centre (a tile corner).
    r = Rig("furnace_item_drop", _grid(6), 60, slots=wood(80))
    r.ent("target", "stone-furnace", 3, 1, 0)
    r.op(1, "mine_entity", "target").op(50, "stop")
    rigs.append(r)
    # Contents that only partly fit: which goes first, and whether a stack
    # that does not fit is skipped for one that does.
    furnace = {"fuel": [["coal", 5]], "source": [["iron-ore", 5]],
               "result": [["iron-plate", 5]]}  # fmt: skip
    for k, (name, slots, ename, x, y, d, contents) in enumerate(
        [
            ("furnace_room_plates", [*wood(79), ["iron-plate", 95]], "stone-furnace", 3, 1, 0,
             furnace),
            ("furnace_room_ore", [*wood(79), ["iron-ore", 45]], "stone-furnace", 3, 1, 0,
             furnace),
            ("furnace_room_coal", [*wood(79), ["coal", 45]], "stone-furnace", 3, 1, 0, furnace),
            ("chest_skip_to_plates", [*wood(79), ["iron-plate", 95]], "wooden-chest", 2.5, 0.5,
             0, {"chest": [["coal", 10], ["iron-plate", 10]]}),
            ("drill_fuel_no_room", [*wood(79), ["coal", 50]], "burner-mining-drill", 3, 1, S,
             {"fuel": [["coal", 5]]}),
            ("drill_fuel_room2", [*wood(79), ["coal", 48]], "burner-mining-drill", 3, 1, S,
             {"fuel": [["coal", 5]]}),
            ("inserter_no_room", wood(80), "burner-inserter", 2.5, 0.5, N,
             {"fuel": [["coal", 2]], "held": ["iron-plate", 1]}),
            ("inserter_room_fuel_only", [*wood(79), ["coal", 48]], "burner-inserter", 2.5, 0.5,
             N, {"fuel": [["coal", 2]], "held": ["iron-plate", 1]}),
        ]
    ):  # fmt: skip
        r = Rig(f"partial_{name}", _grid(7 + k), 160, slots=slots)
        if ename in ("stone-furnace", "burner-mining-drill"):
            for dx in (2, 3):
                for dy in (0, 1):
                    r.res(dx, dy)
        r.ent("target", ename, x, y, d, **contents)
        r.op(1, "mine_entity", "target").op(150, "stop")
        rigs.append(r)
    return rigs


def beltpick_rigs() -> list[Rig]:
    """A belt built over items on the ground: what it takes, onto which lane, where."""
    rigs = []
    k = 0
    for d, dname in ((N, "n"), (E, "e"), (S, "s"), (W, "w")):
        r = Rig(f"bp_centre_{dname}", _grid(k), 60, slots=[["transport-belt", 5]])
        r.ent("pile", "item-on-ground", 2.5, 0.5, stack=["iron-plate", 1])
        r.op(1, "place", "belt", "transport-belt", 2.5, 0.5, d)
        rigs.append(r)
        k += 1
    offsets = [(0, -100), (0, -51), (0, 51), (0, 100), (-100, 0), (100, 0), (-51, -52),
               (0, -130), (0, -140), (60, 60)]  # fmt: skip
    for dx, dy in offsets:
        r = Rig(f"bp_off_{dx}_{dy}", _grid(k), 60, slots=[["transport-belt", 5]])
        r.ent("pile", "item-on-ground", 2.5 + dx / 256, 0.5 + dy / 256, stack=["coal", 1])
        r.op(1, "place", "belt", "transport-belt", 2.5, 0.5, E)
        rigs.append(r)
        k += 1
    r = Rig("bp_two", _grid(k), 60, slots=[["transport-belt", 5]])
    r.ent("p1", "item-on-ground", 2.5 - 60 / 256, 0.5, stack=["coal", 1])
    r.ent("p2", "item-on-ground", 2.5 + 60 / 256, 0.5, stack=["iron-plate", 1])
    r.op(1, "place", "belt", "transport-belt", 2.5, 0.5, E)
    rigs.append(r)
    k += 1
    # A drill jammed on its own pile, then a belt built on its drop tile.
    r = Rig("bp_drill_jam", _grid(k), 700, slots=[["transport-belt", 5]])
    for dx, dy in ((2, 0), (3, 0), (2, 1), (3, 1)):
        r.res(dx, dy)
    r.ent("drill", "burner-mining-drill", 3, 1, S, fuel=[["coal", 5]])
    r.op(300, "place", "belt", "transport-belt", 3.5, 2.5, E)
    rigs.append(r)
    return rigs


def select_rigs() -> list[Rig]:
    """Which of two piles over one tile centre the mod's resource mine selects."""
    rigs = []
    specs = [
        ("sym_ab", [("a", -36, "iron-plate"), ("b", 36, "coal")]),
        ("sym_ba", [("b", 36, "coal"), ("a", -36, "iron-plate")]),
        ("near_first", [("a", -20, "iron-plate"), ("b", 40, "coal")]),
        ("near_second", [("b", 40, "coal"), ("a", -20, "iron-plate")]),
        ("vertical", [("a", 0, "iron-plate", -36), ("b", 0, "coal", 36)]),
    ]
    for k, (name, piles) in enumerate(specs):
        r = Rig(f"sel_{name}", _grid(k), 120)
        r.res(2, 0)
        for spec in piles:
            label, dx, item = spec[:3]
            dy = spec[3] if len(spec) > 3 else 0
            r.ent(label, "item-on-ground", 2.5 + dx / 256, 0.5 + dy / 256, stack=[item, 1])
        r.op(1, "mine", 2, 0).op(100, "stop")
        rigs.append(r)
    return rigs


def reenter_rigs() -> list[Rig]:
    """Carried out of reach while mining, and back into it."""
    rigs = []
    r = Rig("reenter_resource", _grid(0), 600, start=(0.5, 0.5))
    for x in range(-8, 12):
        r.ent(f"b{x}", "transport-belt", x + 0.5, 0.5, E)
    r.res(0, -2)
    r.op(1, "mine", 0, -2)
    for x in range(-8, 12):  # turn every belt round at t=100
        r.op(100, "rotate", f"b{x}").op(100, "rotate", f"b{x}")
    r.op(580, "stop")
    rigs.append(r)
    # An entity at the edge of reach, carried out of it mid-mine.
    r = Rig("carry_out_of_entity_reach", _grid(1), 200, start=(0.5, 0.5))
    for x in range(-1, 12):
        r.ent(f"b{x}", "transport-belt", x + 0.5, 0.5, E)
    r.ent("drill", "burner-mining-drill", -10, 1, S)
    r.op(1, "mine_entity", "drill").op(180, "stop")
    rigs.append(r)
    return rigs


def reach_mine_rigs() -> list[Rig]:
    """How far away an entity or a pile is still mined (not carried)."""
    rigs = []
    k = 0
    # Offsets of the entity's centre east of the character, which stands at
    # (0.5, 0.5): centre distances 9.5 .. 10.75, box distances less by the
    # collision half-size (chest 89/256, drill 179/256, pile 35/256).
    for name, y in (("wooden-chest", 0.5), ("burner-mining-drill", 1.0), ("item-on-ground", 0.5)):
        for off in (9.5, 9.75, 9.875, 10.0, 10.125, 10.25, 10.5, 10.75):
            x = 0.5 + off
            r = Rig(f"mreach_{name}_{off}", _grid(k), 80)
            if name == "item-on-ground":
                r.ent("target", name, x, y, stack=["iron-plate", 1])
            else:
                r.ent("target", name, x, y)
            r.op(1, "mine_entity", "target").op(70, "stop")
            rigs.append(r)
            k += 1
    # A resource: centre distance 2.7 .. 2.85 east.
    for off in (2.6875, 2.703125, 2.75, 2.78125, 2.8125, 2.84375):
        r = Rig(f"mreach_ore_{off}", _grid(k), 140, start=(2.5 - off, 0.5))
        r.res(2, 0)
        r.op(1, "mine", 2, 0).op(130, "stop")
        rigs.append(r)
        k += 1
    return rigs


def select2_rigs() -> list[Rig]:
    """Two piles over one tile centre: is nearer Euclidean or per axis?"""
    rigs = []
    for k, order in enumerate((("a", "b"), ("b", "a"))):
        piles = {"a": (30, 30, "iron-plate"), "b": (0, 38, "coal")}
        r = Rig(f"sel_metric_{''.join(order)}", _grid(k), 60)
        r.res(2, 0)
        for label in order:
            dx, dy, item = piles[label]
            r.ent(label, "item-on-ground", 2.5 + dx / 256, 0.5 + dy / 256, stack=[item, 1])
        r.op(1, "mine", 2, 0).op(50, "stop")
        rigs.append(r)
    # The character's own box next to the drop positions: ring 1 around the
    # ore, the character standing 141/256 south of its centre.
    r = Rig("spill_by_character", _grid(2), 1100, start=(2.5, 0.5 + 141 / 256),
            slots=[*wood(79), ["iron-ore", 50]])  # fmt: skip
    r.res(2, 0)
    r.ent("blocker", "item-on-ground", 2.5, 0.5 - 52 / 256, stack=["iron-plate", 1])
    r.op(1, "mine", 2, 0).op(1080, "stop")
    rigs.append(r)
    # Ring 2 next to a furnace, a drill and an inserter (ring 1 filled by
    # script, the character well south).
    for k, (name, x, y, d) in enumerate((("stone-furnace", 5, 1, 0),
                                         ("burner-mining-drill", 5, 1, S),
                                         ("burner-inserter", 4.5, 0.5, N))):  # fmt: skip
        r = Rig(f"spill_by_{name}", _grid(3 + k), 2100, start=(3.5, 2.625),
                slots=[*wood(79), ["iron-ore", 50]])  # fmt: skip
        r.res(3, 0)
        r.ent("blocker", "item-on-ground", 3.5, 0.5 - 52 / 256, stack=["iron-plate", 1])
        for j, (dx, dy) in enumerate(RING1):
            if (dx, dy) != (0, -88):
                r.ent(f"r{j}", "item-on-ground", 3.5 + dx / 256, 0.5 + dy / 256,
                      stack=["iron-plate", 1])  # fmt: skip
        r.ent("thing", name, x, y, d)
        r.op(1, "mine", 3, 0).op(2080, "stop")
        rigs.append(r)
    # Furnace contents order: room for coal and plates, not ore; coal and ore,
    # not plates.
    furnace = {"fuel": [["coal", 5]], "source": [["iron-ore", 5]],
               "result": [["iron-plate", 5]]}  # fmt: skip
    for k, (name, slots) in enumerate(
        (("furnace_coal_plates", [*wood(78), ["coal", 45], ["iron-plate", 95]]),
         ("furnace_coal_ore", [*wood(78), ["coal", 45], ["iron-ore", 45]]))
    ):  # fmt: skip
        r = Rig(f"partial_{name}", _grid(6 + k), 100, slots=slots)
        for dx in (2, 3):
            for dy in (0, 1):
                r.res(dx, dy)
        r.ent("target", "stone-furnace", 3, 1, 0, **furnace)
        r.op(1, "mine_entity", "target").op(90, "stop")
        rigs.append(r)
    return rigs


def extra_rigs() -> list[Rig]:
    """What the entity family left open: a furnace mid-craft, a jammed drill's
    pending ore, a wall's own item, two piles close on one lane of a new belt."""
    rigs = []
    ore4 = ((2, 0), (3, 0), (2, 1), (3, 1))
    r = Rig("furnace_ingredient_no_room", _grid(0), 120, slots=[*wood(79), ["coal", 45]])
    for dx, dy in ore4:
        r.res(dx, dy)
    r.ent("target", "stone-furnace", 3, 1, 0, fuel=[["coal", 5]], source=[["iron-ore", 1]])
    r.op(20, "mine_entity", "target").op(110, "stop")
    rigs.append(r)
    r = Rig("furnace_ingredient_room", _grid(1), 120, slots=wood(76))
    for dx, dy in ore4:
        r.res(dx, dy)
    r.ent("target", "stone-furnace", 3, 1, 0, fuel=[["coal", 5]], source=[["iron-ore", 1]])
    r.op(20, "mine_entity", "target").op(110, "stop")
    rigs.append(r)
    for k, slots in enumerate((wood(76), [*wood(79), ["coal", 46]])):
        r = Rig(f"drill_pending_ore_{'room' if k == 0 else 'no_room'}", _grid(2 + k), 620,
                slots=slots)  # fmt: skip
        for dx, dy in ore4:
            r.res(dx, dy)
        r.ent("target", "burner-mining-drill", 3, 1, S, fuel=[["coal", 5]])
        r.op(560, "mine_entity", "target").op(610, "stop")
        rigs.append(r)
    r = Rig("wall_item_no_room", _grid(4), 60, slots=wood(80))
    r.ent("target", "stone-wall", 2.5, 0.5, 0)
    r.op(1, "mine_entity", "target").op(50, "stop")
    rigs.append(r)
    r = Rig("bp_two_close_one_lane", _grid(5), 60, slots=[["transport-belt", 5]])
    r.ent("p1", "item-on-ground", 2.5 - 20 / 256, 0.5, stack=["coal", 1])
    r.ent("p2", "item-on-ground", 2.5 + 20 / 256, 0.5 + 72 / 256, stack=["iron-plate", 1])
    r.op(1, "place", "belt", "transport-belt", 2.5, 0.5, E)
    rigs.append(r)
    r = Rig("bp_two_close_one_lane_b", _grid(6), 60, slots=[["transport-belt", 5]])
    r.ent("p1", "item-on-ground", 2.5 - 20 / 256, 0.5 + 72 / 256, stack=["coal", 1])
    r.ent("p2", "item-on-ground", 2.5 + 20 / 256, 0.5, stack=["iron-plate", 1])
    r.op(1, "place", "belt", "transport-belt", 2.5, 0.5, E)
    rigs.append(r)
    # Two piles 40 apart along one lane, on belts running west, north and
    # south, created downstream-first and upstream-first: which goes on first,
    # and so which is pushed back.
    k = 7
    for d, dname, (ux, uy) in ((W, "w", (-1, 0)), (N, "n", (0, -1)), (S, "s", (0, 1))):
        rx, ry = -uy, ux  # right of travel
        down = (20 * ux + 72 * rx, 20 * uy + 72 * ry)
        up = (-20 * ux + 72 * rx, -20 * uy + 72 * ry)
        for order in ("downfirst", "upfirst"):
            r = Rig(f"bp_two_close_{dname}_{order}", _grid(k), 60, slots=[["transport-belt", 5]])
            first, second = (down, up) if order == "downfirst" else (up, down)
            r.ent("p1", "item-on-ground", 2.5 + first[0] / 256, 0.5 + first[1] / 256,
                  stack=["coal", 1])  # fmt: skip
            r.ent("p2", "item-on-ground", 2.5 + second[0] / 256, 0.5 + second[1] / 256,
                  stack=["iron-plate", 1])  # fmt: skip
            r.op(1, "place", "belt", "transport-belt", 2.5, 0.5, d)
            rigs.append(r)
            k += 1
    return rigs


def beltpick2_rigs() -> list[Rig]:
    """Several piles on one tile when a belt is built there: the order they go
    on, and how one pushes another along its lane."""
    rigs = []
    k = 0
    ring9 = [(0, 0), *RING1]

    def rig(name, piles, d):
        nonlocal k
        r = Rig(name, _grid(k), 40, slots=[["transport-belt", 5]])
        items = ("coal", "iron-plate", "stone", "iron-ore", "copper-ore", "wood")
        for j, (dx, dy) in enumerate(piles):
            r.ent(f"p{j}", "item-on-ground", 2.5 + dx / 256, 0.5 + dy / 256,
                  stack=[items[j % len(items)], 1])  # fmt: skip
        r.op(1, "place", "belt", "transport-belt", 2.5, 0.5, d)
        rigs.append(r)
        k += 1

    for d, dname in ((E, "e"), (W, "w"), (N, "n"), (S, "s")):
        rig(f"bp_ring9_{dname}", ring9, d)
    rig("bp_ring9_rev_e", list(reversed(ring9)), E)
    # Same along-position, both on lane 2 of an east belt.
    rig("bp_same_along_ab", [(0, 0), (0, 88)], E)
    rig("bp_same_along_ba", [(0, 88), (0, 0)], E)
    # Three on one lane of an east belt, 40 apart: every creation order.
    three = {"d": (40, 72), "m": (0, 72), "u": (-40, 72)}
    for order in ("dmu", "dum", "mdu", "mud", "udm", "umd"):
        rig(f"bp_three_{order}", [three[c] for c in order], E)
    return rigs


def beltpick3_rigs() -> list[Rig]:
    """Two piles on one lane of a new east belt: the spacing at which the
    second is still taken, and what happens past the lane's end."""
    rigs = []
    k = 0
    # (name, along offsets of the two piles; lane 2, lateral 0 and +72 so
    # they do not collide). Target = 128 - along.
    pairs = [
        ("gap64", 88, 24),    # targets 40, 104
        ("gap60", 88, 28),    # 40, 100
        ("gap70", 88, 18),    # 40, 110
        ("same216", -88, -88),  # 216, 216
        ("end", -72, -102),   # 200, 230
        ("gap64_back", -24, -88),  # 152, 216
    ]  # fmt: skip
    for name, a, b in pairs:
        for order in ("ab", "ba"):
            r = Rig(f"bp3_{name}_{order}", _grid(k), 40, slots=[["transport-belt", 5]])
            pa = ("pa", 2.5 + a / 256, 0.5, "coal")
            pb = ("pb", 2.5 + b / 256, 0.5 + 72 / 256, "iron-plate")
            for label, x, y, item in (pa, pb) if order == "ab" else (pb, pa):
                r.ent(label, "item-on-ground", x, y, stack=[item, 1])
            r.op(1, "place", "belt", "transport-belt", 2.5, 0.5, E)
            rigs.append(r)
            k += 1
    # Five and six piles on one lane, 40 apart (alternately 0 and 72/256 off
    # the centre line so no two collide): how many the lane takes.
    for n in (5, 6):
        for order in ("asc", "desc"):
            targets = [10 + 40 * j for j in range(n)]
            if order == "desc":
                targets = targets[::-1]
            r = Rig(f"bp4_n{n}_{order}", _grid(k), 40, slots=[["transport-belt", 5]])
            items = ("coal", "iron-plate", "stone", "iron-ore", "copper-ore", "wood")
            for j, t in enumerate(targets):
                lateral = 72 if (t // 40) % 2 else 0
                r.ent(f"p{j}", "item-on-ground", 2.5 + (128 - t) / 256, 0.5 + lateral / 256,
                      stack=[items[j], 1])  # fmt: skip
            r.op(1, "place", "belt", "transport-belt", 2.5, 0.5, E)
            rigs.append(r)
            k += 1
    return rigs


FAMILIES = {
    "pile": pile_rigs,
    "entity": entity_rigs,
    "carry": carry_rigs,
    "cover": cover_rigs,
    "spill": spill_rigs,
    "beltpick": beltpick_rigs,
    "select": select_rigs,
    "reenter": reenter_rigs,
    "reach": reach_mine_rigs,
    "select2": select2_rigs,
    "extra": extra_rigs,
    "beltpick2": beltpick2_rigs,
    "beltpick3": beltpick3_rigs,
}

# ---------------------------------------------------------------- engine side

RUN = r"""
storage.hm = {log = {}, failed = {}, notes = {}}
local P = storage.hm
local DIR = {north = defines.direction.north, east = defines.direction.east,
             south = defines.direction.south, west = defines.direction.west}
for _, rig in ipairs(RIGS) do
  paint(rig.base[1] - 10, rig.base[2] - 10, rig.base[1] + 16, rig.base[2] + 12)
end
local order, start = {}, 0
for k, rig in ipairs(RIGS) do
  order[k] = {rig = rig, t0 = start, byt = {}}
  for _, op in ipairs(rig.ops) do
    order[k].byt[op[1]] = order[k].byt[op[1]] or {}
    table.insert(order[k].byt[op[1]], op)
  end
  start = start + rig.ticks + 1
end
local cur = {k = 0, res = {}, E = {}, labels = {}, mine = nil, mine_e = nil, walk = nil}
local function note(...)
  P.notes[#P.notes + 1] = {RIGS[cur.k].name, cur.t, ...}
end
local function fill(e, spec)
  local c = spec.contents or {}
  if c.chest then
    local inv = e.get_inventory(defines.inventory.chest)
    for _, st in ipairs(c.chest) do inv.insert({name = st[1], count = st[2]}) end
  end
  if c.fuel then
    for _, st in ipairs(c.fuel) do e.get_fuel_inventory().insert({name = st[1], count = st[2]}) end
  end
  if c.source then
    local inv = e.get_inventory(defines.inventory.furnace_source)
    for _, st in ipairs(c.source) do inv.insert({name = st[1], count = st[2]}) end
  end
  if c.result then
    local inv = e.get_inventory(defines.inventory.furnace_result)
    for _, st in ipairs(c.result) do inv.insert({name = st[1], count = st[2]}) end
  end
  if c.held then e.held_stack.set_stack({name = c.held[1], count = c.held[2]}) end
  if c.lanes then
    for _, it in ipairs(c.lanes) do
      if not e.get_transport_line(it[1]).insert_at(it[2] / 256, {name = it[3], count = 1}) then
        note("lane insert refused", spec.label)
      end
    end
  end
end
local function setup(k)
  local rig = RIGS[k]
  local bx, by = rig.base[1], rig.base[2]
  ch.walking_state = {walking = false}
  ch.mining_state = {mining = false}
  ch.teleport({bx + rig.start[1], by + rig.start[2]})
  local inv = ch.get_main_inventory()
  inv.clear()
  for i, st in ipairs(rig.slots) do inv[i].set_stack({name = st[1], count = st[2]}) end
  cur.res, cur.E, cur.labels = {}, {}, {}
  for _, r in ipairs(rig.resources) do
    local e = s.create_entity({name = r[3], position = {bx + r[1] + 0.5, by + r[2] + 0.5},
                               amount = r[4]})
    cur.res[#cur.res + 1] = e
  end
  for _, spec in ipairs(rig.entities) do
    local e
    local pos = {bx + spec.x, by + spec.y}
    if spec.name == "item-on-ground" then
      local st = spec.contents.stack
      e = s.create_entity({name = "item-on-ground", position = pos,
                           stack = {name = st[1], count = st[2]}})
    else
      e = s.create_entity({name = spec.name, position = pos, direction = spec.d,
                           force = "player"})
      if e then fill(e, spec) end
    end
    if not e then P.failed[#P.failed + 1] = {rig.name, spec.label, spec.name, spec.x, spec.y} end
    cur.E[spec.label] = e
    cur.labels[#cur.labels + 1] = spec.label
  end
  cur.mine, cur.mine_e, cur.walk, cur.last = nil, nil, nil, nil
end
local function apply(k, op)
  local rig = RIGS[k]
  local bx, by = rig.base[1], rig.base[2]
  local kind = op[2]
  if kind == "mine" then
    cur.mine_e = nil
    cur.mine = {bx + op[3] + 0.5, by + op[4] + 0.5}
  elseif kind == "mine_entity" then
    cur.mine = nil
    cur.mine_e = op[3]
  elseif kind == "stop" then
    cur.mine, cur.mine_e = nil, nil
    ch.mining_state = {mining = false}
  elseif kind == "walk" then
    cur.walk = DIR[op[3]]
  elseif kind == "halt" then
    cur.walk = nil
    ch.walking_state = {walking = false}
  elseif kind == "slot" then
    local inv = ch.get_main_inventory()
    if op[5] == 0 then inv[op[3]].clear()
    else inv[op[3]].set_stack({name = op[4], count = op[5]}) end
  elseif kind == "rotate" then
    local e = cur.E[op[3]]
    if not (e and e.valid and e.rotate()) then note("rotate failed", op[3]) end
  elseif kind == "place" then
    local label, name = op[3], op[4]
    local spec = {name = name, position = {bx + op[5], by + op[6]}, direction = op[7],
                  force = ch.force, build_check_type = defines.build_check_type.manual}
    if s.can_place_entity(spec) then
      local e = s.create_entity({name = name, position = spec.position,
                                 direction = op[7], force = ch.force})
      if e then
        ch.get_main_inventory().remove({name = name, count = 1})
        cur.E[label] = e
        cur.labels[#cur.labels + 1] = label
        note("placed", label, math.floor(e.position.x * 256 + 0.5) - bx * 256,
             math.floor(e.position.y * 256 + 0.5) - by * 256)
      else
        note("create_entity failed", label)
      end
    else
      note("can_place_entity refused", label)
    end
  else
    error("op " .. kind)
  end
end
local function rel(pos, bx, by)
  return {math.floor(pos.x * 256 + 0.5) - bx * 256, math.floor(pos.y * 256 + 0.5) - by * 256}
end
local function line_items(line)
  local o = {}
  for _, it in pairs(line.get_detailed_contents()) do
    o[#o + 1] = {math.floor(it.position * 256 + 0.5), it.stack.name}
  end
  table.sort(o, function(a, b) return a[1] < b[1] end)
  return o
end
local INVS = {chest = defines.inventory.chest, fuel = defines.inventory.fuel,
              source = defines.inventory.furnace_source,
              result = defines.inventory.furnace_result}
local function reading()
  local rig = RIGS[cur.k]
  local bx, by = rig.base[1], rig.base[2]
  local inv = ch.get_main_inventory()
  local counts = {}
  for _, name in ipairs(TRACKED) do counts[#counts + 1] = inv.get_item_count(name) end
  local amounts = {}
  for i, e in ipairs(cur.res) do amounts[i] = (e and e.valid) and e.amount or -1 end
  local piles = {}
  for _, e in pairs(s.find_entities_filtered({area = {{bx - 10, by - 10}, {bx + 16, by + 12}},
                                             type = "item-entity"})) do
    local p = rel(e.position, bx, by)
    piles[#piles + 1] = {p[1], p[2], e.stack.name, e.stack.count}
  end
  table.sort(piles, function(a, b) return a[1] < b[1] or (a[1] == b[1] and a[2] < b[2]) end)
  local ents, lanes, invs = {}, {}, {}
  for _, label in ipairs(cur.labels) do
    local e = cur.E[label]
    if e and e.valid then
      local p = rel(e.position, bx, by)
      ents[label] = {e.name, p[1], p[2]}
      if e.type == "transport-belt" and not label:match("^b%-?%d+$") then
        lanes[label] = {line_items(e.get_transport_line(1)), line_items(e.get_transport_line(2))}
      end
      if e.type ~= "item-entity" and e.type ~= "transport-belt" then
        local o = {}
        for key, which in pairs(INVS) do
          local ok, iv = pcall(function() return e.get_inventory(which) end)
          if ok and iv then
            local c = iv.get_contents()
            if next(c) then
              local list = {}
              for _, st in pairs(c) do list[#list + 1] = {st.name, st.count} end
              table.sort(list, function(a, b) return a[1] < b[1] end)
              o[key] = list
            end
          end
        end
        if e.type == "inserter" and e.held_stack.valid_for_read then
          o.held = {e.held_stack.name, e.held_stack.count}
        end
        if next(o) then invs[label] = o end
      end
    else
      ents[label] = "gone"
    end
  end
  local sel = ""
  if ch.selected and ch.selected.valid then
    local p = rel(ch.selected.position, bx, by)
    sel = {ch.selected.name, p[1], p[2]}
  end
  local pp = rel(ch.position, bx, by)
  return {p = pp, m = ch.mining_state.mining and true or false,
          g = g(ch.character_mining_progress), w = ch.walking_state.walking and true or false,
          inv = counts, e = inv.count_empty_stacks(), a = amounts, sel = sel,
          piles = piles, ents = ents, lanes = lanes, inv_of = invs}
end
P.t0 = game.tick + 1
local old = script.get_event_handler(defines.events.on_tick)
local body
script.on_event(defines.events.on_tick, function(ev)
  if old then old(ev) end
  if P.last == game.tick or P.err then return end
  P.last = game.tick
  local ok, msg = pcall(body)
  if not ok then P.err = tostring(msg) end
end)
body = function()
  local T = game.tick - P.t0
  for k, o in ipairs(order) do
    local t = T - o.t0
    if t >= 0 and t <= o.rig.ticks then
      cur.t = t
      if t == 0 then cur.k = k setup(k) end
      for _, op in ipairs(o.byt[t] or {}) do apply(k, op) end
      if cur.walk then ch.walking_state = {walking = true, direction = cur.walk} end
      if cur.mine then
        ch.update_selected_entity(cur.mine)
        ch.mining_state = {mining = true, position = cur.mine}
      elseif cur.mine_e then
        local e = cur.E[cur.mine_e]
        if e and e.valid then
          ch.update_selected_entity(e.position)
          ch.mining_state = {mining = true, position = e.position}
        else
          -- The mod's poller: the target is gone, so it stops.
          ch.mining_state = {mining = false}
          cur.mine_e = nil
        end
      end
      local key = helpers.table_to_json(reading())
      if key ~= cur.last then
        cur.last = key
        P.log[#P.log + 1] = {k, t, key}
      end
    end
  end
end
return start
"""


def run(family: str) -> dict:
    rigs = [r.to_dict() for r in FAMILIES[family]()]
    body = (PRE + "local TRACKED = " + _lua_value(list(TRACKED)) + "\n"
            + "local RIGS = " + _lua_value(rigs) + "\n" + RUN)  # fmt: skip
    label = f"probe-hm2-{family}-{os.getpid()}"
    _, rows, failed, notes = _world(label, body, ticks=1)
    logs: dict = {r["name"]: [] for r in rigs}
    for k, t, key in rows:
        logs[rigs[k - 1]["name"]].append([t, json.loads(key)])
    return {"rigs": {r["name"]: r for r in rigs}, "tracked": list(TRACKED), "log": logs,
            "create_failed": failed or [], "notes": notes or []}  # fmt: skip


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--family", choices=sorted(FAMILIES), required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    started = time.perf_counter()
    doc = run(args.family)
    from factoriorl.worker import WorkerManager

    doc["engine"] = WorkerManager().engine.to_dict()
    doc["wall_seconds"] = round(time.perf_counter() - started, 1)
    out = Path(args.out) if args.out else EVIDENCE / f"handmine2-{args.family}.json.xz"
    out.write_bytes(lzma.compress(json.dumps(doc, sort_keys=True).encode()))
    print(f"wrote {out} ({out.stat().st_size // 1024} KiB), {doc['wall_seconds']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
