# Belts and burner inserters, measured

What a tick-exact simulator has to do to reproduce transport belts and burner
inserters as Factorio 2.0.60 runs them. Every number below comes from
`docs/evidence/sim-mechanics-m4-logistics.json`, written by
`tools/probe_logistics.py` (about 85 s on the laptop). Ticks are written
`t=N`: game ticks after the probe fuelled its rigs. The world is tick-paused
and advanced one tick at a time with `env.advance(1)`, and every rig is read
after every tick (3,000 ticks, no gaps).

Units: belt positions are in **1/256 tile**. Every position of every item on
every sampled tick was an exact multiple of 1/256, so integer arithmetic is
enough.

## Rigs

| # | rig | key in `samples` |
|---|---|---|
| 1 | ten east belts, six plates per lane fed at the back | `straight` |
| 2 | four east belts into a right turn and three south, one plate per lane | `curve` |
| 3 | four-belt north feed sideloading an east run, three plates per lane | `side_main`, `side_feed` |
| 4 | south-facing burner drill onto a single east belt; belt extended at t=1500 | `drill`, `drill_belt` |
| 5 | chest to burner inserter to chest, plates | `c2c`, `c2c_dst` |
| 6 | four-belt ore line, inserter at its end, stone furnace | `bend*` |
| 7 | 16-belt east line fed every 20 ticks; inserter on the near lane / far lane | `flow*`, `flow2*` |
| 8 | smelting furnace, inserter, east belt | `fout*` |
| 9 | chest to inserter to unfuelled furnace, ore and coal | `fill_*` |
| 10 | inserters with one coal / one wood moving plates | `fuel*`, `wood*` |
| 11 | drill onto a two-belt run, inserter at the end | `tick_*` |
| 12 | chest to inserter onto a belt running east, west, south | `drop1..3` |
| 13 | one item per rig reaching a sideload at 8 sub-tile phases, both feed lanes | `phase1..16` |
| 14 | drill into a chest, inserter out of it, built in both orders | `order_a`, `order_b` |
| 15 | drill onto a belt tile that an inserter picks from | `same` |
| 16 | inserter built with no fuel, moving coal chest to chest | `self` |
| 17 | inserter filling a fuelled, working furnace with ore | `hot` |

All seventeen built (195 `create_entity` calls, none failed) and all moved
items the intended way.

## Belts

### Coordinates

- Each belt entity owns one transport line per lane. On a straight belt both
  lines are 256 long (`line_length = 1`).
- **Position 0 is the downstream edge, 256 the upstream edge.** For an east
  belt centred on x=130.5, `get_line_item_position(0)` is x=131 and
  `get_line_item_position(1)` is x=130.0039.
- **Lane 1 is the left of the direction of travel, lane 2 the right.** For an
  east belt at y=130.5, lane 1 runs at y=130.265625 and lane 2 at y=130.734375:
  60/256 either side of the centre line.
- Lines of consecutive belts are separate objects (`line_equals` is false
  between belt 1 and belt 2) but chain: `total_segment_length` reads 10 on
  every belt of the ten-belt run.

### Speed and hand-over

- Belt speed is **8/256 per tick** (`belt_speed = 0.03125`). An item at 248 on
  belt 1 reads 240, 232, ... one tick at a time (`straight`, t=0..).
- Crossing belts is continuous: position p becomes p-8, and if that is below 0
  the item moves to the next belt at p-8+256. Plate 1 of `straight` reads 0 on
  belt 1 at t=31 and 248 on belt 2 at t=32; the same every 32 ticks.
- There is no per-belt delay: every one of the nine belt boundaries in
  `straight` takes exactly 32 ticks.

### Spacing and compression

- Minimum gap between items on one lane: **64** (four items per lane per
  belt). `insert_at_back` on a moving line was accepted at t=0, 8, 16, 24, ...:
  once the previous item is 64 ahead.
- At the end of a line the front item stops at **exactly 0**. Everything
  behind compresses to 64 apart: at t=400 belt 10 holds 0, 64, 128, 192 on
  both lanes and belt 9 holds 0, 64 (`straight`).
- Arrival is a clamp, not a fractional step: the first plate went in at 248 on
  belt 1 of 10 and reached 0 on belt 10 at t=319 = (248 + 9x256)/8.

### Insertion reads back one step on

An item placed at nominal position p on a line that is free to move reads
**p-8 on the tick it was placed**:

- `insert_at(0.5 + k/256)` on the phase rigs' feed belts reads 120 + k at t=0,
  with no tick run since the insert (setup and the first sample are both on
  game tick 0, `fuelled_tick`), then 112 + k at t=1.
- `insert_at_back` reads 248, not 256 (`straight`, t=0).
- A drill output and an inserter drop aimed at 128 read 120 on the output tick
  (`drill_belt` t=243, `drop1` t=47).

When the item cannot move (the gap ahead is already 64, or it is at 0) it
reads p: `drill_belt` t=723 reads 128 behind items at 0 and 64, and
`insert_at(0)` reads 0. The simplest model: **an insertion happens before the
line moves in that tick** (or, for script inserts between ticks, it is moved
once immediately).

### Curves (right turn, east into south)

- The turn belt's lines are **295** long on the outer lane (lane 1, the left
  lane on a right turn) and **106** on the inner lane (lane 2). Chain totals
  for 4 + turn + 3 belts: 2087 (8.15234375 tiles) and 1898 (7.4140625).
- Transit is continuous through the turn with the same 8 per tick. Lane 1
  enters the turn at t=128 at 287 (=295-8), reads 7 at t=163 and 255 on the
  next belt at t=164. Lane 2 enters at 98 (=106-8) at t=128, reads 2 at t=140
  and 250 on the next belt at t=141.
- End stops: lane 1 at t=260, lane 2 at t=237, both at 0; (248 + chain -
  256)/8 rounded up in each case.
- A left turn was not measured. The mirror image (outer = lane 2) is the
  obvious guess and needs a rig before it is relied on.

### Sideloading

A belt pointing into the side of another belt (not its end) feeds **both of
its lanes onto the one lane of the target on the side it comes from**. Feed
running north into the south side of an east belt: everything joins main
lane 2, on the belt the feed points at.

**Single item, exact** (`phase1..16`, one item each, empty main): an item at
feed position k (k = 0..7) at t=15 appears on the main belt at t=16 at

- **180 + k** from feed lane 1 (the feed's west lane), and
- **59 + k** from feed lane 2 (its east lane).

That is `E - (8 - k)` with the overshoot carried: E = 188 and 67. 188 is the
distance from the main belt's downstream edge to the feed lane's centre line
(0.734375 tile). Geometry says 68 for the other lane and the engine uses
**67** -- the same 67/256 shows up as the start of the inner curve lane
(x=134.26171875 in `setup.lines.curve`). Use the measured values.

**Several items at once is not fully explained.** In rig 3 (three plates per
feed lane, 8 ticks apart, both lanes arriving on the same ticks) the entries
were, feed lane 1: 172 (t=128), 180 (t=136), 187 (t=144); feed lane 2: 59
(t=128), 59 (t=136), and the third held at feed position 0 from t=143 to t=159
and entered at 123, i.e. 64 behind the item ahead. The 172 and 187 do not fit
the single-item rule. See "Open questions".

## Burner mining drill onto a belt

- A south-facing drill centred on (x, y) drops at (x+0.5, y+1.296875). On an
  east belt below it that is position **128** on **lane 1** (the drop point is
  in the north half of the belt tile).
- The mining cycle wraps (progress 0.99999 at t=241, 1/240 at t=242) and the
  ore appears on the belt the **next** tick: t=243, then 483, 723 (every 240).
  On the output tick it reads 120 (moved once, see above).
- On a one-belt line the lane fills to 0, 64, 128 and the fourth output is
  refused: status `waiting_for_space_in_destination` from t=963, progress
  frozen at 1/240. Position 192 stays empty: the drill does not put the item
  behind the drop point on a stopped line.
- When the belt was extended at t=1500 the waiting output landed at **t=1501**
  and read **184** (192 before the move), not 128. The cadence restarted from
  there: 1741, 1981, 2221, then blocked again at t=2461.
- Drill into a chest is the same: `order_a` / `order_b` show the item in the
  chest at t=243. The chest drill reports `waiting_for_space_in_destination`
  for the one tick t=242 in between; the belt drills report `working`.

## Burner inserter

### Prototype and geometry

`rotation_speed` 0.013 turns/tick, `extension_speed` 0.035 tiles/tick, pickup
(0, -1), drop (0, 1.2), `max_energy_usage` 2400 J/tick, one fuel slot,
`chases_belt_items` true. **Direction points at the pickup tile**: built
facing north at (160.5, 130.5), it picks at (160.5, 129.5) and drops at
(160.5, 131.69921875). Only north-facing inserters were built here;
"Inserter belt pickup" below covers all four facings.

### Built state: a quarter of a wood

A burner inserter from `create_entity` has an **empty fuel slot but is already
burning wood with 500,000 J remaining** (a quarter of one wood's 2 MJ), energy
buffer 0, hand retracted to 179/256 on the pickup side (`setup.burning_at_build`,
every inserter at t=0). It works without any fuel added. The fuel you insert is
only loaded when that runs out: t=551 for a chest-to-chest inserter, where
`remaining` jumps to 3,999,889.99 -- the 110 J still owed that tick is taken
from the new coal.

### Chest to chest cycle (`c2c`)

| event | tick |
|---|---|
| buffer filled (0 to 2560 J) | t=1 |
| first pickup (hand extends 179 to 256 at about 9/256 per tick) | t=9 |
| drop, item in the destination chest | t=47 |
| next pickup | t=85 |
| drop | t=123 |

**38 ticks pickup to drop, 38 drop to pickup, 76 per item.** 0.5 turn / 0.013 is
38.46. The arm takes 37 full steps (0.481 turn) and on the 38th arrives: once
the extension is on target, a rotation with less than one step left after the
step is completed in that tick (rule 2 of "Inserter belt pickup").

### Energy

Burner draw per tick, from `remaining_burning_fuel` differences (`c2c`,
`fuel`, `wood`):

- t=1: 2,560 (fills the buffer); t=2..9 (first approach): 1,750 each.
- Every half-swing, both directions: **5 ticks at 2,400 then 33 ticks at 650
  = 33,450 J**. A chest-to-chest cycle is **66,900 J per item** (t=85..161,
  summed).
- Waiting (`waiting_for_source_items`, `waiting_for_space_in_destination`):
  **0 J/tick**. There is no drain. The buffer is not always 2,560 while
  waiting: an inserter that falls asleep keeps what its last move left
  (810 or 1,910 are common); see rule 6 of "Inserter belt pickup".
- All of these are one rule: **50 kJ per turn of rotation plus 50 kJ per tile
  of extension** actually charged that tick (0.013 x 50,000 = 650, 0.035 x
  50,000 = 1,750, both = 2,400). Chase amounts such as 745.46 or 1,935.71 in
  `flow` are partial steps of the same rule; the chase probe matches every
  tick to 0.03 J.

Check against exhaustion (`wood`: the built-in quarter wood, then one wood):
the wood is loaded at t=551 and runs out at t=2825 (remaining 0, buffer
1,099.96); at t=2826 the buffer is 0 and status `no_fuel`, the hand stopped
mid-swing still holding a plate. 2 MJ / 66,900 J = 29.9 cycles = 2,272 ticks;
observed 2,274. 37 plates delivered by then (39 for the coal inserter by
t=3000, still running on its coal with 1.85 MJ left).

### Hand position

`held_stack_position` is the arm's polar position plus a drawn lift. Its
**x** offset from the inserter is exactly `-L sin(2 pi a)` of the arm state
in "Inserter belt pickup", truncated toward zero to 1/256 tile, on every tick
of every chase rig. Its **y** offset is `-L cos(2 pi a)` minus a lift that
rises and falls over a swing (up to about 150/256 on a full half turn, 0 at
both ends, 0 while chasing from rest); that is why its distance goes above
1.5 tiles mid-swing (`c2c`, t=10..46). The lift never feeds back into the
logic and is not modelled. The item swings through the west side because of
the half-turn rule (rule 2).

### Waiting inserters react one tick late

A waiting inserter acts on the tick **after** its source changes:

- drill output into a chest at t=243, picked at **t=244**, identically with the
  inserter built after the drill (`order_a`) and before it (`order_b`);
- furnace result at t=194, picked at t=195 (`fout`);
- room in a working furnace's source at t=240, next ore picked at t=241 (`hot`);
- item appears on the pickup belt tile at t=243, hand starts moving at t=244
  (`same`). In `tick_ins` and `bend` the hand starts on the tick the item
  crosses into the pickup tile (t=259, t=96), i.e. one tick after it reached
  position 0 of the belt before, the tile boundary.

Build order made no difference in the one rig that tested it. Drill output and
inserter drop both land before belt movement in their tick (the p-8 readback).

### Pickup from a belt

- **Stationary items at a belt end are like a chest**: the pickup is exactly 38
  ticks after the previous drop (`bend`: drop t=134, pickup t=172). The item
  taken was the one on the tile centre, position 128; lane 1 and lane 2 both
  had one there and it took **lane 2** (t=101 and t=172). Whether that is "near
  lane" or "right lane" needs an inserter on the other side of a belt.
- **An arriving item is chased.** `same`: item appears t=243 at 120, hand moves
  t=244, grabbed at t=250 with the item at 64; dropped 36 ticks later (t=286);
  back 38 later (t=324). `tick_ins`: pickup t=265, drop t=300 (35). `bend`:
  pickup t=101, drop t=134 (33). The grab-to-drop time shrinks by however much
  the arm already rotated while chasing.
- **From a moving belt it catches every swing, but the timing depends on the
  chase.** 20-tick feed, near lane: 40 items taken by t=3000, pick-to-drop 31
  to 38 ticks, drop-to-pick 32 to 45; far lane: 38 items. It is deterministic
  -- the near lane repeats exactly with a 140-tick period (2 items) from t=325,
  the far lane with 220 ticks (3 items) from t=481, hand positions identical.
  The chase kinematics are now reproduced tick by tick: see "Inserter belt
  pickup" below (both of these rigs replay exactly).

### Drop onto a belt

- The item goes onto the lane whose half of the tile contains the drop point,
  at the drop point's distance from the belt's downstream edge:
  - belt running east: **lane 2**, position 128 (reads 120 on the drop tick);
  - belt running west: **lane 1** (the south half is the left lane going west),
    128;
  - belt running south, away from the inserter: the drop point is on the
    centre line; it went to **lane 2**, position 77 (reads 69).
- Blocking: with 0, 64, 128 on the drop belt's lane the inserter holds the
  item, `waiting_for_space_in_destination` from t=883 (`drop1..3`). As with
  the drill, 192 is never used.
- Furnace to belt (`fout`): result picked the tick after it appears, dropped 38
  ticks later onto lane 2 at 128.

### Fill limits

- **Ore into a stone furnace: 2 in the source slot.** Unfuelled (`fill_ore`):
  delivers at t=47 and t=123, then waits empty-handed at the pickup.
  Fuelled and smelting (`hot`): the source never exceeds 2 and is topped up
  one tick after each craft takes an ore (t=240 to 241, 432 to 433).
- **Coal into the fuel slot: 5** (`fill_coal`, last delivered t=351).
- The limit is checked **before picking up**: the arm returns to the pickup
  and waits without taking an item. Status reads
  `waiting_for_source_items` even though the chest is full.

### Self-refuel

An inserter moving fuel keeps its own slot stocked (`self`: built with only
the quarter wood, moving coal). It picked coal at t=9 and at **t=37** put it
into **its own fuel slot**, not the destination; the next pickup was t=65 and
that coal went to the chest (t=103). When the wood ran out at t=454 it loaded
that coal and the coal in its hand went into itself again at t=491. Timing of
the self-insert (28 ticks after pickup; 37 ticks after the slot emptied
mid-swing) is recorded but not explained.

## Inserter belt pickup

How a burner inserter takes items off a yellow belt, moving or stopped, to
the tick. `tools/inserter_model.py` is the executable form of this section;
it replays every rig below from its first tick and compares with the engine.

### Evidence

`tools/probe_inserter_chase.py` builds rigs of one burner inserter, a belt
through its pickup tile and a wooden chest, and records every tick: hand
position, whether it holds an item, burner energy (fuel left + buffer, and the
buffer alone), and every item (unique id, lane, position) on the pickup belt
and its two neighbours. Five families, `docs/evidence/inserter-chase-*.json.xz`:

| family | rigs | ticks | what varies |
|---|---|---|---|
| `single` | 256 | 400 | one item reaching an idle inserter: facing N/E/S/W; belt crossing left-to-right, right-to-left, running toward the inserter (item stops at the belt end), running away; near/far lane; item position mod 8 |
| `stream` | 252 | 3,000 | items fed every 8..40 ticks on the near, far or both lanes, all facings, both crossings, queues at a belt end; half the rigs start with an item in hand |
| `wake` | 80 | 900 | sleeping inserters, 500+ ticks after build: when the buffer is refilled |
| `loss` | 524 | 900 | an item the returning hand fails to catch, lost with the hand at many distances |
| `pair` | 40 | 2,000 | two inserters on opposite sides of one belt tile, both lanes fed, facings N/E (and their opposites), both crossings |

`tools/probe_inserter_wake.py` (`docs/evidence/inserter-wake.json`) adds items
to belt lines in different ways and watches sleeping inserters wake. The two
moving-belt rigs of the first probe (`flow`, `flow2`) are replayed as a sixth
family, `m4`.

**Agreement.** With the burner buffer taken from the engine each tick (which
isolates everything but rule 6), all 1,154 rigs replay exactly: 1,487,998
rig-ticks, 12,578 hand fills. Exactly means, on every tick: the hand's x
offset (below), whether the hand is full, which item was picked up and on
which tick, and the energy spent (largest error 0.022 J). With rule 6
computing the buffer too, every rig is exact except in the first ~300 ticks
after its belts were built (see "Not yet exact").

### Frame and state

Everything below is in the inserter's frame: its centre at (0, 0), its pickup
tile ahead at (0, -1), the chest behind; units 1/256 tile; y down. The four
facings are rotations of this frame, except for the half-turn rule in 2.

- The arm is an orientation `a` (turns, **anticlockwise** from straight ahead,
  so 0.25 is the inserter's left) and a length `L`. The hand is at
  `(-L sin 2 pi a, -L cos 2 pi a)`.
- As built: `a = 0`, `L = 179.2` (`starting_distance` 0.7).
- Pickup point: `a = 0, L = 256`. Drop point: `a = 0.5, L = 307.2` (1.2
  tiles; `drop_position` reads 1.19921875 but the arm uses 1.2).
- The engine reports `held_stack_position` as the inserter's position plus the
  hand vector **truncated toward zero** to 1/256 in each world axis. The x of
  that is exact; the y adds a drawn lift during swings (see "Hand position").
- A belt item's position is its lane's centre line (60/256 either side of the
  belt's centre, lane 1 on the left of travel) at its distance from the belt's
  downstream edge.

### The rules

**1. Order.** Belts move first in a tick, then the inserter. It sees items
where this tick's belt move left them, and an item that crosses onto the
pickup belt is chased in the same tick (`single`: item enters at 255, hand
moves that tick). Items the script adds between ticks are seen at once.

**2. Moving toward a target** `(a_t, L_t)`, each tick, `ROT = 0.013` turn,
`EXT = 0.035 tile = 8.96/256`:

- Extension, `dl = L_t - L`:
  - `|dl| < 0.001 tile` (0.256/256): `L = L_t`, **no energy** (`stream`: 0.253
    free, 0.279 charged);
  - `|dl| <= EXT`: `L = L_t`, charged `|dl|`;
  - otherwise `L += EXT` toward it, charged EXT, and if less than EXT is then
    left, `L = L_t` (still charged EXT). So the extension arrives when under
    two steps remain: the first approach 179.2 to 256 takes 8 ticks, 256 to
    307.2 takes 5.
- Rotation, `d` = the short way from `a` to `a_t`:
  - `|d| <= ROT`: `a = a_t`, charged `|d|`;
  - otherwise `a += ROT` toward it, charged ROT, and **if the extension
    reached its target this tick** and less than ROT is then left, `a = a_t`.
    Without the extension condition this rule breaks (`m4` far lane t=258:
    0.0246 turn left, no snap, extension still moving; `single` rig 0 t=170:
    0.0198 left, snap, extension done).
  - An **exact half turn** is decided in world orientation (clockwise from
    north, [0, 1)): +0.5 goes anticlockwise, -0.5 clockwise, against the sign.
    So a north-facing inserter swings from pickup to drop and back through the
    west both ways, a south-facing one out through the east and back through
    the west, an east- or west-facing one back through the north (`single`,
    all 64 rigs of each facing). Differences that are not exactly 0.5 take the
    short way.
- Arrived = both axes on target this tick.
- Energy = 50,000 J x (turns + tiles) as charged above. If the buffer holds
  less (rule 6), the extension is paid first and moves in proportion to what
  it gets; rotation gets the rest (`single`: 810 J buys 4.15/256 of extension
  and no rotation). As after a full step, if less than EXT is then left the
  extension is on its target (the second probe's `seg_*_8`, `seg_b_7`: 810
  J buy 0.0162 of a 0.0337 extension and the hand is on the item; fifth
  probe).

**3. Holding an item:** target the drop point. On arrival the item goes into
the chest that tick; the swing back starts next tick. Chest to chest this is
the 38/38 cycle above.

**4. Empty: which target.**

- The item being chased, as long as it is on the pickup belt (either lane,
  position 0..255 of that belt entity). Items on the next belt do not count.
- Otherwise a new one among the items on the pickup belt:
  **the lane nearer the inserter first** (for a belt that runs along the arm,
  where both lanes are equally near, **lane 1**, the left of travel), then
  **the item furthest upstream** (largest position). Not the nearest, and not
  the one needing least rotation (`stream` both-lane rigs; `m4` t=358, t=1134).
- The choice sticks: a newer item entering the belt does not replace it.
- No item: target the pickup point. At rest with nothing to chase the
  inserter waits there.
- **On arrival at an item it is picked up that tick**, off the belt at once;
  the hand is exactly on the item's position. That can be the tick the item
  crosses onto the pickup belt (`pair` rig 15, t=317).

**5. The chased item leaves the pickup belt.** If the hand is **over the
pickup belt's tile** (x in [-128, 128], y in [-384, -128]) at that moment, the
inserter does nothing this tick (no move, 0 J) and chooses again next tick.
Otherwise it chooses again at once and moves (`loss`: hand x -121.5 idles,
-132.2 moves; `stream` rigs 6, 8, 9). The tile edges themselves are not
pinned: x = +-128 is the natural value and the data only bound it to
121.5..132.2; the y edges were never approached.

**6. The burner buffer** (2,560 J when full):

- Refilled after every tick the inserter is awake, whether or not it moved.
- It **falls asleep** when it ends a tick at the pickup point with nothing to
  chase and **no item anywhere on its pickup belt's line** (the connected
  chain of belts, upstream and downstream). Asleep, it keeps the buffer as its
  last move left it: 810 after the first approach, 1,910 after a return swing.
  If the line holds items it stays awake at the pickup point (`wake` case B).
- It **wakes** when an item is added to that line: inserted anywhere on it,
  35 belts upstream or downstream of the pickup belt included; entering it
  from a sideload (when the item joins the line, not when it goes on the
  feed); dropped onto it by another inserter. Not for items on an unconnected
  belt. Waking without a move refills the buffer.
- An item that arrives on the pickup belt with no add since the inserter fell
  asleep (dropped straight onto the pickup belt, for instance) wakes it into
  a move paid from the stale buffer (`single` "away" rigs: 810 J).
- **Young belts wake late.** For roughly the first 300 ticks after the belts
  were built, an add wakes the inserter only at a later tick (0 to 287 ticks
  after the belts were built, observed), and items on the line do not keep it
  awake. The tick depends on where the rig is in the world (moving the whole
  grid by a few tiles changes it) and not on when the inserter was built
  (belts built at t=0 and inserters at t=400: all immediate). After ~400
  ticks every add woke its inserter at once (256 of 256 rigs, adds at t=400
  and t=800).
  **Explained by the third probe** ("Belt-line segments"): "its line" is the
  *segment* its pickup belt belongs to, and a young belt is a segment of its
  own until its chain merges, 1 to 600 ticks after it was built.

**7. Two inserters on one belt tile** (one each side, `pair`): the one built
later updates first in a tick. So when it takes an item the other was
chasing, the other finds it gone in the same tick (rule 5 applies at once);
when the earlier-built one takes an item, the later one still saw it that
tick, moved toward it, and finds it gone the next tick. This held for all
four relative placements; whether sleeping and waking can change the order
was not tested.

### Exact vs not yet exact

**Exact** (every rig, every tick, all quantities above):

- belts crossing in front of the inserter in either direction, belts running
  toward it (items stopped at the belt end) and away from it;
- near lane, far lane, both lanes;
- all four facings;
- single items at every position mod 8, streams every 8 to 40 ticks, items
  queued and stopped, an inserter starting with an item in hand or given one
  mid-run;
- missed items, retargeting, lost targets;
- two inserters taking from one belt tile, one each side;
- the energy of every tick, including partial moves;
- the buffer, sleep and wake once the belts are more than ~300 ticks old.

**Not yet exact:**

- **The young-belt wake delay** (rule 6). It only matters when an inserter is
  asleep with a part-empty buffer and its first move is limited by that. In
  the probes that is the first pickup after build: with the delay ignored,
  242 of 256 `single` pickups and 10,107 of 10,312 `stream` pickups are on
  the engine's tick; in each diverging rig the first differing pickup is one
  tick later in the engine, and every divergence starts between t=166 and
  t=173. The per-rig delay could not be predicted.
  **Decision (2026-09-24): accepted.** The simulator ignores the delay. Parity
  comparisons may differ by one tick on the first pickup in the first ~300
  ticks after belts are built, and are exact after that.
  **Explained and reproduced since** (third and fourth probes): the inserter
  watches its belt-line segment, and young belts are segments of their own
  until their chain merges. factory-sim compares these rigs from t=0.
- **Hand y** (the drawn lift) is not modelled; nothing depends on it.
- **Not measured at all:** curved or sideloaded pickup belts, underground
  belts and splitters, more than two inserters on a line, fast or express
  belts, drop targets other than a chest, a chase interrupted by running out
  of fuel, and the exact tile edges of rule 5.
- The energy of a tick agrees to 0.03 J, not bit for bit; decisions near a
  snap threshold could in principle differ by that rounding. None did.

## Second probe: turns, ground, chests, order, mining

`tools/probe_logistics2.py` (about 75 s) built about 460 rigs for what the
first probe left open, read every tick for 1,500 ticks, and wrote
`docs/evidence/sim-mechanics-m4-logistics2.json.xz`. The rig families are
listed in the probe's docstring. factory-sim rebuilds them in
`tests/logistics_rigs2.py` and compares every reading on every tick
(`tests/test_mechanics_logistics2.py`, `tests/test_character_logistics2.py`);
the rigs it leaves out are named there with the reason.

### Belts

- **Left turns** mirror right turns: lane 1 is the inner lane (106/256 long),
  lane 2 the outer (295/256). Right turns are the other way round (lane 1
  295, lane 2 106).
- **Sideload from the target's left** (`sl_left_*`): the item joins the
  target's lane 1 (the near lane). From the feed's lane 2 it enters at
  180 + k, from lane 1 at 59 + k, k the sub-tile phase: the mirror of the
  right-hand rule.
- **Side feeds and shape** (`side_*`): a belt fed from one side at the start
  or end of a line becomes a turn; fed from both sides, or with a belt behind
  it, it stays straight. A head-on feed stops at position 0.
- **Rotating a loaded turn back to straight** (`rot_*`): items are re-placed
  on each lane by `new = floor(p * (L_new - 1) / L_old)` whenever the lane
  length changes, in the same script call. The `rotate_and_mine` parity trace
  is exact with this rule.
- **Simultaneous sideload arrivals** (`sim_*`): not settled. When both feed
  lanes reach the target on one tick, which item moves first depends on the
  insertion order, the number of belt lines crossed and the rest of the world;
  the first probe's `side` rig and the `three` rigs disagree under any single
  order tried.
  **Explained by the third probe** ("Update order and simultaneous
  sideloads"): the order is the lines' activation order, which the lines'
  merging changes; "the rest of the world" is the rig's position.

### Inserters and drills on turns and the ground

- **Drops onto a turn** (`tdrop_*`, `tdrill_*`): the item goes on the inner
  lane if the drop point is nearer the inner corner, otherwise on the outer,
  at half the lane's length (53 on the inner, 147 on the outer). Drills
  behave the same.
- **Pickups from a turn** (`tpick_*`): lane 1 first, then the
  furthest-upstream item, as on straight belts. Where the arm aims for an
  item on the turn's arc is not pinned to 1/256: `setup.lines` puts the arcs
  at radius 188/256 and 67/256 about the inner corner, and hand x and energy
  still differ from the first move.
  **Third probe:** the arm aims at the item's point as
  `get_line_item_position` gives it, tabled ("Turn item points").
- **Ground** (`ground_*`, `gpair_*`): with nothing at the drop point the
  inserter drops an item pile on the ground at the drop point, unless a pile
  is already there, and then waits. With nothing at the pickup point it takes
  from item piles on that tile.
- **Full chests** (`full_*`): an inserter holding an item for a full chest
  waits at the drop point until room appears (`RELEASE` frees room by
  script).
- **Fill limits** (`fill_*`): fuel into a drill, a furnace or an inserter
  stops at 5; ore into a furnace at 2; a drill takes no ore; iron plates go
  into no furnace; stone stops at 4 in a furnace (it makes stone bricks).
- **Mixed chests** (`mix_*`): the inserter takes from the last slot holding
  an item its target wants.
- **Update order** (`order_*`, `wake3`, `chain_*`, `woken_*`): inserters
  update from the last in the update list to the first; a new inserter joins
  the end. One waiting on a machine or chest sleeps and leaves the list; any
  change to that machine's or its target's contents wakes it, and it rejoins
  the end, so those that fell asleep first run first on the next tick, never
  on the tick they were woken. An inserter waiting on a belt line sleeps on
  the line and stays in the list.
- **Energy running out mid-move** (`pe_*`): with less than a full tick the
  extension is paid first and the rotation gets `(float)((budget - e_ext) /
  50000)` turns; what remains below 1e-9 J stays in the buffer and the status
  stays `working`. All 24 rigs are exact.
- **Self-refuel redirect** (`sr_*`): when the fuel slot empties while the
  arm is carrying fuel, the arm turns toward its own slot from wherever it is,
  loads it and swings back to the drop.
- **Drill status** (`dstat_*`): a drill whose output was refused reads
  `working` in the same script call as a change to its output target: its
  belt extended or rotated, room made in its chest, the pile at its drop
  point removed. A chest built nearby does not change it.

### Chests, mining and the character

- **Prototypes** (`setup.prototypes`): wooden chest collision box ±0.3477,
  mining time 0.1 s; transport belt ±0.3984, 0.1 s; burner inserter ±0.1484,
  0.1 s; character ±0.1992.
- **Placement under the character** (`setup.placement`): a chest or inserter
  can't be placed with the character 0 or 0.3 tiles from the tile centre; it
  can from 0.6 on. A belt can be placed under the character anywhere.
- **Mining returns** (`mine`): a chest gives its slots in order, then
  itself; a belt gives lane 1 front to back, lane 2 front to back, then
  itself; an inserter gives its fuel, itself, then what it held.
- **The character on belts** (`char_*`): standing, it's carried 8/256 a
  tick along a straight belt; walking with the belt it moves 46 a tick,
  walking against it 30 a tick in the walking direction. Standing on a lane
  line, at a belt end and carried into a chest are reproduced tick by tick.
  Carriage round a turn (`char_rturn`, `char_lturn`) is not reproduced.

### How factory-sim compares

- **Young belts** (see "Not yet exact"): compared from t=0 since the fourth
  probe; they were compared from t=300 with a fuel offset before.
- **Hand y**: the drawn lift of a swing that didn't start from rest is not
  known. The simulator marks such records and the comparison takes the
  engine's y (`fsim.trace.relax_hand_y`); hand x is compared exactly.
- **Not reproduced yet:** pickups from a turn by an inserter on the side
  opposite the feeding belt (`tpick_r_e`, `tpick_l_e`: hand x 1/256 off on
  two ticks each; the four other `tpick_*` rigs are exact); an item added
  onto a pickup belt that runs along the arm into a sleeping inserter
  (`seg_*_8`, `seg_b_7`: the arm's first move, not segments; matched by
  the fifth probe's reading of that move); and carriage
  round a turn. Simultaneous sideload arrivals (`side_both`, `side_turn_two`,
  `sim_*`) and the `smelting_chain` sleep at t=776 were reproduced with
  belt-line segments in the fourth probe.

## Third probe: segments, turn points, sideload order

`tools/probe_logistics3.py` measures what the three behaviours left open by
the second probe have in common, and `tools/check_logistics3.py` replays the
rules below on the stored evidence without the engine (52 of 52 sideload
cases, 12 of 12 young wakes, 2 of 2 splits as predicted). Evidence, all in
`docs/evidence/`:

| file | what |
|---|---|
| `logistics3-turns.json.xz` | `get_line_item_position` at every position of both lanes of all eight turns |
| `logistics3-delay.json.xz` | the merge delay of every tile and lane in x -6..35, y -22..6 (the parity scenes) |
| `logistics3-delay-<tag>.json.xz` | the same for the rigs checked below: `wake-young` (`inserter-wake.json`), `probe2-sideload` (the second probe's sideload rigs), `probe1side`, `at200`, `at300` |
| `logistics3-segments.json.xz` | which belts' lines are one object, every tick, in three parity scenes |
| `logistics3-sideload.json.xz` | 32 sideload rigs, one fresh world each |
| `logistics3-adhoc.json.xz` | the one-off runs the rules below were read from (named in the file) |

### Belt-line segments

A lane's lines on consecutive belts start as separate objects and are later
merged into one, a *segment*: `LuaTransportLine.line_equals` turns true
between the belts, and `total_segment_length` (the whole chain, already on
the build tick) does not show it.

- **Merge delay.** Every tile and lane has a fixed delay `d`, 1 to 600 ticks.
  A chain (straight belts and turns alike) merges whole on the tick the first
  of its belts' delays, counted from when that belt was built, runs out: the
  minimum over its belts. Evidence: 640 two-belt chains; 240 three-belt
  chains, each merging at the smaller of its two pairs' ticks; every chain of
  three parity scenes, turn included.
- `d` is measured one tile at a time: a lone belt, with a neighbour destroyed
  and rebuilt every tick so the neighbour's own delay never runs out.
- `d` depends on the tile and the lane only: not on the belt's facing (north,
  east, south and west agree), the build order (a grid built in reverse gives
  the same), the build tick (a grid built at t=37 merges 37 later, all 640),
  other entities, the map seed or the surface (a new surface with another
  seed: 82 of 82 equal). It is not periodic in x or y and the two lanes are
  unrelated. Common hashes of the tile or its 1/256 coordinates (FNV-1a,
  splitmix64, MurmurHash3 finalisers, with the lane mixed in several ways)
  match at chance level: the function is not identified.
- **Re-arming.** A belt whose delay ran out with nothing to merge with starts
  it again when a neighbour is built; one whose delay is still running keeps
  its deadline. A first belt at t=0 and its neighbour at t=700 merge exactly
  the pair's delay after t=700; with the neighbour at t=1, 37 or 100 they
  merge at the first belt's own deadline when that comes first.
- **Boundaries.** An entity that works on a lane marks a boundary two belts
  downstream of its belt, between belt k+2 and k+3 when it works on belt k:
  a burner drill's output (eight rigs, and the smelting chain), an inserter
  dropping onto the lane or picking from the belt (`logistics_belt_pickup`,
  `logistics_sideload_merge`). A sideload onto belt k marked one between k+1
  and k+2 (one observation). A drop or sideload marks only its lane; a
  pickup, both lanes. Script inserts mark none. Merging stops at boundaries.
- **When a boundary splits a merged segment:** at the tick of the first such
  interaction plus `d` of the segment's downstream-most belt. Smelting chain,
  lane 1 (belts 1 to 9 merged at t=39): drill's first output t=242, `d` of
  belt 9 (12,-4) is 492, split at 734 into belts 1-4 and 5-9. Belt pickup:
  first drops t=46, `d` of the last belt 87 (lane 2) and 516 (lane 1), split
  at 133 and 562. Eight drill rigs: first output + `d` of the last belt, all
  exact. A chain that had not merged yet merges straight into its pieces at
  its merge tick (sideload merge: lane 2 at 159 into 1-4, 5-7, 8-10).
- **An inserter watches its pickup belt's segment**, not the whole chain:
  rule 6 above with "line" read as "segment". That is the young-belt wake
  delay: before the chain merges, an item added upstream is on another
  segment, and the inserter wakes when the chain merges or when the item
  reaches its belt, whichever comes first. All twelve `young` rigs of
  `inserter-wake.json` wake at exactly that tick (one record earlier, by that
  file's convention). It is also the `logistics_smelting_chain` case: at
  t=536 the lane was one segment and the ore on belt 4 kept the inserter
  awake; at t=776, after the split at 734, the ore on belt 4 is on the other
  segment, the inserter sleeps with 1,910 J, and it wakes at t=803 when the
  ore crosses onto belt 5, its own segment.

### Update order and simultaneous sideloads

- **Order.** Segments move in last-activated-first order: a segment that
  gets an item while it has none goes first. Of two items put on the two
  lanes of one feed in one script call, the one put second moves first.
  Other lines activated or emptied before, between or after do not change
  the order of the two (decoy rigs).
- **Crossing a belt** onto a line that is a segment of its own activates it,
  so on a young feed each belt crossed reverses the order of the two lanes;
  on a merged lane nothing is activated. Feeds of 1 to 4 belts at (300, 300)
  alternate with the number of belts crossed; at (183, 113) the feed's lane 2
  merged at t=42 and lane 1 at t=85, so the crossings at t=48 and t=80
  re-activate lane 1 only and it stays first; with the belts 600 ticks old
  the item put second always goes first, whatever the number of belts.
- **A lane that merges while it carries items** keeps its place in the order
  if its item is already on the chain's downstream-most belt, and otherwise
  goes last. (Keeping its place always fails `sim_three_l1first`; going last
  always fails `sim_f4_l2first`.)
- **Two items sideloading onto one empty target segment in one tick:** the
  first inserted wakes the segment without moving it; the second insertion
  finds it awake and not yet moved this tick and moves it first, so the item
  inserted first goes 8/256 further (172 + k or 51 + k instead of 180 + k,
  59 + k). With the target already carrying items it has moved before either
  insertion and neither goes further (target-traffic rigs: an item on the
  target moves freely whether it was placed before, between or after the
  feed items). **Fourth probe:** that holds for a target whose items move; a
  target whose items are all stopped is asleep, and the first sideload wakes
  it like an empty one ("Stopped lines sleep").
- With the measured merge delays these rules predict which item goes
  further in all 52 cases checked: the 24 script-fed and 6 inserter-fed rigs
  of `logistics3-sideload.json.xz`, the 18 two-item `sim_*` rigs and the two
  feeds of `side_both` of the second probe (including `three_l1first`, which
  had gone the other way from its twin: it does so alone in a fresh world at
  its position too), the first probe's `side` rig, and
  `logistics_sideload_merge` at t=127, where every line is still its own
  segment (the feed's lanes merge at t=139 and t=325, the main's at 159 and
  174), the items cross two belts, and lane 2 goes further: so the lane-1
  drop was activated first in t=47. Two-inserter rigs agree: with the west
  inserter built first, lane 2 goes further, for feeds of 1, 2 and 3 belts.
  **Settled by the fourth probe:** a burner inserter drops on the *far* lane,
  so the west inserter drops on lane 2 of a north feed and the east one on
  lane 1; the lane-1 drop that comes first is the last-built inserter's, as
  in the chest rigs ("Two inserters on one tick").

- **Acceptance.** In the `three` rigs a sideloaded item went on exactly 64
  behind its target (lane-2 item, t=160: target 59, item ahead at 59, placed
  at 123), where the simulator's `lane_insert` refuses at 64 and accepts
  below. Earlier the same item was refused with the items ahead chained up
  to 192 behind (t=144 to 159). **Settled by the fourth probe:** 59 is the
  entry point (67) less the item's move; measured from the entry, 123 is 56
  behind it, and the simulator already put it there. A sweep of the boundary
  found the rule ("Sideload acceptance").

### Turn item points

An item on a turn is at the point `get_line_item_position` gives for its
position: on the 1/256 grid, near quarter circles of radius 188 and 67 about
the inner corner, but not a circle rounded any simple way. The eight turns
are one table rotated, a left turn the mirror image of a right turn with its
lanes swapped, exactly. With the arm aiming at those points, four of the six
`tpick_*` rigs agree with the engine on every tick. From the side opposite
the feeding belt (`tpick_r_e`, `tpick_l_e`) energy agrees every tick but on
two ticks each, as the hand reaches an item, its x is 1/256 off (-346 where
the arm's own position rounds to -345, -237 where it rounds to -238): how the
engine rounds the arm at a target an exact number of 1/256 away is not
reproduced (float variants of the orientation, length and drawing tried).

### What the simulator needs

To reproduce items 1 and 3 exactly the simulator needs `d` for every tile an
entity can be built on, and the rules above: merge and re-arm timers,
boundaries and split timers, inserters watching segments, segments ordered
by activation with the catch-up move. Options:

1. **A measured table of `d`.** It is a pure function of tile and lane, so a
   table is exact on every map, where it covers. `--family delay` took 26 s
   for the 2,436 tile-lanes of the parity scenes' area. The
   simulator would then be exact inside the table and could refuse or flag
   belts outside it.
2. **Identify the hash**, which would make it general. Not found.
3. **Keep the gaps** as they stand.

Details still open under any option: the drop order of two inserters in one
tick (above), boundaries for sideloads and pickups in other geometries, and
what removing or rotating a belt of a merged segment does.

**Decision (2026-09-25): option 1**, done in the fourth probe: the table
covers the scene area and the probes' rig areas, and factory-sim reproduces
segments from it. The drop order is settled; the other two details were
settled by the fifth probe.

## Fourth probe: the delay map, update order, sleeping lines, acceptance

`tools/probe_logistics4.py` measures what factory-sim needs to reproduce
belt-line segments. It also settles the two details the third probe left
open. Evidence, all in `docs/evidence/`:

| file | what |
|---|---|
| `logistics4-delay.json` | where the merge-delay table factory-sim ships comes from: area, engine, method, corrections, the sha256 of its values, and every check below |
| `logistics4-delaycheck.json.xz` | 600 random tiles re-measured in fresh worlds: facing north twice, east, south and west once each |
| `logistics4-order.json.xz` | 14 rigs of two burner inserters acting on the same tick, in both build orders |
| `logistics4-sleep.json.xz` | 6 rigs: a stopped line freed three ways, on young and on old belts |
| `logistics4-accept.json.xz` | 1,296 sideloads and 81 drops arriving next to an item moving along the lane |

The full maps (`--family delaymap`, about 180 KiB each) stay in `runtime/`.
The compact table is in factory-sim (`fsim/data/belt-delay.*`), built by its
`tools/make_belt_delay.py`.

### The delay map

- **Area.** Scenes are built inside the scene box, which is 64 tiles either
  side of the origin by default (`Blueprint.radius`; factory-sim's
  generators use 48). The mod clears and reports neutral entities only
  inside it. The agent can walk beyond the box, and a belt_smelting task
  with sites 20 to 40 tiles apart still fits inside it. The table covers
  [-128, 128)² tiles: four times the box, with 64 tiles of room past it on
  every side. It also covers the areas the logistics probes built their rigs
  in, [96, 352)² and [296, 340) x [456, 480), so that factory-sim can
  replay them. That is 132,128 tiles and 264,256 tile-lanes, with `d` from 1
  to 600.
- **Method.** The method is the third probe's: a belt at every measured tile
  (six worlds, one lattice offset each), with its downstream neighbour
  destroyed and rebuilt every tick. The value is the first tick the two
  lines are equal. One 256 x 256 facing takes about 2.5 minutes on the
  desktop.
- **The method's one flaw.** When the neighbour's own tile has `d = 1`, the
  rebuilt neighbour's timer runs out before the next rebuild, and the
  reading is 1 whatever the measured tile's `d` is. Every rectangle was
  therefore measured twice, facing north (neighbour at y-1) and facing east
  (neighbour at x+1), and the two maps were combined:
  - equal readings stand;
  - where they differ, one reads 1, its neighbour's `d` is 1, and the other
    reading is the tile's `d`;
  - 459 north readings and 456 east readings were corrected this way, with
    no conflicts;
  - no tile read 1 with both neighbours at 1, so no tile was left undecided.

  The third probe's tables carry the same flaw: 17 of their 9,626 readings.
- **Checks.**
  - All 9,626 third-probe readings agree with the table, except those 17.
  - The 32 x 32 tiles measured in two rectangles agree, in both facings.
  - 600 random tiles were re-measured in fresh worlds, one run per facing.
    Facing north (twice), east, south and west, 1,196, 1,196, 1,200, 1,200
    and 1,198 of 1,200 tile-lanes agree. Every difference is the flaw above,
    in that run's own facing.
  - The two north runs are identical, so the measurement is deterministic.
  - `d` does not depend on facing, and lane 1 is the lane left of travel in
    every facing, as the third probe found.
- **The hash.** Not identified. `d` is uniform on 1..600, and the two lanes
  and neighbouring tiles are uncorrelated (|r| < 0.005). Two families of
  candidate were tried, with the tile or 1/256 coordinates and the lane
  encoded several ways, and mapped to 1..600 by mod 599/600/601 or
  multiply-shift:
  - hash finalisers: FNV-1a, splitmix64, MurmurHash3, lowbias32, Wang, PCG;
  - generators: LCGs, minstd, taus88 and mt19937.

  None beats chance: the best matched 17 of 3,000.

### Two inserters on one tick

The last-built inserter acts first, dropping onto a belt as well as into a
chest. The third probe had it the other way for belt drops because it
misread which inserter drops on which lane. A burner inserter drops 1.2
tiles out, past the belt's centre line, so it drops on the far lane: west of
a north feed it drops on lane 2, east of it on lane 1.

`order` built 14 rigs and gave each inserter a different plate, so the
engine's item ids on the belt record the drop order directly. In all 14 the
last built acted first:

- drops on a feed, with the chests built before either inserter;
- drops with no coal;
- inserters north and south of an east belt;
- two drops competing for one free chest slot;
- two pickups competing for one plate, dropped onto belts or into chests.

This agrees with `logistics_sideload_merge`: the east inserter, built
second, drops on lane 1, and its drop activates lane 1 first.

### Stopped lines sleep

- **A segment in whose update no item moves goes to sleep:** it leaves the
  activation order. It wakes, to the end of the order, when:
  - an item goes onto it: a drop, a sideload, an item crossing from the belt
    behind, or a script insert;
  - an item comes off it: an inserter's pickup or a script's `remove_item`;
  - a belt is built, removed or turned;
  - what it runs into moves. When the segment ahead on its chain, or the
    lane it sideloads onto, moves, a sleeping segment behind it wakes and
    moves at once, on the same tick.
- **`sleep`**: a line of seven young belts, stopped at its end, was freed
  three ways: a script took the front item, an inserter took the back item
  of the last belt, or an eighth belt was built on the end. Every item
  behind moved on the next tick (t=21, t=37, t=21), young belts and old
  alike.
- **Consequences:**
  - The catch-up applies to a stopped target. In the `ins_*` rigs the second
    pair of sideloads (t=139) lands on a lane holding two stopped items. The
    first sideload wakes the lane without moving it, and the second moves
    it, so the item inserted first goes 8 further.
  - A segment moves the one ahead of it first only if that one is awake.

### Sideload acceptance

`accept` placed one feed item so that it reaches the main belt as an item on
the target lane passes the entry point. It covered 40 positions either side,
every phase of the 8-tick step, and both feed lanes.

- **The rule.** Let E be the entry point (67 or 188) and p the feed item's
  position before its move. Positions are read on the target lane after
  this tick's move.
  - Items at or ahead of E are ahead of the new item.
  - The new item goes at the first place 64 clear of every item ahead, but
    only if that place is less than 64 behind E.
  - With nothing ahead within 64, it goes at E + p - 8.
  - An item exactly at E refuses the sideload, and the feed item waits a
    tick.
  - An item one step behind E is not consulted, and the new item goes on 2
    to 8 ahead of it.
- **The `three` case.** The "exactly 64" of the third probe is 56 in these
  terms, and the simulator already put that item there. Its rule measured
  from E + p instead of E, which the sweep showed wrong for items between
  the two points.
- **Drops.** Inserter drops past a moving item follow the simulator's rule
  at every phase (81 of 81).
- **Script inserts.** A script insert within 8 of a belt's downstream edge
  reads 0 on that belt, not on the next one.

### Timing, as factory-sim implements it

- **Merges.** A belt built between ticks at t0 merges on tick t0 + d. An
  inserter watching the chain acts on the merge tick itself (`flow2`: lane 1
  merges and the inserter refills at t=76).
- **Splits.** A drop, a drill's output or a sideload counts from the tick
  before the item first reads on the belt, since the insertion reads one
  step on. A pickup counts from its own tick. `logistics_belt_pickup`: drops
  read at t=47 split the line at 46 + 87 = 133. `logistics_smelting_chain`:
  the ore read at t=243 splits the lane at 242 + 492 = 734.
- **Wakes.** An item crossing onto a watched segment wakes the inserter on
  the same tick (t=803). A drop by another entity wakes it from the next
  tick, as before.
- **Boundaries.** Every attached entity's boundary counts at a split,
  whether it has worked yet or not: at t=133 the two pickup inserters of
  `logistics_belt_pickup` had taken nothing, and the line split at their
  boundaries too. The trigger is the first item an entity puts on or takes
  off the chain. **Fifth probe:** not so. Each entity's boundary is set off
  on its own. Those two pickups were waiting when the drops came, and a
  waiting pickup is set off then ("Boundaries" of the fifth probe).

### How factory-sim compares

With the table and these rules, factory-sim matches the engine exactly:

- **Parity traces:** all five logistics traces, free-running, synced and
  tick by tick. `KNOWN_GAPS` is empty.
- **Segment classes** of the three parity scenes the third probe logged,
  every tick.
- **First probe:** the rigs, from t=0. The young-belt tolerance is gone,
  and so is the `side_main` gap.
- **Second probe:** `side_both`, `side_turn_two` and all 22 `sim_*` rigs,
  every tick.
- **Third probe:** the 32 sideload rigs.
- **Fourth probe:** the 14 order rigs, the 6 sleep rigs and all 1,377
  acceptance rigs.

A belt built outside the table stops the simulator with
`fsim.BeltDelayMissing`, rather than guess.

**Not measured, and chosen** (all measured since, in the fifth probe):

- the place in the order of the pieces that a split or a removed belt
  leaves holding items: they move last, like a merge;
- what turning a belt does to its timers: nothing;
- belt loops: they move as a whole, before the segments.

**Still open:** boundaries for sideloads and pickups in other geometries,
and what removing or rotating a belt of a merged segment does. Settled by
the fifth probe.

## Fifth probe: changed belts, loops, boundaries, the arm's first move

`tools/probe_logistics5.py` measures what factory-sim had chosen or left
open about segments. A rig is data: a base tile and timed operations (build a
belt, a wooden chest, a burner inserter or a burner mining drill on iron ore
laid under it, `insert_at` an item, `rotate` or `destroy` an entity). The same rig list runs in the engine and in
factory-sim's `tests/logistics_rigs5.py`. Every tick it changes, each rig
logs every item on every belt (with the engine's ids), which belt lanes are
one line object (`line_equals`, compared across both lanes), every
inserter's hand and every chest. 366 rigs in twelve families, one evidence
file each, `docs/evidence/logistics5-<family>.json.xz`:

| family | rigs | what |
|---|---|---|
| `order` | 13 | two long feeds onto one old, empty main lane; an item on each lands on the same tick; one feed is split at a boundary, loses a belt or has one turned while its item rides it |
| `change` | 26 | lines of 8 belts, young and old: a belt turned (once, back, four times, 180 degrees), removed and built again; a turn turned; a pending split with a belt turned; merge timers re-armed |
| `feedchg` | 16 | a five-belt feed sideloading onto an old main: each feed belt near the front, and the main belts around the target, removed and rebuilt, turned, turned four times |
| `loop` | 15 | closed loops, young and old: 2 x 2, 4 x 3 and 6 x 4, sparse and compressed, fed by a sideload onto an empty or a busy loop lane, an inserter dropping onto one and one taking from it |
| `loop2` | 27 | 4 x 3 loops built from five different belts, along the flow and against it, young and old; a merged loop with a belt removed and rebuilt, or turned four times |
| `bound` | 54 | inserters picking and dropping from either side of an old line at its start, middle and end, on and next to turns, several on one line, some never working; feeds from either side, at the ends, before a turn, onto a west line |
| `dist` | 60 | a drop, a pickup or a feed at one belt followed by 11 patterns of straights and left and right turns |
| `drill` | 36 | a burner mining drill, built on an old line, dropping onto one belt from the north (lane 1) or the south (lane 2), followed by 34 patterns chosen so that their lane sums pass through every value from 468 to 768 a lane can reach; two drills dropping onto a turn |
| `loop3` | 80 | 2 x 2 loops, clockwise and anticlockwise, built from either of two belts: an inserter at each of the eight tiles outside, taking a plate put on the inner or the outer lane, or dropping two plates |
| `trig` | 22 | what sets a boundary off: idle and full-chest pickups, script inserts upstream, downstream and onto the pickup belt, idle drops, empty feeds, pickups built after the items |
| `trig2` | 9 | which boundaries a split uses; the delay a second split on an upstream piece counts |
| `trig3` | 8 | boundaries one, two and three belts apart, together and one after the other |

Each family ran in one to three fresh worlds on the laptop, 25 to 35 s each.

### The inserter's first move from a short buffer (`seg_*_8`, `seg_b_7`)

The four second-probe rigs set aside as "item added onto the pickup belt"
were not about segments. The item goes onto the belt that runs along the arm
into the sleeping inserter, 112/256 from its edge, so the extension to the
item, 0.0337 tile, is less than one step (0.035). The inserter wakes with the
810 J it kept, which buys 0.0162 tile. The model moved the arm 0.0162 and
paid the rest of the extension on the next tick (2,400 J where the engine
paid 2,164.31 J). The engine does what it does after a full step: with less
than a step then left, the arm is on the target. That tick's 810 J buy the
whole extension, and the next tick pays only the rotation plus the next
0.0303 tile. Rule 2 of "Inserter belt pickup", for a short buffer, becomes:
the extension moves in proportion to what it gets, and if less than EXT is
then left, it snaps to the target. With that the four rigs agree on every
tick. The `pe_*` rigs, whose buffers run out mid-swing, agree as before.

### Changed belts

Think of the edges between belts as nodes. Each belt's downstream edge
joins:

- the downstream edge of the belt feeding it;
- the downstream edge of the belt it feeds;
- for a feed's front (its sideload link), the downstream edge of the belt it
  sideloads onto.

When a belt is built, turned or removed, **each belt on the four tiles next
to it** (connected to it or not) has every edge within 1 of its own
downstream edge cut, within 2 for a removal. Rotations cut before and after
the turn. On a straight line this cuts:

| change at belt k | belts left on their own |
|---|---|
| built, or turned (even four times in one tick, back where it was) | k-1, k, k+1, k+2 |
| removed | k-2, k-1, k+1, k+2, k+3 |
| a feed's front built | the target belt and the one after it |
| a feed's front removed | the target belt, the one before it and the two after |
| a feed belt one short of the front built / removed | the edge after the target / the target belt and the one after |

A belt running past a new feed's side, like the column of a turn that
comes back alongside the feed, has its own cut (`dist` `side_s_RSSSS`,
`side_n_LRSSS`: two and four more belts than the target's cut).

**Timers.** Every piece next to a cut restarts its merge timer at the tick
of the change, with d of the piece's head lane, when the cut parted lanes
that were one. A piece that was not merged across the cut keeps a timer that
is still running (`rm_*_young`), and starts one that has run out
(`rot_k3_old180`). So after a removal the pieces merge again at the tick of
the removal plus the least of their heads' delays: `rm_k0_old` at 650 + 53,
where 53 is d of the remaining piece's head (m7), not of any belt cut loose.
A turn of a young belt in place changes nothing that was not merged.

**Membership.** A turn or a removal parts a merged segment only where the
cut falls. The rest stays one object: `rot_k7_old` keeps m0..m5 together,
and `rot_turn_old` keeps m0..m2.

### Where the pieces go in the order

The `order` rigs are sensitive both ways. Moving the split piece to the end
of the order, or to its front, fails one of each pair (`split_S1`/`S2`,
`rm_S1`/`S2`, `rot_S1`/`S2`).

- **The piece holding the old head keeps its place**: after a split at a
  boundary, a removed belt or a turned one, and when the head's belt is cut
  off on its own (`rm11_*`).
- Pieces with a new head that hold items keep moving: last in that tick when
  the segment was awake, asleep with it when it was not. Which of them moves
  first cannot show: they only feed the piece ahead, which a segment always
  moves first.

### Loops

- **A loop is a chain whose front is its oldest belt.** Its lanes merge like
  a chain's, the whole loop at the first timer to run out. A loop that is one
  segment has its seam at the downstream edge of the lane of its oldest belt
  (`loop2`: built from l0, l1, l3, l5 or l8, along the flow or against it,
  the seam is after the first belt built; after l0 is removed and rebuilt,
  after l1).
- **A one-segment loop moves as one line from the seam.** Its front item is
  held by nothing ahead. An item that crosses the seam onto the loop's own
  back moves again, as far as it went in: 6 before the seam reads 256 - 4
  after it, 0 reads 256 - 16 (`rect_*`, `sq_*`, `big_old`). A script insert
  within 8 of the seam reads 0 before it (`sq_old`).
- **A loop of several segments moves segment by segment**, in the activation
  order. Each moves the one ahead first, as on a chain. Round the loop, that
  comes back to the segment the move started from. An item crossing into it,
  which has not moved yet, moves a full step again when it does
  (`drop_old`: 2 reads 106 - 14).
- A boundary in force on a loop parts it at the seam too (`feed_old`: pieces
  l8..l0 and l1..l7). A loop broken open parts at the seam, and that piece
  restarts its timer (`rm_l6`: l0 alone, merging again d of its own after the
  removal).

### Boundaries

An entity marks a boundary on the lane it works on:

- an inserter picking up: both lanes of the belt;
- an inserter dropping: the lane it drops on;
- a drill: the lane its output lands on;
- a feed: the lane it sideloads onto.

**Where.** The boundary is at the upstream edge of the belt holding the
point R along the lane from the downstream edge of the entity's belt,
searching toward the front of the chain. A turn's lanes are 295 and 106
long.

- Inserters: R lies in (618, 657], bracketed on its own for drops and for
  pickups, on both lanes.
- Drills: R lies in (618, 657] too, measured on its own (`drill`: the lower
  end from 9 rigs whose lane reaches 618, the upper end from 7 that reach
  657, both lanes, and a drill dropping onto a turn).
- Sideloads: R lies in (362, 401] from the target belt's edge.

On a straight line that is k+2|k+3 and k+1|k+2, as before. After an inner
turn it is one belt further: 256 + 106 = 362 is not enough. The 20 drop, 20
pickup and 20 feed rigs of `dist`, the 36 of `drill`, the turn rigs of
`bound` and the loop rigs all agree.

**On a loop the search stops at the front.** It does not pass the loop's
front lane, the lane of its oldest belt, whose downstream edge is the seam.
On the outer lane of a 2 x 2 loop (`loop3`, 295 a lane), from the belt two
behind the front the boundary falls between the belt behind the front and
the front (295 + 295 + 295 = 885 holds R), and the loop splits there and at
the seam. From the front itself, or from the belt behind it, the search
reaches the front first, and nothing splits (a search round the loop would
have split both). On the inner lane (106 a lane) the front is at most 318
away: no inserter, picking up or dropping, splits a 2 x 2 loop there.

**The intervals cannot be narrowed further, and need not be.** The only
lane sum strictly inside (618, 657) is six inner lanes, 636, which only a
2 x 2 loop has, and there the search meets the front after three lanes.
None lies strictly inside (362, 401). So every value in each interval gives
the same boundaries on every belt geometry factory-sim models, and no rig
can tell them apart. factory-sim uses 640 and 384.

**When: a boundary is set off**:

- by the entity's first item: a drop, a drill's output, or a feed's item
  arriving, a tick before the item reads on the belt;
- for a pickup, by the first time the inserter looks for an item on that
  lane. That is:
  - when it chooses one on its pickup belt: only that lane
    (`pick_k0_both`: with items on both lanes of its pickup belt, the near
    lane at the insert, the other when it has dropped the first item into
    its chest);
  - or, with none on its pickup belt, when it waits while its lane's
    segment holds an item it would take. An item added to the segment
    elsewhere makes a waiting inserter look as of that tick: a script's at
    the insert, an entity's a tick before it reads.

Set off nothing:

- an idle inserter;
- an inserter whose chest is full (`pick_full`);
- an inserter dropping from an empty chest (`idle_drop_*`, `idle_drop_k7_drop_k2`);
- a feed nothing has come down (`side_put_*`);
- a script insert by itself.

The fourth probe's "every attached entity's boundary counts at a split" was
this. In `logistics_belt_pickup` the pickups were waiting when the drops came
at t=46, and were set off then.

**Split.** When a boundary is set off at tick T, the lane's chain is made
into its pieces again at T plus d of the chain's front lane (not of the
piece's head: a second split, on the upstream piece, counts the same d,
`seq_a`/`b`/`c`). "Made into its pieces" is what a merge does. Every edge is
joined but those at the boundaries in force, and those are parted even
where the lanes were one. So belts cut apart by a turn in the meantime are
one again (`split_rot_k5_4x`), and a boundary set off a few ticks before
another's split goes with it (`multi_side_k2_drop_k6`). A merge by a timer
does the same (`turn_side_before`: a merge at t=770 parts m6|m7).

**Close together.** Take the entities in force on a chain from upstream.
One whose belt lies between the belt of the last one kept and that one's
boundary marks none. So of two drops one or two belts apart only the
upstream one counts (`drops_k3_k4`, `drops_k2_k4`, `drops_k2_k3_k4`); three
apart both count (`drops_k2_k5`).

- If the downstream one comes later, nothing happens (`late_down`).
- If the upstream one comes later, then at its split its boundary appears
  and the downstream one goes, its two pieces one again (`late_up`: m5|m6
  parted and m6|m7 joined, both at t=807).

### Script inserts

A script insert within 8 of a belt's downstream edge reads on the next belt
(`256 - 8 + p`) when the lane is merged with it (`bound` pickups, t=620: 0
reads 248 on the next belt). It reads 0 on its own belt when the lane is a
segment of its own, as in the fourth probe, and before a loop's seam.

### How factory-sim compares

With these rules factory-sim matches all 366 rigs, every reading, every
tick (`tests/test_segments5.py`). The four second-probe rigs `seg_a_8`,
`seg_ar_8`, `seg_b_7` and `seg_d_8` are compared again
(`test_mechanics_logistics2.py`). Every earlier logistics test still passes.
Those are:

- the five parity traces, free-running, synced and tick by tick;
- the segment classes of the third probe;
- the first and second probes' rigs;
- the third probe's sideload rigs and the fourth probe's order, sleep and
  acceptance rigs.

### A loaded state and the activation order

A load could leave factory-sim with one placement of its own: where in the
activation order a segment goes when a loaded hidden state gives it items
the simulator did not have there. Whether the engine's order could be loaded
instead:

- **It cannot be read.** `LuaTransportLine` in 2.0.60 (`runtime-api.json`)
  has `can_insert_at`, `can_insert_at_back`, `clear`, `force_insert_at`,
  `get_contents`, `get_detailed_contents`, `get_item_count`,
  `get_line_item_position`, `insert_at`, `insert_at_back`, `line_equals`,
  `remove_item`, `input_lines`, `line_length`, `output_lines`, `owner` and
  `total_segment_length`: nothing about whether a line is awake or its place
  in the update order. `LuaEntity.active`, `disabled_by_script`,
  `is_updatable`, `frozen` and `status` are entity flags, and `unit_number`
  gives creation order, which the activation order does not follow: it
  follows when items arrive on empty segments.
- **It can be inferred from the history**, by replaying it, which is what
  factory-sim does. The one-step sync check resets once and steps alongside
  the recording, loading the recorded state over its own at every decision.
  It keeps its own segments: membership, timers, boundaries set off, sleep
  and order.
- **The placement is never reached by a check.** A load changes the order
  only through belt contents that differ from the simulator's. Over the one-
  step sync of all 18 golden traces (2,520 loads) no load changed a single
  belt lane. It can only happen after the simulator's belt contents have
  already differed from the recording, which the comparison after the
  previous step reports, or when the first recorded state differs from the
  installed scene, which free-running reports. Free-running and
  tick-by-tick checks never load.

It would matter only for loading an engine state into a simulator that has
not replayed its history, which nothing does now. There more than the order
is missing: segment membership (which `line_equals` could capture), merge and
split timers, boundaries set off, and sleeping segments.

**Decision (2026-09-25): refuse.** `Sim.load_hidden` checks before it loads
anything. A load that would change the items (name and position; ids are
only names) of a belt-line segment that still holds items afterwards raises
`fsim.BeltOrderUnknown`, as a belt outside the delay table raises
`fsim.BeltDelayMissing`, rather than guess. A load that leaves the belts as
the simulator has them, or that empties a segment, goes through: an empty
segment is in no order. Every one-step sync of the golden traces still
passes (`tests/test_parity.py`), and `tests/test_load_order.py` triggers the
error.

## Hand-mining and reach (v3 decisions)

`tools/probe_handmine.py`, raw Lua, every tick recorded. The `mine` family
drives the character the way the mod's `POLLS.mine` does (selection and
`mining_state` re-asserted at the tile every tick) over 27 rigs
(`docs/evidence/handmine-mine.json.xz`); the `reach` family asks
`can_reach_entity` for seven entity types from 16 x 16 sub-tile character
positions at every whole-tile offset up to 12, 1.12 million positions, plus
rays at 1/256 (`docs/evidence/handmine-reach.json.xz`).

### Hand-mining

- **Speed and time.** `mining_speed` 0.5, modifier 0; iron ore, copper ore,
  coal and stone all have `mining_time` 1. Progress is seconds / mining time,
  seconds growing by 0.5 / 60 a tick, and reads 0 on the tick mining is first
  asked for. The first item arrives **121 ticks** after the ask, each next one
  120 later, whatever the character's sub-tile position (`time_*`,
  `subtile_*`).
- **When an item comes.** Once progress passes 1, strictly: a pile (0.025 s)
  reads exactly 1 on its third tick and is taken on the fourth; a chest (0.1 s)
  on the thirteenth; a drill (0.3 s, whose seconds pass 1 on the 36th) on the
  36th. factory-sim compared `<` 1 before this; only the pile case differed.
- **Amount.** Each item takes 1 from the resource; at 0 the entity is gone and
  nothing more is mined (the mod's poller then settles the action).
- **Interrupted.** Stopping keeps the progress for that target: resumed the
  same tile after 60 idle ticks, or after 1, it continues from where it was
  (`stop_resume`, `stop_one_tick`). Mining another tile starts from 0, and
  going back to the first starts from 0 again.
- **Walking.** Asked to walk while mining, the character does not move and
  mining goes on unchanged; asked to mine while walking, it stops walking
  (`walk_while_mining`, `mine_while_walking`). The mod's `move` cancels a
  running mine first, so neither happens through the action space.
- **Reach.** The engine makes no progress on a tile beyond resource reach
  (`beyond_reach`); the mod refuses such a mine before it starts.
- **Full inventory.** The mod refuses a mine when the main inventory has no
  empty slot, even if a part stack of the item has room (`full_partial_stack`
  shows the engine would take it). When the inventory fills during a mine, the
  next item still comes off the resource (amount -1, and it counts as produced)
  but lands on the ground at the resource's position, one item; the pile then
  lies over the tile and is mined instead, every fourth tick, and never taken
  (`full_no_room`, `full_partial_stack`, `full_mine_pile`). Where the centre
  is taken, it lands round it: "Mining at the edges" below.
- **A tile something stands on.** The mod mines a resource by selecting at its
  position, and selection prefers any entity whose selection box holds that
  point: a belt, chest, inserter, wall, furnace, drill or pile over the ore is
  mined instead and goes into the inventory, and then no resource is mined
  while mining stays asked for (`cover_*`, `under_drill`), only another entity
  at that point ("Mining at the edges") -- the mod's mine then runs until a
  move cancels it. Stopping and asking again mines the ore
  (`cover_then_retry`). A pile 0.203 tiles off the tile centre (where a drill
  or inserter drops) is not over it (`cover_pile_off_centre`); the character's
  own box never is (`self_cover_*`); a pile cannot lie on a belt.

Selection-box half-sizes: chest, belt and wall 0.5, inserter 0.3984375,
furnace 0.796875, drill 1, pile 0.16796875.

### Reach

- **Entities.** `can_reach_entity` is true exactly when the straight-line
  distance from the character's position to the entity's **collision box** is
  at most 10 (`reach_distance`), boundary included: no exception in 1.12
  million positions, and every ray's last accepted position is at distance
  10.0. The selection box, the bounding box and the centre each disagree on
  thousands. Collision half-sizes: chest 0.34765625, belt 0.3984375, inserter
  0.1484375, wall 0.2890625, furnace and drill 0.69921875, pile 0.13671875.
  It gates `mine_at`, `rotate_at`, `give_to` and `take_from` alike.
- **Placing.** The mod refuses a placement farther than `build_distance` (10)
  from the character's position to the requested position (a slot's tile
  centre), straight-line; the entity's size does not enter. That is the
  dx = -7 / dx = +7 asymmetry: the same slot is 9.9 tiles away from one
  sub-tile position and 10.2 from another.
- **Hand-mining.** The mod refuses a resource farther than
  `resource_reach_distance` (2.7) from the character's position to the
  resource's centre, straight-line.

### Mining at the edges

`tools/probe_handmine2.py`, the same harness, 13 families
(`docs/evidence/handmine2-*.json.xz`): piles and entities mined with part of
the room they need, where what does not fit lands, a belt built over piles,
which of several coverers is mined, and the character carried by a belt while
it mines.

- **A pile with part of the room.** The inventory takes what fits and the pile
  stays with the rest; mining goes on, and once there is room the rest is
  taken (`pile`). A pile made by a full inventory or a drop holds one item, so
  through the action space this does not arise.
- **An entity with contents.** What it holds goes first, each stack as much as
  fits, in order: a chest's slots from the first; a furnace's fuel, source,
  result; a drill's or an inserter's fuel. The first stack that does not all
  fit keeps the entity standing: the next stack is not tried even when it
  would fit (`entity`, `extra`), and the character mines it again and again,
  each completion taking what now fits (the mod's mine runs until a move
  cancels it). With everything out the entity is removed, and then what cannot
  keep it standing follows, into the inventory or onto the ground round where
  it stood: a belt's lane 1 then lane 2, front to back; a furnace's ingredient
  in progress; the entity's own item; a drill's pending ore or an inserter's
  hand.
- **Where it lands.** On a grid of 88/256 round the drop point: the point
  itself, then ring after ring, each clockwise from its top-left corner (top
  row left to right, right column down, bottom row right to left, left column
  up), at the first point where a pile's collision box (0.13671875) overlaps no
  other pile and no colliding entity's collision box; the character does not
  block, and water was not measured. One item a pile (`spill`, `cover`). A full
  inventory's ore does the same round the tile centre, so with a drill's pile
  52/256 north of the centre -- too far off to be selected there (a pile's
  selection half-size is 43/256), near enough to keep a drop off it -- the ore
  lands round it and mining goes on
  (`hand_mine_spills` fills ring 1 without its top-middle point, then ring 2
  from the one point the drill's box leaves, clear of it by 2/256).
- **Carried while mining.** A belt carries the character while it mines; the
  target is not pinned. Each tick mines from where the character's walking
  step left it, before the belt's carry, so the tick the carry takes it out of
  reach still mines; while the target is out of reach from where the character
  stands, progress reads 0, nothing is mined, and what was is kept: back in
  reach, carried or walked and asked again, it goes on from there (`carry`,
  `reenter`, `reach`, and the `hand_mine_carried` trace: 26 ticks kept where
  the last reading was 25). Reach per tick is the rule of "Reach" above: a
  resource within 2.7 of its box (25/256), an entity within 10 of its
  collision box.
- **Which coverer.** Selection at a point takes a building over a pile, and
  among piles the one nearest the point, the newest on a tie (`select`,
  `select2`). After an entity is taken, mining still asked for mines only
  another entity at the point, never the resource under it, until mining stops
  (`cover`). A building can stand over a pile: a pile a full inventory left on
  a tile centre stays under a chest built there, and mining the tile takes the
  chest, then the pile (`hand_mine_build_over`).
- **A belt built over piles.** It takes the piles on its tile onto its lanes,
  the newest first, each where a drop at its position would go and behind
  what is already ahead of it on that lane; one that would land more than 64
  behind its point is dropped round the belt instead (`beltpick`,
  `beltpick2`, `beltpick3`). An item pushed past the lane's upstream end reads
  at the lane's last position. A jammed drill whose pile is taken this way goes
  on onto the belt (`hand_mine_build_over`).

factory-sim implements each rule; four parity traces pin them through the
mod's own actions, tick by tick: `hand_mine_spills`, `hand_mine_contents` (a
chest stopping at its plates with its wood behind them, a furnace's fuel before
its ore, a loaded belt and a jammed drill with no room for most of what they
hold), `hand_mine_build_over` and `hand_mine_carried`. What is not exact yet:

- with nine piles under a new belt, five or six of them going onto one lane,
  the engine refused one item more than the rule in 4 of the 5 `bp_ring9_*`
  rigs (`beltpick2`); the other 48 rigs with piles under a new belt agree. Not
  reduced to a rule;
- which of several piles on one tile a tile handle resolves to (the mod's
  `find_entities_filtered{limit = 1}`): factory-sim takes the first made;
- a belt built over piles next to a loaded belt, and water in a drop's rings:
  not measured;
- a one-step sync cannot load the progress kept out of reach, since the engine
  reads 0 there; factory-sim keeps its own when a recorded 0 falls out of
  reach.

### Decisions (user, 2026-09-25)

1. **Hand-mining is in v3.** `parameterized-v3` gains `mine_tile` at index 22,
   after every `v1` operation: a resource tile named by placement slot (the
   tile the grid planes already show the ore on) and a count from the amount
   dimension (1, 5, 20), sent to the mod's `mine` action as `mine_at` is.
   Target-by-tile rather than new rows, because resource tiles have no row in
   the entity table and a row per tile would crowd out the entities. The
   action vector was `MultiDiscrete[23, 97, 226, 5, 19, 4]`; with `take_fuel`
   and `finish` (below) it is `MultiDiscrete[25, 97, 226, 5, 19, 4]`. factory-sim
   implements every rule above, pinned tick by tick by the `hand_mine_rules`
   parity trace (a spill with the inventory filling mid-mine, a refusal when
   full, the ore tile under a chest); `WorldV3.mine_resource(x, y, amount)`.
2. **The v3 mask is legal where the game accepts.** A target row is legal
   when it is visible and within reach by the rule above; a placement slot when
   its tile is free and its centre within build distance; the same slot is
   also legal for `mine_tile` when a visible resource tile within resource
   reach lies on it. What a factorised mask could not say was left to the
   decoder or the game: the placement dimension shared by `place_at` and
   `mine_tile` (the union), a row shared by every verb that takes one, a 2x2
   machine's other three tiles and an item's own collision (the game's
   `can_place_entity`), and "no free inventory slot" (the mod's refusal).
   Superseded by the per-operation masks ("v3 masks: per operation", below),
   which keep only the placement-per-item case.
3. **Remembered rows carry no belt or inserter detail.** Accepted as is:
   memory keeps a record's name, type, position, direction, contents and
   recipe, nothing else. Under decision 2 a remembered row is also masked.
4. **Handcrafting stays off for Stage 1.** `belt_smelting` hands out a fixed
   inventory, so nothing needs crafting, and an unmeasured mechanic would break
   parity. It is measured and implemented exactly together with Stage 2's
   recipe dimension.

## v3 masks per operation, `finish` and taking fuel (user decisions, 2026-09-25)

One revision of the v3 action layout, before anything trains on it. The
action vector is `MultiDiscrete[25, 97, 226, 5, 19, 4]`: `parameterized-v1`'s
22 operations, `mine_tile` (22), `take_fuel` (23) and `finish` (24); the flat
mask is 376 entries, the per-operation masks 25 x 351 bits a step (25 rows of
88 hex digits in a trace), and the self vector 13 slots. `parameterized-v1`'s
digest (`7222fb372fe51f63`) and every `v1` and `v2` tensor, mask and vector
are unchanged; `parameterized-v3`'s is `6cba96b001f19107`.

### What the engine accepts, measured

`tools/probe_inventory.py` reads the facts off the running engine
(`docs/evidence/inventory-prototypes.json.xz`, `inventory-fuel.json.xz`), and
`factoriorl.inventory_rules` holds them, pinned to that evidence by
`tests/unit/test_inventory_rules.py`; factory-sim carries the same table
(`csrc/fsim_rl.c`).

- **The character.** 80 main inventory slots. Stack sizes: 100 for plates,
  gears, belts and wood, 10 for a steam engine, 20 for an offshore pump, 50
  for everything else in `ITEMS_V3`.
- **Fuel.** Coal (4 MJ) and wood (2 MJ); both chemical.
- **Turning.** `LuaEntity.rotate` turns a belt, an inserter, a drill and a
  boiler; a steam engine and an offshore pump take a direction but refuse to
  turn once built.
- **Inventories.** A wooden chest's 16 slots take anything; a burner's fuel
  inventory (drill, furnace, inserter, boiler) is one slot that takes coal or
  wood; a stone furnace's source is one slot that takes iron ore, copper ore
  or stone, 54 of them by one insert (over the stack of 50); its result slot
  takes anything by script, which is where a give of coal lands once the fuel
  slot is full.
- **Mining.** A ground pile is minable but yields nothing, and the mod's
  `mine` refuses it ("yields nothing"); a pile over a resource tile shares the
  tile's handle, which resolves to the resource, and is mined through it
  ("Mining at the edges").
- **Part of the room.** The mod's transfer takes `min(count, available)` if
  `can_insert` says at least one fits, inserts what fits, puts the rest back
  and reports `rejected`/`no_space` -- and what moved stays moved. With no
  room at all nothing moves (`take_partial`, `give_partial`,
  `take_fuel_partial`).
- **Fuel out of a burner** (decision 3 below). A transfer from a drill,
  furnace, inserter or boiler naming its fuel item comes out of the fuel slot,
  inventory 1 (`defines.inventory.fuel`, the number `chest` shares): the mod's
  `inventory_of` tries it first. More than is there moves what is there. The
  item burning is not in the slot: it stays in the burner and burns on (its
  `remaining_burning_fuel` runs down unchanged), so a furnace given one coal
  has an empty slot from its first tick and nothing to take.

### The decisions

1. **v3 masks: per operation (option C).** Per operation, a mask over each
   argument dimension (target, placement, direction, item, amount): a value is
   legal when some whole argument combination holding it is one the game
   accepts, and an operation when it has one. A dimension an operation does
   not read offers only its sentinel, as does every dimension of an illegal
   operation, so no row is ever all false. The flat mask stays, as the
   operations then per dimension the union of the legal operations' rows, so
   stock sb3's `MaskableMultiCategorical` runs on it unchanged. FactorioRL:
   `ParameterizedEnv.operation_masks()` (and `packed_operation_masks()` for a
   record); factory-sim: `fsim_rl_opmask3` and `RlEnv.op_masks()`, identical,
   pinned per decision by the contract golden's `op_masks`
   (`tools/v3_contract_golden.py`, factory-sim `tests/test_rl_contract.py`).
   The rules, all from what the observation shows:
   - `place_at`: an item held that places an entity and whose recipe is
     enabled, a free slot within build distance, any facing;
   - `mine_at`: a visible row within reach, not a pile off a resource tile;
     a pile on a resource tile within resource reach of the tile's centre;
     no mine running and a free main slot (the mod refuses both);
   - `mine_tile`: a slot holding a visible resource tile within resource
     reach, the same two refusals;
   - `rotate_at`, `rotate_at_reverse`: a visible row within reach that turns;
   - `give_to`: a row within reach and an item held that one of its
     inventories takes at least one of, by the room its record shows;
   - `take_from`: a row within reach holding the item (fuel included, as the
     mod reads it), the item in `take_from`'s domain, and room for at least
     one;
   - `take_fuel`: a row within reach whose fuel slot holds fuel, and room for
     at least one of it;
   - `finish`, `wait` and the moves: always.
   Room is computed from the item totals the observation carries, with every
   other item taken to lie in whole stacks: an upper bound, so a legal action
   is never masked and "no room" is exact.

   Known observable limitation (user decision): placement per item is not in
   it. A 2x2 machine's other three tiles and an item's own collision are still
   the game's `can_place_entity`, refused `collision`. And the rows are
   projections: a target legal for one item and an item legal for another
   target can make a pair the game refuses (a belt given to a furnace), which
   is the game's refusal, never a decode failure.
2. **The free main slots are observed.** `local-v3` v2 publishes
   `character.slots = {free, total}` (`count_empty_stacks` and the slot
   count), and the v3 self vector's 13th slot is `free / total`. It is what
   the mine refusal reads; item counts alone cannot tell "full" from "a part
   stack has room".
3. **Taking fuel is its own verb, `take_fuel`** (op 23): a row and an amount;
   the item is the one the fuel slot holds, read from the record. Chosen over
   a fuel option on `take_from` because the per-operation mask gives it its
   own row -- the burners with fuel in the slot and room for it -- with no item
   dimension to fill (a fuel slot holds one item), while `take_from` keeps its
   `v1` meaning and domain. `take_from` naming a fuel item already takes from
   the fuel slot when that item is in its domain (the mod tries the fuel
   inventory first), as measured, and its mask says so.
4. **`finish`** (op 24): no arguments, always legal. The task's verification
   window runs at once through `run_verification`, the path the exhausted
   budget takes, and the episode terminates on its result; nothing is sent to
   the mod. A task without a window ends on its success condition as it
   stands. factory-sim: `rl_finish`; the program API: `WorldV3.finish()`, and a
   `v3` program's return is its `finish`.

Pinned by the `v3_take_fuel` trace (fuel out of a drill, a furnace and an
inserter mid-burn, with room for part of it, none, and after room is made;
tick by tick) and the `v3_finish` trace (a `mine_tile`, then `finish` with a
line running), both recorded under the `v3` catalog and `local-v3`; every
trace's `local-v3` observations (`--sensor`) are also factory-sim's own
rendering, checked decision by decision (`test_the_local_v3_sensor`).
factory-sim's `Policy(action_space="v3")` draws the arguments under the
sampled operation's row and scores a stored action under the row of the
operation it stores, so the PPO ratio is taken under the masks the action was
sampled under (`tests/test_policy_v3.py`); there is no `v3` training loop yet
(the batched `VecEnv` is `v1`/`v2`).

## Open questions and rigs to settle them

1. **Sideload with traffic.** The single-item rule is exact; entries of 172 and
   187 under contention are not. Rig: one feed lane at a time, pairs of items
   at controlled gaps (via `insert_at`) arriving on an empty and on a moving
   main lane.
   **Third probe:** the 172 is the catch-up move of an empty target segment;
   the 187 is the spacing rule (64 behind the item ahead); see "Update order
   and simultaneous sideloads".
2. **Where a drop lands when the drop point is taken on a moving line.** The
   unblocked drill put its item at 192 (read 184) behind an item at 128,
   while on a stopped line 192 was refused. Hypothesis: the item goes at the
   first free position at or behind the drop point, accepted only if it is
   less than 64 behind the drop point, with the insertion seeing the line
   before this tick's move. Rig: an inserter holding an item over a slowly
   filled moving lane, items placed with `insert_at` at 120..200.
3. **Lane choice on pickup**: settled by "Inserter belt pickup" rule 4 (the
   nearer lane, then the furthest-upstream item; the `bend` rig's lane 2 is
   its near lane) and, for inserters on both sides of one belt, rule 7.
4. **Left turns**: settled by the second probe (lane lengths mirror). Still
   open: where an item sits on a turn's arc for an inserter picking from it
   (`tpick_*`), and simultaneous sideload arrivals (`sim_*`).
5. **The energy mechanism**: settled, 50 kJ per turn plus 50 kJ per tile (see
   "Energy" and rule 2). The young-belt wake delay of rule 6: settled by the
   third and fourth probes (segments).
6. **Coal exhaustion** was not reached in 3,000 ticks (4 MJ is about 4,550
   ticks at 66,900 J per item); the wood run covers the stop behaviour.
7. **Self-refuel timing**: settled by the `sr_*` rigs (the arm turns to its
   own slot from wherever it is when the slot empties).

## Recorded state

What the parity recorder (`tools/record_parity_trace.py`) writes for belts,
burner inserters and wooden chests, so a simulator can export the same thing.
It comes from `world.hidden_state` in `mod/factoriorl/world.lua` and reaches
the trace through `Normaliser.hidden`. Records for every other entity type keep
exactly the keys they had before, so traces recorded before these fields
existed still match.

The traces are recorded under the task's `local-v2`. `--sensor` replays each
under `local-v3` and keeps its wire observations (`<name>.local-v3.jsonl.xz`),
for the `v3` contract (`tools/v3_contract_golden.py`); everything else the
replay records -- tensor hashes, mask, goal, truth, hidden state, transitions
-- is checked identical to the trace, and the observations differ only by
`local-v3`'s additions (belt `lanes` and `shape`, inserter `pickup`, `drop`
and `held`, drill `drop`, the profile's name and version). All 22 traces
replay identically, and the `v3` hashes and masks encoded from the sensor's
own observations are the ones the earlier mapping from `hidden` gave.

Common to every record in `hidden.entities`, as before:

- **`position`**: `[x, y]`, integers, map coordinates in 1/256 tile. A belt,
  inserter or chest sits on its tile centre, e.g. tile (4, 1) is `[1152, 384]`.
- **`direction`**: 0..15 (north 0, east 4, south 8, west 12).
- **`status`**: the `defines.entity_status` name.
  - belts read `working`;
  - chests read `normal`;
  - inserters read `working`, `waiting_for_source_items`,
    `waiting_for_space_in_destination` or `no_fuel`.
- **`energy`**: a decimal string, `%.17g`.
- Entities are sorted by `position[1]`, then `position[0]`, then `name`.

### Transport belt (`transport-belt`)

| key | shape | meaning |
|---|---|---|
| `belt_shape` | `"straight"` / `"left"` / `"right"` | `LuaEntity.belt_shape`. East into south is `"right"` |
| `lanes` | `[lane1, lane2]` | lane 1 is left of the direction of travel, lane 2 is right |
| `lanes[i]` | list of `[name, position, id]` | items on that lane of this belt, from `get_transport_line(i).get_detailed_contents()` |
| `position` (in an item) | integer, 1/256 tile | distance from **this belt's** downstream end, `floor(p*256 + 0.5)` |
| `id` (in an item) | integer >= 1 | item identity within the episode (see below) |

Other keys on a belt record:

- `energy` is `"0"`;
- `inventories` is `{}`;
- there is no burner field.

Position ranges:

- a straight line runs 0..255;
- the outer lane of a right turn runs 0..294, and the inner lane 0..105.

Items are listed in ascending `position`, with ties broken by engine id. The
same item keeps its id as it crosses from one belt to the next.

**Ids are renamed.** The engine's `unique_id` comes from a counter that
carries on across episodes, so it differs between two recordings of one world.
`Normaliser` renames ids 1, 2, 3, ... in the order the trace first meets them:

- records in trace order;
- within a record, entities in the order above;
- lane 1 before lane 2;
- items in list order.

A simulator reproduces the names by assigning any unique id per item and
running its exports through the same renaming (a copy of `Normaliser._item`).
Items that first appear in the same record are named in list order, not in
creation order.

When the recorder checks a tick trace against its decision trace
(`comparable_hidden`), item ids are dropped. Two items made within one decision
are met in creation order at tick resolution and in list order at decision
resolution. `[name, position]` must still agree.

### Burner inserter (`inserter`)

| key | shape | meaning |
|---|---|---|
| `direction` | 0..15 | points at the **pickup** side (north picks from y-1) |
| `held` | `{"name": str, "count": int}`, **absent** when the hand is empty | `held_stack` |
| `held_stack_position` | `[x, y]`, 1/256 tile | where the hand is drawn; always present |
| `pickup_position` | `[x, y]`, 1/256 tile | north-facing at tile (x, y): `[256x+128, 256y-128]` |
| `drop_position` | `[x, y]`, 1/256 tile | north-facing at tile (x, y): `[256x+128, 256y+435]`, i.e. 1.19921875 tiles out |
| `energy` | decimal string, J | buffer, up to 2560 |
| `remaining_burning_fuel` | decimal string, J | `burner.remaining_burning_fuel` |
| `currently_burning` | item name, absent when nothing | `"wood"` on a freshly built inserter (the built-in quarter wood), then the fuel it loads |
| `inventories.fuel` | `{"size": 1, "stacks": [[1, name, count]]}` | fuel slot |
| `inventories.burnt_result` | `{"size": 0, "stacks": []}` | always empty |

`held_stack_position` is the drawn hand (see "Hand position" above): its x
follows the arm state of "Inserter belt pickup" exactly, its y adds a drawn
lift during swings that is not modelled. A simulator that models the arm
reproduces x but not y. Relaxing y on the simulator side is a decision for
that side; this record keeps the engine's value.

### Wooden chest (`container`)

- `inventories.chest` is `{"size": 16, "stacks": [[slot, name, count], ...]}`,
  slot by slot (1-based, only non-empty slots). This was already recorded for
  containers.
- There is no fuel or burnt-result key.

### World digest

The per-decision `digest` (a hash of `world.digest()`'s lines) appends fields
only to the lines of belts and inserters:

- **belts** get `|shape=<belt_shape>|lanes=1:<name>@<pos>,...;2:...`. The line
  carries no item ids, because they are not state a reset restores;
- **inserters** get `|held=<name>=<count>`, or `|held=` when the hand is empty.

Every other line is unchanged.

### Scenarios

Five scenarios in `tools/record_parity_trace.py` carry `"requires":
"logistics"` in `docs/evidence/sim-parity/index.json`. Each has a decision
trace and a per-tick trace.

- They run on `construct_smelting_line` at 30 ticks per decision, with the
  family's scene replaced by a pre-placed rig in the header's `blueprint`.
- Rigs are laid out relative to the `patch` marker, which is not always
  (0, 0): the family comes from the scenario's seed.
- A simulator that does not yet install belts, inserters or chests can skip
  these by that key.

| scenario | what it covers |
|---|---|
| `logistics_smelting_chain` | south drill onto an east belt; right turn; belt end, then inserter, furnace, inserter, belt, inserter, chest; 120 decisions |
| `logistics_sideload_merge` | ten-belt east run loaded from both sides, with a four-belt north feed (copper, both lanes) sideloading its south side, until everything backs up; 100 decisions |
| `logistics_belt_pickup` | sixteen-belt line loaded on both lanes, with inserters on both sides picking from the moving belt into chests; 100 decisions |
| `logistics_inserter_fuel_exhaustion` | chest-to-chest inserters on the built-in quarter wood only, on one added wood, and on coal they move (self-refuel); then `give_to` one coal to the first, which restarts; 111 decisions |
| `logistics_belt_rotate_and_mine` | eight-belt line fed on both lanes; `rotate_at`, then `rotate_at_reverse`, on a belt mid-flow; `mine_at` that belt, then a compressed one near the end, whose eight plates go to the character; 44 decisions |

All the actions are expressible in `parameterized-v1`, and every trace has 0
unencodable actions. The targeted belts are also within the v2 entity table
(rows 0 and 8), so `parameterized-v2` can express them too.

Tick counts: 3,600, 3,000, 3,000, 3,330 and 1,320. Two runs of every tick
trace hash identically, and each agrees with its decision trace at every
decision boundary. A fresh worker (`--check`) reproduces all five decision
traces.

## Decisions (2026-09-24)

- **Simultaneous sideload arrivals, pickup position on a turn's arc, and the
  belt-line sleep case**: keep reverse-engineering, time-boxed to about two
  days together. The sim stays exact where measured; these remain pinned in
  factory-sim's `KNOWN_GAPS` until matched.
- **Character carried round a turn**: accepted as a gap. Straight belts are
  exact; turn carriage is not compared.
- **Young-belt wake delay**: accepted. Rigs it affects are compared from
  t=300, with a constant remaining-fuel offset allowed.
- **Hand y (drawn lift)**: not modelled. Where it is unknown the comparison
  takes the engine's value (`fsim.trace.relax_hand_y`).

## Decisions (2026-09-25)

- **Belt-line segments**: made exact in factory-sim from a measured table of
  the merge delay (option 1 of "What the simulator needs"), covering the
  scene area and the probes' rig areas; a belt outside the table stops the
  simulator (`fsim.BeltDelayMissing`) rather than guess. This supersedes the
  2026-09-24 entries on simultaneous sideloads, the belt-line sleep case and
  the young-belt wake delay: `KNOWN_GAPS` is empty and young-belt rigs are
  compared from t=0.
- **v3 profile** (user decision): hand-mining added to v3 (`mine_tile`); the
  v3 mask made legal exactly where the game accepts, by the measured reach
  rules; remembered rows accepted without belt or inserter detail;
  handcrafting off for Stage 1, measured and added with Stage 2's recipe
  dimension. The measurements and the four records are in "Hand-mining and
  reach (v3 decisions)" above.
- **A loaded state and the activation order**: a hidden-state load that
  would change the items of a belt-line segment that still holds items
  afterwards stops the simulator (`fsim.BeltOrderUnknown`) instead of
  placing that segment in the activation order itself; see "A loaded state
  and the activation order" above.
- **v3 masks: per operation (option C)**, **`finish`** and **taking fuel**
  (user decisions): one revision of the v3 layout, `MultiDiscrete[25, 97, 226,
  5, 19, 4]`, with the free main slots observed; see "v3 masks per operation,
  `finish` and taking fuel" above, which also records placement per item as a
  known observable limitation.
