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
(160.5, 131.69921875). Only north-facing inserters were built.

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
38.46; the observed 38 is consistent with the pickup tick itself counting as
the first rotation step (39 steps of 0.013 reach 0.507).

### Energy

Burner draw per tick, from `remaining_burning_fuel` differences (`c2c`,
`fuel`, `wood`):

- t=1: 2,560 (fills the buffer); t=2..9 (first approach): 1,750 each.
- Every half-swing, both directions: **5 ticks at 2,400 then 33 ticks at 650
  = 33,450 J**. A chest-to-chest cycle is **66,900 J per item** (t=85..161,
  summed).
- Waiting (`waiting_for_source_items`, `waiting_for_space_in_destination`):
  **0 J/tick**, buffer stays 2,560. There is no drain.
- Chasing belt items draws other amounts (745.46, 1,360.02, 1,935.71, ... in
  `flow`, `tick_ins`), so the table above only holds for a swing between two
  fixed points.

Check against exhaustion (`wood`: the built-in quarter wood, then one wood):
the wood is loaded at t=551 and runs out at t=2825 (remaining 0, buffer
1,099.96); at t=2826 the buffer is 0 and status `no_fuel`, the hand stopped
mid-swing still holding a plate. 2 MJ / 66,900 J = 29.9 cycles = 2,272 ticks;
observed 2,274. 37 plates delivered by then (39 for the coal inserter by
t=3000, still running on its coal with 1.85 MJ left).

### Hand position

`held_stack_position` does not follow a simple polar path: its distance from
the inserter goes above 1.5 tiles mid-swing and the item swings through the
west side (`c2c`, t=10..46). It looks like a drawing position. Model the timing
above, not the hand.

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
  the far lane with 220 ticks (3 items) from t=481, hand positions identical --
  but reproducing it means reproducing the chase kinematics tick by tick.
  **Recommendation: do not model it.** Keep tasks to pickups from chests,
  machines and stopped items at belt ends, where the 38/38 rule holds.

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

## Open questions and rigs to settle them

1. **Sideload with traffic.** The single-item rule is exact; entries of 172 and
   187 under contention are not. Rig: one feed lane at a time, pairs of items
   at controlled gaps (via `insert_at`) arriving on an empty and on a moving
   main lane.
2. **Where a drop lands when the drop point is taken on a moving line.** The
   unblocked drill put its item at 192 (read 184) behind an item at 128,
   while on a stopped line 192 was refused. Hypothesis: the item goes at the
   first free position at or behind the drop point, accepted only if it is
   less than 64 behind the drop point, with the insertion seeing the line
   before this tick's move. Rig: an inserter holding an item over a slowly
   filled moving lane, items placed with `insert_at` at 120..200.
3. **Lane choice on pickup** when two items are equally close: lane 2 both
   times here. Rig: inserters on both sides of one belt.
4. **Left turns, other inserter facings**: only a right turn and north-facing
   inserters were built.
5. **The energy mechanism** behind 5x2,400 + 33x650 is empirical. It is enough
   for fixed chest/machine/belt-end swings; chase swings draw different
   amounts.
6. **Coal exhaustion** was not reached in 3,000 ticks (4 MJ is about 4,550
   ticks at 66,900 J per item); the wood run covers the stop behaviour.
7. **Self-refuel timing** (28 vs 37 ticks) needs a rig that starts the empty
   slot at different points of the swing.

## Recorded state

What the parity recorder (`tools/record_parity_trace.py`) writes for belts,
burner inserters and wooden chests, so a simulator can export the same thing.
It comes from `world.hidden_state` in `mod/factoriorl/world.lua` and reaches
the trace through `Normaliser.hidden`. Records for every other entity type keep
exactly the keys they had before, so traces recorded before these fields
existed still match.

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

`held_stack_position` is the drawn hand, not a polar path (see "Hand
position" above). A simulator that models the timing and not the hand cannot
reproduce it exactly. Relaxing it on the simulator side is a decision for that
side; this record keeps the engine's value.

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
