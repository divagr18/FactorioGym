# Phase 2.0 capability spike — what works on a player-less character

The agent body is a bare `character` LuaEntity in `storage`, not a `LuaPlayer`:
Factorio 2.0 removed `game.create_player` and a headless zero-player server has
no player. Mining and crafting are two of PLAN.md 2.1's ten actions and both
depend on `LuaControl` behaving without anyone controlling it, so this was
measured before the action matrix was designed.

Command: `uv run factoriorl spike-control` → `docs/evidence/phase2-capability-spike.json`.
Engine: Factorio 2.0.60 (build 83512, win64).

Two sources are used together: the pinned build's machine-readable API dump at
`D:\Factorio\doc-html\runtime-api.json` (`api_version 6`) for what *exists*, and
a live worker for what actually *works*.

## Headline: no fallback drivers are needed

Both risky actions are native. `drivers.mining = "native"` and
`drivers.crafting = "native"` in the action profile — no mod-driven timer, no
documented approximation, nothing to disclose under PLAN.md section 2.

| Capability | Result |
|---|---|
| `update_selected_entity` → `selected` sticks | ✅ `selected_name: "iron-ore"`, `can_reach: true` at 2.55 tiles |
| Native mining via `mining_state` | ✅ progress 0 → 0.25 over 30 ticks, 1 ore delivered by 150 ticks |
| Native crafting via `begin_crafting` | ✅ `started: 2`, queue size 1, ingredients debited at start (10 → 6 plates), 2 gears delivered within 90 ticks |
| Assembled placement | ✅ `can_place_entity` true → `create_entity` → unit_number 12; second attempt `can_place_again: false` |
| `rotate{}` with no `by_player` | ✅ belt direction 0 → 4 |
| `cursor_stack` | ✅ accessible and non-nil |
| `register_on_object_destroyed` | ✅ registration number returned, fires despite `raise_destroy = false` |
| Resource `unit_number` | ❌ **absent** (a chest has one: 11) |

Mining timing checks out against the prototype: iron ore's `mining_time` is 1 s
and the character's mining speed is 0.5/s, so a full mine is 120 ticks — matching
the observed 0.25 progress after 30.

## Placement has no native path — settled, not open

`build_from_cursor` is declared on `LuaPlayer`, **not** on `LuaControl`. There is
no character build API. Placement must be assembled from
`can_place_entity{build_check_type = manual}` + `create_entity` + inventory
debit, which the spike confirms works and correctly reports collision.

`docs/ACTION_MATRIX.md` must therefore state what the assembled path excludes:
fast-replace, ghost revival, tile building, undo, and `on_built_entity`.

## Entity identity needs two forms

Resources are `ResourceEntityPrototype` with `flags = {"placeable-neutral"}`,
not `EntityWithOwnerPrototype`, and the spike confirms they carry **no
`unit_number`**. Since `unit_number` is documented as never reused for the
lifetime of a save, identity keyed on it makes PLAN 2.4's "rebuilt entities do
not inherit stale identity" true by construction — but only for entities that
have one.

So the handle allocator needs a `unit` form keyed on `unit_number` and a `tile`
form keyed on `(surface, tx, ty)` plus a generation counter for resources, trees
and rocks.

## Research: the tech tree is trigger-gated at the root

This took the longest to pin down and changes how research must be modelled.

`add_research` returned `false` for *every* technology tried, including
`electronics` and `steam-power` — which have no prerequisites, are enabled, and
are unresearched. `research_enabled` was true. Writing `force.research_queue`
succeeded but left the queue empty.

The cause is in the base game's data, not in our runtime: **Factorio 2.0's early
technologies are trigger-based**, completed by an action rather than selected.

```
steam-power              research_trigger = craft 50 iron-plate
electronics              research_trigger = craft 10 copper-plate
automation-science-pack  research_trigger = craft a lab   (prereqs: steam-power, electronics)
```

**7 technologies are trigger-based; 189 are queueable.** Every zero-prerequisite
technology is in the trigger set, so at a fresh start *nothing is selectable at
all* — the tree is entered by crafting, not by choosing.

The positive case then confirms the action works: after researching the trigger
chain through the evaluator, `add_research("logistics")` returns **true** and
`current_research` becomes `logistics`. `rocket-silo`, with unmet prerequisites,
returns false.

Consequences for the action matrix:

- `research` must distinguish three distinct reasons for refusal, not collapse
  them into one `tech_locked`: prerequisites unmet, technology already
  researched, and **technology is trigger-based and therefore not selectable**.
  The last one is not a failure of the agent's plan; it is a statement that the
  goal is reached by crafting.
- `current_research` is **read-only**; `research_queue` is the writable path.
- Early progression couples research to crafting, which matters for Phase 3 task
  design and Phase 9's research progression.

## Observation query cost is not a problem

Measured by timing 200 repetitions of each query inside one Lua call and
subtracting the cost of 200 no-ops:

| Query | Per call | Returned |
|---|---|---|
| `find_entities_filtered{radius = 32}` | **0.011 ms** | 29 entities |
| `find_entities_filtered{radius = 32, type = "resource"}` | ~0.000 ms | 15 resources |
| `find_tiles_filtered{65×65 box, collision_mask = "water_tile"}` | **0.018 ms** | 0 tiles |

Two conclusions for Phase 2.3. First, the three-query observation design is far
from being the bottleneck at reference-scene scale — 0.03 ms against a 16.7 ms
round trip. Second, `find_tiles_filtered` with a collision mask is effectively
free and returns *only obstacles*, so terrain must never be built by iterating
`get_tile` over 4225 tiles. Re-measure at Phase 7 factory scale, where entity
counts are two orders of magnitude larger.

## Reach values, from the prototype rather than a constant

`build_distance = 10`, `reach_distance = 10`, `resource_reach_distance = 2.7`,
`item_pickup_distance = 1`, `drop_item_distance = 10`.

The mod's `REACH_DISTANCE = 6` is wrong in both directions, and there is no
single reach: build, item and resource reach are three different numbers.
`can_reach_entity` applies the engine's own bounding-box-aware rule and should be
used for entity targets instead of centre-to-centre Euclidean distance.

This does **not** invalidate Phase 0 evidence: its transfer is at 5.37 tiles
(inside both 6 and 10) and `test_out_of_reach_transfer_fails` is at 18.06
(outside both), so both assertions hold at either value.
