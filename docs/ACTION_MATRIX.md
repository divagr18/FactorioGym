# Action matrix

Generated from `src/factoriorl/action_matrix.py`, which mirrors
`mod/factoriorl/matrix.lua`. Do not edit by hand — run
`uv run factoriorl action-matrix`.

Reach kinds come from the character prototype: build 10, entity 10,
resource 2.7. `none` means the action is not a physical interaction.

| Action | Ongoing | Cancellable | Reach | Inventory | Required | Optional |
|---|---|---|---|---|---|---|
| `move` | yes | yes | none | no | `direction` | `ticks` |
| `mine` | yes | yes | resource | yes | `handle` | `count` |
| `craft` | yes | yes | none | yes | `recipe` | `count` |
| `place` | no | no | build | yes | `item`, `position` | `direction` |
| `rotate` | no | no | entity | no | `handle` | `reverse` |
| `transfer` | no | no | entity | yes | `from`, `to`, `item`, `count` | -- |
| `set_recipe` | no | no | entity | yes | `handle` | `recipe` |
| `research` | no | no | none | no | -- | `technology`, `cancel` |
| `wait` | no | no | none | no | -- | -- |
| `cancel` | no | no | none | no | `target_request_id` | -- |

## Failure codes

| Action | Codes |
|---|---|
| `move` | `missing_field`, `bad_type`, `precondition` |
| `mine` | `unknown_handle`, `target_missing`, `out_of_reach`, `not_mineable`, `no_space`, `busy` |
| `craft` | `recipe_unavailable`, `tech_locked`, `no_items`, `bad_type` |
| `place` | `no_items`, `collision`, `out_of_reach`, `tech_locked`, `invalid_target` |
| `rotate` | `unknown_handle`, `target_missing`, `out_of_reach`, `invalid_target` |
| `transfer` | `no_items`, `no_space`, `out_of_reach`, `unknown_handle`, `target_missing`, `precondition` |
| `set_recipe` | `recipe_unavailable`, `tech_locked`, `invalid_target`, `out_of_reach`, `unknown_handle`, `target_missing` |
| `research` | `tech_locked`, `tech_not_selectable`, `invalid_target` |
| `wait` | -- |
| `cancel` | `unknown_request_id`, `not_cancellable` |

## What the assembled placement path excludes

`build_from_cursor` is declared on `LuaPlayer`, not `LuaControl`, and the
agent body is a bare character entity, so there is no native character
build path. `place` is assembled from `can_place_entity` +
`create_entity` + an inventory debit, which excludes fast-replace, ghost
revival, tile building, undo, and the `on_built_entity` event.

## Research is not always selectable

Factorio 2.0 gates the root of the technology tree behind triggers rather
than selection: 7 of 196 technologies complete by crafting or mining and
cannot be queued at all, and every zero-prerequisite technology is one of
them. `research` reports `tech_not_selectable` for those, separately from
`tech_locked` for unmet prerequisites — telling an agent to wait for
prerequisites that will never arrive would be wrong.
