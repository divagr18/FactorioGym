"""Python mirror of ``mod/factoriorl/matrix.lua`` (DESIGN.md 2.1).

The Lua table is the authority the dispatcher reads; this is the Python view of
the same contract. Two tests keep them from drifting: an engine-free contract
test compares this against the names and flags declared in the Lua source, and
an engine test compares it against the live ``describe`` response, so a worker
can never be running a matrix the client did not validate against.

A hand-maintained matrix would be stale within a week, so ``docs/ACTION_MATRIX.md``
is generated from this rather than written.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from factoriorl.protocol import ActionType, ErrorCode


class Reach(StrEnum):
    """Which of the character's three reach distances an action uses.

    There is no single reach: the prototype declares build 10, reach 10 and
    resource 2.7. ``NONE`` marks an action that is not a physical interaction.
    """

    NONE = "none"
    ENTITY = "entity"
    BUILD = "build"
    RESOURCE = "resource"


@dataclass(frozen=True)
class ActionSpec:
    name: ActionType
    ongoing: bool
    supersedes: bool
    cancellable: bool
    reach: Reach
    mutates_inventory: bool
    required: tuple[str, ...]
    optional: tuple[str, ...] = ()
    failure_codes: tuple[ErrorCode, ...] = field(default_factory=tuple)


A = ActionType
E = ErrorCode

#: Declaration order matches matrix.lua's ORDER, which fixes doc and test order.
ACTIONS: dict[ActionType, ActionSpec] = {
    A.MOVE: ActionSpec(
        name=A.MOVE,
        ongoing=True,
        # A new move supersedes a running one, so a policy emitting a direction
        # every step never has to spend an action cancelling first.
        supersedes=True,
        cancellable=True,
        reach=Reach.NONE,
        mutates_inventory=False,
        required=("direction",),
        optional=("ticks",),
        failure_codes=(E.MISSING_FIELD, E.BAD_TYPE, E.PRECONDITION),
    ),
    A.MINE: ActionSpec(
        name=A.MINE,
        ongoing=True,
        supersedes=False,
        cancellable=True,
        reach=Reach.RESOURCE,
        mutates_inventory=True,
        required=("handle",),
        optional=("count",),
        failure_codes=(
            E.UNKNOWN_HANDLE,
            E.TARGET_MISSING,
            E.OUT_OF_REACH,
            E.NOT_MINEABLE,
            E.NO_SPACE,
            E.BUSY,
        ),
    ),
    A.CRAFT: ActionSpec(
        name=A.CRAFT,
        ongoing=True,
        supersedes=False,
        cancellable=True,
        reach=Reach.NONE,
        mutates_inventory=True,
        required=("recipe",),
        optional=("count",),
        failure_codes=(E.RECIPE_UNAVAILABLE, E.TECH_LOCKED, E.NO_ITEMS, E.BAD_TYPE),
    ),
    A.PLACE: ActionSpec(
        name=A.PLACE,
        ongoing=False,
        supersedes=False,
        cancellable=False,
        reach=Reach.BUILD,
        mutates_inventory=True,
        required=("item", "position"),
        optional=("direction",),
        failure_codes=(
            E.NO_ITEMS,
            E.COLLISION,
            E.OUT_OF_REACH,
            E.TECH_LOCKED,
            E.INVALID_TARGET,
        ),
    ),
    A.ROTATE: ActionSpec(
        name=A.ROTATE,
        ongoing=False,
        supersedes=False,
        cancellable=False,
        reach=Reach.ENTITY,
        mutates_inventory=False,
        required=("handle",),
        optional=("reverse",),
        failure_codes=(
            E.UNKNOWN_HANDLE,
            E.TARGET_MISSING,
            E.OUT_OF_REACH,
            E.INVALID_TARGET,
        ),
    ),
    A.TRANSFER: ActionSpec(
        name=A.TRANSFER,
        ongoing=False,
        supersedes=False,
        cancellable=False,
        reach=Reach.ENTITY,
        mutates_inventory=True,
        required=("from", "to", "item", "count"),
        failure_codes=(
            E.NO_ITEMS,
            E.NO_SPACE,
            E.OUT_OF_REACH,
            E.UNKNOWN_HANDLE,
            E.TARGET_MISSING,
            E.PRECONDITION,
        ),
    ),
    A.SET_RECIPE: ActionSpec(
        name=A.SET_RECIPE,
        ongoing=False,
        supersedes=False,
        cancellable=False,
        reach=Reach.ENTITY,
        mutates_inventory=True,
        required=("handle",),
        optional=("recipe",),
        failure_codes=(
            E.RECIPE_UNAVAILABLE,
            E.TECH_LOCKED,
            E.INVALID_TARGET,
            E.OUT_OF_REACH,
            E.UNKNOWN_HANDLE,
            E.TARGET_MISSING,
        ),
    ),
    A.RESEARCH: ActionSpec(
        name=A.RESEARCH,
        ongoing=False,
        supersedes=False,
        cancellable=False,
        reach=Reach.NONE,
        mutates_inventory=False,
        required=(),
        optional=("technology", "cancel"),
        # TECH_NOT_SELECTABLE is separate from TECH_LOCKED because 2.0 gates the
        # root of the tech tree behind crafting triggers, not prerequisites.
        failure_codes=(E.TECH_LOCKED, E.TECH_NOT_SELECTABLE, E.INVALID_TARGET),
    ),
    A.WAIT: ActionSpec(
        name=A.WAIT,
        ongoing=False,
        supersedes=False,
        cancellable=False,
        reach=Reach.NONE,
        mutates_inventory=False,
        required=(),
    ),
    A.CANCEL: ActionSpec(
        name=A.CANCEL,
        ongoing=False,
        supersedes=False,
        cancellable=False,
        reach=Reach.NONE,
        mutates_inventory=False,
        required=("target_request_id",),
        failure_codes=(E.UNKNOWN_REQUEST_ID, E.NOT_CANCELLABLE),
    ),
}

ORDER: tuple[ActionType, ...] = tuple(ACTIONS)

#: Actions that span decision intervals and therefore report `running` first.
ONGOING: frozenset[ActionType] = frozenset(a for a, s in ACTIONS.items() if s.ongoing)


def to_markdown() -> str:
    """Render the matrix as the committed documentation table."""
    lines = [
        "# Action matrix",
        "",
        "Generated from `src/factoriorl/action_matrix.py`, which mirrors",
        "`mod/factoriorl/matrix.lua`. Do not edit by hand — run",
        "`uv run factoriorl action-matrix`.",
        "",
        "Reach kinds come from the character prototype: build 10, entity 10,",
        "resource 2.7. `none` means the action is not a physical interaction.",
        "",
        "| Action | Ongoing | Cancellable | Reach | Inventory | Required | Optional |",
        "|---|---|---|---|---|---|---|",
    ]
    for action in ORDER:
        spec = ACTIONS[action]
        lines.append(
            f"| `{spec.name.value}` | {'yes' if spec.ongoing else 'no'} "
            f"| {'yes' if spec.cancellable else 'no'} | {spec.reach.value} "
            f"| {'yes' if spec.mutates_inventory else 'no'} "
            f"| {', '.join(f'`{f}`' for f in spec.required) or '--'} "
            f"| {', '.join(f'`{f}`' for f in spec.optional) or '--'} |"
        )
    lines += [
        "",
        "## Failure codes",
        "",
        "| Action | Codes |",
        "|---|---|",
    ]
    for action in ORDER:
        spec = ACTIONS[action]
        codes = ", ".join(f"`{c.value}`" for c in spec.failure_codes) or "--"
        lines.append(f"| `{spec.name.value}` | {codes} |")
    lines += [
        "",
        "## What the assembled placement path excludes",
        "",
        "`build_from_cursor` is declared on `LuaPlayer`, not `LuaControl`, and the",
        "agent body is a bare character entity, so there is no native character",
        "build path. `place` is assembled from `can_place_entity` +",
        "`create_entity` + an inventory debit, which excludes fast-replace, ghost",
        "revival, tile building, undo, and the `on_built_entity` event.",
        "",
        "## Research is not always selectable",
        "",
        "Factorio 2.0 gates the root of the technology tree behind triggers rather",
        "than selection: 7 of 196 technologies complete by crafting or mining and",
        "cannot be queued at all, and every zero-prerequisite technology is one of",
        "them. `research` reports `tech_not_selectable` for those, separately from",
        "`tech_locked` for unmet prerequisites — telling an agent to wait for",
        "prerequisites that will never arrive would be wrong.",
        "",
    ]
    return "\n".join(lines)
