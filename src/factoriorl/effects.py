"""What an action was supposed to do, and whether it did it.

the literature synthesis §5 (Ghallab/Nau/Traverso, PDF
94-98): executing a command and observing its intended effect are different
things, and a controller must verify the consequence appropriate to its goal.
Its examples are exactly ours -- "a successful transfer means items moved; it
does not establish resumed production. A completed belt placement means an
entity was placed; it does not establish orientation, connected flow, or
downstream output."

So an action record carries the *commanded* action, the *expected* observable
effect with a deadline, and separately the *observed* outcome. Predicted and
observed never share a field: "issued fuel transfer" must not become "machine
is fuelled" without a successful result.

Nothing here diagnoses anything or chooses an action. It is a record shape plus
a deadline check, so a controller built on it is declared assistance rather
than hidden inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class EffectState(StrEnum):
    """Where an expectation stands. `UNKNOWN` is a first-class answer."""

    PENDING = "pending"
    CONFIRMED = "confirmed"
    REFUSED = "refused"
    TIMED_OUT = "timed_out"
    UNKNOWN = "unknown"


#: How a commanded action's effect is checked. Named so a trace says which
#: consequence was verified, not merely that something was verified.
class EffectKind(StrEnum):
    ENTITY_PRESENT = "entity_present"
    ENTITY_FACING = "entity_facing"
    ITEMS_MOVED = "items_moved"
    INVENTORY_FELL = "inventory_fell"
    CHARACTER_MOVED = "character_moved"
    OUTPUT_ROSE = "output_rose"
    #: Commanded with no verifiable local consequence declared.
    NONE = "none"


@dataclass(frozen=True)
class Expectation:
    """The effect an action was issued for, and by when.

    `deadline_ticks` is a declared bound, not a guess dressed as a fact. A
    transfer lands immediately; a belt repair reaching a sink takes 340-600
    ticks for a 6-14 tile line, so a monitor that gave up in one step would
    call every correct repair a failure.
    """

    kind: EffectKind
    deadline_ticks: int = 0
    detail: dict = field(default_factory=dict)


@dataclass
class ActionRecord:
    """One commanded action, its expectation, and what was observed.

    Predicted and observed are separate fields on purpose.
    """

    action_key: str
    commanded: dict
    expectation: Expectation
    issued_tick: int
    status: str | None = None
    error: str | None = None
    observed: EffectState = EffectState.PENDING
    observed_tick: int | None = None

    @property
    def settled(self) -> bool:
        return self.observed is not EffectState.PENDING

    def refuse(self, error: str | None, tick: int) -> None:
        """The engine declined it; there is no effect to wait for."""
        self.status = "rejected"
        self.error = error
        self.observed = EffectState.REFUSED
        self.observed_tick = tick

    def confirm(self, tick: int) -> None:
        self.observed = EffectState.CONFIRMED
        self.observed_tick = tick

    def expire(self, tick: int) -> None:
        """The deadline passed without the effect appearing.

        Distinct from `REFUSED`: the engine accepted the command and the world
        did not follow. A controller that cannot tell those apart will retry
        the wrong one.
        """
        self.observed = EffectState.TIMED_OUT
        self.observed_tick = tick

    def to_dict(self) -> dict:
        return {
            "action_key": self.action_key,
            "commanded": self.commanded,
            "expected_effect": self.expectation.kind.value,
            "deadline_ticks": self.expectation.deadline_ticks,
            "expected_detail": self.expectation.detail,
            "issued_tick": self.issued_tick,
            "status": self.status,
            "error": self.error,
            "observed": self.observed.value,
            "observed_tick": self.observed_tick,
        }


#: The consequence each verb is issued for. Deliberately conservative: a verb
#: whose local effect is not observable declares `NONE` rather than a claim
#: nothing can check.
EXPECTATIONS: dict[str, Expectation] = {
    "place": Expectation(EffectKind.ENTITY_PRESENT, deadline_ticks=0),
    "rotate": Expectation(EffectKind.ENTITY_FACING, deadline_ticks=0),
    "transfer": Expectation(EffectKind.ITEMS_MOVED, deadline_ticks=0),
    "mine": Expectation(EffectKind.INVENTORY_FELL, deadline_ticks=120),
    "move": Expectation(EffectKind.CHARACTER_MOVED, deadline_ticks=60),
    # Selecting a recipe has no local consequence the sensor reports for a
    # furnace, which auto-selects; claiming one would be a fiction.
    "set_recipe": Expectation(EffectKind.NONE),
    "craft": Expectation(EffectKind.NONE, deadline_ticks=300),
    "cancel": Expectation(EffectKind.NONE),
    "wait": Expectation(EffectKind.NONE),
}


def expectation_for(action: str) -> Expectation:
    return EXPECTATIONS.get(action, Expectation(EffectKind.NONE))


def observe(record: ActionRecord, observation: dict, tick: int) -> ActionRecord:
    """Advance one record against the world, from the observation alone.

    Only the effect the record *declared* is checked. A confirmed placement
    says an entity is on that tile and nothing about flow reaching a sink,
    which is the distinction §5 exists to keep.
    """
    if record.settled:
        return record

    kind = record.expectation.kind
    detail = record.expectation.detail
    if kind is EffectKind.NONE:
        record.observed = EffectState.UNKNOWN
        record.observed_tick = tick
        return record

    if kind is EffectKind.ENTITY_PRESENT:
        wanted = detail.get("position")
        if wanted is not None and _entity_at(observation, wanted):
            record.confirm(tick)
    elif kind is EffectKind.ENTITY_FACING:
        wanted = detail.get("position")
        facing = detail.get("direction")
        entity = _entity_at(observation, wanted) if wanted is not None else None
        if entity is not None and facing is not None and entity.get("d") == facing:
            record.confirm(tick)
    elif kind is EffectKind.INVENTORY_FELL or kind is EffectKind.ITEMS_MOVED:
        item = detail.get("item")
        before = detail.get("before")
        if item is not None and before is not None:
            now = (observation.get("inventory") or {}).get(item, 0)
            if now != before:
                record.confirm(tick)
    elif kind is EffectKind.CHARACTER_MOVED:
        before = detail.get("position")
        now = (observation.get("character") or {}).get("position")
        if before is not None and now is not None and list(now) != list(before):
            record.confirm(tick)

    if not record.settled and tick - record.issued_tick >= record.expectation.deadline_ticks:
        record.expire(tick)
    return record


def _entity_at(observation: dict, position) -> dict | None:
    import math

    tile = (math.floor(position[0]), math.floor(position[1]))
    for record in observation.get("entities") or []:
        point = record.get("p")
        if point and (math.floor(point[0]), math.floor(point[1])) == tile:
            return record
    return None
