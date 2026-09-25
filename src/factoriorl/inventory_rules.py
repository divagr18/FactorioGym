"""What the engine accepts where: the facts the `v3` per-operation masks rest on.

Read off the running engine by `tools/probe_inventory.py` and recorded in
`docs/evidence/inventory-prototypes.json.xz` (slot counts, stack sizes, fuel
values, what each item places, which entities turn, what each inventory
accepts, what mining gives) and `docs/evidence/inventory-fuel.json.xz`
(transfers through the mod's own sequence). `tests/unit/test_inventory_rules.py`
holds every constant here to that evidence, so none of them is typed in on
trust. factory-sim's `csrc/fsim_rl.c` carries the same table, held to the same
file.
"""

from __future__ import annotations

#: The character's main inventory.
MAIN_SLOTS = 80

#: Stack sizes of the `v3` items (`encoders.ITEMS_V3`).
STACK_SIZE: dict[str, int] = {
    "iron-ore": 50,
    "copper-ore": 50,
    "coal": 50,
    "stone": 50,
    "iron-plate": 100,
    "copper-plate": 100,
    "stone-furnace": 50,
    "iron-gear-wheel": 100,
    "transport-belt": 100,
    "wood": 100,
    "small-electric-pole": 50,
    "wooden-chest": 50,
    "burner-mining-drill": 50,
    "burner-inserter": 50,
    "assembling-machine-1": 50,
    "boiler": 50,
    "steam-engine": 10,
    "offshore-pump": 20,
}

#: Items with a fuel value (both chemical).
FUEL_ITEMS = frozenset({"coal", "wood"})

#: Item -> the entity it places.
PLACES: dict[str, str] = {
    "stone-furnace": "stone-furnace",
    "transport-belt": "transport-belt",
    "small-electric-pole": "small-electric-pole",
    "wooden-chest": "wooden-chest",
    "burner-mining-drill": "burner-mining-drill",
    "burner-inserter": "burner-inserter",
    "assembling-machine-1": "assembling-machine-1",
    "boiler": "boiler",
    "steam-engine": "steam-engine",
    "offshore-pump": "offshore-pump",
}

#: Entities `LuaEntity.rotate` turns. A steam engine and an offshore pump take a
#: direction but refuse to turn once built.
ROTATABLE = frozenset({"transport-belt", "burner-inserter", "burner-mining-drill", "boiler"})

#: The ores a stone furnace's source slot takes (steel is not researched, so
#: not iron plate).
SMELTABLE = frozenset({"iron-ore", "copper-ore", "stone"})

#: How many of an ore one insert puts into an empty stone furnace's source
#: slot: 54, over the stack of 50 (`inventory-prototypes`, `capacity`).
SOURCE_CAPACITY = 54

#: Per entity, the inventories the mod's transfer tries, in its order
#: (`actions.lua`, `inventory_of`), as (the observation record's field, slots,
#: the items it accepts or None for any, and per item how many one slot takes
#: where that is not the stack size). A furnace's result slot takes any item by
#: script, which is where a give of coal lands once the fuel slot is full
#: (`inventory-fuel`, `give_partial`).
INVENTORIES: dict[str, tuple[tuple[str, int, frozenset | None, dict], ...]] = {
    "wooden-chest": (("contents", 16, None, {}),),
    "stone-furnace": (
        ("fuel", 1, FUEL_ITEMS, {}),
        ("contents", 1, SMELTABLE, dict.fromkeys(SMELTABLE, SOURCE_CAPACITY)),
        ("output", 1, None, {}),
    ),
    "burner-mining-drill": (("fuel", 1, FUEL_ITEMS, {}),),
    "burner-inserter": (("fuel", 1, FUEL_ITEMS, {}),),
    "boiler": (("fuel", 1, FUEL_ITEMS, {}),),
}

#: Minable by the character, yielding nothing: `actions.lua`'s `mine` refuses
#: it ("yields nothing"). A pile over a resource tile shares the tile's handle,
#: which resolves to the resource, and is mined through it.
YIELDS_NOTHING = frozenset({"item-on-ground"})


def slots_of(item: str, count: int) -> int:
    """The fewest slots `count` of `item` can fill: whole stacks. An item
    outside `STACK_SIZE` counts as one slot, the fewest it can take."""
    if count <= 0:
        return 0
    size = STACK_SIZE.get(item)
    return 1 if size is None else -(-count // size)


def room(held: dict | None, slots: int, item: str, per_slot: int | None = None) -> int:
    """How many of `item` an inventory of `slots` holding `held` (item totals)
    can take at most; `per_slot`, how many of it one slot takes, where that is
    not its stack size.

    Totals do not say how the stacks lie, so everything else is taken to lie in
    whole stacks, the fewest slots it can fill: this is an upper bound, and 0
    here means the engine has no room at all. (Stacks of `item` itself do not
    matter: however its own count is split, the room it leaves is the same.)
    """
    size = per_slot or STACK_SIZE.get(item)
    if size is None:
        return 0
    held = held or {}
    others = sum(slots_of(name, int(count)) for name, count in held.items() if name != item)
    return max(0, size * (slots - others) - int(held.get(item, 0)))


def accepts(record: dict, item: str) -> bool:
    """Whether a give of `item` to this entity moves at least one.

    The first inventory the mod tries that can take one (`can_insert`) is where
    it goes; a give larger than the room there moves what fits and is reported
    `no_space` (`inventory-fuel`, `give_partial`), and moves nothing only when
    no inventory has room.
    """
    for field, slots, taken, per_slot in INVENTORIES.get(record.get("name"), ()):
        if taken is not None and item not in taken:
            continue
        if room(record.get(field), slots, item, per_slot.get(item)) >= 1:
            return True
    return False


def holds(record: dict, item: str) -> int:
    """How many of `item` the inventories a transfer reads hold."""
    return sum(
        int((record.get(field) or {}).get(item, 0))
        for field, *_rest in INVENTORIES.get(record.get("name"), ())
    )


def fuel_item(record: dict) -> str | None:
    """The item in an entity's fuel slot, or None. A burner's fuel inventory is
    one slot (`inventory-prototypes`), so it holds one item at most."""
    if not any(field == "fuel" for field, *_rest in INVENTORIES.get(record.get("name"), ())):
        return None
    for item, count in (record.get("fuel") or {}).items():
        if count:
            return item
    return None
