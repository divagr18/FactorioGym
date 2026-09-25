"""`factoriorl.inventory_rules` against the engine evidence it was read from.

`docs/evidence/inventory-prototypes.json.xz` and `inventory-fuel.json.xz` were
written by `tools/probe_inventory.py` on the running engine; every constant the
`v3` per-operation masks use must be what they recorded.
"""

from __future__ import annotations

import json
import lzma
from pathlib import Path

import pytest

from factoriorl import inventory_rules as rules
from factoriorl.encoders import ITEMS_V3

EVIDENCE = Path(__file__).resolve().parents[2] / "docs" / "evidence"


def _load(name: str) -> dict:
    path = EVIDENCE / f"inventory-{name}.json.xz"
    if not path.is_file():
        pytest.skip(f"no {path.name}")
    return json.loads(lzma.decompress(path.read_bytes()))


def test_slots_stacks_fuel_and_places_are_the_engines():
    data = _load("prototypes")
    assert rules.MAIN_SLOTS == data["character"]["main_slots"]
    items = data["items"]
    assert set(items) == set(ITEMS_V3)
    assert rules.STACK_SIZE == {name: rec["stack_size"] for name, rec in items.items()}
    assert rules.FUEL_ITEMS == {name for name, rec in items.items() if rec["fuel_value"] > 0}
    assert rules.PLACES == {
        name: rec["place_result"] for name, rec in items.items() if "place_result" in rec
    }


def test_turning_mining_and_inventories_are_the_engines():
    entities = _load("prototypes")["entities"]
    assert rules.ROTATABLE == {name for name, rec in entities.items() if rec["rotated"]}
    assert rules.YIELDS_NOTHING == {
        name for name, rec in entities.items() if rec["minable"] and "mines_to" not in rec
    }
    everything = set(ITEMS_V3)
    for name, rec in entities.items():
        measured = [
            (inv["size"], set(inv["accepts"]), inv["capacity"])
            for _index, inv in sorted(rec["inventories"].items(), key=lambda kv: int(kv[0]))
        ]
        declared = [
            (
                slots,
                everything if taken is None else set(taken),
                {
                    item: slots * (per_slot.get(item) or rules.STACK_SIZE[item])
                    for item in (everything if taken is None else taken)
                },
            )
            for _field, slots, taken, per_slot in rules.INVENTORIES.get(name, ())
        ]
        assert measured == declared, name
    assert rules.SMELTABLE == set(entities["stone-furnace"]["inventories"]["2"]["accepts"])


def _transfers(data: dict, rig: str) -> list[dict]:
    return [note[3] for note in data["transfers"] if note[0] == rig]


def test_fuel_comes_out_of_the_fuel_slot():
    data = _load("fuel")
    for rig in ("drill", "furnace", "inserter", "boiler"):
        first, second = _transfers(data, rig)[:2]
        # Inventory 1 is the fuel slot of every burner (`defines.inventory.fuel`).
        assert first["from_index"] == 1 and first["status"] == "completed", rig
        # More than is there: what is there, all of it.
        assert second["wanted"] == second["available"] == second["inserted"], rig
    # One coal burns at once; the burning one is not in the slot.
    (only,) = _transfers(data, "furnace_one_coal")
    assert only["code"] == "no_items"


def test_part_of_the_room_moves_what_fits_and_is_refused():
    data = _load("fuel")
    take = _transfers(data, "take_partial")
    assert (take[0]["inserted"], take[0]["put_back"], take[0]["code"]) == (3, 2, "no_space")
    assert take[1]["can_insert"] is False and take[1]["code"] == "no_space"
    give = _transfers(data, "give_partial")
    assert (give[0]["inserted"], give[0]["put_back"]) == (2, 3)
    # The fuel slot full, the next inventory that takes coal is the result slot.
    assert give[1]["to_index"] == 3 and give[1]["status"] == "completed"


def test_room_is_an_upper_bound_from_totals():
    # 79 slots of wood and 47 coal: room for 3 coal, none for plates.
    held = {"wood": 7900, "coal": 47}
    assert rules.room(held, rules.MAIN_SLOTS, "coal") == 3
    assert rules.room(held, rules.MAIN_SLOTS, "iron-plate") == 0
    # A fuel slot.
    assert rules.room({"coal": 48}, 1, "coal") == 2
    assert rules.room({"coal": 48}, 1, "wood") == 0
    assert rules.room({}, 1, "wood") == 100
    # A furnace's source slot takes 54 ore.
    assert rules.accepts(
        {"name": "stone-furnace", "contents": {"iron-ore": 50}, "output": {"iron-plate": 1}},
        "iron-ore",
    )
    assert not rules.accepts(
        {"name": "stone-furnace", "contents": {"iron-ore": 54}, "output": {"iron-plate": 1}},
        "iron-ore",
    )
    furnace = {"name": "stone-furnace", "fuel": {"coal": 50}}
    assert rules.accepts(furnace, "coal")  # into the result slot
    assert rules.fuel_item(furnace) == "coal"
    assert rules.fuel_item({"name": "wooden-chest", "contents": {"coal": 5}}) is None
