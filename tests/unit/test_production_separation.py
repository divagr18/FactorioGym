"""A4.2: machine output, handcrafting and mining are three different numbers.

The engine will not separate them. `get_item_production_statistics` counts
everything entering the force's inventory space, so a plate a furnace made and a
plate a character hand-crafted are one number, and mined ore is in there too.

So `world.truth` derives the machine column by subtraction and publishes all
three parts. These tests cover the Python half -- that an open world measures
the derived column and discovers its own item set, and that a benchmark task's
behaviour is byte-identical to before.

The engine half is `tools/probe_production.py` and
`docs/evidence/a4-production.json`: 8 ore mined by hand and 0 in the machine
column, 1 gear hand-crafted and 0 in the machine column, 5 plates smelted and 5
in the machine column.
"""

from __future__ import annotations

from factoriorl import worlds
from factoriorl.production import ProductionMetrics
from factoriorl.tasks import get


def _record(metrics: ProductionMetrics, ticks: list[int], truth_key: str, series: list[dict]):
    for tick, counts in zip(ticks, series, strict=True):
        metrics.record({"tick": tick}, {truth_key: counts})
    return metrics.report()


def test_a_task_still_measures_everything_the_force_produced():
    """Unchanged, and it has to be: every existing result was measured this way."""
    metrics = ProductionMetrics.for_task(get("plate_line").spec)
    assert metrics.source == "produced"
    assert not metrics.discover
    assert metrics.items  # the task declares its items in its predicates


def test_an_open_world_measures_the_machine_column():
    """`for_task` would give it `items=()` and a report of `{}` in every field,
    because `open_world.spec_for` declares no success predicates at all."""
    assert ProductionMetrics.for_task(_open_spec()).items == ()

    metrics = ProductionMetrics.for_world(worlds.get("open_factory"))
    assert metrics.source == "machine_produced"
    assert metrics.discover


def _open_spec():
    from factoriorl.open_world import spec_for

    return spec_for(worlds.get("open_factory"))


def test_an_open_world_discovers_the_items_it_actually_produces():
    """A world has no predicates to read an item list off, and produces
    whatever the agent decides to. The set cannot be known in advance."""
    metrics = ProductionMetrics.for_world(worlds.get("open_factory"))
    report = _record(
        metrics,
        [0, 3600, 7200],
        "machine_produced",
        [{}, {"iron-plate": 5.0}, {"iron-plate": 11.0, "copper-plate": 2.0}],
    )

    assert set(metrics.items) == {"iron-plate", "copper-plate"}
    assert report["cumulative_produced"]["iron-plate"] == 11.0
    assert report["counts"] == "machine_produced"


def test_time_to_first_output_is_not_time_to_first_sustained_output():
    """A4.2 asks for time to first machine production. One plate answers that
    and does not answer whether output was sustained across a window."""
    metrics = ProductionMetrics.for_world(worlds.get("open_factory"))
    report = _record(
        metrics,
        [0, 600, 1200],
        "machine_produced",
        [{}, {"iron-plate": 1.0}, {"iron-plate": 1.0}],
    )

    assert report["ticks_to_first_output"]["iron-plate"] == 600
    # One plate and then nothing is not sustained output, and the window has
    # not even elapsed yet.
    assert report["ticks_to_first_sustained_output"]["iron-plate"] is None


def test_an_item_discovered_late_does_not_invent_earlier_production():
    """Every sample before an item appeared is zero for it, which is true."""
    metrics = ProductionMetrics.for_world(worlds.get("open_factory"))
    _record(
        metrics,
        [0, 3600],
        "machine_produced",
        [{"iron-plate": 4.0}, {"iron-plate": 8.0, "copper-plate": 3.0}],
    )

    first_sample = metrics._samples[0][1]
    assert first_sample.get("copper-plate", 0.0) == 0.0


def test_the_report_says_which_column_it_counted():
    """`produced` includes handcrafting and mining; `machine_produced` does not.
    A number that does not say which it is cannot be compared to another one."""
    task = ProductionMetrics.for_task(get("plate_line").spec)
    world = ProductionMetrics.for_world(worlds.get("open_factory"))

    assert task.report()["counts"] == "produced"
    assert world.report()["counts"] == "machine_produced"
