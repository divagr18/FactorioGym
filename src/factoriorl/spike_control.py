"""Phase 2.0 capability spike: what works on a player-less character?

The agent body is a bare ``character`` LuaEntity in ``storage``, not a
``LuaPlayer`` -- Factorio 2.0 removed ``game.create_player`` and a headless
zero-player server has no player. The pinned build's API dump
(``D:\\Factorio\\doc-html\\runtime-api.json``) confirms ``LuaEntity``'s parent is
``LuaControl``, which *declares* ``mining_state``, ``begin_crafting``,
``crafting_queue``, ``cursor_stack`` and the reach properties.

Declaration is not the same as working with nobody controlling it. Mining and
crafting are two of the ten actions in PLAN.md 2.1, and both depend on this, so
it gets measured before the action matrix is designed rather than after.

This deliberately uses the evaluator channel (``bridge.run``), not the typed
protocol: the point is to probe raw engine behavior, including things the typed
protocol will never expose.

Run: ``uv run factoriorl spike-control``
"""

from __future__ import annotations

import json
import time
from typing import Any

from factoriorl.paths import evidence_dir
from factoriorl.rcon import LuaError, RCONClient
from factoriorl.session import WorkerSession
from factoriorl.worker import WorkerManager

#: Ore is placed outside the initial view so the spike also exercises the
#: "resource exists but is not yet visible" case Phase 2.4 cares about.
ORE_POSITION = (12, 0)


class Probe:
    """Runs Lua through the evaluator channel and records what happened."""

    def __init__(self, client: RCONClient) -> None:
        self._client = client
        self.results: dict[str, Any] = {}

    def run(self, name: str, code: str) -> Any:
        try:
            value = self._client.lua(code)
            self.results[name] = {"ok": True, "value": value}
            return value
        except (LuaError, OSError) as exc:
            self.results[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            return None


def _setup_world() -> str:
    """Create ore, a furnace target and stone next to the character."""
    x, y = ORE_POSITION
    return f"""
    local srf = game.surfaces["nauvis"]
    local ch = storage.frrl_character
    ch.teleport({{0, 0}})
    for dx = 0, 3 do
      for dy = -1, 1 do
        local p = {{ {x} + dx, {y} + dy }}
        if srf.can_place_entity({{name = "iron-ore", position = p}}) then
          srf.create_entity({{name = "iron-ore", position = p, amount = 500}})
        end
      end
    end
    for dx = 0, 2 do
      srf.create_entity({{name = "stone", position = {{ {x} + dx, {y} + 4 }}, amount = 300}})
    end
    local inv = ch.get_inventory(defines.inventory.character_main)
    inv.clear()
    inv.insert({{name = "stone-furnace", count = 2}})
    inv.insert({{name = "transport-belt", count = 2}})
    return {{
      ore = srf.count_entities_filtered({{
        area = {{{{ {x} - 1, {y} - 2 }}, {{ {x} + 5, {y} + 6 }}}}, type = "resource"}}),
      character = ch.position.x .. "," .. ch.position.y,
    }}
    """


def run_control_spike(worker_id: str = "spike-control") -> dict[str, Any]:
    manager = WorkerManager()
    handle = manager.launch(worker_id)
    report: dict[str, Any] = {"engine": handle.engine.to_dict(), "probes": {}}
    try:
        client = RCONClient(handle.spec.rcon_endpoint, timeout=30.0)
        client.connect()
        probe = Probe(client)
        try:
            probe.run("setup", _setup_world())

            # --- Reach and prototype constants -------------------------------
            probe.run(
                "reach_properties",
                """
                local ch = storage.frrl_character
                return {
                  build_distance = ch.build_distance,
                  reach_distance = ch.reach_distance,
                  resource_reach_distance = ch.resource_reach_distance,
                  item_pickup_distance = ch.item_pickup_distance,
                  drop_item_distance = ch.drop_item_distance,
                }
                """,
            )

            # --- 1. Does `selected` stick without a player? -------------------
            x, y = ORE_POSITION
            probe.run(
                "selected_sticks",
                f"""
                local srf = game.surfaces["nauvis"]
                local ch = storage.frrl_character
                ch.teleport({{ {x} - 2, {y} }})
                local ore = srf.find_entities_filtered({{
                  position = {{ {x}, {y} }}, radius = 1, type = "resource"}})[1]
                if not ore then return {{ error = "no ore found" }} end
                ch.update_selected_entity(ore.position)
                local sel = ch.selected
                return {{
                  had_ore = true,
                  selected_present = sel ~= nil,
                  selected_name = sel and sel.name or nil,
                  can_reach = ch.can_reach_entity(ore),
                  distance = math.sqrt((ch.position.x - ore.position.x)^2
                                     + (ch.position.y - ore.position.y)^2),
                }}
                """,
            )

            # --- 12. Do resources carry a unit_number? -----------------------
            probe.run(
                "resource_unit_number",
                f"""
                local srf = game.surfaces["nauvis"]
                local ore = srf.find_entities_filtered({{
                  position = {{ {x}, {y} }}, radius = 1, type = "resource"}})[1]
                local chest = srf.create_entity({{
                  name = "wooden-chest", position = {{ -5, -5 }}, force = "player"}})
                return {{
                  resource_unit_number = ore and ore.unit_number or nil,
                  resource_has_unit_number = (ore and ore.unit_number) ~= nil,
                  chest_unit_number = chest and chest.unit_number or nil,
                  mining_time = ore and ore.prototype.mineable_properties.mining_time or nil,
                }}
                """,
            )

            # --- 6. cursor_stack on a player-less character ------------------
            probe.run(
                "cursor_stack",
                """
                local ch = storage.frrl_character
                local ok, cs = pcall(function() return ch.cursor_stack end)
                return { accessible = ok, is_nil = (cs == nil) }
                """,
            )

            # --- 2/3. Native mining ------------------------------------------
            probe.run(
                "mining_start",
                f"""
                local srf = game.surfaces["nauvis"]
                local ch = storage.frrl_character
                local inv = ch.get_inventory(defines.inventory.character_main)
                local before = inv.get_item_count("iron-ore")
                local ore = srf.find_entities_filtered({{
                  position = {{ {x}, {y} }}, radius = 1, type = "resource"}})[1]
                ch.update_selected_entity(ore.position)
                ch.mining_state = {{ mining = true, position = ore.position }}
                return {{
                  ore_before = before,
                  mining_flag = ch.mining_state.mining,
                  progress = ch.character_mining_progress,
                }}
                """,
            )

            with WorkerSession(handle, timeout=30.0) as session:
                session.status()
                session.advance(30)
                probe.run(
                    "mining_after_30t",
                    """
                    local ch = storage.frrl_character
                    local inv = ch.get_inventory(defines.inventory.character_main)
                    return {
                      ore_count = inv.get_item_count("iron-ore"),
                      progress = ch.character_mining_progress,
                      still_mining = ch.mining_state.mining,
                    }
                    """,
                )
                session.advance(120)
                probe.run(
                    "mining_after_150t",
                    """
                    local ch = storage.frrl_character
                    local inv = ch.get_inventory(defines.inventory.character_main)
                    ch.mining_state = { mining = false }
                    return {
                      ore_count = inv.get_item_count("iron-ore"),
                      progress = ch.character_mining_progress,
                    }
                    """,
                )

                # --- 4/5. Native crafting --------------------------------------
                probe.run(
                    "crafting_begin",
                    """
                    local ch = storage.frrl_character
                    local inv = ch.get_inventory(defines.inventory.character_main)
                    inv.insert({ name = "iron-plate", count = 10 })
                    local craftable = ch.get_craftable_count("iron-gear-wheel")
                    local started = ch.begin_crafting({ recipe = "iron-gear-wheel", count = 2 })
                    return {
                      craftable_count = craftable,
                      started = started,
                      queue_size = #ch.crafting_queue,
                      plates_after_start = inv.get_item_count("iron-plate"),
                      recipe_energy = prototypes.recipe["iron-gear-wheel"].energy,
                    }
                    """,
                )
                session.advance(90)
                probe.run(
                    "crafting_after_90t",
                    """
                    local ch = storage.frrl_character
                    local inv = ch.get_inventory(defines.inventory.character_main)
                    return {
                      gears = inv.get_item_count("iron-gear-wheel"),
                      plates = inv.get_item_count("iron-plate"),
                      queue_size = ch.crafting_queue and #ch.crafting_queue or 0,
                      queue_progress = ch.crafting_queue_progress,
                    }
                    """,
                )

                # --- 7. Assembled placement ------------------------------------
                probe.run(
                    "placement",
                    """
                    local srf = game.surfaces["nauvis"]
                    local ch = storage.frrl_character
                    local inv = ch.get_inventory(defines.inventory.character_main)
                    local pos = { ch.position.x + 2, ch.position.y }
                    local can = srf.can_place_entity({
                      name = "stone-furnace", position = pos, force = "player",
                      build_check_type = defines.build_check_type.manual })
                    local built = nil
                    if can then
                      local e = srf.create_entity({
                        name = "stone-furnace", position = pos, force = "player" })
                      if e then
                        inv.remove({ name = "stone-furnace", count = 1 })
                        built = e.unit_number
                      end
                    end
                    local can_again = srf.can_place_entity({
                      name = "stone-furnace", position = pos, force = "player",
                      build_check_type = defines.build_check_type.manual })
                    return {
                      can_place = can, built_unit_number = built,
                      can_place_again = can_again,
                      furnaces_left = inv.get_item_count("stone-furnace"),
                    }
                    """,
                )

                # --- 8. Rotation without by_player -----------------------------
                probe.run(
                    "rotation",
                    """
                    local srf = game.surfaces["nauvis"]
                    local ch = storage.frrl_character
                    local belt = srf.create_entity({ name = "transport-belt",
                      position = { ch.position.x, ch.position.y + 2 }, force = "player" })
                    local before = belt.direction
                    local rotated = belt.rotate({})
                    local furnace = srf.find_entities_filtered({
                      name = "stone-furnace", limit = 1 })[1]
                    local furnace_rotated = furnace and furnace.rotate({}) or nil
                    return {
                      belt_before = before, belt_rotated = rotated,
                      belt_after = belt.direction,
                      furnace_rotate_result = furnace_rotated,
                    }
                    """,
                )

                # --- 9/10. Research and recipe selection ------------------------
                probe.run(
                    "research_and_recipe",
                    """
                    local force = game.forces["player"]
                    local out = {}
                    out.research_enabled = force.research_enabled
                    local root = nil
                    for name, tech in pairs(force.technologies) do
                      if not tech.researched and tech.enabled then
                        local n = 0
                        for _ in pairs(tech.prerequisites) do n = n + 1 end
                        if n == 0 then root = name break end
                      end
                    end
                    out.root = root

                    -- a) add_research with a name string
                    out.add_by_name = force.add_research(root)
                    out.current_a = force.current_research and force.current_research.name or nil

                    -- b) add_research with the LuaTechnology object
                    out.add_by_object = force.add_research(force.technologies[root])
                    out.current_b = force.current_research and force.current_research.name or nil

                    -- c) write the research queue directly (current_research is read-only)
                    local ok_q, err_q = pcall(function()
                      force.research_queue = { root }
                    end)
                    out.queue_write_ok = ok_q
                    out.queue_write_err = ok_q and nil or tostring(err_q)
                    out.current_c = force.current_research and force.current_research.name or nil
                    out.queue_len_after = #force.research_queue

                    -- What blocks it? Inspect the technology itself.
                    local t = force.technologies[root]
                    out.tech_enabled = t.enabled
                    out.tech_researched = t.researched
                    out.tech_unit_count = t.research_unit_count
                    out.tech_visible = t.visible_when_disabled

                    -- Every zero-prerequisite technology in 2.0 base is
                    -- trigger-based (craft N of an item), so none is queueable
                    -- at game start. Classify the tree, then prove the positive
                    -- case by satisfying prerequisites through the evaluator.
                    local triggered, queueable = 0, 0
                    for name, tech in pairs(force.technologies) do
                      if tech.prototype.research_trigger then
                        triggered = triggered + 1
                      else
                        queueable = queueable + 1
                      end
                    end
                    out.trigger_based_count = triggered
                    out.queueable_count = queueable
                    out.root_is_trigger_based = t.prototype.research_trigger ~= nil
                    out.root_trigger = t.prototype.research_trigger
                      and t.prototype.research_trigger.type or nil

                    -- Positive case: research the trigger chain outright, then a
                    -- unit-based technology must become selectable.
                    force.technologies["steam-power"].researched = true
                    force.technologies["electronics"].researched = true
                    force.technologies["automation-science-pack"].researched = true
                    out.add_after_prereqs = force.add_research("logistics")
                    out.current_after_prereqs = force.current_research
                      and force.current_research.name or nil
                    out.logistics_is_trigger_based =
                      force.technologies["logistics"].prototype.research_trigger ~= nil

                    -- Unsatisfiable case, for contrast.
                    out.add_locked = force.add_research("rocket-silo")
                    out.circuit_recipe_enabled = force.recipes["electronic-circuit"].enabled
                    return out
                    """,
                )

                # --- 11. Lifecycle detection -----------------------------------
                probe.run(
                    "object_destroyed_registration",
                    """
                    local srf = game.surfaces["nauvis"]
                    local chest = srf.create_entity({
                      name = "wooden-chest", position = { -8, -8 }, force = "player" })
                    local reg = script.register_on_object_destroyed(chest)
                    storage.frrl_spike_reg = reg
                    local unit = chest.unit_number
                    chest.destroy({ raise_destroy = false })
                    return { registration_number = reg, unit_number = unit,
                             destroyed = true }
                    """,
                )

                # --- 13. Observation query cost --------------------------------
                for label, body in (
                    ("noop", "local _ = 1"),
                    ("entities", "n = #srf.find_entities_filtered({position = p, radius = 32})"),
                    (
                        "resources",
                        "n = #srf.find_entities_filtered("
                        "{position = p, radius = 32, type = 'resource'})",
                    ),
                    (
                        "water_tiles",
                        "n = #srf.find_tiles_filtered({area = "
                        "{{p.x - 32, p.y - 32}, {p.x + 32, p.y + 32}}, "
                        "collision_mask = 'water_tile'})",
                    ),
                ):
                    reps = 200
                    code = (
                        "local srf = game.surfaces['nauvis'] "
                        "local p = storage.frrl_character.position "
                        "local n = 0 "
                        f"for _ = 1, {reps} do {body} end "
                        "return n"
                    )
                    started = time.perf_counter()
                    value = probe.run(f"query_{label}", code)
                    elapsed_ms = (time.perf_counter() - started) * 1000.0
                    probe.results[f"query_{label}"]["total_ms"] = round(elapsed_ms, 3)
                    probe.results[f"query_{label}"]["reps"] = reps
                    probe.results[f"query_{label}"]["count"] = value

            report["probes"] = probe.results
        finally:
            client.close()
    finally:
        report["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        evidence_dir().mkdir(parents=True, exist_ok=True)
        (evidence_dir() / "phase2-capability-spike.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        manager.cleanup(handle)
    return report
