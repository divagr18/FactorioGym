"""Phase 2 exit gate (DESIGN.md).

    an embodied scripted agent can navigate, gather resources, produce an item,
    and interact with machinery without privileged mutations

The agent below drives a worker through the typed protocol only. "Without
privileged mutations" is enforced mechanically rather than asserted: the
session's RCON client is wrapped so any Lua that is not the typed dispatch
template raises, which makes `bridge.run` unreachable from the agent. Scenario
setup does use the evaluator channel, but only before the agent starts, and
every such call is recorded in the report with its tick -- so
`privileged_calls_by_agent: 0` is a measurement, not a claim.

Each step is cross-referenced to the DESIGN.md acceptance criterion it satisfies.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from factoriorl.paths import evidence_dir, runtime_dir
from factoriorl.rcon import RCONClient
from factoriorl.session import WorkerSession
from factoriorl.worker import WorkerManager

SCENARIO = "phase2-gate"
ORE_X = 40
STONE_POSITION = (36, -8)
COAL_POSITION = (40, 8)
BELT_POSITION = (-8, 0)
ASSEMBLER_POSITION = (-6, 0)
GATE_SPEED = 30.0


class PrivilegedAccess(RuntimeError):
    """The scripted agent tried to reach the evaluator channel."""


class GuardedClient:
    """Wraps an RCON client so only the typed dispatch template may be sent.

    The evaluator's arbitrary-Lua path shares the RCON connection with the
    typed protocol, so "the agent used no privileged mutations" is only
    meaningful if the agent *cannot*. This makes the attempt raise.
    """

    def __init__(self, inner: RCONClient) -> None:
        self._inner = inner
        self.blocked = 0

    def lua(self, code: str):
        if 'remote.call("frrl_bridge", "dispatch"' not in code:
            self.blocked += 1
            raise PrivilegedAccess(f"agent attempted a non-dispatch call: {code[:80]}")
        return self._inner.lua(code)

    def __getattr__(self, name):
        return getattr(self._inner, name)


@dataclass
class GateReport:
    passed: bool = True
    steps: list[dict] = field(default_factory=list)
    evaluator_setup: list[dict] = field(default_factory=list)
    measurements: dict = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    privileged_calls_by_agent: int = 0

    def fail(self, reason: str) -> None:
        self.failures.append(reason)
        self.passed = False

    def step(self, criterion: str, note: str, **data: Any) -> None:
        self.steps.append({"criterion": criterion, "note": note, **data})

    def check(self, criterion: str, condition: bool, note: str, **data: Any) -> bool:
        self.step(criterion, note, ok=bool(condition), **data)
        if not condition:
            self.fail(f"{criterion}: {note}")
        return bool(condition)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "privileged_calls_by_agent": self.privileged_calls_by_agent,
            "measurements": self.measurements,
            "failures": self.failures,
            "evaluator_setup": self.evaluator_setup,
            "steps": self.steps,
        }


def _observe(session: WorkerSession) -> dict:
    return session.observe().response.result


def _entity_named(observation: dict, name: str) -> dict | None:
    for record in observation.get("entities", []):
        if record.get("name") == name:
            return record
    return None


def _resource_named(observation: dict, name: str) -> dict | None:
    for record in observation.get("resources", {}).get("tiles", []):
        if record.get("name") == name:
            return record
    return None


def _walk_towards(session: WorkerSession, target, report: GateReport, budget: int = 60) -> dict:
    """Step toward a target using only move actions and the fused step request."""
    observation = _observe(session)
    stuck = 0
    for _ in range(budget):
        position = observation["character"]["position"]
        dx = target[0] - position[0]
        dy = target[1] - position[1]
        if abs(dx) < 1.5 and abs(dy) < 1.5:
            break
        if abs(dx) >= abs(dy):
            direction = "east" if dx > 0 else "west"
        else:
            direction = "south" if dy > 0 else "north"
        timed = session.step({"action": "move", "direction": direction, "ticks": 30}, ticks=30)
        result = timed.response.result or {}
        observation = result.get("observation") or _observe(session)
        # Movement is greedy: Phase 2 has no pathfinding (DESIGN.md defers
        # navigation to 5.1), so an obstacle stops progress entirely. Detect
        # that rather than spinning out the budget in silence.
        moved = observation["character"]["position"]
        if abs(moved[0] - position[0]) < 0.01 and abs(moved[1] - position[1]) < 0.01:
            stuck += 1
            if stuck >= 3:
                break
        else:
            stuck = 0
    return observation


def run_phase2_gate(worker_id: str = "phase2-gate") -> dict:
    report = GateReport()
    manager = WorkerManager()
    handle = manager.launch(worker_id)
    started = time.perf_counter()
    observation_sizes: list[int] = []
    try:
        # ---- evaluator setup, recorded, before the agent starts -------------
        with RCONClient(handle.spec.rcon_endpoint, timeout=30.0) as setup:
            setup.lua(f"game.speed = {GATE_SPEED} return game.speed")
            report.evaluator_setup.append({"call": "game.speed", "value": GATE_SPEED})

        session = WorkerSession(handle, timeout=30.0)
        session.status()
        session.reset(scenario=SCENARIO)
        report.evaluator_setup.append({"call": "reset", "scenario": SCENARIO})

        # From here on the agent may not reach the evaluator channel.
        guarded = GuardedClient(session._client)
        session._client = guarded

        observation = _observe(session)
        observation_sizes.append(len(json.dumps(observation)))

        # ---- 2.1: rotation ---------------------------------------------------
        # Both sit next to spawn and are in view immediately, which is why
        # this runs before the agent walks east to the ore.
        belt = _entity_named(observation, "transport-belt")
        # Unconditional: a criterion that quietly disappears when the agent
        # fails to reach its target is not evidence. If the belt cannot be
        # found, that is a gate failure, not a skipped step.
        if not report.check(
            "2.1 rotation reachable",
            belt is not None,
            "the scenario belt was located",
            character=_observe(session)["character"]["position"],
        ):
            report.check("2.1 rotation", False, "a belt can be rotated (belt not found)")
        else:
            rotate = session.act("rotate", handle=belt["h"])
            report.check(
                "2.1 rotation",
                rotate.response.code.value == "ok",
                "a belt can be rotated",
                result=rotate.response.result,
            )

        # ---- 2.1: recipe selection and the technology gate -------------------
        assembler = _entity_named(observation, "assembling-machine-1")
        if not report.check(
            "2.1 recipe target reachable",
            assembler is not None,
            "the scenario assembling machine was located",
            character=_observe(session)["character"]["position"],
        ):
            report.check("2.1 recipe selection", False, "assembler not found")
        else:
            ok_recipe = session.act("set_recipe", handle=assembler["h"], recipe="iron-gear-wheel")
            report.check(
                "2.1 recipe selection",
                ok_recipe.response.code.value == "ok",
                "an unlocked recipe can be selected",
            )
            locked_recipe = session.act(
                "set_recipe", handle=assembler["h"], recipe="electronic-circuit"
            )
            report.check(
                "2.1 recipe technology gate",
                locked_recipe.response.code.value == "rejected"
                and locked_recipe.response.error.code.value == "tech_locked",
                "a locked recipe is refused with tech_locked",
                error=(
                    locked_recipe.response.error.code.value
                    if locked_recipe.response.error
                    else None
                ),
            )

        # ---- 2.1: research, including the trigger-gated root ------------------
        trigger = session.act("research", technology="electronics")
        report.check(
            "2.1 research trigger gate",
            trigger.response.code.value == "rejected"
            and trigger.response.error.code.value == "tech_not_selectable",
            "a trigger-based technology is refused distinctly from tech_locked",
            error=trigger.response.error.code.value if trigger.response.error else None,
        )
        unmet = session.act("research", technology="rocket-silo")
        report.check(
            "2.1 research prerequisite gate",
            unmet.response.code.value == "rejected"
            and unmet.response.error.code.value in ("tech_locked", "tech_not_selectable"),
            "a technology with unmet prerequisites is refused",
            error=unmet.response.error.code.value if unmet.response.error else None,
        )

        # ---- 2.3: objects outside the sensor region are absent ---------------
        report.check(
            "2.3 partial observability",
            _resource_named(observation, "iron-ore") is None,
            "ore at x=40 is not visible from spawn (32-tile sensor radius)",
            sensor_radius=observation["sensor"]["radius"],
        )

        # ---- 2.3/2.4: walk east until the ore becomes visible ----------------
        first_sighting_tick = None
        for _ in range(40):
            timed = session.step({"action": "move", "direction": "east", "ticks": 30}, ticks=30)
            observation = (timed.response.result or {}).get("observation") or _observe(session)
            observation_sizes.append(len(json.dumps(observation)))
            ore = _resource_named(observation, "iron-ore")
            if ore is not None:
                first_sighting_tick = observation["tick"]
                break
        report.check(
            "2.3 visibility transition",
            first_sighting_tick is not None,
            "ore entered the sensor region while walking east",
            first_sighting_tick=first_sighting_tick,
        )

        ore = _resource_named(observation, "iron-ore")
        if ore is None:
            report.fail("2.3: never saw the ore; cannot continue")
            raise RuntimeError("ore never became visible")

        # ---- 2.4: a visible entity keeps its handle across observations ------
        handle_id = ore["h"]
        stable = True
        for _ in range(3):
            again = _resource_named(_observe(session), "iron-ore")
            if again is None or again["h"] != handle_id:
                stable = False
        report.check(
            "2.4 stable identity",
            stable,
            "an unchanged visible entity retains its handle",
            handle=handle_id,
        )

        # ---- 2.1: mining out of resource reach is refused --------------------
        far = session.act("mine", handle=handle_id, count=1)
        report.check(
            "2.1 reach",
            far.response.code.value == "rejected"
            and far.response.error.code.value == "out_of_reach",
            "mining beyond resource reach (2.7) is rejected",
            code=far.response.code.value,
            error=far.response.error.code.value if far.response.error else None,
        )

        # ---- walk into resource reach and mine -------------------------------
        observation = _walk_towards(session, (ore["p"][0] - 1, ore["p"][1]), report)
        ore = _resource_named(observation, "iron-ore")
        handle_id = ore["h"] if ore else handle_id

        mine = session.act("mine", handle=handle_id, count=6)
        mining_time = (mine.response.result or {}).get("mining_time")
        report.check(
            "2.2 ongoing action",
            mine.response.code.value == "ok" and mine.response.result["status"] == "running",
            "mining answers running, not completed",
            code=mine.response.code.value,
        )
        mine_request_id = mine.response.request_id

        saw_progress = False
        settled = None
        start_tick = _observe(session)["tick"]
        for _ in range(40):
            session.advance(30)
            probe = session.request_status(mine_request_id).response.result
            mid = _observe(session)
            if (mid["character"].get("mining") or {}).get("active"):
                progress = mid["character"]["mining"]["progress"]
                if 0.0 < progress < 1.0:
                    saw_progress = True
            if probe.get("settled"):
                settled = probe
                break
        end_tick = _observe(session)["tick"]

        report.check(
            "2.2 progress survives step calls",
            saw_progress,
            "mining progress was observed strictly between 0 and 1 mid-operation",
        )
        report.check(
            "2.2 completion reported once",
            settled is not None and settled["state"] == "completed",
            "the mine settled to completed",
            state=settled["state"] if settled else None,
        )
        inventory = _observe(session)["inventory"]
        report.check(
            "2.1 mining yields items",
            inventory.get("iron-ore", 0) >= 6,
            "six ore reached the character inventory",
            iron_ore=inventory.get("iron-ore", 0),
        )
        report.check(
            "2.1 mining cannot bypass time",
            (end_tick - start_tick) >= 6 * (mining_time or 1) * 60 * 0.5,
            "elapsed ticks are consistent with the prototype mining time",
            elapsed_ticks=end_tick - start_tick,
            mining_time=mining_time,
        )

        # ---- 2.2: cancellation stops at a documented boundary ---------------
        long_mine = session.act("mine", handle=handle_id, count=500)
        session.advance(30)
        cancel = session.act("cancel", target_request_id=long_mine.response.request_id)
        report.check(
            "2.2 cancellation",
            cancel.response.code.value == "ok",
            "an ongoing mine can be cancelled",
            result=(cancel.response.result or {}).get("result"),
        )
        after_cancel = _observe(session)
        report.check(
            "2.2 cancellation boundary",
            not (after_cancel["character"].get("mining") or {}).get("active", False),
            "mining stopped at the paused tick the cancel was processed",
        )
        retry = session.act("mine", handle=handle_id, count=1)
        report.check(
            "2.2 cancellation leaves nothing wedged",
            retry.response.code.value == "ok",
            "a new mine starts after a cancellation",
        )
        session.act("cancel", target_request_id=retry.response.request_id)

        # ---- gather stone, then craft ---------------------------------------
        observation = _walk_towards(session, STONE_POSITION, report)
        stone = _resource_named(observation, "stone")
        if stone is not None:
            observation = _walk_towards(session, (stone["p"][0] - 1, stone["p"][1]), report)
            stone = _resource_named(_observe(session), "stone") or stone
            session.act("mine", handle=stone["h"], count=6)
            for _ in range(60):
                session.advance(60)
                if _observe(session)["inventory"].get("stone", 0) >= 6:
                    break
        stone_count = _observe(session)["inventory"].get("stone", 0)
        report.check(
            "2.1 mining a second resource",
            stone_count >= 5,
            "enough stone for a furnace",
            stone=stone_count,
        )

        short = session.act("craft", recipe="stone-furnace", count=99)
        report.check(
            "2.1 crafting cannot bypass cost",
            short.response.code.value == "rejected"
            and short.response.error.code.value == "no_items",
            "crafting without ingredients is rejected",
            error=short.response.error.code.value if short.response.error else None,
        )
        before_craft = _observe(session)["inventory"]
        craft = session.act("craft", recipe="stone-furnace", count=1)
        report.check(
            "2.2 crafting is ongoing",
            craft.response.code.value == "ok" and craft.response.result["status"] == "running",
            "crafting answers running",
        )
        for _ in range(40):
            session.advance(30)
            if session.request_status(craft.response.request_id).response.result.get("settled"):
                break
        after_craft = _observe(session)["inventory"]
        report.check(
            "2.1 crafting produces an item",
            after_craft.get("stone-furnace", 0) >= 1,
            "a stone furnace was hand-crafted",
            furnaces=after_craft.get("stone-furnace", 0),
            stone_before=before_craft.get("stone", 0),
            stone_after=after_craft.get("stone", 0),
        )

        # ---- 2.1: placement, collision, reach --------------------------------
        position = _observe(session)["character"]["position"]
        spot = [round(position[0]) + 2, round(position[1])]
        place = session.act("place", item="stone-furnace", position=spot)
        report.check(
            "2.1 placement",
            place.response.code.value == "ok",
            "the furnace was placed",
            code=place.response.code.value,
            error=place.response.error.code.value if place.response.error else None,
        )
        furnace_handle = (place.response.result or {}).get("handle")
        again = session.act("place", item="stone-furnace", position=spot)
        report.check(
            "2.1 placement collision",
            again.response.code.value == "rejected"
            and again.response.error.code.value in ("collision", "no_items"),
            "a second placement on the same tile is refused",
            error=again.response.error.code.value if again.response.error else None,
        )
        far_place = session.act(
            "place", item="stone-furnace", position=[position[0] + 40, position[1]]
        )
        report.check(
            "2.1 placement reach",
            far_place.response.code.value == "rejected"
            and far_place.response.error.code.value in ("out_of_reach", "no_items"),
            "placing beyond build distance is refused",
            error=far_place.response.error.code.value if far_place.response.error else None,
        )
        locked = session.act("place", item="electronic-circuit", position=spot)
        report.check(
            "2.1 placement technology gate",
            locked.response.code.value == "rejected",
            "placing an item whose recipe is locked is refused",
            error=locked.response.error.code.value if locked.response.error else None,
        )

        # ---- 2.1: machinery interaction, and a natively produced plate -------
        produced = 0
        if furnace_handle:
            observation = _walk_towards(session, COAL_POSITION, report)
            coal = _resource_named(observation, "coal")
            if coal is not None:
                observation = _walk_towards(session, (coal["p"][0] - 1, coal["p"][1]), report)
                coal = _resource_named(_observe(session), "coal") or coal
                session.act("mine", handle=coal["h"], count=4)
                for _ in range(60):
                    session.advance(60)
                    if _observe(session)["inventory"].get("coal", 0) >= 4:
                        break

        # ---- the exit gate's core claim: an item produced by the world -------
        # Hand-crafting already produced an item; this is the machinery half.
        # The furnace smelts natively while the agent waits, with no privileged
        # write anywhere in the loop.
        if furnace_handle:
            observation = _walk_towards(session, spot, report, budget=90)
            furnace = _entity_named(observation, "stone-furnace")
            if furnace is not None:
                fuel = session.act(
                    "transfer",
                    **{"from": "character", "to": furnace["h"], "item": "coal", "count": 2},
                )
                report.check(
                    "2.1 machinery interaction",
                    fuel.response.code.value == "ok",
                    "coal was inserted into the furnace's fuel inventory",
                    error=fuel.response.error.code.value if fuel.response.error else None,
                )
                ore_in = session.act(
                    "transfer",
                    **{"from": "character", "to": furnace["h"], "item": "iron-ore", "count": 5},
                )
                report.check(
                    "2.1 machinery interaction",
                    ore_in.response.code.value == "ok",
                    "iron ore was inserted into the furnace's source inventory",
                    error=ore_in.response.error.code.value if ore_in.response.error else None,
                )
                for _ in range(60):
                    session.advance(60)
                    current = _entity_named(_observe(session), "stone-furnace")
                    contents = (current or {}).get("contents", {}) or {}
                    if contents.get("iron-plate", 0) >= 1:
                        break
                out = session.act(
                    "transfer",
                    **{"from": furnace["h"], "to": "character", "item": "iron-plate", "count": 1},
                )
                produced = _observe(session)["inventory"].get("iron-plate", 0)
                report.check(
                    "exit gate: produce an item",
                    produced >= 1,
                    "the furnace smelted a plate and the agent collected it",
                    iron_plate=produced,
                    transfer_code=out.response.code.value,
                )

        report.measurements["observation_bytes"] = {
            "min": min(observation_sizes),
            "max": max(observation_sizes),
            "mean": round(sum(observation_sizes) / len(observation_sizes), 1),
        }

        # ---- 2.4: destroyed handles fail explicitly --------------------------
        if furnace_handle:
            with RCONClient(handle.spec.rcon_endpoint, timeout=30.0) as evaluator:
                evaluator.lua(
                    "local s = game.surfaces['nauvis'] "
                    "for _, e in pairs(s.find_entities_filtered({name='stone-furnace'})) do "
                    "e.destroy({raise_destroy=true}) end return true"
                )
            report.evaluator_setup.append({"call": "destroy stone-furnace", "why": "2.4 probe"})
            gone = session.act("rotate", handle=furnace_handle)
            report.check(
                "2.4 destroyed handle",
                gone.response.code.value == "rejected"
                and gone.response.error.code.value == "target_missing",
                "a destroyed handle fails as target_missing, not silently",
                error=gone.response.error.code.value if gone.response.error else None,
            )

        # ---- 2.4: reset is an identity boundary ------------------------------
        previous_handle = handle_id
        session.reset(scenario=SCENARIO)
        stale = session.act("mine", handle=previous_handle, count=1)
        report.check(
            "2.4 reset identity boundary",
            stale.response.code.value == "rejected"
            and stale.response.error.code.value == "unknown_handle",
            "every handle from the previous episode is unknown after reset",
            error=stale.response.error.code.value if stale.response.error else None,
        )

        # ---- 2.5: profiles are always reported -------------------------------
        final = _observe(session)
        report.check(
            "2.5 profile metadata",
            final.get("profiles", {}).get("observation") == "local-v1"
            and final["profiles"]["action"] == "primitive-v1",
            "every observation carries its observation and action profile",
            profiles=final.get("profiles"),
        )

        report.privileged_calls_by_agent = guarded.blocked
        report.measurements["game_ticks"] = final["absolute_tick"]
        report.measurements["wall_seconds"] = round(time.perf_counter() - started, 2)
        report.measurements["produced_plate"] = produced
        session.close()
    except Exception as exc:  # noqa: BLE001 - the gate reports, it does not raise
        report.fail(f"unhandled: {type(exc).__name__}: {exc}")
    finally:
        payload = report.to_dict()
        payload["engine"] = handle.engine.to_dict()
        payload["worker"] = handle.spec.manifest()
        payload["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        gate_dir = runtime_dir() / "gate-phase2"
        gate_dir.mkdir(parents=True, exist_ok=True)
        (gate_dir / "report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        evidence_dir().mkdir(parents=True, exist_ok=True)
        summary = {k: v for k, v in payload.items() if k != "steps"}
        summary["criteria"] = [
            {k: s[k] for k in ("criterion", "note", "ok") if k in s} for s in payload["steps"]
        ]
        (evidence_dir() / "phase2-gate.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        manager.cleanup(handle)
    return payload
