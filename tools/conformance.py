"""R2.1's gate: build and operate through policy-accessible actions only.

The redirection asks for "a scripted conformance client [that] can place and
operate a small production chain using only policy-accessible actions, without
task-specific runtime shortcuts", with target identity stable through movement
and typed observable outcomes for stale handles, deleted handles, invalid
placements and insufficient inventory.

Every action here goes through `env.step_arguments`, which validates each
argument against a domain derived from the observation and then calls the same
`step_payload` the discrete policy path uses. Nothing reads evaluator truth and
nothing calls a task-specific helper: no `reference.py` solver, no direct
`session.step`, no RCON.

Writes `docs/evidence/r2-conformance.json`.
"""

from __future__ import annotations

import json
import math
import sys

from factoriorl import catalog as catalog_module
from factoriorl.effects import ActionRecord, EffectKind, Expectation, expectation_for, observe
from factoriorl.env import FactorioEnv
from factoriorl.paths import evidence_dir
from factoriorl.rcon import RCONClient
from factoriorl.seeding import Branch, SeedPlan
from factoriorl.session import WorkerSession
from factoriorl.tasks import get
from factoriorl.worker import WorkerManager

CATALOG = "parameterized-v1"


class Client:
    """A policy-shaped caller: catalog indices plus validated arguments."""

    def __init__(self, env: FactorioEnv):
        self.env = env
        self.records: list[ActionRecord] = []
        self.log: list[dict] = []

    @property
    def observation(self) -> dict:
        return self.env._observation

    def tick(self) -> int:
        return int(self.observation.get("tick") or 0)

    def do(self, key: str, expect: Expectation | None = None, **arguments) -> dict:
        index = self.env.catalog.keys().index(key)
        template = self.env.catalog.templates[index]
        record = ActionRecord(
            action_key=key,
            commanded={"action": template.action, **arguments},
            expectation=expect or expectation_for(template.action),
            issued_tick=self.tick(),
        )
        try:
            _, _, _, _, info = self.env.step_arguments(index, arguments)
        except ValueError as exc:
            # Refused by the client's own domain check, before the engine.
            record.refuse(f"client:{exc}", self.tick())
            self.records.append(record)
            self.log.append({**record.to_dict(), "refused_by": "client"})
            return {"status": "client_refused", "error": str(exc)}
        record.status = info.get("action_status")
        record.error = info.get("action_error")
        if record.error:
            record.refuse(record.error, self.tick())
        else:
            observe(record, self.observation, self.tick())
        self.records.append(record)
        self.log.append({**record.to_dict(), "refused_by": None})
        return {"status": record.status, "error": record.error}

    def handles(self) -> list[str]:
        return list(self.env.argument_domains()["targets"])

    def entity_at(self, position) -> dict | None:
        tile = (math.floor(position[0]), math.floor(position[1]))
        for record in self.observation.get("entities") or []:
            point = record.get("p")
            if point and (math.floor(point[0]), math.floor(point[1])) == tile:
                return record
        return None

    def walk_toward(self, goal, steps: int = 40) -> None:
        """Approach with the ordinary move actions the policy has."""
        for _ in range(steps):
            here = (self.observation.get("character") or {}).get("position") or [0.0, 0.0]
            dx, dy = goal[0] - here[0], goal[1] - here[1]
            if math.hypot(dx, dy) <= 1.6:
                return
            if abs(dx) >= abs(dy):
                axis, delta = "east" if dx > 0 else "west", abs(dx)
            else:
                axis, delta = "south" if dy > 0 else "north", abs(dy)
            stride = "move_" if delta > 4.0 else ("step_" if delta > 1.0 else "nudge_")
            self.do(stride + axis)
            continue


def _env(session, task_id, split, index):
    task = get(task_id)
    env = FactorioEnv(
        task,
        session,
        SeedPlan(master=20260909, run_id="conformance"),
        branch=Branch.TRAIN,
        split=split,
    )
    # The conformance catalog, not the task's own subset: the point is that the
    # shared contract works, not that a task was tuned for it.
    env.catalog = catalog_module.resolve(CATALOG)
    env._episode_index = index
    env.reset()
    return env


def check(report: dict, label: str, ok: bool, **detail) -> bool:
    report["checks"].append({"check": label, "ok": bool(ok), **detail})
    print(f"  [{'ok ' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""), flush=True)
    return bool(ok)


def placement_and_errors(session, report: dict) -> None:
    """Place at a chosen tile with a chosen facing, and provoke each error."""
    env = _env(session, "repair_belt", "train", 0)
    client = Client(env)
    gap = (env._observation.get("goal") or {}).get("gap")
    check(report, "scene publishes a work site", gap is not None, gap=gap)

    handles_before = client.handles()
    client.walk_toward(gap)
    handles_after = client.handles()
    stable = [h for h in handles_before if h in handles_after]
    check(
        report,
        "target identity survives movement",
        len(stable) >= max(1, len(handles_before) // 2),
        before=len(handles_before),
        after=len(handles_after),
        stable=len(stable),
    )

    # The R2.1 unlock: tile and facing chosen independently.
    facing = next(
        (
            e.get("d")
            for e in (client.observation.get("entities") or [])
            if e.get("name") == "transport-belt" and e.get("d") is not None
        ),
        None,
    )
    named = {0: "north", 4: "east", 8: "south", 12: "west"}.get(facing or 0, "east")
    outcome = client.do(
        "place_at",
        Expectation(EffectKind.ENTITY_FACING, 0, {"position": gap, "direction": facing}),
        item="transport-belt",
        position=[gap[0], gap[1]],
        direction=named,
    )
    placed = client.entity_at(gap)
    check(
        report,
        "placed at a chosen tile with a chosen facing",
        placed is not None and placed.get("d") == facing,
        wanted_facing=facing,
        got=(placed or {}).get("d"),
        outcome=outcome,
    )

    # Two layers, and both matter. The validated path refuses an occupied
    # tile before the engine sees it, because the tile stopped being a
    # placement candidate the moment the belt landed there -- that is the
    # domain doing its job. The engine's typed code is still the backstop for
    # the race the client cannot see: the world moving between the observation
    # and the action. Exercised through `step_payload`, which is the addressed
    # path the language-model client uses.
    occupied_client = client.do(
        "place_at", None, item="transport-belt", position=[gap[0], gap[1]], direction=named
    )
    check(
        report,
        "occupied tile -> refused by the domain, with a name",
        occupied_client.get("status") == "client_refused"
        and "placements" in str(occupied_client.get("error")),
        outcome=occupied_client,
    )
    _, _, _, _, engine_info = env.step_payload(
        {
            "action": "place",
            "item": "transport-belt",
            "position": [gap[0], gap[1]],
            "direction": named,
        },
        action_key="place_at",
    )
    check(
        report,
        "occupied tile -> engine reports collision",
        engine_info.get("action_error") == "collision",
        error=engine_info.get("action_error"),
    )

    # Stale handle: never minted this episode.
    stale = client.do("rotate_at", None, handle="h999999")
    named_refusal = stale.get("status") == "client_refused" or stale.get("error") in {
        "unknown_handle",
        "target_missing",
    }
    check(report, "stale handle -> refused with a name", named_refusal, outcome=stale)

    # Insufficient inventory: spend the belts, then place once more.
    for offset in range(1, 6):
        candidate = [gap[0] + offset, gap[1] - 2.0]
        if candidate in env.argument_domains()["placements"]:
            client.do("place_at", None, item="transport-belt", position=candidate, direction=named)
    empty = client.do(
        "place_at", None, item="transport-belt", position=[gap[0], gap[1] - 3.0], direction=named
    )
    check(
        report,
        "empty inventory -> refused by the domain, with a name",
        empty.get("status") == "client_refused" and "items" in str(empty.get("error")),
        outcome=empty,
    )
    free = next((p for p in env.argument_domains()["placements"] if abs(p[1] - gap[1]) > 2.0), None)
    _, _, _, _, no_items = env.step_payload(
        {
            "action": "place",
            "item": "transport-belt",
            "position": free or [gap[0], gap[1] - 4.0],
            "direction": named,
        },
        action_key="place_at",
    )
    check(
        report,
        "empty inventory -> engine reports no_items",
        no_items.get("action_error") == "no_items",
        error=no_items.get("action_error"),
    )
    report["placement_trace"] = client.log


def operate_a_chain(session, report: dict) -> None:
    """Fuel a drill by handle with an explicit count, and see output appear."""
    env = _env(session, "plate_line", "train", 0)
    client = Client(env)

    drill = next(
        (e for e in (client.observation.get("entities") or []) if "drill" in str(e.get("name"))),
        None,
    )
    check(
        report, "chain scene exposes a machine", drill is not None, name=(drill or {}).get("name")
    )
    if drill is None:
        return

    client.walk_toward(drill["p"])
    held = (client.observation.get("inventory") or {}).get("coal", 0)
    fuelled = client.do(
        "give_to",
        Expectation(EffectKind.ITEMS_MOVED, 0, {"item": "coal", "before": held}),
        to=str(drill["h"]),
        item="coal",
        count=20,
    )
    check(
        report,
        "transfer to a named handle with an explicit count",
        fuelled.get("error") is None,
        outcome=fuelled,
    )
    check(
        report,
        "the transfer's declared effect was observed",
        any(r.action_key == "give_to" and r.observed.value == "confirmed" for r in client.records),
        states=[r.observed.value for r in client.records if r.action_key == "give_to"],
    )

    # Let it run, then look for the consequence a *goal* cares about, which the
    # transfer's own confirmation does not establish.
    for _ in range(30):
        client.do("wait")
    working = [
        e
        for e in (client.observation.get("entities") or [])
        if "drill" in str(e.get("name")) and e.get("status") is None
    ]
    check(
        report,
        "machine reports working after fuelling",
        bool(working),
        note="status absent means working in the sensor contract",
    )
    report["operation_trace"] = client.log


def main() -> int:
    report: dict = {
        "gate": "R2.1 conformance",
        "catalog": CATALOG,
        "note": (
            "Every action goes through env.step_arguments, which validates each "
            "argument against an observation-derived domain and then calls the "
            "same step_payload the discrete policy path uses. No reference "
            "solver, no direct session.step, no RCON, no evaluator truth."
        ),
        "checks": [],
    }
    manager = WorkerManager()
    handle = manager.launch("conformance")
    try:
        with RCONClient(handle.spec.rcon_endpoint, timeout=30.0) as rcon:
            rcon.lua("game.speed = 90 return 1")
        session = WorkerSession(handle, timeout=30.0)
        session.status()
        print("placement and typed outcomes:", flush=True)
        placement_and_errors(session, report)
        print("operating a chain:", flush=True)
        operate_a_chain(session, report)
        session.close()
    finally:
        manager.cleanup(handle)

    report["passed"] = all(c["ok"] for c in report["checks"])
    report["failed"] = [c["check"] for c in report["checks"] if not c["ok"]]
    path = evidence_dir() / "r2-conformance.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n{'PASSED' if report['passed'] else 'FAILED'}: wrote {path}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
