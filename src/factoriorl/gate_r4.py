"""R4 exit gate: fresh observations, a real outage, three declared tracks.

    R4.1 -- one refresh path: anything that advances time or writes to the
    world leaves the env's cached observation current.
    R4.2 -- a disruption whose production effect is measured against declared
    criteria, from a validated common state, with a control arm.
    R4.3 -- three declared tracks, each reporting its own hints and assistance
    and reaching its own outcome.

Four clauses, each measured, none asserted.

**Fresh by construction, not by discipline.** The Phase 5 demonstration
advanced 3,600 ticks with `session.step` around the env and injected its outage
with raw Lua, so the env never learned either had happened. Its first
post-outage prompt -- the decision meant to diagnose the outage -- said "game
tick 450" while the world was at 4050, and reported a drill holding 48 coal and
`working` at the moment it was empty. This gate advances through `env.advance`
and disrupts through a *declared* `TaskSpec.disruptions` entry, then compares
the env's cached tick against the engine's own `game.tick` read over a separate
connection. A cache that agrees with the engine is the measurement; the code
path being correct is not.

**The outage has to actually stop production.** `empty_fuel` clears fuel
inventories and leaves `entity.energy` alone, so a burner keeps producing for
about 1,170 ticks afterwards -- measured, and cross-checked against 2,999,833 J
at 150 kW. The gate reads the arms evidence rather than re-deriving it, and
refuses if that evidence did not validate its own common state: R4.2 requires a
*validated* pre-intervention state, and three replayed prefixes are not the
same thing as one.

**A track is a different benchmark, so each gets its own outcome.** The gate
reports `supplied_knowledge` and `assistance` per track in
`r3-agent-baseline.json`'s shape, and never aggregates the three into one
number -- a repair result and a diagnosis result are not comparable, and a
single headline would invite exactly that comparison.

Run: uv run python -m factoriorl.gate_r4 [--episodes N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from factoriorl import assistance as assistance_module
from factoriorl import recovery as recovery_module
from factoriorl.engine_config import resolve_game_speed
from factoriorl.env import FactorioEnv
from factoriorl.paths import evidence_dir
from factoriorl.rcon import RCONClient
from factoriorl.seeding import Branch, SeedPlan
from factoriorl.session import WorkerSession
from factoriorl.tasks import get
from factoriorl.tasks.reference import solve
from factoriorl.worker import WorkerManager

GATE_MASTER = 20260909

#: One task per declared track. `repair` is the family whose fault tile *is*
#: published, kept in the gate precisely so the contrast with `diagnosis` is
#: measured rather than described.
TRACKS = (
    ("repair", "repair_belt"),
    ("diagnosis", "diagnose_line"),
    ("persistent_operation", "keep_line_running"),
)

#: The task whose declared disruption the freshness clause uses.
DISRUPTED_TASK = "keep_line_running"

ARMS_EVIDENCE = "r4-recovery-arms.json"
DECAY_EVIDENCE = "r4-burner-decay.json"


def check(report: dict, label: str, ok: bool, **detail) -> bool:
    report["checks"].append({"check": label, "ok": bool(ok), **detail})
    print(f"  [{'ok ' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""), flush=True)
    return bool(ok)


def engine_tick(endpoint: str) -> int:
    """The engine's own tick, over a separate connection.

    Separate on purpose: asking the same session that populated the cache
    whether the cache is current would prove nothing.
    """
    with RCONClient(endpoint, timeout=30.0) as client:
        return int(str(client.lua("return game.tick")).strip())


# ------------------------------------------------------- R4.1, on the engine


def clause_the_cache_is_never_stale(report: dict, session, endpoint: str) -> dict:
    """Advance, disrupt, and check the env's view against the engine's."""
    task = get(DISRUPTED_TASK)
    env = FactorioEnv(
        task,
        session,
        SeedPlan(master=GATE_MASTER, run_id="gate-r4"),
        branch=Branch.TRAIN,
        split="train",
    )
    env._episode_index = 0
    env.reset()

    at_reset = int(env._observation.get("tick") or 0)
    check(
        report,
        "the env's tick matches the engine's at reset",
        at_reset == engine_tick(endpoint),
        env=at_reset,
        engine=engine_tick(endpoint),
    )

    # Past the declared outage, in `decision_ticks` chunks through the typed
    # path, which is what keeps the production window *sampled* rather than
    # jumped -- a two-sample window spanning 3,600 ticks satisfied
    # `SUSTAINED_OUTPUT`'s span test before R4.2 added a density test.
    disruption = task.spec.disruptions[0]
    env.advance(disruption.at_tick + 2 * env.spec_.decision_ticks)
    after = int(env._observation.get("tick") or 0)
    engine = engine_tick(endpoint)
    check(
        report,
        "the env's tick matches the engine's after advancing past the outage",
        after == engine,
        env=after,
        engine=engine,
        advanced_through="env.advance",
    )

    window = env._truth.get("window") or []
    ticks = [row[0] for row in window]
    widest = max((b - a for a, b in zip(ticks[:-1], ticks[1:], strict=True)), default=0)
    check(
        report,
        "the production window is sampled, not jumped",
        bool(ticks) and widest <= env.spec_.decision_ticks,
        samples=len(ticks),
        widest_gap=widest,
        decision_ticks=env.spec_.decision_ticks,
    )

    applied = list(env._disruptions_applied)
    check(
        report,
        "the declared disruption fired, and the engine says what it touched",
        len(applied) == 1 and applied[0]["ok"] and bool(applied[0]["touched"]),
        applied=applied,
    )
    check(
        report,
        "it fired at or after its declared tick, never before",
        bool(applied) and applied[0]["applied_at_tick"] >= disruption.at_tick,
        declared=disruption.at_tick,
        applied_at=applied[0]["applied_at_tick"] if applied else None,
    )

    # The fresh-observation claim, at the one moment the old driver got wrong:
    # the first observation after the world was written to from outside.
    fuel_after = {
        record.get("name"): record.get("fuel")
        for record in (env._observation.get("entities") or [])
        if record.get("name") in ("burner-mining-drill", "stone-furnace")
    }
    check(
        report,
        "the first post-outage observation shows the emptied fuel",
        bool(fuel_after) and all(not value for value in fuel_after.values()),
        fuel=fuel_after,
        note="`resync` ran inside the refresh path; the driver that fired raw "
        "Lua reported 48 coal here",
    )

    # And nothing announced it.
    check(
        report,
        "the outage is unannounced: no marker, no event",
        not (env._observation.get("goal") or {}),
        goal=env._observation.get("goal") or {},
        published_markers=list(task.spec.public_markers),
    )
    return {
        "at_reset": at_reset,
        "after_advance": after,
        "disruptions_applied": applied,
        "resyncs": list(env._resyncs),
    }


# ------------------------------------------------------- R4.2, from evidence


def clause_the_outage_was_real_and_controlled(report: dict) -> dict:
    """Read the arms and the decay curve, and refuse a confounded comparison.

    Deliberately not re-derived here. The arms need five fresh workers -- one
    per run, because the `craft-item iron-plate` research trigger's counter is
    cumulative on the force and unresettable, so sequential episodes on one
    worker cannot reach the same technology state -- and a gate that re-ran
    them would be a second implementation of the measurement.
    """
    body: dict = {}
    for name in (ARMS_EVIDENCE, DECAY_EVIDENCE):
        path = evidence_dir() / name
        if not path.is_file():
            check(report, f"{name} exists", False, hint="run tools/recovery_arms.py")
            return body
        body[name] = json.loads(path.read_text(encoding="utf-8"))

    arms = body[ARMS_EVIDENCE]
    decay = body[DECAY_EVIDENCE]

    check(
        report,
        "the arms started from a validated common state",
        bool((arms.get("common_state") or {}).get("validated")),
        digest_lines=(arms.get("common_state") or {}).get("digest_lines"),
        mismatched=(arms.get("common_state") or {}).get("mismatched_arms"),
    )
    # The criteria have to be the module's, not numbers chosen after the fact.
    declared = recovery_module.Criteria().to_dict()
    check(
        report,
        "the arms were judged by the declared criteria, unmodified",
        (arms.get("criteria") or {}) == declared,
        declared=declared,
    )
    comparison = arms.get("comparison") or {}
    outage = comparison.get("outage") or {}
    check(
        report,
        "the control arm suffered a confirmed outage, so the disruption can stop the line",
        outage.get("no_action") == "confirmed",
        control_outage=outage.get("no_action"),
    )
    check(
        report,
        "the agent arm was run more than once",
        int(comparison.get("agent_runs") or 0) > 1,
        runs=comparison.get("agent_runs"),
        note="one run is not a measurement: `give_coal_N` fuels the nearest "
        "entity and two runs of the same model differed by 39 plates",
    )
    # The number the withdrawn claim needed and did not have.
    found = decay.get("findings") or {}
    check(
        report,
        "the run-on after a fuel clear is measured, and agrees with the arithmetic",
        bool(found.get("measured_agrees_with_arithmetic")),
        ticks_producing_after_removal=found.get("ticks_producing_after_removal"),
        implied=found.get("implied_run_on_ticks"),
    )
    check(
        report,
        "no arm is reported as a recovery it did not demonstrate",
        arms.get("outcome") in ("recovery_demonstrated", "loss_averted", "agent_failed"),
        outcome=arms.get("outcome"),
        per_run=arms.get("per_agent_run_outcome"),
    )
    return {
        "outcome": arms.get("outcome"),
        "comparison": comparison,
        "run_on_ticks": found.get("ticks_producing_after_removal"),
    }


# -------------------------------------------------------------------- R4.3


def clause_each_track_is_declared(report: dict) -> dict:
    """Per-track hints and assistance, engine-free, in the baseline's shape."""
    tracks: dict = {}
    for track, task_id in TRACKS:
        spec = get(task_id).spec
        published = set(spec.public_markers)
        withheld = set(spec.fault_markers) - published
        tracks[track] = {
            "task": {"id": spec.id, "version": spec.version, "track": spec.track},
            "supplied_knowledge": {
                "task_description": spec.description,
                "published_markers": sorted(published),
                "withheld_fault_markers": sorted(withheld),
            },
            "assistance": assistance_module.describe_assistance(spec),
            "declared_disruptions": [d.to_dict() for d in spec.disruptions],
        }
        check(report, f"{track}: the task declares this track", spec.track == track, task=task_id)

    check(
        report,
        "the repair track declares that it publishes the fault's location",
        "fault-location:published" in tracks["repair"]["assistance"],
        assistance=tracks["repair"]["assistance"],
    )
    check(
        report,
        "the diagnosis track publishes no marker at all",
        not tracks["diagnosis"]["supplied_knowledge"]["published_markers"],
        published=tracks["diagnosis"]["supplied_knowledge"]["published_markers"],
    )
    check(
        report,
        "and no shaping component points at what it withheld",
        not [
            reward.name
            for reward in get("diagnose_line").spec.rewards
            if reward.shaping
            and getattr(reward.predicate, "marker", None)
            in set(get("diagnose_line").spec.fault_markers)
        ],
        note="rewards._measure's CHARACTER_WITHIN branch reads truth, so such a "
        "component would hand back through the reward what the observation withholds",
    )
    check(
        report,
        "the persistent-operation track declares its disruption on the task",
        len(tracks["persistent_operation"]["declared_disruptions"]) == 1,
        disruptions=tracks["persistent_operation"]["declared_disruptions"],
    )
    return tracks


def clause_each_track_is_reachable(report: dict, session, episodes: int) -> dict:
    """The reference reaches each track's objective on the held-out split.

    Solvability, not skill. Each reference's disclosed assistance is recorded
    beside its rate, because `solve_diagnose_line` deliberately does *not*
    diagnose -- it applies every fix -- and `solve_keep_line_running`
    deliberately does *not* read the declared outage tick. A reference that
    used privileged knowledge the agent lacks would be demonstrating a
    different task's solvability.
    """
    outcomes: dict = {}
    for track, task_id in TRACKS:
        solved = 0
        traces = []
        for index in range(episodes):
            env = FactorioEnv(
                get(task_id),
                session,
                SeedPlan(master=GATE_MASTER, run_id=f"gate-r4-{track}"),
                branch=Branch.TRAIN,
                split="test",
            )
            env._episode_index = index
            env.reset()
            trace = solve(env)
            solved += 1 if trace.succeeded else 0
            traces.append(
                {
                    "episode": index,
                    "succeeded": bool(trace.succeeded),
                    "stuck_reason": trace.stuck_reason,
                    "final_tick": int(env._observation.get("tick") or 0),
                    "produced": (env._truth.get("produced") or {}).get("iron-plate"),
                    "disruptions_applied": list(env._disruptions_applied),
                }
            )
        outcomes[track] = {
            "task": task_id,
            "split": "test",
            "episodes": episodes,
            "reference_solved": solved,
            "reference_rate": round(solved / episodes, 3) if episodes else None,
            "reference_assistance": REFERENCE_ASSISTANCE[track],
            "traces": traces,
        }
        check(
            report,
            f"{track}: the reference reaches the objective on the held-out split",
            solved == episodes,
            solved=f"{solved}/{episodes}",
        )
    return outcomes


#: What each reference is given that the agent is not. Declared, because "the
#: reference solves it" means nothing without it.
REFERENCE_ASSISTANCE = {
    "repair": "marker positions from truth, including the published fault tile",
    "diagnosis": (
        "marker positions from truth. It does *not* read which machine is "
        "starved: it applies every fix, so its success is not evidence that the "
        "diagnosis is easy"
    ),
    "persistent_operation": (
        "marker positions from truth. It does *not* read the declared outage "
        "tick, which the agent is never told; it refuels on a cycle instead"
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episodes", type=int, default=3, help="episodes per track")
    args = parser.parse_args()

    report: dict = {
        "gate": "R4 fresh observations, a measured outage, three declared tracks",
        "clauses": [
            "R4.1: every path that advances time or writes to the world leaves the "
            "env's cached observation current",
            "R4.2: the disruption's production effect is measured against criteria "
            "declared beforehand, from a validated common state, with a control arm",
            "R4.3: three declared tracks, each with its own hints, assistance and outcome",
            "each track's objective is reachable on its held-out split",
        ],
        "checks": [],
    }

    print("three declared tracks (engine-free):", flush=True)
    report["tracks"] = clause_each_track_is_declared(report)

    print("\nthe outage was real, and measured against a control:", flush=True)
    report["recovery"] = clause_the_outage_was_real_and_controlled(report)

    manager = WorkerManager()
    handle = manager.launch("gate-r4")
    started = time.perf_counter()
    session = None
    try:
        endpoint = handle.spec.rcon_endpoint
        with RCONClient(endpoint, timeout=30.0) as client:
            client.lua(f"game.speed = {resolve_game_speed()} return game.speed")
        session = WorkerSession(handle, timeout=60.0)
        session.status()

        print("\nthe env's view of the world, against the engine's:", flush=True)
        report["freshness"] = clause_the_cache_is_never_stale(report, session, endpoint)

        print("\neach track's objective, reached by its reference:", flush=True)
        report["reachable"] = clause_each_track_is_reachable(report, session, args.episodes)
    finally:
        if session is not None:
            try:
                session.close()
            except OSError:
                pass
        manager.cleanup(handle)

    report["elapsed_seconds"] = round(time.perf_counter() - started, 1)
    report["passed"] = all(c["ok"] for c in report["checks"])
    report["failed"] = [c["check"] for c in report["checks"] if not c["ok"]]
    # Deliberately no aggregate rate across tracks: a repair result and a
    # diagnosis result are not comparable, and one headline would invite the
    # comparison this split exists to prevent.
    report["per_track_outcome"] = {
        track: {
            "assistance": report["tracks"][track]["assistance"],
            "reference_rate": (report.get("reachable") or {}).get(track, {}).get("reference_rate"),
        }
        for track, _task in TRACKS
    }
    path = evidence_dir() / "r4-tracks.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n{'PASSED' if report['passed'] else 'FAILED'}: wrote {path}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
