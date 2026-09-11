"""Wiring one agent run to a real worker (PLAN.md 5.3).

Kept apart from :mod:`factoriorl.agent.loop` for one reason: the loop must be
runnable without an engine, because every test of it runs offline and CI has no
Factorio. This module is the only place in the package that launches a worker,
so importing the loop, the adapters, the summary or the parser never drags in
worker management.

Nothing here is exercised by the unit tests -- it needs an engine, and PLAN
section 4 is explicit that engine-dependent work is not done until it has run
against a real engine. It is written to match ``learn/train.py``'s setup exactly
so the two produce comparable provenance, and it is marked here as awaiting an
engine run rather than presented as verified.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from factoriorl import assistance as assistance_module
from factoriorl.agent.adapters import ModelAdapter
from factoriorl.agent.loop import AgentConfig, AgentLoop
from factoriorl.agent.narrate import Narrator
from factoriorl.agent.sampler import Sampler
from factoriorl.engine_config import resolve_game_speed


#: The paused server loop rate. Same value ``learn/train.py`` uses, for the same
#: reason: RCON round-trip latency is bounded by the server tick, so a run at
#: the default speed pays 16.6 ms per round trip instead of 1.5 ms.
#:
#: Resolved per call rather than at import, which is how it used to be: a module
#: constant is read once when the process starts, so a caller that wanted a
#: different speed -- a real-time run, say -- silently got whatever the first
#: import had decided.
def default_speed() -> float:
    return resolve_game_speed()


#: Real time. One second of wall clock is one second of game time, so a decision
#: the model spends five seconds on costs the world five seconds. Every measured
#: benchmark run uses `default_speed()` instead, which is far faster and
#: unwatchable.
REALTIME_SPEED = 1.0


def run_task(
    config: AgentConfig,
    adapter: ModelAdapter,
    *,
    master_seed: int = 20260907,
    game_speed: float | None = None,
    free_running: bool = False,
    on_ready=None,
    until=None,
) -> dict[str, Any]:
    """Launch a worker, play ``config.episodes`` episodes, return the result.

    Imports are local so that this module can exist in a package whose other
    modules must stay importable without the ``rl`` extra installed.
    """
    from factoriorl import manifest as manifest_module
    from factoriorl.env import FactorioEnv
    from factoriorl.rcon import RCONClient
    from factoriorl.seeding import Branch, SeedPlan, seed_everything
    from factoriorl.session import WorkerSession
    from factoriorl.tasks import get
    from factoriorl.worker import WorkerManager

    run_id = manifest_module.new_run_id(config.run_prefix)
    seeded = seed_everything(master_seed)
    plan = SeedPlan(master=master_seed, run_id=run_id)
    task = get(config.task_id)

    manager = WorkerManager()
    # Run-scoped worker id, for the reason `learn/train.py` gives: a fixed name
    # collides the moment two runs of the same task overlap, and the second dies
    # on the first one's write-data lock.
    handle = manager.launch(f"agent-{config.task_id}-{run_id[-8:]}")
    session = None
    try:
        speed = default_speed() if game_speed is None else float(game_speed)
        with RCONClient(handle.spec.rcon_endpoint, timeout=30.0) as client:
            client.lua(f"game.speed = {speed} return game.speed")
        session = WorkerSession(handle, timeout=30.0)
        session.status()
        if free_running:
            # The world is no longer paused between decisions, so an action
            # lands at whatever tick it arrives at. That makes the run a
            # demonstration rather than a measurement, and it is recorded as one
            # in the provenance below rather than left to be inferred.
            session.configure(free_running=True)
        branch = Branch.TRAIN if config.split == "train" else Branch.EVAL
        env = FactorioEnv(
            task,
            session,
            plan,
            branch=branch,
            split=config.split,
            shaping=config.shaping,
        )
        if config.skills:
            # Appends indices after the primitive catalog and leaves primitive
            # indices where they are, which is the contract `action_vocabulary`
            # relies on to describe a wrapper it knows nothing about.
            from factoriorl.skills import SkillEnv

            env = SkillEnv(env)
        loop = AgentLoop(
            env,
            adapter,
            config,
            run_id=run_id,
            provenance={
                "engine": handle.engine.to_dict(),
                "workers": [handle.spec.manifest()],
                "seeds": {**plan.to_dict(), "seeded": seeded},
                # Every assistance this run received, composed. `wait_batch`
                # repeats a `wait` the model just chose without asking again,
                # which is a real help on a task whose measurement window is
                # 200 decisions of waiting -- so it is named here rather than
                # left to be inferred from the config.
                "assistance": assistance_module.describe_assistance(
                    task.spec,
                    extra=((f"wait-batch:{config.wait_batch}",) if config.wait_batch else ()),
                ),
                "clock": {
                    "game_speed": speed,
                    "free_running": free_running,
                    "measurement": (
                        "demonstration: the world runs while the model thinks, so "
                        "an action lands at whatever tick it arrives at"
                        if free_running
                        else "exact stepping"
                    ),
                },
            },
        )
        # The run clock starts here and not before: A0.3 puts it at "the first
        # gameplay observation after engine readiness", so worker launch, mod
        # packaging and save creation do not eat the agent's time. `loop.run()`
        # takes the first observation on its next line.
        if on_ready is not None:
            on_ready()
        return loop.run(until=until)
    finally:
        if session is not None:
            try:
                session.close()
            except OSError:
                pass
        manager.cleanup(handle)


def finalize(session, checkpointer, *, observation: dict | None = None) -> dict:
    """Bring the world to a stop and take the last snapshot of it.

    Roadmap A4.3: "At the deadline stop new actions, cancel controller-held
    activity, pause the world for the final snapshot/save, then stop the
    worker. The paused finalization period is outside gameplay and recorded
    separately."

    None of it happened before. In particular the world was still *running*
    while `game.server_save` was issued, which in a realtime run means the save
    holds a slightly different world from the last observation the agent saw --
    and a run report presents the two side by side.

    Every step is recorded, including the ones that failed, because a
    finalization that half-worked is the case a reader most needs to know
    about. A step that raises must not cost the run its final save, which is
    the whole point of finalizing.
    """
    started = time.perf_counter()
    record: dict = {
        "measures": "wall time after the last decision; outside gameplay",
        "steps": [],
    }

    def step(name: str, action) -> None:
        at = time.perf_counter()
        entry: dict = {"step": name}
        try:
            entry["result"] = action()
        except Exception as failure:  # noqa: BLE001 - recorded, never fatal
            entry["error"] = f"{type(failure).__name__}: {failure}"
        entry["seconds"] = round(time.perf_counter() - at, 3)
        record["steps"].append(entry)

    # 1. Cancel what the controller still has running. A walk or a mine left in
    #    flight would keep moving the character while the save is taken.
    inflight = [
        str(entry["request_id"])
        for entry in (observation or {}).get("inflight") or []
        if isinstance(entry, dict) and entry.get("request_id")
    ]
    record["cancelled"] = inflight
    for request_id in inflight:
        step(
            f"cancel:{request_id}",
            lambda rid=request_id: session.act("cancel", target_request_id=rid).response.result,
        )

    # 2. Pause. `free_running` is the one knob that changes what happens rather
    #    than how fast it happens, and turning it off is what makes the world
    #    stand still for the snapshot.
    step("pause", lambda: session.configure(free_running=False).response.result)

    # 3. The snapshot, of a world that is now not moving -- with the tick read
    #    either side of it. "The world was paused for the save" is otherwise a
    #    claim about a knob rather than a statement about the world, and this is
    #    the difference between the two: if the pause did not take, these two
    #    numbers differ and the artifact says so.
    def _tick() -> int | None:
        try:
            return int((session.truth().response.result or {}).get("tick") or 0)
        except Exception:  # noqa: BLE001 - a missing reading is not a failed save
            return None

    record["tick_before_save"] = _tick()
    step("save", lambda: checkpointer.save("final"))
    record["tick_after_save"] = _tick()
    # One tick is the floor, not a tolerance chosen to make this pass.
    # `game.tick_paused = true` set from inside a tick lets that tick finish, so
    # the world advances by exactly one and then stops. Measured either side of
    # the save: 59 ticks before `configure(free_running=False)` was made to
    # pause immediately, 1 after.
    drift = None
    if record["tick_before_save"] is not None and record["tick_after_save"] is not None:
        drift = record["tick_after_save"] - record["tick_before_save"]
    record["ticks_across_the_save"] = drift
    record["world_was_still_for_the_save"] = drift is not None and drift <= 1

    record["seconds"] = round(time.perf_counter() - started, 3)
    record["failed_steps"] = [entry["step"] for entry in record["steps"] if "error" in entry]
    return record


def run_world(
    mode,
    config: AgentConfig,
    adapter: ModelAdapter,
    *,
    master_seed: int = 20260910,
    game_speed: float | None = None,
    free_running: bool = True,
    on_ready=None,
    on_finished=None,
    clock_reader=None,
    until=None,
    checkpoint_seconds: float | None = None,
    resume_from: Path | None = None,
    launch_client: bool = False,
    client_warmup: float = 30.0,
    hold_open: float = 0.0,
    narrate: bool = True,
    overlay: bool = True,
) -> dict[str, Any]:
    """Play an open generated world -- `factoriorl.worlds` -- rather than a task.

    Structurally the same as :func:`run_task` and deliberately not folded into
    it. The two differ at every step that matters: the worker is created on the
    natural terrain surface rather than the benchmark one, the environment
    initialises the map instead of painting a scene over it, and the run is not
    scored. Sharing one function would mean four branches through the only code
    path that launches an engine.

    ``free_running`` defaults to True here and False there, which is the honest
    default for each: a benchmark episode is a measurement and needs the world
    paused between decisions; an open world is played in real time and a paused
    world would misrepresent what the agent was doing with its thirty minutes.
    """
    from factoriorl import knowledge as knowledge_module
    from factoriorl import manifest as manifest_module
    from factoriorl.agent.checkpoint import DEFAULT_INTERVAL_SECONDS, Checkpointer
    from factoriorl.agent.viewer import Viewer
    from factoriorl.freeplay import starting_inventory
    from factoriorl.open_world import OpenWorldEnv
    from factoriorl.rcon import RCONClient
    from factoriorl.seeding import seed_everything
    from factoriorl.session import WorkerSession
    from factoriorl.worker import WorkerManager

    run_id = manifest_module.new_run_id(config.run_prefix)
    seeded = seed_everything(master_seed)

    manager = WorkerManager()
    # Read before the worker is launched: an unreadable freeplay definition
    # should fail in a second, not after a ninety-second map generation.
    freeplay = starting_inventory(manager.engine.executable)
    viewer = None
    sampler = None
    handle = manager.launch(
        f"world-{mode.id}-{run_id[-8:]}",
        map_seed=master_seed,
        terrain=mode.terrain,
        # `terrain` is ignored when a save is given -- the save carries the
        # world it was generated with -- but it is still passed so the worker
        # manifest records the surface this world is on rather than
        # defaulting to the benchmark one and quietly saying the wrong thing.
        save_source=resume_from,
    )
    session = None
    try:
        speed = REALTIME_SPEED if game_speed is None else float(game_speed)
        with RCONClient(handle.spec.rcon_endpoint, timeout=30.0) as client:
            client.lua(f"game.speed = {speed} return game.speed")
        session = WorkerSession(handle, timeout=30.0)
        session.status()
        if free_running:
            session.configure(free_running=True)
        # A resume leaves the character, its inventory and the force's research
        # exactly as the save holds them -- and, since the fresh-gate fix, leaves
        # the factory standing.
        resuming = resume_from is not None
        env = OpenWorldEnv(
            mode,
            session,
            inventory=freeplay["items"],
            seed=master_seed,
            fresh=not resuming,
        )
        # Started before the first decision, because a join stalls the server
        # while it transfers the map and a stall *inside* a step is an
        # infrastructure failure rather than a task outcome.
        viewer = Viewer.start(
            requested=launch_client,
            executable=handle.engine.executable,
            address=f"127.0.0.1:{handle.spec.ports.game}",
            mod_directory=handle.spec.mod_directory,
            warmup_seconds=client_warmup,
        )
        viewer.wait_for_join()
        checkpointer = Checkpointer(
            session=session,
            write_data=handle.spec.write_data,
            run_dir=manifest_module.runs_dir() / run_id,
            interval_seconds=(
                DEFAULT_INTERVAL_SECONDS if checkpoint_seconds is None else checkpoint_seconds
            ),
        )
        # Fetched once, after the world exists and before the first decision.
        # Cached on disk by engine build and mod digest, so a second run on the
        # same machine pays nothing for it. A failure here is recorded and the
        # run continues: an agent without the recipe book is worse off, not
        # stopped, and losing a thirty-minute run to a missing lookup table
        # would be the wrong trade.
        knowledge, knowledge_error = {}, None
        try:
            knowledge = knowledge_module.cached(
                session,
                engine_build=str(handle.engine.build),
                mod_digest=manifest_module.mod_source_digest(),
            )
        except Exception as failure:  # noqa: BLE001 - recorded, never fatal
            knowledge_error = f"{type(failure).__name__}: {failure}"
        rendered_knowledge = knowledge_module.render(knowledge) if knowledge else ""

        # Built before the loop so the first decision is narrated too, and
        # given the session rather than the env: the overlay is drawn with
        # `rendering`, which the sensor cannot see, so nothing here reaches the
        # agent's observation. See `factoriorl.agent.narrate`.
        narrator = (
            Narrator(session, console=narrate, overlay=overlay) if (narrate or overlay) else None
        )
        loop = AgentLoop(
            env,
            adapter,
            config,
            run_id=run_id,
            narrator=narrator,
            static_knowledge=rendered_knowledge,
            provenance={
                "engine": handle.engine.to_dict(),
                "workers": [handle.spec.manifest()],
                "seeds": {"master": master_seed, "seeded": seeded, "map_seed": master_seed},
                # Composed, not asserted. An open world is given navigation
                # and bounded sequences, and a run that received help while
                # recording `none` is exactly what `factoriorl.assistance`
                # exists to stop.
                "assistance": (
                    "+".join(mode.assistance) if mode.assistance else assistance_module.STATIC
                ),
                "world": mode.to_dict(),
                "starting_inventory": None if resuming else freeplay,
                "knowledge": {
                    "counts": (knowledge or {}).get("counts"),
                    "rendered_chars": len(rendered_knowledge),
                    "error": knowledge_error,
                    "placement": (
                        "the transcript's static prefix, so it is sent once and "
                        "served from the provider's cache thereafter"
                    ),
                },
                "resumed_from": str(resume_from) if resuming else None,
                "viewer": viewer.to_dict(),
                "clock": {
                    "game_speed": speed,
                    "free_running": free_running,
                    "measurement": (
                        "demonstration: an open world is played in real time and "
                        "is not scored; no rate measured here is comparable to a "
                        "benchmark task"
                    ),
                },
            },
        )
        if clock_reader is not None:
            # So the prompt can state how much run time and allowance are left.
            # "decision 503 of 100000" is not planning guidance for a run about
            # to hit a thirty-minute wall.
            loop.clock_reader = clock_reader
        if on_ready is not None:
            on_ready()
        # Before the first action, per A1.3: a world that dies on decision one
        # should still leave something to resume from.
        checkpointer.save("initial")
        # Started after the initial save so the first sample describes a world
        # that has been snapshotted, and stopped before finalization so the
        # samples cover gameplay and nothing else.
        sampler = Sampler(
            session=session,
            destination=manifest_module.runs_dir() / run_id / "production.jsonl",
            metrics=env.metrics,
        ).start()
        result = loop.run(until=until, on_decision=checkpointer.maybe_save)
        # Gameplay is over here, and the clock has to say so *here*: everything
        # below is finalization, and `RunClock.to_dict()` already claimed to
        # exclude the final snapshot while `clock.stop()` ran after it.
        if on_finished is not None:
            on_finished()
        result["sampling"] = sampler.stop().to_dict()
        sampler = None  # stopped here, so the `finally` has nothing left to do
        result["finalization"] = finalize(session, checkpointer, observation=env._observation)
        result["checkpoints"] = checkpointer.to_dict()
        result["viewer"] = viewer.to_dict(
            reordered_replies=getattr(session._client, "reordered_replies", None)
        )
        return result
    finally:
        # A run that raised still has to leave the sampler stopped: it is a
        # daemon thread, so it would otherwise keep reading a session the
        # `finally` below is about to close.
        if sampler is not None:
            sampler.stop()
        # After the final checkpoint, so the last thing on screen is the world
        # that was actually saved.
        if viewer is not None:
            viewer.stop(hold_open_seconds=hold_open)
        if session is not None:
            try:
                session.close()
            except OSError:
                pass
        # Every verified checkpoint has already been copied into the run
        # directory by now, which matters: `cleanup` removes the worker
        # directory wholesale and the engine writes its saves inside it.
        manager.cleanup(handle)
