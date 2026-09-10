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

from pathlib import Path
from typing import Any

from factoriorl import assistance as assistance_module
from factoriorl.agent.adapters import ModelAdapter
from factoriorl.agent.loop import AgentConfig, AgentLoop
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


def run_world(
    mode,
    config: AgentConfig,
    adapter: ModelAdapter,
    *,
    master_seed: int = 20260910,
    game_speed: float | None = None,
    free_running: bool = True,
    on_ready=None,
    until=None,
    checkpoint_seconds: float | None = None,
    resume_from: Path | None = None,
    launch_client: bool = False,
    client_warmup: float = 30.0,
    hold_open: float = 0.0,
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
        loop = AgentLoop(
            env,
            adapter,
            config,
            run_id=run_id,
            provenance={
                "engine": handle.engine.to_dict(),
                "workers": [handle.spec.manifest()],
                "seeds": {"master": master_seed, "seeded": seeded, "map_seed": master_seed},
                "assistance": assistance_module.STATIC,
                "world": mode.to_dict(),
                "starting_inventory": None if resuming else freeplay,
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
        if on_ready is not None:
            on_ready()
        # Before the first action, per A1.3: a world that dies on decision one
        # should still leave something to resume from.
        checkpointer.save("initial")
        result = loop.run(until=until, on_decision=checkpointer.maybe_save)
        checkpointer.save("final")
        result["checkpoints"] = checkpointer.to_dict()
        result["viewer"] = viewer.to_dict(
            reordered_replies=getattr(session._client, "reordered_replies", None)
        )
        return result
    finally:
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
