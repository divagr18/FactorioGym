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

from typing import Any

from factoriorl import assistance as assistance_module
from factoriorl.agent.adapters import ModelAdapter
from factoriorl.agent.loop import AgentConfig, AgentLoop
from factoriorl.engine_config import resolve_game_speed

#: The paused server loop rate. Same value ``learn/train.py`` uses, for the same
#: reason: RCON round-trip latency is bounded by the server tick, so a run at
#: the default speed pays 16.6 ms per round trip instead of 1.5 ms. Set here so
#: an agent run and a training run measure game time on the same footing.
RUN_SPEED = resolve_game_speed()


def run_task(
    config: AgentConfig,
    adapter: ModelAdapter,
    *,
    master_seed: int = 20260907,
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
        with RCONClient(handle.spec.rcon_endpoint, timeout=30.0) as client:
            client.lua(f"game.speed = {RUN_SPEED} return game.speed")
        session = WorkerSession(handle, timeout=30.0)
        session.status()
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
            },
        )
        return loop.run()
    finally:
        if session is not None:
            try:
                session.close()
            except OSError:
                pass
        manager.cleanup(handle)
