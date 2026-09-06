"""Interactive probe: exact stepping, movement, transfer, reset.

Run: uv run python tools/probe_session.py
"""

import time

from factoriorl.session import WorkerSession
from factoriorl.worker import WorkerManager


def main() -> None:
    manager = WorkerManager()
    handle = manager.launch("probe")
    try:
        with WorkerSession(handle) as session:
            status = session.status()
            print("status:", status.response.result, f"{status.round_trip_ms:.1f}ms")
            print("episode:", session.episode_id)

            # Stepping: paused world, exact advances.
            for ticks in (1, 30, 120):
                timed = session.advance(ticks)
                print(f"advance {ticks}: tick={timed.response.tick} result={timed.response.result}")

            # Inter-request idle must not advance the simulation.
            obs_a = session.observe().response.result["absolute_tick"]
            time.sleep(0.5)
            obs_b = session.observe().response.result["absolute_tick"]
            print("idle drift:", obs_b - obs_a)

            # Movement south for 30 ticks.
            move = session.act("move", direction="south", ticks=30)
            print("move:", move.response.result)
            session.advance(30)
            obs = session.observe().response.result
            print("after move+30t: pos", obs["character"]["position"], "tick", obs["tick"])

            # Transfer from src chest to character.
            tr = session.act(
                "transfer", **{"from": "src", "to": "character", "item": "iron-plate", "count": 10}
            )
            print("transfer:", tr.response.code, tr.response.result)

            # Reset and compare against the reference scene.
            reset = session.reset()
            print("reset:", reset.response.result, "new episode:", session.episode_id)
            obs2 = session.observe().response.result
            print("post-reset src:", obs2["entities"]["src"])
            print("post-reset char:", obs2["character"]["position"], obs2["inventory"])
    finally:
        manager.shutdown(handle)


if __name__ == "__main__":
    main()
