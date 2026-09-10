"""Gate A3, against the engine: does persistent agency reach a real prompt?

The offline fixtures in `tests/unit/test_persistent_agency.py` prove the plan,
the notes and the repeated-failure block behave correctly. They prove it against
a stub environment, which cannot disagree with the engine about what an
observation looks like -- and every A3 mechanism ends in *rendered text a model
reads*, assembled from a live observation.

So this drives the real `AgentLoop` over a real generated world with a scripted
adapter standing in for the provider. No key, no provider, no cost. What it
checks is exactly what a stub cannot:

  - the objective reaches the transcript's **static prefix**, not the per-turn
    body, which is where it has to be for the cache to serve it;
  - a plan stated on turn 1 is still in front of the model on turn 3;
  - a note renders under the unverified heading, never as an observation;
  - three identical refusals against the real refusal machinery -- not a stub's
    idea of a failure -- produce the stall block.

Run:
  uv run python tools/probe_agency.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from factoriorl import worlds  # noqa: E402
from factoriorl.agent.adapters import ScriptedAdapter  # noqa: E402
from factoriorl.agent.loop import AgentConfig, AgentLoop  # noqa: E402
from factoriorl.agent.memory import STALL_THRESHOLD  # noqa: E402
from factoriorl.freeplay import starting_inventory  # noqa: E402
from factoriorl.open_world import OpenWorldEnv  # noqa: E402
from factoriorl.session import WorkerSession  # noqa: E402
from factoriorl.worker import WorkerManager  # noqa: E402

#: A placement far outside build range. Refused by the mod every time, with the
#: same reason every time, which is what makes it a *repeated* failure rather
#: than a sequence of different ones.
UNREACHABLE = [9000.5, 9000.5]


def script(env: OpenWorldEnv) -> list[str]:
    """What the stand-in provider says, turn by turn."""
    place = env.catalog.keys().index("place_at")
    wait = env.catalog.keys().index("wait_for")
    stall = json.dumps(
        {
            "action": place,
            "arguments": {
                "item": "stone-furnace",
                "position": UNREACHABLE,
                "direction": "north",
            },
            "reason": "trying the same impossible placement again",
        }
    )
    return [
        json.dumps(
            {
                "action": wait,
                "arguments": {"until": "nothing_in_flight", "seconds": 10},
                "reason": "look around first",
                "plan": "smelt iron, then build a second furnace on the coal",
                "note": "the crash site is west and there is ore to the east",
            }
        ),
        # Deliberately silent about both from here on: the point is that they
        # persist without being restated.
        *[stall] * (STALL_THRESHOLD + 1),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--out", default=str(ROOT / "docs" / "evidence" / "a3-agency.json"))
    args = parser.parse_args()

    mode = worlds.get("open_factory")
    manager = WorkerManager()
    freeplay = starting_inventory(manager.engine.executable)
    report: dict = {
        "measures": "roadmap A3's gate, rendered against a live observation",
        "world": mode.to_dict(),
        "engine": manager.engine.to_dict(),
        "provider": "scripted -- no key, no call, no cost",
    }
    started = time.perf_counter()

    handle = manager.launch("probe-a3", map_seed=args.seed, terrain=mode.terrain)
    try:
        session = WorkerSession(handle, timeout=60.0)
        session.status()
        session.configure(free_running=True)
        env = OpenWorldEnv(mode, session, inventory=freeplay["items"], seed=args.seed)
        env.reset()

        adapter = ScriptedAdapter(script(env))
        loop = AgentLoop(
            env,
            adapter,
            AgentConfig(task_id=mode.id),
            run_id="probe-a3",
            run_dir=Path(handle.spec.directory) / "probe-a3",
        )
        loop.run_dir.mkdir(parents=True, exist_ok=True)
        loop.run_segment(0, fresh_memory=True, budget=len(adapter.script))

        prefix = loop.transcript.static_prefix
        first, last = loop.decisions[0], loop.decisions[-1]
        # What the model was actually shown on the final turn: the rendered
        # observation plus the memory block that is appended to it.
        final_prompt = last.summary.render() + "\n" + loop.memory.render()

        report["decisions"] = len(loop.decisions)
        report["resolutions"] = [d.resolution for d in loop.decisions]
        report["attempt_failures"] = [
            [
                (a.outcome.failure.value, a.outcome.detail)
                for a in d.attempts
                if a.outcome is not None and hasattr(a.outcome, "failure")
            ]
            for d in loop.decisions
        ]
        report["memory_render"] = loop.memory.render()
        report["memory_attempts"] = len(loop.memory.attempts)
        report["objective_in_static_prefix"] = "YOUR OBJECTIVE" in prefix
        report["objective_absent_from_turn"] = "YOUR OBJECTIVE" not in final_prompt
        report["static_prefix_chars"] = len(prefix)
        report["plan_recorded"] = first.plan
        report["note_recorded"] = first.note
        # Read out of the transcript, which holds the turns exactly as they
        # were sent. Rewinding `loop.memory` after the fact cannot reconstruct
        # turn two: the memory object was mutated in place, and by the end the
        # plan has correctly been closed as stalled. Asking the artifact what
        # was sent is the only version of this question that has an answer.
        user_turns = [t.content for t in loop.transcript.turns if t.role == "user"]
        report["user_turns"] = len(user_turns)
        report["plan_under_current_plan_on_turn_two"] = any(
            "CURRENT PLAN\n  smelt iron, then build a second furnace on the coal" in turn
            for turn in user_turns
        )
        report["note_survived_without_restatement"] = sum(
            "the crash site is west" in turn for turn in user_turns
        )
        report["plan_still_present"] = "smelt iron, then build a second furnace" in final_prompt
        report["prompt_chars"] = len(final_prompt)
        report["memory_chars"] = len(loop.memory.render())
        report["note_under_unverified_heading"] = (
            "THE AGENT'S OWN NOTES (unverified -- you wrote these)" in final_prompt
            and "the crash site is west" in final_prompt
        )
        report["stall_reported"] = "REPEATED FAILURE -- this is not working" in final_prompt
        report["stall_block"] = [
            line for line in final_prompt.splitlines() if "failed" in line and "times" in line
        ]
        report["plan_closed_as_stalled"] = any(
            entry.get("outcome") == "stalled" for entry in loop.memory.plans
        )
        report["stalled_on"] = [d.stalled_on for d in loop.decisions if d.stalled_on]
        # The refusals were real refusals from the mod, not a stub's flag.
        report["refusals"] = [
            (d.result or {}).get("action_error") or (d.result or {}).get("action_status")
            for d in loop.decisions[1:]
        ]
        report["final_prompt"] = final_prompt
        session.close()
    finally:
        manager.cleanup(handle)

    report["wall_seconds"] = round(time.perf_counter() - started, 1)
    report["verdict"] = {
        "objective_is_cached_not_repeated": bool(
            report["objective_in_static_prefix"] and report["objective_absent_from_turn"]
        ),
        "plan_persists_without_restatement": bool(report["plan_still_present"]),
        "plan_was_current_in_a_turn_actually_sent": bool(
            report["plan_under_current_plan_on_turn_two"]
        ),
        "note_persisted_without_restatement": report["note_survived_without_restatement"] > 1,
        # Counted on a stable phrase. This used to name the exact size of the
        # placement domain ("121 legal values"), which then changed the moment
        # the character's own tile stopped being offered -- a probe failing
        # because the code got better is a probe testing the wrong thing.
        "refusal_is_stated_once_not_twice": (
            report["memory_render"].count("legal values for placements") == 1
        ),
        "note_is_labelled_unverified": bool(report["note_under_unverified_heading"]),
        "repeated_failure_reported": bool(report["stall_reported"]),
        "stalled_plan_closed": bool(report["plan_closed_as_stalled"]),
        "refusals_came_from_the_engine": all(bool(r) for r in report["refusals"]),
    }
    Path(args.out).write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report["verdict"], indent=2))
    print(f"wrote {args.out}")
    return 0 if all(report["verdict"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
