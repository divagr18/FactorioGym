"""Gate A4.3: is the run measured on a wall clock, and stopped properly?

This drives the real `run_world` -- the same entry point `factoriorl agent`
uses -- over a real generated world in **realtime**, with a scripted provider
that deliberately takes several seconds to answer. No key, no provider, no cost.

The slowness is the experiment, not an accident. In realtime the world keeps
running while the model thinks, so a sampler that follows the agent's turn
records nothing during exactly the interval that matters. A4.3 asks for samples
"approximately every five seconds, **including during model calls**", and the
only way to show that is to make the model calls long and then count the samples
that landed inside them.

What it checks:

  - samples arrive on the wall clock, and arrive *while the provider is
    thinking*;
  - the sampler shares the agent's connection without losing anything -- the
    decisions still land, and `reordered_replies` stays sane;
  - finalization cancels, pauses, then saves, in that order;
  - the world is **still** across the save: the tick either side is the same
    number, which is the difference between "we set a knob" and "the world
    stopped";
  - gameplay time and finalization time are separate numbers.

Run:
  uv run python tools/probe_measurement.py
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
from factoriorl.agent.adapters import ModelReply, ScriptedAdapter  # noqa: E402
from factoriorl.agent.budget import RunClock  # noqa: E402
from factoriorl.agent.loop import AgentConfig  # noqa: E402
from factoriorl.agent.runner import run_world  # noqa: E402

#: How long the stand-in provider pretends to think. Longer than the sampler's
#: five-second interval, so a sample *must* land inside a model call for the
#: cadence claim to hold.
THINKING_SECONDS = 7.0

#: Decisions to play. Four slow turns is around half a minute of gameplay --
#: enough for several sampling intervals and short enough to run repeatedly.
DECISIONS = 4


class SlowAdapter(ScriptedAdapter):
    """A scripted adapter that takes its time, so the sampler has a gap to fill."""

    def __init__(self, script, *, seconds: float = THINKING_SECONDS) -> None:
        super().__init__(script)
        self.seconds = seconds
        self.thinking: list[tuple[float, float]] = []

    def complete(self, request) -> ModelReply:
        started = time.time()
        time.sleep(self.seconds)
        self.thinking.append((started, time.time()))
        return super().complete(request)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--out", default=str(ROOT / "docs" / "evidence" / "a4-measurement.json"))
    args = parser.parse_args()

    mode = worlds.get("open_factory")
    # `wait_for` with a legal duration: harmless, and it keeps the world turning
    # without the probe pretending to play well.
    reply = json.dumps(
        {
            "action": 26,
            "arguments": {"until": "world_changes", "seconds": 3},
            "reason": "let the world run while the sampler watches",
        }
    )
    adapter = SlowAdapter([reply] * (DECISIONS + 2))
    config = AgentConfig(task_id=mode.id, max_steps=DECISIONS)
    clock = RunClock(limit_seconds=600.0)

    started = time.perf_counter()
    result = run_world(
        mode,
        config,
        adapter,
        master_seed=args.seed,
        game_speed=1.0,
        free_running=True,
        on_ready=clock.start,
        on_finished=clock.stop,
        until=clock.expired,
    )
    wall = round(time.perf_counter() - started, 1)

    run_dir = Path(result["run_dir"])
    samples = [
        json.loads(line)
        for line in (run_dir / "production.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    finalization = result.get("finalization") or {}
    sampling = result.get("sampling") or {}

    # A4.1's artifact set, checked on disk rather than in the return value: an
    # interrupted run leaves the directory and not the dict, and the directory
    # is what anyone reads later.
    artifacts = {
        name: (run_dir / name).exists()
        for name in (
            "config.json",
            "manifest.json",
            "status.json",
            "decisions.jsonl",
            "tool_events.jsonl",
            "production.jsonl",
            "result.json",
            "summary.json",
        )
    }
    tool_events = [
        json.loads(line)
        for line in (run_dir / "tool_events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    saves = sorted(str(path.name) for path in (run_dir / "saves").glob("*.zip"))

    # A sample counts as "during a model call" if its wall clock falls inside
    # one of the intervals the adapter was asleep for. Both are measured, not
    # assumed: the adapter records when it started and stopped thinking.
    origin = time.time() - (time.perf_counter() - started)
    during = 0
    for row in samples:
        at = origin + float(row["wall_seconds"])
        if any(begin <= at <= end for begin, end in adapter.thinking):
            during += 1

    report = {
        "measures": "roadmap A4.3: wall-clock sampling and an orderly stop",
        "world": mode.to_dict(),
        "provider": f"scripted, {THINKING_SECONDS}s per call -- no key, no cost",
        "wall_seconds": wall,
        "gameplay_seconds": round(clock.elapsed_seconds, 1),
        "finalization_seconds": finalization.get("seconds"),
        "decisions": len(result.get("episodes") or []) and result["episodes"][0].get("steps"),
        "sampling": sampling,
        "samples": len(samples),
        "samples_during_a_model_call": during,
        "sample_gaps_seconds": [
            round(b["wall_seconds"] - a["wall_seconds"], 2)
            for a, b in zip(samples, samples[1:], strict=False)
        ],
        "read_ms": [row["read_ms"] for row in samples],
        "first_sample": samples[0] if samples else None,
        "last_sample": samples[-1] if samples else None,
        "finalization": finalization,
        "checkpoints": result.get("checkpoints"),
        "reordered_replies": (result.get("viewer") or {}).get("reordered_replies"),
        "run_dir": str(run_dir),
        "artifacts": artifacts,
        "tool_events": len(tool_events),
        "summary": summary,
        "saves": saves,
    }
    report["verdict"] = {
        "samples_were_written": len(samples) >= 3,
        "samples_landed_during_model_calls": during >= 1,
        "cadence_is_about_five_seconds": all(
            gap <= 3 * sampling.get("interval_seconds", 5.0)
            for gap in report["sample_gaps_seconds"]
        ),
        "no_sample_failed": sampling.get("failures") == 0,
        "ticks_advanced_while_the_model_thought": (
            bool(samples) and samples[-1]["tick"] > samples[0]["tick"]
        ),
        "finalization_ran_in_order": [entry["step"] for entry in finalization.get("steps") or []][
            -2:
        ]
        == ["pause", "save"],
        "no_finalization_step_failed": not finalization.get("failed_steps"),
        "the_world_was_still_for_the_save": bool(finalization.get("world_was_still_for_the_save")),
        "gameplay_and_finalization_are_separate": (
            finalization.get("seconds") is not None
            and clock.elapsed_seconds > 0
            and clock.elapsed_seconds < wall
        ),
        "every_a4_1_artifact_was_written": all(artifacts.values()),
        "tool_events_are_per_action": len(tool_events) >= summary["decisions"],
        "the_summary_separates_decisions_from_actions": (
            summary["tool_actions"] >= summary["decisions"]
        ),
        "an_initial_and_a_final_save_exist": (
            any("initial" in name for name in saves) and any("final" in name for name in saves)
        ),
        "the_final_save_is_verified": any(
            entry.get("label") == "final" and entry.get("verified")
            for entry in (result.get("checkpoints") or {}).get("saves") or []
        ),
    }
    Path(args.out).write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report["verdict"], indent=2))
    print(
        f"{len(samples)} samples, {during} during a model call; "
        f"gameplay {report['gameplay_seconds']}s, finalization "
        f"{report['finalization_seconds']}s of {wall}s wall"
    )
    print(f"wrote {args.out}")
    return 0 if all(report["verdict"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
