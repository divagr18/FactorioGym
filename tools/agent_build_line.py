"""R3.2: a bounded LLM baseline on `build_line`, and what it cost.

R3.2's gate: *"Replay shows agent-issued construction, commissioning, and the
complete measurement window. Report supplied knowledge, assistance, game time,
decisions, inference usage, and throughput. A failed policy remains a valid
baseline result; the construction demonstration gate stays unmet until an agent
actually passes."*

So this reports all six, and it reports a failure as a result. The three things
worth saying about how it is set up:

**Supplied knowledge is enumerated, not summarised.** The system prompt, its
digest, the task's own declared description, and the argument-domain rules the
prompt states are all written into the report verbatim. R3.2 also forbids
embedding the reference build sequence in runtime assistance, so the report
records the check for that -- `FORBIDDEN_IN_PROMPT` names the geometry phrases
the reference solver had to measure on an engine, and the prompt is scanned for
them. The task's objective is fair game; the geometry is not.

**Waiting is batched, and that is an assistance.** `build_line`'s window is
3,600 game ticks against a 30-tick decision, so a construction run is ~240
decisions of which ~200 are waiting for a furnace. Re-asking a paid provider to
re-read a 2,700-character prompt 200 times to say "wait" again measures the
price of patience. `--wait-batch` repeats a wait the model *just chose*,
stopping as soon as the observation changes, and appears in the manifest's
`assistance` field as `wait-batch:N`.

**Nothing here can complete the run.** No reference solver is imported, and the
loop only ever repeats an action the model chose. `reference.solve` refuses the
eval branch outright, which is a separate guard.

Run:
    uv run python tools/agent_build_line.py --episodes 1 --wait-batch 12
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from factoriorl.agent.adapters import OpenAICompatibleAdapter  # noqa: E402
from factoriorl.agent.loop import (  # noqa: E402
    FORBIDDEN_IN_PROMPT,
    SYSTEM_PROMPT,
    AgentConfig,
    _prompt_digest,
)
from factoriorl.agent.runner import run_task  # noqa: E402
from factoriorl.agent.summary import SUMMARY_ENCODING_VERSION  # noqa: E402
from factoriorl.paths import evidence_dir  # noqa: E402
from factoriorl.tasks import get  # noqa: E402

TASK = "build_line"


def _manifest(result: dict) -> dict:
    """The run's own manifest, which is where `assistance` is recorded."""
    import json as _json

    path = Path(result.get("run_dir") or "") / "manifest.json"
    if not path.is_file():
        return {}
    return _json.loads(path.read_text(encoding="utf-8"))


def _write_replay(result: dict) -> str | None:
    """Build the replay page, so the gate's first clause is an artefact.

    R3.2's gate is worded about what the *replay* shows, not about what a
    summary claims, so the page is produced by the same run that produced the
    numbers rather than left for someone to remember to build.
    """
    run_dir = Path(result.get("run_dir") or "")
    if not run_dir.is_dir():
        return None
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        import replay as replay_module

        return str(replay_module.build(run_dir, None))
    except Exception as exc:  # noqa: BLE001 - a missing replay is reported, not fatal
        print(f"replay not written: {type(exc).__name__}: {exc}", flush=True)
        return None


def _decisions(result: dict) -> list[dict]:
    """Every decision, read from the run's own `decisions.jsonl`.

    The loop streams decisions to disk and returns only aggregates, so reading
    `result["episodes"][i]["decisions"]` found nothing and this report showed
    "0 decisions" for a run that made 17. Reading the replay file is also the
    right source for a gate whose wording is about what the *replay* shows.
    """
    import json as _json

    path = Path(result.get("run_dir") or "") / "decisions.jsonl"
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(_json.loads(line))
    return rows


#: Reported so a cost/success trade-off is possible at all, which synthesis
#: section 10 asks for: "report a cost/success trade-off for compact PPO, LLMs,
#: and hybrids rather than a single score hiding different compute budgets".
def _usage_totals(result: dict) -> dict:
    prompt = completion = calls = 0
    for decision in _decisions(result):
        for attempt in decision.get("attempts") or []:
            usage = attempt.get("usage") or {}
            calls += 1
            prompt += int(usage.get("prompt_tokens") or 0)
            completion += int(usage.get("completion_tokens") or 0)
    return {
        "provider_calls": calls,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


def _last_tick(result: dict, episode: int | None) -> int | None:
    """The last game tick the replay recorded for one episode."""
    ticks = [
        int((row.get("observation") or {}).get("tick") or 0)
        for row in _decisions(result)
        if episode is None or row.get("episode") == episode
    ]
    return max(ticks) if ticks else None


def _construction_evidence(result: dict) -> dict:
    """Which placements the agent issued, and what the engine said.

    The gate asks the *replay* to show agent-issued construction. The replay is
    the decision record, so this reads it rather than re-deriving anything: an
    action key, the arguments the model supplied, and the engine's own verdict.
    """
    placements: list[dict] = []
    transfers: list[dict] = []
    waits_batched = 0
    decisions = 0
    fallbacks = 0
    failures: dict[str, int] = {}
    for decision in _decisions(result):
        decisions += 1
        outcome = decision.get("result") or {}
        waits_batched += int(outcome.get("batched_waits") or 0)
        if decision.get("resolution") != "model":
            fallbacks += 1
        for attempt in decision.get("attempts") or []:
            # `failure` is a top-level field on the recorded attempt, not
            # nested under an `outcome`. Reading the wrong key reported no
            # decision failures for a run whose aggregate counted 79.
            failure = attempt.get("failure") or attempt.get("error_kind")
            if failure:
                failures[failure] = failures.get(failure, 0) + 1
        row = {
            "episode": decision.get("episode"),
            "step": decision.get("step"),
            "arguments": decision.get("arguments") or {},
            "status": outcome.get("action_status"),
            "error": outcome.get("action_error"),
            "reason": decision.get("reason"),
        }
        if decision.get("action_key") == "place_at":
            placements.append(row)
        elif decision.get("action_key") in ("give_to", "take_from"):
            transfers.append(row)
    return {
        "decisions": decisions,
        "fallback_decisions": fallbacks,
        "waits_repeated_without_asking": waits_batched,
        "placements_issued": placements,
        "placements_accepted": [p for p in placements if not p["error"]],
        "transfers_issued": transfers,
        "transfers_accepted": [t for t in transfers if not t["error"]],
        "decision_failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--model", default=os.environ.get("FRRL_AGENT_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=300,
        help="hard ceiling on decisions per episode; a paid provider makes an "
        "unbounded episode an unbounded bill",
    )
    parser.add_argument(
        "--wait-batch",
        type=int,
        default=12,
        help="repeat a wait the model chose, up to this many times, stopping as "
        "soon as the observation changes. Declared as an assistance.",
    )
    parser.add_argument("--out", default="r3-agent-baseline.json")
    parser.add_argument(
        "--from-run",
        default=None,
        help="rebuild the report from an existing run directory instead of "
        "playing again. The replay is the source of every number here, so a "
        "reporting fix must not cost another episode of inference.",
    )
    args = parser.parse_args()

    if not args.from_run and not os.environ.get(args.api_key_env):
        print(f"{args.api_key_env} is not set in the environment")
        return 2

    leaked = [phrase for phrase in FORBIDDEN_IN_PROMPT if phrase.lower() in SYSTEM_PROMPT.lower()]
    if leaked:
        # Refused rather than reported: a baseline run whose prompt contains the
        # reference geometry is not a baseline, and publishing it as one is the
        # failure R3.2's wording exists to prevent.
        print(f"refusing to run: the system prompt contains {leaked}")
        return 1

    spec = get(TASK).spec
    config = AgentConfig(
        task_id=TASK,
        episodes=args.episodes,
        split=args.split,
        max_steps=args.max_steps,
        wait_batch=args.wait_batch,
        run_prefix="r3agent",
    )
    adapter = OpenAICompatibleAdapter(
        base_url=args.base_url,
        model=args.model,
        api_key_env=args.api_key_env,
        timeout=args.timeout,
    )
    if args.from_run:
        run_dir = Path(args.from_run)
        if not run_dir.is_dir():
            run_dir = ROOT / "runtime" / "runs" / args.from_run
        if not run_dir.is_dir():
            print(f"no such run directory: {args.from_run}")
            return 2
        result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        # Older runs recorded no `run_dir`; the path is what was asked for.
        result["run_dir"] = str(run_dir)
        print(f"rebuilding the report from {run_dir}", flush=True)
    else:
        print(f"running {TASK} on {args.split} with {args.model}", flush=True)
        result = run_task(config, adapter)

    report = {
        "gate": "R3.2 bounded LLM baseline",
        "task": {"id": TASK, "version": spec.version, "split": args.split},
        # ---- the six things the gate asks to be reported -----------------
        "supplied_knowledge": {
            "system_prompt": SYSTEM_PROMPT,
            "system_prompt_digest": _prompt_digest(),
            "task_description": spec.description,
            "summary_encoding_version": SUMMARY_ENCODING_VERSION,
            "observation_profile": spec.observation_profile,
            "action_profile": spec.action_profile,
            "catalog": spec.catalog,
            "reference_geometry_withheld": sorted(FORBIDDEN_IN_PROMPT),
            "reference_geometry_present_in_prompt": leaked,
            "note": (
                "The task's declared description is supplied; the placement "
                "geometry the reference builder measured on an engine is not. "
                "No reference solver is imported by this tool."
            ),
        },
        "assistance": _manifest(result).get("profiles", {}).get("assistance"),
        "profiles": _manifest(result).get("profiles", {}),
        "game_time": {
            "ticks_per_decision": spec.decision_ticks,
            "max_game_ticks": spec.max_game_ticks,
            "episodes": [
                {
                    "episode": e.get("episode"),
                    # `final_tick` and `production` are recorded per episode by
                    # the loop. A run made before that landed still has the
                    # game time in its replay, so it is read from there rather
                    # than reported as unknown.
                    "final_tick": e.get("final_tick")
                    if e.get("final_tick") is not None
                    else _last_tick(result, e.get("episode")),
                    "final_tick_source": (
                        "episode record" if e.get("final_tick") is not None else "replay"
                    ),
                    "steps": e.get("steps"),
                    "success": e.get("success"),
                    "stopped": e.get("stopped"),
                    "production": e.get("production"),
                }
                for e in result.get("episodes") or []
            ],
        },
        "decisions": _construction_evidence(result),
        "inference_usage": _usage_totals(result),
        # Two different throughputs, kept apart. "how fast did the provider
        # answer" and "how much iron did the line make per 3600 ticks" are both
        # real and neither substitutes for the other -- and a report that said
        # only the first would describe the model's speed as the task's.
        "throughput": {
            "model": {
                "provider_calls": result.get("model_calls"),
                "latency_ms": result.get("latency_ms") or {},
                "decisions_per_provider_call": (
                    round(result["decisions"] / result["model_calls"], 3)
                    if result.get("model_calls")
                    else None
                ),
            },
            "production_per_3600_ticks": [
                {
                    "episode": e.get("episode"),
                    "final_output_rate": (e.get("production") or {}).get("final_output_rate"),
                    "cumulative_produced": (e.get("production") or {}).get("cumulative_produced"),
                    "ticks_to_first_sustained_output": (e.get("production") or {}).get(
                        "ticks_to_first_sustained_output"
                    ),
                }
                for e in result.get("episodes") or []
            ],
        },
        # ---- the verdict, stated the way the gate states it --------------
        # Computed from the episodes: the loop's aggregate carries latency and
        # usage, not a success rate, so reading one from it reported None.
        "success_rate": (
            round(
                sum(1 for e in result.get("episodes") or [] if e.get("success"))
                / len(result["episodes"]),
                4,
            )
            if result.get("episodes")
            else None
        ),
        "run_id": result.get("run_id"),
        "run_dir": str(result.get("run_dir") or ""),
    }
    successes = sum(1 for e in result.get("episodes") or [] if e.get("success"))
    report["episodes_played"] = len(result.get("episodes") or [])
    report["verdict"] = (
        "an agent constructed and commissioned the line"
        if successes
        else "the agent did not complete construction; a failed baseline is a valid result"
    )
    report["construction_demonstration_gate_met"] = bool(successes)

    report["replay"] = _write_replay(result)
    path = evidence_dir() / args.out
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    usage = report["inference_usage"]
    evidence = report["decisions"]
    print(
        # `episodes_played`, not `args.episodes`: in `--from-run` the episode
        # count comes from the run being read, and printing the flag's default
        # reported a two-episode run as "0/1".
        f"\nsuccess {successes}/{report['episodes_played']}; "
        f"{evidence['decisions']} decisions, "
        f"{evidence['waits_repeated_without_asking']} batched waits, "
        f"{len(evidence['placements_accepted'])}/{len(evidence['placements_issued'])} "
        f"placements accepted, {usage['total_tokens']} tokens over "
        f"{usage['provider_calls']} calls",
        flush=True,
    )
    print(f"wrote {path}", flush=True)
    # 0 either way: a failed agent is a valid baseline result and must not read
    # as a broken tool.
    return 0


if __name__ == "__main__":
    sys.exit(main())
