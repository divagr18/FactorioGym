"""Record what a trained policy actually did, decision by decision.

Nothing in the RL path logs a policy's actions. `agent/loop.py` records
per-decision traces for the language-model path and `tools/replay.py` renders
them; for a trained policy it prints "no primitive trace recorded for this
run". `per_scene` says how an episode *ended* and never how far it got, so
three separate questions are stuck on the same absence:

* **The four scenes nothing solves.** 3007, 3026, 3033 and 3040 were missed by
  all six cells of the paired 4.4 comparison, every one of the 24 attempts
  burning exactly 120 decisions and ending on a `precondition` rejection. Six
  independently trained policies failing identically is a property of the
  scenes, but a rate cannot say whether the scene is unsolvable or the policy
  merely never gets there. The scripted solver is refused on eval-branch scenes
  by design (`ReferenceOnEvaluatedEpisode`, R3.1), so the trace is the
  remaining route.
* **R5.2's probe** requires confirming "policies' actual actions rather than
  inferring them solely from generator distributions".
* **R5.1's report** requires identifying "where attempts fail".

This is deliberately not wired into `evaluate_parallel`. Recording every
decision of every evaluated episode would add a large amount of data to every
run's result for the sake of a handful of scenes, and it would touch the path
that produces published evidence. This drives named scenes with a loaded
checkpoint instead, changing nothing about how runs are scored. If R5-C ends up
consuming it routinely it belongs in `src/`; until then one file is the honest
size.

Uses the frozen holdout's own seed plan, so a named index is the same scene the
cells were scored on rather than a lookalike, and records the blueprint digest
so that can be checked rather than trusted.

Run:
  uv run python tools/trace_scenes.py --checkpoint <run_id> --episodes 3007,3026,3033,3040 \\
      --controls 3000,3001 --out docs/evidence/r5-deliver-hardcore-trace.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from factoriorl import manifest as manifest_module  # noqa: E402
from factoriorl.engine_config import resolve_game_speed  # noqa: E402
from factoriorl.env import FactorioEnv  # noqa: E402
from factoriorl.rcon import RCONClient  # noqa: E402
from factoriorl.seeding import Branch, SeedPlan  # noqa: E402
from factoriorl.session import WorkerSession  # noqa: E402
from factoriorl.skills import SkillEnv  # noqa: E402
from factoriorl.tasks import get  # noqa: E402
from factoriorl.worker import WorkerManager  # noqa: E402


def holdout_plan(path: str) -> tuple[SeedPlan, dict]:
    """The frozen holdout's own seed plan, so a named index is the real scene.

    A scene is a pure function of (master, run_id, branch, index) and `run_id`
    enters the seed, so rebuilding the plan from anything else produces
    different worlds under the same numbers.
    """
    frozen = json.loads(Path(path).read_text(encoding="utf-8"))["holdout"]
    spec = frozen["seed_plan"]
    return SeedPlan(master=spec["master"], run_id=spec["run_id"]), frozen


def marker_distances(observation: dict, truth: dict) -> dict:
    """How far the character is from each published marker, per decision.

    The cheapest possible answer to "how far did it get": a policy that never
    approaches the source is failing somewhere different from one that reaches
    it and cannot transfer the item.
    """
    position = (observation.get("character") or {}).get("position") or [0.0, 0.0]
    markers = truth.get("markers") or {}
    return {
        name: round(math.dist(position, point), 2)
        for name, point in markers.items()
        if isinstance(point, (list, tuple)) and len(point) == 2
    }


def trace_episode(env, inner, model, index: int, deterministic: bool = True) -> dict:
    """Roll one named scene out and record every decision.

    `env` is what the policy acts through (possibly a `SkillEnv`); `inner` is
    the `FactorioEnv` underneath, which is where the scene index, the world
    observation and evaluator truth live.
    """
    inner._episode_index = index - 1
    # The env returns the *encoded* observation the policy trains on, so using
    # what it hands back is identical to the training path by construction --
    # re-encoding here would risk a different profile or goal vector and score
    # the policy on a tensor it never saw.
    observation, _info = env.reset()
    truth = inner._truth or {}
    steps: list[dict] = []
    total = 0.0
    terminated = truncated = False

    while True:
        mask = env.action_masks()
        batched = {key: value[None, ...] for key, value in observation.items()}
        action, _ = model.predict(
            batched, action_masks=mask[None, ...], deterministic=deterministic
        )
        chosen = int(action[0] if hasattr(action, "__len__") else action)

        observation, reward, terminated, truncated, info = env.step(chosen)
        total += float(reward)
        steps.append(
            {
                "step": info.get("steps"),
                "action_index": chosen,
                # The key, not just the index: an index means nothing without
                # the catalog it was resolved against.
                "action_key": info.get("action_key"),
                "action_status": info.get("action_status"),
                # Per step, unlike `per_scene`, which keeps only the terminal
                # one -- so a run of rejections is visible rather than summarised
                # to its last member.
                "action_error": info.get("action_error"),
                "reward": round(float(reward), 4),
                "distance_to": marker_distances(inner._observation, inner._truth or {}),
                "carrying": {
                    item: count
                    for item, count in (
                        (inner._observation.get("character") or {}).get("inventory") or {}
                    ).items()
                    if count
                },
            }
        )
        if terminated or truncated:
            break

    return {
        "episode_index": index,
        "layout_family": inner._family.name if inner._family else None,
        "solved": bool(info.get("success")),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "decisions": len(steps),
        "total_reward": round(total, 4),
        "markers": truth.get("markers"),
        "terminal_action_error": info.get("action_error"),
        # A census of what the policy tried, which is the thing `per_scene`
        # cannot give: it keeps one terminal error and no action identities.
        "action_key_counts": _counts(step["action_key"] for step in steps),
        "action_error_counts": _counts(step["action_error"] for step in steps),
        "closest_approach": _closest(steps),
        "steps": steps,
    }


def _counts(values) -> dict:
    from collections import Counter

    return dict(Counter(values))


def _closest(steps: list[dict]) -> dict:
    """Nearest the policy ever came to each marker.

    "It never got within 8 tiles of the source" and "it stood on the source and
    could not pick up" are different failures, and this is the one number that
    separates them.
    """
    best: dict = {}
    for step in steps:
        for name, distance in (step.get("distance_to") or {}).items():
            if name not in best or distance < best[name]:
                best[name] = distance
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", default="deliver")
    parser.add_argument("--checkpoint", required=True, help="run id, or path to model.zip")
    parser.add_argument("--episodes", required=True, help="comma-separated episode indices")
    parser.add_argument(
        "--controls",
        default="",
        help="indices the policy is known to solve. Without these a failure on "
        "the scenes of interest cannot be distinguished from a broken harness",
    )
    parser.add_argument("--holdout", default="docs/evidence/holdout_v3.json")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--skills",
        action="store_true",
        help="wrap in SkillEnv. Must match the checkpoint: the action space is "
        "part of the architecture, and a mismatch is a shape error at predict "
        "time. `shaping_comparison` enables skills by default, so the 4.4 "
        "checkpoints need this",
    )
    parser.add_argument("--sampled", action="store_true", help="sample instead of argmax")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    from sb3_contrib import MaskablePPO

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        checkpoint = manifest_module.runs_dir() / args.checkpoint / "model.zip"
    if not checkpoint.is_file():
        raise SystemExit(f"no checkpoint at {checkpoint}")

    wanted = [int(v) for v in args.episodes.split(",") if v.strip()]
    controls = [int(v) for v in args.controls.split(",") if v.strip()]
    plan, frozen = holdout_plan(args.holdout)
    task = get(args.task)

    manager = WorkerManager()
    handle = manager.launch("trace")
    started = time.perf_counter()
    try:
        with RCONClient(handle.spec.rcon_endpoint, timeout=30.0) as client:
            client.lua(f"game.speed = {resolve_game_speed()} return game.speed")
        session = WorkerSession(handle, timeout=30.0)
        session.status()
        model = MaskablePPO.load(checkpoint, device="cpu")
        inner = FactorioEnv(task, session, plan, branch=Branch.EVAL, split=args.split)
        # The action space is part of what the checkpoint was trained against,
        # so this has to match the run's `config.skills` or `predict` fails on a
        # shape mismatch -- 13 primitive actions against 19 with skills.
        env = SkillEnv(inner) if args.skills else inner

        traces = []
        for group, indices in (("subject", wanted), ("control", controls)):
            for index in indices:
                trace = trace_episode(env, inner, model, index, deterministic=not args.sampled)
                trace["group"] = group
                traces.append(trace)
                print(
                    f"{group:8s} ep {index}: solved={trace['solved']} "
                    f"decisions={trace['decisions']} "
                    f"closest={trace['closest_approach']} "
                    f"errors={trace['action_error_counts']}",
                    flush=True,
                )
    finally:
        manager.shutdown(handle)

    report = {
        "records": "what a trained policy did, per decision, on named frozen scenes",
        "task": {"id": task.spec.id, "version": task.spec.version},
        "checkpoint": str(checkpoint),
        "holdout": {
            "id": frozen.get("id"),
            "content_hash": frozen.get("content_hash"),
            "seed_plan": frozen.get("seed_plan"),
        },
        "arm": "argmax" if not args.sampled else "sampled",
        "action_space": "skills" if args.skills else "primitive",
        "wall_seconds": round(time.perf_counter() - started, 1),
        "traces": traces,
    }
    print(json.dumps({k: v for k, v in report.items() if k != "traces"}, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
