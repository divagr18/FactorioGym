"""Run factory-sim builder programs on the real engine, beside the simulator.

A builder program is a `def build(world):` that factory-sim's program search
evolved, scored only in the simulator. Here the same program plays Factorio:
factory-sim's `World` and `play` drive an engine backend over the same
`ParameterizedEnv(profile="v2")` a policy uses, so every call the program
makes is one decision of the same action space, legal by the engine's own
mask and decoder. Each episode is also replayed in the simulator on the same
scene, and the two results are recorded side by side.

Scenes come from the frozen holdout's seed plan (`holdout_v3`: master
20260908, run `holdout-v3`, eval branch), so an index here names the same scene
factory-sim scored. Every episode checks that: the engine's scene digest must
equal factory-sim's digest of the same index.

    uv run python tools/program_transfer.py \\
        --program-db ../factory-sim/runs/evolve-v3-s1/genealogy.sqlite \\
        --first-index 1000 --episodes 32

Needs factory-sim installed in this environment (`pip install factory-sim`, or
an editable install of a checkout).
"""

from __future__ import annotations

import argparse
import json
import lzma
import math
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evolve import evaluate, sandbox  # noqa: E402
from fsim.obsview import ObsView  # noqa: E402
from fsim.program_api import play, run_episode  # noqa: E402

from factoriorl.engine_config import resolve_game_speed  # noqa: E402
from factoriorl.env import FactorioEnv  # noqa: E402
from factoriorl.parameterized import ParameterizedEnv  # noqa: E402
from factoriorl.rcon import RCONClient  # noqa: E402
from factoriorl.seeding import Branch, SeedPlan  # noqa: E402
from factoriorl.session import WorkerSession  # noqa: E402
from factoriorl.tasks import get  # noqa: E402
from factoriorl.worker import WorkerManager  # noqa: E402

OUT = ROOT / "docs" / "evidence" / "program-transfer.json"
HOLDOUT_PLAN = SeedPlan(master=evaluate.HOLDOUT_MASTER, run_id=evaluate.HOLDOUT_RUN_ID)
OP_WAIT = 21
WAIT = (OP_WAIT, 0, 0, 0, 0, 0)
#: The simulator stops a program after 2 s; on the engine every action waits for
#: the game, so the same limit would stop a correct program mid-build.
ENGINE_TIME_LIMIT_S = 900.0


class EngineBackend:
    """factory-sim's backend protocol over a reset `ParameterizedEnv` (v2)."""

    def __init__(self, penv: ParameterizedEnv, observation: dict) -> None:
        self.penv = penv
        self._obs = observation
        self._done = False
        self.steps = 0
        self.info: dict = {}
        self.decode_failures = 0
        self.max_steps = int(penv.env.spec_.max_decision_steps)
        nvec = [int(n) for n in penv.action_space.nvec]
        self._bounds = np.cumsum([0, *nvec])

    @property
    def obs(self) -> dict:
        return self._obs

    @property
    def done(self) -> bool:
        return self._done

    def steps_left(self) -> int:
        return self.max_steps - self.steps

    def legal(self, vector) -> bool:
        """The engine's own check: every component inside its dimension's mask,
        and a vector the decoder accepts."""
        if vector[0] == OP_WAIT:
            return True
        mask = self.penv.action_masks()
        for d, value in enumerate(vector):
            if not mask[self._bounds[d] + int(value)]:
                return False
        return self.penv.decode(np.asarray(vector, dtype=np.int64))[2] is None

    def step(self, vector) -> float:
        before = ObsView(self._obs).char_pos
        obs, _, terminated, truncated, info = self.penv.step(np.asarray(vector, dtype=np.int64))
        self._obs, self.info = obs, info
        self._done = bool(terminated or truncated)
        self.steps += 1
        self.decode_failures += int(bool(info.get("decode_failure")))
        after = ObsView(obs).char_pos
        return math.hypot(after[0] - before[0], after[1] - before[1])

    def plate_tick(self) -> int | None:
        return None  # an evaluator statistic the engine run does not need

    def finish(self, on_step) -> tuple[bool, int]:
        for _ in range(self.max_steps + 1):
            if self._done:
                break
            self.step(WAIT)
            on_step()
        verification = self.info.get("verification") or {}
        return bool(self.info.get("success")), int(verification.get("machine_output") or 0)


def load_program(db: Path) -> dict:
    """The run's final program: best validation score, ties to the older one --
    the rule the run itself selected by."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    row = con.execute(
        "SELECT id, code, val_mean FROM candidates WHERE val_mean IS NOT NULL "
        "ORDER BY val_mean DESC, created, id LIMIT 1"
    ).fetchone()
    con.close()
    return {"run": db.parent.name, "id": row[0], "val_mean": row[2], "code": row[1]}


def run_program(task, session, program: dict, first_index: int, episodes: int) -> list[dict]:
    env = FactorioEnv(task, session, HOLDOUT_PLAN, branch=Branch.EVAL, split="test")
    captured: dict = {}
    install = env._install

    def capturing_install(blueprint):
        captured["payload"] = blueprint.to_dict(
            public_markers=env.spec_.public_markers,
            extra_tracked_items=env.spec_.extra_tracked_items,
        )
        return install(blueprint)

    env._install = capturing_install
    penv = ParameterizedEnv(env, profile="v2")
    build = sandbox.load(program["code"])
    rows = []
    for k in range(episodes):
        index = evaluate.HOLDOUT_START_INDEX + first_index + k
        observation, reset_info = penv.reset(options={"scene_index": index})
        payload = captured["payload"]
        started = time.perf_counter()
        backend = EngineBackend(penv, observation)
        engine = play(build, backend, time_limit_s=ENGINE_TIME_LIMIT_S)
        seconds = time.perf_counter() - started
        sim = run_episode(build, payload, task=task.spec.id, time_limit_s=ENGINE_TIME_LIMIT_S)
        rows.append(
            {
                "index": index,
                "family": reset_info.get("layout_family"),
                "blueprint": reset_info.get("blueprint"),
                "digest_matches_factory_sim": reset_info.get("blueprint")
                == evaluate.scene_digest(payload),
                "engine": {
                    "success": engine.success,
                    "verified_output": engine.verified_output,
                    "decisions": engine.decisions,
                    "refusals": engine.refusals,
                    "failures": engine.failures,
                    "decode_failures": backend.decode_failures,
                    "error": engine.error,
                    "trace": engine.trace,
                    "seconds": round(seconds, 1),
                },
                "simulator": {
                    "success": sim.success,
                    "verified_output": sim.verified_output,
                    "decisions": sim.decisions,
                    "refusals": sim.refusals,
                    "failures": sim.failures,
                    "error": sim.error,
                },
            }
        )
        print(
            program["run"],
            index,
            "engine",
            engine.success,
            engine.verified_output,
            "sim",
            sim.success,
            sim.verified_output,
            f"{seconds:.0f}s",
            flush=True,
        )
    return rows


def summarise(rows: list[dict]) -> dict:
    n = len(rows)

    def rate(side):
        return sum(r[side]["success"] for r in rows) / n if n else 0.0

    return {
        "episodes": n,
        "engine_success": rate("engine"),
        "simulator_success": rate("simulator"),
        "agree": sum(r["engine"]["success"] == r["simulator"]["success"] for r in rows),
        "digests_match": sum(r["digest_matches_factory_sim"] for r in rows),
        "engine_decode_failures": sum(r["engine"]["decode_failures"] for r in rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--program-db", type=Path, action="append", required=True)
    parser.add_argument("--task", default="construct_smelting_line")
    parser.add_argument(
        "--first-index",
        type=int,
        default=1000,
        help="offset into the holdout stream; 1000 on is the slice the programs were reported on",
    )
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    task = get(args.task)
    programs = [load_program(db) for db in args.program_db]
    manager = WorkerManager()
    handle = manager.launch("program-transfer")
    report: dict = {
        "about": "factory-sim builder programs played on the real engine, beside the simulator",
        "task": task.spec.id,
        "task_version": task.spec.version,
        "action_space": "parameterized-v2",
        "seed_plan": {**HOLDOUT_PLAN.to_dict(), "branch": "eval"},
        "holdout_indices": [
            evaluate.HOLDOUT_START_INDEX + args.first_index,
            evaluate.HOLDOUT_START_INDEX + args.first_index + args.episodes - 1,
        ],
        "engine": {k: v for k, v in manager.engine.to_dict().items() if k != "executable"},
        "programs": {},
    }
    episodes: list[dict] = []
    try:
        with RCONClient(handle.spec.rcon_endpoint, timeout=30.0) as client:
            client.lua(f"game.speed = {resolve_game_speed()} return game.speed")
        session = WorkerSession(handle, timeout=120.0)
        session.status()
        for program in programs:
            rows = run_program(task, session, program, args.first_index, args.episodes)
            report["programs"][program["run"]] = {
                "candidate": program["id"],
                "val_mean": program["val_mean"],
                "source": program["code"],
                "summary": summarise(rows),
            }
            episodes += [{"program": program["run"], **r} for r in rows]
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", "utf-8")
        session.close()
    finally:
        manager.cleanup(handle)
    episodes_path = args.out.with_name(args.out.stem + ".episodes.jsonl.xz")
    report["episodes_file"] = episodes_path.name
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", "utf-8")
    lines = "".join(json.dumps(r, sort_keys=True) + "\n" for r in episodes)
    episodes_path.write_bytes(lzma.compress(lines.encode(), preset=9))
    print(json.dumps({name: p["summary"] for name, p in report["programs"].items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
