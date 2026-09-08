"""R3.1 exit gate: bounded construction, measured on a real engine.

    The reference builder completes construction under the same embodied action
    rules. The scene cannot already satisfy the production objective.
    Acceptance counts newly constructed required components and sustained
    output, not just final inventory or one transient plate.

Three clauses, three measurements, none of them asserted.

**Same embodied action rules.** Every action the builder takes goes through
`env.step_arguments`, which re-checks each argument against a domain derived
from the *observation*. The session's RCON client is wrapped so any Lua that is
not the typed dispatch template raises, following `gate_phase2.py`: "the agent
used no privileged mutations" is only meaningful if it *cannot*. Scene install
happens before the builder starts and is counted separately.

**The scene cannot already satisfy the objective.** Checked twice, because the
engine-free check and the engine check can fail for different reasons: once
from the blueprint via `validate_all`, and once on the live engine by
evaluating the success predicate at step 0 of every episode.

**Newly constructed components and sustained output.** `BUILT` counts
placements the `place` handler recorded, so a preplaced machine cannot
contribute -- the install path does not go through that handler, and this gate
measures that by reporting `placed_counts` beside `built`. Sustained output is
demonstrated by *breaking* a working line: the drill is removed through the
ordinary `mine_at` action, a full window is allowed to elapse, and the gate
records that cumulative production stays high while the windowed rate falls to
zero and success goes false. A monotone counter cannot make that distinction,
which is the whole reason the predicate exists.

Run: uv run python -m factoriorl.gate_r3 [--episodes N]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from typing import Any

from factoriorl.engine_config import resolve_game_speed
from factoriorl.env import FactorioEnv
from factoriorl.paths import evidence_dir
from factoriorl.rcon import RCONClient
from factoriorl.seeding import Branch, SeedPlan
from factoriorl.session import WorkerSession
from factoriorl.tasks import get, validate_all
from factoriorl.tasks.reference import Driver, SolveTrace, solve_build_line
from factoriorl.tasks.spec import PredicateKind
from factoriorl.worker import WorkerManager

TASK = "build_line"
SPLITS = ("train", "val", "test")
GATE_MASTER = 20260909


class PrivilegedAccess(RuntimeError):
    """The builder tried to reach the evaluator's arbitrary-Lua channel."""


class GuardedClient:
    """Only the typed dispatch template may be sent once the builder starts.

    Copied from `gate_phase2.py` for the same reason it exists there: the
    evaluator's arbitrary-Lua path shares the RCON connection with the typed
    protocol, so "the builder used the same action rules as a policy" has to be
    enforced rather than described.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.armed = False
        self.blocked = 0
        self.evaluator_calls = 0

    def lua(self, code: str):
        if 'remote.call("frrl_bridge", "dispatch"' not in code:
            if self.armed:
                self.blocked += 1
                raise PrivilegedAccess(f"non-dispatch call while armed: {code[:80]}")
            self.evaluator_calls += 1
        return self._inner.lua(code)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def check(report: dict, label: str, ok: bool, **detail) -> bool:
    report["checks"].append({"check": label, "ok": bool(ok), **detail})
    print(f"  [{'ok ' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""), flush=True)
    return bool(ok)


# --------------------------------------------------------- the three clauses


def clause_scene_cannot_be_solved_already(report: dict) -> None:
    """Engine-free half: the blueprint itself must not satisfy success."""
    validation = validate_all(sample_seeds=16)
    entry = validation["tasks"][TASK]
    already = [p for p in entry["problems"] if "already satisfied at reset" in p]
    check(
        report,
        "blueprint does not satisfy the objective at reset (16 seeds x 3 families)",
        not already,
        offenders=already[:3],
    )
    check(
        report,
        "the reset check covers every success clause of this task",
        entry["reset_check_covers_success"],
        undecidable=entry["undecidable_success"],
    )
    check(
        report,
        "no structural problem in any sampled scene",
        entry["ok"],
        problems=entry["problems"][:3],
    )


def _build_one(env: FactorioEnv, guard: GuardedClient) -> tuple[SolveTrace, Driver, dict]:
    """Run the builder on the current episode with the guard armed."""
    trace = SolveTrace(task=TASK, budget=env.spec_.max_decision_steps)
    driver = Driver(env, trace)
    at_reset = {p.describe(): p.evaluate(env._observation, env._truth) for p in env.spec_.success}
    guard.armed = True
    try:
        solve_build_line(driver)
    finally:
        guard.armed = False
    trace.succeeded = driver.success
    return trace, driver, at_reset


def _build_without_waiting(driver: Driver) -> bool:
    """`solve_build_line` minus its terminal wait loop.

    Reuses the solver's own candidate selection and placement rather than
    reimplementing the geometry, so the line this gate breaks is the same line
    the builder builds.
    """
    from factoriorl.tasks import reference as ref

    patch = driver.marker("patch")
    if patch is None:
        driver.trace.stuck_reason = "no patch marker in task truth"
        return False
    ore = driver.resource_tiles()
    if not ore:
        if not driver.walk_to(patch, tolerance=2.0, interact_range=8.0):
            return False
        ore = driver.resource_tiles()

    def buildable(centre):
        return all((centre[0] + dx, centre[1] + dy) in ore for dx in (-1, 0) for dy in (-1, 0))

    goal = (math.floor(patch[0]), math.floor(patch[1]))
    centres = sorted(
        (c for c in {(x + 1, y + 1) for x, y in ore} if buildable(c)),
        key=lambda c: (abs(c[0] - goal[0]) + abs(c[1] - goal[1]), c),
    )
    for centre in centres[: ref.BUILD_ATTEMPTS]:
        if ref._build_at(driver, centre):
            return True
        if driver.terminated or driver.truncated:
            return False
    return False


def clause_builder_completes_construction(report: dict, session, guard, episodes: int) -> dict:
    """The builder finishes, through the shared contract, on every split."""
    task = get(TASK)
    outcomes: dict[str, list[dict]] = {}
    for split in SPLITS:
        rows = []
        for index in range(episodes):
            env = FactorioEnv(
                task,
                session,
                # The gate's own plan, on the TRAIN branch: `reference.solve`
                # refuses the EVAL branch outright, and a gate that ran the
                # builder on evaluated scenes would be the violation this
                # clause exists to prevent.
                SeedPlan(master=GATE_MASTER, run_id="gate-r3"),
                branch=Branch.TRAIN,
                split=split,
            )
            env._episode_index = index - 1
            env.reset()
            trace, driver, at_reset = _build_one(env, guard)
            truth = env._truth
            rows.append(
                {
                    "split": split,
                    "episode": index,
                    "succeeded": trace.succeeded,
                    "steps": trace.steps,
                    "stuck_reason": trace.stuck_reason,
                    "success_at_reset": at_reset,
                    "built": dict(truth.get("built") or {}),
                    "placed_counts": dict(truth.get("placed_counts") or {}),
                    "working_counts": dict(truth.get("working_counts") or {}),
                    "production": env.metrics.report(),
                    "actions": trace.action_log[:12],
                }
            )
            print(
                f"    {split}[{index}] succeeded={trace.succeeded} steps={trace.steps} "
                f"built={rows[-1]['built']}",
                flush=True,
            )
            # Deliberately not `env.close()`: that closes the *session*, which
            # every episode here shares. Closing it after the first episode
            # left the socket at None and the second reset failed on send.
        outcomes[split] = rows

    flat = [row for rows in outcomes.values() for row in rows]
    check(
        report,
        "the reference builder completes construction on every split",
        all(row["succeeded"] for row in flat),
        splits={s: sum(r["succeeded"] for r in rows) for s, rows in outcomes.items()},
        episodes_per_split=episodes,
    )
    check(
        report,
        "success was false at step 0 of every episode on the live engine",
        all(not any(row["success_at_reset"].values()) for row in flat),
    )
    check(
        report,
        "the builder issued no privileged Lua while it was running",
        guard.blocked == 0,
        blocked=guard.blocked,
        evaluator_calls_outside_the_builder=guard.evaluator_calls,
    )
    check(
        report,
        "both required components were counted as newly built by the agent",
        all(
            row["built"].get("burner-mining-drill", 0) >= 1
            and row["built"].get("stone-furnace", 0) >= 1
            for row in flat
        ),
    )
    # Acceptance must not be reachable by final inventory: nothing the agent
    # holds is counted, and the scene preplaces no machine at all.
    check(
        report,
        "the scene preplaced no machine, so every counted build is the agent's",
        all(
            row["placed_counts"].get("burner-mining-drill", 0)
            == row["built"].get("burner-mining-drill", 0)
            for row in flat
        ),
    )
    report["outcomes"] = outcomes
    return outcomes


def _snapshot(env, window) -> dict:
    return {
        "tick": env._observation.get("tick"),
        "produced": dict(env._truth.get("produced") or {}),
        "window_output": window.window_output(env._truth),
        "window_elapsed": window.window_elapsed(env._truth),
        "success": all(p.evaluate(env._observation, env._truth) for p in env.spec_.success),
        "rate": env.metrics.report()["final_output_rate"],
    }


def clause_sustained_not_transient(report: dict, session, guard, outcomes: dict) -> None:
    """Build a line, kill it, and show acceptance never arrives.

    The break has to happen *before* the criterion is met, not after: success
    terminates the episode, so once the line has been accepted there is no
    further action to take -- the first version of this check tried to mine the
    drill after success and got a refused action with no reason, because the
    episode was already over.

    Breaking early is the sharper measurement anyway. The threshold is reached
    in cumulative terms around tick 2850 while a full window has not elapsed
    until 3600, so there is an interval in which a `PRODUCED` criterion at the
    same threshold *would* accept and the sustained one does not. The drill is
    removed in that interval, two full windows are allowed to pass, and what
    each measurement then says is recorded. The positive control is not
    simulated: it is the episodes above, every one of which reached the
    criterion with a running line.
    """
    task = get(TASK)
    env = FactorioEnv(
        task,
        session,
        SeedPlan(master=GATE_MASTER, run_id="gate-r3-break"),
        branch=Branch.TRAIN,
        split="train",
    )
    env._episode_index = -1
    env.reset()

    window = next(p for p in env.spec_.success if p.kind is PredicateKind.SUSTAINED_OUTPUT)
    trace = SolveTrace(task=TASK, budget=env.spec_.max_decision_steps)
    driver = Driver(env, trace)

    # Build, but not the solver's terminal wait loop -- that would carry the
    # episode past acceptance and terminate it.
    guard.armed = True
    try:
        built = _build_without_waiting(driver)
    finally:
        guard.armed = False
    if not built:
        check(report, "a line was built to break", False, stuck=trace.stuck_reason)
        return

    def tick() -> int:
        return int(env._observation.get("tick") or 0)

    running: dict = {}
    break_tick = 0
    guard.armed = True
    try:
        # Run until the cumulative total has passed the threshold but the
        # window has not yet elapsed.
        while tick() < window.over_ticks - 3 * env.spec_.decision_ticks:
            if driver.terminated or driver.truncated or not driver.do("wait"):
                break
        running = _snapshot(env, window)

        drills = driver.entities_of_type("mining-drill")
        if not drills:
            check(report, "the built drill is addressable for removal", False)
            return
        removed = driver.do("mine_at", handle=str(drills[0]["h"]))
        if not removed:
            check(
                report,
                "the drill could be mined through the catalog",
                False,
                why=trace.stuck_reason,
                terminated=driver.terminated,
                truncated=driver.truncated,
            )
            return
        break_tick = tick()

        # Two windows, not one: the furnace keeps smelting the ore already in
        # it after the drill goes, so a single window would still straddle that
        # trickle and the zero would not mean what it says.
        deadline = break_tick + 2 * window.over_ticks
        while tick() < deadline:
            if driver.terminated or driver.truncated or not driver.do("wait"):
                break
    finally:
        guard.armed = False

    stopped = _snapshot(env, window)
    report["break_test"] = {
        "running": running,
        "break_tick": break_tick,
        "stopped": stopped,
        "threshold": window.at_least,
        "over_ticks": window.over_ticks,
    }

    item = window.item
    accepted = [r for rows in outcomes.values() for r in rows if r["succeeded"]]
    check(
        report,
        "a continuously running line does reach the criterion (positive control)",
        len(accepted) > 0,
        episodes_accepted=len(accepted),
        rates=[r["production"]["final_output_rate"].get(item) for r in accepted[:3]],
    )
    check(
        report,
        "the line was producing before it was broken",
        running["produced"].get(item, 0) > 0,
        produced=running["produced"].get(item),
        at_tick=running["tick"],
    )
    check(
        report,
        "cumulative production did not fall when the line died",
        stopped["produced"].get(item, 0) >= running["produced"].get(item, 0),
        before=running["produced"].get(item),
        after=stopped["produced"].get(item),
    )
    check(
        report,
        "a PRODUCED criterion at the same threshold would have accepted this run",
        stopped["produced"].get(item, 0) >= window.at_least,
        produced=stopped["produced"].get(item),
        threshold=window.at_least,
    )
    check(
        report,
        "the windowed output is zero after two windows with the line dead",
        stopped["window_output"] == 0.0,
        before=running["window_output"],
        after=stopped["window_output"],
    )
    check(
        report,
        "the sustained criterion never accepted the line that ran and stopped",
        not running["success"] and not stopped["success"],
        at_break=running["success"],
        at_end=stopped["success"],
    )
    check(
        report,
        "the reported output rate is zero for the stopped line",
        stopped["rate"].get(item, 1.0) == 0.0,
        before=running["rate"].get(item),
        after=stopped["rate"].get(item),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episodes", type=int, default=3, help="episodes per split")
    args = parser.parse_args()

    report: dict = {
        "gate": "R3.1 bounded construction",
        "task": TASK,
        "clauses": [
            "the reference builder completes construction under the same embodied action rules",
            "the scene cannot already satisfy the production objective",
            "acceptance counts newly constructed required components and sustained "
            "output, not just final inventory or one transient plate",
        ],
        "checks": [],
    }

    print("scene cannot already satisfy the objective (engine-free):", flush=True)
    clause_scene_cannot_be_solved_already(report)

    manager = WorkerManager()
    handle = manager.launch("gate-r3")
    started = time.perf_counter()
    try:
        with RCONClient(handle.spec.rcon_endpoint, timeout=30.0) as client:
            client.lua(f"game.speed = {resolve_game_speed()} return game.speed")
        session = WorkerSession(handle, timeout=60.0)
        session.status()
        # `WorkerSession` holds its client privately; the guard is installed
        # over it so `dispatch_raw` -- the only path any action takes -- goes
        # through the check.
        guard = GuardedClient(session._client)
        session._client = guard
        print("reference builder, through env.step_arguments only:", flush=True)
        outcomes = clause_builder_completes_construction(report, session, guard, args.episodes)
        print("sustained output versus a line that ran and stopped:", flush=True)
        clause_sustained_not_transient(report, session, guard, outcomes)
        session._client = guard._inner
        session.close()
    finally:
        manager.cleanup(handle)

    report["elapsed_seconds"] = round(time.perf_counter() - started, 1)
    report["passed"] = all(c["ok"] for c in report["checks"])
    report["failed"] = [c["check"] for c in report["checks"] if not c["ok"]]
    path = evidence_dir() / "r3-construction.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n{'PASSED' if report['passed'] else 'FAILED'}: wrote {path}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
