"""Scripted reference solutions (PLAN.md 3.2).

    For each family provide: ... A scripted solution.

The point of these is not to solve the game. It is to answer one question
mechanically: **is this task reachable through the action catalog the policy
is given?** A solver that has full evaluator knowledge -- exact marker
positions, exact container contents -- and still cannot finish has proved the
task is unsolvable *through the catalog*, not merely unlearned. That
distinguishes a task defect from a training shortfall, which is the distinction
PLAN section 4 asks to be made before a gate is judged.

They live in their own module so the training entrypoint provably never imports
them; a unit test asserts that, because PLAN 4.2 forbids scripted solution
labels reaching ordinary PPO training.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from factoriorl.env import FactorioEnv


@dataclass
class SolveTrace:
    """What the solver did, and where it got stuck if it did."""

    task: str
    succeeded: bool = False
    steps: int = 0
    budget: int = 0
    stuck_reason: str | None = None
    action_log: list[str] = field(default_factory=list)
    last_error: str | None = None

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "succeeded": self.succeeded,
            "steps": self.steps,
            "budget": self.budget,
            "budget_fraction": round(self.steps / self.budget, 3) if self.budget else None,
            "stuck_reason": self.stuck_reason,
            "last_error": self.last_error,
            "actions": self.action_log[-24:],
        }


class Driver:
    """Thin helper: address catalog actions by key, and observe as we go."""

    def __init__(self, env: FactorioEnv, trace: SolveTrace) -> None:
        self.env = env
        self.trace = trace
        self.index = {template.key: i for i, template in enumerate(env.catalog.templates)}
        self.terminated = False
        self.truncated = False
        self.success = False

    def available(self, key: str) -> bool:
        return key in self.index

    def do(self, key: str) -> bool:
        """Take one catalog action. False once the episode is over."""
        if self.terminated or self.truncated:
            return False
        if key not in self.index:
            self.trace.stuck_reason = f"catalog has no action '{key}'"
            return False
        action = self.index[key]
        if not self.env.action_masks()[action]:
            # Masked out: the action exists but its precondition is unmet.
            self.trace.stuck_reason = f"'{key}' is masked out"
            return False
        _, _, self.terminated, self.truncated, info = self.env.step(action)
        self.trace.steps += 1
        self.trace.action_log.append(key)
        if info.get("action_error"):
            self.trace.last_error = f"{key}: {info['action_error']}"
        self.success = bool(info.get("success"))
        return True

    # ---- state the solver is allowed to see (it is evaluator-side) --------

    @property
    def position(self) -> tuple[float, float]:
        return tuple((self.env._observation.get("character") or {}).get("position", [0, 0]))

    def marker(self, name: str) -> tuple[float, float] | None:
        target = (self.env._truth.get("markers") or {}).get(name)
        return tuple(target) if target else None

    def container(self, name: str) -> dict:
        return (self.env._truth.get("containers") or {}).get(name, {})

    def inventory(self) -> dict:
        return self.env._observation.get("inventory") or {}

    def nearest_resource(self) -> tuple[float, float] | None:
        tiles = (self.env._observation.get("resources") or {}).get("tiles", [])
        if not tiles:
            return None
        here = self.position
        best = min(tiles, key=lambda t: math.dist(here, t["p"]))
        return tuple(best["p"])

    def walk_to(
        self,
        target,
        tolerance: float = 1.4,
        budget: int = 80,
        interact_range: float = 0.0,
    ) -> bool:
        """Greedy axis-first walk. Phase 2 has no pathfinding (PLAN defers it
        to 5.1), so an obstacle stops progress; detect that instead of spinning."""
        stalled = 0
        for _ in range(budget):
            here = self.position
            dx, dy = target[0] - here[0], target[1] - here[1]
            if abs(dx) <= tolerance and abs(dy) <= tolerance:
                return True
            # A long stride covers ~4.45 tiles and a short one ~1.0, so use
            # the short stride once inside long-stride range -- otherwise the
            # walk overshoots and oscillates forever, which is exactly what the
            # first run of these solvers exposed.
            far = max(abs(dx), abs(dy)) > 5.0
            prefix = "move" if far else "step"
            key = (
                (f"{prefix}_east" if dx > 0 else f"{prefix}_west")
                if abs(dx) >= abs(dy)
                else (f"{prefix}_south" if dy > 0 else f"{prefix}_north")
            )
            if key not in self.index:
                key = key.replace("step_", "move_")
            if not self.do(key):
                return False
            moved = self.position
            if math.dist(here, moved) < 0.05:
                stalled += 1
                if stalled >= 3:
                    # Entity reach is 10 tiles, so being blocked a few tiles
                    # short of a machine is close enough to interact with it --
                    # the goal was never to stand on top of it.
                    if math.dist(moved, target) <= interact_range:
                        return True
                    self.trace.stuck_reason = (
                        f"blocked walking toward {target} at {moved} "
                        "(no pathfinding until Phase 5.1)"
                    )
                    return False
            else:
                stalled = 0
        self.trace.stuck_reason = f"walk budget exhausted heading to {target}"
        return False


# --------------------------------------------------------------- solutions


def solve_navigate(driver: Driver) -> None:
    goal = driver.marker("goal")
    if goal is None:
        driver.trace.stuck_reason = "no 'goal' marker in task truth"
        return
    driver.walk_to(goal, tolerance=1.0)


def solve_deliver(driver: Driver) -> None:
    src, dst = driver.marker("src"), driver.marker("dst")
    if src is None or dst is None:
        driver.trace.stuck_reason = "missing src/dst markers"
        return
    for _ in range(4):
        if driver.inventory().get("iron-plate", 0) >= 10:
            break
        if not driver.walk_to(src, interact_range=6.0):
            return
        if not driver.do("take_iron-plate_20"):
            return
    if not driver.walk_to(dst, interact_range=6.0):
        return
    driver.do("give_iron-plate_20")


def solve_supply_furnace(driver: Driver) -> None:
    ore_chest, furnace = driver.marker("ore_chest"), driver.marker("furnace")
    if ore_chest is None or furnace is None:
        driver.trace.stuck_reason = "missing ore_chest/furnace markers"
        return
    for _ in range(6):
        if driver.success or driver.terminated:
            return
        if not driver.walk_to(ore_chest, interact_range=6.0):
            return
        if not driver.do("take_iron-ore_20"):
            return
        if not driver.walk_to(furnace, interact_range=6.0):
            return
        if not driver.do("give_iron-ore_5"):
            return
        for _ in range(12):
            if driver.success or driver.terminated:
                return
            if not driver.do("wait"):
                return


def solve_mine_smelt(driver: Driver) -> None:
    furnace = driver.marker("furnace")
    patch = driver.marker("patch")
    if furnace is None or patch is None:
        driver.trace.stuck_reason = "missing furnace/patch markers"
        return
    for _ in range(6):
        if driver.success or driver.terminated:
            return
        if not driver.walk_to(patch, tolerance=2.0):
            return
        ore = driver.nearest_resource()
        if ore is None:
            driver.trace.stuck_reason = "no resource visible at the patch"
            return
        # Resource reach is 2.7, tighter than entity reach.
        if not driver.walk_to(ore, tolerance=2.0):
            return
        if not driver.do("mine_nearest_5"):
            return
        # Mining is ongoing: it needs intervals to finish before the ore is in
        # the inventory to hand over.
        for _ in range(10):
            if driver.inventory().get("iron-ore", 0) >= 1:
                break
            if not driver.do("wait"):
                return
        if not driver.walk_to(furnace, interact_range=6.0):
            return
        if not driver.do("give_coal_5"):
            return
        if not driver.do("give_iron-ore_5"):
            return
        for _ in range(15):
            if driver.success or driver.terminated:
                return
            if not driver.do("wait"):
                return


def solve_repair_belt(driver: Driver) -> None:
    """Walk to where a belt must go and place it.

    This is the solver that exposes the catalog defect: `place_transport_belt`
    binds its position to the character's tile plus two east, so the only way
    to place into a gap is to stand exactly two tiles west of it.
    """
    gap = _belt_gap(driver)
    if gap is None:
        driver.trace.stuck_reason = "could not locate the belt gap from the observation"
        return
    # Placement is welded to character + (2, 0).
    stand = (gap[0] - 1, gap[1])
    if not driver.walk_to(stand, tolerance=0.4, budget=120):
        return
    if not driver.do("place_transport_belt_east"):
        return
    for _ in range(30):
        if driver.success or driver.terminated:
            return
        if not driver.do("wait"):
            return


def _belt_gap(driver: Driver) -> tuple[float, float] | None:
    belts = [
        e for e in driver.env._observation.get("entities", []) if e.get("name") == "transport-belt"
    ]
    if not belts:
        return None
    # floor, not round: entity positions are tile centres (x.5), and banker's
    # rounding maps 4.5 -> 4 but 5.5 -> 6, inventing gaps that are not there.
    ys = [math.floor(e["p"][1]) for e in belts]
    line_y = max(set(ys), key=ys.count)
    xs = sorted({math.floor(e["p"][0]) for e in belts if math.floor(e["p"][1]) == line_y})
    for left, right in zip(xs, xs[1:], strict=False):
        if right - left > 1:
            return (float(left + 1), float(line_y))
    return None


def solve_restore_power(driver: Driver) -> None:
    """Same shape as repair_belt: find the pole gap and stand two west of it."""
    poles = [
        e
        for e in driver.env._observation.get("entities", [])
        if e.get("name") == "small-electric-pole"
    ]
    drill = driver.marker("drill")
    if drill is None:
        driver.trace.stuck_reason = "no 'drill' marker in task truth"
        return
    xs = sorted({math.floor(e["p"][0]) for e in poles if math.floor(e["p"][1]) == -8})
    gap_x = None
    for left, right in zip(xs, xs[1:], strict=False):
        if right - left > 4:
            gap_x = left + 4
            break
    if gap_x is None and xs:
        gap_x = xs[-1] + 4
    if gap_x is None:
        driver.trace.stuck_reason = "no poles visible to infer the gap from"
        return
    if not driver.walk_to((gap_x - 1, -8.0), tolerance=0.45, budget=140):
        return
    if not driver.do("place_small_electric_pole_east"):
        return
    for _ in range(20):
        if driver.success or driver.terminated:
            return
        if not driver.do("wait"):
            return


SOLVERS = {
    "navigate": solve_navigate,
    "deliver": solve_deliver,
    "supply_furnace": solve_supply_furnace,
    "mine_smelt": solve_mine_smelt,
    "repair_belt": solve_repair_belt,
    "restore_power": solve_restore_power,
}


def solve(env: FactorioEnv) -> SolveTrace:
    """Run the reference solution for `env`'s task on its current episode."""
    task_id = env.spec_.id
    trace = SolveTrace(task=task_id, budget=env.spec_.max_decision_steps)
    solver = SOLVERS.get(task_id)
    if solver is None:
        trace.stuck_reason = f"no reference solution registered for {task_id}"
        return trace
    driver = Driver(env, trace)
    solver(driver)
    trace.succeeded = driver.success
    if not trace.succeeded and trace.stuck_reason is None:
        trace.stuck_reason = "solver finished without reaching the success predicate"
    return trace


def random_rollout(env: FactorioEnv, rng: np.random.Generator, budget: int) -> bool:
    """The floor a reference solution is read against."""
    for _ in range(budget):
        legal = np.flatnonzero(env.action_masks())
        _, _, terminated, truncated, info = env.step(int(rng.choice(legal)))
        if terminated or truncated:
            return bool(info.get("success"))
    return False
