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
            # Three strides: ~4.45 tiles, ~1.04, ~0.30. Each tier must be
            # finer than the error it is asked to close, or the walk oscillates
            # on that stride's lattice instead of converging. Two tiers was not
            # enough: a 1.04-tile minimum stride cannot land within 0.4 of a
            # tile centre, which is what every placement needs.
            error = max(abs(dx), abs(dy))
            prefix = "move" if error > 5.0 else ("step" if error > 1.5 else "nudge")
            key = (
                (f"{prefix}_east" if dx > 0 else f"{prefix}_west")
                if abs(dx) >= abs(dy)
                else (f"{prefix}_south" if dy > 0 else f"{prefix}_north")
            )
            if key not in self.index:
                key = key.replace("nudge_", "step_")
            if key not in self.index:
                key = key.replace("step_", "move_")
            if not self.do(key):
                return False
            moved = self.position
            if math.dist(here, moved) < 0.05:
                stalled += 1
                # Entity reach is 10 tiles, so being blocked a few tiles short
                # of a machine is close enough to interact with it -- the goal
                # was never to stand on top of it.
                if math.dist(moved, target) <= interact_range:
                    return True
                # Greedy axis-first movement walks straight into any obstacle
                # placed across the direct line, which is exactly what a
                # structural holdout like `screened_depot` puts there. Slide
                # along the blocking face instead of giving up: commit to the
                # perpendicular axis for a few strides, alternating sides on
                # each successive stall so a wall is escaped whichever end is
                # nearer. This is still not pathfinding (PLAN defers that to
                # 5.1) -- it is enough to round a convex obstacle, and a
                # reference solver is allowed to be more capable than the
                # policy it validates the task for.
                if stalled > MAX_SIDESTEPS:
                    self.trace.stuck_reason = (
                        f"blocked walking toward {target} at {moved} "
                        "(no pathfinding until Phase 5.1)"
                    )
                    return False
                sideways = ("north", "south") if abs(dx) >= abs(dy) else ("west", "east")
                side = sideways[stalled % 2]
                for _ in range(SIDESTEP_STRIDES):
                    if not self.do(
                        f"step_{side}" if f"step_{side}" in self.index else f"move_{side}"
                    ):
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
    # Carry twenty and deposit twenty. Depositing five of a twenty-ore carry
    # threw away three quarters of each trip, which was invisible while the
    # target was five plates and made the task unsolvable the moment it rose.
    #
    # Then wait long enough for the furnace to actually work: a stone furnace
    # smelts an iron plate in 3.2 s, so twenty plates is 64 s, which at a
    # 30-tick decision interval is about 128 decisions. Twelve was never going
    # to be enough -- the solver was walking away mid-smelt.
    for _ in range(3):
        if driver.success or driver.terminated:
            return
        if not driver.walk_to(ore_chest, interact_range=6.0):
            return
        if not driver.do("take_iron-ore_20"):
            return
        if not driver.walk_to(furnace, interact_range=6.0):
            return
        if not driver.do("give_iron-ore_20"):
            return
        for _ in range(140):
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
        # Mining is ongoing, and slow: about four decision intervals per ore.
        # Wait for the *whole* batch, because `give_iron-ore_5` transfers five
        # or fails with `no_items` -- and walking to the furnace cancels
        # mining, so there is no second chance to top up. Waiting for one ore
        # and leaving is why this family reported `no_items` while mining was
        # in fact working perfectly.
        carried = 0
        for _ in range(40):
            carried = driver.inventory().get("iron-ore", 0)
            if carried >= GIVE_BATCH:
                break
            if not driver.do("wait"):
                return
        if carried < GIVE_BATCH:
            driver.trace.stuck_reason = (
                f"mined only {carried} ore in 40 intervals, need {GIVE_BATCH}"
            )
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
    """Repair every defect on the line, not just the first one.

    The layout families differ in *how many* and *what kind* of defect there is
    -- `double_gap` has two holes, `misrotation` leaves a belt facing the wrong
    way -- so a solver that places one belt and stops passes `gap` and fails
    the rest. It fixes whatever the observation still shows is wrong.
    """
    for _ in range(6):
        if driver.success or driver.terminated:
            break
        gap = _belt_gap(driver)
        if gap is not None:
            # Approach from the south and place north. Standing west of the gap
            # and placing east would be the obvious route, but placement binds
            # to floor(position), so the standing tile selects the target -- and
            # the tile west of a gap holds a belt. South wins over north because
            # the catalog welds facing to the placement offset: a north-placed
            # belt is one clockwise rotation from the east the line needs.
            if not driver.walk_to((gap[0], gap[1] + 1.0), tolerance=0.35, budget=140):
                return
            if not driver.do("place_transport_belt_north"):
                return
            if not _rotate_to_east(driver, gap):
                return
            continue
        crooked = _misrotated_belt(driver)
        if crooked is None:
            break
        if not driver.walk_to((crooked[0], crooked[1] + 1.0), tolerance=0.35, budget=140):
            return
        if not _rotate_to_east(driver, crooked):
            return
    for _ in range(40):
        if driver.success or driver.terminated:
            return
        if not driver.do("wait"):
            return


def _rotate_to_east(driver: Driver, position) -> bool:
    """Turn the belt at `position` until it faces east, or give up."""
    for _ in range(4):
        belt = _entity_at(driver, position)
        if belt is None or belt.get("d") == _EAST:
            return True
        if not driver.do("rotate_target"):
            return False
    return True


def _misrotated_belt(driver: Driver):
    belts = [
        e for e in driver.env._observation.get("entities", []) if e.get("name") == "transport-belt"
    ]
    if not belts:
        return None
    ys = [math.floor(e["p"][1]) for e in belts]
    line_y = max(set(ys), key=ys.count)
    for belt in sorted(belts, key=lambda e: e["p"][0]):
        if math.floor(belt["p"][1]) == line_y and belt.get("d") != _EAST:
            return (belt["p"][0], belt["p"][1])
    return None


#: How many times a walk may slide along a blocking face before it is
#: declared stuck, and how far each slide goes.
MAX_SIDESTEPS = 6
SIDESTEP_STRIDES = 3

#: Pole spacing along the chain. A wider run than this is a missing pole.
POLE_SPACING = 4

#: `give_<item>_5` moves five at a time or nothing.
GIVE_BATCH = 5

#: Factorio 2.0 uses 16 directions, so east is 4 rather than 2.
_EAST = 4


def _entity_at(driver: Driver, position, radius: float = 0.4):
    for entity in driver.env._observation.get("entities", []):
        point = entity.get("p") or [0, 0]
        if math.dist(point, position) <= radius:
            return entity
    return None


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
            # Tile centre, not tile index: everything downstream is world
            # coordinates, and tile n spans [n, n+1).
            return (left + 1.5, line_y + 0.5)
    return None


def solve_restore_power(driver: Driver) -> None:
    """Walk the pole chain until the gap is visible, then fill it.

    Inferring the gap once from spawn does not work and the reason is the point
    of the environment: a chain of nine poles puts the drill near x=37, well
    past the 32-tile sensor radius, so the pole list is *truncated*. The old
    fallback -- "no interior gap visible, so the gap must be past the last pole
    I can see" -- then placed into the middle of an intact chain and collided.

    Walking east re-scans as more of the line comes into view, and the gap is
    only trusted when it is bounded on both sides: by two poles, or by a pole
    and the drill. An unbounded guess is exactly the wrong one.
    """
    drill = driver.marker("drill")
    if drill is None:
        driver.trace.stuck_reason = "no 'drill' marker in task truth"
        return

    for _ in range(12):
        if driver.success or driver.terminated:
            return
        gap = _pole_gap(driver, drill)
        if gap is not None:
            row = gap[1]
            if not driver.walk_to((gap[0], row - 1.0), tolerance=0.35, budget=160):
                return
            # Approach from the north and place south: the tile west of the gap
            # holds a pole. Poles are rotationally symmetric, so unlike a belt
            # there is nothing to correct afterwards.
            if not driver.do("place_small_electric_pole_south"):
                return
            for _ in range(20):
                if driver.success or driver.terminated:
                    return
                if not driver.do("wait"):
                    return
            continue
        # Nothing conclusive in view: move along the line toward the drill.
        here = driver.position
        if abs(here[0] - drill[0]) <= 3.0:
            driver.trace.stuck_reason = "reached the drill without finding a bounded gap"
            return
        if not driver.walk_to((here[0] + 8.0, drill[1]), tolerance=1.5, budget=60):
            return


def _pole_gap(driver: Driver, drill) -> tuple[float, float] | None:
    """A gap in the pole chain that is bounded on both sides, or None."""
    poles = [
        e
        for e in driver.env._observation.get("entities", [])
        if e.get("name") == "small-electric-pole"
    ]
    if not poles:
        return None
    # The chain's row is the modal one: the poles feeding the solar farm sit on
    # their own row, and picking the wrong one would look like a one-pole chain.
    rows = [math.floor(e["p"][1]) for e in poles]
    row = max(set(rows), key=rows.count)
    xs = sorted({math.floor(e["p"][0]) for e in poles if math.floor(e["p"][1]) == row})
    for left, right in zip(xs, xs[1:], strict=False):
        if right - left > POLE_SPACING:
            return (left + POLE_SPACING + 0.5, row + 0.5)
    # The drill is the chain's right-hand boundary, so a last pole too far from
    # it is a gap even with nothing beyond to bracket it. Only trust this when
    # the drill is actually in view.
    if xs and math.floor(drill[1]) == row and drill[0] - xs[-1] > POLE_SPACING:
        return (xs[-1] + POLE_SPACING + 0.5, row + 0.5)
    return None


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
