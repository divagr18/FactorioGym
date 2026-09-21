"""Scripted reference solutions (DESIGN.md 3.2).

    For each family provide: ... A scripted solution.

The point of these is not to solve the game. It is to answer one question
mechanically: **is this task reachable through the action catalog the policy
is given?** A solver that has full evaluator knowledge -- exact marker
positions, exact container contents -- and still cannot finish has proved the
task is unsolvable *through the catalog*, not merely unlearned. That
distinguishes a task defect from a training shortfall, which is the distinction
DESIGN section 4 asks to be made before a gate is judged.

They live in their own module so the training entrypoint provably never imports
them; a unit test asserts that, because DESIGN 4.2 forbids scripted solution
labels reaching ordinary PPO training.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from factoriorl.env import FactorioEnv
from factoriorl.seeding import Branch


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


def _brief(arguments: dict) -> str:
    """Arguments as they appear in a trace: short, ordered, no numpy repr."""
    parts = []
    for name in sorted(arguments):
        value = arguments[name]
        if isinstance(value, (list, tuple)) and len(value) == 2:
            parts.append(f"{name}=({value[0]:g},{value[1]:g})")
        else:
            parts.append(f"{name}={value}")
    return " ".join(parts)


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

    def do(self, key: str, **arguments) -> bool:
        """Take one catalog action. False once the episode is over.

        With arguments, this goes through `step_arguments`, which re-checks
        every value against the domain the *policy* could see. The two checks
        are not redundant and the order matters: `action_masks` already refuses
        an action one of whose argument domains is empty, so reaching the
        per-argument check means the domain had values and this particular
        value was not among them. That is a solver bug worth naming, and a
        `ValueError` here is recorded as one rather than crashing the run --
        the whole point of a scripted solution is to distinguish "the task is
        unreachable through the catalog" from "the script is wrong".
        """
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
        try:
            step = (
                self.env.step_arguments(action, arguments) if arguments else self.env.step(action)
            )
        except ValueError as exc:
            self.trace.stuck_reason = f"'{key}' argument refused: {exc}"
            self.trace.last_error = str(exc)
            return False
        _, _, self.terminated, self.truncated, info = step
        self.trace.steps += 1
        self.trace.action_log.append(key if not arguments else f"{key}({_brief(arguments)})")
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

    def entities_of_type(self, type_name: str) -> list[dict]:
        return [
            record
            for record in (self.env._observation.get("entities") or [])
            if record.get("type") == type_name
        ]

    def resource_tiles(self) -> set[tuple[int, int]]:
        """Ore tiles, floored. A drill needs ore beneath it, so this is what
        decides where a drill can go."""
        return {
            (math.floor(tile["p"][0]), math.floor(tile["p"][1]))
            for tile in ((self.env._observation.get("resources") or {}).get("tiles") or [])
            if tile.get("p")
        }

    def placement_for(self, tile: tuple[int, int]) -> list[float] | None:
        """The domain value naming `tile`, or None if it is not offered.

        `step_arguments` compares the argument against the domain by equality,
        so the solver has to hand back the value the domain actually holds
        rather than an equal-looking one it computed itself.
        """
        for candidate in self.env.argument_domains()["placements"]:
            if (math.floor(candidate[0]), math.floor(candidate[1])) == tile:
                return candidate
        return None

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
        """Greedy axis-first walk. Phase 2 has no pathfinding (DESIGN defers it
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
                # nearer. This is still not pathfinding (DESIGN defers that to
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
#: Drill centres to try before giving up. The first is nearest the patch
#: centre; a collision or a range refusal falls through to the next rather than
#: ending the episode, because one refused placement is not an unsolvable task.
BUILD_ATTEMPTS = 4

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


def solve_plate_line(driver: Driver) -> None:
    """Commission the line: fuel the drill, fuel the furnace, then wait.

    Two well-placed transfers is the whole task, and where the character stands
    is the whole difficulty. `give_coal_20` moves coal to the *nearest* entity
    and the drill and furnace are two tiles apart, so the walk has to end on the
    far side of the machine being fuelled -- north of the drill, south of the
    furnace. Standing between them fuels whichever happens to be nearer, and an
    agent that fuels the drill twice never lights the furnace.

    Measured against the engine: the drill takes 19 coal and the furnace 20, and
    3,600 ticks later the pair has produced 14 plates with both at status 1.
    """
    drill, furnace = driver.marker("drill"), driver.marker("furnace")
    if drill is None or furnace is None:
        driver.trace.stuck_reason = "missing drill/furnace markers"
        return

    # Approach each machine from the side away from the other, so "nearest" is
    # unambiguous when the transfer lands.
    drill_side = (drill[0], drill[1] - 2.0)
    furnace_side = (furnace[0], furnace[1] + 2.0)

    if not driver.walk_to(drill_side, tolerance=1.2):
        driver.trace.stuck_reason = "could not reach the drill's approach"
        return
    if not driver.do("give_coal_20"):
        return
    if not driver.walk_to(furnace_side, tolerance=1.2):
        driver.trace.stuck_reason = "could not reach the furnace's approach"
        return
    if not driver.do("give_coal_20"):
        return

    # Smelting is the slow part: about one plate per 240 ticks, so thirty plates
    # is roughly 7,200 ticks, and the solver has to stand still for them.
    for _ in range(320):
        if driver.success or driver.terminated:
            return
        if not driver.do("wait"):
            return


def solve_build_line(driver: Driver) -> None:
    """Build a drill and a furnace, fuel both, and let the line run.

    The geometry is measured, not guessed -- see `families/build_line.py`. In
    short: a requested tile centre `t + 0.5` snaps to the integer centre
    `t + 1`; a south-facing drill at `D` drops into tile `(D.x, D.y + 1)`; and
    a 2x2 furnace covers tiles `x in {cx-1, cx}`, `y in {cy-1, cy}`, so the two
    centres that catch the drop without overlapping the drill are
    `(D.x, D.y + 2)` and `(D.x + 1, D.y + 2)`.

    The drill's position is read back from the observation rather than assumed
    after placing, so a snap the solver got wrong shows up as a stuck reason
    here instead of as a silently misaligned furnace.
    """
    patch = driver.marker("patch")
    if patch is None:
        driver.trace.stuck_reason = "no 'patch' marker in task truth"
        return

    ore = driver.resource_tiles()
    if not ore:
        # The character starts off the patch and ore is in the observation, so
        # walking is what makes the patch visible.
        if not driver.walk_to(patch, tolerance=2.0, interact_range=8.0):
            return
        ore = driver.resource_tiles()
    if not ore:
        driver.trace.stuck_reason = "no ore tiles visible after walking to the patch"
        return

    # A drill snapped to centre D occupies tiles x in {D.x-1, D.x},
    # y in {D.y-1, D.y}. Require ore under all four: `can_place_entity` refuses
    # a drill with nothing to mine, and requiring the whole footprint keeps the
    # choice valid on the narrow strip as well as the square.
    def buildable(centre: tuple[int, int]) -> bool:
        return all((centre[0] + dx, centre[1] + dy) in ore for dx in (-1, 0) for dy in (-1, 0))

    goal = (math.floor(patch[0]), math.floor(patch[1]))
    centres = sorted(
        (c for c in {(x + 1, y + 1) for x, y in ore} if buildable(c)),
        key=lambda c: (abs(c[0] - goal[0]) + abs(c[1] - goal[1]), c),
    )
    if not centres:
        driver.trace.stuck_reason = (
            f"no drill centre has ore under all four tiles ({len(ore)} ore tiles visible)"
        )
        return

    for drill_centre in centres[:BUILD_ATTEMPTS]:
        if _build_at(driver, drill_centre):
            break
        if driver.terminated or driver.truncated:
            return
    else:
        return

    # Nothing left to do but let it run. Each wait is `decision_ticks`, and the
    # window is 3600 ticks, so this is the bulk of the episode by design: the
    # task is not finished when the line is built, it is finished when the line
    # has been running.
    while not driver.success and not (driver.terminated or driver.truncated):
        if not driver.do("wait"):
            return


def _build_at(driver: Driver, drill_centre: tuple[int, int]) -> bool:
    """Place and fuel a drill/furnace pair for one candidate drill centre."""
    # Stand clear of both footprints and within `PLACEMENT_RADIUS` of each. The
    # pair spans y in {D.y-1 .. D.y+2}, so a spot three tiles to the east is
    # off the structure and inside build distance for all of it.
    standing = (drill_centre[0] + 3.5, drill_centre[1] + 0.5)
    if not driver.walk_to(standing, tolerance=1.2, interact_range=2.0):
        return False

    drill_tile = (drill_centre[0] - 1, drill_centre[1] - 1)
    position = driver.placement_for(drill_tile)
    if position is None:
        driver.trace.stuck_reason = f"tile {drill_tile} is not an offered placement"
        return False
    if not driver.do("place_at", item="burner-mining-drill", position=position, direction="south"):
        return False
    if driver.trace.last_error:
        # A collision or a range refusal is worth trying the next centre for,
        # not worth ending the episode over.
        driver.trace.last_error = None
        return False

    drills = driver.entities_of_type("mining-drill")
    if not drills:
        driver.trace.stuck_reason = "placed a drill but no mining-drill is observable"
        return False
    # Read the snap back rather than assuming it.
    built = min(drills, key=lambda r: math.dist(r["p"], [drill_centre[0], drill_centre[1]]))
    drill_x, drill_y = math.floor(built["p"][0]), math.floor(built["p"][1])

    furnace_tile = (drill_x - 1, drill_y + 1)
    furnace_position = driver.placement_for(furnace_tile)
    if furnace_position is None:
        driver.trace.stuck_reason = f"tile {furnace_tile} is not an offered placement"
        return False
    if not driver.do(
        "place_at", item="stone-furnace", position=furnace_position, direction="north"
    ):
        return False
    if driver.trace.last_error:
        driver.trace.stuck_reason = f"furnace placement refused: {driver.trace.last_error}"
        return False

    furnaces = driver.entities_of_type("furnace")
    if not furnaces:
        driver.trace.stuck_reason = "placed a furnace but no furnace is observable"
        return False
    furnace = min(furnaces, key=lambda r: math.dist(r["p"], [drill_x, drill_y + 2]))

    # Fuel by handle, so neither machine can absorb the whole supply -- the trap
    # `plate_line` documents, where `give_coal_20` targeted whatever was nearest
    # and the furnace never lit.
    for target in (built, furnace):
        if not driver.do("give_to", to=str(target["h"]), item="coal", count=20):
            return False
        if driver.trace.last_error:
            driver.trace.stuck_reason = f"fuelling refused: {driver.trace.last_error}"
            return False
    return True


def solve_diagnose_line(driver: Driver) -> None:
    """Apply every fix, in an order that works on all four fault kinds.

    **This solver does not diagnose, and that is deliberate.** A reference
    solution exists to show that each generated scene is solvable inside its
    budget -- DESIGN 3.2's solvability check -- and nothing more. Reading truth to
    find out which machine is dry and then fixing only that one would make the
    reference's success depend on privileged information, and would make "the
    reference solves it" read as evidence that the diagnosis is easy. It is
    evidence of neither: this walks to both machines, fuels both, and clears the
    furnace's output whether or not it was blocked.

    So the reference's disclosed assistance for this family is *not* fault
    localisation. It is that it never has to choose: the agent's difficulty is
    picking a fix from an observation, and the solver skips the choice by doing
    all of them.

    `give_coal_20` fuels the *nearest* entity and the machines are two tiles
    apart, so each approach ends on the far side of its own machine -- north of
    the drill, south of the furnace -- exactly as `solve_plate_line` does.
    """
    drill, furnace = driver.marker("drill"), driver.marker("furnace")
    if drill is None or furnace is None:
        driver.trace.stuck_reason = "missing drill/furnace markers"
        return

    if not driver.walk_to((drill[0], drill[1] - 2.0), tolerance=1.2):
        driver.trace.stuck_reason = "could not reach the drill's approach"
        return
    if not driver.do("give_coal_20"):
        return
    if not driver.walk_to((furnace[0], furnace[1] + 2.0), tolerance=1.2):
        driver.trace.stuck_reason = "could not reach the furnace's approach"
        return
    if not driver.do("give_coal_20"):
        return

    # Clear the output slot, with enough headroom for the whole episode rather
    # than just for the window. Six withdrawals was the first attempt and it
    # made the reference *lose to random* on the test split -- 0.40 against
    # 0.60 -- because the furnace refilled to a hundred partway through the
    # 400-decision wait and stopped again while the solver stood still. A stone
    # furnace's result slot holds one stack of a hundred, the line makes about
    # 15 plates per 3,600 ticks, and the budget is 15,000 ticks: sixty plates
    # of headroom cannot be consumed inside it, so twelve withdrawals end the
    # fault instead of postponing it.
    for _ in range(12):
        if driver.success or driver.terminated:
            return
        if not driver.do("take_iron-plate_5"):
            break

    # Then stand still: acceptance cannot arrive before tick 3,600 and needs a
    # full 3,600-tick window after that, so this is roughly 240 decisions of
    # waiting and the budget is 500.
    for _ in range(400):
        if driver.success or driver.terminated:
            return
        if not driver.do("wait"):
            return


def solve_keep_line_running(driver: Driver) -> None:
    """Top both machines up on a cycle until the window is satisfied.

    **It does not read the declared outage tick, deliberately.** The tick is in
    `TaskSpec.disruptions` and the solver could look it up, but the agent is
    never told it -- R4.3 asks for a disruption detectable only by monitoring --
    and a reference that timed its repair off a field the agent cannot see
    would be demonstrating a different task's solvability. Cycling costs a few
    decisions and needs no privileged knowledge, so the reference's disclosed
    assistance stays "it knows where the machines are", which every family's
    reference has.

    Each round puts five coal in each machine. Five coal is 20 MJ against a
    150 kW drill, which is about 8,000 ticks -- so a single post-outage top-up
    is enough, and the cycle exists to guarantee one lands after the outage
    rather than to keep the line alive by brute force. Twelve rounds fit in the
    120 coal the scene provides, against a 12,000-tick budget.

    Approaches from the far side of each machine, as `solve_plate_line` does:
    `give_coal_5` fuels the *nearest* entity and the two machines are two tiles
    apart, so standing between them fuels whichever happens to be closer.
    """
    drill, furnace = driver.marker("drill"), driver.marker("furnace")
    if drill is None or furnace is None:
        driver.trace.stuck_reason = "missing drill/furnace markers"
        return
    drill_side = (drill[0], drill[1] - 2.0)
    furnace_side = (furnace[0], furnace[1] + 2.0)

    for _ in range(12):
        if driver.success or driver.terminated:
            return
        if not driver.walk_to(drill_side, tolerance=1.2):
            driver.trace.stuck_reason = "could not reach the drill's approach"
            return
        if not driver.do("give_coal_5"):
            return
        if driver.success or driver.terminated:
            return
        if not driver.walk_to(furnace_side, tolerance=1.2):
            driver.trace.stuck_reason = "could not reach the furnace's approach"
            return
        if not driver.do("give_coal_5"):
            return
        # Then stand still and let it smelt. Acceptance cannot arrive before
        # tick 8,400 -- the outage at 3,600, plus the measured 1,200-tick run-on
        # the fuel clear does not stop, plus a full 3,600-tick window -- so most
        # of this solver's decisions are waiting, as they are on every family
        # whose objective is production.
        for _ in range(24):
            if driver.success or driver.terminated:
                return
            if not driver.do("wait"):
                return


SOLVERS = {
    "navigate": solve_navigate,
    "deliver": solve_deliver,
    "supply_furnace": solve_supply_furnace,
    "plate_line": solve_plate_line,
    "mine_smelt": solve_mine_smelt,
    "repair_belt": solve_repair_belt,
    "restore_power": solve_restore_power,
    "build_line": solve_build_line,
    "diagnose_line": solve_diagnose_line,
    "keep_line_running": solve_keep_line_running,
}


def _solver_for(task_id: str):
    """The reference solver for a task: its own declaration first, then `SOLVERS`.

    `RegisteredTask` has carried a `solve` field since it was written and
    **nothing ever read it**. The real dispatch was this module's hard-coded
    `SOLVERS` dict, which meant the repository's own claim -- "adding a task
    means dropping a module here" (`tasks/families/__init__.py`) -- was false:
    a family that dropped in a module and declared its solver got
    `no reference solution registered`, which makes `tools/solvability.py`
    call it *defective* and fails `gate_phase3`'s 0.99 reference threshold.
    Authoring a task therefore required editing framework internals, which is
    exactly what R6's gate says a user must not have to do.

    A task's own declaration wins, so a new family is self-contained. `SOLVERS`
    stays as the fallback because the ten existing solvers genuinely belong
    here -- they share `Driver`, `_belt_gap`, `_pole_gap` and the rest, and
    moving them into ten family modules would duplicate that machinery for no
    gain.
    """
    from factoriorl import tasks as tasks_module

    try:
        declared = tasks_module.get(task_id).solve
    except Exception:
        # A task absent from the registry is not this function's problem to
        # report; the caller's `None` branch says it better.
        declared = None
    return declared or SOLVERS.get(task_id)


class ReferenceOnEvaluatedEpisode(RuntimeError):
    """A scripted solution was pointed at an evaluated episode.

    R3.1 requires that the reference builder "must not complete the evaluated
    agent's run". Import hygiene already keeps this module out of `train.py`
    -- `tests/unit/test_manifest_and_isolation.py` asserts the import direction -- but
    import hygiene cannot stop a *gate* or an analysis script, both of which
    legitimately import both halves, from handing a solver an eval env. So the
    obligation is enforced where it can actually be violated, in the same
    spirit as `gate_phase2.py`'s guarded RCON client: the attempt raises.

    `Branch.EVAL` is the discriminator because it is already set at every
    evaluation construction site in `train.py`, which means this cannot be
    bypassed by forgetting a new flag.
    """


def solve(env: FactorioEnv) -> SolveTrace:
    """Run the reference solution for `env`'s task on its current episode."""
    if getattr(env, "branch", None) is Branch.EVAL:
        raise ReferenceOnEvaluatedEpisode(
            f"refusing to run the {env.spec_.id} reference solution on an episode "
            "drawn from the eval branch: a scripted solution must never complete "
            "an evaluated run (R3.1)"
        )
    task_id = env.spec_.id
    trace = SolveTrace(task=task_id, budget=env.spec_.max_decision_steps)
    solver = _solver_for(task_id)
    if solver is None:
        trace.stuck_reason = (
            f"no reference solution for {task_id}: it is absent from SOLVERS and its "
            "RegisteredTask declares no `solve`"
        )
        return trace
    driver = Driver(env, trace)
    solver(driver)
    trace.succeeded = driver.success
    if not trace.succeeded and trace.stuck_reason is None:
        trace.stuck_reason = "solver finished without reaching the success predicate"
    return trace


def random_rollout(env, rng: np.random.Generator, budget: int) -> bool:
    """The floor a reference solution is read against.

    The floor has to be measured in the space the policy will actually train
    on, which is the point `tools/solvability.py` makes at length. A
    parameterized catalog therefore has to be sampled per dimension: sampling
    a flat index would either crash on the missing arguments or, worse, keep
    picking the argument-free actions and report a floor for a space nobody
    trains in.
    """
    # One sampler for every random floor. This loop and the trainer's baselines
    # used to be separate implementations of "uniform over legal values", and
    # the trainer's was wrong for factorized spaces.
    from factoriorl.parameterized import sample_masked

    for _ in range(budget):
        step = env.step(sample_masked(env.action_space, env.action_masks(), rng))
        _, _, terminated, truncated, info = step
        if terminated or truncated:
            return bool(info.get("success"))
    return False
