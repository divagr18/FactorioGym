"""A filmed demo: the agent sets up iron, copper and coal in the real game.

`watch_program.py` plays an evolved program on a holdout scene. This plays one
scripted builder, written against factory-sim's `World` API, on a scene made
for watching: three small patches -- coal to the north, iron to the south-west,
copper to the south-east, about 25 tiles apart -- and a character carrying the
kit a burner start needs.

What the agent does, every step one decision of `ParameterizedEnv(profile="v2")`
through the mod's own action dispatcher (so the overlay shows each one):

1. walks to the coal patch and places two burner drills facing each other, so
   each drops coal into the other's fuel slot, plus a third drill filling a
   wooden chest; fuels all three;
2. walks to the iron patch: two burner drills dropping into two stone furnaces,
   fuelled;
3. walks to the copper patch: the same, smelting copper;
4. makes supply rounds: coal from the chest and the coal pair, then iron and
   copper plates out of the furnaces and coal top-ups into the machines;
5. walks to the middle and the camera pulls back over all three sites.

    uv run python tools/demo_play.py                       # watch only
    uv run python tools/demo_play.py --shots runtime/videos/demo-shots
    uv run python tools/demo_play.py --record runtime/videos/demo-raw.mp4

The scene and the episode limits are the demo's own (a longer construction
budget, no verification window); no measured task, catalog or observation is
changed. Like every watched run it is a demonstration, not a measurement.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from fsim.program_api import World  # noqa: E402
from program_transfer import OP_WAIT, EngineBackend  # noqa: E402
from watch_agent import launch_client, wait_for_client  # noqa: E402
from watch_program import start_recording, stop_recording  # noqa: E402

from factoriorl.agent.viewer import enable_overlay  # noqa: E402
from factoriorl.engine_config import resolve_engine_config  # noqa: E402
from factoriorl.env import FactorioEnv  # noqa: E402
from factoriorl.parameterized import ParameterizedEnv  # noqa: E402
from factoriorl.rcon import RCONClient  # noqa: E402
from factoriorl.seeding import Branch, SeedPlan  # noqa: E402
from factoriorl.session import WorkerSession  # noqa: E402
from factoriorl.tasks import RegisteredTask, get  # noqa: E402
from factoriorl.tasks.spec import Blueprint, ResourceSpec  # noqa: E402
from factoriorl.worker import WorkerManager  # noqa: E402

OP_PLACE, OP_GIVE, OP_TAKE = 12, 16, 17

# --------------------------------------------------------------------- scene
#
# Patch rectangles as (x0, x1, y0, y1), inclusive tile bounds.
COAL = (-3, 3, -14, -8)
IRON = (-16, -10, 5, 11)
COPPER = (10, 16, 5, 11)
START = (0.5, 0.5)
MIDDLE = (0.5, 1.5)
KIT = {"burner-mining-drill": 7, "stone-furnace": 4, "wooden-chest": 1, "coal": 60}


def _tiles(rect):
    x0, x1, y0, y1 = rect
    return [(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]


def demo_blueprint(_family=None, _rng=None) -> Blueprint:
    resources = [ResourceSpec("coal", (float(x), float(y)), 20000) for x, y in _tiles(COAL)]
    resources += [ResourceSpec("iron-ore", (float(x), float(y)), 20000) for x, y in _tiles(IRON)]
    resources += [
        ResourceSpec("copper-ore", (float(x), float(y)), 20000) for x, y in _tiles(COPPER)
    ]
    return Blueprint(
        resources=tuple(resources),
        character_position=START,
        character_inventory=dict(KIT),
        markers={"patch": (0.0, -11.0)},
        unlock_recipes=("burner-mining-drill", "stone-furnace", "wooden-chest"),
        radius=48,
    )


def demo_task() -> RegisteredTask:
    """construct_smelting_line's catalog and observation, on the demo scene,
    with room for a five-minute film: no decision cap to speak of and a
    construction budget of an hour, so no verification window cuts in."""
    base = get("construct_smelting_line")
    spec = replace(base.spec, max_decision_steps=100_000, max_game_ticks=216_000 + 3_600)
    return RegisteredTask(spec=spec, generate=demo_blueprint, solve=None)


# ------------------------------------------------------------------- pacing


class Timeline:
    """When, in the recording, the agent was only waiting -- the stretches
    `make_clip.py` speeds up -- and when each phase began. Seconds from the
    moment the recorder was started."""

    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self.idle: list[list[float]] = []
        self.phases: list[tuple[str, float]] = []
        self._idle_since: float | None = None

    def now(self) -> float:
        return round(time.perf_counter() - self.t0, 2)

    def set_idle(self, idle: bool) -> None:
        if idle and self._idle_since is None:
            self._idle_since = self.now()
        elif not idle and self._idle_since is not None:
            self.idle.append([self._idle_since, self.now()])
            self._idle_since = None

    def phase(self, name: str) -> None:
        self.phases.append((name, self.now()))

    def write(self, path: Path) -> None:
        self.set_idle(False)
        data = {"idle": self.idle, "phases": self.phases, "end": self.now()}
        path.write_text(json.dumps(data, indent=1) + "\n", "utf-8")


class PacedBackend(EngineBackend):
    """The transfer backend, with a pause after each build or transfer so a
    viewer can read it. The world keeps running through the pause (free-running),
    exactly as it would while a person looks at what they just did."""

    def __init__(self, penv, observation, pause: float, timeline: Timeline, shoot=None) -> None:
        super().__init__(penv, observation)
        self.pause = pause
        self.timeline = timeline
        #: Called as shoot(op, n) right after the n-th build or transfer, while
        #: its floating text is still up -- for checking framing.
        self.shoot = shoot
        self.acted = 0

    def step(self, vector) -> float:
        op = int(vector[0])
        self.timeline.set_idle(op == OP_WAIT)
        moved = super().step(vector)
        if op in (OP_PLACE, OP_GIVE, OP_TAKE):
            self.acted += 1
            if self.shoot is not None:
                self.shoot(op, self.acted)
            time.sleep(self.pause)
        return moved


# ------------------------------------------------------------------ program


def demo_program(world: World, *, linger: float, rounds: int, on_phase=None) -> None:
    """The builder the agent runs. Everything it knows comes from `world`:
    where it stands, what it holds, the ore it can see and the entity table."""
    tolerance = 0.3
    stride = 38 / 256

    def phase(name):
        if on_phase is not None:
            on_phase(name)

    def step_towards(goal):
        here = world.me()
        dx, dy = goal[0] - here[0], goal[1] - here[1]
        if abs(dx) <= tolerance and abs(dy) <= tolerance:
            return False
        if abs(dx) >= abs(dy):
            distance, direction = abs(dx), ("E" if dx > 0 else "W")
        else:
            distance, direction = abs(dy), ("S" if dy > 0 else "N")
        if distance >= 30 * stride:
            world.move(direction, "long")
        elif distance >= 7 * stride:
            world.move(direction, "step")
        else:
            world.move(direction, "nudge")
        return True

    def walk(goal):
        for _ in range(200):
            if not step_towards(goal):
                return

    def nearest(kind, point):
        best, best_d = None, 0.0
        for e in world.entities():
            if e.kind == kind:
                d = math.hypot(e.x - point[0], e.y - point[1])
                if best is None or d < best_d:
                    best, best_d = e, d
        return best

    def wait(seconds):
        """Wait decisions (30 ticks each) for `seconds` of real time."""
        until = time.perf_counter() + seconds
        while time.perf_counter() < until:
            world.wait()

    def held(item):
        return world.inventory().get(item, 0)

    def check_ore(kind, anchors):
        seen = set(world.ore_tiles(kind))
        for ax, ay in anchors:
            square = {(ax, ay), (ax + 1, ay), (ax, ay + 1), (ax + 1, ay + 1)}
            if not square <= seen:
                print(f"  note: {kind} not seen under the drill at {(ax, ay)}", flush=True)

    def drill_at(anchor):
        return nearest("mining-drill", (anchor[0] + 1, anchor[1] + 1))

    def furnace_at(anchor):
        return nearest("furnace", (anchor[0] + 1, anchor[1] + 1))

    def fuel(entity, amount=5):
        if entity is not None and held("coal") >= 1:
            world.give(entity, "coal", amount)

    # -- the coal patch: a self-fuelling pair, and a drill filling a chest
    pair_a, pair_b, chest_drill = (-3, -10), (-1, -10), (2, -10)
    chest = (3, -8)
    coal_stand = (0.5, -6.5)
    # -- the smelting sites: two drills over the bottom rows, furnaces below
    iron_drills, iron_furnaces = [(-15, 9), (-12, 9)], [(-15, 11), (-12, 11)]
    iron_stand = (-12.5, 7.5)
    copper_drills, copper_furnaces = [(11, 9), (14, 9)], [(11, 11), (14, 11)]
    copper_stand = (13.5, 7.5)

    phase("start")
    wait(3)

    phase("coal")
    walk(coal_stand)
    check_ore("coal", [pair_a, pair_b, chest_drill])
    world.place("burner-mining-drill", pair_a[0], pair_a[1], "E")
    world.place("burner-mining-drill", pair_b[0], pair_b[1], "W")
    fuel(drill_at(pair_a))
    fuel(drill_at(pair_b))
    world.place("burner-mining-drill", chest_drill[0], chest_drill[1], "S")
    world.place("wooden-chest", chest[0], chest[1], "N")
    fuel(drill_at(chest_drill))
    wait(linger)

    def smelting_site(kind, stand, drills, furnaces):
        walk(stand)
        check_ore(kind, drills)
        for d, f in zip(drills, furnaces, strict=True):
            world.place("burner-mining-drill", d[0], d[1], "S")
            world.place("stone-furnace", f[0], f[1], "N")
            fuel(drill_at(d))
            fuel(furnace_at(f))

    phase("iron")
    smelting_site("iron-ore", iron_stand, iron_drills, iron_furnaces)
    wait(linger)
    phase("copper")
    smelting_site("copper-ore", copper_stand, copper_drills, copper_furnaces)
    wait(linger * 1.5)

    for round_ in range(rounds):
        phase(f"round {round_ + 1}")
        walk(coal_stand)
        box = nearest("container", (chest[0] + 0.5, chest[1] + 0.5))
        if box is not None:
            world.take(box, "coal", 20)
        for anchor in (pair_a, pair_b):
            d = drill_at(anchor)
            if d is not None:
                world.take(d, "coal", 20)
        fuel(drill_at(chest_drill))
        wait(linger / 2)
        for plate, stand, drills, furnaces in (
            ("iron-plate", iron_stand, iron_drills, iron_furnaces),
            ("copper-plate", copper_stand, copper_drills, copper_furnaces),
        ):
            walk(stand)
            for d, f in zip(drills, furnaces, strict=True):
                oven = furnace_at(f)
                if oven is not None:
                    world.take(oven, plate, 20)
                fuel(drill_at(d))
                fuel(oven)
            wait(linger)

    phase("overview")
    walk(MIDDLE)


# --------------------------------------------------------------------- camera


def lua_players(runner, body: str) -> object:
    return runner.lua(
        f"local n = 0 for _, p in pairs(game.connected_players) do {body} n = n + 1 end return n"
    )


def screenshot(runner, name: str, zoom: float, folder: str = "demo") -> None:
    """A frame from the client's own renderer, GUI included: written to the
    client's script-output (`%APPDATA%/Factorio/script-output/<folder>`)."""
    lua_players(
        runner,
        "game.take_screenshot{player = p, by_player = p, resolution = {1920, 1080}, "
        f"zoom = {zoom}, show_gui = true, show_entity_info = true, "
        f"path = '{folder}/{name}.png'}}",
    )


def pull_back(runner, start: float, end: float, seconds: float) -> None:
    """Zoom every spectator from `start` to `end`, smoothly, in real time."""
    frames = max(1, int(seconds * 20))
    for k in range(1, frames + 1):
        z = start + (end - start) * (k / frames)
        lua_players(runner, f"p.zoom = {z:.4f}")
        time.sleep(seconds / frames)


# ----------------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--game-speed", type=float, default=1.0)
    parser.add_argument("--join-wait", type=float, default=25.0)
    parser.add_argument("--zoom", type=float, default=1.25, help="camera zoom while working")
    parser.add_argument("--final-zoom", type=float, default=0.7, help="zoom of the last shot")
    parser.add_argument("--pause", type=float, default=1.4, help="s after each place/transfer")
    parser.add_argument("--linger", type=float, default=8.0, help="s spent watching a site")
    parser.add_argument("--rounds", type=int, default=2, help="supply rounds after building")
    parser.add_argument("--hold", type=float, default=14.0, help="s on the final wide shot")
    parser.add_argument("--title", default="Agent playing Factorio: burner start")
    parser.add_argument("--record", type=Path, help="mp4 path; omit to only watch")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--monitor", type=int, default=0, help="ddagrab output index")
    parser.add_argument(
        "--shots",
        default=None,
        help="take a client screenshot at each phase into script-output/<SHOTS>",
    )
    args = parser.parse_args()

    task = demo_task()
    engine = resolve_engine_config()
    manager = WorkerManager()
    handle = manager.launch("demo-play")
    client = recorder = None
    started = time.perf_counter()
    try:
        with RCONClient(handle.spec.rcon_endpoint, timeout=30.0) as rcon:
            rcon.lua(f"game.speed = {args.game_speed} return game.speed")
        address = f"127.0.0.1:{handle.spec.ports.game}"
        client = launch_client(engine.executable, address, handle.spec.mod_directory)
        session = WorkerSession(handle, timeout=120.0)
        session.status()
        session.configure(free_running=True)
        runner = session._client
        warning = enable_overlay(runner, title=args.title, zoom=args.zoom)
        if warning:
            print(warning, flush=True)
        plan = SeedPlan(master=0, run_id="demo-play")
        env = FactorioEnv(task, session, plan, branch=Branch.EVAL, split="test")
        penv = ParameterizedEnv(env, profile="v2")
        observation, _info = penv.reset(options={"scene_index": 0})
        wait_for_client(args.join_wait)
        # The join wait ran the world on; nothing has been built, so only the
        # clock moved. Put the camera where it should be before filming.
        lua_players(runner, f"p.zoom = {args.zoom}")
        # A cleaner frame: entity info on (drill output arrows, what each
        # furnace is smelting), the research prompt and side buttons off.
        lua_players(
            runner,
            "local v = p.game_view_settings v.show_entity_info = true "
            "v.show_research_info = false v.show_side_menu = false "
            "v.show_alert_gui = false v.show_shortcut_bar = false p.game_view_settings = v",
        )
        time.sleep(2)
        timeline = Timeline()
        if args.record:
            print(f"recording monitor {args.monitor} -> {args.record}", flush=True)
            recorder = start_recording(args.record, args.fps, args.monitor)
            timeline = Timeline()
            time.sleep(2)
        film_start = time.perf_counter()

        def on_phase(name: str) -> None:
            print(f"[{time.perf_counter() - film_start:6.1f}s] {name}", flush=True)
            timeline.phase(name)
            if args.shots:
                screenshot(runner, name.replace(" ", "-"), args.zoom, args.shots)

        shoot = None
        if args.shots:

            def shoot(op: int, n: int) -> None:
                if n in (1, 4, 16, 30):
                    screenshot(runner, f"action-{n:02d}-op{op}", args.zoom, args.shots)

        backend = PacedBackend(penv, observation, args.pause, timeline, shoot)
        world = World(backend, decision_budget=100_000)
        demo_program(world, linger=args.linger, rounds=args.rounds, on_phase=on_phase)
        print(
            f"[{time.perf_counter() - film_start:6.1f}s] built {len(world._built)}: "
            f"{world._built}; {world.decisions} decisions, {world.refusals} refused intents, "
            f"{world.failures} failed actions; holding {world.inventory()}",
            flush=True,
        )
        for line in world._trace:
            if "refused" in line or "failed" in line:
                print("   ", line, flush=True)
        # Wide shot: re-enable the overlay at the final zoom so its text stays
        # the same size on screen, after a smooth pull-back.
        timeline.set_idle(False)
        timeline.phase("pull back")
        pull_back(runner, args.zoom, args.final_zoom, 4.0)
        enable_overlay(runner, title=args.title, zoom=args.final_zoom)
        timeline.phase("hold")
        if args.shots:
            time.sleep(2)
            screenshot(runner, "final", args.final_zoom, args.shots)
        time.sleep(args.hold)
        timeline.phase("end")
        print(f"[{time.perf_counter() - film_start:6.1f}s] done", flush=True)
        if args.record:
            timeline.write(args.record.with_suffix(".timeline.json"))
        report_state(runner)
        session.close()
    finally:
        if recorder is not None:
            stop_recording(recorder)
        if client is not None:
            client.terminate()
        manager.cleanup(handle)
    print(f"total {time.perf_counter() - started:.0f}s", flush=True)
    return 0


def report_state(runner) -> None:
    """What is standing and what it holds, from the engine, for the log."""
    out = runner.lua(
        "local s = {} for _, e in pairs(game.surfaces[1].find_entities_filtered{"
        "force = 'player', type = {'mining-drill', 'furnace', 'container'}}) do "
        "local inv = e.get_output_inventory() local fuel = e.get_fuel_inventory() "
        "s[#s + 1] = string.format('%s@%.1f,%.1f %s fuel=%d out=%s', e.name, e.position.x, "
        "e.position.y, e.status and tostring(e.status) or '-', "
        "fuel and fuel.get_item_count() or 0, "
        "inv and serpent.line(inv.get_contents()) or '-') end return table.concat(s, ' | ')"
    )
    print("engine state:", out, flush=True)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    sys.exit(main())
