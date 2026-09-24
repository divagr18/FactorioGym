"""Watch -- and record -- one builder program play the real engine.

`program_transfer.py` measures programs headless; this plays one program on one
holdout scene with a graphical client joined as a spectator (the mod keeps its
camera on the agent), and optionally records that client's window with ffmpeg.

    uv run python tools/watch_program.py --programs runtime/jobs/gt/progs_laptop.jsonl \\
        --run grpo-b-00 --index 1000 --record runtime/videos/grpo-b-00.mp4

A watched run adds a `LuaPlayer` and reorders RCON replies (see
`factoriorl/agent/viewer.py`), so it is a demonstration, not a measurement: the
engine and simulator results are printed but nothing is written to
docs/evidence. The episode itself is stepped exactly as in the measured runs.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from evolve import evaluate, sandbox  # noqa: E402
from fsim.program_api import play  # noqa: E402
from program_transfer import (  # noqa: E402
    ENGINE_TIME_LIMIT_S,
    HOLDOUT_PLAN,
    EngineBackend,
    load_programs_jsonl,
)
from watch_agent import launch_client, wait_for_client  # noqa: E402

from factoriorl.agent.viewer import DEFAULT_OVERLAY_ZOOM, enable_overlay  # noqa: E402
from factoriorl.engine_config import resolve_engine_config  # noqa: E402
from factoriorl.env import FactorioEnv  # noqa: E402
from factoriorl.parameterized import ParameterizedEnv  # noqa: E402
from factoriorl.rcon import RCONClient  # noqa: E402
from factoriorl.seeding import Branch  # noqa: E402
from factoriorl.session import WorkerSession  # noqa: E402
from factoriorl.tasks import get  # noqa: E402
from factoriorl.worker import WorkerManager  # noqa: E402


def client_window_title(pid: int, timeout: float = 60.0) -> str:
    """The client's main window title, read from Windows (ffmpeg captures by title)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", f"(Get-Process -Id {pid}).MainWindowTitle"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        if out:
            return out
        time.sleep(1)
    raise RuntimeError("client window never appeared")


def start_recording(out: Path, fps: int, monitor: int) -> subprocess.Popen:
    """Record the whole monitor with Desktop Duplication (`ddagrab`).

    `gdigrab` by window title returns a blank white frame for Factorio: the game
    renders on the GPU and GDI never sees it. Desktop Duplication captures what
    the monitor shows, so the client must be the foreground window.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", f"ddagrab=output_idx={monitor}:framerate={fps}:draw_mouse=0",
        "-vf", "hwdownload,format=bgra,format=yuv420p",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        str(out),
    ]  # fmt: skip
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def stop_recording(proc: subprocess.Popen) -> None:
    try:
        proc.communicate(b"q", timeout=60)  # ffmpeg finalises the file on 'q'
    except subprocess.TimeoutExpired:
        proc.kill()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--programs", type=Path, required=True)
    parser.add_argument("--run", required=True, help="which program (its `run` name)")
    parser.add_argument("--index", type=int, default=1000, help="holdout offset, as in transfer")
    parser.add_argument("--task", default="construct_smelting_line")
    parser.add_argument("--game-speed", type=float, default=1.0)
    parser.add_argument("--join-wait", type=float, default=25.0)
    parser.add_argument("--hold", type=float, default=8.0, help="seconds to keep filming after")
    parser.add_argument("--record", type=Path, help="mp4 path; omit to only watch")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--monitor", type=int, default=0, help="ddagrab output index")
    parser.add_argument(
        "--exact-stepping",
        action="store_true",
        help="pause the world between decisions, as measured runs do (stutters on screen)",
    )
    parser.add_argument(
        "--overlay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="draw each action above the character and a panel of recent ones",
    )
    parser.add_argument(
        "--zoom", type=float, default=DEFAULT_OVERLAY_ZOOM, help="spectator camera zoom"
    )
    args = parser.parse_args()

    program = next(p for p in load_programs_jsonl(args.programs) if p["run"] == args.run)
    build = sandbox.load(program["code"])
    task = get(args.task)
    engine = resolve_engine_config()
    manager = WorkerManager()
    handle = manager.launch(f"watch-program-{args.run}")
    client = recorder = None
    try:
        with RCONClient(handle.spec.rcon_endpoint, timeout=30.0) as rcon:
            rcon.lua(f"game.speed = {args.game_speed} return game.speed")
        address = f"127.0.0.1:{handle.spec.ports.game}"
        client = launch_client(engine.executable, address, handle.spec.mod_directory)
        session = WorkerSession(handle, timeout=120.0)
        session.status()
        if not args.exact_stepping:
            # Measured runs pause the world between decisions (`tick_paused`), which
            # on screen is a stutter every 30 ticks. A program decides in
            # milliseconds, so letting the world run between them looks continuous.
            session.configure(free_running=True)
        if args.overlay:
            warning = enable_overlay(
                session._client, title=f"{args.run} - {args.task}", zoom=args.zoom
            )
            if warning:
                print(warning, flush=True)
        env = FactorioEnv(task, session, HOLDOUT_PLAN, branch=Branch.EVAL, split="test")
        penv = ParameterizedEnv(env, profile="v2")
        index = evaluate.HOLDOUT_START_INDEX + args.index
        observation, info = penv.reset(options={"scene_index": index})
        wait_for_client(args.join_wait)
        if args.record and client is not None:
            title = client_window_title(client.pid)
            print(f"recording monitor {args.monitor} ({title!r}) -> {args.record}", flush=True)
            recorder = start_recording(args.record, args.fps, args.monitor)
            time.sleep(2)
        started = time.perf_counter()
        backend = EngineBackend(penv, observation)
        result = play(build, backend, time_limit_s=ENGINE_TIME_LIMIT_S)
        print(
            f"{args.run} index {index} family {info.get('layout_family')}: "
            f"engine success {result.success}, plates {result.verified_output}, "
            f"{result.decisions} decisions, {time.perf_counter() - started:.0f}s",
            flush=True,
        )
        time.sleep(args.hold)
        session.close()
    finally:
        if recorder is not None:
            stop_recording(recorder)
        if client is not None:
            client.terminate()
        manager.cleanup(handle)
    return 0


if __name__ == "__main__":
    sys.exit(main())
