"""Run the language-model agent on a worker you can watch in a real client.

The worker is the ordinary one: `worker.py` launches the *graphical* Factorio
binary with `--start-server`, so every run this repo has ever done is already a
dedicated server bound to `127.0.0.1:<game port>`. Nothing about the engine
needs changing to see it -- a second Factorio process joins that server as a
normal multiplayer client and renders the same world the agent is driving.

This tool launches one worker, prints the address, optionally starts the client
itself with `--mp-connect`, and then hands the environment to `AgentLoop`.

Three things to know before reading anything off the screen
----------------------------------------------------------
**The world keeps running while the model thinks, by default.** The measured
environment does not: exact stepping is `game.tick_paused` plus a fixed
`decision_ticks` per action, so the world moves for 30 ticks and then freezes
for the one to several seconds an inference takes. That stutter is the
environment working correctly, but it is a bad thing to watch, and watching is
what this tool is for -- so it runs free by default and `--exact-stepping` puts
the pause back. `--game-speed` is forced to 1.0 either way, so motion plays at
real time; the throughput runs use whatever `engine_config` resolves, which is
faster and unwatchable.

**This used to be impossible, and the reason was one line in `rcon.py`.** With
a client connected every request came back with an **empty body** --
`non-json rcon response: ''`, turned into a `ProtocolError`, and the run died
before its first decision. Three hypotheses were tested and correctly ruled
out (the pause, the payload size, transience), and the conclusion drawn was
that a connected client breaks the RCON transport itself. It does not.

What nobody had read was the engine's own log. `WorkerSpec.console_log` is only
the *console* copy; the real log is `write-data/factorio-current.log`, and it
records the failure with no error at all -- a clean join and nothing else. The
console copy is where the answer was, in the command order right after the
join::

    [JOIN] Keno joined the game
    [COMMAND] <server> (command): rcon.print('p')
    [COMMAND] <server> (command): local _ = 1
    [COMMAND] <server> (command): local _ = 1      <- a sentinel, ahead of the
    [COMMAND] <server> (command): local ok, result = pcall(...)  command it was
                                                              meant to follow

`RCONClient.command` sends a sentinel after every command and treated its
arrival as proof the real reply was complete, because "a Source RCON server
answers in order on one connection". That is true only with no players. Attach
a client and the order inverts for anything longer than a trivial command, so
the sentinel was read as an empty response and the real reply arrived during
the *next* call, where its lower id marked it stale and it was drained. One
lost reply per call, forever. Completion is now "both replies seen", in either
order, and `RCONClient.reordered_replies` counts the inversions -- 15 of them
across a six-step probe.

None of the three ruled-out hypotheses was wrong, and size in particular is
now measured rather than argued: with a client attached the same server
answered a 16 KB command and a 64 KB reply without complaint, and does not
split a reply at all (4 MB arrived in one packet).

**Exact stepping survives a client, which is why the default is a preference
rather than a workaround.** Verified with a client in the game: six steps
advanced the tick by exactly 30 each, `game.tick_paused` read `True` between
every one, and the world drifted 0 ticks across three idle seconds. So
`--exact-stepping` genuinely works with something watching; it is simply less
pleasant to watch than a world that never stops.

The one perturbation left is real and unfixable: joining creates a `LuaPlayer`
no measured run has. So nothing here is written to `docs/evidence/`; the run
artifact lands under `runtime/runs/` like any other agent run, and
`tools/replay.py` -- which perturbs nothing -- stays the way to look at a run
that *was* measured.

**Watch the prompt, not just the screen.** Every decision's exact prompt, the
model's reply and the outcome go to `decisions.jsonl` in the run directory
printed at the end. The screen shows what the world did; the log shows what the
model was told and what it asked for, which is where the interesting failures
are -- an agent that walks in circles and one that cannot see a machine's fuel
level look identical on screen. R4 found the second of those by reading a log
whose every entity line said `status 18` and nothing else.

Run:
  set -a; . ./.env; set +a
  uv run python tools/watch_agent.py --task plate_line --model gpt-5.6-luna --launch-client
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from factoriorl.agent.adapters import (  # noqa: E402
    ModelAdapter,
    ModelReply,
    ModelRequest,
    OpenAICompatibleAdapter,
)
from factoriorl.agent.loop import AgentConfig, AgentLoop  # noqa: E402
from factoriorl.agent.viewer import DEFAULT_OVERLAY_ZOOM, enable_overlay  # noqa: E402
from factoriorl.engine_config import resolve_engine_config  # noqa: E402
from factoriorl.env import FactorioEnv  # noqa: E402
from factoriorl.rcon import RCONClient  # noqa: E402
from factoriorl.seeding import Branch, SeedPlan  # noqa: E402
from factoriorl.session import WorkerSession  # noqa: E402
from factoriorl.tasks import get  # noqa: E402
from factoriorl.worker import WorkerManager  # noqa: E402

#: Tasks worth watching, and what you will see. `keep_line_running` is the one
#: with something to watch happen *to* the agent rather than only something the
#: agent does.
WATCHABLE = {
    "plate_line": "a built, unfuelled line; the agent has to reach both machines and fuel them",
    "keep_line_running": (
        "a line already producing, then a declared fuel outage at tick 3600 that "
        "nothing announces -- the agent has to notice and refuel"
    ),
    "diagnose_line": "a stopped line with no marker saying why; one machine is dry and the "
    "furnace's output slot is full",
    "build_line": "nothing built; the agent places the drill and the furnace itself",
    "deliver": "four identical chests, one of them scored",
    "repair_belt": "a belt with a hole in it, and the hole's coordinate is published",
}


#: There used to be a `PatientClient` here that retried any call answered with
#: an empty body, six times a second apart, on the theory that a connected
#: client made the server stall. It never once succeeded on a retry, which was
#: the clue: the reply was not late, it had already been delivered and dropped.
#: With the ordering fixed in `rcon.py` there is nothing to be patient about, so
#: the wrapper is gone rather than kept as a safety net -- a retry that masks an
#: empty answer would also mask a worker that had genuinely died, and a watched
#: run should fail as loudly as a measured one.


class Narrating(ModelAdapter):
    """Prints each decision as it happens, and otherwise is the inner adapter.

    A watch tool that says nothing until the run ends is half a tool: the
    screen shows what the world did and the terminal is where you can see what
    the model was told and what it asked for. `AgentLoop` has no per-decision
    hook, but every decision passes through exactly one adapter call, so the
    seam is here.

    A proxy rather than an edit to the loop, because the loop is what the
    measured runs use and it should not grow a printing mode for one tool. It
    forwards `describe` unchanged, so the run manifest records the real
    adapter's configuration and not this wrapper.
    """

    def __init__(self, inner: ModelAdapter) -> None:
        self.inner = inner
        self.name = inner.name
        self.calls = 0

    def complete(self, request: ModelRequest) -> ModelReply:
        # The rendered summary's own header, rather than a counter kept here:
        # this should report the tick the model was actually shown, which is
        # the whole subject of R4.1.
        header = ""
        for line in request.user.splitlines():
            if "game tick" in line:
                header = line.strip()
                break
        reply = self.inner.complete(request)
        self.calls += 1
        if reply.error:
            print(
                f"  [{self.calls:>3}] {header}\n        provider error: {reply.error[:160]}",
                flush=True,
            )
            return reply
        text = " ".join((reply.text or "").split())
        print(f"  [{self.calls:>3}] {header}\n        {text[:220]}", flush=True)
        return reply

    def describe(self) -> dict:
        return self.inner.describe()

    def redact(self, text: str) -> str:
        return self.inner.redact(text)


def launch_client(executable: Path, address: str, mod_directory: Path) -> subprocess.Popen | None:
    """Start a second Factorio as a multiplayer client, on the server's mods.

    `--mod-directory` is the whole trick, and it was found the hard way: a
    client started on the player's own mod set greets them with *"Your active
    mods don't match the server's. Do you want to synchronize?"*, and
    synchronizing is the wrong answer twice over. `factoriorl` is a local mod
    that `modpack.package_mod` writes into the worker's directory -- it is not
    on the mod portal, so a sync has nowhere to fetch it from -- and a sync
    rewrites the player's real active-mod list to match a throwaway server.

    Pointing the client at the worker's own mod directory makes the two sets
    identical by construction, so no dialog appears and the player's install is
    untouched. Read-only sharing of that directory while the server runs is
    fine; both processes only load from it.

    `--config` is deliberately left alone, so the client keeps the player's
    graphics settings and credentials. The worker gets an isolated config
    precisely so a *measured* run cannot depend on those; a watched run is not
    a measured run.
    """
    cmd = [str(executable), "--mp-connect", address, "--mod-directory", str(mod_directory)]
    try:
        return subprocess.Popen(cmd)
    except OSError as exc:
        print(f"could not start the client ({exc}); connect manually instead", flush=True)
        return None


def wait_for_client(seconds: float) -> None:
    """Give the client time to finish joining before the first decision.

    A join makes the server transfer the map and stall, and the first watched
    run hit that stall *inside* a step: the request raised an infrastructure
    failure, the episode was truncated at decision 3 and excluded from metrics,
    and the log said nothing about a client. Waiting moves the stall to before
    the first decision, where it costs nothing.

    A plain wait, not an RCON poll. The poll this replaced printed its query
    into the **server console**, which the watcher reads across the middle of
    their own screen -- five copies of
    `remote.call("frrl_bridge", "run", "return #game.connected_players")` were
    visible in the first screenshot of a successful join -- and it never
    returned a positive count even with a player plainly standing in the game.
    A tool for watching something should not scribble on the thing being
    watched.
    """
    print(f"  giving the client {seconds:.0f}s to join before the first decision", flush=True)
    time.sleep(seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", default="plate_line", choices=sorted(WATCHABLE))
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=120,
        help="hard ceiling on decisions, independent of the task budget: a paid "
        "provider makes an unbounded episode an unbounded bill",
    )
    parser.add_argument(
        "--game-speed",
        type=float,
        default=1.0,
        help="1.0 so each 30-tick burst plays at real time. The throughput runs "
        "use whatever engine_config resolves, which is faster and unwatchable",
    )
    parser.add_argument(
        "--exact-stepping",
        action="store_true",
        help="pause the world between decisions, the way a measured run does. "
        "Off by default here: a world that freezes for the several seconds the "
        "model spends thinking is a worse thing to watch than one that keeps "
        "running, and watching is the whole purpose of this tool. It is a "
        "choice and not a workaround -- exact stepping was measured working "
        "with a client in the game (six steps, exactly 30 ticks each, "
        "tick_paused true between every one, 0 ticks of idle drift over 3s), so "
        "pass this when you want to see the environment the benchmark actually "
        "measures. Free running **changes what happens**: the world keeps "
        "ticking while the model thinks, so an action lands at whatever tick it "
        "arrives at instead of exactly decision_ticks after the last one, which "
        "is why a run either way is a demonstration and never a measurement",
    )
    parser.add_argument(
        "--echo-commands",
        action="store_true",
        help="send Lua as `/c` instead of `/silent-command`, so the engine logs "
        "and broadcasts every command. Factorio prints a broadcast command "
        "verbatim to every connected player, and this repo's commands are a page "
        "of pcall/remote.call wrapper each -- so on a watched run that is the "
        "whole screen. Silent is the default; pass this when the point is to "
        "audit what the run asked the engine to do from the engine's own log",
    )
    parser.add_argument(
        "--launch-client",
        action="store_true",
        help="start a second Factorio and connect it. Without this the address "
        "is printed and you join through Multiplayer -> Connect to address",
    )
    parser.add_argument(
        "--wait-batch",
        type=int,
        default=0,
        help="repeat a chosen `wait` this many extra times without asking again. "
        "`plate_line` needs 30 plates, which is about 240 decisions of which "
        "some 200 are waiting for a furnace -- so without this a run measures "
        "the price of patience rather than the agent. It is an **assistance** "
        "and the manifest records it; it only ever repeats a `wait` the model "
        "just chose, so it cannot substitute for a decision the model did not "
        "make. It also makes a watched run far less tedious: the world advances "
        "in longer stretches between pauses",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="do not print each decision as it happens. The default prints the "
        "tick the model was shown and the reply it gave, which is the half of a "
        "watched run the screen cannot show",
    )
    parser.add_argument(
        "--client-warmup",
        type=float,
        default=30.0,
        help="seconds to wait after starting the client, before the first "
        "decision. The join stalls the server while it transfers the map, and a "
        "stall inside a step is an infrastructure failure, not a task outcome",
    )
    parser.add_argument(
        "--hold-open",
        type=float,
        default=60.0,
        help="seconds to keep the server alive after the run ends, so the final "
        "state stays on screen. The client is disconnected when it exits",
    )
    parser.add_argument(
        "--overlay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="draw each action above the character and keep a panel of the last "
        "few with their outcomes, in the watching client only. Drawing and GUI "
        "are not simulation, so what the agent does is unchanged either way",
    )
    parser.add_argument(
        "--zoom",
        type=float,
        default=DEFAULT_OVERLAY_ZOOM,
        help="spectator camera zoom, set once when the client joins (with --overlay)",
    )
    args = parser.parse_args()

    if not os.environ.get(args.api_key_env):
        print(f"{args.api_key_env} is not set in the environment")
        print("run:  set -a; . ./.env; set +a")
        return 2

    engine = resolve_engine_config()
    manager = WorkerManager()
    # Unique per run, not the bare "watch": the worker id names the write-data
    # directory, and Factorio takes an exclusive lock on it. Two watched runs
    # overlapping -- easy, because `--hold-open` deliberately keeps the first
    # server alive so the final state stays on screen -- failed the second one
    # at launch with "Couldn't create lock file ... Is another instance already
    # running?".
    handle = manager.launch(f"watch-{os.getpid()}")
    session = None
    client = None
    try:
        endpoint = handle.spec.rcon_endpoint
        address = f"127.0.0.1:{handle.spec.ports.game}"
        with RCONClient(endpoint, timeout=30.0) as rcon:
            rcon.lua(f"game.speed = {args.game_speed} return game.speed")

        print("\n" + "=" * 68, flush=True)
        print(f"  watching:  {args.task} -- {WATCHABLE[args.task]}", flush=True)
        print(f"  model:     {args.model}", flush=True)
        print(f"  connect:   Multiplayer -> Connect to address -> {address}", flush=True)
        print(
            "             (joining by hand? pass Factorio "
            f"--mod-directory {handle.spec.mod_directory} or it will ask to sync mods)",
            flush=True,
        )
        print(f"  engine:    {engine.executable}", flush=True)
        print("=" * 68, flush=True)
        print(
            "\nthe world keeps running while the model thinks, so what you see\n"
            "is continuous. That is --free-running, the default here and not\n"
            "what the benchmark measures: pass --exact-stepping for a paused\n"
            "world that advances exactly 30 ticks per decision instead.\n"
            "a joined client adds a LuaPlayer the measured runs do not have, so\n"
            "nothing here is written to docs/evidence.\n",
            flush=True,
        )

        session = WorkerSession(handle, timeout=60.0)
        # Set before the first request, so every command in the run -- scenario
        # install, reset, step, truth -- goes out the same way.
        session._client.silent = not args.echo_commands
        session.status()

        # Free running by default, because a world that freezes for every
        # second the model spends thinking is a poor thing to watch. Exact
        # stepping is available and was measured working with a client attached
        # -- see the module docstring -- so this is a preference, not a
        # workaround for a transport that cannot do the other thing.
        free_running = not args.exact_stepping
        if free_running:
            session.configure(free_running=True)
            print(
                "  FREE RUNNING: the world is not paused between decisions, so "
                "this run is a demonstration and not a measurement -- an action "
                "lands at whatever tick it arrives at",
                flush=True,
            )

        if args.overlay:
            warning = enable_overlay(
                session._client, title=f"{args.task} - {args.model}", zoom=args.zoom
            )
            if warning:
                print(f"  {warning}", flush=True)

        env = FactorioEnv(
            get(args.task),
            session,
            SeedPlan(master=int(time.time()) % 100000, run_id="watch"),
            branch=Branch.TRAIN,
            split=args.split,
        )
        # Install every scene the loop will use, *before* a client exists, so
        # the watcher joins the task rather than the world the worker booted
        # into -- the freeplay crash site with the two-chest `reference` scene
        # beside it.
        #
        # This used to carry a second, load-bearing reason: `scenario_define`
        # is the biggest request the protocol has -- for `plate_line` every one
        # of 49 ore tiles, about ten kilobytes -- and it was believed to be too
        # big to survive a connected client. It is not. With a client in the
        # game the same server answered a 16 KB command and a 64 KB reply, and
        # `prepare_scene` itself was measured succeeding with the client
        # already joined. The install stays here for the first reason alone.
        #
        # `prepare_scene` installs without resetting, and `_install` caches by
        # digest. `FactorioEnv.reset` **increments** `_episode_index`, so the
        # index is rewound afterwards to leave the loop drawing episode 0.
        for index in range(args.episodes):
            env.prepare_scene(index)
        env._episode_index = -1
        env.reset()
        # Rewound, so the loop's first reset draws episode 0 again -- the scene
        # already installed and now on the watcher's screen.
        env._episode_index = -1
        print(
            f"  scene installed: {args.task}, {args.split} split, "
            f"{args.episodes} episode(s) pre-installed",
            flush=True,
        )

        if args.launch_client:
            client = launch_client(engine.executable, address, handle.spec.mod_directory)
            if client is not None:
                wait_for_client(args.client_warmup)
        provider = OpenAICompatibleAdapter(
            base_url=args.base_url,
            model=args.model,
            api_key_env=args.api_key_env,
            timeout=args.timeout,
        )
        adapter = provider if args.quiet else Narrating(provider)
        config = AgentConfig(
            task_id=args.task,
            episodes=args.episodes,
            split=args.split,
            max_steps=args.max_steps,
            wait_batch=args.wait_batch,
            run_prefix="watch",
        )
        loop = AgentLoop(
            env,
            adapter,
            config,
            provenance={
                "engine": handle.engine.to_dict(),
                "workers": [handle.spec.manifest()],
                "watched": True,
                "free_running": free_running,
                "note": (
                    "a human client was connected to this server, which creates a "
                    "LuaPlayer no measured run has. Demonstration only."
                ),
            },
        )
        result = loop.run()
        if session._client.reordered_replies:
            # Not a warning. It is the count of calls whose sentinel arrived
            # before the command's own reply, which is what a connected player
            # does to the reply order and what used to end the run. A non-zero
            # number here is the transport doing its job.
            print(
                f"\nthe sentinel overtook the real reply on "
                f"{session._client.reordered_replies} call(s) and both were "
                "waited for. That count is zero on every player-free run.",
                flush=True,
            )
        if provider.healed_parameters:
            # Not a footnote: dropping `temperature` means this run sampled at
            # the provider's default and is not reproducible from its seed.
            print(
                "\nthe provider rejected parameters and they were substituted: "
                + ", ".join(
                    f"{row['rejected']} -> {row['sent_instead']}"
                    for row in provider.healed_parameters
                ),
                flush=True,
            )
        print(
            "\n"
            + json.dumps(
                {
                    "run_dir": str(loop.run_dir),
                    "episodes": [
                        {
                            "episode": row.get("episode"),
                            "steps": row.get("steps"),
                            "success": row.get("success"),
                            "final_tick": row.get("final_tick"),
                            "production": (row.get("production") or {}).get("cumulative_produced"),
                        }
                        for row in (result.get("episodes") or [])
                    ],
                },
                indent=1,
            ),
            flush=True,
        )

        if args.hold_open > 0:
            print(f"\nholding the server open for {args.hold_open:.0f}s", flush=True)
            deadline = time.perf_counter() + args.hold_open
            while time.perf_counter() < deadline:
                time.sleep(1.0)
    finally:
        if session is not None:
            try:
                session.close()
            except OSError:
                pass
        if client is not None and client.poll() is None:
            client.terminate()
        manager.cleanup(handle)
    return 0


if __name__ == "__main__":
    sys.exit(main())
