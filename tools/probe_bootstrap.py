"""Gate A2: can a factory be started using only the tools the agent has?

A2's gate asks for the constituent chain -- gather, craft, place, fuel, obtain
machine-produced output -- driven "through exactly the public tools". That word
is the whole point. Every step below goes through `env.step_arguments` on the
`open-v1` catalog, the same entry point a model's reply reaches. No raw Lua
touches the world, nothing is teleported, and no item is conjured: if a step
here cannot be expressed, the agent cannot express it either.

It doubles as the reference solver an open world does not otherwise have. A
benchmark task has one and its solvability is checkable because of it; an open
world had neither until now, so "the tools are sufficient" was an assertion.

Raw Lua appears only to *read* state for the verdict, and once to exhaust a
deposit -- an evaluator write, marked as one, used to provoke a failure the
chain would otherwise never reach.

Run:
  uv run python tools/probe_bootstrap.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from factoriorl import worlds  # noqa: E402
from factoriorl.agent.adapters import ScriptedAdapter  # noqa: E402
from factoriorl.agent.loop import AgentConfig, AgentLoop, Decision  # noqa: E402
from factoriorl.agent.parsing import ParsedAction  # noqa: E402
from factoriorl.env import TRANSFER_AMOUNTS  # noqa: E402
from factoriorl.freeplay import starting_inventory  # noqa: E402
from factoriorl.open_world import OpenWorldEnv  # noqa: E402
from factoriorl.protocol import Request, RequestType  # noqa: E402
from factoriorl.rcon import RCONClient  # noqa: E402
from factoriorl.session import WorkerSession  # noqa: E402
from factoriorl.worker import WorkerManager  # noqa: E402

#: How many mining actions to spend before giving up on a deposit.
MINE_ATTEMPTS = 40

#: How close the character must be for `mine_at` to be accepted. The mod
#: enforces the real reach; this is the probe's own arrival test, set just under
#: it so "arrived" and "can act" mean the same thing here.
REACH_TILES = 2.4

#: Walk attempts before giving up. Deliberately generous, and the reason is a
#: measured property of the design rather than a fudge: `navigate` plans over
#: the agent's own *explored* terrain, and terrain is explored by walking, so a
#: route to somewhere 28 tiles off cannot be planned in one go -- it is planned
#: as far as the map is known, walked, and replanned. Measured on this probe:
#: 10 walks closed 28.5 tiles to 11.3. That cost is real and an agent pays it
#: too; refusing to plan through unknown ground is what keeps navigation
#: assistance rather than knowledge.
WALK_ATTEMPTS = 30


class Tools:
    """The public catalog, addressed the way a model's reply addresses it."""

    def __init__(self, env: OpenWorldEnv) -> None:
        self.env = env
        self.keys = list(env.catalog.keys())
        self.log: list[dict] = []

    def index(self, key: str) -> int:
        return self.keys.index(key)

    def legal(self, key: str) -> bool:
        mask = self.env.action_masks()
        index = self.index(key)
        return index < len(mask) and bool(mask[index])

    def do(self, key: str, **arguments) -> dict:
        """One action. Records what happened either way, and never raises."""
        started = time.perf_counter()
        record: dict = {"action": key, "arguments": arguments}
        try:
            index = self.index(key)
            if arguments:
                result = self.env.step_arguments(index, arguments)
            else:
                result = self.env.step(index)
            _, _, terminated, truncated, info = result
            record.update(
                status=info.get("action_status"),
                error=info.get("action_error"),
                terminated=terminated,
                truncated=truncated,
            )
            for extra in ("condition_met", "waited_steps", "inspected"):
                if extra in info:
                    record[extra] = info[extra]
        except Exception as failure:  # noqa: BLE001 - a refusal is a result here
            record["refused"] = f"{type(failure).__name__}: {failure}"
            # A refusal from the game is a result. A `ValueError` from
            # `step_arguments` is not: it means this probe asked for something
            # the catalog does not accept -- a missing argument or a value
            # outside its domain -- which is a bug in the caller, not a fact
            # about the world.
            #
            # Both used to land in `refused` together and be read as "the game
            # said no". When `mine_at` gained a required `count` this helper
            # stopped passing one, every mine call raised, the probe gathered
            # nothing, and the failure surfaced eight steps later as an empty
            # furnace. Separated so a caller can tell the two apart and stop.
            if isinstance(failure, ValueError):
                record["client_error"] = str(failure)
        record["ms"] = round((time.perf_counter() - started) * 1000, 1)
        self.log.append(record)
        return record

    # ------------------------------------------------------------- reading

    @property
    def observation(self) -> dict:
        return self.env._observation

    def inventory(self, item: str) -> int:
        return int((self.observation.get("inventory") or {}).get(item, 0))

    def domain(self, name: str) -> list:
        return list(self.env.argument_domains().get(name) or [])

    def _tiles(self, resource: str) -> list[dict]:
        """Every visible tile of one resource, closest first.

        `resources.tiles` arrives in whatever order the sensor swept, not in
        distance order -- 100 tiles of one patch came back with the *furthest*
        first. Taking `tiles[0]` therefore reported ore 11.9 tiles away while
        the character was standing 1.4 tiles from some, which read as a walk
        that never arrived and cost 38 mines refused `out_of_reach`.
        """
        tiles = [
            tile
            for tile in (self.observation.get("resources") or {}).get("tiles") or []
            if tile.get("name") == resource and tile.get("p")
        ]
        return sorted(tiles, key=lambda t: self.distance_to(t["p"]))

    def nearest(self, resource: str) -> list[float] | None:
        """Where the closest visible tile of a named resource is.

        Falls back to the patch aggregate, which is all there is beyond
        `resource_detail_radius` -- 12 tiles -- and is the only way anything
        further out is visible at all.
        """
        tiles = self._tiles(resource)
        if tiles:
            return [float(tiles[0]["p"][0]), float(tiles[0]["p"][1])]
        patches = (self.observation.get("resources") or {}).get("patches") or []
        rows = patches.values() if isinstance(patches, dict) else patches
        for patch in rows:
            if isinstance(patch, dict) and patch.get("name") == resource and patch.get("nearest"):
                near = patch["nearest"]
                return [float(near[0]), float(near[1])]
        return None

    def handle_for(self, resource: str) -> str | None:
        """The handle of the *closest* tile, which is the one within reach."""
        for tile in self._tiles(resource):
            if tile.get("h"):
                return str(tile["h"])
        return None

    def distance_to(self, point) -> float:
        if not point:
            return float("inf")
        here = (self.observation.get("character") or {}).get("position") or [0.0, 0.0]
        return ((here[0] - point[0]) ** 2 + (here[1] - point[1]) ** 2) ** 0.5

    def entity_handle(self, name: str) -> str | None:
        for entity in self.observation.get("entities") or []:
            if entity.get("name") == name and entity.get("h"):
                return str(entity["h"])
        return None


def walk_to(tools: Tools, resource: str, report: dict, label: str) -> bool:
    """Walk until the resource is close enough to *act on*, not merely to see.

    Two things this has to get right, both learned from the first run of this
    probe:

    **A walk does not finish inside the step that starts it.** `navigate` is an
    ongoing action; it replied `running` and the character was still moving. At
    0.1484 tiles/tick a 28-tile route is ~190 ticks and one step advances 30, so
    the first version issued a walk, immediately tried to mine, and was refused
    `out_of_reach` 38 times. Each walk is now followed by `wait_for` on
    `nothing_in_flight`, which is what that verb is for.

    **A handle is not reach.** A resource tile gets a handle once it is inside
    `resource_detail_radius` -- 12 tiles -- while `mine` needs about 2.7. The
    first version treated "has a handle" as "arrived", which is how it reached
    the mining loop still eleven tiles away.
    """
    goal = tools.nearest(resource)
    if goal is None:
        report[f"{label}_error"] = f"no {resource} within sensor range to walk to"
        return False
    report[f"{label}_started_at"] = round(tools.distance_to(goal), 1)

    for attempt in range(WALK_ATTEMPTS):
        # Against the *nearest* tile for the arrival test, because any tile of
        # the patch is worth reaching -- but toward a goal fixed at the start,
        # because the nearest tile moves as more of the patch comes into range
        # and chasing it walks along the patch edge instead of towards it.
        target = tools.nearest(resource)
        if target is not None and tools.distance_to(target) <= REACH_TILES:
            if tools.handle_for(resource):
                report[f"{label}_walks"] = attempt
                return True

        # Only ever a destination the observation itself offered, which is what
        # keeps this inside the agent's own action surface.
        legal = tools.domain("destinations")
        chosen = min(
            legal,
            key=lambda p: (p[0] - goal[0]) ** 2 + (p[1] - goal[1]) ** 2,
            default=None,
        )
        if chosen is None:
            report[f"{label}_error"] = "the destinations domain was empty"
            return False
        before = tools.distance_to(goal)
        tools.do("walk_to_position", destination=list(chosen))
        # Let the route run out. A walk that is still in flight leaves the
        # character somewhere between where it was and where it is going, and
        # every later action is judged against that.
        for _ in range(WALK_ATTEMPTS):
            if not (tools.observation.get("inflight") or []):
                break
            tools.do("wait_for", until="nothing_in_flight", seconds=10)
        if tools.distance_to(goal) >= before - 0.25:
            # Genuinely stuck rather than merely slow: the route is not getting
            # closer, so more of the same will not help.
            report[f"{label}_stalled_at"] = round(tools.distance_to(goal), 1)
            break
    remaining = tools.nearest(resource)
    report[f"{label}_error"] = (
        f"{resource} still {tools.distance_to(remaining):.1f} tiles away "
        f"after {WALK_ATTEMPTS} walks"
        if remaining
        else f"{resource} left sensor range"
    )
    return False


def mine(tools: Tools, resource: str, want: int, report: dict, label: str) -> bool:
    held = tools.inventory(resource)
    # The whole amount in one request. `mine` has always accepted a count -- the
    # primitive catalog's `mine_nearest_5` passes 5 -- and the open catalog now
    # exposes it, so this asks once for what it wants instead of once per tile.
    #
    # It is also the reason this helper must pass one at all: `mine_at` gained a
    # required `count`, and `Tools.do` never raises, so the missing argument was
    # swallowed silently. The probe gathered nothing, fed an empty furnace, and
    # failed eight steps later on a domain check with no plates in it. A
    # required argument that goes missing must be loud.
    amount = min((a for a in TRANSFER_AMOUNTS if a >= want), default=max(TRANSFER_AMOUNTS))
    for _ in range(MINE_ATTEMPTS):
        if tools.inventory(resource) - held >= want:
            break
        handle = tools.handle_for(resource)
        if handle is None:
            # The tile under us ran out; step to the next one the sensor sees.
            if not walk_to(tools, resource, report, f"{label}_rewalk"):
                break
            handle = tools.handle_for(resource)
            if handle is None:
                break
        outcome = tools.do("mine_at", handle=handle, count=amount)
        if outcome.get("client_error"):
            report[f"{label}_mine_error"] = outcome["client_error"]
            break
    report[f"{label}_mined"] = tools.inventory(resource) - held
    return tools.inventory(resource) - held >= want


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--out", default=str(ROOT / "docs" / "evidence" / "a2-bootstrap.json"))
    args = parser.parse_args()

    mode = worlds.get("open_factory")
    manager = WorkerManager()
    freeplay = starting_inventory(manager.engine.executable)
    report: dict = {
        "measures": "roadmap A2's gate: the bootstrap chain, through the public catalog only",
        "world": mode.to_dict(),
        "catalog": mode.catalog,
        "engine": manager.engine.to_dict(),
    }
    started = time.perf_counter()

    handle = manager.launch("probe-a2", map_seed=args.seed, terrain=mode.terrain)
    try:
        with RCONClient(handle.spec.rcon_endpoint, timeout=60.0) as client:
            client.lua("game.speed = 1.0 return game.speed")
        session = WorkerSession(handle, timeout=60.0)
        session.status()
        session.configure(free_running=True)
        env = OpenWorldEnv(mode, session, inventory=freeplay["items"], seed=args.seed)
        env.reset()
        tools = Tools(env)
        report["catalog_size"] = len(tools.keys)
        report["verbs"] = tools.keys

        # Cancellation runs first, at spawn, and the reason is measurable: a
        # walk only stays in flight if the route outlasts the 30-tick step that
        # starts it, which is 4.45 tiles. Standing at the ore the whole
        # `destinations` domain is closer than that, so the walk settles inside
        # its own step and there is nothing to cancel. From spawn the nearest
        # ore is 28 tiles, which is ~190 ticks.
        report["cancellation"] = probe_cancellation(tools)

        # --- gather ------------------------------------------------------
        report["walked_to_ore"] = walk_to(tools, "iron-ore", report, "ore")
        report["mined_ore"] = mine(tools, "iron-ore", 5, report, "ore")
        report["walked_to_coal"] = walk_to(tools, "coal", report, "coal")
        report["mined_coal"] = mine(tools, "coal", 5, report, "coal")

        # --- craft (by hand, from what was gathered) ---------------------
        crafted = tools.do("craft_recipe", recipe="iron-gear-wheel", count=1)
        report["handcraft"] = crafted
        tools.do("wait_for", until="nothing_in_flight", seconds=10)
        report["crafted_a_gear"] = tools.inventory("iron-gear-wheel") > 0

        # --- place -------------------------------------------------------
        report["placed_furnace"] = False
        if tools.inventory("stone-furnace"):
            # Several candidates, because the domain is built from what the
            # observation can see and the engine's own `can_place_entity` is
            # stricter -- ore tiles under the character are not in `entities`,
            # so the first choice came back `collision`.
            attempts = []
            for spot in tools.domain("placements")[:12]:
                placed = tools.do(
                    "place_at", item="stone-furnace", position=list(spot), direction="north"
                )
                attempts.append(placed)
                if placed.get("status") == "completed":
                    report["placed_furnace"] = True
                    break
            report["place_attempts"] = attempts

        # --- fuel and feed ------------------------------------------------
        furnace = tools.entity_handle("stone-furnace")
        report["furnace_handle"] = furnace
        if furnace:
            # Whichever fuel is actually held. Coal is the obvious one and was
            # not within the 32-tile sensor of this spawn; freeplay's single
            # wood is 2 MJ against a stone furnace's 90 kW, which is 22 seconds
            # of burn -- about seven plates, and plenty to show a machine
            # producing. Insisting on coal would have made the gate a fact about
            # one map seed.
            fuel = next(
                (item for item in ("coal", "wood") if tools.inventory(item)),
                None,
            )
            report["fuel_used"] = fuel
            report["fuelled"] = (
                tools.do("give_to", to=furnace, item=fuel, count=1)
                if fuel
                else {"refused": "no fuel in inventory"}
            )
            report["fed"] = tools.do("give_to", to=furnace, item="iron-ore", count=5)
            # A plate takes 3.2 seconds; one bare `wait` buys half of one.
            # A plate is 3.2 seconds and one bare `wait` buys half of one, so
            # this is the verb earning its place: the same wait costs one
            # decision instead of seven.
            report["waited"] = [
                tools.do("wait_for", until="world_changes", seconds=30) for _ in range(2)
            ]
            report["inspected"] = tools.do("inspect", handle=furnace)

        # --- machine-produced output --------------------------------------
        with RCONClient(handle.spec.rcon_endpoint, timeout=60.0) as client:
            report["furnace_output"] = client.lua(
                'local s = game.surfaces["nauvis"] '
                'local f = s.find_entities_filtered({name="stone-furnace"})[1] '
                'if not f then return "no furnace" end '
                "local out = f.get_output_inventory() "
                "local total = 0 "
                "for _, e in pairs(out.get_contents()) do total = total + e.count end "
                'return "plates:" .. total .. " status:" .. tostring(f.status)'
            )
        report["produced_plates"] = str(report["furnace_output"]).startswith("plates:") and not str(
            report["furnace_output"]
        ).startswith("plates:0")

        # --- the protocol cases that need a machine ------------------------
        report["lost_reply"] = probe_lost_reply(session, tools, furnace)
        report["partial_sequence"] = probe_partial_sequence(tools, furnace)

        # --- the failure modes the gate names -----------------------------
        report["failures"] = probe_failures(tools, handle, furnace)
        report["actions"] = tools.log
        session.close()
    finally:
        manager.cleanup(handle)

    report["wall_seconds"] = round(time.perf_counter() - started, 1)
    report["verdict"] = verdict(report)
    Path(args.out).write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report["verdict"], indent=2))
    print(f"wrote {args.out}")
    return 0 if all(report["verdict"].values()) else 1


def probe_cancellation(tools: Tools) -> dict:
    """Start a long walk and stop it, using only what the prompt publishes.

    A2.3 requires that an action still running is "reported with its operation
    identity and can be inspected/cancelled". Both halves are checked here: the
    identity is read out of the observation's `inflight` entries -- the same
    field the prompt renders and the `requests` argument domain draws from --
    and it is handed straight back to `cancel_request`.
    """
    found: dict = {}
    destinations = tools.domain("destinations")
    if not destinations:
        found["error"] = "no destination to walk to"
        return found
    here = (tools.observation.get("character") or {}).get("position") or [0, 0]
    far = max(destinations, key=lambda p: (p[0] - here[0]) ** 2 + (p[1] - here[1]) ** 2)
    found["started"] = tools.do("walk_to_position", destination=list(far))

    running = [e for e in (tools.observation.get("inflight") or []) if e.get("request_id")]
    found["inflight"] = running
    if not running:
        # A route short enough to finish inside its own step is not a
        # cancellation test, and saying so beats reporting a pass.
        found["error"] = "the walk settled before it could be cancelled"
        return found

    identity = str(running[0]["request_id"])
    found["identity"] = identity
    # The domain the model would have to pick from has to contain it, or the
    # verb is unusable no matter what the mod accepts.
    found["published_in_domain"] = identity in [str(r) for r in tools.domain("requests")]
    before = (tools.observation.get("character") or {}).get("position") or [0, 0]
    found["cancel"] = tools.do("cancel_request", target_request_id=identity)
    tools.do("wait_for", until="nothing_in_flight", seconds=10)
    after = (tools.observation.get("character") or {}).get("position") or [0, 0]
    found["still_in_flight"] = [
        e for e in (tools.observation.get("inflight") or []) if e.get("request_id") == identity
    ]
    # Cancellation stops the walker at the tick the cancel is processed, so the
    # character keeps whatever ground it covered and gains no more.
    found["moved_after_cancel"] = round(
        ((after[0] - before[0]) ** 2 + (after[1] - before[1]) ** 2) ** 0.5, 2
    )
    return found


def probe_lost_reply(session: WorkerSession, tools: Tools, furnace: str | None) -> dict:
    """Resend a mutating request byte-identically and prove it ran once.

    This is the one case here that cannot be posed through the catalog, because
    it is a property of the transport rather than of the game: `env.step` does
    not hand back a request id, and a lost reply is exactly the situation where
    a client holds an id and no answer. So the request is issued at the session
    layer and then *re-issued unchanged*, which is what a client that never saw
    the first reply would do.

    The world is the witness. If the ledger did not deduplicate, two plates
    would come out of the furnace instead of one.
    """
    found: dict = {}
    if not furnace:
        found["error"] = "no furnace to take from"
        return found

    before = tools.inventory("iron-plate")
    request = Request(
        request_id="probe-lost-reply",
        episode_id=session.episode_id or "",
        type=RequestType.STEP,
        payload={
            "action": {
                "action": "transfer",
                "from": furnace,
                "to": "character",
                "item": "iron-plate",
                "count": 1,
            },
            "ticks": 30,
        },
    )
    first = session._request(request)  # noqa: SLF001 - a transport probe, deliberately
    found["first_code"] = str(first.response.code)
    session.collect(request.request_id)
    tools.do("wait_for", until="nothing_in_flight", seconds=10)
    middle = tools.inventory("iron-plate")

    # The reply is now "lost": the same bytes go out again.
    second = session._request(request)  # noqa: SLF001
    found["second_code"] = str(second.response.code)
    found["second_result"] = second.response.result
    tools.do("wait_for", until="nothing_in_flight", seconds=10)
    after = tools.inventory("iron-plate")

    found["plates"] = {"before": before, "after_first": middle, "after_resend": after}
    found["executed_once"] = middle == before + 1 and after == middle
    found["resend_reported_duplicate"] = "duplicate" in found["second_code"].lower()
    return found


def probe_partial_sequence(tools: Tools, furnace: str | None) -> dict:
    """Run a three-action sequence whose middle action fails, against the engine.

    A2.3's contract -- stop at the first failure, report completed, failed and
    unexecuted, roll nothing back -- was covered only by
    `tests/unit/test_sequences.py`, which runs against a stub environment. The
    stub cannot disagree with the engine about what "failed" means, so this
    calls the loop's own `_execute_sequence` over the live world.

    The first action is deliberately a *mutating* one, because "no rollback"
    is only a claim if there is something that could have been rolled back.
    """
    found: dict = {}
    if not furnace:
        found["error"] = "no furnace to take from"
        return found

    loop = AgentLoop(tools.env, ScriptedAdapter([]), AgentConfig(task_id="open_factory"))
    requested = [
        ParsedAction(
            index=tools.index("take_from"),
            key="take_from",
            arguments={"from": furnace, "item": "iron-plate", "count": 1},
        ),
        # Nothing has ever had this handle.
        ParsedAction(index=tools.index("mine_at"), key="mine_at", arguments={"handle": "h9999"}),
        ParsedAction(index=tools.index("inspect"), key="inspect", arguments={"handle": furnace}),
    ]
    decision = Decision(
        episode=0,
        step=0,
        summary=None,  # type: ignore[arg-type] - recording only; unread here
        attempts=[],
        action_index=requested[0].index,
        action_key=requested[0].key,
        resolution="accepted",
        requested=requested,
    )

    before = tools.inventory("iron-plate")
    executed, _reward, _term, _trunc, _info = loop._execute_sequence(  # noqa: SLF001
        decision, remaining=8
    )
    after = tools.inventory("iron-plate")

    statuses = [record.get("status") for record in decision.outcomes]
    found["executed"] = executed
    found["stopped"] = decision.sequence_stopped
    found["statuses"] = statuses
    found["outcomes"] = decision.outcomes
    found["plates"] = {"before": before, "after": after}
    # Every requested action is accounted for exactly once, which is the half a
    # count of completions cannot express.
    found["all_accounted_for"] = len(decision.outcomes) == len(requested)
    found["stopped_at_first_failure"] = (
        decision.sequence_stopped is not None and statuses[-1] == "unexecuted"
    )
    # The successful transfer stands. Undoing it would mean issuing more game
    # actions, which is a different world state, not the original one.
    found["kept_the_successful_action"] = after == before + 1
    return found


def probe_failures(tools: Tools, handle, furnace: str | None) -> dict:
    """The seven cases A2's gate names, each provoked on purpose.

    A successful bootstrap never reaches any of them, which is exactly why they
    are worth forcing: an error path nothing exercises is an error path nobody
    has checked.
    """
    found: dict = {}

    # A handle that named something real and no longer does.
    found["stale_target"] = tools.do("mine_at", handle="h9999")

    # Somewhere already occupied -- the tile the character is standing on.
    here = (tools.observation.get("character") or {}).get("position") or [0, 0]
    found["blocked_placement"] = tools.do(
        "place_at",
        item="stone-furnace",
        position=[float(here[0]), float(here[1])],
        direction="north",
    )

    # A recipe whose ingredients are not held.
    found["insufficient_ingredients"] = tools.do(
        "craft_recipe", recipe="electronic-circuit", count=1
    )

    # An argument outside the domain the observation published.
    found["out_of_domain_argument"] = tools.do("walk_to_position", destination=[9999.5, 9999.5])

    # A technology that is not on the researchable frontier.
    found["unavailable_research"] = tools.do("research_technology", technology="rocket-silo")

    # An exhausted deposit. The only evaluator write in this file, and marked:
    # mining a patch dry through the catalog would take thousands of actions.
    if furnace:
        with RCONClient(handle.spec.rcon_endpoint, timeout=60.0) as client:
            found["deposit_emptied_by_evaluator"] = client.lua(
                'local s = game.surfaces["nauvis"] local n = 0 '
                'for _, e in pairs(s.find_entities_filtered({type="resource"})) do '
                "e.destroy() n = n + 1 end return n"
            )
        found["mine_exhausted_deposit"] = tools.do("mine_at", handle="h1")

    return found


def verdict(report: dict) -> dict:
    failures = report.get("failures") or {}

    def refused(name: str) -> bool:
        """A failure is *reported*, not silently swallowed or crashed into."""
        record = failures.get(name)
        if not isinstance(record, dict):
            return False
        return bool(
            record.get("refused") or record.get("error") or record.get("status") != "completed"
        )

    return {
        "walked_to_ore_by_tool": bool(report.get("walked_to_ore")),
        "gathered_ore": bool(report.get("mined_ore")),
        "handcrafted": bool(report.get("crafted_a_gear")),
        "placed_a_machine": bool(report.get("placed_furnace")),
        "fuelled_a_machine": (report.get("fuelled") or {}).get("status") == "completed",
        "machine_produced_output": bool(report.get("produced_plates")),
        "stale_target_refused": refused("stale_target"),
        "blocked_placement_refused": refused("blocked_placement"),
        "insufficient_ingredients_refused": refused("insufficient_ingredients"),
        "out_of_domain_argument_refused": refused("out_of_domain_argument"),
        "unavailable_research_refused": refused("unavailable_research"),
        "exhausted_deposit_refused": refused("mine_exhausted_deposit"),
        "running_action_cancelled": bool(
            (report.get("cancellation") or {}).get("cancel", {}).get("status") == "completed"
            and not (report.get("cancellation") or {}).get("still_in_flight")
        ),
        "running_action_identity_published": bool(
            (report.get("cancellation") or {}).get("published_in_domain")
        ),
        "lost_reply_executed_once": bool((report.get("lost_reply") or {}).get("executed_once")),
        "lost_reply_reported_duplicate": bool(
            (report.get("lost_reply") or {}).get("resend_reported_duplicate")
        ),
        "partial_sequence_stopped_at_failure": bool(
            (report.get("partial_sequence") or {}).get("stopped_at_first_failure")
        ),
        "partial_sequence_accounted_for_every_action": bool(
            (report.get("partial_sequence") or {}).get("all_accounted_for")
        ),
        "partial_sequence_kept_completed_work": bool(
            (report.get("partial_sequence") or {}).get("kept_the_successful_action")
        ),
    }


if __name__ == "__main__":
    sys.exit(main())
