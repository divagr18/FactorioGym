"""Record golden traces a simulator must reproduce, decision by decision (M2).

A fast simulator is only worth training on if it agrees with Factorio, and
agreement has to be checked one decision at a time: a single item of difference
changes which actions are legal, and from there two backends compare different
episodes. So each scenario here is played on the engine through the policy's
own interface -- `parameterized-v1`, the `MultiDiscrete` action space -- and
every decision is written down:

- the action, as the vector a policy emits and as the key and arguments it
  decodes to;
- the wire observation, a hash of every encoded tensor, the flat action mask
  and the goal vector;
- reward, termination, success, the reward components and any verification;
- the evaluator truth the reward and goal read;
- a digest of the world, and the exact hidden state (`world_digest` with
  `hidden`): machine progress, stored energy, slot layouts, the character's
  mining and walking state, ground items, resource amounts, the handle registry
  and the operations still running. A simulator in one-step sync mode loads
  that state and steps once; in free-running mode it compares against it.

**Determinism is checked, not assumed.** Each scenario is recorded, then played
a second time from a fresh episode by feeding the recorded *vectors* back
through `ParameterizedEnv.step` -- the path a simulator-trained checkpoint takes.
The two traces must hash identically. That proves the engine side is
reproducible and that the vectors alone are enough to replay it, which is what a
simulator will be handed.

What differs between two recordings for reasons that are not the world is
normalised before hashing, and nothing else: the episode id, the absolute game
tick (every tick is made episode-relative), and session request ids (renamed in
order of first appearance). Wall-clock time is kept out of the trace.

Run (needs the engine):
  uv run python tools/record_parity_trace.py --all
  uv run python tools/record_parity_trace.py drill_furnace_facings walk_and_reach
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import lzma
import math
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from factoriorl import encoders  # noqa: E402
from factoriorl.engine_config import resolve_game_speed  # noqa: E402
from factoriorl.env import FactorioEnv  # noqa: E402
from factoriorl.manifest import mod_source_digest  # noqa: E402
from factoriorl.parameterized import DIMENSIONS, ParameterizedEnv, sample_masked  # noqa: E402
from factoriorl.rcon import RCONClient  # noqa: E402
from factoriorl.seeding import Branch, SeedPlan  # noqa: E402
from factoriorl.session import WorkerSession  # noqa: E402
from factoriorl.tasks import get  # noqa: E402
from factoriorl.tasks.reference import Driver, SolveTrace, _build_at  # noqa: E402
from factoriorl.tasks.reference import solve as reference_solve  # noqa: E402
from factoriorl.tasks.spec import Blueprint, ResourceSpec  # noqa: E402
from factoriorl.worker import WorkerManager  # noqa: E402

OUT_DIR = ROOT / "docs" / "evidence" / "sim-parity"
MASTER_SEED = 20260915

#: Session request ids carry a per-session random prefix ("step-07d5db43-7"), and
#: the part a step issues is suffixed (":act").
REQUEST_ID = re.compile(r"^[a-z_]+-[0-9a-f]{8}-\d+(:[a-z_]+)?$")

#: Truth the reward, the goal vector and the verifier read. `window` is left out
#: because it is derived from `produced` per decision, which is recorded;
#: `production_note` and `scenario` are prose and bookkeeping.
TRUTH_KEYS = (
    "tick",
    "markers",
    "containers",
    "working",
    "produced",
    "working_counts",
    "placed_counts",
    "stored_energy",
    "remaining_burning_fuel",
    "built",
    "machine_produced",
    "handcrafted",
    "mined_by_hand",
    "by_hand_source",
    "verification",
)

#: Hidden-state fields that are lists. The engine's JSON encoder writes an empty
#: table as `{}`, so an empty list arrives as an object; these are made lists
#: again so an empty scene and a simulator's `[]` hash the same.
HIDDEN_LISTS = ("entities", "ground_items", "resources")


#: Keys under which the mod publishes an absolute `game.tick` in an observation.
ABSOLUTE_TICKS = frozenset(
    {
        "started_tick",
        "deadline_tick",
        "cancelled_at_tick",
        "observed_tick",
        "last_seen",
        "last_tick",
        "seen",
    }
)


class EpisodeOver(Exception):
    """A scenario asked for a decision after the episode ended."""


# ---------------------------------------------------------------- normalising


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _listify(value):
    return [] if value == {} else value


def _tensor_hashes(encoded: dict) -> dict:
    out = {}
    for key in sorted(encoded):
        array = np.ascontiguousarray(encoded[key])
        digest = hashlib.sha256()
        digest.update(f"{array.dtype.str}|{array.shape}|".encode())
        digest.update(array.tobytes())
        out[key] = digest.hexdigest()[:16]
    return out


def _request_order(request_id: str) -> tuple[int, str]:
    """`step-07d5db43-12:act` -> (12, ":act")."""
    head, _, suffix = request_id.partition(":")
    return int(head.rsplit("-", 1)[1]), suffix


class Normaliser:
    """Removes what differs between two recordings of the same world."""

    def __init__(self) -> None:
        self.base_tick = 0
        self.request_ids: dict[str, str] = {}

    def begin(self, observation: dict) -> None:
        self.base_tick = int(observation.get("absolute_tick") or 0) - int(
            observation.get("tick") or 0
        )

    def claim(self, *values) -> None:
        """Name every request id in `values` that has no name yet.

        In the order the session issued them -- the counter in the id, then the
        suffix -- not in the order a walk over the record meets them. A walk
        follows each dict's key order, which is the engine's JSON order and not
        something another backend reproduces; the counter order is the same in
        any session that issued the same requests in the same order.
        """
        found: set[str] = set()

        def walk(value):
            if isinstance(value, str) and REQUEST_ID.match(value):
                found.add(value)
            elif isinstance(value, dict):
                for item in value.values():
                    walk(item)
            elif isinstance(value, list):
                for item in value:
                    walk(item)

        for value in values:
            walk(value)
        for value in sorted(found - set(self.request_ids), key=_request_order):
            self.request_ids[value] = f"r{len(self.request_ids) + 1}"

    def _rename(self, value):
        if isinstance(value, str) and REQUEST_ID.match(value):
            if value not in self.request_ids:
                self.claim(value)
            return self.request_ids[value]
        if isinstance(value, dict):
            return {k: self._rename(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._rename(v) for v in value]
        return value

    def _relative(self, tick):
        return None if tick is None else int(tick) - self.base_tick

    def _ticks(self, value, key=None):
        """Engine ticks made episode-relative, wherever the mod publishes one.

        Most published ticks are already relative (`tick`, an event's `tick`);
        a few are `game.tick` as-is -- an operation's `started_tick`, a
        cancellation's `cancelled_at_tick`, memory's `last_seen`. They are
        recognised by name. A relative tick wrongly shifted here would differ
        between the recording and its replay, which start at different engine
        ticks, so the replay comparison is what checks this list.
        """
        if isinstance(value, dict):
            return {k: self._ticks(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [self._ticks(v, key) for v in value]
        if key in ABSOLUTE_TICKS and isinstance(value, int) and not isinstance(value, bool):
            return self._relative(value)
        return value

    def observation(self, observation: dict) -> dict:
        body = {k: v for k, v in observation.items() if k not in ("episode_id", "absolute_tick")}
        return self._rename(self._ticks(body))

    def truth(self, truth: dict) -> dict:
        body = {k: truth[k] for k in TRUTH_KEYS if k in truth}
        if "tick" in body:
            body["tick"] = self._relative(body["tick"])
        return self._rename(body)

    def hidden(self, hidden: dict) -> dict:
        body = dict(hidden)
        body["tick"] = self._relative(body.get("tick"))
        for key in HIDDEN_LISTS:
            body[key] = _listify(body.get(key) or [])
        for record in body["entities"]:
            # A mapping by inventory name, so an entity with none stays `{}`.
            record["inventories"] = record.get("inventories") or {}
            for inventory in record["inventories"].values():
                inventory["stacks"] = _listify(inventory.get("stacks") or [])
        character = body.get("character")
        if character and character.get("main"):
            character["main"]["stacks"] = _listify(character["main"].get("stacks") or [])
        handles = dict(body.get("handles") or {})
        order = []
        for entry in _listify(handles.get("order") or []):
            entry = dict(entry)
            entry["first_seen"] = self._relative(entry.get("first_seen"))
            if entry.get("destroyed_tick") is not None:
                entry["destroyed_tick"] = self._relative(entry["destroyed_tick"])
            order.append(entry)
        handles["order"] = order
        body["handles"] = handles
        inflight = dict(body.get("inflight") or {})
        entries = []
        for entry in _listify(inflight.get("entries") or []):
            entry = dict(entry)
            entry["started_tick"] = self._relative(entry.get("started_tick"))
            if entry.get("deadline_tick") is not None:
                entry["deadline_tick"] = self._relative(entry["deadline_tick"])
            entries.append(entry)
        inflight["entries"] = entries
        body["inflight"] = inflight
        return self._rename(body)


# ---------------------------------------------------------------- recording


class Recorder:
    """Hooks a `FactorioEnv` so every decision, however it is issued, is recorded.

    The hook sits on the instance's `step` and `step_arguments`, which is where
    a reference solver, a scripted scenario and `ParameterizedEnv.step` all
    arrive. `advance` -- the verifier's action-locked window -- goes around
    them, so it is recorded as part of the decision that triggered it, which is
    also what a policy experiences.
    """

    def __init__(self, env: FactorioEnv, session: WorkerSession) -> None:
        self.env = env
        self.penv = ParameterizedEnv(env)
        self.session = session
        self.normaliser = Normaliser()
        self.records: list[dict] = []
        #: Set by a driver that chose a vector itself (a random rollout, a
        #: replay), so the recorded vector is the one the policy emitted even
        #: when it decoded to a no-op.
        self.pending_vector: list[int] | None = None
        self._install_hooks()

    def _install_hooks(self) -> None:
        original_step = self.env.step
        original_arguments = self.env.step_arguments

        def step(action):
            return self._decide(int(action), {}, lambda: original_step(action))

        def step_arguments(action, arguments):
            return self._decide(
                int(action), dict(arguments), lambda: original_arguments(action, arguments)
            )

        self.env.step = step
        self.env.step_arguments = step_arguments

    def begin(self, encoded: dict) -> None:
        self.normaliser.begin(self.env._observation)
        self.records.append(self._state(encoded, transition=None))

    def _decide(self, operation: int, arguments: dict, call: Callable):
        key = self.env.catalog.templates[operation].key
        action: dict = {"key": key, "arguments": arguments, "decode_failure": None}
        if self.pending_vector is not None:
            vector = [int(v) for v in self.pending_vector]
            decoded_operation, _, failure = self.penv.decode(vector)
            action["vector"] = vector
            action["decode_failure"] = failure
            if failure is None and decoded_operation != operation:
                action["decode_failure"] = f"decoded {decoded_operation}, stepped {operation}"
        else:
            try:
                action["vector"] = [int(v) for v in self.penv.encode(operation, arguments)]
            except ValueError as exc:
                # An action no policy could take. Recorded, counted in the
                # index, and replayed by key rather than silently substituted.
                action["vector"] = None
                action["unencodable"] = str(exc)
        encoded, reward, terminated, truncated, info = call()
        self.pending_vector = None
        transition = {
            "action": action,
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "success": bool(info.get("success")),
            "action_status": info.get("action_status"),
            "action_error": info.get("action_error"),
            "reward_components": {
                k: float(v) for k, v in sorted((info.get("reward_components") or {}).items())
            },
        }
        if info.get("verification") is not None:
            transition["verification"] = info["verification"]
        self.records.append(self._state(encoded, transition))
        return encoded, reward, terminated, truncated, info

    def _state(self, encoded: dict, transition: dict | None) -> dict:
        normaliser = self.normaliser
        digest = self.session.world_digest(hidden=True).response.result or {}
        lines = digest.get("lines") or []
        mask = self.penv.action_masks()
        return {
            "decision": len(self.records),
            "transition": transition,
            "tick": int(self.env._observation.get("tick") or 0),
            "observation": (
                normaliser.claim(self.env._observation, self.env._truth, digest)
                or normaliser.observation(self.env._observation)
            ),
            "tensors": _tensor_hashes(encoded),
            "mask": "".join("1" if bit else "0" for bit in mask),
            "goal": [float(v) for v in self.env._goal_vector()],
            "truth": normaliser.truth(self.env._truth),
            "digest": hashlib.sha256("\n".join(lines).encode()).hexdigest()[:16],
            "hidden": normaliser.hidden(digest.get("hidden") or {}),
        }


def trace_hash(header: dict, records: list[dict]) -> str:
    digest = hashlib.sha256()
    digest.update(_canonical(header).encode())
    for record in records:
        digest.update(b"\n")
        digest.update(_canonical(record).encode())
    return digest.hexdigest()


def first_difference(a, b, path="") -> str | None:
    if type(a) is not type(b):
        return f"{path}: {type(a).__name__} != {type(b).__name__}"
    if isinstance(a, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                return f"{path}.{key}: present on one side only"
            found = first_difference(a[key], b[key], f"{path}.{key}")
            if found:
                return found
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} != {len(b)}"
        for index, (x, y) in enumerate(zip(a, b, strict=True)):
            found = first_difference(x, y, f"{path}[{index}]")
            if found:
                return found
        return None
    return None if a == b else f"{path}: {a!r} != {b!r}"


# ---------------------------------------------------------------- scripting


class Script:
    """A scenario's hands: catalog actions by key, and world lookups by tile.

    Every argument handed to an action is a value the policy's own domain holds
    at that decision, looked up rather than computed, so each scripted action
    has a vector.
    """

    def __init__(self, env: FactorioEnv) -> None:
        self.env = env
        self.keys = env.catalog.keys()
        self.trace = SolveTrace(task=env.spec_.id, budget=env.spec_.max_decision_steps)
        self.driver = Driver(env, self.trace)
        self.over = False
        self.log: list[dict] = []

    # ---- actions
    def do(self, key: str, **arguments) -> dict:
        if self.over:
            raise EpisodeOver(key)
        operation = self.keys.index(key)
        if arguments:
            result = self.env.step_arguments(operation, arguments)
        else:
            result = self.env.step(operation)
        _, _, terminated, truncated, info = result
        self.over = bool(terminated or truncated)
        self.log.append(
            {"key": key, "status": info.get("action_status"), "error": info.get("action_error")}
        )
        return info

    def wait(self, decisions: int) -> None:
        for _ in range(decisions):
            if self.over:
                return
            self.do("wait")

    def place(self, item: str, centre: tuple[int, int], direction: str) -> dict:
        """Place a 2x2 machine so it snaps to integer `centre`.

        A requested tile centre `t + 0.5` snaps a 2x2 entity to `t + 1`
        (`families/build_line.py`), so the tile to ask for is `centre - 1`.
        """
        tile = (centre[0] - 1, centre[1] - 1)
        return self.do("place_at", item=item, position=self.placement(tile), direction=direction)

    def give(self, handle: str, item: str, count: int) -> dict:
        return self.do("give_to", to=handle, item=item, count=count)

    def take(self, handle: str, item: str, count: int) -> dict:
        return self.do("take_from", **{"from": handle}, item=item, count=count)

    # ---- lookups
    @property
    def observation(self) -> dict:
        return self.env._observation

    def placement(self, tile: tuple[int, int]) -> list[float]:
        position = self.driver.placement_for(tile)
        if position is None:
            raise RuntimeError(f"tile {tile} is not an offered placement")
        return position

    def entity(self, name: str, centre: tuple[float, float]) -> str:
        records = [e for e in self.observation.get("entities") or [] if e.get("name") == name]
        if not records:
            raise RuntimeError(f"no {name} is observable")
        best = min(records, key=lambda r: math.dist(r["p"], centre))
        return str(best["h"])

    def tile_handle(self, tile: tuple[int, int]) -> str | None:
        for record in (self.observation.get("resources") or {}).get("tiles") or []:
            p = record.get("p")
            if p and (math.floor(p[0]), math.floor(p[1])) == tile:
                return str(record["h"])
        return None

    def held(self, item: str) -> int:
        return int((self.observation.get("inventory") or {}).get(item, 0))


# ---------------------------------------------------------------- scenes


def _patch_centre(blueprint: Blueprint) -> tuple[int, int]:
    cx, cy = blueprint.markers["patch"]
    return (math.floor(cx), math.floor(cy))


def stand_on_patch(inventory: dict[str, int]) -> Callable[[Blueprint], Blueprint]:
    """The character on the patch centre's tile, holding `inventory`."""

    def edit(blueprint: Blueprint) -> Blueprint:
        px, py = _patch_centre(blueprint)
        return dataclasses.replace(
            blueprint,
            character_position=(px + 0.5, py + 0.5),
            character_inventory=dict(inventory),
        )

    return edit


def small_deposits(blueprint: Blueprint) -> Blueprint:
    """Three tiles of 1, 2 and 3 ore beside the start, and nothing else to mine."""
    sx = math.floor(blueprint.character_position[0])
    sy = math.floor(blueprint.character_position[1])
    return dataclasses.replace(
        blueprint,
        character_position=(sx + 0.5, sy + 0.5),
        resources=(
            ResourceSpec("iron-ore", (float(sx + 1), float(sy)), amount=2),
            ResourceSpec("iron-ore", (float(sx + 1), float(sy + 1)), amount=1),
            ResourceSpec("iron-ore", (float(sx + 2), float(sy)), amount=3),
        ),
    )


def east_of_the_wall(blueprint: Blueprint) -> Blueprint:
    """`obstructed_patch` puts a five-tile wall at x = 5; start four tiles east of it."""
    return dataclasses.replace(blueprint, character_position=(9.3, 0.4))


# ---------------------------------------------------------------- scenarios


def walk_and_reach(s: Script) -> None:
    # Into the wall: a full stride stops short at the collision boundary, and a
    # second makes no progress at all.
    s.do("move_west")
    s.do("move_west")
    s.do("step_west")
    # Along its face, then round the north end.
    s.do("nudge_north")
    s.do("nudge_north")
    s.do("step_north")
    s.do("step_north")
    s.do("move_west")
    s.do("move_north")
    s.do("move_west")
    s.do("move_west")
    s.do("move_south")
    # Back east onto the patch's edge, from the far side of the wall.
    s.do("move_east")
    s.do("step_east")
    # Reach: resource reach is 2.7 tiles, so the nearest tile is mined and the
    # farthest visible one is refused.
    here = s.observation["character"]["position"]
    # Among the tiles a policy can address: the target dimension holds 32.
    addressable = set(s.env.argument_domains()["targets"][:32])
    tiles = [
        t
        for t in (s.observation.get("resources") or {}).get("tiles") or []
        if t.get("h") in addressable
    ]
    near = min(tiles, key=lambda t: math.dist(t["p"], here))
    far = max(tiles, key=lambda t: math.dist(t["p"], here))
    s.do("mine_at", handle=str(far["h"]))
    s.do("mine_at", handle=str(near["h"]))
    s.wait(5)


def hand_mine_exhaustion(s: Script) -> None:
    sx = math.floor(s.observation["character"]["position"][0])
    sy = math.floor(s.observation["character"]["position"][1])
    a, b, c = (sx + 1, sy), (sx + 1, sy + 1), (sx + 2, sy)
    # A second mine while the first is running, then walking away mid-mine.
    s.do("mine_at", handle=s.tile_handle(a))
    s.do("mine_at", handle=s.tile_handle(c))
    s.wait(5)
    s.do("mine_at", handle=s.tile_handle(a))
    s.do("nudge_south")
    s.wait(5)
    # Then every tile to exhaustion, one ore per `mine_at`, as a policy must.
    for tile in (a, b, c):
        for _ in range(8):
            handle = s.tile_handle(tile)
            if handle is None or s.over:
                break
            s.do("mine_at", handle=handle)
            s.wait(4)
    s.wait(2)


def placement_footprints(s: Script) -> None:
    px = math.floor(s.observation["character"]["position"][0])
    py = math.floor(s.observation["character"]["position"][1])
    # A furnace whose footprint covers the character's own tile: offered by the
    # domain, refused by the character's bounding box.
    s.place("stone-furnace", (px, py), "north")
    # Two drills on ore.
    s.place("burner-mining-drill", (px, py - 2), "north")
    s.place("burner-mining-drill", (px - 2, py), "west")
    # A furnace overlapping the first drill's footprint.
    s.place("stone-furnace", (px, py - 3), "north")
    # A drill half on the patch, and one wholly off it.
    s.place("burner-mining-drill", (px + 4, py + 1), "east")
    s.place("burner-mining-drill", (px + 5, py - 3), "east")
    # Rotation moves a drill's drop position.
    drill = s.entity("burner-mining-drill", (px, py - 2))
    s.do("rotate_at", handle=drill)
    s.do("rotate_at_reverse", handle=drill)
    s.do("rotate_at_reverse", handle=drill)
    # Picking an empty machine back up.
    s.do("mine_at", handle=s.entity("burner-mining-drill", (px - 2, py)))
    s.wait(3)


FACING_LINES = (
    # (drill centre, facing, furnace centre), as offsets from the patch centre.
    # Furnace offsets are the productive ones measured per facing in
    # `docs/evidence/section8-symmetry.json`.
    ((0, -2), "north", (0, -4)),
    ((3, 1), "east", (5, 1)),
    ((1, 3), "south", (1, 5)),
    ((-2, 0), "west", (-4, 0)),
)


def drill_furnace_facings(s: Script) -> None:
    px = math.floor(s.observation["character"]["position"][0])
    py = math.floor(s.observation["character"]["position"][1])
    for drill, facing, furnace in FACING_LINES:
        s.place("burner-mining-drill", (px + drill[0], py + drill[1]), facing)
        s.place("stone-furnace", (px + furnace[0], py + furnace[1]), "north")
    for drill, _, furnace in FACING_LINES:
        s.give(s.entity("burner-mining-drill", (px + drill[0], py + drill[1])), "coal", 5)
        s.give(s.entity("stone-furnace", (px + furnace[0], py + furnace[1])), "coal", 5)
    s.wait(60)


def drill_jam_and_build_over_pile(s: Script) -> None:
    px = math.floor(s.observation["character"]["position"][0])
    py = math.floor(s.observation["character"]["position"][1])
    s.place("burner-mining-drill", (px, py - 2), "north")
    s.give(s.entity("burner-mining-drill", (px, py - 2)), "coal", 5)
    # No furnace: the drill drops ore on the ground and stops on its own pile.
    s.wait(20)
    # Then build the furnace on the pile's tile.
    s.place("stone-furnace", (px, py - 4), "north")
    s.give(s.entity("stone-furnace", (px, py - 4)), "coal", 5)
    s.wait(20)


def burner_run_on(s: Script) -> None:
    px = math.floor(s.observation["character"]["position"][0])
    py = math.floor(s.observation["character"]["position"][1])
    s.place("burner-mining-drill", (px, py - 2), "north")
    s.place("stone-furnace", (px, py - 4), "north")
    drill = s.entity("burner-mining-drill", (px, py - 2))
    furnace = s.entity("stone-furnace", (px, py - 4))
    # One coal each: 4 MJ is 1,600 ticks of drill and about 2,670 of furnace,
    # and each keeps running on its stored buffer after the slot empties.
    s.give(drill, "coal", 1)
    s.give(furnace, "coal", 1)
    s.wait(95)
    s.give(drill, "coal", 5)
    s.give(furnace, "coal", 5)
    s.wait(15)


def transfer_clamping(s: Script) -> None:
    px = math.floor(s.observation["character"]["position"][0])
    py = math.floor(s.observation["character"]["position"][1])
    s.place("stone-furnace", (px + 2, py + 1), "north")
    furnace = s.entity("stone-furnace", (px + 2, py + 1))
    s.give(furnace, "iron-ore", 5)
    s.give(furnace, "iron-ore", 20)  # holds 5 more: clamped
    s.take(furnace, "iron-ore", 20)  # the furnace holds 10: clamped
    s.give(furnace, "iron-ore", 5)
    # A take during a hand-mine. The mine poller counts the character's ore, so
    # ore taken from the furnace while mining is what it sees.
    s.do("mine_at", handle=s.tile_handle((px, py + 1)) or s.tile_handle((px - 1, py)))
    s.take(furnace, "iron-ore", 5)
    s.wait(5)
    # Fuel it, let it smelt, and take more plates than it has.
    s.give(furnace, "coal", 20)  # holds 3: clamped
    s.give(furnace, "iron-ore", 5)
    s.wait(14)
    s.take(furnace, "iron-plate", 20)
    s.wait(2)


def mine_machine_returns_contents(s: Script) -> None:
    px = math.floor(s.observation["character"]["position"][0])
    py = math.floor(s.observation["character"]["position"][1])
    s.place("burner-mining-drill", (px, py - 2), "north")
    s.place("stone-furnace", (px, py - 4), "north")
    drill = s.entity("burner-mining-drill", (px, py - 2))
    furnace = s.entity("stone-furnace", (px, py - 4))
    s.give(drill, "coal", 5)
    s.give(furnace, "coal", 5)
    s.give(furnace, "iron-ore", 5)
    s.wait(16)
    # A furnace holding fuel, ore and plates, then a drill holding fuel.
    s.do("mine_at", handle=furnace)
    s.wait(3)
    s.do("mine_at", handle=drill)
    s.wait(3)


def build_line_reference(s: Script) -> None:
    reference_solve(s.env)


def construct_smelting_line_reference(s: Script) -> None:
    """The reference line, then the budget run out, so the verifier runs on its own.

    Not `reference.solve`, which calls `run_verification` directly: an RL
    episode reaches verification by truncating, and that is the path a
    simulator has to reproduce.
    """
    for x, y in sorted(s.driver.resource_tiles()):
        if _build_at(s.driver, (x + 1, y + 1)) or s.driver.terminated or s.driver.truncated:
            break
    s.over = s.driver.terminated or s.driver.truncated
    while not s.over:
        s.do("wait")


def construct_smelting_line_exploit(s: Script) -> None:
    """`tools/probe_verification_provenance.py`'s exploit arm: a furnace, no drill,
    hand-mined ore fed on the last construction decisions."""
    driver, env = s.driver, s.env
    ore = driver.nearest_resource()
    if ore is None or not driver.walk_to(ore, tolerance=1.0):
        raise RuntimeError(f"could not reach ore: {s.trace.stuck_reason}")
    furnace = None
    for position in env.argument_domains()["placements"]:
        s.trace.last_error = None
        s.do("place_at", item="stone-furnace", position=list(position), direction="north")
        if env._observation.get("entities") and driver.entities_of_type("furnace"):
            furnace = str(driver.entities_of_type("furnace")[0]["h"])
            break
    if furnace is None:
        raise RuntimeError("no offered tile accepted a furnace")
    s.give(furnace, "coal", 20)
    while s.held("iron-ore") < 14:
        if env._observation.get("inflight"):
            s.do("wait")
            continue
        here = env._observation["character"]["position"]
        tiles = [
            t
            for t in (env._observation.get("resources") or {}).get("tiles") or []
            if t.get("h") and t.get("name") == "iron-ore"
        ]
        s.do("mine_at", handle=str(min(tiles, key=lambda t: math.dist(t["p"], here))["h"]))
    last_tick = env.construction_tick_limit - 3 * env.spec_.decision_ticks
    last_step = env.spec_.max_decision_steps - 3
    while env._steps < last_step and int(env._observation.get("tick") or 0) < last_tick:
        s.do("wait")
    s.give(furnace, "iron-ore", 20)
    while not s.over:
        s.do("wait")


def masked_random_rollout(s: Script, recorder: Recorder, decisions: int = 200) -> None:
    """Uniform over legal values per dimension: an untrained policy's first actions.

    The rest are what a script would think to do; this is what a policy
    actually does, decode failures included.
    """
    rng = np.random.default_rng(MASTER_SEED)
    penv = recorder.penv
    for _ in range(decisions):
        if s.over:
            return
        vector = sample_masked(penv.action_space, penv.action_masks(), rng)
        recorder.pending_vector = [int(v) for v in vector]
        _, _, terminated, truncated, _ = penv.step(vector)
        s.over = bool(terminated or truncated)


@dataclass(frozen=True)
class Scenario:
    name: str
    task: str
    split: str
    episode_index: int
    about: str
    drive: Callable
    edit: Callable[[Blueprint], Blueprint] | None = None
    wants_recorder: bool = False


SMELTING = "construct_smelting_line"

SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        "walk_and_reach",
        SMELTING,
        "test",
        0,
        "strides into a wall, along its face and round it; resource reach",
        walk_and_reach,
        edit=east_of_the_wall,
    ),
    Scenario(
        "hand_mine_exhaustion",
        SMELTING,
        "train",
        0,
        "hand-mining 121-tick ore, a second mine mid-mine, walking away, tiles to zero",
        hand_mine_exhaustion,
        edit=small_deposits,
    ),
    Scenario(
        "placement_footprints",
        SMELTING,
        "train",
        0,
        "2x2 snapping, refusals for the character's box, overlap and bare ground, rotation",
        placement_footprints,
        edit=stand_on_patch({"burner-mining-drill": 6, "stone-furnace": 4, "coal": 10}),
    ),
    Scenario(
        "drill_furnace_facings",
        SMELTING,
        "train",
        0,
        "a drill feeding a furnace in each of four facings, fuelled and run",
        drill_furnace_facings,
        edit=stand_on_patch({"burner-mining-drill": 4, "stone-furnace": 4, "coal": 40}),
    ),
    Scenario(
        "drill_jam_and_build_over_pile",
        SMELTING,
        "train",
        0,
        "a drill stopped by its own ground pile, then a furnace built on the pile",
        drill_jam_and_build_over_pile,
        edit=stand_on_patch({"burner-mining-drill": 1, "stone-furnace": 1, "coal": 10}),
    ),
    Scenario(
        "burner_run_on",
        SMELTING,
        "train",
        0,
        "one coal each: slots empty, stored energy runs on, machines stop, refuel",
        burner_run_on,
        edit=stand_on_patch({"burner-mining-drill": 1, "stone-furnace": 1, "coal": 12}),
    ),
    Scenario(
        "transfer_clamping",
        SMELTING,
        "train",
        0,
        "gives and takes clamped to what is there, and a take during a hand-mine",
        transfer_clamping,
        edit=stand_on_patch({"stone-furnace": 1, "coal": 3, "iron-ore": 10}),
    ),
    Scenario(
        "mine_machine_returns_contents",
        SMELTING,
        "train",
        0,
        "picking up a working furnace and drill returns them with their contents",
        mine_machine_returns_contents,
        edit=stand_on_patch(
            {"burner-mining-drill": 1, "stone-furnace": 1, "coal": 20, "iron-ore": 5}
        ),
    ),
    Scenario(
        "build_line_reference",
        "build_line",
        "train",
        0,
        "the build_line reference solver, to success",
        build_line_reference,
    ),
    Scenario(
        "construct_smelting_line_reference",
        SMELTING,
        "train",
        0,
        "the reference line, then the budget run out into automatic verification",
        construct_smelting_line_reference,
    ),
    Scenario(
        "construct_smelting_line_exploit",
        SMELTING,
        "train",
        0,
        "a hand-fed furnace and no drill, which verification must score zero",
        construct_smelting_line_exploit,
    ),
    Scenario(
        "masked_random_rollout",
        "build_line",
        "train",
        1,
        "200 uniformly random legal vectors, decode failures included",
        masked_random_rollout,
        wants_recorder=True,
    ),
)


# ---------------------------------------------------------------- running


def make_env(scenario: Scenario, session: WorkerSession, task=None) -> tuple[FactorioEnv, dict]:
    """A fresh env whose next reset installs the scenario's scene."""
    env = FactorioEnv(
        task or get(scenario.task),
        session,
        SeedPlan(master=MASTER_SEED, run_id=f"sim-parity-{scenario.name}"),
        branch=Branch.TRAIN,
        split=scenario.split,
    )
    installed: dict = {}

    def prepare_scene(episode_index: int) -> str:
        families = env._families()
        rng = env.seed_plan.generator_rng(env.branch, episode_index)
        env._family = families[rng.randrange(len(families))]
        blueprint = env.task.generate(env._family, rng)
        if scenario.edit is not None:
            blueprint = scenario.edit(blueprint)
        installed["payload"] = blueprint.to_dict(
            public_markers=env.spec_.public_markers,
            extra_tracked_items=env.spec_.extra_tracked_items,
        )
        installed["family"] = env._family.name
        return env._install(blueprint)

    env.prepare_scene = prepare_scene
    return env, installed


def header_for(scenario: Scenario, env: FactorioEnv, recorder: Recorder, installed: dict) -> dict:
    spec = env.spec_
    return {
        "scenario": scenario.name,
        "about": scenario.about,
        "task": spec.id,
        "task_version": spec.version,
        "split": scenario.split,
        "episode_index": scenario.episode_index,
        "layout_family": installed["family"],
        "blueprint": installed["payload"],
        "decision_ticks": spec.decision_ticks,
        "max_decision_steps": spec.max_decision_steps,
        "max_game_ticks": spec.max_game_ticks,
        "construction_tick_limit": env.construction_tick_limit,
        "observation_profile": spec.observation_profile,
        "action_profile": spec.action_profile,
        "catalog": spec.catalog,
        "catalog_digest": env.catalog.digest(),
        "catalog_keys": list(env.catalog.keys()),
        "verification": spec.verification.to_dict() if spec.verification else None,
        "action_space": {
            "dimensions": [name for name, _ in DIMENSIONS],
            "nvec": [int(n) for n in recorder.penv.action_space.nvec],
            "items": list(encoders.ITEMS),
        },
    }


def record(scenario: Scenario, session: WorkerSession, replay: list[dict] | None = None):
    env, installed = make_env(scenario, session)
    recorder = Recorder(env, session)
    encoded, _ = env.reset(options={"scene_index": scenario.episode_index})
    recorder.begin(encoded)
    script = Script(env)
    error = None
    try:
        if replay is None:
            if scenario.wants_recorder:
                scenario.drive(script, recorder)
            else:
                scenario.drive(script)
        else:
            for action in replay:
                if script.over:
                    break
                if action["vector"] is not None:
                    recorder.pending_vector = action["vector"]
                    _, _, terminated, truncated, _ = recorder.penv.step(
                        np.asarray(action["vector"], dtype=np.int64)
                    )
                    script.over = bool(terminated or truncated)
                else:
                    script.do(action["key"], **action["arguments"])
    except EpisodeOver as exc:
        error = f"scenario acted after the episode ended ({exc})"
    header = header_for(scenario, env, recorder, installed)
    return header, recorder.records, error


#: Scenarios replayed tick by tick with `--tick-resolution`: the mechanics ones.
#: The three long ones are a 600-decision reference, an exploit and a random
#: rollout, whose mechanics these already cover, at 18,000 ticks apiece.
TICK_SCENARIOS = (
    "walk_and_reach",
    "hand_mine_exhaustion",
    "placement_footprints",
    "drill_furnace_facings",
    "drill_jam_and_build_over_pile",
    "burner_run_on",
    "transfer_clamping",
    "mine_machine_returns_contents",
)


def comparable_hidden(hidden: dict) -> dict:
    """Hidden state without what a one-tick replay changes by construction.

    The replay issues 29 extra `wait` steps per decision. They settle events and
    take in-flight sequence numbers, and so shift request renaming, but touch
    nothing in the world.
    """
    body = {k: v for k, v in hidden.items() if k not in ("event_seq",)}
    inflight = dict(body.get("inflight") or {})
    inflight.pop("next_seq", None)
    inflight["entries"] = [
        {k: v for k, v in entry.items() if k not in ("request_id", "seq")}
        for entry in inflight.get("entries") or []
        if entry.get("action") != "advance"
    ]
    body["inflight"] = inflight
    # A handle is minted when an observation first sees its entity, and the
    # replay observes every tick: a pile that appears at tick 273 is first
    # seen at 273 here and at 300 in the decision trace. Names and order must
    # still agree; the tick they were first seen need not.
    handles = dict(body.get("handles") or {})
    handles["order"] = [
        {k: v for k, v in entry.items() if k not in ("first_seen", "destroyed_tick")}
        for entry in handles.get("order") or []
    ]
    body["handles"] = handles
    return body


def record_ticks(scenario: Scenario, session: WorkerSession, actions: list[dict]):
    """Replay recorded actions one tick at a time, keeping hidden state per tick.

    Each decision becomes its action on a one-tick step, then `wait` steps to
    make up the decision's ticks. `wait` changes nothing in the world, so the
    state at every decision boundary must equal the decision trace's; the
    caller checks that before trusting the ticks in between.
    """
    base = get(scenario.task)
    ticks_per_decision = base.spec.decision_ticks
    spec = dataclasses.replace(base.spec, decision_ticks=1, max_decision_steps=10**9)
    env, installed = make_env(scenario, session, task=dataclasses.replace(base, spec=spec))
    env.reset(options={"scene_index": scenario.episode_index})
    normaliser = Normaliser()
    normaliser.begin(env._observation)

    def hidden() -> dict:
        digest = session.world_digest(hidden=True).response.result or {}
        normaliser.claim(digest)
        return normaliser.hidden(digest.get("hidden") or {})

    rows = [hidden()]
    script = Script(env)
    for action in actions:
        if script.over:
            break
        script.do(action["key"], **action["arguments"])
        rows.append(hidden())
        for _ in range(ticks_per_decision - 1):
            if script.over:
                break
            script.do("wait")
            rows.append(hidden())
    header = {
        "scenario": scenario.name,
        "task": base.spec.id,
        "task_version": base.spec.version,
        "ticks_per_decision": ticks_per_decision,
        "blueprint": installed["payload"],
    }
    return header, rows


def write_trace(path: Path, header: dict, records: list[dict]) -> None:
    """One JSON line for the header, then one per decision, xz-compressed.

    xz rather than gzip because consecutive decisions repeat most of the world
    -- the resource list, the handle registry -- at a distance past gzip's
    32 KB window: a 600-decision trace is 22 MB of JSON, 2 MB gzipped and 66 KB
    as xz. xz writes no timestamp, so the same world gives the same bytes.
    """
    lines = [_canonical(header)] + [_canonical(r) for r in records]
    payload = ("\n".join(lines) + "\n").encode()
    path.write_bytes(lzma.compress(payload, preset=9 | lzma.PRESET_EXTREME))


def read_trace(path: Path) -> tuple[dict, list[dict]]:
    lines = lzma.decompress(path.read_bytes()).decode().splitlines()
    return json.loads(lines[0]), [json.loads(line) for line in lines[1:]]


def summarise(header: dict, records: list[dict]) -> dict:
    transitions = [r["transition"] for r in records if r["transition"]]
    last = records[-1]
    statuses: dict[str, int] = {}
    for t in transitions:
        name = f"{t['action']['key']}:{t['action_status']}"
        if t["action_error"]:
            name += f":{t['action_error']}"
        statuses[name] = statuses.get(name, 0) + 1
    verification = next((t["verification"] for t in transitions if "verification" in t), None)
    return {
        "task": header["task"],
        "task_version": header["task_version"],
        "layout_family": header["layout_family"],
        "decisions": len(transitions),
        "final_tick": last["tick"],
        "return": round(sum(t["reward"] for t in transitions), 6),
        "success": bool(transitions and transitions[-1]["success"]),
        "terminated": bool(transitions and transitions[-1]["terminated"]),
        "truncated": bool(transitions and transitions[-1]["truncated"]),
        "unencodable_actions": sum(1 for t in transitions if t["action"].get("vector") is None),
        "decode_failures": sum(1 for t in transitions if t["action"].get("decode_failure")),
        "verification": verification
        and {
            k: verification.get(k)
            for k in ("machine_output", "uncapped_output", "machine_source", "success", "reward")
        },
        "action_outcomes": dict(sorted(statuses.items())),
        "final_entities": [
            {"name": e["name"], "status": e.get("status")}
            for e in last["hidden"]["entities"]
            if e.get("force") == "player"
        ],
        "final_ground_items": len(last["hidden"]["ground_items"]),
    }


def locate_difference(header, records, header2, records2) -> dict:
    """The first decision two traces disagree at, and the first field there."""
    at = next(
        (
            i
            for i, (a, b) in enumerate(zip(records, records2, strict=False))
            if _canonical(a) != _canonical(b)
        ),
        min(len(records), len(records2)),
    )
    if first_difference(header, header2):
        return {"decision": None, "path": "header" + first_difference(header, header2)}
    if at < min(len(records), len(records2)):
        return {"decision": at, "path": first_difference(records[at], records2[at])}
    return {"decision": at, "path": f"lengths {len(records)} != {len(records2)}"}


def tick_resolution(scenario: Scenario, session: WorkerSession, index: dict) -> int:
    """Record one scenario tick by tick, twice, and check it against its trace."""
    started = time.perf_counter()
    decision_header, decisions = read_trace(OUT_DIR / f"{scenario.name}.jsonl.xz")
    actions = [r["transition"]["action"] for r in decisions if r["transition"]]
    header, rows = record_ticks(scenario, session, actions)
    header2, rows2 = record_ticks(scenario, session, actions)
    first, second = trace_hash(header, rows), trace_hash(header2, rows2)
    step = header["ticks_per_decision"]
    mismatch = None
    for record in decisions:
        at = record["decision"] * step
        if at >= len(rows):
            break
        found = first_difference(comparable_hidden(record["hidden"]), comparable_hidden(rows[at]))
        if found:
            mismatch = {"decision": record["decision"], "tick": at, "path": found}
            break
    name = f"{scenario.name}.ticks.jsonl.xz"
    write_trace(OUT_DIR / name, header, rows)
    entry = index.setdefault(scenario.name, {})
    entry["ticks"] = {
        "trace": name,
        "trace_sha256": first,
        "ticks": len(rows) - 1,
        "repeat_identical": first == second,
        "matches_decision_trace": mismatch is None,
        "first_mismatch": mismatch,
        "wall_seconds": round(time.perf_counter() - started, 1),
    }
    print(
        f"{scenario.name:36} ticks={len(rows) - 1:5d} repeat={first == second} "
        f"matches_decisions={mismatch or True}",
        flush=True,
    )
    return 0 if first == second and mismatch is None else 1


def replay_note(entry: dict) -> str:
    if "replay_identical" not in entry:
        return "skipped"
    return "same" if entry["replay_identical"] else str(entry.get("replay_first_difference"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scenarios", nargs="*", help="scenario names (default: --all)")
    parser.add_argument("--all", action="store_true", help="record every scenario")
    parser.add_argument("--list", action="store_true", help="list scenarios and exit")
    parser.add_argument(
        "--no-replay", action="store_true", help="skip the second, vector-replayed recording"
    )
    parser.add_argument(
        "--tick-resolution",
        action="store_true",
        help="replay the mechanics scenarios one tick at a time and record hidden state",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="record and compare with the committed traces instead of writing them",
    )
    args = parser.parse_args()
    by_name = {s.name: s for s in SCENARIOS}
    if args.list:
        for s in SCENARIOS:
            print(f"{s.name:36} {s.task:26} {s.about}")
        return 0
    unknown = [n for n in args.scenarios if n not in by_name]
    if unknown:
        parser.error(f"unknown scenarios: {unknown}")
    chosen = SCENARIOS if args.all or not args.scenarios else [by_name[n] for n in args.scenarios]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index_path = OUT_DIR / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    manager = WorkerManager()
    handle = manager.launch("sim-parity")
    failures = 0
    try:
        with RCONClient(handle.spec.rcon_endpoint, timeout=30.0) as client:
            client.lua(f"game.speed = {resolve_game_speed()} return game.speed")
        session = WorkerSession(handle, timeout=120.0)
        session.status()
        engine = manager.engine.to_dict()
        if args.tick_resolution:
            names = args.scenarios or list(TICK_SCENARIOS)
            for name in names:
                failures += tick_resolution(by_name[name], session, index)
                index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", "utf-8")
            chosen = []
        for scenario in chosen:
            started = time.perf_counter()
            header, records, error = record(scenario, session)
            first = trace_hash(header, records)
            if args.check:
                # A fresh worker against the committed trace: the determinism
                # claim across processes, not just across two episodes.
                committed_header, committed = read_trace(OUT_DIR / f"{scenario.name}.jsonl.xz")
                same = trace_hash(committed_header, committed) == first
                failures += 0 if same and error is None else 1
                where = (
                    "same"
                    if same
                    else locate_difference(committed_header, committed, header, records)
                )
                print(f"{scenario.name:36} check={where}", flush=True)
                continue
            entry = {
                "about": scenario.about,
                "trace": f"{scenario.name}.jsonl.xz",
                "trace_sha256": first,
                "error": error,
                **summarise(header, records),
            }
            if not args.no_replay:
                replay_actions = [r["transition"]["action"] for r in records if r["transition"]]
                header2, records2, error2 = record(scenario, session, replay=replay_actions)
                second = trace_hash(header2, records2)
                entry["replay_sha256"] = second
                entry["replay_identical"] = first == second
                if first != second:
                    entry["replay_first_difference"] = locate_difference(
                        header, records, header2, records2
                    )
                if error2:
                    entry["replay_error"] = error2
            write_trace(OUT_DIR / entry["trace"], header, records)
            entry["wall_seconds"] = round(time.perf_counter() - started, 1)
            entry["mod_source_digest"] = mod_source_digest()
            entry["engine_build"] = engine.get("build") or engine.get("version")
            index[scenario.name] = entry
            ok = error is None and entry.get("replay_identical", True)
            failures += 0 if ok else 1
            print(
                f"{scenario.name:36} decisions={entry['decisions']:4d} "
                f"success={entry['success']!s:5} return={entry['return']:.3f} "
                f"unencodable={entry['unencodable_actions']} "
                f"replay={replay_note(entry)} "
                f"{entry['wall_seconds']}s",
                flush=True,
            )
            index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", "utf-8")
        session.close()
    finally:
        manager.cleanup(handle)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
