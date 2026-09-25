"""The `v3` tensors and masks of every golden trace, for the simulator's contract test.

The golden traces (`docs/evidence/sim-parity`) were recorded under `local-v2`
and `parameterized-v1`, so their tensor hashes are the `v1` layout's. This
re-encodes every recorded decision under the `v3` layout and action profile
with FactorioRL's own code -- `encoders.encode(layout=LAYOUT_V3)`,
`FactorioEnv.marker_slots` and `ParameterizedEnv(profile="v3").action_masks`
-- and writes one hash set and one flat mask per decision. factory-sim steps
the same recorded vectors and must produce the same `v3` tensors and masks bit
for bit (`tests/test_rl_contract.py`).

**What stands in for the engine.** `local-v3` adds belt lanes and shape, the
inserter's pickup, drop and hand, and a drill's drop point to each visible
record (`sensor.logistics_detail`). The traces predate that profile, so those
fields are rebuilt here from the engine state each record also carries
(`hidden`), by the mapping the Lua function applies:

- belt: `lanes` = the number of items on each of `hidden.lanes`,
  `shape` = `hidden.belt_shape`;
- inserter: `pickup` / `drop` = `hidden.pickup_position` / `drop_position`
  (1/256 tiles), `held` = `hidden.held.name`;
- burner drill: `drop` = its centre plus the engine-measured offset for its
  facing (`tasks.potentials.DROP_OFFSETS`; `hidden` does not carry it).

The sweep keeps 48 entities under `local-v2` and 96 under `local-v3`; no trace
has more than 48 in view, so the recorded sweep is the `local-v3` sweep too
(checked below). Re-recording the traces under `local-v3` on the engine
replaces this bridge with the sensor's own output; the hashes must not move.

    uv run python tools/v3_contract_golden.py
    uv run python tools/v3_contract_golden.py --copy-to ../factory-sim/tests/golden
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from factoriorl import catalog as catalog_module  # noqa: E402
from factoriorl import encoders  # noqa: E402
from factoriorl.env import PLACEMENT_RADIUS_V3, FactorioEnv  # noqa: E402
from factoriorl.parameterized import ParameterizedEnv  # noqa: E402
from factoriorl.tasks import get  # noqa: E402
from factoriorl.tasks.potentials import DROP_OFFSETS  # noqa: E402

EVIDENCE = ROOT / "docs" / "evidence" / "sim-parity"
OUT = EVIDENCE / "v3_contract.json.xz"
VERSION = 1
#: `local-v2`'s `entity_cap`; a recorded sweep this full may have clipped what
#: `local-v3` would have shown.
LOCAL_V2_CAP = 48


def read_trace(path: Path) -> tuple[dict, list[dict]]:
    lines = lzma.decompress(path.read_bytes()).decode().splitlines()
    return json.loads(lines[0]), [json.loads(line) for line in lines[1:]]


def tensor_hashes(encoded: dict) -> dict:
    """`record_parity_trace._tensor_hashes`, restated so this runs without the engine stack."""
    out = {}
    for key in sorted(encoded):
        array = np.ascontiguousarray(encoded[key])
        digest = hashlib.sha256()
        digest.update(f"{array.dtype.str}|{array.shape}|".encode())
        digest.update(array.tobytes())
        out[key] = digest.hexdigest()[:16]
    return out


def _tiles(point) -> list[float]:
    return [point[0] / 256, point[1] / 256]


def logistics_fields(hidden: dict) -> dict:
    """What `sensor.logistics_detail` would add to this entity's record."""
    name = hidden.get("name")
    out: dict = {}
    if name == "transport-belt":
        lanes = hidden.get("lanes") or [[], []]
        out["lanes"] = [len(lanes[0]), len(lanes[1])]
        if hidden.get("belt_shape"):
            out["shape"] = hidden["belt_shape"]
    elif name == "burner-inserter":
        if hidden.get("pickup_position"):
            out["pickup"] = _tiles(hidden["pickup_position"])
        if hidden.get("drop_position"):
            out["drop"] = _tiles(hidden["drop_position"])
        held = hidden.get("held")
        if held and held.get("name"):
            out["held"] = held["name"]
    elif name == "burner-mining-drill":
        dx, dy = DROP_OFFSETS[int(hidden.get("direction", 0))]
        x, y = _tiles(hidden["position"])
        out["drop"] = [x + dx, y + dy]
    return out


def bridge(observation: dict, hidden: dict) -> dict:
    """The recorded observation with each visible record's `local-v3` fields."""
    by_place = {
        (e.get("name"), tuple(e.get("position") or ())): e for e in hidden.get("entities") or []
    }
    entities = []
    for record in observation.get("entities") or []:
        key = (record.get("name"), (round(record["p"][0] * 256), round(record["p"][1] * 256)))
        engine = by_place.get(key)
        entities.append({**record, **logistics_fields(engine)} if engine else dict(record))
    return {**observation, "entities": entities}


def v3_env(task_id: str) -> tuple[FactorioEnv, ParameterizedEnv]:
    """A `FactorioEnv` that answers from a set observation, under the `v3` catalog."""
    env = FactorioEnv(get(task_id), session=None, seed_plan=None)
    env.catalog = catalog_module.resolve("parameterized-v3", env.spec_.catalog_subset)
    env.v3 = True
    env.layout = encoders.LAYOUT_V3
    env.placement_radius = PLACEMENT_RADIUS_V3
    return env, ParameterizedEnv(env, profile="v3")


def scenario(name: str, entry: dict) -> list[dict]:
    header, records = read_trace(EVIDENCE / entry["trace"])
    env, penv = v3_env(header["task"])
    public = tuple(header["blueprint"].get("public_markers") or ())
    if public != tuple(env.spec_.public_markers):
        raise SystemExit(f"{name}: blueprint markers {public} != spec {env.spec_.public_markers}")
    out = []
    for record in records:
        observation = record["observation"]
        if len(observation.get("entities") or []) >= LOCAL_V2_CAP:
            raise SystemExit(f"{name} decision {record['decision']}: the sweep is at its cap")
        observation = bridge(observation, record.get("hidden") or {})
        env._observation = observation
        goal = np.concatenate(
            [np.asarray(record["goal"], dtype=np.float32), env.marker_slots(observation)]
        )
        encoded = encoders.encode(observation, goal, layout=encoders.LAYOUT_V3)
        mask = "".join("1" if b else "0" for b in penv.action_masks())
        out.append(
            {"decision": record["decision"], "tensors": tensor_hashes(encoded), "mask": mask}
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", help="scenarios (default: every one in the index)")
    parser.add_argument("--copy-to", type=Path, help="also copy the file into this directory")
    args = parser.parse_args()
    index = json.loads((EVIDENCE / "index.json").read_text(encoding="utf-8"))
    names = args.names or sorted(index)
    document = {
        "version": VERSION,
        "about": "v3 tensor hashes and masks of every golden trace (tools/v3_contract_golden.py)",
        "layout": {
            "entities": [encoders.MAX_ENTITIES_V3, encoders.ENTITY_FEATURES_V3],
            "items": list(encoders.ITEMS_V3),
            "goal": encoders.GOAL_FEATURES_V3,
        },
        "catalog_digest": catalog_module.resolve("parameterized-v3").digest(),
        "scenarios": {},
    }
    for name in names:
        document["scenarios"][name] = scenario(name, index[name])
        print(f"ok {name}: {len(document['scenarios'][name])} decisions")
    payload = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    OUT.write_bytes(lzma.compress(payload, preset=9))
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
    if args.copy_to:
        shutil.copyfile(OUT, args.copy_to / OUT.name)
        print(f"copied to {args.copy_to / OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
