"""The `v3` tensors and masks of every golden trace, for the simulator's contract test.

The golden traces (`docs/evidence/sim-parity`) are recorded under the task's
`local-v2` and `parameterized-v1`, so their tensor hashes are the `v1`
layout's. Each is also replayed on the engine under the `local-v3` sensor
(`tools/record_parity_trace.py --sensor`), which keeps every decision's wire
observation with what `local-v3` adds -- belt lanes and shape, an inserter's
pickup, drop and hand, a drill's drop point -- in `<name>.local-v3.jsonl.xz`,
and checks that everything else the trace holds comes out identical. This
encodes those observations under the `v3` layout and action profile with
FactorioRL's own code -- `encoders.encode(layout=LAYOUT_V3)`,
`FactorioEnv.marker_slots` and `ParameterizedEnv(profile="v3").action_masks`
-- and writes one hash set and one flat mask per decision. factory-sim steps
the same recorded vectors and must produce the same `v3` tensors and masks bit
for bit (`tests/test_rl_contract.py`).

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

EVIDENCE = ROOT / "docs" / "evidence" / "sim-parity"
OUT = EVIDENCE / "v3_contract.json.xz"
VERSION = 1


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


def trace_hash(header: dict, records: list[dict]) -> str:
    """`record_parity_trace.trace_hash`, restated for the same reason."""
    digest = hashlib.sha256()
    digest.update(_canonical(header).encode())
    for record in records:
        digest.update(b"\n")
        digest.update(_canonical(record).encode())
    return digest.hexdigest()


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sensor_observations(name: str, entry: dict) -> dict[int, dict]:
    """The trace's `local-v3` observations by decision, checked against the index."""
    sensor = entry.get("local_v3")
    if not sensor:
        raise SystemExit(f"{name}: no local-v3 observations; record_parity_trace.py --sensor")
    if not sensor.get("v1_fields_identical"):
        raise SystemExit(f"{name}: the local-v3 replay departs from the trace: {sensor}")
    header, rows = read_trace(EVIDENCE / sensor["trace"])
    if trace_hash(header, rows) != sensor["trace_sha256"]:
        raise SystemExit(f"{name}: {sensor['trace']} does not hash to the index")
    if header["source_trace_sha256"] != entry["trace_sha256"]:
        raise SystemExit(f"{name}: {sensor['trace']} replays another recording of the trace")
    return {row["decision"]: row["observation"] for row in rows}


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
    sensed = sensor_observations(name, entry)
    env, penv = v3_env(header["task"])
    public = tuple(header["blueprint"].get("public_markers") or ())
    if public != tuple(env.spec_.public_markers):
        raise SystemExit(f"{name}: blueprint markers {public} != spec {env.spec_.public_markers}")
    out = []
    for record in records:
        observation = sensed[record["decision"]]
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
