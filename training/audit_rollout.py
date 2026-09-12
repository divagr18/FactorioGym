"""Audit a schema-v2 learner rollout without loading a model or Factorio."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rollout_artifacts import RolloutArtifactError, audit_run


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps({"run": str(args.run), "audit": audit_run(args.run)}, sort_keys=True))
    except RolloutArtifactError as exc:
        raise SystemExit(f"rollout audit failed: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
