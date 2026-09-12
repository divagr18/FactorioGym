"""Run the authenticated Windows-side bridge for WSL language-model rollouts.

Example (on the Windows Factorio host):
    uv run python tools/agentic_bridge.py --token-env FACTORIORL_BRIDGE_TOKEN
"""

from __future__ import annotations

import argparse
import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import RLock

from factoriorl.agentic.bridge import BridgeRequestError, FactorioBridge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", default="construct_smelting_line")
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--token-env", required=True)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument(
        "--game-speed",
        type=float,
        default=None,
        help="explicit Factorio pacing speed recorded by the worker environment",
    )
    return parser.parse_args()


def build_bridge(task_id: str, split: str, seed: int) -> tuple[FactorioBridge, object, object]:
    from factoriorl.env import FactorioEnv
    from factoriorl.manifest import new_run_id
    from factoriorl.seeding import Branch, SeedPlan
    from factoriorl.session import WorkerSession
    from factoriorl.tasks import get
    from factoriorl.worker import WorkerManager

    manager = WorkerManager()
    run_id = new_run_id("agentic-bridge")
    handle = manager.launch(f"agentic-{task_id}-{run_id[-8:]}")
    session = WorkerSession(handle, timeout=30.0)
    env = FactorioEnv(
        get(task_id),
        session,
        SeedPlan(master=seed, run_id=run_id),
        branch=Branch.TRAIN if split == "train" else Branch.EVAL,
        split=split,
    )
    return FactorioBridge(env, task_id), manager, handle


def main() -> int:
    args = parse_args()
    if args.game_speed is not None:
        if args.game_speed <= 0:
            raise SystemExit("--game-speed must be positive")
        os.environ["FACTORIO_RL_GAME_SPEED"] = str(args.game_speed)
    token = os.environ.get(args.token_env)
    if not token:
        raise SystemExit(f"required token environment variable is unset: {args.token_env}")
    bridge, manager, handle = build_bridge(args.task, args.split, args.seed)
    request_lock = RLock()

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, status: HTTPStatus, body: dict) -> None:
            encoded = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _authorized(self) -> bool:
            return self.headers.get("Authorization") == f"Bearer {token}"

        def do_GET(self) -> None:  # noqa: N802
            if not self._authorized():
                self._reply(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
            elif self.path == "/v1/health":
                self._reply(
                    HTTPStatus.OK,
                    {"ok": True, "task": args.task, "game_speed": args.game_speed},
                )
            elif self.path == "/v1/observe":
                with request_lock:
                    self._reply(HTTPStatus.OK, bridge.observe())
            else:
                self._reply(HTTPStatus.NOT_FOUND, {"error": "unknown endpoint"})

        def do_POST(self) -> None:  # noqa: N802
            if not self._authorized():
                self._reply(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length)) if length else {}
                with request_lock:
                    if self.path == "/v1/reset":
                        result = bridge.reset(body.get("seed"))
                    elif self.path == "/v1/act":
                        result = bridge.act(body.get("index"), body.get("arguments"), body.get("target"))
                    elif self.path == "/v1/finish":
                        result = bridge.finish()
                    else:
                        self._reply(HTTPStatus.NOT_FOUND, {"error": "unknown endpoint"})
                        return
                self._reply(HTTPStatus.OK, result)
            except (BridgeRequestError, ValueError, json.JSONDecodeError) as exc:
                self._reply(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

        def log_message(self, _format: str, *_args) -> None:
            return

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        json.dumps(
            {"host": args.host, "port": args.port, "task": args.task, "game_speed": args.game_speed}
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()
        try:
            bridge.env.session.close()
        finally:
            manager.cleanup(handle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
