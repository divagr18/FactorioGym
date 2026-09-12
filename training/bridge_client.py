"""Tiny WSL-side client for the restricted Factorio rollout bridge."""

from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.error import HTTPError
from urllib.request import Request, urlopen


class BridgeError(RuntimeError):
    pass


@dataclass(frozen=True)
class BridgeClient:
    base_url: str
    token: str
    timeout_seconds: float = 60.0

    def _call(self, method: str, path: str, body: dict | None = None) -> dict:
        payload = None if body is None else json.dumps(body).encode("utf-8")
        request = Request(
            f"{self.base_url.rstrip('/')}{path}",
            data=payload,
            method=method,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                decoded = json.loads(response.read())
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise BridgeError(f"bridge {method} {path} returned {exc.code}: {detail}") from exc
        if "error" in decoded:
            raise BridgeError(str(decoded["error"]))
        return decoded

    def health(self) -> dict:
        return self._call("GET", "/v1/health")

    def reset(self, seed: int | None = None) -> dict:
        return self._call("POST", "/v1/reset", {} if seed is None else {"seed": seed})

    def observe(self) -> dict:
        return self._call("GET", "/v1/observe")

    def act(self, index: int, arguments: dict | None = None, target: str | None = None) -> dict:
        body = {"index": index, "arguments": arguments or {}}
        if target is not None:
            body["target"] = target
        return self._call("POST", "/v1/act", body)

    def finish(self) -> dict:
        return self._call("POST", "/v1/finish", {})
