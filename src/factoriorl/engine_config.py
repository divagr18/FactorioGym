"""Engine discovery and version pinning (PLAN.md 0.2).

The actual game build used for integration is pinned: it is recorded in every
run manifest and never silently switched. Resolution order:

1. ``factoriorl.engine_config.from_json`` with an explicit ``executable``.
2. ``FACTORIO_RL_ENGINE`` environment variable.
3. ``.factoriorl.user.json`` at the workspace root (gitignored, user-local).
4. Default Windows installation path.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from factoriorl.errors import StartupFailure, StartupFailureKind
from factoriorl.paths import workspace_root

ENGINE_ENV_VAR = "FACTORIO_RL_ENGINE"
SPEED_ENV_VAR = "FACTORIO_RL_GAME_SPEED"
USER_CONFIG_NAME = ".factoriorl.user.json"
DEFAULT_WINDOWS_EXE = r"D:\Factorio\bin\x64\factorio.exe"

#: Paused-loop and tick multiplier. 74% of a step is the predicted wait for the
#: decision interval to elapse, `ticks / (60 * speed)`, so this sets throughput
#: directly -- and the best value is a property of the *machine*, not of the
#: system. Measured on two:
#:
#:     laptop (8c mobile)       60: 11.16 ms   90: 8.82   120: 9.73   200: 10.34
#:     desktop (Ryzen 5 5600T)  60: 11.18 ms   90: 8.20   120: 6.85   240:  6.51
#:
#: The laptop's optimum is 90 and the desktop's is 120, because the useful
#: ceiling is the engine's own tick rate: once `after_send` stops falling a
#: shorter sleep only makes the settle prediction optimistic, the collect poll
#: count climbs above 1, and the extra RCON traffic costs more than the sleep
#: saves. So 90 is the default and a faster machine declares its own.
#:
#: It matters where there is *one* worker -- solvability, agent runs,
#: demonstration collection. At six workers the engines are core-limited and
#: cannot reach even the speed-90 rate: a repeated A/B measured 402 against 408
#: steps/s for 90 against 120, no difference, with five times the spread.
DEFAULT_GAME_SPEED = 90.0

#: Pinned engine build (PLAN.md 0.2). Verified on this workstation.
EXPECTED_VERSION = "2.0.60"
EXPECTED_BUILD = 83512
MINIMUM_SUPPORTED_BUILD = 83512

_VERSION_RE = re.compile(r"^Version:\s*([0-9.]+)\s*\(build\s+(\d+),", re.MULTILINE)


@dataclass(frozen=True)
class EngineConfig:
    """Resolved Factorio executable plus the pinned build metadata."""

    executable: Path
    version: str
    build: int
    platform: str = "win64"

    def to_dict(self) -> dict:
        data = asdict(self)
        data["executable"] = str(self.executable)
        return data

    def assert_compatible(self) -> None:
        """Fail loudly if the resolved build is not the pinned one."""
        if self.build < MINIMUM_SUPPORTED_BUILD:
            raise StartupFailure(
                StartupFailureKind.ENGINE_ERROR,
                f"engine build {self.build} is older than pinned minimum {MINIMUM_SUPPORTED_BUILD}",
            )
        if self.version != EXPECTED_VERSION or self.build != EXPECTED_BUILD:
            raise StartupFailure(
                StartupFailureKind.ENGINE_ERROR,
                f"engine {self.version} (build {self.build}) differs from pinned "
                f"{EXPECTED_VERSION} (build {EXPECTED_BUILD}); update the pin "
                f"deliberately instead of silently switching builds",
            )


def _candidate_paths() -> list[Path]:
    candidates: list[Path] = []
    env = os.environ.get(ENGINE_ENV_VAR)
    if env:
        candidates.append(Path(env))
    user_cfg = workspace_root() / USER_CONFIG_NAME
    if user_cfg.is_file():
        try:
            cfg = json.loads(user_cfg.read_text(encoding="utf-8"))
            exe = cfg.get("engine", {}).get("executable")
            if exe:
                candidates.append(Path(exe))
        except (OSError, json.JSONDecodeError):
            pass
    candidates.append(Path(DEFAULT_WINDOWS_EXE))
    return candidates


def resolve_game_speed() -> float:
    """Engine speed for this machine: env var, then user config, then default.

    Same resolution order as the executable, for the same reason: it is a local
    fact, it must not require editing the package, and it has to be recorded in
    a run's manifest so two machines' throughput numbers stay comparable.
    """
    raw = os.environ.get(SPEED_ENV_VAR)
    if raw:
        try:
            value = float(raw)
        except ValueError:
            raise StartupFailure(
                f"{SPEED_ENV_VAR}={raw!r} is not a number",
                kind=StartupFailureKind.UNKNOWN,
            ) from None
        if value > 0:
            return value

    user_cfg = workspace_root() / USER_CONFIG_NAME
    if user_cfg.exists():
        try:
            cfg = json.loads(user_cfg.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cfg = {}
        value = ((cfg or {}).get("engine") or {}).get("game_speed")
        if isinstance(value, int | float) and value > 0:
            return float(value)
    return DEFAULT_GAME_SPEED


def resolve_engine_config() -> EngineConfig:
    """Locate the Factorio executable and probe its version string."""
    last_error = "no candidate executable configured"
    for path in _candidate_paths():
        if not path.is_file():
            last_error = f"executable not found: {path}"
            continue
        version, build = probe_version(path)
        return EngineConfig(executable=path.resolve(), version=version, build=build)
    raise StartupFailure(StartupFailureKind.MISSING_EXECUTABLE, last_error)


def probe_version(executable: Path) -> tuple[str, int]:
    """Run ``factorio --version`` and parse version and build number."""
    try:
        proc = subprocess.run(
            [str(executable), "--version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except OSError as exc:
        raise StartupFailure(
            StartupFailureKind.MISSING_EXECUTABLE, f"cannot execute {executable}: {exc}"
        ) from exc
    output = proc.stdout + proc.stderr
    match = _VERSION_RE.search(output)
    if match is None:
        raise StartupFailure(
            StartupFailureKind.ENGINE_ERROR, f"unparseable version output: {output[:200]!r}"
        )
    return match.group(1), int(match.group(2))
