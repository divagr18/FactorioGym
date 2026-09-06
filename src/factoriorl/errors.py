"""Error taxonomy.

Worker crashes are infrastructure failures, never ordinary task failures.
Invalid actions surface as structured :class:`~factoriorl.protocol.ActionResult`
objects with a reason; they do not raise and never silently become another action.
"""

from __future__ import annotations

from enum import Enum


class StartupFailureKind(Enum):
    """Why a worker failed to come up."""

    MISSING_EXECUTABLE = "missing_executable"
    INVALID_SAVE = "invalid_save"
    PORT_UNAVAILABLE = "port_unavailable"
    ENGINE_ERROR = "engine_error"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class FactorioRLError(Exception):
    """Base class for all FactorioRL errors."""


class StartupFailure(FactorioRLError):
    """Worker process failed to start; classified by kind."""

    def __init__(self, kind: StartupFailureKind, message: str) -> None:
        super().__init__(f"{kind.value}: {message}")
        self.kind = kind
        self.message = message


class WorkerFailure(FactorioRLError):
    """A running worker failed; the run evidence is preserved."""


class InfrastructureFailure(WorkerFailure):
    """Worker crash or transport loss: an infrastructure failure, not a task failure."""


class ProtocolError(FactorioRLError):
    """Malformed or unsupported protocol message."""


class UnsupportedProtocolError(ProtocolError):
    """Peer speaks a protocol version this side does not implement."""


class StaleEpisodeError(ProtocolError):
    """Request references an episode that has ended (reset or new worker)."""
