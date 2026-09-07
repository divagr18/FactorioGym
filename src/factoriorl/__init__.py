"""FactorioRL: embodied Factorio environment for RL policies and language-model agents.

Importing this package never launches Factorio; engine access is explicit via
:mod:`factoriorl.engine` and worker commands.
"""

from factoriorl.errors import (
    FactorioRLError,
    InfrastructureFailure,
    ProtocolError,
    StaleEpisodeError,
    StartupFailure,
    UnsupportedProtocolError,
    WorkerFailure,
)
from factoriorl.protocol import (
    PROTOCOL_VERSION,
    ActionResult,
    ActionStatus,
    ActionType,
    ErrorBody,
    ErrorCode,
    Request,
    RequestType,
    Response,
    ResultCode,
)

__version__ = "0.1.0"

__all__ = [
    "PROTOCOL_VERSION",
    "ActionStatus",
    "ActionResult",
    "ActionType",
    "ErrorCode",
    "RequestType",
    "ErrorBody",
    "FactorioRLError",
    "InfrastructureFailure",
    "ProtocolError",
    "Request",
    "Response",
    "ResultCode",
    "StartupFailure",
    "StaleEpisodeError",
    "UnsupportedProtocolError",
    "WorkerFailure",
    "__version__",
]
