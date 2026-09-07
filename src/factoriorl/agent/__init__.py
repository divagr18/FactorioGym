"""Model adapters and the agent loop (PLAN.md 5.3).

A language-model agent plays through the same contracts a compact policy uses:
the same wire observation (rendered by :mod:`factoriorl.agent.summary` instead of
:mod:`factoriorl.encoders`), the same catalog action indices, and the same mask.
The provider sits behind :class:`ModelAdapter`, which sees only strings.

``factoriorl.agent.runner`` is deliberately not imported here: it launches a
worker, and everything else in this package must stay importable -- and
testable -- with no engine, no network and no credential.
"""

from __future__ import annotations

from factoriorl.agent.adapters import (
    AnthropicMessagesAdapter,
    ModelAdapter,
    ModelReply,
    ModelRequest,
    OpenAICompatibleAdapter,
    ScriptedAdapter,
)
from factoriorl.agent.loop import (
    MAX_ATTEMPTS_PER_DECISION,
    MAX_CONSECUTIVE_FALLBACKS,
    SYSTEM_PROMPT,
    AgentConfig,
    AgentLoop,
    Decision,
)
from factoriorl.agent.parsing import DecisionFailure, ParsedAction, ParseFailure, parse_action
from factoriorl.agent.summary import (
    LegalAction,
    ObservationSummary,
    TaskBrief,
    action_vocabulary,
    legal_actions,
    summarise,
)

__all__ = [
    "MAX_ATTEMPTS_PER_DECISION",
    "MAX_CONSECUTIVE_FALLBACKS",
    "SYSTEM_PROMPT",
    "AgentConfig",
    "AgentLoop",
    "AnthropicMessagesAdapter",
    "Decision",
    "DecisionFailure",
    "LegalAction",
    "ModelAdapter",
    "ModelReply",
    "ModelRequest",
    "ObservationSummary",
    "OpenAICompatibleAdapter",
    "ParseFailure",
    "ParsedAction",
    "ScriptedAdapter",
    "TaskBrief",
    "action_vocabulary",
    "legal_actions",
    "parse_action",
    "summarise",
]
