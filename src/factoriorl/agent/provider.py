"""One place that builds a model provider with its spend cap attached.

Before this, `factoriorl agent` and `tools/demonstration.py` each constructed an
adapter themselves. That is not merely duplication: the demonstration built a
bare `OpenAICompatibleAdapter` and so had **no spend cap at all**. Harmless while
it pointed at a local endpoint, and a real liability the moment any entrypoint
points at a paid provider -- which is what roadmap A5 does.

So the rule this module exists to make structural: **an adapter reaches a run
already wrapped.** `build()` returns the guarded adapter, and the raw provider
alongside it only because the caller needs its redactor for the artifact it
writes. There is no path here that hands back an unguarded provider to run with.
"""

from __future__ import annotations

from dataclasses import dataclass

from factoriorl.agent.adapters import (
    AnthropicMessagesAdapter,
    ModelAdapter,
    OpenAICompatibleAdapter,
)
from factoriorl.agent.budget import Budget, RunClock
from factoriorl.agent.guarded import BudgetedAdapter

#: Default ceiling on one run's provider spend, in US dollars. The roadmap's
#: first-run allowance.
DEFAULT_MAX_COST_USD = 5.0

#: Default wall-clock ceiling, measured from the first gameplay observation.
DEFAULT_MAX_WALL_SECONDS = 1800.0


@dataclass
class Provider:
    """A guarded adapter and the pieces a caller needs to report on it."""

    #: What the loop should be given. Always the guarded one.
    adapter: BudgetedAdapter
    #: The unwrapped provider, for its redactor and its `describe()`. Never for
    #: running: an adapter that skipped the proxy would skip the cap.
    inner: ModelAdapter
    budget: Budget
    clock: RunClock

    def redact(self, text: str) -> str:
        return self.inner.redact(text)

    def to_dict(self) -> dict:
        return {
            "budget": self.budget.to_dict(),
            "clock": self.clock.to_dict(),
            "refusals": list(self.adapter.refusals),
        }


def build(
    *,
    model: str,
    adapter: str = "openai",
    base_url: str = "http://127.0.0.1:8080/v1",
    api_key_env: str = "",
    timeout: float = 60.0,
    temperature: float | None = 0.0,
    token_parameter: str = "max_tokens",
    thinking: bool | None = None,
    reasoning_effort: str | None = None,
    max_tokens: int = 4096,
    max_cost_usd: float = DEFAULT_MAX_COST_USD,
    max_wall_seconds: float = DEFAULT_MAX_WALL_SECONDS,
) -> Provider:
    """Build a provider with its cap and clock already attached.

    Raises `factoriorl.pricing.UnknownModelPrice` if the model has no
    snapshotted price. That happens here, before a worker is launched, because
    discovering it after a ninety-second map generation is the wrong time --
    A0.3's "fail the paid preflight instead of guessing".
    """
    budget = Budget(cap_usd=max_cost_usd, model=model, max_output_tokens=max_tokens)
    clock = RunClock(limit_seconds=max_wall_seconds)

    if adapter == "anthropic":
        inner: ModelAdapter = AnthropicMessagesAdapter(
            model=model,
            api_key_env=api_key_env or "ANTHROPIC_API_KEY",
            timeout=timeout,
        )
    else:
        inner = OpenAICompatibleAdapter(
            base_url=base_url,
            model=model,
            api_key_env=api_key_env or None,
            timeout=timeout,
            temperature=temperature,
            token_parameter=token_parameter,
            thinking=thinking,
            reasoning_effort=reasoning_effort,
        )
    return Provider(
        adapter=BudgetedAdapter(inner, budget, clock),
        inner=inner,
        budget=budget,
        clock=clock,
    )
