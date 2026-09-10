"""What a model call costs, snapshotted rather than guessed.

`docs/AGENTIC_ROADMAP-2026-09-10.md` A0.3 requires a spend cap enforced *before*
dispatch, and says plainly what to do when the inputs to that arithmetic are not
available: "If a safe input-token upper bound or current pricing cannot be
established, fail the paid preflight instead of guessing."

So this module is a table, not a lookup service. `price()` raises on a model it
does not know. A provider that quietly changes an alias should stop a run rather
than bill it at a number this repository made up.

Two deliberate choices
----------------------
**Everything is priced at peak.** The published rates carry an off-peak discount,
and the off-peak figures are recorded below for transparency, but no figure this
module returns is ever *reduced* by them. The peak window is provider policy that
can move without notice, the run clock is local, and a cost that is quietly too
low is a worse failure than one that is too high. `is_peak()` exists to report
which window a run happened to fall in, not to discount it.

**Cache-hit and cache-miss input are separate prices, not an average.** For
`deepseek-flash` a cache hit is **fifty times** cheaper than a miss --
$0.006 against $0.30 per million. That ratio is why
`src/factoriorl/agent/transcript.py` exists and why the append-only prompt
discipline is a correctness property of the budget rather than a nicety: at ~300
decisions the same run costs about $1.20 with prefix caching and about $34
without, against a $5 cap.

Consequently a reservation must assume the *miss* rate for every input token. A
run cannot know in advance that its prefix survived in the provider's cache, and
the roadmap asks for "all input at the uncached peak rate".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

#: Dollars per million tokens. Every rate here is the **peak** rate.
PER_MILLION = 1_000_000


@dataclass(frozen=True)
class ModelPrice:
    """Published rates for one model id, at one retrieval date."""

    model: str
    input_cache_hit: float
    input_cache_miss: float
    output: float
    context_window: int
    max_output: int
    retrieved: str
    source_url: str
    #: Recorded so a reader can see the discount exists. Never applied.
    off_peak: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "unit": "usd_per_million_tokens",
            "basis": "peak rates; off-peak discounts are recorded but never applied",
            "input_cache_hit": self.input_cache_hit,
            "input_cache_miss": self.input_cache_miss,
            "output": self.output,
            "context_window": self.context_window,
            "max_output": self.max_output,
            "retrieved": self.retrieved,
            "source_url": self.source_url,
            "off_peak_reference": dict(self.off_peak),
        }


#: Retrieved 2026-09-10 from the official pricing page. `deepseek-v4-pro` is
#: listed because requests to it are routed to V4.1 Flash and billed at the Flash
#: price as of 2026-09-14 -- a run that names it should still cost what it costs.
PRICES: dict[str, ModelPrice] = {
    "deepseek-flash": ModelPrice(
        model="deepseek-flash",
        input_cache_hit=0.006,
        input_cache_miss=0.30,
        output=1.20,
        context_window=1_000_000,
        max_output=384_000,
        retrieved="2026-09-10",
        source_url="https://api-docs.deepseek.com/quick_start/pricing/",
        off_peak={"input_cache_hit": 0.003, "input_cache_miss": 0.15, "output": 0.60},
    ),
    "deepseek-v4-pro": ModelPrice(
        model="deepseek-v4-pro",
        input_cache_hit=0.044,
        input_cache_miss=1.32,
        output=3.96,
        context_window=1_000_000,
        max_output=384_000,
        retrieved="2026-09-10",
        source_url="https://api-docs.deepseek.com/quick_start/pricing/",
        off_peak={"input_cache_hit": 0.022, "input_cache_miss": 0.66, "output": 1.98},
    ),
}

#: A local inference endpoint -- llama.cpp, vLLM, LM Studio, Ollama -- bills
#: nothing, and that is a *known* price rather than an unknown one. Without this
#: the default `factoriorl agent` invocation, which points at a local server the
#: way `doctor-agent` does, would be refused by a cap it can never breach.
#:
#: Zero is asserted here only for endpoints that are free by construction. A
#: hosted model whose price nobody has checked still raises, which is the case
#: the refusal exists for.
_FREE = dict(
    input_cache_hit=0.0,
    input_cache_miss=0.0,
    output=0.0,
    context_window=0,
    max_output=0,
    retrieved="n/a",
    source_url="https://localhost",
)

PRICES["local-model"] = ModelPrice(model="local-model", **_FREE)
PRICES["scripted"] = ModelPrice(model="scripted", **_FREE)

#: Aliases the provider documents as resolving to a priced model.
ALIASES: dict[str, str] = {
    "deepseek-v4-flash": "deepseek-flash",
    "deepseek-v4-flash-vision-exp": "deepseek-flash",
}


class UnknownModelPrice(LookupError):
    """Raised rather than guessing. A0.3: fail the preflight instead."""


def price(model: str) -> ModelPrice:
    """The published rates for `model`, or raise.

    Raising is the feature. A missing entry means nobody has checked what this
    model costs, and a budget computed from an invented number is not a budget.
    """
    resolved = ALIASES.get(model, model)
    if resolved not in PRICES:
        known = ", ".join(sorted(set(PRICES) | set(ALIASES)))
        raise UnknownModelPrice(
            f"no snapshotted price for model {model!r}. A spend cap cannot be enforced "
            f"against a price this repository does not have, so this is refused rather "
            f"than guessed. Add it to factoriorl.pricing.PRICES with its retrieval date "
            f"and source, or pick one of: {known}"
        )
    return PRICES[resolved]


def is_peak(when: datetime | None = None) -> bool:
    """Whether `when` falls in the provider's peak window.

    Reporting only -- see the module docstring. Published as 01:00-04:00 and
    06:00-10:00 UTC, Monday to Friday.
    """
    moment = (when or datetime.now(UTC)).astimezone(UTC)
    if moment.weekday() >= 5:
        return False
    hour = moment.hour
    return 1 <= hour < 4 or 6 <= hour < 10


def cost_usd(
    model: str,
    *,
    cache_hit_tokens: int = 0,
    cache_miss_tokens: int = 0,
    output_tokens: int = 0,
) -> float:
    """Dollars for one call, given the token split the provider reported."""
    rates = price(model)
    total = (
        cache_hit_tokens * rates.input_cache_hit
        + cache_miss_tokens * rates.input_cache_miss
        + output_tokens * rates.output
    )
    return total / PER_MILLION


def worst_case_usd(model: str, *, input_tokens: int, max_output_tokens: int) -> float:
    """The reservation: every input token a cache miss, output at its ceiling.

    A run cannot know before dispatch whether its prefix is still resident in the
    provider's cache, so the reservation assumes it is not. The roadmap's wording
    is "all input at the uncached peak rate"; with a 50x hit/miss ratio this is a
    large over-estimate on a well-cached run, which is why `Budget` treats it as
    a *hold* that is released on reconciliation rather than as spend.
    """
    return cost_usd(
        model,
        cache_miss_tokens=input_tokens,
        output_tokens=max_output_tokens,
    )
