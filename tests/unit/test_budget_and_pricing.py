"""The spend cap and the run clock, which must bind before a request is sent.

Every assertion here maps to a clause of the agent roadmap
A0.3 and to a row of its Gate A0 table: valid output, missing usage, retry
accounting, deadline expiry, and no dispatch after exhaustion.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from factoriorl import pricing
from factoriorl.agent.budget import (
    Budget,
    BudgetExhausted,
    RunClock,
    normalise_usage,
)

MODEL = "deepseek-flash"


# --- the price table --------------------------------------------------------


def test_an_unknown_model_is_refused_rather_than_guessed() -> None:
    with pytest.raises(pricing.UnknownModelPrice) as caught:
        pricing.price("some-model-nobody-checked")
    # The message has to say what to do, because it fires during a preflight.
    assert "factoriorl.pricing.PRICES" in str(caught.value)


def test_documented_aliases_resolve_to_a_priced_model() -> None:
    assert pricing.price("deepseek-v4-flash").model == MODEL


def test_a_cache_hit_is_fifty_times_cheaper_than_a_miss() -> None:
    """The ratio the whole transcript design rests on. If it moves, that design
    needs revisiting rather than silently continuing to assume a 50x saving."""
    rates = pricing.price(MODEL)
    assert rates.input_cache_miss / rates.input_cache_hit == pytest.approx(50.0)


def test_every_price_records_where_and_when_it_came_from() -> None:
    for model, entry in pricing.PRICES.items():
        assert entry.retrieved, f"{model} has no retrieval date"
        assert entry.source_url.startswith("https://"), f"{model} has no source"


def test_off_peak_rates_are_recorded_but_never_applied() -> None:
    """Peak-only pricing is deliberate: a cost that is quietly too low is worse
    than one that is too high, and the peak window is provider policy."""
    weekend = datetime(2026, 9, 12, 2, 0, tzinfo=UTC)  # a Saturday
    assert pricing.is_peak(weekend) is False
    peak = pricing.cost_usd(MODEL, cache_miss_tokens=1_000_000)
    assert peak == pytest.approx(pricing.price(MODEL).input_cache_miss)


# --- normalising what providers report --------------------------------------


def test_deepseeks_explicit_cache_split_is_read() -> None:
    record = normalise_usage(
        {
            "prompt_tokens": 1000,
            "prompt_cache_hit_tokens": 960,
            "prompt_cache_miss_tokens": 40,
            "completion_tokens": 120,
        }
    )
    assert (record.cache_hit_tokens, record.cache_miss_tokens) == (960, 40)
    assert record.output_tokens == 120
    assert record.cache_split_reported is True
    assert record.shape == "deepseek"


def test_openais_nested_cached_count_is_read() -> None:
    record = normalise_usage(
        {
            "prompt_tokens": 500,
            "completion_tokens": 60,
            "prompt_tokens_details": {"cached_tokens": 384},
        }
    )
    assert (record.cache_hit_tokens, record.cache_miss_tokens) == (384, 116)
    assert record.shape == "openai"


def test_anthropics_cache_read_and_write_are_distinguished() -> None:
    # A cache *write* is billed like fresh input, not like a read.
    record = normalise_usage(
        {
            "input_tokens": 100,
            "cache_read_input_tokens": 900,
            "cache_creation_input_tokens": 50,
            "output_tokens": 30,
        }
    )
    assert record.cache_hit_tokens == 900
    assert record.cache_miss_tokens == 150
    assert record.shape == "anthropic"


def test_a_total_without_a_split_is_priced_as_all_miss() -> None:
    """An unknown is resolved against the run, never in its favour."""
    record = normalise_usage({"prompt_tokens": 800, "completion_tokens": 10})
    assert record.cache_hit_tokens == 0
    assert record.cache_miss_tokens == 800
    assert record.reported is True
    assert record.cache_split_reported is False


def test_absent_usage_is_reported_as_unmeasured_not_as_zero() -> None:
    for payload in (None, {}, {"note": "no counts here"}):
        record = normalise_usage(payload)
        assert record.reported is False, payload
        assert record.input_tokens == 0


# --- the cap ----------------------------------------------------------------


def test_a_reservation_prices_every_input_token_as_a_miss() -> None:
    budget = Budget(cap_usd=5.0, model=MODEL, max_output_tokens=4096)
    held = budget.reserve(input_tokens=100_000)
    expected = (100_000 * 0.30 + 4096 * 1.20) / 1_000_000
    assert held.usd == pytest.approx(expected)
    # The hold is visible in what is left, before any call has been made.
    assert budget.remaining_usd == pytest.approx(5.0 - expected)
    assert budget.committed_usd == 0.0


def test_settling_releases_the_hold_and_commits_the_real_cost() -> None:
    budget = Budget(cap_usd=5.0, model=MODEL, max_output_tokens=4096)
    budget.reserve(input_tokens=100_000)
    entry = budget.settle(
        {
            "prompt_tokens": 100_000,
            "prompt_cache_hit_tokens": 99_000,
            "prompt_cache_miss_tokens": 1_000,
            "completion_tokens": 400,
        }
    )
    real = (99_000 * 0.006 + 1_000 * 0.30 + 400 * 1.20) / 1_000_000
    assert budget.committed_usd == pytest.approx(real)
    assert budget.outstanding is None
    assert entry["billed_from"] == "usage"
    # The whole point: the hold was ~50x the real cost and was not charged.
    assert entry["reserved_usd"] > entry["spend_usd"] * 20


def test_a_call_with_no_usage_is_charged_at_its_reservation_and_flagged() -> None:
    """A0.3: retain the reservation for requests with uncertain billing."""
    budget = Budget(cap_usd=5.0, model=MODEL, max_output_tokens=4096)
    held = budget.reserve(input_tokens=10_000)
    entry = budget.settle(None)
    assert budget.committed_usd == pytest.approx(held.usd)
    assert budget.uncertain_calls == 1
    assert entry["billed_from"] == "reservation"
    assert "conservative" in budget.to_dict()["accounting"]


def test_a_request_that_could_breach_the_cap_is_never_dispatched() -> None:
    budget = Budget(cap_usd=0.01, model=MODEL, max_output_tokens=4096)
    with pytest.raises(BudgetExhausted) as caught:
        budget.reserve(input_tokens=1_000_000)
    message = str(caught.value)
    assert "not sent" in message
    assert budget.outstanding is None, "a refused reservation must not be held"


def test_only_one_request_may_be_outstanding() -> None:
    budget = Budget(cap_usd=5.0, model=MODEL, max_output_tokens=4096)
    budget.reserve(input_tokens=1000)
    with pytest.raises(RuntimeError, match="already outstanding"):
        budget.reserve(input_tokens=1000)


def test_settling_without_a_reservation_is_a_programming_error() -> None:
    budget = Budget(cap_usd=5.0, model=MODEL, max_output_tokens=4096)
    with pytest.raises(RuntimeError, match="nothing was dispatched"):
        budget.settle({"prompt_tokens": 1})


def test_an_unknown_model_fails_at_construction_not_at_dispatch() -> None:
    with pytest.raises(pricing.UnknownModelPrice):
        Budget(cap_usd=5.0, model="not-in-the-table", max_output_tokens=100)


def test_the_cache_hit_rate_is_reported_and_starts_unknown() -> None:
    budget = Budget(cap_usd=5.0, model=MODEL, max_output_tokens=4096)
    assert budget.cache_hit_rate is None
    budget.reserve(input_tokens=1000)
    budget.settle({"prompt_cache_hit_tokens": 970, "prompt_cache_miss_tokens": 30})
    assert budget.cache_hit_rate == pytest.approx(0.97)


def test_caching_is_what_makes_the_five_dollar_cap_reachable() -> None:
    """The plan's arithmetic, as an executable claim rather than a paragraph.

    Same 300-decision conversation, priced twice: once with an append-only prefix
    that the provider caches, once rebuilt from scratch each turn. If this ever
    stops holding, the transcript discipline has stopped paying for itself.
    """
    turns, per_turn, out = 300, 2_500, 800
    cached = uncached = 0.0
    for index in range(1, turns + 1):
        prefix = per_turn * (index - 1)
        cached += pricing.cost_usd(
            MODEL, cache_hit_tokens=prefix, cache_miss_tokens=per_turn, output_tokens=out
        )
        uncached += pricing.cost_usd(MODEL, cache_miss_tokens=prefix + per_turn, output_tokens=out)
    assert cached < 5.0, f"the cached run should fit the cap, cost ${cached:.2f}"
    assert uncached > 20.0, f"the uncached run should blow it, cost ${uncached:.2f}"


# --- the run clock ----------------------------------------------------------


def test_the_clock_does_not_run_before_gameplay_begins() -> None:
    """It starts at the first gameplay observation, so a slow engine launch does
    not consume the agent's thirty minutes."""
    clock = RunClock(limit_seconds=0.0)
    assert clock.expired() is False
    clock.start()
    assert clock.expired() is True


def test_a_request_timeout_cannot_outlive_the_run() -> None:
    clock = RunClock(limit_seconds=1000.0)
    clock.start()
    clock.started_at -= 995.0  # 5 seconds left
    assert clock.deadline_for_request(180.0) == pytest.approx(5.0, abs=0.5)
    clock.started_at -= 100.0  # already past
    assert clock.deadline_for_request(180.0) == 0.0


def test_before_the_clock_starts_a_request_keeps_its_own_ceiling() -> None:
    clock = RunClock(limit_seconds=60.0)
    assert clock.deadline_for_request(30.0) == 30.0


def test_stopping_the_clock_freezes_elapsed_time() -> None:
    """The paused finalisation at the end of a run is outside gameplay."""
    clock = RunClock(limit_seconds=60.0)
    clock.start()
    clock.stop()
    frozen = clock.elapsed_seconds
    assert clock.elapsed_seconds == frozen
    assert clock.to_dict()["measures"].startswith("gameplay only")
