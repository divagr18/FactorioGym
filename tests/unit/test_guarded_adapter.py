"""The cap and the deadline must stop a request *before* it is sent.

Gate A0 names the property directly: a mock delayed response proves no
additional request starts after budget exhaustion. So every assertion here is
against the wrapped adapter's own call count -- what actually left the machine --
rather than against a log line saying it did not.
"""

from __future__ import annotations

import json

import pytest

from factoriorl.agent.adapters import ModelRequest, ScriptedAdapter
from factoriorl.agent.budget import Budget, RunClock
from factoriorl.agent.guarded import BudgetedAdapter, estimate_input_tokens

MODEL = "deepseek-flash"


def _request(text: str = "observe the world") -> ModelRequest:
    return ModelRequest(system="rules", user=text)


def _answer() -> str:
    return json.dumps({"action": "wait", "reason": "ok"})


def _guarded(cap: float = 5.0, *, clock: RunClock | None = None, usage: dict | None = None):
    inner = ScriptedAdapter([_answer()] * 10, usage=usage)
    budget = Budget(cap_usd=cap, model=MODEL, max_output_tokens=4096)
    return inner, budget, BudgetedAdapter(inner, budget, clock)


# --- the estimate -----------------------------------------------------------


def test_the_token_estimate_is_an_upper_bound_not_a_guess() -> None:
    """Two characters per token is pessimistic on purpose: English runs about
    four, so ordinary text reserves roughly double what it needs. Over-reserving
    costs only headroom, because the hold is released on settlement."""
    request = ModelRequest(system="a" * 100, user="b" * 100)
    assert estimate_input_tokens(request) > 200 / 4


def test_the_estimate_counts_a_history_when_there_is_one() -> None:
    flat = ModelRequest(system="s", user="u")
    history = ModelRequest(
        system="s",
        user="u",
        messages=({"role": "system", "content": "s"}, {"role": "user", "content": "u" * 500}),
    )
    assert estimate_input_tokens(history) > estimate_input_tokens(flat)


# --- refusing ---------------------------------------------------------------


def test_no_request_is_sent_once_the_budget_is_exhausted() -> None:
    inner, budget, guarded = _guarded(cap=0.0001)
    reply = guarded.complete(_request("x" * 10_000))

    assert inner.calls == 0, "a request was dispatched against an exhausted cap"
    assert reply.ok is False
    assert reply.error_kind == "budget_exhausted"
    assert "not sent" in reply.error


def test_retrying_after_exhaustion_still_sends_nothing() -> None:
    """The loop retries a failed call. Each retry must be refused again rather
    than eventually slipping one through."""
    inner, budget, guarded = _guarded(cap=0.0001)
    for _ in range(5):
        guarded.complete(_request("x" * 10_000))
    assert inner.calls == 0
    assert len(guarded.refusals) == 5


def test_no_request_is_sent_once_the_deadline_has_passed() -> None:
    clock = RunClock(limit_seconds=0.0)
    clock.start()
    inner, budget, guarded = _guarded(clock=clock)

    reply = guarded.complete(_request())
    assert inner.calls == 0
    assert reply.error_kind == "deadline_expired"
    assert budget.outstanding is None, "a refused request must not hold a reservation"


def test_a_refusal_leaves_no_reservation_behind() -> None:
    """A hold that is never released would wedge every later request."""
    inner, budget, guarded = _guarded(cap=0.0001)
    guarded.complete(_request("x" * 10_000))
    assert budget.outstanding is None
    assert budget.calls == 0, "a request that never left must not count as a call"


# --- accounting a call that does happen -------------------------------------


def test_a_dispatched_call_is_reserved_then_settled_from_real_usage() -> None:
    usage = {
        "prompt_tokens": 1000,
        "prompt_cache_hit_tokens": 950,
        "prompt_cache_miss_tokens": 50,
        "completion_tokens": 20,
    }
    inner, budget, guarded = _guarded(usage=usage)
    reply = guarded.complete(_request())

    assert inner.calls == 1
    assert reply.ok
    assert budget.outstanding is None
    assert budget.calls == 1
    assert budget.cache_hit_tokens == 950
    assert budget.cache_hit_rate == pytest.approx(0.95)


def test_a_call_the_provider_did_not_measure_keeps_its_reservation() -> None:
    inner, budget, guarded = _guarded(usage=None)
    guarded.complete(_request())
    assert budget.uncertain_calls == 1
    assert budget.committed_usd > 0.0
    assert budget.ledger[-1]["billed_from"] == "reservation"


def test_an_adapter_that_raises_does_not_wedge_the_budget() -> None:
    """`complete` is contractually forbidden from raising. If one does anyway,
    the hold must still clear -- otherwise a single misbehaving provider stops
    the run for a reason nobody could diagnose from the artifact."""

    class Exploding(ScriptedAdapter):
        def complete(self, request):
            raise RuntimeError("this adapter breaks its contract")

    budget = Budget(cap_usd=5.0, model=MODEL, max_output_tokens=4096)
    guarded = BudgetedAdapter(Exploding([_answer()]), budget)
    with pytest.raises(RuntimeError):
        guarded.complete(_request())
    assert budget.outstanding is None
    assert budget.uncertain_calls == 1


# --- the deadline reaching the provider -------------------------------------


def test_a_request_timeout_is_narrowed_to_what_is_left_of_the_run() -> None:
    """A0.3: a provider request inherits the remaining deadline. A 180-second
    timeout would otherwise run well past a deadline that has already passed."""

    class Recording(ScriptedAdapter):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.timeout = 180.0
            self.seen: list[float] = []

        def complete(self, request):
            self.seen.append(self.timeout)
            return super().complete(request)

    clock = RunClock(limit_seconds=1000.0)
    clock.start()
    clock.started_at -= 997.0  # three seconds left
    inner = Recording([_answer()])
    budget = Budget(cap_usd=5.0, model=MODEL, max_output_tokens=4096)
    BudgetedAdapter(inner, budget, clock).complete(_request())

    assert inner.seen[0] == pytest.approx(3.0, abs=0.5)
    assert inner.timeout == 180.0, "the adapter's own timeout must be restored"


# --- what the manifest sees -------------------------------------------------


def test_the_proxy_reports_the_provider_not_itself() -> None:
    inner, _, guarded = _guarded()
    assert guarded.name == inner.name
    assert guarded.describe()["adapter"] == inner.describe()["adapter"]


def test_the_limits_are_published_beside_the_provider_settings() -> None:
    inner, budget, guarded = _guarded(clock=RunClock(limit_seconds=1800.0))
    described = guarded.describe()["limits"]
    assert described["budget"]["cap_usd"] == 5.0
    assert described["clock"]["limit_seconds"] == 1800.0
    assert "characters" in described["input_token_estimate"]
    assert "conservative" in described["budget"]["accounting"]


def test_redaction_still_reaches_the_wrapped_adapters_credential() -> None:
    class Secretive(ScriptedAdapter):
        def redact(self, text: str) -> str:
            return text.replace("sk-secret", "<redacted>")

    budget = Budget(cap_usd=5.0, model=MODEL, max_output_tokens=4096)
    guarded = BudgetedAdapter(Secretive([_answer()]), budget)
    assert guarded.redact("key sk-secret here") == "key <redacted> here"
