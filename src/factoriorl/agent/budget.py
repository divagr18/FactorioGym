"""Spend and wall-clock limits that bind *before* a request is dispatched.

the agent roadmap A0.3, in full force: one outstanding
request; reserve the worst-case cost before dispatch; stop before dispatch if it
could exceed the remaining budget; use actual usage when available and retain the
reservation for requests with uncertain billing; report the result as
conservative accounting rather than a verified provider invoice.

Why a reservation is a *hold* and not a spend
---------------------------------------------
Input grows quadratically over an append-only conversation -- turn *i* re-sends
everything before it -- so the worst-case reservations summed across ~300
decisions come to roughly $34, while the actual cost with prefix caching is about
$1.20. Treating each reservation as spend would refuse dispatch around turn 110
on a run that was never going to cost more than a quarter of its cap.

The roadmap resolves this with "use one outstanding model request": at most one
hold exists at a time, so the test is

    committed + outstanding <= cap

and the hold is released the moment real usage arrives. When usage does *not*
arrive the hold is converted to committed spend instead of released -- an
unmeasured call is assumed to have cost its worst case, and the run is flagged so
the final figure is never presented as exact.

What this is not
----------------
Not an invoice. Token counts come from the provider's own `usage` block, prices
come from a table snapshotted on a date recorded beside them
(`factoriorl.pricing`), and every rate is the peak rate. The number is an upper
bound this repository can defend, not a reconciliation against a bill.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from factoriorl.pricing import cost_usd, price, worst_case_usd

#: Input token shapes seen in the wild, and how each reports prefix caching.
#: DeepSeek splits the prompt explicitly; OpenAI nests a cached count inside
#: `prompt_tokens_details`; Anthropic uses its own `*_input_tokens` names.
_TOTAL_INPUT_KEYS = ("prompt_tokens", "input_tokens")
_TOTAL_OUTPUT_KEYS = ("completion_tokens", "output_tokens")


class BudgetExhausted(RuntimeError):
    """Raised instead of dispatching a request that could breach the cap."""


class DeadlineExpired(RuntimeError):
    """Raised instead of dispatching a request that could outlive the run clock."""


@dataclass(frozen=True)
class Usage:
    """One call's token counts, normalised across provider shapes."""

    cache_hit_tokens: int
    cache_miss_tokens: int
    output_tokens: int
    #: False when the provider reported no usable counts at all, which is what
    #: makes a call "uncertainly billed" and keeps its reservation.
    reported: bool
    #: False when the provider gave a total but no cache split, so every input
    #: token was priced as a miss. Distinguishes "no cache" from "unknown cache".
    cache_split_reported: bool
    shape: str

    @property
    def input_tokens(self) -> int:
        return self.cache_hit_tokens + self.cache_miss_tokens

    def to_dict(self) -> dict:
        return {
            "cache_hit_tokens": self.cache_hit_tokens,
            "cache_miss_tokens": self.cache_miss_tokens,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reported": self.reported,
            "cache_split_reported": self.cache_split_reported,
            "shape": self.shape,
        }


def _first_int(payload: dict, keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return int(value)
    return None


def normalise_usage(usage: dict[str, Any] | None) -> Usage:
    """Map any provider's `usage` block onto one shape.

    `AgentLoop.aggregate` currently sums whatever numeric keys a provider happened
    to send, so a run that touched two providers produces four different token
    names and no total means anything. Normalising here fixes that at the source.

    **An unreported cache split is priced as all-miss**, never as all-hit: an
    unknown is resolved against the run, not in its favour.
    """
    if not usage:
        return Usage(0, 0, 0, reported=False, cache_split_reported=False, shape="absent")

    output = _first_int(usage, _TOTAL_OUTPUT_KEYS) or 0

    # DeepSeek: an explicit hit/miss split of the prompt.
    hit = _first_int(usage, ("prompt_cache_hit_tokens",))
    miss = _first_int(usage, ("prompt_cache_miss_tokens",))
    if hit is not None or miss is not None:
        total = _first_int(usage, _TOTAL_INPUT_KEYS)
        hit = hit or 0
        if miss is None:
            miss = max((total or hit) - hit, 0)
        return Usage(hit, miss, output, True, True, "deepseek")

    # Anthropic: cache reads and cache writes are separate from fresh input.
    read = _first_int(usage, ("cache_read_input_tokens",))
    written = _first_int(usage, ("cache_creation_input_tokens",))
    if read is not None or written is not None:
        fresh = _first_int(usage, ("input_tokens",)) or 0
        return Usage(read or 0, fresh + (written or 0), output, True, True, "anthropic")

    # OpenAI: a nested cached count, when the endpoint reports one at all.
    details = usage.get("prompt_tokens_details")
    total = _first_int(usage, _TOTAL_INPUT_KEYS)
    if isinstance(details, dict):
        cached = _first_int(details, ("cached_tokens",))
        if cached is not None and total is not None:
            return Usage(cached, max(total - cached, 0), output, True, True, "openai")

    if total is None and output == 0:
        return Usage(0, 0, 0, reported=False, cache_split_reported=False, shape="unrecognised")

    # A total with no split: every input token priced as a miss.
    return Usage(0, total or 0, output, True, False, "total-only")


@dataclass
class Reservation:
    input_tokens: int
    max_output_tokens: int
    usd: float


@dataclass
class Budget:
    """A spend cap with exactly one outstanding hold.

    `cap_usd` covers *everything* the roadmap says it covers: live provider
    checks, retries, and the run itself.
    """

    cap_usd: float
    model: str
    max_output_tokens: int
    committed_usd: float = 0.0
    calls: int = 0
    uncertain_calls: int = 0
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0
    output_tokens: int = 0
    outstanding: Reservation | None = None
    #: Per-call records, so a run can be audited rather than trusted.
    ledger: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Fails here rather than at the first dispatch: an unknown price is a
        # preflight failure, and the preflight is where it should surface.
        price(self.model)

    @property
    def remaining_usd(self) -> float:
        held = self.outstanding.usd if self.outstanding else 0.0
        return self.cap_usd - self.committed_usd - held

    def reserve(self, input_tokens: int) -> Reservation:
        """Hold the worst case for one call, or refuse to dispatch it."""
        if self.outstanding is not None:
            raise RuntimeError(
                "a reservation is already outstanding; A0.3 permits one in-flight "
                "request so that the hold and the cap can be compared meaningfully"
            )
        usd = worst_case_usd(
            self.model,
            input_tokens=input_tokens,
            max_output_tokens=self.max_output_tokens,
        )
        if self.committed_usd + usd > self.cap_usd:
            raise BudgetExhausted(
                f"dispatching this request could cost up to ${usd:.4f} against "
                f"${self.cap_usd - self.committed_usd:.4f} left of a ${self.cap_usd:.2f} "
                f"cap, so it is not sent. Committed so far: ${self.committed_usd:.4f} "
                f"over {self.calls} calls"
            )
        self.outstanding = Reservation(input_tokens, self.max_output_tokens, usd)
        return self.outstanding

    def settle(self, usage: dict[str, Any] | None) -> dict:
        """Release the hold and commit what the call actually cost.

        With no usable `usage` the hold becomes the committed figure -- the
        roadmap's "retain the reservation for requests with uncertain billing" --
        and the call is counted as uncertain so the total is never reported as
        exact.
        """
        held = self.outstanding
        if held is None:
            raise RuntimeError("settle() without a reservation: nothing was dispatched")
        self.outstanding = None
        self.calls += 1

        record = normalise_usage(usage)
        if record.reported:
            spend = cost_usd(
                self.model,
                cache_hit_tokens=record.cache_hit_tokens,
                cache_miss_tokens=record.cache_miss_tokens,
                output_tokens=record.output_tokens,
            )
        else:
            spend = held.usd
            self.uncertain_calls += 1

        self.committed_usd += spend
        self.cache_hit_tokens += record.cache_hit_tokens
        self.cache_miss_tokens += record.cache_miss_tokens
        self.output_tokens += record.output_tokens
        entry = {
            "call": self.calls,
            "reserved_usd": round(held.usd, 6),
            "spend_usd": round(spend, 6),
            "billed_from": "usage" if record.reported else "reservation",
            **record.to_dict(),
        }
        self.ledger.append(entry)
        return entry

    @property
    def cache_hit_rate(self) -> float | None:
        """Share of input tokens served from the provider's prefix cache.

        `None` until some input has been counted. This is the number the
        append-only transcript exists to move, so it is reported rather than
        assumed.
        """
        total = self.cache_hit_tokens + self.cache_miss_tokens
        if total == 0:
            return None
        return round(self.cache_hit_tokens / total, 4)

    def to_dict(self) -> dict:
        return {
            "cap_usd": self.cap_usd,
            "model": self.model,
            "committed_usd": round(self.committed_usd, 6),
            "remaining_usd": round(self.remaining_usd, 6),
            "calls": self.calls,
            "uncertain_calls": self.uncertain_calls,
            "cache_hit_tokens": self.cache_hit_tokens,
            "cache_miss_tokens": self.cache_miss_tokens,
            "output_tokens": self.output_tokens,
            "cache_hit_rate": self.cache_hit_rate,
            "accounting": (
                "conservative: peak rates from a snapshotted table, worst-case "
                "reservation before dispatch, unmeasured calls charged at their "
                "reservation. Not a provider invoice"
            ),
            "ledger": list(self.ledger),
        }


@dataclass
class RunClock:
    """The wall-clock deadline, started at first gameplay rather than at launch.

    A0.3: "The 30-minute run clock starts immediately before the first gameplay
    observation after engine readiness." Worker launch and the paused
    finalisation at the end are outside gameplay and are recorded separately, so
    a slow engine start does not eat the agent's time.
    """

    limit_seconds: float
    started_at: float | None = None
    stopped_at: float | None = None

    def start(self) -> None:
        if self.started_at is None:
            self.started_at = time.perf_counter()

    def stop(self) -> None:
        if self.started_at is not None and self.stopped_at is None:
            self.stopped_at = time.perf_counter()

    @property
    def elapsed_seconds(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.stopped_at if self.stopped_at is not None else time.perf_counter()
        return end - self.started_at

    @property
    def remaining_seconds(self) -> float:
        return max(self.limit_seconds - self.elapsed_seconds, 0.0)

    def expired(self) -> bool:
        """True once the deadline has passed. Before `start()` it never has."""
        return self.started_at is not None and self.elapsed_seconds >= self.limit_seconds

    def deadline_for_request(self, ceiling: float) -> float:
        """A per-request timeout that cannot outlive the run.

        A0.3: "A provider request and every tool wait inherit the remaining
        deadline." Without this a 180-second provider timeout can run 150 seconds
        past a deadline that has already passed.
        """
        if self.started_at is None:
            return ceiling
        return max(min(ceiling, self.remaining_seconds), 0.0)

    def to_dict(self) -> dict:
        return {
            "limit_seconds": self.limit_seconds,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "remaining_seconds": round(self.remaining_seconds, 3),
            "expired": self.expired(),
            "started": self.started_at is not None,
            "measures": "gameplay only; worker launch and final snapshot excluded",
        }
