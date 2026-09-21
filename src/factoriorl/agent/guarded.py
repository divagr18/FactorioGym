"""An adapter that will not dispatch a request the run cannot afford.

the agent roadmap A0.3 wants the cap and the deadline to bind
*before* a request is sent, not to be noticed afterwards. This wraps any
:class:`~factoriorl.agent.adapters.ModelAdapter` and does exactly that, which
means the enforcement lives in one place rather than being sprinkled through the
loop.

Why a proxy rather than a change to `AgentLoop`
-----------------------------------------------
Every provider call in the package funnels through ``adapter.complete``. A proxy
therefore sees all of them -- including the ones the loop cannot see. In
particular ``OpenAICompatibleAdapter`` retries internally when a provider rejects
a parameter by name, and on its HTTP-error path it returns no ``usage`` at all,
so those retries are billed and currently invisible. Counting attempts here
rather than counting the loop's decisions is what makes the ledger honest.

The pattern is the one ``tools/watch_agent.py`` already uses for its narrating
proxy, so this is a shape the repository knows.

Refusing is a reply, not an exception
-------------------------------------
The adapter contract says ``complete`` must not raise: a failed call is a
``ModelReply`` carrying an error and a measured latency, because the caller needs
the failure *and* its cost in the record. A refusal is a failure of that kind, so
it comes back as a reply with ``error_kind`` of ``budget_exhausted`` or
``deadline_expired``. The loop then does what it does with any provider failure
-- bounded retries, then a recorded fallback -- and each retry is refused again
without a request leaving the machine.

That is the property Gate A0 asks for: *no additional request starts after budget
exhaustion*. It is asserted against the wrapped adapter's own call count, not
against a log line.
"""

from __future__ import annotations

from factoriorl.agent.adapters import ModelAdapter, ModelReply, ModelRequest
from factoriorl.agent.budget import Budget, BudgetExhausted, RunClock

#: Characters per token at the *pessimistic* end, used to turn a request into a
#: token count for the reservation. Ordinary English runs about four; two is
#: chosen so a request full of punctuation, digits or handles still reserves
#: enough. Over-reserving is close to free here -- the reservation is a hold
#: released the moment real usage arrives -- while under-reserving would let a
#: run exceed the cap it was given, which is the failure that matters.
PESSIMISTIC_CHARS_PER_TOKEN = 2.0


def estimate_input_tokens(request: ModelRequest) -> int:
    """A deliberately high estimate of the tokens this request will be billed.

    A0.3 asks for a "safe input-token upper bound", and says to fail rather than
    guess when one cannot be established. Characters are the only thing available
    before dispatch without importing a tokenizer for each provider, so this
    counts them and divides by a pessimistic ratio. The role played is a *bound*,
    not a measurement: the real count comes back in ``usage`` and replaces this
    entirely.
    """
    if request.messages:
        characters = sum(len(message.get("content") or "") for message in request.messages)
    else:
        characters = len(request.system) + len(request.user)
    return int(characters / PESSIMISTIC_CHARS_PER_TOKEN) + 1


class BudgetedAdapter(ModelAdapter):
    """Wrap an adapter so a spend cap and a run deadline bind before dispatch."""

    def __init__(
        self,
        inner: ModelAdapter,
        budget: Budget,
        clock: RunClock | None = None,
    ) -> None:
        self.inner = inner
        self.budget = budget
        self.clock = clock
        #: Requests this proxy declined to send, by reason. The count that
        #: matters for Gate A0 is on the *inner* adapter; this is for the record.
        self.refusals: list[dict] = []

    @property
    def name(self) -> str:  # type: ignore[override]
        # The manifest should say which provider ran, not that a wrapper did.
        return self.inner.name

    def complete(self, request: ModelRequest) -> ModelReply:
        if self.clock is not None and self.clock.expired():
            return self._refuse(
                "deadline_expired",
                f"the {self.clock.limit_seconds:.0f}s run clock expired "
                f"{self.clock.elapsed_seconds - self.clock.limit_seconds:.1f}s ago, "
                f"so this request was not sent",
            )

        tokens = estimate_input_tokens(request)
        try:
            self.budget.reserve(tokens)
        except BudgetExhausted as exhausted:
            return self._refuse("budget_exhausted", str(exhausted))

        # A request must not outlive the run it belongs to. The adapters keep
        # their timeout as a plain attribute, so it is narrowed for this call and
        # put back afterwards -- a 180-second provider timeout would otherwise
        # run well past a deadline that has already passed.
        restore = None
        if self.clock is not None and hasattr(self.inner, "timeout"):
            restore = self.inner.timeout
            self.inner.timeout = self.clock.deadline_for_request(restore)
        reply: ModelReply | None = None
        try:
            reply = self.inner.complete(request)
        finally:
            if restore is not None:
                self.inner.timeout = restore
            # Settled in `finally` so an adapter that raises -- against its own
            # contract -- cannot leave a hold outstanding and wedge every later
            # request behind a reservation that never clears. An unmeasured call
            # is charged at its reservation, which is what `settle(None)` does.
            if self.budget.outstanding is not None:
                self.budget.settle(reply.usage if reply is not None else None)
        return reply

    def _refuse(self, kind: str, message: str) -> ModelReply:
        self.refusals.append({"kind": kind, "detail": message})
        return ModelReply(text="", latency_ms=0.0, error=message, error_kind=kind)

    def describe(self) -> dict:
        described = dict(self.inner.describe())
        described["limits"] = {
            "budget": self.budget.to_dict(),
            "clock": self.clock.to_dict() if self.clock else None,
            "refusals": list(self.refusals),
            "input_token_estimate": (
                f"characters / {PESSIMISTIC_CHARS_PER_TOKEN} before dispatch, "
                f"replaced by the provider's reported usage on settlement"
            ),
        }
        return described

    def redact(self, text: str) -> str:
        # The credential belongs to the wrapped adapter; this proxy never sees it.
        return self.inner.redact(text)
