"""An append-only message history, so the provider can cache the prefix.

Why this exists
---------------
`factoriorl.pricing` records the number that forces it: for `deepseek-flash` an
input token served from the provider's prefix cache costs **$0.006 per million**
against **$0.30** for one that is not -- fifty times cheaper. Priced over a
300-decision run at ~2,500 volatile tokens a turn, an append-only conversation
costs **$1.19** at a 99.3% hit rate; rebuilding the prompt each turn costs
**$34.15**. Against the roadmap's $5 cap, prefix caching is not an optimisation.
It is the difference between a run that can happen and one that cannot.

DeepSeek's cache is automatic and matches on an **exact prefix**
(`api-docs.deepseek.com/guides/kv_cache`, retrieved 2026-09-10). There is no
parameter to set and nothing to opt into; the only thing a client controls is
whether request *N* is a byte-exact prefix of request *N+1*. That is the single
invariant this class exists to hold.

What it replaces
----------------
`AgentLoop._ask` built `ModelRequest(system=SYSTEM_PROMPT, user=<everything>)`
fresh on every attempt: the rendered observation, the memory block, and -- on a
retry -- a correction *spliced into the same user string*. Three consequences,
all of them cache-hostile:

- the volatile observation sat in the same message every time, so nothing after
  the system prompt could ever match;
- a retry rewrote a message that had already been sent, which is a prefix
  mutation, not an append;
- the model was never shown its own rejected answer, only a description of why
  it was wrong.

Here a retry appends the assistant's actual reply and then a new user turn. The
history only ever grows.

The rules, and what each one costs to break
-------------------------------------------
**Never mutate a turn that has been sent.** Editing anything before the last
message invalidates the cached prefix from that point on, so every later token
reverts to the miss rate.

**A failed *call* re-sends the identical body.** If the request never reached the
model there is nothing to append -- no assistant turn exists -- and resending
byte-for-byte is a complete cache hit. `append_rejection` is for answers that
arrived and did not validate; transport failures use `resend`.

**Never send `reasoning_content` back.** DeepSeek's thinking-mode guide is
explicit: without the `tools` parameter it "does not need to be passed back; even
if passed to the API, it will be ignored and will not be concatenated into the
context", while *with* `tools` all prior reasoning must be replayed. Our contract
is JSON-in-text and sends no `tools`, so assistant turns carry `content` only.
That keeps the appended turn small and deterministic, which is exactly what a
stable prefix needs.

**Compaction is a declared cache reset.** Dropping or summarising old turns
rewrites the prefix and throws away every cached token. It is therefore explicit,
recorded with the reason, counted, and never automatic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Roles the transcript will emit. `system` is positional and appears once.
USER = "user"
ASSISTANT = "assistant"
SYSTEM = "system"


@dataclass(frozen=True)
class Turn:
    role: str
    content: str

    def to_message(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class Transcript:
    """The message list, and the guarantee that it only ever grows.

    `static_prefix` is text that is invariant for the whole run -- the task
    brief, the action catalog, argument domains. It is placed immediately after
    the system prompt so it is cached once and never re-sent as a miss. Anything
    that changes between decisions belongs in an appended user turn instead.
    """

    system: str
    static_prefix: str = ""
    turns: list[Turn] = field(default_factory=list)
    resets: list[dict] = field(default_factory=list)
    #: Every message list this transcript has produced, as a digest. Used to
    #: prove the prefix property in tests and to spot a mutation in a live run.
    sent: list[int] = field(default_factory=list)

    # ---------------------------------------------------------------- writing

    def append_user(self, content: str) -> Turn:
        return self._append(USER, content)

    def append_assistant(self, content: str) -> Turn:
        """Record what the model actually said.

        `content` only -- never `reasoning_content`. See the module docstring.
        """
        return self._append(ASSISTANT, content)

    def append_rejection(self, answer: str, correction: str) -> None:
        """The retry shape: the model's own answer, then why it was refused.

        Telling the model what was wrong is what makes a retry different from a
        repeat -- a loop that re-sent an identical prompt once burned its whole
        retry budget on the same malformed answer three times. Doing it by
        appending rather than by rewriting is what keeps the prefix intact.
        """
        self.append_assistant(answer)
        self.append_user(correction)

    def _append(self, role: str, content: str) -> Turn:
        turn = Turn(role=role, content=content)
        self.turns.append(turn)
        return turn

    # ---------------------------------------------------------------- reading

    def messages(self) -> list[dict[str, str]]:
        """The full body to send, most-stable content first."""
        head = self.system
        if self.static_prefix:
            # One message rather than two: a provider that normalises or drops a
            # second system message would silently change the prefix, and the
            # prefix is the thing being protected.
            head = f"{head}\n\n{self.static_prefix}"
        return [{"role": SYSTEM, "content": head}, *(t.to_message() for t in self.turns)]

    def record_sent(self) -> list[dict[str, str]]:
        """Return the body and remember its shape, for the prefix assertion."""
        body = self.messages()
        self.sent.append(len(body))
        return body

    def prefix_of(self, later: list[dict[str, str]]) -> bool:
        """Whether this transcript's current body is a strict prefix of `later`."""
        current = self.messages()
        return len(current) <= len(later) and later[: len(current)] == current

    # ---------------------------------------------------------------- resetting

    def reset(self, reason: str, keep_static: bool = True) -> None:
        """Drop the history. This throws away the cache, so it is recorded.

        Not called anywhere in A0. It exists so that when compaction arrives it
        has to go through a door that counts the cost rather than appearing as an
        unexplained collapse in the cache hit rate.
        """
        self.resets.append({"reason": reason, "turns_dropped": len(self.turns)})
        self.turns.clear()
        if not keep_static:
            self.static_prefix = ""

    def to_dict(self) -> dict:
        return {
            "turns": len(self.turns),
            "static_prefix_chars": len(self.static_prefix),
            "system_chars": len(self.system),
            "requests_sent": len(self.sent),
            "cache_resets": list(self.resets),
            "discipline": (
                "append-only; a sent turn is never mutated, a failed call is "
                "resent byte-identical, and reasoning_content is never returned"
            ),
        }
