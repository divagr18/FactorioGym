"""Turning a provider response into a validated typed action (PLAN.md 5.3).

PLAN 5.3's first acceptance criterion is that "provider responses become
validated typed actions", and its second is that "malformed outputs produce
bounded retries or a recorded failure". Both need the same thing first: the
ways a response can fail to become an action must be *distinguishable*, because
"the model is bad at this task", "the prompt does not describe the format",
"the model keeps choosing masked actions" and "the endpoint is down" call for
four different fixes and produce four identical-looking stalled runs otherwise.

So parsing returns one of two values, and a failure names its kind:

``unparseable``
    Nothing in the text identified an action at all. Usually a prompt problem.
``unknown_action``
    A name was given and the catalog does not contain it. The model invented an
    action -- often a plausible Factorio verb that this catalog does not expose.
``out_of_range``
    An integer was given outside the action space. The model is indexing a
    catalog it imagined rather than the one it was shown.
``illegal_action``
    A real, in-range action that the environment's mask currently forbids. The
    model ignored the legality list it was given; the environment would reject
    it, and PLAN section 2 forbids silently turning it into a different action.

A provider-level failure -- timeout, HTTP error, empty response -- is
deliberately *not* in this enum. It is carried by ``ModelReply.error`` and
recorded as ``provider_error``, because a call that never returned text is not
the model getting the format wrong.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from factoriorl.agent.summary import LegalAction


class DecisionFailure(StrEnum):
    """Why a response did not become an action."""

    UNPARSEABLE = "unparseable"
    UNKNOWN_ACTION = "unknown_action"
    OUT_OF_RANGE = "out_of_range"
    ILLEGAL_ACTION = "illegal_action"
    #: The call itself failed. Not produced here; recorded by the loop so that
    #: one vocabulary covers every reason a decision did not happen.
    PROVIDER_ERROR = "provider_error"


@dataclass(frozen=True)
class ParsedAction:
    index: int
    key: str
    #: Whatever the model said about why, kept verbatim for the replay record.
    reason: str = ""

    def to_dict(self) -> dict:
        return {"index": self.index, "key": self.key, "reason": self.reason}


@dataclass(frozen=True)
class ParseFailure:
    failure: DecisionFailure
    detail: str
    #: What was extracted before validation rejected it, when anything was.
    candidate: str | None = None

    def to_dict(self) -> dict:
        return {"failure": self.failure.value, "detail": self.detail, "candidate": self.candidate}


#: A fenced or bare JSON object anywhere in the response. Models wrap JSON in
#: prose and in code fences no matter how the prompt is worded, so the parser
#: reads the *last* object in the text: when a model restates the format example
#: before answering, the first object is the example and taking it would execute
#: the instructions instead of the decision.
_OBJECT = re.compile(r"\{[^{}]*\}", re.DOTALL)

#: The fallback shape, for a model that answers in a line rather than JSON.
_LINE = re.compile(r"\baction\b\s*[:=]\s*\"?([A-Za-z0-9_\-]+)\"?", re.IGNORECASE)


def _extract(text: str) -> tuple[Any, str] | None:
    """Pull the action field out of a response, or ``None``."""
    for match in reversed(_OBJECT.findall(text or "")):
        try:
            payload = json.loads(match)
        except ValueError:
            continue
        if isinstance(payload, dict) and "action" in payload:
            return payload["action"], str(payload.get("reason") or "")
    line = _LINE.search(text or "")
    if line:
        return line.group(1), ""
    return None


def parse_action(
    text: str,
    legal: tuple[LegalAction, ...],
    vocabulary: tuple[tuple[str, str], ...],
) -> ParsedAction | ParseFailure:
    """Validate one response against the catalog and the current mask.

    ``vocabulary`` is every action index that exists; ``legal`` is the subset
    the environment will currently accept. Both are needed to tell "you invented
    an action" from "that action exists but is not available right now", and
    that distinction is the difference between a prompt bug and a masking bug.
    """
    extracted = _extract(text)
    if extracted is None:
        return ParseFailure(
            DecisionFailure.UNPARSEABLE,
            "no JSON object with an 'action' field and no 'action:' line",
        )
    raw, reason = extracted
    legal_by_index = {a.index: a for a in legal}
    keys = [key for key, _ in vocabulary]

    # An integer, or a string that is entirely digits: the model chose by index.
    index: int | None = None
    if isinstance(raw, bool):
        # `bool` is an `int` in Python, and `{"action": true}` reaching the
        # environment as action 1 would be a silent misexecution.
        return ParseFailure(DecisionFailure.UNPARSEABLE, "action was a boolean", str(raw))
    if isinstance(raw, int):
        index = raw
    elif isinstance(raw, str) and raw.strip().lstrip("-").isdigit():
        index = int(raw.strip())
    elif not isinstance(raw, str):
        return ParseFailure(
            DecisionFailure.UNPARSEABLE,
            f"action was {type(raw).__name__}, not a name or an index",
            str(raw),
        )

    if index is None:
        name = raw.strip()
        if name not in keys:
            return ParseFailure(
                DecisionFailure.UNKNOWN_ACTION,
                f"{name!r} is not in the action catalog",
                name,
            )
        index = keys.index(name)

    if not 0 <= index < len(vocabulary):
        return ParseFailure(
            DecisionFailure.OUT_OF_RANGE,
            f"index {index} is outside the action space of {len(vocabulary)}",
            str(index),
        )
    if index not in legal_by_index:
        return ParseFailure(
            DecisionFailure.ILLEGAL_ACTION,
            f"action {index} ({keys[index]}) is masked out in this state",
            keys[index],
        )
    return ParsedAction(index=index, key=keys[index], reason=reason)
