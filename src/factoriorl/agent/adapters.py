"""Provider-independent model adapters (PLAN.md 5.3).

PLAN 5.3 asks for "a provider-independent agent loop with adapters for a
configurable local inference endpoint and optional API providers", and lists as
an acceptance criterion that "local and API models use the same observation and
action contracts". The way that is made *structurally* true rather than merely
intended is the shape of this interface: an adapter receives a
:class:`ModelRequest` -- two strings and a token ceiling -- and returns a
:class:`ModelReply` -- one string plus measurement. An adapter never sees the
environment, the observation, the catalog or the mask, so it cannot acquire a
private channel to the game; and it never returns an action, so every provider's
output goes through the same validation in :mod:`factoriorl.agent.parsing`.

No SDK is imported. ``pyproject.toml`` separates a PEP 621 ``rl`` extra from a
PEP 735 ``train`` group precisely so the base install stays small, and an agent
loop that dragged a provider SDK into it would undo that. Both HTTP adapters use
``urllib`` from the standard library.

**Latency is measured around every call and reported even when the call fails.**
A provider that times out is a result, not an absence of one: PLAN 5.3 wants
inference latency recorded, and a record that silently omitted the slow failures
would misreport exactly the cases anyone reads the record to find.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from factoriorl.agent.credentials import read_credential, redactor

#: Default ceiling on a decision response. The model is asked for one small JSON
#: object, but this is not the place to be clever: an adaptive-thinking model
#: spends output tokens before it writes any text, so a cap tuned to the size of
#: the answer returns a truncated message with no text block at all and every
#: decision is recorded as unparseable output rather than as the truncation it
#: actually was.
DEFAULT_MAX_TOKENS = 2048

#: Floor applied by API adapters whose models think before answering, for the
#: reason above.
THINKING_MAX_TOKENS_FLOOR = 1024

#: Per-call HTTP timeout. Bounded because a decision loop that blocks forever on
#: one provider stall produces neither a result nor a failure, and the run
#: artifact would end mid-episode with nothing saying why.
DEFAULT_TIMEOUT_SECONDS = 60.0


@dataclass(frozen=True)
class ModelRequest:
    """Everything an adapter is given. Deliberately provider-neutral."""

    system: str
    user: str
    max_tokens: int = DEFAULT_MAX_TOKENS


@dataclass(frozen=True)
class ModelReply:
    """Everything an adapter returns, including the measurement.

    ``error`` set means the *call* failed -- transport, timeout, HTTP status, or
    a response body that was not the shape the provider documents. That is a
    different thing from a call that succeeded and returned text the loop could
    not turn into an action, and the two are kept distinguishable all the way
    into the run artifact.
    """

    text: str
    latency_ms: float
    usage: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    error_kind: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "latency_ms": round(self.latency_ms, 3),
            "usage": self.usage,
            "meta": self.meta,
            "error": self.error,
            "error_kind": self.error_kind,
        }


class ModelAdapter(ABC):
    """One provider behind the contract every provider shares."""

    #: Short name recorded in the run manifest.
    name: str = "adapter"

    @abstractmethod
    def complete(self, request: ModelRequest) -> ModelReply:
        """Answer one request. Must not raise: a failed call is a ``ModelReply``
        carrying ``error`` and a measured latency, because the caller needs the
        failure *and* its cost in the record."""

    @abstractmethod
    def describe(self) -> dict:
        """Configuration for the run manifest. Must contain no credential.

        Implementations build this from non-secret fields only and record the
        *name* of the environment variable a key was read from plus whether one
        was present, which is what a later reader actually needs in order to
        reproduce the run.
        """

    def redact(self, text: str) -> str:
        """Remove this adapter's own credentials from text about to be recorded.

        Default is identity because an adapter with no credential has nothing to
        remove. Adapters that hold one override it via
        :func:`factoriorl.agent.credentials.redactor`.
        """
        return text


# ------------------------------------------------------------------- scripted


class ScriptedAdapter(ModelAdapter):
    """A deterministic offline adapter -- part of the package, not a fixture.

    Every test of this package runs with no network and no API key, and CI has
    neither. A scripted adapter is therefore how the loop is exercised at all,
    which makes it a first-class member of the interface rather than something
    that belongs in ``tests/``. It is also the honest way to demonstrate the
    provider-independence claim: if a scripted adapter, a local endpoint and an
    API provider are interchangeable in the loop, the contract is real.

    ``script`` is either a sequence of replies or a callable taking the request
    and the zero-based call index. A sequence entry may be:

    * a ``str``: returned as the model's text;
    * an ``Exception``: raised inside the call, so the loop sees a provider
      failure with a measured latency -- how a timeout is rehearsed offline.

    Running off the end of a sequence is reported as a provider error rather
    than silently repeating the last reply, because a test whose script ran out
    should fail loudly instead of looping on a stale answer.
    """

    name = "scripted"

    def __init__(
        self,
        script: Sequence[str | Exception] | Callable[[ModelRequest, int], str],
        *,
        model: str = "scripted",
        usage: dict[str, Any] | None = None,
    ) -> None:
        self.script = script
        self.model = model
        self.usage = usage or {}
        self.calls = 0
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelReply:
        started = time.perf_counter()
        index = self.calls
        self.calls += 1
        self.requests.append(request)
        try:
            if callable(self.script):
                text = self.script(request, index)
            else:
                if index >= len(self.script):
                    raise IndexError(f"scripted adapter exhausted after {len(self.script)} calls")
                entry = self.script[index]
                if isinstance(entry, Exception):
                    raise entry
                text = entry
        except Exception as exc:  # noqa: BLE001 - a provider failure is a result
            return ModelReply(
                text="",
                latency_ms=(time.perf_counter() - started) * 1000.0,
                error=f"{type(exc).__name__}: {exc}",
                error_kind="scripted",
                meta={"call_index": index},
            )
        return ModelReply(
            text=text,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            usage=dict(self.usage),
            meta={"call_index": index, "model": self.model},
        )

    def describe(self) -> dict:
        return {"adapter": self.name, "model": self.model, "offline": True}


# ----------------------------------------------------------------- HTTP base


class _HTTPAdapter(ModelAdapter):
    """Shared JSON-over-HTTP plumbing for the two real providers.

    ``urllib`` rather than ``requests`` or a provider SDK: adding a runtime
    dependency to the base install to send one POST would be a poor trade
    against the packaging discipline ``pyproject.toml`` already keeps.
    """

    def __init__(self, *, api_key_env: str | None, timeout: float) -> None:
        self.api_key_env = api_key_env
        self.timeout = timeout
        # Name-mangled and never returned by `describe`. The loop holds the
        # adapter, so a plainly named attribute would be one `vars()` away from
        # the run artifact.
        self.__api_key = read_credential(api_key_env)
        self.__redact = redactor([self.__api_key])

    @property
    def _api_key(self) -> str | None:
        return self.__api_key

    @property
    def has_credential(self) -> bool:
        return bool(self.__api_key)

    def redact(self, text: str) -> str:
        return self.__redact(text)

    def _post(self, url: str, headers: dict[str, str], body: dict) -> tuple[dict, str | None, str]:
        """POST JSON and return ``(payload, error, error_kind)``.

        Errors are values, not exceptions, so the caller can attach the latency
        it measured around this call to the failure itself.
        """
        data = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            # The body of a 4xx often explains the refusal; it is also the most
            # likely place for a provider to echo the credential back, so it is
            # redacted before it becomes part of the record.
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:500]
            except OSError:
                detail = ""
            return {}, self.redact(f"HTTP {exc.code}: {detail}"), "http_status"
        except urllib.error.URLError as exc:
            return {}, self.redact(f"{type(exc).__name__}: {exc.reason}"), "transport"
        except TimeoutError as exc:
            return {}, f"timeout after {self.timeout}s: {exc}", "timeout"
        except OSError as exc:
            return {}, self.redact(f"{type(exc).__name__}: {exc}"), "transport"
        try:
            payload = json.loads(raw)
        except ValueError as exc:
            return {}, f"provider returned non-JSON: {exc}", "malformed_response"
        if not isinstance(payload, dict):
            return {}, "provider returned a JSON value that is not an object", "malformed_response"
        return payload, None, ""


class OpenAICompatibleAdapter(_HTTPAdapter):
    """A configurable local inference endpoint (PLAN 5.3's first adapter).

    The OpenAI chat-completions shape is what local servers speak -- llama.cpp,
    vLLM, LM Studio and Ollama all expose it -- so one adapter covers the whole
    class of local endpoints without this package taking a position on which
    server the user runs. ``base_url`` is configuration, not a constant.

    A key is optional and read from the environment when a variable is named,
    because a local endpoint usually needs none and a hosted OpenAI-compatible
    gateway usually does.
    """

    name = "openai-compatible"

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8080/v1",
        model: str = "local-model",
        api_key_env: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        temperature: float | None = 0.0,
        #: Which field carries the output ceiling. Local servers -- llama.cpp,
        #: vLLM, LM Studio, Ollama -- speak `max_tokens`, and OpenAI's newer
        #: reasoning models reject it outright:
        #:   "Unsupported parameter: 'max_tokens' is not supported with this
        #:    model. Use 'max_completion_tokens' instead."
        #: Both spellings are the same OpenAI-compatible shape, so this is
        #: configuration rather than a second adapter.
        token_parameter: str = "max_tokens",
        extra_body: dict | None = None,
    ) -> None:
        super().__init__(api_key_env=api_key_env, timeout=timeout)
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.token_parameter = token_parameter
        self.extra_body = dict(extra_body or {})

    def complete(self, request: ModelRequest) -> ModelReply:
        headers = {"content-type": "application/json"}
        if self._api_key:
            headers["authorization"] = f"Bearer {self._api_key}"
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
            ],
            self.token_parameter: request.max_tokens,
            "stream": False,
            **self.extra_body,
        }
        # Zero by default: the same observation should produce the same decision
        # when a run is replayed, and sampling noise in a 600-step episode makes
        # a failure impossible to reproduce. `None` omits the field entirely,
        # for models that reject sampling parameters rather than ignoring them.
        if self.temperature is not None:
            body["temperature"] = self.temperature
        started = time.perf_counter()
        payload, error, kind = self._post(f"{self.base_url}/chat/completions", headers, body)
        latency = (time.perf_counter() - started) * 1000.0
        if error:
            return ModelReply(text="", latency_ms=latency, error=error, error_kind=kind)
        choices = payload.get("choices") or []
        if not choices:
            return ModelReply(
                text="",
                latency_ms=latency,
                usage=payload.get("usage") or {},
                error="response contained no choices",
                error_kind="malformed_response",
            )
        message = (choices[0] or {}).get("message") or {}
        return ModelReply(
            text=message.get("content") or "",
            latency_ms=latency,
            usage=payload.get("usage") or {},
            meta={
                "model": payload.get("model"),
                "finish_reason": (choices[0] or {}).get("finish_reason"),
            },
        )

    def describe(self) -> dict:
        return {
            "adapter": self.name,
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
            # The variable's *name* and whether it was set: enough to reproduce
            # the run, and not the secret itself.
            "api_key_env": self.api_key_env,
            "credential_present": self.has_credential,
        }


class AnthropicMessagesAdapter(_HTTPAdapter):
    """An API provider (PLAN 5.3's second adapter), over the Messages API.

    Raw HTTP rather than the ``anthropic`` SDK, for the packaging reason above.
    Two provider-specific details are worth naming because getting them wrong
    fails at runtime rather than at import:

    * ``temperature`` is **not** sent. Sampling parameters were removed on the
      current model family and a request carrying one is rejected with a 400, so
      an adapter that forwarded the loop's determinism preference would fail
      every call.
    * ``max_tokens`` is floored, because these models think before answering and
      the thinking is charged against the same ceiling; too small a cap returns
      a message truncated before any text block, which the loop would otherwise
      record as unparseable output rather than as truncation.
    """

    name = "anthropic-messages"

    #: Wire version header the Messages API requires on every request.
    API_VERSION = "2023-06-01"

    def __init__(
        self,
        *,
        model: str = "claude-opus-5",
        api_key_env: str = "ANTHROPIC_API_KEY",
        base_url: str = "https://api.anthropic.com",
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        extra_body: dict | None = None,
    ) -> None:
        super().__init__(api_key_env=api_key_env, timeout=timeout)
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.extra_body = dict(extra_body or {})

    def complete(self, request: ModelRequest) -> ModelReply:
        headers = {
            "content-type": "application/json",
            "anthropic-version": self.API_VERSION,
        }
        if self._api_key:
            headers["x-api-key"] = self._api_key
        body = {
            "model": self.model,
            "max_tokens": max(request.max_tokens, THINKING_MAX_TOKENS_FLOOR),
            "system": request.system,
            "messages": [{"role": "user", "content": request.user}],
            **self.extra_body,
        }
        started = time.perf_counter()
        payload, error, kind = self._post(f"{self.base_url}/v1/messages", headers, body)
        latency = (time.perf_counter() - started) * 1000.0
        if error:
            return ModelReply(text="", latency_ms=latency, error=error, error_kind=kind)
        blocks = payload.get("content") or []
        text = "".join(
            block.get("text") or ""
            for block in blocks
            if isinstance(block, dict) and block.get("type") == "text"
        )
        stop_reason = payload.get("stop_reason")
        meta = {
            "model": payload.get("model"),
            "stop_reason": stop_reason,
            "stop_details": payload.get("stop_details"),
        }
        if not text:
            # A refusal or a truncation both arrive as HTTP 200 with no usable
            # text. Naming the stop reason here is what keeps that from being
            # filed as "the model wrote nonsense".
            return ModelReply(
                text="",
                latency_ms=latency,
                usage=payload.get("usage") or {},
                meta=meta,
                error=f"response contained no text block (stop_reason={stop_reason})",
                error_kind="empty_response",
            )
        return ModelReply(
            text=text,
            latency_ms=latency,
            usage=payload.get("usage") or {},
            meta=meta,
        )

    def describe(self) -> dict:
        return {
            "adapter": self.name,
            "model": self.model,
            "base_url": self.base_url,
            "api_version": self.API_VERSION,
            "api_key_env": self.api_key_env,
            "credential_present": self.has_credential,
        }
