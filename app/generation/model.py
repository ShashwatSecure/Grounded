"""Chat-model adapters.

The application depends only on the `ChatModel` interface. `GroqChatModel` is
the production adapter (wrapping `langchain-groq`'s `ChatGroq`), and
`FakeChatModel` is a deterministic, offline test double. Nothing outside this
module should import `langchain_groq` directly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.domain import Usage
from app.errors import (
    ModelAuthError,
    ModelError,
    ModelRetryExhaustedError,
)


@dataclass(frozen=True)
class ChatMessage:
    """A single role/content message, independent of any provider SDK."""

    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class ModelResponse:
    text: str
    usage: Usage
    latency_ms: float


class ChatModel(Protocol):
    """Interface every chat-model adapter must implement."""

    def generate(self, messages: list[ChatMessage]) -> ModelResponse: ...


class _RetryableModelError(ModelError):
    """Internal marker used to trigger tenacity retries; never raised to callers."""


@dataclass
class GroqChatModel:
    """Chat model adapter backed by Groq via `langchain-groq`.

    Retries apply only to transient failures (timeouts, rate limits, server
    errors). Authentication and validation failures fail immediately.
    """

    api_key: str
    model: str
    timeout_seconds: float = 30.0
    max_tokens: int = 800
    max_retries: int = 3
    _client: object | None = field(default=None, init=False, repr=False)

    def _get_client(self):
        if self._client is None:
            try:
                from langchain_groq import ChatGroq
            except ImportError as exc:  # pragma: no cover - import guard
                raise ModelError(
                    "langchain-groq is not installed. Install project dependencies."
                ) from exc
            self._client = ChatGroq(
                api_key=self.api_key,
                model=self.model,
                timeout=self.timeout_seconds,
                max_tokens=self.max_tokens,
            )
        return self._client

    def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        start = time.monotonic()

        @retry(
            reraise=True,
            stop=stop_after_attempt(max(1, self.max_retries + 1)),
            wait=wait_exponential(multiplier=0.5, max=8),
            retry=retry_if_exception_type(_RetryableModelError),
        )
        def _call() -> ModelResponse:
            try:
                from langchain_core.messages import (
                    AIMessage,
                    BaseMessage,
                    HumanMessage,
                    SystemMessage,
                )

                lc_messages: list[BaseMessage] = []
                for m in messages:
                    if m.role == "system":
                        lc_messages.append(SystemMessage(content=m.content))
                    elif m.role == "assistant":
                        lc_messages.append(AIMessage(content=m.content))
                    else:
                        lc_messages.append(HumanMessage(content=m.content))

                client = self._get_client()
                result = client.invoke(lc_messages)
            except Exception as exc:  # normalize provider errors
                raise _classify_provider_error(exc) from exc

            usage = _extract_usage(result)
            latency_ms = (time.monotonic() - start) * 1000
            return ModelResponse(text=result.content, usage=usage, latency_ms=latency_ms)

        try:
            return _call()
        except _RetryableModelError as exc:
            raise ModelRetryExhaustedError(
                f"Model call failed after {self.max_retries + 1} attempts: {exc}"
            ) from exc


def _extract_usage(result: object) -> Usage:
    usage_meta = getattr(result, "usage_metadata", None) or {}
    input_tokens = usage_meta.get("input_tokens") if usage_meta else None
    output_tokens = usage_meta.get("output_tokens") if usage_meta else None
    return Usage(input_tokens=input_tokens, output_tokens=output_tokens)


def _classify_provider_error(exc: Exception) -> ModelError:
    """Map a raw provider exception into a bounded application error."""
    message = str(exc)
    lowered = message.lower()
    if "timeout" in lowered or "timed out" in lowered:
        return _RetryableModelError(f"Model call timed out: {message}")
    if "401" in lowered or "unauthorized" in lowered or "invalid api key" in lowered:
        return ModelAuthError(f"Model authentication failed: {message}")
    if "429" in lowered or "rate limit" in lowered:
        return _RetryableModelError(f"Model rate limited: {message}")
    if any(code in lowered for code in ("500", "502", "503", "504")):
        return _RetryableModelError(f"Model server error: {message}")
    return ModelError(f"Model call failed: {message}")


class FakeChatModel:
    """Deterministic, offline chat model for tests.

    Returns a canned response (or one produced by an injected `responder`
    callable) without making any network call.
    """

    def __init__(
        self,
        response_text: str = "Fake answer.",
        usage: Usage | None = None,
        responder=None,
        raise_error: ModelError | None = None,
    ) -> None:
        self.response_text = response_text
        self.usage = usage or Usage(input_tokens=10, output_tokens=5)
        self.responder = responder
        self.raise_error = raise_error
        self.calls: list[list[ChatMessage]] = []

    def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        self.calls.append(messages)
        if self.raise_error is not None:
            raise self.raise_error
        text = self.responder(messages) if self.responder else self.response_text
        return ModelResponse(text=text, usage=self.usage, latency_ms=1.0)


def create_groq_model(
    api_key: str,
    model: str,
    timeout_seconds: float,
    max_tokens: int,
    max_retries: int,
) -> GroqChatModel:
    """Factory used by the application to build the production chat model."""
    return GroqChatModel(
        api_key=api_key,
        model=model,
        timeout_seconds=timeout_seconds,
        max_tokens=max_tokens,
        max_retries=max_retries,
    )
