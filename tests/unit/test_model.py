from __future__ import annotations

import pytest

from app.errors import (
    ModelAuthError,
    ModelRetryExhaustedError,
)
from app.generation.model import (
    ChatMessage,
    FakeChatModel,
    _classify_provider_error,
    _RetryableModelError,
)


def test_fake_chat_model_returns_canned_response() -> None:
    model = FakeChatModel(response_text="Hello from fake model.")
    result = model.generate([ChatMessage(role="user", content="hi")])
    assert result.text == "Hello from fake model."
    assert result.usage.input_tokens == 10
    assert len(model.calls) == 1


def test_fake_chat_model_uses_responder_callback() -> None:
    model = FakeChatModel(responder=lambda messages: f"echo:{messages[-1].content}")
    result = model.generate([ChatMessage(role="user", content="ping")])
    assert result.text == "echo:ping"


def test_fake_chat_model_raises_injected_error() -> None:
    model = FakeChatModel(raise_error=ModelAuthError("bad key"))
    with pytest.raises(ModelAuthError):
        model.generate([ChatMessage(role="user", content="hi")])


@pytest.mark.parametrize(
    ("message", "expected_type"),
    [
        ("Request timed out after 30s", _RetryableModelError),
        ("401 Unauthorized: invalid api key", ModelAuthError),
        ("429 Too Many Requests: rate limit exceeded", _RetryableModelError),
        ("503 Service Unavailable", _RetryableModelError),
        ("Some other failure", type(_classify_provider_error(Exception("Some other failure")))),
    ],
)
def test_classify_provider_error(message: str, expected_type: type) -> None:
    error = _classify_provider_error(Exception(message))
    assert isinstance(error, expected_type)


def test_model_retry_exhausted_error_message_is_sanitized() -> None:
    err = ModelRetryExhaustedError("failed with api_key=gsk_supersecretvalue123")
    assert "gsk_supersecretvalue123" not in str(err)
    assert "REDACTED" in str(err)
