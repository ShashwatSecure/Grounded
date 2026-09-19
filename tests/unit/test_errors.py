from __future__ import annotations

from app.errors import ConfigError, sanitize_message


def test_sanitize_message_redacts_api_key_assignment() -> None:
    message = "failed to init client with api_key=sk-abcdef1234567890"
    sanitized = sanitize_message(message)
    assert "sk-abcdef1234567890" not in sanitized
    assert "REDACTED" in sanitized


def test_sanitize_message_redacts_authorization_bearer() -> None:
    message = "request failed: Authorization: Bearer abcdef123456789"
    sanitized = sanitize_message(message)
    assert "abcdef123456789" not in sanitized


def test_sanitize_message_redacts_groq_key_prefix() -> None:
    message = "client error near gsk_ABCDEF1234567890 in headers"
    sanitized = sanitize_message(message)
    assert "gsk_ABCDEF1234567890" not in sanitized


def test_sanitize_message_leaves_normal_text_untouched() -> None:
    message = "Indexed 12 documents and 184 chunks."
    assert sanitize_message(message) == message


def test_grounded_error_sanitizes_on_construction() -> None:
    err = ConfigError("bad config: api_key=super-secret-value")
    assert "super-secret-value" not in str(err)
