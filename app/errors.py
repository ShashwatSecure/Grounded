"""Application-level error types and secret-safe error sanitization."""

from __future__ import annotations

import re
from re import Match

_SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key\s*[:=]\s*)(\S+)"),
    re.compile(r"(?i)(authorization:\s*bearer\s+)(\S+)"),
    re.compile(r"(?i)(bearer\s+)([A-Za-z0-9\-_.]{8,})"),
    re.compile(r"(gsk_[A-Za-z0-9]+)"),  # Groq API key prefix
]


def _redact(match: Match[str]) -> str:
    groups = match.groups()
    if len(groups) >= 2:
        return f"{groups[0]}***REDACTED***"
    return "***REDACTED***"


def sanitize_message(message: str) -> str:
    """Redact API keys, bearer tokens, and similar secrets from a message."""
    sanitized = message
    for pattern in _SECRET_PATTERNS:
        sanitized = pattern.sub(_redact, sanitized)
    return sanitized


class GroundedError(Exception):
    """Base class for all application errors. Messages are sanitized on creation."""

    def __init__(self, message: str) -> None:
        super().__init__(sanitize_message(message))


class ConfigError(GroundedError):
    """Raised when application configuration is invalid or missing."""


class DocumentLoadError(GroundedError):
    """Raised when a document cannot be discovered, read, or decoded."""


class EmbeddingError(GroundedError):
    """Raised when embedding generation fails."""


class VectorStoreError(GroundedError):
    """Raised when the vector store cannot be read, written, or queried."""


class RetrievalError(GroundedError):
    """Raised when retrieval fails for reasons other than empty results."""


class ModelError(GroundedError):
    """Base class for chat-model provider errors."""


class ModelTimeoutError(ModelError):
    """Raised when a chat-model call exceeds its configured timeout."""


class ModelAuthError(ModelError):
    """Raised when the chat-model provider rejects credentials."""


class ModelRateLimitError(ModelError):
    """Raised when the chat-model provider reports rate limiting."""


class ModelRetryExhaustedError(ModelError):
    """Raised when all bounded retry attempts have been exhausted."""


class AnswerError(GroundedError):
    """Raised when a grounded answer cannot be produced or validated."""
