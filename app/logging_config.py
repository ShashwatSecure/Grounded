"""Structured JSON logging configuration.

Uses the standard library `logging` module with a JSON formatter rather than a
third-party dependency such as structlog, to keep the dependency footprint
small. Secret-like values (API keys, bearer tokens) are redacted before any
record is emitted.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any

from app.errors import sanitize_message

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()) | {"message"}


class JsonFormatter(logging.Formatter):
    """Formats each log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": sanitize_message(record.getMessage()),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = sanitize_message(value) if isinstance(value, str) else value
        if record.exc_info:
            payload["exc_info"] = sanitize_message(self.formatException(record.exc_info))
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging to emit single-line JSON records to stdout."""
    root = logging.getLogger()
    root.setLevel(level.upper())
    root.handlers.clear()
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)


def log_event(logger: logging.Logger, level: int, event: str, **fields: Any) -> None:
    """Emit a structured event log line with arbitrary extra fields.

    Example: log_event(logger, logging.INFO, "retrieval.completed", retrieved_count=3)
    """
    logger.log(level, event, extra=fields)
