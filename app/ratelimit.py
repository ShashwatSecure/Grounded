"""A minimal file-persisted rate limiter for the CLI `ask` command.

Each CLI invocation is a new process, so the limiter persists recent call
timestamps to a small JSON file under the index directory to enforce a
sliding-window limit across invocations.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from app.errors import GroundedError


class RateLimitExceededError(GroundedError):
    """Raised when the configured number of calls per window is exceeded."""


class RateLimiter:
    def __init__(self, state_path: Path, max_calls: int, window_seconds: float = 60.0) -> None:
        self.state_path = state_path
        self.max_calls = max_calls
        self.window_seconds = window_seconds

    def check_and_record(self, now: float | None = None) -> None:
        """Raise RateLimitExceededError if over the limit; otherwise record this call."""
        now = time.time() if now is None else now
        timestamps = self._load()
        cutoff = now - self.window_seconds
        timestamps = [t for t in timestamps if t >= cutoff]

        if len(timestamps) >= self.max_calls:
            raise RateLimitExceededError(
                f"Rate limit exceeded: max {self.max_calls} requests per "
                f"{int(self.window_seconds)}s. Please wait and try again."
            )

        timestamps.append(now)
        self._save(timestamps)

    def _load(self) -> list[float]:
        if not self.state_path.exists():
            return []
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []

    def _save(self, timestamps: list[float]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(timestamps), encoding="utf-8")
