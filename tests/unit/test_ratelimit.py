from __future__ import annotations

from pathlib import Path

import pytest

from app.ratelimit import RateLimiter, RateLimitExceededError


def test_rate_limiter_allows_calls_within_limit(tmp_path: Path) -> None:
    limiter = RateLimiter(state_path=tmp_path / "rl.json", max_calls=3, window_seconds=60)
    for _ in range(3):
        limiter.check_and_record(now=1000.0)


def test_rate_limiter_blocks_calls_over_limit(tmp_path: Path) -> None:
    limiter = RateLimiter(state_path=tmp_path / "rl.json", max_calls=2, window_seconds=60)
    limiter.check_and_record(now=1000.0)
    limiter.check_and_record(now=1000.0)
    with pytest.raises(RateLimitExceededError):
        limiter.check_and_record(now=1000.0)


def test_rate_limiter_allows_again_after_window_expires(tmp_path: Path) -> None:
    limiter = RateLimiter(state_path=tmp_path / "rl.json", max_calls=1, window_seconds=10)
    limiter.check_and_record(now=1000.0)
    with pytest.raises(RateLimitExceededError):
        limiter.check_and_record(now=1005.0)
    limiter.check_and_record(now=1011.0)  # outside the 10s window, allowed again


def test_rate_limiter_persists_state_across_instances(tmp_path: Path) -> None:
    state_path = tmp_path / "rl.json"
    RateLimiter(state_path=state_path, max_calls=1, window_seconds=60).check_and_record(
        now=1000.0
    )
    second_limiter = RateLimiter(state_path=state_path, max_calls=1, window_seconds=60)
    with pytest.raises(RateLimitExceededError):
        second_limiter.check_and_record(now=1010.0)
