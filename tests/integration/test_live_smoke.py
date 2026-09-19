"""Live smoke test: verifies real connectivity to the Groq API.

Skipped by default. Run explicitly with:

    RUN_LIVE_TESTS=1 python -m pytest tests/integration/test_live_smoke.py -m live -s

Requires GROQ_API_KEY to be set in the environment. This test never prints the
API key, the full prompt, or the full response — only success/failure,
provider error category, latency, and token usage.
"""

from __future__ import annotations

import os

import pytest

from app.config import load_settings
from app.errors import ModelError
from app.generation.model import ChatMessage, GroqChatModel

pytestmark = pytest.mark.live


@pytest.mark.skipif(
    os.environ.get("RUN_LIVE_TESTS") != "1",
    reason="Live tests are opt-in; set RUN_LIVE_TESTS=1 to run.",
)
def test_live_groq_smoke_test() -> None:
    settings = load_settings()
    api_key = settings.require_groq_api_key()

    model = GroqChatModel(
        api_key=api_key,
        model=settings.groq_model,
        timeout_seconds=settings.model_timeout_seconds,
        max_tokens=16,
        max_retries=1,
    )

    try:
        response = model.generate(
            [ChatMessage(role="user", content="Reply with the single word: pong")]
        )
    except ModelError as exc:
        print(f"live_smoke_test outcome=failed error_type={type(exc).__name__}")
        raise

    print(
        "live_smoke_test outcome=success "
        f"latency_ms={response.latency_ms:.1f} "
        f"input_tokens={response.usage.input_tokens} "
        f"output_tokens={response.usage.output_tokens}"
    )
    assert response.text
