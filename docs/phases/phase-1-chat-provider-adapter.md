# Phase 1 — Chat Provider Adapter

## What was built

[app/generation/model.py](../../app/generation/model.py):

- `ChatMessage` / `ModelResponse`: provider-agnostic message and response
  types, so nothing outside this module ever imports `langchain_groq` or
  `langchain_core.messages` directly.
- `ChatModel` (a `Protocol`): the interface the rest of the app depends on.
- `GroqChatModel`: the production adapter. Lazily constructs `ChatGroq` (so
  importing this module never requires network access), converts
  `ChatMessage` to `langchain_core` message types, applies a **timeout**
  (`timeout_seconds`), a **max token** budget, and a **bounded retry policy**
  via `tenacity` (`stop_after_attempt`, `wait_exponential`) that only retries
  a private `_RetryableModelError` marker.
- `_classify_provider_error()`: maps raw provider exceptions to
  `ModelAuthError` (never retried), `ModelTimeoutError`/rate-limit/5xx
  (wrapped as retryable), or a generic `ModelError`.
- `FakeChatModel`: a deterministic offline double used by every other test in
  the suite — returns a canned string, an injected `responder` callback, or
  raises an injected error, and records every call in `self.calls`.

## Why these choices

- **Interface-first design**: the plan requires "Unit tests use a fake model
  and do not call Groq." Putting a `Protocol` between the app and
  `langchain-groq` means `Answerer`, the CLI, and the evaluation harness never
  need to know which implementation they're talking to.
- **Retries only for transient failures**: authentication errors
  (`ModelAuthError`) and generic validation failures are *not* wrapped in the
  retryable marker, so they fail immediately instead of wasting time retrying
  something that can never succeed — this satisfies the Reliability
  Requirement "Non-retryable authentication and validation failures fail
  immediately."
- **Error classification by message content**: `langchain-groq`/the
  underlying HTTP client don't expose a single stable exception taxonomy
  across all failure modes, so classifying by status code/keywords in the
  message is the most robust cross-version approach, and the message itself
  is sanitized by `ModelError`'s `GroundedError` base class before it can
  leak upstream.
- **`FakeChatModel.calls` recording**: lets tests assert not just the output
  but *whether the model was called at all* (e.g. it must never be called
  when there's no retrieved context).

## How it was verified

- [tests/unit/test_model.py](../../tests/unit/test_model.py): covers
  `FakeChatModel` canned/callback/error behavior and `_classify_provider_error`
  classification for timeout / 401 / 429 / 5xx / unknown messages.
- [tests/unit/test_errors.py](../../tests/unit/test_errors.py):
  `ModelRetryExhaustedError` messages containing a fake Groq key are
  confirmed redacted.
- [tests/integration/test_live_smoke.py](../../tests/integration/test_live_smoke.py):
  opt-in only (`RUN_LIVE_TESTS=1 ... -m live`), uses a real `GROQ_API_KEY`,
  sends a minimal one-word prompt, and prints only success/failure, error
  category, latency, and token usage — never the key or the full
  prompt/response.

## Exit criteria status

- [x] A live smoke test can ask one question (opt-in, documented above).
- [x] Unit tests use a fake model and do not call Groq.
- [x] Provider failures do not expose keys or raw headers (sanitized via
      `GroundedError`/`sanitize_message`).
