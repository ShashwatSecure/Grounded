# Phase 8 — Production Readiness Baseline

## What was built

- **Structured logs with correlation/retrieval IDs**: already established in
  Phases 0/4 (`app/logging_config.py`'s JSON formatter; `Retriever` mints a
  `retrieval_id` per call). The CLI's `ask` command additionally logs an
  `ask.completed`/`ask.failed` event with `outcome`, `latency_ms`,
  `source_count`, and token usage — never the question text or answer text.
- **Latency and token metrics**: `Answer.latency_ms` and `Answer.usage`
  (input/output tokens, when the provider returns them) are computed in
  `Answerer` and surfaced in both the CLI log event and
  `scripts/evaluate.py`'s summary (P50/P95 latency, average tokens).
- **Retry with exponential backoff and a retry limit**: `GroqChatModel`
  (Phase 1) via `tenacity.wait_exponential` + `stop_after_attempt`; exhausting
  retries raises `ModelRetryExhaustedError` rather than looping forever.
- **Model timeout handling**: `GroqChatModel.timeout_seconds` is passed
  straight into `ChatGroq(timeout=...)`.
- **Input and output size limits**: `MAX_INPUT_CHARACTERS` is enforced in the
  CLI's `ask` command before any retrieval/model call happens;
  `MODEL_MAX_TOKENS` bounds output size.
- **Rate limiting for the CLI**
  ([app/ratelimit.py](../../app/ratelimit.py)): a small file-persisted
  sliding-window `RateLimiter` (`RATE_LIMIT_PER_MINUTE`, default 20/min),
  since each CLI invocation is a separate process with no shared in-memory
  state. Wired into the `ask` command before the chat model is invoked, and
  raises a `RateLimitExceededError` (a `GroundedError`) when exceeded.
- **Health/readiness checks** (`grounded doctor` /
  `python -m app.cli doctor`): reports, without ever leaking secret values,
  whether configuration loaded, whether `DOCUMENTS_PATH` exists, whether an
  index is present and non-empty, and whether `GROQ_API_KEY` is set — exits
  non-zero if anything is not ready, so it can be used as a scripted
  readiness gate.
- **CI workflow** ([.github/workflows/ci.yml](../../.github/workflows/ci.yml)):
  on every push/PR to `master`, installs the project, runs `ruff check .`,
  `mypy app`, the offline test suite (`-m "not live"`), a `pip-audit --strict`
  dependency vulnerability scan, and a `gitleaks` secret scan.
- **Linting and type checking config**: `[tool.ruff]`/`[tool.ruff.lint]` and
  `[tool.mypy]` in `pyproject.toml` (already added in Phase 0, exercised here).

## Why these choices

- **File-based rate limiter instead of an in-memory one**: a CLI process
  exits after each command, so an in-memory limiter would reset on every
  invocation and enforce nothing. Persisting a small JSON timestamp list
  alongside the vector index (same directory convention as conversation
  history) makes the limit meaningful across repeated `ask` invocations
  without requiring a database or daemon.
- **`doctor` as a CLI command, not a background job**: this is a CLI
  application, not a long-running service, so "health and readiness checks in
  the application layer" is implemented as an explicit, scriptable diagnostic
  command rather than an HTTP `/healthz` endpoint (which belongs to the
  *Future Extensions* FastAPI service layer, out of scope for this milestone).
- **`pip-audit --strict` and `gitleaks` as separate CI steps**: the plan
  requires "dependency and secret scanning" as distinct concerns from
  linting/type-checking/tests; keeping them as separate steps means a
  dependency CVE or a leaked secret is visible as its own failing check
  rather than buried in a generic "tests failed" signal.
- **mypy could not be run locally in this environment**: this workspace's
  Windows security policy (AppLocker/WDAC) blocks loading a DLL that mypy's
  compiled backend needs (`ImportError: ... An Application Control policy
  has blocked this file`), even though `mypy --version` works. This is an
  environment restriction, not a code defect — CI runs on a standard Ubuntu
  GitHub-hosted runner where this restriction does not apply, so `mypy app`
  is verified there instead. `ruff check .` was run and is clean locally.

## How it was verified

```
python -m ruff check .           # All checks passed!
python -m pytest tests/unit tests/integration tests/evaluation -q
# 79 passed, 1 skipped

python -m app.cli doctor
# [OK] configuration / [OK] documents_path / [OK] index / [NOT READY] groq_api_key
# (exit code 1 until GROQ_API_KEY is set, as expected on a fresh checkout)
```

[tests/unit/test_ratelimit.py](../../tests/unit/test_ratelimit.py) covers:
allowing calls within the limit, blocking once the limit is hit, allowing
again once the sliding window expires, and persisting state across separate
`RateLimiter` instances (simulating separate CLI invocations).

## Exit criteria status

- [x] Failures are diagnosable without logging sensitive content (JSON logs,
      `sanitize_message`, `doctor` command).
- [x] External calls have bounded time and retry behavior (`timeout_seconds`,
      `tenacity` retry with exponential backoff and `stop_after_attempt`).
- [x] CI blocks changes that fail tests or static checks (ruff/mypy/pytest
      are required, non-continue-on-error steps in `ci.yml`).
