# Phase 0 — Project Bootstrap

## What was built

- Greenfield project layout under the workspace root (`app/`, `data/`, `tests/`,
  `scripts/`), matching [plan.md](../../plan.md#6-greenfield-project-structure).
- [pyproject.toml](../../pyproject.toml): dependency groups (`dev` extras),
  ruff config, mypy config, pytest config (`testpaths`, `pythonpath`,
  live-test marker added in Phase 8).
- [.env.example](../../.env.example) and [.gitignore](../../.gitignore) — no
  secrets committed, `.env` and generated index artifacts are ignored.
- [app/config.py](../../app/config.py): `pydantic-settings`-based `Settings`
  with validated ranges (`gt=0`, `ge=0`, `le=1.0`), a custom validator that
  rejects `chunk_overlap >= chunk_size`, and `require_groq_api_key()` which
  only raises when the key is actually needed (not for `--help` or `index`).
- [app/errors.py](../../app/errors.py): a `GroundedError` exception hierarchy
  used everywhere in the app, plus `sanitize_message()` which redacts
  API-key-shaped strings (`api_key=...`, `Authorization: Bearer ...`,
  `gsk_...`) from every error message at construction time.
- [app/domain.py](../../app/domain.py): the core dataclasses (`Document`,
  `Chunk`, `RetrievedContext`, `Usage`, `Answer`) shared by every layer.
- [app/logging_config.py](../../app/logging_config.py): stdlib `logging` +
  a custom `JsonFormatter` (no `structlog` dependency), redacting secrets in
  every log line via the same `sanitize_message()`.
- A minimal [app/cli.py](../../app/cli.py) (`index`/`ask`/`reset-history`
  stubs at this stage) wired to `load_settings()` so `--help` never requires
  `GROQ_API_KEY`.
- Sample local documents (`data/documents/policies.md`,
  `data/documents/shipping.md`) used by later phases' tests and examples.

## Why these choices

- **pydantic-settings over hand-rolled env parsing**: gives fail-fast
  validation, type coercion, and a single source of truth for configuration,
  satisfying "Fail fast when required settings are missing" and "Validate
  numeric ranges" from the plan's Configuration section.
- **Secret redaction at the exception-base-class level** (`GroundedError.__init__`)
  rather than at each call site: guarantees every error raised anywhere in the
  app is sanitized, instead of relying on every developer remembering to do it.
- **stdlib JSON logging instead of `structlog`**: the plan allows either;
  stdlib keeps the dependency surface smaller while still producing
  structured, greppable JSON log lines.
- **API key required lazily, not at settings-load time**: the plan's Phase 0
  exit criteria explicitly requires `--help` to work without an API key, and
  `index` never needs one (only embeddings, not the chat model).

## How it was verified

- `uv sync` succeeds into a clean `.venv`.
- `python -m app.cli --help` prints usage without `GROQ_API_KEY` set.
- `python -m app.cli ask "test"` (no key set) exits with a clear, non-secret
  `Configuration error: GROQ_API_KEY is not set...` message (exit code 2).
- [tests/unit/test_config.py](../../tests/unit/test_config.py) and
  [tests/unit/test_errors.py](../../tests/unit/test_errors.py) cover
  validation rules and redaction behavior.

## Exit criteria status

- [x] The project installs into a clean virtual environment.
- [x] The CLI displays help without requiring an API key.
- [x] Invalid configuration produces a clear, non-secret error.
