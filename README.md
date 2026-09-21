# Grounded

A small, production-minded GenAI document assistant built with retrieval-augmented
generation (RAG). Ask questions about your own Markdown/text documents from the
command line and get answers grounded in cited sources — or a clear "I don't know"
when the documents don't contain the answer.

See [plan.md](plan.md) for the full project specification and
[docs/phases/](docs/phases/) for a written record of how and why each phase was
implemented.

## Features

- Local document ingestion (Markdown / plain text) with checksum-based re-indexing
- Local vector similarity search (no external vector DB required)
- Grounded chat answers via Groq, with citation validation
- Bounded conversation history
- Structured JSON logs with no secret leakage
- Fully offline unit/integration tests (fake model + embedding adapters)
- Optional live smoke test and evaluation harness

## Requirements

- Python 3.12+
- A [Groq](https://console.groq.com) API key (only required for `ask`, not for `index`)

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management and
packaging.

```bash
uv sync
cp .env.example .env
# then edit .env and set GROQ_API_KEY
```

`uv sync` creates `.venv` and installs the project plus the `dev` dependency
group from `uv.lock`. Prefix commands with `uv run` (e.g. `uv run python -m
app.cli --help`), or activate `.venv` as usual.

On Windows, the same setup can be run with the helper scripts:

```powershell
.\scripts\setup.ps1
.\scripts\start.ps1 --help
```

The other common commands are also available as scripts:

```powershell
.\scripts\start.ps1 index .\data\documents
.\scripts\start.ps1 ask "What is the cancellation policy?"
.\scripts\test.ps1
.\scripts\build.ps1
```

If PowerShell blocks local scripts, run them with `-ExecutionPolicy Bypass` for
the current command.

## Usage

```bash
# Show CLI help (does not require an API key)
python -m app.cli --help

# Index the documents in ./data/documents
python -m app.cli index ./data/documents

# Ask a question (requires GROQ_API_KEY)
python -m app.cli ask "What is the cancellation policy?"

# Clear conversation history
python -m app.cli reset-history

# Check configuration, index, and provider readiness
python -m app.cli doctor
```

## Configuration

All configuration is read from environment variables or a `.env` file — see
[.env.example](.env.example) for the full list and [app/config.py](app/config.py)
for validation rules. There are no hard-coded secrets or endpoints.

## Testing

```bash
# Fast, offline unit + integration tests (no network, no API key)
uv run pytest tests/unit tests/integration

# Optional live smoke test (requires GROQ_API_KEY, makes a real API call)
uv run pytest tests/integration -m live --run-live

# Evaluation harness (offline by default)
uv run python scripts/evaluate.py
```

## Project layout

See [plan.md](plan.md#6-greenfield-project-structure) for the full target layout.

## Implementation log

Each implementation phase is documented, including design rationale, in
[docs/phases/](docs/phases/):

- [Phase 0 — Bootstrap](docs/phases/phase-0-bootstrap.md)
- [Phase 1 — Chat Provider Adapter](docs/phases/phase-1-chat-provider-adapter.md)
- [Phase 2 — Document Loading](docs/phases/phase-2-document-loading.md)
- [Phase 3 — Chunking and Indexing](docs/phases/phase-3-chunking-and-indexing.md)
- [Phase 4 — Retrieval](docs/phases/phase-4-retrieval.md)
- [Phase 5 — Grounded Answer Generation](docs/phases/phase-5-grounded-answer-generation.md)
- [Phase 6 — Conversation History](docs/phases/phase-6-conversation-history.md)
- [Phase 7 — Testing and Evaluation](docs/phases/phase-7-testing-and-evaluation.md)
- [Phase 8 — Production Readiness Baseline](docs/phases/phase-8-production-readiness.md)
