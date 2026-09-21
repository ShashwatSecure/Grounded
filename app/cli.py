"""Command-line interface for the Grounded document assistant."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from app.config import Settings, load_settings
from app.conversation.history import load_history, save_history
from app.errors import ConfigError, GroundedError
from app.generation.answerer import Answerer
from app.generation.model import GroqChatModel
from app.ingestion.indexer import Indexer
from app.logging_config import configure_logging, log_event
from app.ratelimit import RateLimiter
from app.retrieval.embeddings import HuggingFaceEmbeddingProvider
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import NumpyVectorStore

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)
logger = logging.getLogger("app.cli")

_HISTORY_FILENAME = "history.json"
_RATE_LIMIT_FILENAME = "rate_limit_state.json"


def _build_vector_store(settings: Settings) -> NumpyVectorStore:
    return NumpyVectorStore(
        index_path=Path(settings.index_path), index_version=settings.index_version_key()
    )


def _build_embedding_provider(settings: Settings) -> HuggingFaceEmbeddingProvider:
    return HuggingFaceEmbeddingProvider(model_name=settings.embedding_model)


@app.command()
def index(
    path: str = typer.Argument(
        None, help="Directory of documents to index. Defaults to DOCUMENTS_PATH."
    ),
) -> None:
    """Index documents so they can be retrieved and cited in answers."""
    settings = load_settings()
    documents_path = Path(path) if path else settings.documents_path

    embedding_provider = _build_embedding_provider(settings)
    vector_store = _build_vector_store(settings)
    indexer = Indexer(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    try:
        summary = indexer.index_directory(documents_path)
    except GroundedError as exc:
        typer.echo(f"Indexing failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(
        f"Indexed {summary.documents_indexed} documents and {summary.chunks_indexed} chunks "
        f"({summary.documents_skipped_unchanged} unchanged, "
        f"{len(summary.failed_files)} failed)."
    )
    for source, message in summary.failed_files:
        typer.echo(f"  failed: {source}: {message}", err=True)


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask about the indexed documents."),
) -> None:
    """Ask a grounded question about the indexed documents."""
    settings = load_settings()

    normalized_question = question.strip()
    if not normalized_question:
        typer.echo("Question must not be empty.", err=True)
        raise typer.Exit(code=1)
    if len(normalized_question) > settings.max_input_characters:
        typer.echo(
            "Question exceeds the maximum allowed length of "
            f"{settings.max_input_characters} characters.",
            err=True,
        )
        raise typer.Exit(code=1)

    try:
        api_key = settings.require_groq_api_key()
    except GroundedError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    rate_limiter = RateLimiter(
        state_path=Path(settings.index_path) / _RATE_LIMIT_FILENAME,
        max_calls=settings.rate_limit_per_minute,
    )
    try:
        rate_limiter.check_and_record()
    except GroundedError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    history_path = Path(settings.index_path) / _HISTORY_FILENAME
    history = load_history(
        history_path, settings.max_history_messages, settings.max_input_characters
    )
    history.add_user(normalized_question)

    embedding_provider = _build_embedding_provider(settings)
    vector_store = _build_vector_store(settings)
    vector_store.load()
    retriever = Retriever(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        top_k=settings.top_k,
        similarity_threshold=settings.similarity_threshold,
    )
    chat_model = GroqChatModel(
        api_key=api_key,
        model=settings.groq_model,
        timeout_seconds=settings.model_timeout_seconds,
        max_tokens=settings.model_max_tokens,
        max_retries=settings.model_max_retries,
    )
    answerer = Answerer(retriever=retriever, chat_model=chat_model)

    try:
        answer = answerer.answer(normalized_question, history=history.messages()[:-1])
    except GroundedError as exc:
        log_event(logger, logging.ERROR, "ask.failed", error_type=type(exc).__name__)
        typer.echo(f"Could not generate an answer: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    history.add_assistant(answer.text)
    save_history(history, history_path)

    log_event(
        logger,
        logging.INFO,
        "ask.completed",
        outcome="grounded" if answer.grounded else "uncertain",
        latency_ms=answer.latency_ms,
        source_count=len(answer.sources),
        input_tokens=answer.usage.input_tokens,
        output_tokens=answer.usage.output_tokens,
    )

    typer.echo("\nAnswer:")
    typer.echo(answer.text)
    if answer.sources:
        typer.echo("\nSources:")
        for source in answer.sources:
            typer.echo(f"- {source}")


@app.command("reset-history")
def reset_history() -> None:
    """Reset the CLI's persisted conversation history."""
    settings = load_settings()
    history_path = Path(settings.index_path) / _HISTORY_FILENAME
    if history_path.exists():
        history_path.unlink()
    typer.echo("Conversation history has been reset.")


@app.command()
def doctor() -> None:
    """Report configuration, index, and provider readiness without leaking secrets."""
    settings = load_settings()
    checks: list[tuple[str, bool, str]] = []

    checks.append(("configuration", True, "loaded and validated"))

    documents_exist = Path(settings.documents_path).is_dir()
    checks.append(
        (
            "documents_path",
            documents_exist,
            str(settings.documents_path) if documents_exist else "directory not found",
        )
    )

    manifest_path = Path(settings.index_path) / "manifest.json"
    if manifest_path.exists():
        vector_store = _build_vector_store(settings)
        vector_store.load()
        index_ready = len(vector_store) > 0
        detail = f"{len(vector_store)} chunks indexed" if index_ready else "index is empty"
    else:
        index_ready = False
        detail = "no index found; run `index` first"
    checks.append(("index", index_ready, detail))

    has_api_key = bool(settings.groq_api_key)
    checks.append(
        (
            "groq_api_key",
            has_api_key,
            "present" if has_api_key else "not set (required for `ask`)",
        )
    )

    all_ready = all(ok for _, ok, _ in checks)
    for name, ok, detail in checks:
        status = "OK" if ok else "NOT READY"
        typer.echo(f"[{status}] {name}: {detail}")

    if not all_ready:
        raise typer.Exit(code=1)


def main() -> None:
    try:
        settings = load_settings()
        configure_logging(settings.log_level)
    except ConfigError as exc:
        typer.echo(f"Configuration error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    app()


if __name__ == "__main__":
    main()

