"""Command-line interface for the Grounded document assistant."""

from __future__ import annotations

from pathlib import Path

import typer

from app.config import Settings, load_settings
from app.errors import ConfigError, GroundedError
from app.ingestion.indexer import Indexer
from app.logging_config import configure_logging
from app.retrieval.embeddings import HuggingFaceEmbeddingProvider
from app.retrieval.vector_store import NumpyVectorStore

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)


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
def ask() -> None:
    """Reserve the question-answering command for a later phase."""
    load_settings()
    typer.echo("Question answering will be implemented in a later phase.")


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

