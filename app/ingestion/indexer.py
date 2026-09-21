"""Orchestrates loading, splitting, embedding, and persisting the document index."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from app.domain import Chunk
from app.errors import EmbeddingError
from app.ingestion.loaders import load_documents
from app.ingestion.splitter import split_document
from app.retrieval.embeddings import EmbeddingProvider
from app.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)

_EMBED_BATCH_SIZE = 64


@dataclass
class IndexSummary:
    documents_indexed: int = 0
    documents_skipped_unchanged: int = 0
    chunks_indexed: int = 0
    failed_files: list[tuple[str, str]] = field(default_factory=list)


class Indexer:
    """Indexes a directory of documents into the configured vector store."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def index_directory(self, documents_path: Path) -> IndexSummary:
        self.vector_store.load()
        existing_checksums = self.vector_store.document_checksums()

        documents, load_errors = load_documents(documents_path)
        summary = IndexSummary(failed_files=[(str(p), msg) for p, msg in load_errors])

        for document in documents:
            if existing_checksums.get(document.document_id) == document.checksum:
                summary.documents_skipped_unchanged += 1
                continue

            # Re-indexing a changed file: drop its previously indexed chunks first.
            self.vector_store.remove_document(document.document_id)

            try:
                chunks = split_document(document, self.chunk_size, self.chunk_overlap)
                self._embed_and_add(chunks)
            except EmbeddingError as exc:
                summary.failed_files.append((document.source, str(exc)))
                continue

            summary.documents_indexed += 1
            summary.chunks_indexed += len(chunks)

        self.vector_store.persist()
        logger.info(
            "indexing.completed",
            extra={
                "documents_indexed": summary.documents_indexed,
                "documents_skipped_unchanged": summary.documents_skipped_unchanged,
                "chunks_indexed": summary.chunks_indexed,
                "failed_count": len(summary.failed_files),
            },
        )
        return summary

    def _embed_and_add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        for i in range(0, len(chunks), _EMBED_BATCH_SIZE):
            batch = chunks[i : i + _EMBED_BATCH_SIZE]
            texts = [c.content for c in batch]
            try:
                embeddings = self.embedding_provider.embed_documents(texts)
            except Exception as exc:
                raise EmbeddingError(f"Failed to embed batch: {exc}") from exc
            self.vector_store.add(batch, embeddings)
