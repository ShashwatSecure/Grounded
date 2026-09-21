from __future__ import annotations

from pathlib import Path

from app.ingestion.indexer import Indexer
from app.retrieval.embeddings import FakeEmbeddingProvider
from app.retrieval.vector_store import NumpyVectorStore


def _write_sample_documents(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "policies.md").write_text(
        "# Policies\n\nCancellations are allowed within 14 days of purchase.\n\n"
        "Warranty covers manufacturing defects for 1 year.",
        encoding="utf-8",
    )
    (root / "shipping.md").write_text(
        "# Shipping\n\nDomestic shipping takes 3-5 business days.\n\n"
        "International shipping takes 7-21 business days.",
        encoding="utf-8",
    )


def _build_indexer(tmp_path: Path) -> tuple[Indexer, NumpyVectorStore]:
    embedder = FakeEmbeddingProvider(dimension=32)
    store = NumpyVectorStore(index_path=tmp_path / "index", index_version="v1")
    indexer = Indexer(
        embedding_provider=embedder, vector_store=store, chunk_size=200, chunk_overlap=20
    )
    return indexer, store


def test_indexing_documents_produces_chunks_in_store(tmp_path: Path) -> None:
    documents_path = tmp_path / "documents"
    _write_sample_documents(documents_path)
    indexer, store = _build_indexer(tmp_path)

    summary = indexer.index_directory(documents_path)

    assert summary.documents_indexed == 2
    assert summary.chunks_indexed > 0
    assert len(store) == summary.chunks_indexed
    assert (tmp_path / "index" / "manifest.json").exists()


def test_reindexing_unchanged_files_skips_and_does_not_duplicate(tmp_path: Path) -> None:
    documents_path = tmp_path / "documents"
    _write_sample_documents(documents_path)
    indexer, _store = _build_indexer(tmp_path)

    first_summary = indexer.index_directory(documents_path)

    # Build a fresh indexer/store pointed at the same persisted index directory,
    # simulating a second CLI invocation.
    embedder = FakeEmbeddingProvider(dimension=32)
    second_store = NumpyVectorStore(index_path=tmp_path / "index", index_version="v1")
    second_indexer = Indexer(
        embedding_provider=embedder, vector_store=second_store, chunk_size=200, chunk_overlap=20
    )

    second_summary = second_indexer.index_directory(documents_path)

    assert second_summary.documents_skipped_unchanged == 2
    assert second_summary.documents_indexed == 0
    assert len(second_store) == first_summary.chunks_indexed


def test_changed_file_is_reindexed_without_duplicating_other_chunks(tmp_path: Path) -> None:
    documents_path = tmp_path / "documents"
    _write_sample_documents(documents_path)
    indexer, _store = _build_indexer(tmp_path)
    indexer.index_directory(documents_path)

    # Modify one file, then re-index with a fresh indexer against the persisted index.
    (documents_path / "policies.md").write_text(
        "# Policies\n\nCancellations are allowed within 30 days now.", encoding="utf-8"
    )
    embedder = FakeEmbeddingProvider(dimension=32)
    store = NumpyVectorStore(index_path=tmp_path / "index", index_version="v1")
    reindexer = Indexer(
        embedding_provider=embedder, vector_store=store, chunk_size=200, chunk_overlap=20
    )
    summary = reindexer.index_directory(documents_path)

    assert summary.documents_indexed == 1
    assert summary.documents_skipped_unchanged == 1
    store.load()
    contents = " ".join(c.content for c in store.chunks if c.source == "policies.md")
    assert "30 days" in contents


def test_index_version_change_invalidates_existing_index(tmp_path: Path) -> None:
    documents_path = tmp_path / "documents"
    _write_sample_documents(documents_path)
    embedder = FakeEmbeddingProvider(dimension=32)
    store = NumpyVectorStore(index_path=tmp_path / "index", index_version="v1")
    indexer = Indexer(
        embedding_provider=embedder, vector_store=store, chunk_size=200, chunk_overlap=20
    )
    indexer.index_directory(documents_path)

    # Simulate an embedding-model change: same directory, new index_version.
    new_store = NumpyVectorStore(index_path=tmp_path / "index", index_version="v2-different-model")
    new_indexer = Indexer(
        embedding_provider=embedder, vector_store=new_store, chunk_size=200, chunk_overlap=20
    )
    summary = new_indexer.index_directory(documents_path)

    assert summary.documents_indexed == 2
    assert summary.documents_skipped_unchanged == 0
