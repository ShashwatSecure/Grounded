from __future__ import annotations

from app.domain import Chunk
from app.retrieval.embeddings import FakeEmbeddingProvider
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import NumpyVectorStore


def _chunk(content: str, source: str, position: int = 0, document_id: str = "doc1") -> Chunk:
    return Chunk(
        chunk_id=f"{document_id}-{position}",
        document_id=document_id,
        source=source,
        position=position,
        content=content,
    )


def _build_store(
    tmp_path, chunks: list[Chunk], embedder: FakeEmbeddingProvider
) -> NumpyVectorStore:
    store = NumpyVectorStore(index_path=tmp_path, index_version="v1")
    embeddings = embedder.embed_documents([c.content for c in chunks])
    store.add(chunks, embeddings)
    return store


def test_retrieve_returns_matching_chunk_for_known_question(tmp_path) -> None:
    embedder = FakeEmbeddingProvider(dimension=32)
    chunks = [
        _chunk("cancellation policy allows refunds within 14 days", "policies.md", 0),
        _chunk("shipping takes 3 to 5 business days domestically", "shipping.md", 1),
    ]
    store = _build_store(tmp_path, chunks, embedder)
    retriever = Retriever(embedder, store, top_k=1, similarity_threshold=0.0)

    result = retriever.retrieve("what is the cancellation policy refund days")

    assert not result.is_empty
    assert result.chunks[0].source == "policies.md"


def test_retrieve_returns_empty_context_for_irrelevant_question(tmp_path) -> None:
    embedder = FakeEmbeddingProvider(dimension=32)
    chunks = [_chunk("cancellation policy allows refunds within 14 days", "policies.md", 0)]
    store = _build_store(tmp_path, chunks, embedder)
    retriever = Retriever(embedder, store, top_k=5, similarity_threshold=0.9)

    result = retriever.retrieve("completely unrelated banana spaceship topic")

    assert result.is_empty
    assert result.chunks == []
    assert result.scores == []


def test_retrieve_respects_top_k(tmp_path) -> None:
    embedder = FakeEmbeddingProvider(dimension=32)
    chunks = [_chunk(f"document number {i} about widgets", f"doc{i}.md", i) for i in range(5)]
    store = _build_store(tmp_path, chunks, embedder)
    retriever = Retriever(embedder, store, top_k=2, similarity_threshold=0.0)

    result = retriever.retrieve("widgets document")

    assert len(result.chunks) <= 2


def test_retrieve_sorts_results_by_relevance(tmp_path) -> None:
    embedder = FakeEmbeddingProvider(dimension=32)
    chunks = [
        _chunk("apples oranges bananas fruit", "fruit.md", 0),
        _chunk("apples oranges bananas fruit apples oranges", "fruit2.md", 1),
    ]
    store = _build_store(tmp_path, chunks, embedder)
    retriever = Retriever(embedder, store, top_k=2, similarity_threshold=0.0)

    result = retriever.retrieve("apples oranges bananas")

    assert result.scores == sorted(result.scores, reverse=True)


def test_retrieve_generates_unique_retrieval_ids(tmp_path) -> None:
    embedder = FakeEmbeddingProvider(dimension=32)
    chunks = [_chunk("some content about widgets", "doc.md", 0)]
    store = _build_store(tmp_path, chunks, embedder)
    retriever = Retriever(embedder, store, top_k=1, similarity_threshold=0.0)

    first = retriever.retrieve("widgets")
    second = retriever.retrieve("widgets")

    assert first.retrieval_id != second.retrieval_id
