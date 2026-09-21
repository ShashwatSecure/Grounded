"""Query-time retrieval: embed a question and fetch the most relevant chunks."""

from __future__ import annotations

import logging
import uuid

from app.domain import Chunk, RetrievedContext
from app.retrieval.embeddings import EmbeddingProvider
from app.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """Retrieves the top-k chunks above a similarity threshold for a query."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        top_k: int,
        similarity_threshold: float,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

    def retrieve(self, query: str) -> RetrievedContext:
        retrieval_id = str(uuid.uuid4())
        query_embedding = self.embedding_provider.embed_query(query)
        results = self.vector_store.search(query_embedding, self.top_k)
        filtered = [(c, s) for c, s in results if s >= self.similarity_threshold]

        logger.info(
            "retrieval.completed",
            extra={
                "retrieval_id": retrieval_id,
                "retrieved_count": len(results),
                "kept_count": len(filtered),
                "top_similarity": results[0][1] if results else None,
            },
        )

        chunks: list[Chunk] = [c for c, _ in filtered]
        scores: list[float] = [s for _, s in filtered]
        return RetrievedContext(
            chunks=chunks, scores=scores, query=query, retrieval_id=retrieval_id
        )
