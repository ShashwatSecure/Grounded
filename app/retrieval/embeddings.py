"""Embedding providers.

The application depends only on the `EmbeddingProvider` interface.
`HuggingFaceEmbeddingProvider` is the production adapter (a local
sentence-transformers model via `langchain-huggingface`), and
`FakeEmbeddingProvider` is a deterministic, offline test double that requires
no model download and no network access.
"""

from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np

from app.errors import EmbeddingError


class EmbeddingProvider(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class HuggingFaceEmbeddingProvider:
    """Wraps a local sentence-transformers model via `langchain-huggingface`.

    The model is loaded lazily on first use so that importing this module (and
    running the offline test suite, which uses `FakeEmbeddingProvider`
    instead) never requires the heavier `sentence-transformers`/`torch`
    dependencies to be exercised.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._embedder = None

    def _get_embedder(self):
        if self._embedder is None:
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
            except ImportError as exc:  # pragma: no cover - import guard
                raise EmbeddingError(
                    "langchain-huggingface/sentence-transformers is not installed."
                ) from exc
            self._embedder = HuggingFaceEmbeddings(model_name=self.model_name)
        return self._embedder

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            return self._get_embedder().embed_documents(texts)
        except Exception as exc:
            raise EmbeddingError(f"Failed to embed documents: {exc}") from exc

    def embed_query(self, text: str) -> list[float]:
        try:
            return self._get_embedder().embed_query(text)
        except Exception as exc:
            raise EmbeddingError(f"Failed to embed query: {exc}") from exc


class FakeEmbeddingProvider:
    """Deterministic hashed-bag-of-words embeddings for offline tests.

    The same text always produces the same vector, and texts that share more
    words produce vectors with higher cosine similarity, which is enough to
    exercise chunking/retrieval logic deterministically without a real
    embedding model. Word counts (not signed values) are hashed into buckets
    so that shared vocabulary reliably increases similarity instead of
    occasionally cancelling out.
    """

    def __init__(self, dimension: int = 32) -> None:
        self.dimension = dimension

    def _embed_one(self, text: str) -> list[float]:
        vector = np.zeros(self.dimension, dtype=np.float32)
        words = text.lower().split()
        for word in words:
            digest = hashlib.sha256(word.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dimension
            vector[idx] += 1.0
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)
