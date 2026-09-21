"""Local vector store.

`NumpyVectorStore` is a pure-Python, dependency-light local vector store
(cosine similarity over a numpy array), persisted as a `.npy` vector file plus
a JSON manifest. It is used as the first local vector store instead of
FAISS/Chroma to avoid native-library install friction on developer machines,
while remaining swappable behind the `VectorStore` interface.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

import numpy as np

from app.domain import Chunk
from app.errors import VectorStoreError

_VECTORS_FILE = "vectors.npy"
_MANIFEST_FILE = "manifest.json"


class VectorStore(Protocol):
    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None: ...

    def search(self, query_embedding: list[float], top_k: int) -> list[tuple[Chunk, float]]: ...

    def remove_document(self, document_id: str) -> None: ...

    def document_checksums(self) -> dict[str, str]: ...

    def persist(self) -> None: ...

    def load(self) -> None: ...

    def __len__(self) -> int: ...


class NumpyVectorStore:
    """A cosine-similarity vector store backed by an in-memory numpy array."""

    def __init__(self, index_path: Path, index_version: str) -> None:
        self.index_path = Path(index_path)
        self.index_version = index_version
        self._chunks: list[Chunk] = []
        self._vectors: np.ndarray | None = None  # shape (n, dim), L2-normalized rows

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise VectorStoreError("chunks and embeddings must have the same length")
        if not chunks:
            return
        new_vectors = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(new_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        new_vectors = new_vectors / norms
        self._vectors = new_vectors if self._vectors is None else np.vstack(
            [self._vectors, new_vectors]
        )
        self._chunks.extend(chunks)

    def remove_document(self, document_id: str) -> None:
        if not self._chunks:
            return
        keep_indices = [i for i, c in enumerate(self._chunks) if c.document_id != document_id]
        self._chunks = [self._chunks[i] for i in keep_indices]
        vectors = self._vectors
        self._vectors = vectors[keep_indices] if vectors is not None and keep_indices else None

    def document_checksums(self) -> dict[str, str]:
        """Return the last-indexed checksum for each known document_id.

        Used by the indexer to skip re-embedding unchanged files.
        """
        checksums: dict[str, str] = {}
        for chunk in self._chunks:
            checksums.setdefault(chunk.document_id, chunk.metadata.get("checksum", ""))
        return checksums

    def search(self, query_embedding: list[float], top_k: int) -> list[tuple[Chunk, float]]:
        if self._vectors is None or len(self._chunks) == 0:
            return []
        query = np.array(query_embedding, dtype=np.float32)
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm
        scores = self._vectors @ query
        k = min(top_k, len(self._chunks))
        top_indices = np.argsort(-scores)[:k]
        return [(self._chunks[i], float(scores[i])) for i in top_indices]

    def persist(self) -> None:
        self.index_path.mkdir(parents=True, exist_ok=True)
        if self._vectors is not None:
            np.save(self.index_path / _VECTORS_FILE, self._vectors)
        manifest = {
            "index_version": self.index_version,
            "chunks": [asdict(c) for c in self._chunks],
        }
        with open(self.index_path / _MANIFEST_FILE, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    def load(self) -> None:
        manifest_file = self.index_path / _MANIFEST_FILE
        if not manifest_file.exists():
            self._chunks = []
            self._vectors = None
            return
        try:
            with open(manifest_file, encoding="utf-8") as f:
                manifest = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            raise VectorStoreError(f"Failed to read vector store manifest: {exc}") from exc

        if manifest.get("index_version") != self.index_version:
            # Embedding model or chunk config changed since this index was built.
            self._chunks = []
            self._vectors = None
            return

        self._chunks = [Chunk(**c) for c in manifest.get("chunks", [])]
        vectors_file = self.index_path / _VECTORS_FILE
        self._vectors = (
            np.load(vectors_file) if vectors_file.exists() and self._chunks else None
        )

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks)
