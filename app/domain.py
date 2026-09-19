"""Core domain contracts shared across the application layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class Document:
    """A single loaded source document."""

    document_id: str
    source: str
    content: str
    checksum: str
    file_type: str
    size_bytes: int
    modified_time: datetime


@dataclass(frozen=True)
class Chunk:
    """A retrievable slice of a document."""

    chunk_id: str
    document_id: str
    source: str
    position: int
    content: str
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievedContext:
    """Chunks retrieved for a single query, ordered by relevance."""

    chunks: list[Chunk]
    scores: list[float]
    query: str
    retrieval_id: str

    @property
    def is_empty(self) -> bool:
        return len(self.chunks) == 0


@dataclass
class Usage:
    """Token usage reported by the chat model, when available."""

    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass
class Answer:
    """A generated, grounded response returned to the user."""

    text: str
    sources: list[str]
    grounded: bool
    usage: Usage
    latency_ms: float
