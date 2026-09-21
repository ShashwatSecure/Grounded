from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.domain import Document
from app.ingestion.splitter import split_document


def _make_document(content: str, document_id: str = "doc1") -> Document:
    return Document(
        document_id=document_id,
        source="sample.md",
        content=content,
        checksum="abc123",
        file_type="md",
        size_bytes=len(content),
        modified_time=datetime(2024, 1, 1, tzinfo=UTC),
    )


def test_split_document_empty_content_returns_no_chunks() -> None:
    document = _make_document("   \n  ")
    assert split_document(document, chunk_size=100, chunk_overlap=10) == []


def test_split_document_short_content_returns_single_chunk() -> None:
    document = _make_document("This is a short document.")
    chunks = split_document(document, chunk_size=100, chunk_overlap=10)
    assert len(chunks) == 1
    assert chunks[0].content == "This is a short document."
    assert chunks[0].position == 0


def test_split_document_preserves_document_and_source_metadata() -> None:
    document = _make_document("word " * 500)
    chunks = split_document(document, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.document_id == document.document_id
        assert chunk.source == document.source
        assert chunk.metadata["checksum"] == document.checksum


def test_split_document_positions_are_sequential() -> None:
    document = _make_document("word " * 500)
    chunks = split_document(document, chunk_size=100, chunk_overlap=20)
    positions = [c.position for c in chunks]
    assert positions == list(range(len(chunks)))


def test_split_document_chunk_ids_are_unique() -> None:
    document = _make_document("word " * 500)
    chunks = split_document(document, chunk_size=100, chunk_overlap=20)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_split_document_rejects_overlap_ge_chunk_size() -> None:
    document = _make_document("some content")
    with pytest.raises(ValueError):
        split_document(document, chunk_size=50, chunk_overlap=50)


def test_split_document_breaks_on_paragraph_when_possible() -> None:
    content = ("Paragraph one has some words in it. " * 3) + "\n\n" + ("Paragraph two. " * 10)
    document = _make_document(content)
    chunks = split_document(document, chunk_size=120, chunk_overlap=20)
    assert len(chunks) >= 2
    assert "Paragraph two" not in chunks[0].content
