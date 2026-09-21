"""Configurable text chunking that preserves document and position metadata."""

from __future__ import annotations

from app.domain import Chunk, Document


def split_document(document: Document, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Split a document's content into overlapping chunks.

    Splitting is character-based but prefers to break on paragraph, then
    sentence, then whitespace boundaries near the target size, so chunks stay
    reasonably coherent instead of cutting mid-word.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    text = document.content
    if not text.strip():
        return []

    chunks: list[Chunk] = []
    start = 0
    position = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        if end < text_length:
            end = _find_break_point(text, start, end)
        piece = text[start:end].strip()
        if piece:
            chunks.append(
                Chunk(
                    chunk_id=f"{document.document_id}-{position}",
                    document_id=document.document_id,
                    source=document.source,
                    position=position,
                    content=piece,
                    metadata={"file_type": document.file_type, "checksum": document.checksum},
                )
            )
            position += 1
        if end >= text_length:
            break
        start = max(end - chunk_overlap, start + 1)

    return chunks


def _find_break_point(text: str, start: int, proposed_end: int) -> int:
    """Search backward from `proposed_end` for a paragraph/sentence/word boundary."""
    window_start = max(start + 1, proposed_end - 200)
    paragraph_break = text.rfind("\n\n", window_start, proposed_end)
    if paragraph_break != -1:
        return paragraph_break + 2
    sentence_break = max(
        text.rfind(". ", window_start, proposed_end),
        text.rfind(".\n", window_start, proposed_end),
    )
    if sentence_break != -1:
        return sentence_break + 2
    space_break = text.rfind(" ", window_start, proposed_end)
    if space_break != -1:
        return space_break + 1
    return proposed_end
