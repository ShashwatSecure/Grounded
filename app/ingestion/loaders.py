"""Document discovery and loading for Markdown and plain-text files."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

from app.domain import Document
from app.errors import DocumentLoadError

SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt"}


def discover_files(root: Path) -> list[Path]:
    """Return supported files under `root`, sorted for deterministic ordering."""
    root = Path(root)
    if not root.exists():
        raise DocumentLoadError(f"Documents path does not exist: {root}")
    if not root.is_dir():
        raise DocumentLoadError(f"Documents path is not a directory: {root}")

    files = [
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(files)


def _checksum(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def load_document(path: Path, root: Path) -> Document | None:
    """Load a single document, returning None for empty (whitespace-only) files.

    Raises DocumentLoadError for unsupported extensions, unreadable files, or
    content that is not valid UTF-8.
    """
    path = Path(path)
    root = Path(root)
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise DocumentLoadError(f"Unsupported file extension: {path.name}")

    try:
        raw_bytes = path.read_bytes()
    except OSError as exc:
        raise DocumentLoadError(f"Could not read file {path.name}: {exc}") from exc

    try:
        content = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DocumentLoadError(f"File {path.name} is not valid UTF-8 text: {exc}") from exc

    if not content.strip():
        return None

    try:
        stat = path.stat()
        modified_time = datetime.fromtimestamp(stat.st_mtime, tz=UTC)
        size_bytes = stat.st_size
    except OSError as exc:
        raise DocumentLoadError(f"Could not stat file {path.name}: {exc}") from exc

    relative_source = path.relative_to(root).as_posix()
    checksum = _checksum(content)
    document_id = hashlib.sha256(relative_source.encode("utf-8")).hexdigest()[:16]

    return Document(
        document_id=document_id,
        source=relative_source,
        content=content,
        checksum=checksum,
        file_type=path.suffix.lower().lstrip("."),
        size_bytes=size_bytes,
        modified_time=modified_time,
    )


def load_documents(root: Path) -> tuple[list[Document], list[tuple[Path, str]]]:
    """Load all supported documents under `root`.

    Returns (documents, errors); errors contains (path, message) for files that
    failed to load so a single bad file does not stop the whole run.
    """
    root = Path(root)
    documents: list[Document] = []
    errors: list[tuple[Path, str]] = []
    for path in discover_files(root):
        try:
            document = load_document(path, root)
        except DocumentLoadError as exc:
            errors.append((path, str(exc)))
            continue
        if document is not None:
            documents.append(document)
    return documents, errors
