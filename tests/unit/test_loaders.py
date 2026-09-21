from __future__ import annotations

from pathlib import Path

import pytest

from app.errors import DocumentLoadError
from app.ingestion.loaders import discover_files, load_document, load_documents


def test_discover_files_finds_supported_extensions_only(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("hello", encoding="utf-8")
    (tmp_path / "b.txt").write_text("world", encoding="utf-8")
    (tmp_path / "c.pdf").write_text("skip me", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "d.markdown").write_text("nested", encoding="utf-8")

    found = discover_files(tmp_path)
    names = sorted(p.name for p in found)
    assert names == ["a.md", "b.txt", "d.markdown"]


def test_discover_files_missing_directory_raises() -> None:
    with pytest.raises(DocumentLoadError):
        discover_files(Path("does-not-exist-anywhere"))


def test_load_document_valid_file_preserves_source_metadata(tmp_path: Path) -> None:
    sub = tmp_path / "docs"
    sub.mkdir()
    file_path = sub / "policy.md"
    file_path.write_text("# Policy\n\nSome content.", encoding="utf-8")

    document = load_document(file_path, tmp_path)

    assert document is not None
    assert document.source == "docs/policy.md"
    assert document.file_type == "md"
    assert "Some content." in document.content
    assert len(document.checksum) == 64


def test_load_document_empty_file_returns_none(tmp_path: Path) -> None:
    file_path = tmp_path / "empty.txt"
    file_path.write_text("   \n  ", encoding="utf-8")
    assert load_document(file_path, tmp_path) is None


def test_load_document_unsupported_extension_raises(tmp_path: Path) -> None:
    file_path = tmp_path / "data.csv"
    file_path.write_text("a,b,c", encoding="utf-8")
    with pytest.raises(DocumentLoadError):
        load_document(file_path, tmp_path)


def test_load_document_invalid_encoding_raises(tmp_path: Path) -> None:
    file_path = tmp_path / "bad.txt"
    file_path.write_bytes(b"\xff\xfe\x00\x01invalid")
    with pytest.raises(DocumentLoadError):
        load_document(file_path, tmp_path)


def test_load_documents_collects_errors_without_stopping(tmp_path: Path) -> None:
    (tmp_path / "good.md").write_text("Good content here.", encoding="utf-8")
    bad_path = tmp_path / "bad.txt"
    bad_path.write_bytes(b"\xff\xfe\x00\x01invalid")

    documents, errors = load_documents(tmp_path)

    assert len(documents) == 1
    assert documents[0].source == "good.md"
    assert len(errors) == 1
    assert errors[0][0] == bad_path


def test_checksum_changes_when_content_changes(tmp_path: Path) -> None:
    file_path = tmp_path / "note.txt"
    file_path.write_text("version one", encoding="utf-8")
    doc1 = load_document(file_path, tmp_path)

    file_path.write_text("version two", encoding="utf-8")
    doc2 = load_document(file_path, tmp_path)

    assert doc1 is not None and doc2 is not None
    assert doc1.checksum != doc2.checksum
