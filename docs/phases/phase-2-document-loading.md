# Phase 2 — Document Loading

## What was built

[app/ingestion/loaders.py](../../app/ingestion/loaders.py):

- `discover_files(root)`: recursively finds `.md`, `.markdown`, `.txt` files
  under a root directory, sorted for deterministic ordering; raises
  `DocumentLoadError` if the root doesn't exist or isn't a directory.
- `load_document(path, root)`: reads a file as raw bytes, decodes strict
  UTF-8 (raising `DocumentLoadError` with the filename on a decode failure —
  never a raw traceback), returns `None` for whitespace-only ("empty")
  files, and otherwise returns a `Document` with a relative POSIX `source`
  path, a SHA-256 `checksum` of the content, and a short SHA-256-derived
  `document_id` computed from the relative path (stable across re-runs).
- `load_documents(root)`: loads every discovered file and returns
  `(documents, errors)` — one bad file (bad encoding, unsupported extension)
  is recorded in `errors` and does **not** stop the rest of the run.

## Why these choices

- **`document_id` derived from the relative path, not content**: this keeps
  the same document's ID stable across edits, which is required for the
  Phase 3 indexer to detect "this document changed" (same `document_id`,
  different `checksum`) versus "this is a new document."
- **Checksum of decoded content (not raw bytes)**: guards against
  false-positive "changed" detection from encoding artifacts and matches
  what's actually chunked/embedded.
- **Errors collected, not raised, in `load_documents`**: satisfies "Indexing
  can resume after an individual file failure" (Reliability Requirements) —
  a single corrupt file must not block indexing the rest of the directory.
- **UTF-8 only, explicit rejection of unsupported extensions**: matches the
  plan's in-scope statement ("Markdown and plain-text documents") and the
  Phase 2 task "Reject unsupported extensions."

## How it was verified

[tests/unit/test_loaders.py](../../tests/unit/test_loaders.py) covers:
- discovery filtering by extension across nested directories,
- a missing documents directory raising `DocumentLoadError`,
- valid file loading preserving the relative `source` and metadata,
- empty (whitespace-only) files returning `None`,
- unsupported extensions and invalid UTF-8 both raising `DocumentLoadError`,
- `load_documents` collecting one file's error without dropping the other
  valid document,
- checksum changing when content changes.

## Exit criteria status

- [x] Loader tests cover valid, empty, unsupported, and unreadable files.
- [x] Source metadata (relative path, file type, size, modified time)
      survives loading into the `Document` dataclass.
