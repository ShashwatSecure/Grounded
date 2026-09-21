# Phase 3 — Chunking and Indexing

## What was built

- [app/ingestion/splitter.py](../../app/ingestion/splitter.py):
  `split_document()` performs character-based chunking with a configurable
  `chunk_size`/`chunk_overlap`, but prefers to break at a paragraph (`\n\n`),
  then a sentence (`. `), then a whitespace boundary within the last 200
  characters of the target window, so chunks don't cut mid-word. Every chunk
  carries `document_id`, `source`, a sequential `position`, and
  `metadata = {"file_type", "checksum"}` (the parent document's checksum —
  see below).
- [app/retrieval/embeddings.py](../../app/retrieval/embeddings.py):
  `EmbeddingProvider` protocol, `HuggingFaceEmbeddingProvider` (lazy-imports
  `langchain_huggingface.HuggingFaceEmbeddings`, so importing this module
  never requires `torch`/`sentence-transformers` to be exercised), and
  `FakeEmbeddingProvider` — a deterministic hashed-bag-of-words embedding
  used by every offline test (see "Design decision" below).
- [app/retrieval/vector_store.py](../../app/retrieval/vector_store.py):
  `NumpyVectorStore` — an in-memory, cosine-similarity vector store backed by
  a single `numpy` array, persisted as `vectors.npy` + a `manifest.json`
  (chunks + `index_version`). `document_checksums()` exposes each indexed
  document's last-seen checksum so the indexer can detect changes without
  reaching into private state.
- [app/ingestion/indexer.py](../../app/ingestion/indexer.py): `Indexer`
  orchestrates load → skip-if-unchanged → remove-old-chunks-if-changed →
  split → embed (in batches of 64) → add to the store → persist, and returns
  an `IndexSummary` (documents indexed/skipped/chunks/failed files).

## Why these choices

- **Local `NumpyVectorStore` instead of FAISS/Chroma**: the plan allows
  either "Chroma or FAISS for the first local vector store." FAISS's
  native wheel can be a source of install friction on some Windows/Python
  version combinations, and Chroma pulls in a heavier dependency tree for a
  single-user CLI. A small, dependency-light, pure-`numpy` implementation is
  fully sufficient for local document collections and is hidden behind a
  `VectorStore` protocol, so swapping in FAISS/Chroma later is a drop-in
  change with no callers to update.
- **`index_version` embedded in the persisted manifest**: satisfies the
  Reliability Requirement "Index data is versioned by embedding model and
  configuration... Configuration changes that invalidate an index trigger
  re-indexing." `Settings.index_version_key()` combines
  `embedding_model:chunk_size:chunk_overlap`; `NumpyVectorStore.load()`
  discards a persisted index whose version doesn't match the current
  version instead of silently mixing incompatible embeddings.
- **Checksum stored in chunk metadata, not a separate manifest table**:
  keeps the "is this document unchanged?" check colocated with the data it
  describes (`VectorStore.document_checksums()`), instead of a second
  parallel data structure that could drift out of sync.
- **`FakeEmbeddingProvider` uses plain word-count hashing, not signed
  hashing**: an earlier version added a random `+1`/`-1` sign per hashed word
  bucket. That occasionally made cosine similarity between clearly related
  texts come out *negative*, which silently broke the
  `similarity_threshold` filter in tests (a real, observed bug — see
  Phase 4 below). Switching to unsigned counts makes similarity scale
  monotonically with shared vocabulary, which is what a fake test double
  for a real embedding model should approximate.
- **Batch embedding (64 chunks/batch)**: bounds memory/request size when
  using a real embedding backend, per "Generate embeddings in batches."

## How it was verified

- [tests/unit/test_splitter.py](../../tests/unit/test_splitter.py): empty
  content, single-chunk short content, metadata propagation, sequential
  positions, unique chunk IDs, overlap validation, and paragraph-preferring
  breaks.
- [tests/integration/test_indexing_flow.py](../../tests/integration/test_indexing_flow.py):
  full directory indexing produces the expected chunk count; **re-indexing
  unchanged files skips them and does not duplicate chunks**; a changed file
  is re-embedded while untouched files are skipped; changing `index_version`
  (simulating an embedding-model change) invalidates and fully rebuilds the
  index.
- Manual end-to-end run: `python -m app.cli index ./data/documents` using the
  real `HuggingFaceEmbeddingProvider` (downloads
  `sentence-transformers/all-MiniLM-L6-v2` on first run) successfully
  indexed the sample documents and persisted `data/index/manifest.json` +
  `vectors.npy`.

## Exit criteria status

- [x] Re-indexing unchanged files does not duplicate chunks.
- [x] Chunk metadata identifies the original source (`chunk.source`,
      `chunk.document_id`).
- [x] Indexing errors identify the affected file
      (`IndexSummary.failed_files`) without leaking secrets (errors are
      `GroundedError` subclasses, sanitized).
