# Phase 4 — Retrieval

## What was built

[app/retrieval/retriever.py](../../app/retrieval/retriever.py): `Retriever`
combines an `EmbeddingProvider` and a `VectorStore`:

1. Embeds the query.
2. Calls `vector_store.search(query_embedding, top_k)` — the store itself
   returns results already sorted by cosine similarity, descending.
3. Filters out any result below `similarity_threshold`.
4. Logs a structured `retrieval.completed` event (`retrieval_id`,
   `retrieved_count`, `kept_count`, `top_similarity`) — no document content.
5. Returns a `RetrievedContext` with an explicit `is_empty` property used
   everywhere downstream to detect "no usable context."

## Why these choices

- **Filtering happens in the retriever, not the vector store**: the vector
  store's job is pure similarity search; the *policy* of "what counts as
  relevant enough" belongs to the retrieval layer so it can be tuned via
  configuration (`SIMILARITY_THRESHOLD`) without touching storage code.
- **`RetrievedContext.is_empty` as the single source of truth for "no
  context"**: `Answerer` (Phase 5) checks this one property instead of
  re-deriving "no chunks passed the threshold" logic itself.
- **A `retrieval_id` per call**: gives every retrieval a correlation ID for
  logs/debugging, per the plan's Observability Requirements field list.

## A real bug found and fixed during this phase

While writing `tests/unit/test_retriever.py`, a test asserting that a
clearly-relevant chunk ranks above an irrelevant one failed intermittently.
The root cause was in `FakeEmbeddingProvider` (Phase 3): its original
signed-hash scheme could produce a *negative* cosine similarity between two
texts that share vocabulary, purely by chance of which hash buckets got a
`-1`. Since `similarity_threshold=0.0` in that test, a negative score was
incorrectly filtered out, making the retriever report "no context" for a
clearly answerable question. Fixed by switching `FakeEmbeddingProvider` to
unsigned word-count hashing (see Phase 3 doc) — cosine similarity between
related texts is now reliably non-negative and roughly proportional to
shared vocabulary, which is the property the test (and the real retriever
logic) depends on.

## How it was verified

[tests/unit/test_retriever.py](../../tests/unit/test_retriever.py):
- a known, on-topic question retrieves the expected chunk/source,
- an unrelated question with a high threshold returns an explicitly empty
  `RetrievedContext`,
- `top_k` is respected,
- results come back sorted by descending score,
- two calls produce distinct `retrieval_id`s.

## Exit criteria status

- [x] Known questions retrieve expected documents.
- [x] Irrelevant questions return no usable context (`is_empty=True`).
- [x] Retrieval behavior is covered by deterministic, offline tests (no
      network, `FakeEmbeddingProvider` + `NumpyVectorStore` only).
