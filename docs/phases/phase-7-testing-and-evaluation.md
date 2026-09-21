# Phase 7 — Testing and Evaluation

## What was built

- **Unit tests** (`tests/unit/`): one file per pure component —
  `test_config.py`, `test_errors.py`, `test_model.py`, `test_loaders.py`,
  `test_splitter.py`, `test_retriever.py`, `test_prompts.py`,
  `test_history.py`, `test_answerer.py`, `test_ratelimit.py`. All run fully
  offline using `FakeEmbeddingProvider`/`FakeChatModel`/`tmp_path`.
- **Integration tests** (`tests/integration/`):
  `test_indexing_flow.py` (documents → index, including re-indexing and
  index-version invalidation), `test_question_flow.py` (index → retrieval →
  grounded answer, missing-context behavior, provider-failure propagation,
  multi-turn conversation history), and `test_live_smoke.py` (opt-in, real
  Groq call — see Phase 1 doc).
- **Evaluation dataset**
  ([tests/evaluation/questions.jsonl](../../tests/evaluation/questions.jsonl)):
  20 questions covering every required category — direct fact lookup,
  multi-document, paraphrased, ambiguous, unsupported, prompt injection
  embedded in a document, prompt injection in the user's question, a very
  long question, and two empty/whitespace-only questions. A new sample
  document, [data/documents/faq.md](../../data/documents/faq.md), embeds an
  HTML-comment "SYSTEM NOTE" instructing the model to always answer
  "OVERRIDDEN" — a direct test of the Phase 5 untrusted-context defense.
- **Evaluation harness** ([scripts/evaluate.py](../../scripts/evaluate.py)):
  indexes `DOCUMENTS_PATH` and runs every dataset question through the real
  `Retriever`/`Answerer` pipeline, computing retrieval recall@k, source
  citation precision, groundedness rate, refusal accuracy, P50/P95 latency,
  and average input/output tokens. Defaults to `FakeEmbeddingProvider` +
  `FakeChatModel` (fully offline, deterministic, safe for CI); `--live`
  switches to the real HuggingFace + Groq adapters and requires an explicit
  `y` confirmation before making network calls.
- [tests/evaluation/test_evaluation_dataset.py](../../tests/evaluation/test_evaluation_dataset.py):
  asserts the dataset itself is well-formed (20-50 records, required fields,
  unique IDs, all required categories present) and that
  `run_evaluation(live=False)` runs end-to-end with zero errors and produces
  every metric.

## Why these choices

- **Offline evaluation by default, `--live` opt-in with confirmation**:
  matches the plan's "Run evaluation without requiring a live provider by
  default" and "Add an optional live evaluation command with explicit
  confirmation" verbatim. The offline mode exists to validate the *harness
  mechanics* (retrieval → citation-checking → metrics math) deterministically
  in CI; it is not a substitute for judging real answer quality, which needs
  a real embedding model and a real chat model (`--live`). This is called out
  explicitly so the numbers from a default run aren't mistaken for a quality
  score.
- **`faq.md` with an embedded fake "system note"**: a concrete, realistic
  prompt-injection payload (an HTML comment that looks like configuration
  metadata) rather than an obviously fake string, so the grounding policy is
  tested against a plausible attack shape.
- **One evaluation record per category, not just one overall pass/fail**:
  lets `test_dataset_covers_all_required_categories` catch dataset regressions
  (e.g. someone deleting the injection question) independently of whether the
  harness itself still runs.

## How it was verified

```
python -m pytest tests/unit tests/integration tests/evaluation -m "not live" -q
# 79 passed, 1 skipped (the opt-in live smoke test)

python scripts/evaluate.py
# prints a JSON summary with every required metric populated
```

## Exit criteria status

- [x] Tests pass in a clean environment (`uv sync` then
      `pytest`, no network access needed).
- [x] Evaluation results are reproducible (deterministic fake adapters by
      default; `index-eval`'s `index_version` is fixed per mode).
- [x] The project documents known failure cases: the offline evaluation's
      retrieval/groundedness numbers are expected to be low because
      `FakeEmbeddingProvider` is a simple hashed-bag-of-words stand-in, not a
      semantic embedding model — real quality must be assessed with
      `scripts/evaluate.py --live`.
