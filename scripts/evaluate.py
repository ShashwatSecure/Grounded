"""Offline-by-default evaluation harness for the RAG pipeline.

By default this script indexes `DOCUMENTS_PATH` and evaluates questions from
`tests/evaluation/questions.jsonl` using fake embedding and chat-model
adapters, so it runs deterministically with no network access or API key.

Pass --live to instead exercise the real HuggingFace embedding model and Groq
chat model (requires GROQ_API_KEY and network access); this path asks for
explicit confirmation before making any live calls.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import load_settings
from app.generation.answerer import Answerer
from app.generation.model import ChatMessage, FakeChatModel, GroqChatModel
from app.ingestion.indexer import Indexer
from app.retrieval.embeddings import (
    EmbeddingProvider,
    FakeEmbeddingProvider,
    HuggingFaceEmbeddingProvider,
)
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import NumpyVectorStore

QUESTIONS_FILE = (
    Path(__file__).resolve().parent.parent / "tests" / "evaluation" / "questions.jsonl"
)
EVAL_INDEX_PATH = Path(__file__).resolve().parent.parent / "data" / "index-eval"


def _load_questions(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _fake_responder(messages: list[ChatMessage]) -> str:
    """Deterministic stand-in model: cites every source present in the context.

    This does not evaluate real answer quality — it exercises the harness
    (retrieval, citation validation, metrics) without a live model call.
    """
    user_message = messages[-1].content
    sources = re.findall(r"\[source: ([^\]]+)\]", user_message)
    unique_sources = list(dict.fromkeys(sources))
    if not unique_sources:
        return (
            "I could not find enough information in the indexed documents to "
            "answer that question.\nSources: none"
        )
    joined_sources = ", ".join(unique_sources)
    return f"Based on the documents, here is relevant information.\nSources: {joined_sources}"


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    k = (len(sorted_values) - 1) * (pct / 100)
    lower = int(k)
    upper = min(lower + 1, len(sorted_values) - 1)
    if lower == upper:
        return sorted_values[lower]
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (k - lower)


def run_evaluation(live: bool = False) -> dict:
    settings = load_settings()
    documents_path = Path(settings.documents_path)

    embedding_provider: EmbeddingProvider
    if live:
        embedding_provider = HuggingFaceEmbeddingProvider(settings.embedding_model)
        api_key = settings.require_groq_api_key()
        chat_model = GroqChatModel(
            api_key=api_key,
            model=settings.groq_model,
            timeout_seconds=settings.model_timeout_seconds,
            max_tokens=settings.model_max_tokens,
            max_retries=settings.model_max_retries,
        )
        index_version = "eval-live-" + settings.index_version_key()
    else:
        embedding_provider = FakeEmbeddingProvider(dimension=64)
        chat_model = FakeChatModel(responder=_fake_responder)
        index_version = "eval-fake"

    vector_store = NumpyVectorStore(index_path=EVAL_INDEX_PATH, index_version=index_version)
    indexer = Indexer(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    indexer.index_directory(documents_path)

    retriever = Retriever(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        top_k=settings.top_k,
        similarity_threshold=settings.similarity_threshold,
    )
    answerer = Answerer(retriever=retriever, chat_model=chat_model)

    questions = _load_questions(QUESTIONS_FILE)

    latencies: list[float] = []
    input_tokens: list[int] = []
    output_tokens: list[int] = []
    retrieval_hits = 0
    retrievable_questions = 0
    citation_correct = 0
    citation_total = 0
    groundedness_correct = 0
    refusal_correct = 0
    errors = 0
    results = []

    for record in questions:
        question = record["question"]
        expected_sources = set(record.get("expected_sources", []))
        answerable = record.get("answerable", True)

        try:
            retrieved = retriever.retrieve(question)
            answer = answerer.answer(question)
        except Exception as exc:  # evaluation harness must not crash on one bad question
            errors += 1
            results.append({"id": record["id"], "error": str(exc)})
            continue

        latencies.append(answer.latency_ms)
        if answer.usage.input_tokens is not None:
            input_tokens.append(answer.usage.input_tokens)
        if answer.usage.output_tokens is not None:
            output_tokens.append(answer.usage.output_tokens)

        if expected_sources:
            retrievable_questions += 1
            retrieved_sources = {c.source for c in retrieved.chunks}
            if expected_sources & retrieved_sources:
                retrieval_hits += 1

        if answer.sources:
            citation_total += 1
            if expected_sources and set(answer.sources) <= expected_sources:
                citation_correct += 1

        if answerable and answer.grounded:
            groundedness_correct += 1
        if not answerable and not answer.grounded:
            refusal_correct += 1

        results.append(
            {
                "id": record["id"],
                "category": record.get("category"),
                "grounded": answer.grounded,
                "sources": answer.sources,
                "latency_ms": round(answer.latency_ms, 3),
            }
        )

    total = len(questions)
    answerable_count = sum(1 for r in questions if r.get("answerable", True))
    unanswerable_count = total - answerable_count

    summary = {
        "total_questions": total,
        "errors": errors,
        "error_rate": errors / total if total else 0.0,
        "retrieval_recall_at_k": (
            retrieval_hits / retrievable_questions if retrievable_questions else None
        ),
        "citation_precision": citation_correct / citation_total if citation_total else None,
        "groundedness_rate": (
            groundedness_correct / answerable_count if answerable_count else None
        ),
        "refusal_accuracy": (
            refusal_correct / unanswerable_count if unanswerable_count else None
        ),
        "latency_p50_ms": statistics.median(latencies) if latencies else None,
        "latency_p95_ms": _percentile(latencies, 95) if latencies else None,
        "avg_input_tokens": statistics.mean(input_tokens) if input_tokens else None,
        "avg_output_tokens": statistics.mean(output_tokens) if output_tokens else None,
    }
    return {"summary": summary, "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RAG evaluation harness.")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use real Groq + HuggingFace adapters (requires GROQ_API_KEY and network access).",
    )
    args = parser.parse_args()

    if args.live:
        confirm = input(
            "This will make real API calls to Groq using GROQ_API_KEY. Continue? [y/N] "
        )
        if confirm.strip().lower() != "y":
            print("Aborted.")
            return

    report = run_evaluation(live=args.live)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
