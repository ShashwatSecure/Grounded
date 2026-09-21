from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from evaluate import QUESTIONS_FILE, run_evaluation  # noqa: E402

REQUIRED_CATEGORIES = {
    "direct_fact_lookup",
    "multi_document",
    "paraphrased",
    "ambiguous",
    "unsupported",
    "prompt_injection_in_document",
    "prompt_injection_in_question",
    "very_long_question",
    "empty_question",
}


def _load_records() -> list[dict]:
    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_dataset_has_between_20_and_50_questions() -> None:
    records = _load_records()
    assert 20 <= len(records) <= 50


def test_dataset_records_have_required_fields() -> None:
    records = _load_records()
    for record in records:
        assert "id" in record
        assert "question" in record
        assert "expected_sources" in record
        assert "expected_answer_points" in record
        assert "answerable" in record


def test_dataset_ids_are_unique() -> None:
    records = _load_records()
    ids = [r["id"] for r in records]
    assert len(ids) == len(set(ids))


def test_dataset_covers_all_required_categories() -> None:
    records = _load_records()
    categories = {r.get("category") for r in records}
    missing = REQUIRED_CATEGORIES - categories
    assert not missing, f"Missing evaluation categories: {missing}"


def test_offline_evaluation_runs_without_network_and_reports_metrics() -> None:
    report = run_evaluation(live=False)
    summary = report["summary"]

    assert summary["total_questions"] == len(_load_records())
    assert summary["errors"] == 0
    assert summary["retrieval_recall_at_k"] is not None
    assert summary["citation_precision"] is not None
    assert summary["groundedness_rate"] is not None
    assert summary["refusal_accuracy"] is not None
    assert summary["latency_p50_ms"] is not None


def test_offline_evaluation_handles_empty_question_as_unanswerable() -> None:
    report = run_evaluation(live=False)
    empty_question_results = [
        r for r in report["results"] if r.get("category") == "empty_question"
    ]
    assert empty_question_results
    for result in empty_question_results:
        assert result["grounded"] is False
