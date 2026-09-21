from __future__ import annotations

from pathlib import Path

import pytest

from app.conversation.history import ConversationHistory
from app.errors import ModelTimeoutError
from app.generation.answerer import UNCERTAIN_ANSWER, Answerer
from app.generation.model import FakeChatModel
from app.ingestion.indexer import Indexer
from app.retrieval.embeddings import FakeEmbeddingProvider
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import NumpyVectorStore


def _index_sample_documents(tmp_path: Path) -> NumpyVectorStore:
    documents_path = tmp_path / "documents"
    documents_path.mkdir()
    (documents_path / "policies.md").write_text(
        "Cancellations are allowed within 14 days of purchase.", encoding="utf-8"
    )
    embedder = FakeEmbeddingProvider(dimension=32)
    store = NumpyVectorStore(index_path=tmp_path / "index", index_version="v1")
    indexer = Indexer(
        embedding_provider=embedder, vector_store=store, chunk_size=200, chunk_overlap=20
    )
    indexer.index_directory(documents_path)
    return store


def test_question_to_grounded_answer_end_to_end(tmp_path: Path) -> None:
    store = _index_sample_documents(tmp_path)
    embedder = FakeEmbeddingProvider(dimension=32)
    retriever = Retriever(embedder, store, top_k=3, similarity_threshold=0.0)
    model = FakeChatModel(
        response_text="Cancellations are allowed within 14 days.\nSources: policies.md"
    )
    answerer = Answerer(retriever=retriever, chat_model=model)

    answer = answerer.answer("What is the cancellation policy?")

    assert answer.grounded is True
    assert answer.sources == ["policies.md"]
    assert "14 days" in answer.text


def test_question_with_no_matching_context_returns_uncertain(tmp_path: Path) -> None:
    store = _index_sample_documents(tmp_path)
    embedder = FakeEmbeddingProvider(dimension=32)
    retriever = Retriever(embedder, store, top_k=3, similarity_threshold=0.99)
    model = FakeChatModel()
    answerer = Answerer(retriever=retriever, chat_model=model)

    answer = answerer.answer("completely unrelated spaceship banana topic")

    assert answer.text == UNCERTAIN_ANSWER
    assert answer.grounded is False
    assert len(model.calls) == 0


def test_provider_failure_propagates_as_model_error(tmp_path: Path) -> None:
    store = _index_sample_documents(tmp_path)
    embedder = FakeEmbeddingProvider(dimension=32)
    retriever = Retriever(embedder, store, top_k=3, similarity_threshold=0.0)
    model = FakeChatModel(raise_error=ModelTimeoutError("provider timed out"))
    answerer = Answerer(retriever=retriever, chat_model=model)

    with pytest.raises(ModelTimeoutError):
        answerer.answer("What is the cancellation policy?")


def test_conversation_history_across_multiple_turns(tmp_path: Path) -> None:
    store = _index_sample_documents(tmp_path)
    embedder = FakeEmbeddingProvider(dimension=32)
    retriever = Retriever(embedder, store, top_k=3, similarity_threshold=0.0)
    model = FakeChatModel(
        response_text="Cancellations are allowed within 14 days.\nSources: policies.md"
    )
    answerer = Answerer(retriever=retriever, chat_model=model)
    history = ConversationHistory(max_messages=10, max_characters=5000)

    history.add_user("What is the cancellation policy?")
    first_answer = answerer.answer(
        "What is the cancellation policy?", history=history.messages()[:-1]
    )
    history.add_assistant(first_answer.text)

    history.add_user("And what about after 14 days?")
    second_answer = answerer.answer(
        "And what about after 14 days?", history=history.messages()[:-1]
    )
    history.add_assistant(second_answer.text)

    assert len(history) == 4
    # The model must have seen the prior turn as context on the second call.
    second_call_messages = model.calls[-1]
    assert any("cancellation policy" in m.content for m in second_call_messages)
