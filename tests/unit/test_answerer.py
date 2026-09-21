from __future__ import annotations

from app.domain import Chunk, RetrievedContext
from app.generation.answerer import UNCERTAIN_ANSWER, Answerer
from app.generation.model import FakeChatModel
from app.retrieval.embeddings import FakeEmbeddingProvider
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import NumpyVectorStore


class _FixedRetriever:
    """Test double returning a pre-built RetrievedContext regardless of query."""

    def __init__(self, context: RetrievedContext) -> None:
        self._context = context

    def retrieve(self, query: str) -> RetrievedContext:
        return self._context


def _chunk(content: str, source: str) -> Chunk:
    return Chunk(chunk_id="c1", document_id="d1", source=source, position=0, content=content)


def test_answer_returns_uncertain_when_no_context() -> None:
    empty_context = RetrievedContext(chunks=[], scores=[], query="q", retrieval_id="r1")
    retriever = _FixedRetriever(empty_context)
    model = FakeChatModel()
    answerer = Answerer(retriever=retriever, chat_model=model)

    answer = answerer.answer("unrelated question")

    assert answer.text == UNCERTAIN_ANSWER
    assert answer.grounded is False
    assert answer.sources == []
    assert len(model.calls) == 0  # model must not be called with no context


def test_answer_is_grounded_and_cites_valid_source() -> None:
    chunk = _chunk("Refunds are allowed within 14 days.", "policies.md")
    context = RetrievedContext(chunks=[chunk], scores=[0.9], query="q", retrieval_id="r1")
    retriever = _FixedRetriever(context)
    model = FakeChatModel(
        response_text="Refunds are allowed within 14 days.\nSources: policies.md"
    )
    answerer = Answerer(retriever=retriever, chat_model=model)

    answer = answerer.answer("What is the refund window?")

    assert "14 days" in answer.text
    assert answer.sources == ["policies.md"]
    assert answer.grounded is True


def test_answer_strips_hallucinated_sources_not_in_context() -> None:
    chunk = _chunk("Refunds are allowed within 14 days.", "policies.md")
    context = RetrievedContext(chunks=[chunk], scores=[0.9], query="q", retrieval_id="r1")
    retriever = _FixedRetriever(context)
    model = FakeChatModel(
        response_text="Refunds within 14 days.\nSources: policies.md, made-up-file.md"
    )
    answerer = Answerer(retriever=retriever, chat_model=model)

    answer = answerer.answer("What is the refund window?")

    assert answer.sources == ["policies.md"]
    assert "made-up-file.md" not in answer.sources


def test_answer_not_grounded_when_no_sources_cited() -> None:
    chunk = _chunk("Refunds are allowed within 14 days.", "policies.md")
    context = RetrievedContext(chunks=[chunk], scores=[0.9], query="q", retrieval_id="r1")
    retriever = _FixedRetriever(context)
    model = FakeChatModel(response_text="I'm not sure.\nSources: none")
    answerer = Answerer(retriever=retriever, chat_model=model)

    answer = answerer.answer("What is the refund window?")

    assert answer.sources == []
    assert answer.grounded is False


def test_answer_includes_history_messages_sent_to_model() -> None:
    chunk = _chunk("Refunds are allowed within 14 days.", "policies.md")
    context = RetrievedContext(chunks=[chunk], scores=[0.9], query="q", retrieval_id="r1")
    retriever = _FixedRetriever(context)
    model = FakeChatModel(response_text="14 days.\nSources: policies.md")
    answerer = Answerer(retriever=retriever, chat_model=model)

    from app.generation.model import ChatMessage

    history = [
        ChatMessage(role="user", content="hi"),
        ChatMessage(role="assistant", content="hello"),
    ]
    answerer.answer("What is the refund window?", history=history)

    sent_messages = model.calls[0]
    assert sent_messages[0].role == "system"
    assert sent_messages[1].content == "hi"
    assert sent_messages[2].content == "hello"
    assert sent_messages[-1].role == "user"


def test_end_to_end_with_real_retriever_and_fake_embeddings(tmp_path) -> None:
    embedder = FakeEmbeddingProvider(dimension=32)
    store = NumpyVectorStore(index_path=tmp_path, index_version="v1")
    chunk = _chunk("Cancellations are allowed within 14 days of purchase.", "policies.md")
    store.add([chunk], embedder.embed_documents([chunk.content]))
    retriever = Retriever(embedder, store, top_k=3, similarity_threshold=0.0)
    model = FakeChatModel(response_text="Within 14 days.\nSources: policies.md")
    answerer = Answerer(retriever=retriever, chat_model=model)

    answer = answerer.answer("What is the cancellation policy?")

    assert answer.grounded is True
    assert answer.sources == ["policies.md"]
