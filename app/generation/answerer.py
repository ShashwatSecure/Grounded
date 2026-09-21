"""Grounded answer generation: retrieval -> prompt -> model -> validated answer."""

from __future__ import annotations

import re
import time

from app.domain import Answer, RetrievedContext, Usage
from app.generation.model import ChatMessage, ChatModel
from app.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from app.retrieval.retriever import Retriever

UNCERTAIN_ANSWER = (
    "I could not find enough information in the indexed documents to answer that question."
)

_SOURCES_LINE_RE = re.compile(r"(?im)^sources:\s*(.*)$")


class Answerer:
    """Produces a grounded `Answer` for a question, or a controlled refusal."""

    def __init__(self, retriever: Retriever, chat_model: ChatModel) -> None:
        self.retriever = retriever
        self.chat_model = chat_model

    def answer(self, question: str, history: list[ChatMessage] | None = None) -> Answer:
        start = time.monotonic()
        context: RetrievedContext = self.retriever.retrieve(question)

        if context.is_empty:
            return Answer(
                text=UNCERTAIN_ANSWER,
                sources=[],
                grounded=False,
                usage=Usage(),
                latency_ms=(time.monotonic() - start) * 1000,
            )

        messages = [ChatMessage(role="system", content=SYSTEM_PROMPT)]
        messages.extend(history or [])
        messages.append(
            ChatMessage(role="user", content=build_user_prompt(question, context.chunks))
        )

        response = self.chat_model.generate(messages)
        text, cited_sources = _parse_answer(response.text)

        valid_sources = {c.source for c in context.chunks}
        sources = [s for s in cited_sources if s in valid_sources]
        grounded = len(sources) > 0 and text.strip() != UNCERTAIN_ANSWER

        return Answer(
            text=text,
            sources=sources,
            grounded=grounded,
            usage=response.usage,
            latency_ms=(time.monotonic() - start) * 1000,
        )


def _parse_answer(raw_text: str) -> tuple[str, list[str]]:
    """Split the model's raw text into the answer body and its cited source list."""
    match = _SOURCES_LINE_RE.search(raw_text)
    if not match:
        return raw_text.strip(), []
    sources_part = match.group(1).strip()
    body = raw_text[: match.start()].strip()
    if sources_part.lower() in {"none", "n/a", ""}:
        return body, []
    sources = [s.strip() for s in sources_part.split(",") if s.strip()]
    return body, sources
