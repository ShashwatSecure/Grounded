from __future__ import annotations

from app.domain import Chunk
from app.generation.prompts import SYSTEM_PROMPT, build_user_prompt, format_context


def _chunk(content: str, source: str) -> Chunk:
    return Chunk(chunk_id="c1", document_id="d1", source=source, position=0, content=content)


def test_system_prompt_instructs_grounding_and_refusal() -> None:
    assert "only" in SYSTEM_PROMPT.lower()
    assert "i could not find enough information" in SYSTEM_PROMPT.lower()
    assert "untrusted" in SYSTEM_PROMPT.lower()


def test_format_context_labels_each_chunk_with_source() -> None:
    chunks = [_chunk("refunds within 14 days", "policies.md")]
    context = format_context(chunks)
    assert "[source: policies.md]" in context
    assert "refunds within 14 days" in context


def test_format_context_empty_list_returns_placeholder() -> None:
    assert format_context([]) == "(no context retrieved)"


def test_format_context_delimits_multiple_chunks() -> None:
    chunks = [_chunk("first chunk", "a.md"), _chunk("second chunk", "b.md")]
    context = format_context(chunks)
    assert "[source: a.md]" in context
    assert "[source: b.md]" in context
    assert context.index("[source: a.md]") < context.index("[source: b.md]")


def test_build_user_prompt_includes_question_and_delimited_context() -> None:
    chunks = [_chunk("14 day refund window", "policies.md")]
    prompt = build_user_prompt("What is the refund window?", chunks)
    assert "What is the refund window?" in prompt
    assert "<<<CONTEXT_START>>>" in prompt
    assert "<<<CONTEXT_END>>>" in prompt
    assert "[source: policies.md]" in prompt
