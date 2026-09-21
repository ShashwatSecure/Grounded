"""Bounded conversation history, kept separate from system instructions."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

from app.generation.model import ChatMessage


class ConversationHistory:
    """Stores a bounded window of user/assistant turns.

    System instructions are never stored here — callers prepend the system
    prompt themselves each turn, so no history message can ever displace it.
    """

    def __init__(self, max_messages: int, max_characters: int) -> None:
        self.max_messages = max_messages
        self.max_characters = max_characters
        self._messages: deque[ChatMessage] = deque()

    def add_user(self, content: str) -> None:
        self._add(ChatMessage(role="user", content=content))

    def add_assistant(self, content: str) -> None:
        self._add(ChatMessage(role="assistant", content=content))

    def _add(self, message: ChatMessage) -> None:
        if message.role not in ("user", "assistant"):
            raise ValueError("Conversation history only accepts user/assistant messages")
        self._messages.append(message)
        self._trim()

    def _trim(self) -> None:
        while len(self._messages) > self.max_messages:
            self._messages.popleft()
        while self._total_characters() > self.max_characters and len(self._messages) > 1:
            self._messages.popleft()

    def _total_characters(self) -> int:
        return sum(len(m.content) for m in self._messages)

    def messages(self) -> list[ChatMessage]:
        return list(self._messages)

    def reset(self) -> None:
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)

    def to_dict(self) -> dict:
        return {"messages": [{"role": m.role, "content": m.content} for m in self._messages]}

    @classmethod
    def from_dict(
        cls, data: dict, max_messages: int, max_characters: int
    ) -> ConversationHistory:
        history = cls(max_messages, max_characters)
        for item in data.get("messages", []):
            history._add(ChatMessage(role=item["role"], content=item["content"]))
        return history


def load_history(path: Path, max_messages: int, max_characters: int) -> ConversationHistory:
    """Load persisted CLI conversation history, or start empty if none exists."""
    if not path.exists():
        return ConversationHistory(max_messages, max_characters)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ConversationHistory(max_messages, max_characters)
    return ConversationHistory.from_dict(data, max_messages, max_characters)


def save_history(history: ConversationHistory, path: Path) -> None:
    """Persist CLI conversation history to disk between invocations."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

