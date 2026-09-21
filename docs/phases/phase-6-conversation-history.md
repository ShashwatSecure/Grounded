# Phase 6 — Conversation History

## What was built

[app/conversation/history.py](../../app/conversation/history.py):

- `ConversationHistory`: a `collections.deque`-backed store that only ever
  accepts `user`/`assistant` `ChatMessage`s (`_add` raises `ValueError` for
  any other role, e.g. `system`). After every append, `_trim()` first drops
  from the front until `len(messages) <= max_messages`, then keeps dropping
  from the front while the total character count exceeds `max_characters`
  (but never drops the last remaining message).
- `to_dict()`/`from_dict()`: plain JSON-serializable round-trip.
- `load_history(path, ...)`/`save_history(history, path)`: file-based
  persistence so history survives across separate CLI process invocations
  (each `python -m app.cli ask ...` is a new process) — used by
  [app/cli.py](../../app/cli.py) with the history file stored at
  `<INDEX_PATH>/history.json`.
- CLI command `reset-history` deletes that file.

## Why these choices

- **System prompt is never stored in `ConversationHistory`**: `Answerer`
  always constructs `[ChatMessage(role="system", ...), *history, user_message]`
  itself (Phase 5); `ConversationHistory` structurally cannot hold a
  `system` message, so no sequence of user turns can ever displace or
  duplicate the system instructions — directly satisfies "User messages
  cannot replace system instructions."
- **Two independent trim bounds (count and characters)**: the plan asks for
  trimming "by count and character/token budget." Message count bounds
  turn-taking; character budget bounds worst-case prompt size regardless of
  how long individual messages are, protecting context-window usage even
  if a single message is unusually long.
- **File persistence keyed off `INDEX_PATH`**: the CLI is stateless between
  invocations, so history needs to live somewhere; storing it alongside the
  vector index (rather than a new top-level directory) keeps all
  per-installation state in one configurable place.

## How it was verified

[tests/unit/test_history.py](../../tests/unit/test_history.py):
- starts empty; preserves user/assistant ordering,
- trims strictly by message count,
- trims by character budget while never emptying entirely,
- `reset()` clears everything,
- a `system`-role message is rejected with `ValueError`,
- save → load round-trips message content exactly,
- loading a missing file returns an empty history (not an error).

[tests/integration/test_question_flow.py](../../tests/integration/test_question_flow.py)
also exercises `ConversationHistory` across two real `Answerer.answer()`
calls, confirming the second call's prompt actually includes the first
turn's content.

## Exit criteria status

- [x] History is preserved within configured limits (`max_history_messages`,
      `max_input_characters` used as the character budget).
- [x] Old history is removed predictably (oldest-first, deque-based).
- [x] User messages cannot replace system instructions (structurally
      enforced by `ConversationHistory` rejecting non-user/assistant roles,
      and by `Answerer` always prepending the system prompt itself).
