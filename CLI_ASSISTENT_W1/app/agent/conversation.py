"""Conversation history with a bounded context window."""

from __future__ import annotations

from app.llm.base import Message, Role


class Conversation:
    """Holds the messages sent to the model on every turn.

    The system prompt is always kept. Beyond ``max_messages`` other messages,
    the oldest are dropped so requests stay within the model's context window.

    Counting messages rather than tokens is a deliberate simplification: an
    exact token count needs the model's own tokenizer, which the OpenAI-
    compatible API does not expose.
    """

    def __init__(self, system_prompt: str, max_messages: int) -> None:
        if max_messages < 2:
            raise ValueError("max_messages must allow at least one user/assistant exchange")
        self._system = Message(role=Role.SYSTEM, content=system_prompt)
        self._max_messages = max_messages
        self._messages: list[Message] = []

    @property
    def messages(self) -> list[Message]:
        """The system prompt followed by the retained history (a copy)."""
        return [self._system, *self._messages]

    def add(self, message: Message) -> None:
        if message.role is Role.SYSTEM:
            raise ValueError("The system prompt is fixed; add user or assistant messages only")
        self._messages.append(message)
        self._trim()

    def remove_last(self) -> None:
        """Drop the newest message, e.g. a user turn whose request failed."""
        if self._messages:
            self._messages.pop()

    def rollback_to(self, count: int) -> None:
        """Roll back message history to a specific message count if a turn fails."""
        while len(self._messages) > count:
            self._messages.pop()

    def reset(self) -> None:
        self._messages.clear()

    def _trim(self) -> None:
        overflow = len(self._messages) - self._max_messages
        if overflow > 0:
            del self._messages[:overflow]
        # A history that starts with an assistant reply has lost its question.
        while self._messages and self._messages[0].role is not Role.USER:
            self._messages.pop(0)
