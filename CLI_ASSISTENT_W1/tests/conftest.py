"""Shared test doubles."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import pytest

from app.llm.base import LLMError, Message, StreamEvent, ToolDefinition


class FakeProvider:
    """Replays scripted events and records what it was sent."""

    def __init__(self, events: list[StreamEvent], error: LLMError | None = None) -> None:
        self._events = events
        self._error = error
        self.received: list[list[Message]] = []
        self.received_tools: list[list[ToolDefinition] | None] = []

    def stream_chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolDefinition] | None = None,
    ) -> Iterator[StreamEvent]:
        self.received.append(list(messages))
        self.received_tools.append(list(tools) if tools else None)
        yield from self._events
        if self._error is not None:
            raise self._error


@pytest.fixture
def valid_env() -> dict[str, str]:
    return {"LLM_MODEL": "test-model"}
