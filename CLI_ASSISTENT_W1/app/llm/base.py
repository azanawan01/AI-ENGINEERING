"""Provider-neutral LLM interface.

The agent and CLI depend only on the types in this module. Each concrete
provider (see ``openai_compatible.py``) translates between these types and its
own SDK, so swapping providers never touches business logic.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field


class Role(StrEnum):
    """Who authored a message."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCall(BaseModel):
    """A tool invocation requested by the model."""

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    arguments: str


class Message(BaseModel):
    """One entry in the conversation sent to the model."""

    model_config = ConfigDict(frozen=True)

    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class TokenUsage(BaseModel):
    """Token counts reported by the server for one request."""

    model_config = ConfigDict(frozen=True)

    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class SamplingSettings(BaseModel):
    """Sampling parameters applied to every request."""

    model_config = ConfigDict(frozen=True)

    temperature: float = Field(ge=0.0, le=2.0)
    top_p: float = Field(gt=0.0, le=1.0)


@dataclass(frozen=True)
class TextDelta:
    """A fragment of the assistant's reply, emitted as soon as it arrives."""

    text: str


@dataclass(frozen=True)
class ToolCallDelta:
    """A streamed fragment of a tool call."""

    index: int
    id: str | None = None
    name: str | None = None
    arguments_delta: str = ""


@dataclass(frozen=True)
class UsageReport:
    """Token usage for the whole request, emitted once at the end of a stream."""

    usage: TokenUsage


class ToolDefinition(BaseModel):
    """Specification of a tool that the LLM can invoke."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    parameters: dict[str, object]


# Every item a provider stream can yield.
StreamEvent = TextDelta | ToolCallDelta | UsageReport


class LLMError(Exception):
    """Base class for provider failures. Messages are safe to show to users."""


class LLMConnectionError(LLMError):
    """The LLM server could not be reached or did not respond in time."""


class LLMResponseError(LLMError):
    """The LLM server rejected the request or returned an unusable response."""


class LLMProvider(Protocol):
    """Anything that can stream a chat completion."""

    def stream_chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolDefinition] | None = None,
    ) -> Iterator[StreamEvent]:
        """Send ``messages`` and yield events as the reply is generated.

        Args:
            messages: Conversation history to send to the model.
            tools: Optional tool definitions the model may call.

        Raises:
            LLMError: On any provider failure, including mid-stream failures.
        """
        ...
