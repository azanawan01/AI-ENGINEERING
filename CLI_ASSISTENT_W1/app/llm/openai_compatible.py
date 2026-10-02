"""LLM provider for any server that speaks the OpenAI Chat Completions API.

That includes Ollama (``http://localhost:11434/v1``) and OpenAI itself, so
switching between them is a configuration change, not a code change.

This is the only module that imports the OpenAI SDK.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from typing import Any

import openai
from openai import OpenAI
from openai.types.chat import ChatCompletionChunk, ChatCompletionMessageParam

from app.config import AppConfig
from app.llm.base import (
    LLMConnectionError,
    LLMError,
    LLMResponseError,
    Message,
    SamplingSettings,
    StreamEvent,
    TextDelta,
    TokenUsage,
    ToolCallDelta,
    ToolDefinition,
    UsageReport,
)

logger = logging.getLogger(__name__)


class OpenAICompatibleProvider:
    """Streams chat completions through the official OpenAI SDK."""

    def __init__(self, client: OpenAI, model: str, sampling: SamplingSettings) -> None:
        self._client = client
        self._model = model
        self._sampling = sampling

    @property
    def raw_client(self) -> OpenAI:
        """Expose the underlying OpenAI SDK client for direct tool operations."""
        return self._client

    @property
    def model(self) -> str:
        """Model identifier."""
        return self._model

    @classmethod
    def from_config(cls, config: AppConfig) -> OpenAICompatibleProvider:
        client = OpenAI(
            base_url=config.llm_base_url,
            api_key=config.llm_api_key.get_secret_value(),
            timeout=config.llm_request_timeout_seconds,
            # Local servers fail fast when down; retries only delay the error.
            max_retries=0,
        )
        sampling = SamplingSettings(temperature=config.llm_temperature, top_p=config.llm_top_p)
        return cls(client=client, model=config.llm_model, sampling=sampling)

    def stream_chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolDefinition] | None = None,
    ) -> Iterator[StreamEvent]:
        logger.debug(
            "Requesting completion: model=%s messages=%d tools=%d",
            self._model,
            len(messages),
            len(tools) if tools else 0,
        )
        openai_tools = None
        if tools:
            openai_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in tools
            ]

        try:
            kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": [_to_openai_message(message) for message in messages],
                "temperature": self._sampling.temperature,
                "top_p": self._sampling.top_p,
                "stream": True,
                # Without this, streamed responses carry no token counts.
                "stream_options": {"include_usage": True},
            }
            if openai_tools:
                kwargs["tools"] = openai_tools

            stream = self._client.chat.completions.create(**kwargs)
            # Iteration is inside the try: network failures can happen mid-stream.
            for chunk in stream:
                yield from _events_from_chunk(chunk)
        except openai.APIConnectionError as exc:
            # Also covers APITimeoutError, which subclasses it.
            logger.error("Cannot reach LLM server: %s", exc)
            raise LLMConnectionError(
                "Could not reach the LLM server. Is it running, and is LLM_BASE_URL correct?"
            ) from exc
        except openai.APIStatusError as exc:
            logger.error("LLM server returned HTTP %s: %s", exc.status_code, exc.message)
            raise LLMResponseError(_describe_status_error(exc.status_code, self._model)) from exc
        except openai.OpenAIError as exc:
            logger.exception("Unexpected LLM SDK error")
            raise LLMError("The LLM request failed unexpectedly. See the log for details.") from exc


def _to_openai_message(message: Message) -> ChatCompletionMessageParam:
    # The SDK's message param is a union of TypedDicts keyed by role; building
    # it from a plain dict is the documented usage.
    payload: dict[str, Any] = {"role": message.role.value}
    if message.content is not None:
        payload["content"] = message.content
    elif message.tool_calls:
        payload["content"] = ""

    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": tc.arguments,
                },
            }
            for tc in message.tool_calls
        ]
    if message.tool_call_id:
        payload["tool_call_id"] = message.tool_call_id
    if message.name:
        payload["name"] = message.name

    return payload  # type: ignore[return-value]


def _events_from_chunk(chunk: ChatCompletionChunk) -> Iterator[StreamEvent]:
    for choice in chunk.choices:
        if choice.delta.content:
            yield TextDelta(choice.delta.content)
        if choice.delta.tool_calls:
            for tc in choice.delta.tool_calls:
                yield ToolCallDelta(
                    index=tc.index,
                    id=tc.id,
                    name=tc.function.name if tc.function else None,
                    arguments_delta=tc.function.arguments
                    if tc.function and tc.function.arguments
                    else "",
                )
    # The usage chunk arrives last, with an empty ``choices`` list.
    if chunk.usage is not None:
        yield UsageReport(
            TokenUsage(
                prompt_tokens=chunk.usage.prompt_tokens,
                completion_tokens=chunk.usage.completion_tokens,
            )
        )


def _describe_status_error(status_code: int, model: str) -> str:
    if status_code == 404:
        return f"Model '{model}' was not found. Pull it first (for example `ollama pull {model}`)."
    if status_code in (401, 403):
        return "The LLM server rejected the credentials. Check LLM_API_KEY."
    if status_code == 429:
        return "The LLM server is rate limiting requests. Wait a moment and try again."
    return f"The LLM server returned an error (HTTP {status_code})."
