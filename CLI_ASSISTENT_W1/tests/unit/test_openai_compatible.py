"""Provider tests using real SDK chunk types and a fake client (no network)."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

import httpx2  # The OpenAI SDK's own HTTP library, needed to build its error types.
import openai
import pytest
from openai.types.chat import ChatCompletionChunk

from app.llm.base import (
    LLMConnectionError,
    LLMResponseError,
    Message,
    Role,
    SamplingSettings,
    TextDelta,
    TokenUsage,
    UsageReport,
)
from app.llm.openai_compatible import OpenAICompatibleProvider

REQUEST = httpx2.Request("POST", "http://llm.test/v1/chat/completions")


def text_chunk(text: str) -> ChatCompletionChunk:
    return ChatCompletionChunk.model_validate(
        {
            "id": "c1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "test-model",
            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
        }
    )


def usage_chunk(prompt: int, completion: int) -> ChatCompletionChunk:
    return ChatCompletionChunk.model_validate(
        {
            "id": "c1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "test-model",
            "choices": [],
            "usage": {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": prompt + completion,
            },
        }
    )


class FakeCompletions:
    def __init__(self, result: Iterable[ChatCompletionChunk] | Exception) -> None:
        self._result = result
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Iterable[ChatCompletionChunk]:
        self.calls.append(kwargs)
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class FakeClient:
    def __init__(self, result: Iterable[ChatCompletionChunk] | Exception) -> None:
        self.completions = FakeCompletions(result)
        self.chat = self  # Mirrors the SDK's client.chat.completions shape.


def make_provider(client: FakeClient) -> OpenAICompatibleProvider:
    sampling = SamplingSettings(temperature=0.3, top_p=0.9)
    return OpenAICompatibleProvider(client, model="test-model", sampling=sampling)  # type: ignore[arg-type]


MESSAGES = [
    Message(role=Role.SYSTEM, content="be brief"),
    Message(role=Role.USER, content="hi"),
]


def test_stream_yields_text_then_usage() -> None:
    client = FakeClient([text_chunk("Hel"), text_chunk("lo"), usage_chunk(10, 2)])

    events = list(make_provider(client).stream_chat(MESSAGES))

    assert events == [
        TextDelta("Hel"),
        TextDelta("lo"),
        UsageReport(TokenUsage(prompt_tokens=10, completion_tokens=2)),
    ]


def test_request_carries_model_messages_sampling_and_streaming_options() -> None:
    client = FakeClient([])

    list(make_provider(client).stream_chat(MESSAGES))

    call = client.completions.calls[0]
    assert call["model"] == "test-model"
    assert call["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
    ]
    assert call["temperature"] == 0.3
    assert call["top_p"] == 0.9
    assert call["stream"] is True
    assert call["stream_options"] == {"include_usage": True}


def test_empty_deltas_are_skipped() -> None:
    client = FakeClient([text_chunk(""), text_chunk("ok")])

    assert list(make_provider(client).stream_chat(MESSAGES)) == [TextDelta("ok")]


def test_connection_failure_becomes_llm_connection_error() -> None:
    client = FakeClient(openai.APIConnectionError(request=REQUEST))

    with pytest.raises(LLMConnectionError, match="LLM_BASE_URL"):
        list(make_provider(client).stream_chat(MESSAGES))


def test_timeout_becomes_llm_connection_error() -> None:
    client = FakeClient(openai.APITimeoutError(request=REQUEST))

    with pytest.raises(LLMConnectionError):
        list(make_provider(client).stream_chat(MESSAGES))


def test_missing_model_error_tells_user_how_to_fix_it() -> None:
    response = httpx2.Response(404, request=REQUEST)
    client = FakeClient(openai.NotFoundError("model not found", response=response, body=None))

    with pytest.raises(LLMResponseError, match="ollama pull test-model"):
        list(make_provider(client).stream_chat(MESSAGES))


def test_failure_mid_stream_is_translated() -> None:
    def broken_stream() -> Iterator[ChatCompletionChunk]:
        yield text_chunk("partial")
        raise openai.APIConnectionError(request=REQUEST)

    client = FakeClient(broken_stream())
    stream = make_provider(client).stream_chat(MESSAGES)

    assert next(stream) == TextDelta("partial")
    with pytest.raises(LLMConnectionError):
        next(stream)
