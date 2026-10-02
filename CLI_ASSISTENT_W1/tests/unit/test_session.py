import pytest

from app.agent.conversation import Conversation
from app.agent.session import ChatSession
from app.llm.base import LLMConnectionError, Role, TextDelta, TokenUsage, UsageReport
from tests.conftest import FakeProvider


def make_session(provider: FakeProvider) -> tuple[ChatSession, Conversation]:
    conversation = Conversation("system", max_messages=10)
    return ChatSession(provider, conversation), conversation


def test_reply_is_streamed_and_recorded() -> None:
    usage = TokenUsage(prompt_tokens=12, completion_tokens=3)
    provider = FakeProvider([TextDelta("Hel"), TextDelta("lo"), UsageReport(usage)])
    session, conversation = make_session(provider)
    streamed: list[str] = []

    result = session.send("hi", on_text=streamed.append)

    assert streamed == ["Hel", "lo"]
    assert result == usage
    assert [(m.role, m.content) for m in conversation.messages[1:]] == [
        (Role.USER, "hi"),
        (Role.ASSISTANT, "Hello"),
    ]


def test_provider_receives_full_history() -> None:
    provider = FakeProvider([TextDelta("ok")])
    session, _ = make_session(provider)

    session.send("first", on_text=lambda _: None)
    session.send("second", on_text=lambda _: None)

    assert [m.content for m in provider.received[1]] == ["system", "first", "ok", "second"]


def test_missing_usage_returns_none() -> None:
    session, _ = make_session(FakeProvider([TextDelta("ok")]))

    assert session.send("hi", on_text=lambda _: None) is None


def test_failed_turn_leaves_history_unchanged() -> None:
    provider = FakeProvider([TextDelta("partial")], error=LLMConnectionError("down"))
    session, conversation = make_session(provider)

    with pytest.raises(LLMConnectionError):
        session.send("hi", on_text=lambda _: None)

    assert len(conversation.messages) == 1  # Only the system prompt.
