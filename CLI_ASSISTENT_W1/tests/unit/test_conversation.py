import pytest

from app.agent.conversation import Conversation
from app.llm.base import Message, Role


def user(text: str) -> Message:
    return Message(role=Role.USER, content=text)


def assistant(text: str) -> Message:
    return Message(role=Role.ASSISTANT, content=text)


def test_system_prompt_comes_first() -> None:
    conversation = Conversation("system text", max_messages=4)
    conversation.add(user("hi"))

    assert conversation.messages[0] == Message(role=Role.SYSTEM, content="system text")
    assert conversation.messages[1] == user("hi")


def test_oldest_messages_are_dropped_but_system_prompt_is_kept() -> None:
    conversation = Conversation("system", max_messages=4)
    for turn in range(3):
        conversation.add(user(f"q{turn}"))
        conversation.add(assistant(f"a{turn}"))

    contents = [message.content for message in conversation.messages]

    assert contents == ["system", "q1", "a1", "q2", "a2"]


def test_history_never_starts_with_an_assistant_reply() -> None:
    conversation = Conversation("system", max_messages=2)
    conversation.add(user("q0"))
    conversation.add(assistant("a0"))
    conversation.add(user("q1"))  # Trimming to 2 would leave [a0, q1].

    assert [m.content for m in conversation.messages] == ["system", "q1"]


def test_remove_last_and_reset() -> None:
    conversation = Conversation("system", max_messages=4)
    conversation.add(user("q0"))
    conversation.remove_last()
    assert len(conversation.messages) == 1

    conversation.add(user("q1"))
    conversation.reset()
    assert len(conversation.messages) == 1


def test_adding_a_system_message_is_rejected() -> None:
    conversation = Conversation("system", max_messages=4)

    with pytest.raises(ValueError):
        conversation.add(Message(role=Role.SYSTEM, content="override"))
