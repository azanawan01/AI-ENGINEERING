import json
from collections.abc import Iterator, Sequence

from app.agent.conversation import Conversation
from app.agent.session import ChatSession
from app.llm.base import (
    Message,
    Role,
    StreamEvent,
    TextDelta,
    TokenUsage,
    ToolCallDelta,
    ToolDefinition,
    UsageReport,
)
from app.tools.base import ToolRegistry
from app.tools.grammar import FixGrammarTool


class MultiStepFakeProvider:
    """Simulates a multi-turn tool-calling conversation with an LLM."""

    def __init__(self, steps: list[list[StreamEvent]]) -> None:
        self._steps = steps
        self._step_idx = 0
        self.received_histories: list[list[Message]] = []

    def stream_chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolDefinition] | None = None,
    ) -> Iterator[StreamEvent]:
        self.received_histories.append(list(messages))
        if self._step_idx < len(self._steps):
            events = self._steps[self._step_idx]
            self._step_idx += 1
            yield from events


def test_agent_loop_executes_tool_call_and_streams_final_response() -> None:
    # Turn 1: Model calls fix_grammar tool
    step_1_events = [
        ToolCallDelta(
            index=0,
            id="call_123",
            name="fix_grammar",
            arguments_delta='{"text": "she do not goes"}',
        ),
        UsageReport(TokenUsage(prompt_tokens=15, completion_tokens=10)),
    ]
    # Turn 2: Model receives tool output and streams final explanation
    step_2_events = [
        TextDelta("I fixed the grammar for you: She did not go."),
        UsageReport(TokenUsage(prompt_tokens=30, completion_tokens=15)),
    ]

    provider = MultiStepFakeProvider([step_1_events, step_2_events])
    conversation = Conversation("You are a helpful assistant.", max_messages=10)
    registry = ToolRegistry([FixGrammarTool()])
    session = ChatSession(provider, conversation, tools=registry)

    streamed_text: list[str] = []
    invoked_tools: list[tuple[str, str]] = []
    tool_results: list[tuple[str, str]] = []

    usage = session.send(
        "Please fix: she do not goes",
        on_text=streamed_text.append,
        on_tool_call=lambda name, args: invoked_tools.append((name, args)),
        on_tool_result=lambda name, res: tool_results.append((name, res)),
    )

    # 1. Tool execution callbacks fired
    assert len(invoked_tools) == 1
    assert invoked_tools[0][0] == "fix_grammar"
    assert "she do not goes" in invoked_tools[0][1]

    assert len(tool_results) == 1
    assert tool_results[0][0] == "fix_grammar"
    res_data = json.loads(tool_results[0][1])
    assert res_data["status"] == "success"
    assert "did not go" in res_data["corrected_text"]

    # 2. Final text response streamed
    assert "".join(streamed_text) == "I fixed the grammar for you: She did not go."

    # 3. Total usage aggregated across all turns
    assert usage is not None
    assert usage.prompt_tokens == 45
    assert usage.completion_tokens == 25
    assert usage.total_tokens == 70

    # 4. Message history verified
    roles = [m.role for m in conversation.messages]
    assert roles == [
        Role.SYSTEM,
        Role.USER,
        Role.ASSISTANT,
        Role.TOOL,
        Role.ASSISTANT,
    ]
    assert conversation.messages[2].tool_calls is not None
    assert conversation.messages[2].tool_calls[0].name == "fix_grammar"
    assert conversation.messages[3].tool_call_id == "call_123"
    assert conversation.messages[4].content == "I fixed the grammar for you: She did not go."
