"""One chat session: sends user turns to the LLM and records the replies."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable

from app.agent.conversation import Conversation
from app.llm.base import (
    LLMError,
    LLMProvider,
    Message,
    Role,
    TextDelta,
    TokenUsage,
    ToolCall,
    ToolCallDelta,
    UsageReport,
)
from app.tools.base import ToolRegistry

logger = logging.getLogger(__name__)

TextSink = Callable[[str], None]
ToolCallSink = Callable[[str, str], None]
ToolResultSink = Callable[[str, str], None]

MAX_TOOL_TURNS = 5


class ChatSession:
    """Coordinates a conversation with an LLM provider and tool execution loop."""

    def __init__(
        self,
        provider: LLMProvider,
        conversation: Conversation,
        tools: ToolRegistry | None = None,
    ) -> None:
        self._provider = provider
        self._conversation = conversation
        self._tools = tools

    @property
    def tools(self) -> ToolRegistry | None:
        """The ToolRegistry associated with this session, if any."""
        return self._tools


    def send(
        self,
        user_text: str,
        on_text: TextSink,
        on_tool_call: ToolCallSink | None = None,
        on_tool_result: ToolResultSink | None = None,
    ) -> TokenUsage | None:
        """Send one user message, run the tool-calling loop, and stream the reply.

        Returns:
            Aggregated token usage for the request, or ``None`` if not reported.

        Raises:
            LLMError: If any provider request fails. The conversation is rolled
                back to its state prior to this turn.
        """
        initial_history_len = len(self._conversation._messages)
        self._conversation.add(Message(role=Role.USER, content=user_text))

        total_prompt_tokens = 0
        total_completion_tokens = 0
        has_usage = False

        try:
            for _step in range(MAX_TOOL_TURNS):
                reply_parts: list[str] = []
                tool_calls_acc: dict[int, dict[str, str]] = {}
                step_usage: TokenUsage | None = None

                tool_defs = self._tools.definitions if self._tools else None

                for event in self._provider.stream_chat(
                    self._conversation.messages, tools=tool_defs
                ):
                    if isinstance(event, TextDelta):
                        reply_parts.append(event.text)
                        on_text(event.text)
                    elif isinstance(event, ToolCallDelta):
                        idx = event.index
                        if idx not in tool_calls_acc:
                            tool_calls_acc[idx] = {
                                "id": event.id or f"call_{uuid.uuid4().hex[:8]}",
                                "name": event.name or "",
                                "arguments": event.arguments_delta or "",
                            }
                        else:
                            if event.id:
                                tool_calls_acc[idx]["id"] += event.id
                            if event.name:
                                tool_calls_acc[idx]["name"] += event.name
                            if event.arguments_delta:
                                tool_calls_acc[idx]["arguments"] += event.arguments_delta
                    elif isinstance(event, UsageReport):
                        step_usage = event.usage

                if step_usage is not None:
                    has_usage = True
                    total_prompt_tokens += step_usage.prompt_tokens
                    total_completion_tokens += step_usage.completion_tokens

                # If no tool calls were emitted, this assistant reply concludes the turn.
                if not tool_calls_acc:
                    self._conversation.add(
                        Message(role=Role.ASSISTANT, content="".join(reply_parts))
                    )
                    break

                # The model requested one or more tool calls. Assemble them.
                assembled_calls: list[ToolCall] = []
                for idx in sorted(tool_calls_acc.keys()):
                    call_dict = tool_calls_acc[idx]
                    assembled_calls.append(
                        ToolCall(
                            id=call_dict["id"],
                            name=call_dict["name"],
                            arguments=call_dict["arguments"],
                        )
                    )

                assistant_content = "".join(reply_parts) if reply_parts else None
                self._conversation.add(
                    Message(
                        role=Role.ASSISTANT,
                        content=assistant_content,
                        tool_calls=assembled_calls,
                    )
                )

                # Execute each requested tool and record the response
                for call in assembled_calls:
                    if on_tool_call:
                        on_tool_call(call.name, call.arguments)

                    if self._tools:
                        result_json = self._tools.execute(call.name, call.arguments)
                    else:
                        result_json = '{"status": "error", "message": "No tools configured."}'

                    if on_tool_result:
                        on_tool_result(call.name, result_json)

                    self._conversation.add(
                        Message(
                            role=Role.TOOL,
                            content=result_json,
                            tool_call_id=call.id,
                            name=call.name,
                        )
                    )
                # Next loop iteration sends the updated history (with tool results) to the LLM.
        except LLMError:
            self._conversation.rollback_to(initial_history_len)
            raise

        usage = (
            TokenUsage(
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
            )
            if has_usage
            else None
        )
        if usage is not None:
            logger.info(
                "Turn complete: prompt_tokens=%d completion_tokens=%d",
                usage.prompt_tokens,
                usage.completion_tokens,
            )
        return usage

    def reset(self) -> None:
        self._conversation.reset()
