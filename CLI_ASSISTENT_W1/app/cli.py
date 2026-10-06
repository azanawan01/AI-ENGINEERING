"""Interactive terminal loop. Handles user I/O only; no LLM logic lives here."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TextIO

from app.agent.session import ChatSession
from app.llm.base import LLMError, TokenUsage

logger = logging.getLogger(__name__)

EXIT_COMMANDS = frozenset({"/exit", "/quit"})
RESET_COMMAND = "/reset"
TOOLS_COMMANDS = frozenset({"/tools", "/tool"})
HELP_COMMANDS = frozenset({"/help", "/?"})
HELP_TEXT = "Commands: /tools lists available tools, /reset clears conversation, /exit quits."


class ChatCLI:
    def __init__(
        self,
        session: ChatSession,
        show_token_usage: bool,
        read_line: Callable[[str], str] = input,
        output: TextIO | None = None,
    ) -> None:
        self._session = session
        self._show_token_usage = show_token_usage
        self._read_line = read_line
        self._output = output

    def run(self) -> None:
        self._print(f"CLI Assistant ready. {HELP_TEXT}")
        while True:
            try:
                user_text = self._read_line("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                self._print("\nGoodbye.")
                return

            if not user_text:
                continue
            if user_text.lower() in EXIT_COMMANDS:
                self._print("Goodbye.")
                return
            if user_text.lower() == RESET_COMMAND:
                self._session.reset()
                self._print("Conversation cleared.")
                continue
            if user_text.lower() in TOOLS_COMMANDS:
                self._show_tools()
                continue
            if user_text.lower() in HELP_COMMANDS:
                self._show_help()
                continue

            self._handle_turn(user_text)

    def _show_tools(self) -> None:
        if self._session.tools:
            from app.tools import format_tools_summary

            self._print(format_tools_summary(self._session.tools))
        else:
            self._print("No tools currently registered.")

    def _show_help(self) -> None:
        self._print(
            "Commands:\n"
            "  /tools  Show all registered tools and their parameters\n"
            "  /reset  Clear current conversation history\n"
            "  /help   Show available commands\n"
            "  /exit   Quit the assistant (or /quit)"
        )


    def _handle_turn(self, user_text: str) -> None:
        self._write("Assistant: ")
        try:
            usage = self._session.send(
                user_text,
                on_text=self._write,
                on_tool_call=self._on_tool_call,
                on_tool_result=self._on_tool_result,
            )
        except LLMError as exc:
            # Provider errors carry user-safe messages; details are in the log.
            self._print(f"\n[error] {exc}")
            return
        except KeyboardInterrupt:
            logger.info("User interrupted a response")
            self._print("\n[interrupted]")
            return

        self._print("")
        if self._show_token_usage and usage is not None:
            self._print(_format_usage(usage))

    def _on_tool_call(self, tool_name: str, arguments_json: str) -> None:
        self._print(f"\n\n⚙️ [Calling tool: {tool_name}]")
        logger.info("Tool invoked: %s with arguments: %s", tool_name, arguments_json)

    def _on_tool_result(self, tool_name: str, result_json: str) -> None:
        self._print(f"✓ [Tool {tool_name} returned validated response]\n")
        logger.info("Tool %s returned result of %d bytes", tool_name, len(result_json))

    def _write(self, text: str) -> None:
        print(text, end="", flush=True, file=self._output)

    def _print(self, text: str) -> None:
        print(text, file=self._output)


def _format_usage(usage: TokenUsage) -> str:
    return (
        f"[tokens: prompt {usage.prompt_tokens}, "
        f"completion {usage.completion_tokens}, total {usage.total_tokens}]"
    )
