import io
from collections.abc import Callable

from app.agent.conversation import Conversation
from app.agent.session import ChatSession
from app.cli import ChatCLI
from app.llm.base import LLMConnectionError, TextDelta, TokenUsage, UsageReport
from tests.conftest import FakeProvider


def scripted_input(lines: list[str]) -> Callable[[str], str]:
    remaining = iter(lines)

    def read_line(_prompt: str) -> str:
        try:
            return next(remaining)
        except StopIteration:
            raise EOFError from None

    return read_line


def run_cli(provider: FakeProvider, lines: list[str], show_usage: bool = True) -> str:
    session = ChatSession(provider, Conversation("system", max_messages=10))
    output = io.StringIO()
    ChatCLI(session, show_usage, read_line=scripted_input(lines), output=output).run()
    return output.getvalue()


def test_streams_reply_and_shows_usage() -> None:
    usage = UsageReport(TokenUsage(prompt_tokens=5, completion_tokens=1))
    output = run_cli(FakeProvider([TextDelta("Hi"), TextDelta("!"), usage]), ["hello", "/exit"])

    assert "Assistant: Hi!" in output
    assert "[tokens: prompt 5, completion 1, total 6]" in output
    assert output.rstrip().endswith("Goodbye.")


def test_usage_can_be_hidden() -> None:
    usage = UsageReport(TokenUsage(prompt_tokens=5, completion_tokens=1))
    output = run_cli(FakeProvider([TextDelta("Hi"), usage]), ["hello"], show_usage=False)

    assert "[tokens" not in output


def test_provider_error_is_shown_without_traceback_and_loop_continues() -> None:
    provider = FakeProvider([], error=LLMConnectionError("Could not reach the LLM server."))

    output = run_cli(provider, ["hello", "again", "/exit"])

    assert output.count("[error] Could not reach the LLM server.") == 2
    assert "Traceback" not in output


def test_blank_input_is_ignored_and_reset_is_acknowledged() -> None:
    provider = FakeProvider([TextDelta("ok")])

    output = run_cli(provider, ["   ", "/reset", "/exit"])

    assert "Conversation cleared." in output
    assert provider.received == []


def test_tools_command_displays_registered_tools() -> None:
    from app.tools import create_default_registry

    provider = FakeProvider([])
    session = ChatSession(
        provider, Conversation("system", max_messages=10), tools=create_default_registry()
    )
    output = io.StringIO()
    ChatCLI(session, False, read_line=scripted_input(["/tools", "/exit"]), output=output).run()
    val = output.getvalue()
    assert "REGISTERED ASSISTANT TOOLS" in val
    assert "clean_csv" in val
    assert "fix_grammar" in val
    assert "translate_text" in val
    assert "natural_language_database" in val


def test_help_command_displays_help() -> None:
    provider = FakeProvider([])
    output = run_cli(provider, ["/help", "/exit"])
    assert "/tools" in output
    assert "/reset" in output
    assert "/exit" in output

