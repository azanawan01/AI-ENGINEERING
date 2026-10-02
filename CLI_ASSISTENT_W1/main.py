"""Entry point: load configuration, wire dependencies, start the CLI."""

from __future__ import annotations

import sys

from app.agent.conversation import Conversation
from app.agent.prompts import SYSTEM_PROMPT
from app.agent.session import ChatSession
from app.cli import ChatCLI
from app.config import ConfigError, load_config
from app.llm.openai_compatible import OpenAICompatibleProvider
from app.logging_config import configure_logging
from app.tools import create_default_registry


def main() -> int:
    try:
        config = load_config()
    except ConfigError as exc:
        print(f"{exc}\nSee .env.example for the expected settings.", file=sys.stderr)
        return 1

    configure_logging(config.log_level, config.log_file)

    provider = OpenAICompatibleProvider.from_config(config)
    tools = create_default_registry(client=provider.raw_client, model=config.llm_model)
    conversation = Conversation(SYSTEM_PROMPT, max_messages=config.max_history_messages)
    session = ChatSession(provider, conversation, tools=tools)
    ChatCLI(session, show_token_usage=config.show_token_usage).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
