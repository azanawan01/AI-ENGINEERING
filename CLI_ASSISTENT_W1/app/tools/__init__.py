"""Tools package for the CLI assistant."""

from __future__ import annotations

from openai import OpenAI

from app.tools.base import Tool, ToolRegistry
from app.tools.clean_csv import CleanCSVTool
from app.tools.database import NaturalLanguageDatabaseTool
from app.tools.grammar import FixGrammarTool
from app.tools.translate import TranslateTextTool


def create_default_registry(
    client: OpenAI | None = None,
    model: str = "",
    workspace_dir: str = "workspace",
) -> ToolRegistry:
    """Create and return a ToolRegistry with all 4 built-in tools configured."""
    return ToolRegistry(
        [
            CleanCSVTool(workspace_dir=workspace_dir),
            FixGrammarTool(client=client, model=model),
            TranslateTextTool(client=client, model=model),
            NaturalLanguageDatabaseTool(client=client, model=model, workspace_dir=workspace_dir),
        ]
    )


__all__ = [
    "CleanCSVTool",
    "FixGrammarTool",
    "NaturalLanguageDatabaseTool",
    "Tool",
    "ToolRegistry",
    "TranslateTextTool",
    "create_default_registry",
]
