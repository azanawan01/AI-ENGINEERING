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


TOOL_PROMPT_EXAMPLES: dict[str, str] = {
    "clean_csv": "Clean workspace/sample_data.csv, drop duplicates, and visualize salary",
    "fix_grammar": "Can you fix the grammar in: 'She do not goes to school yesterday'?",
    "translate_text": "Translate 'Hello and welcome to our platform' to French with a business tone",
    "natural_language_database": "Show top 5 highest paid employees in workspace/company.db",
}


def format_tools_summary(registry: ToolRegistry) -> str:
    """Format all registered tools into a clean, human-readable terminal string."""
    definitions = registry.definitions
    count = len(definitions)
    lines: list[str] = [
        "=" * 74,
        f"🛠️  REGISTERED ASSISTANT TOOLS ({count} Available)",
        "=" * 74,
    ]
    if not definitions:
        lines.append("\nNo tools currently registered.")
        lines.append("\n" + "=" * 74)
        return "\n".join(lines)

    for i, d in enumerate(definitions, start=1):
        lines.append(f"\n{i}. 🔧 {d.name}")
        lines.append(f"   Description: {d.description}")
        props = d.parameters.get("properties", {})
        req = set(d.parameters.get("required", []))
        if props:
            lines.append("   Parameters:")
            for p, meta in props.items():
                ptype = meta.get("type", "any")
                status = "REQUIRED" if p in req else "optional"
                default_val = meta.get("default")
                def_str = f", default={default_val!r}" if default_val is not None else ""
                desc = meta.get("description", "")
                lines.append(f"     • {p} ({ptype}, {status}{def_str}):")
                if desc:
                    lines.append(f"       {desc}")
        example = TOOL_PROMPT_EXAMPLES.get(d.name)
        if example:
            lines.append(f"   Example Prompt: \"{example}\"")

    lines.append("\n" + "=" * 74)
    return "\n".join(lines)


__all__ = [
    "CleanCSVTool",
    "FixGrammarTool",
    "NaturalLanguageDatabaseTool",
    "Tool",
    "ToolRegistry",
    "TranslateTextTool",
    "create_default_registry",
    "format_tools_summary",
]

