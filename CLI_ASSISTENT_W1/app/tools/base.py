"""Base protocols, definitions, and registry for assistant tools."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from app.llm.base import ToolDefinition

logger = logging.getLogger(__name__)


@runtime_checkable
class Tool(Protocol):
    """Protocol satisfied by any callable tool."""

    @property
    def definition(self) -> ToolDefinition:
        """JSON Schema specification presented to the model."""
        ...

    def execute(self, arguments_json: str) -> str:
        """Validate input arguments and execute the tool.

        Returns:
            A validated JSON string representing the tool result.
        """
        ...


class ToolRegistry:
    """Manages available tools and dispatches calls to them."""

    def __init__(self, tools: Sequence[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        if tools:
            for tool in tools:
                self.register(tool)

    def register(self, tool: Tool) -> None:
        self._tools[tool.definition.name] = tool
        logger.debug("Registered tool: %s", tool.definition.name)

    @property
    def definitions(self) -> list[ToolDefinition]:
        return [tool.definition for tool in self._tools.values()]

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def execute(self, name: str, arguments_json: str) -> str:
        """Find and execute tool by name, catching errors and returning JSON."""
        target_name = name
        if not self.get(target_name):
            try:
                parsed = json.loads(arguments_json)
                if (
                    isinstance(parsed, dict)
                    and "name" in parsed
                    and isinstance(parsed["name"], str)
                    and self.get(parsed["name"])
                ):
                    target_name = parsed["name"]
            except Exception:
                pass

        tool = self.get(target_name)
        if not tool:
            available = list(self._tools.keys())
            return json.dumps(
                {
                    "status": "error",
                    "error_type": "ToolNotFound",
                    "message": f"Tool '{name}' is not registered. Available tools: {available}",
                }
            )

        try:
            return tool.execute(arguments_json)
        except Exception as exc:
            logger.exception("Execution error in tool '%s'", name)
            return json.dumps(
                {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "message": f"Failed executing tool '{name}': {exc}",
                }
            )
