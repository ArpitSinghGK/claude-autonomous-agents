"""Tool abstraction and registry shared by every agent loop.

A ``Tool`` is a typed, self-describing action the agent can take. The registry
renders the Anthropic tool schemas and dispatches ``tool_use`` blocks back to the
Python callables — the seam a real deployment would extend with DB, HTTP, or
vector-store connectors.
"""
from __future__ import annotations

import ast
import operator
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Tool:
    """A single callable the agent may invoke.

    ``input_schema`` is a JSON Schema object exactly as the Anthropic API expects.
    ``run`` receives the validated input dict and returns a string observation.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    run: Callable[[dict[str, Any]], str]

    def to_anthropic(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class ToolRegistry:
    """Holds the available tools and executes tool calls by name."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool
        return tool

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return sorted(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [t.to_anthropic() for t in self._tools.values()]

    def execute(self, name: str, tool_input: dict[str, Any]) -> str:
        """Dispatch a tool call, returning an observation string.

        Errors are returned (not raised) so the loop can feed them back to Claude
        as a recoverable ``tool_result`` with ``is_error=True``.
        """
        tool = self._tools.get(name)
        if tool is None:
            return f"error: unknown tool '{name}'"
        try:
            return tool.run(tool_input)
        except Exception as exc:  # surfaced to the model, not crashed on
            return f"error: {type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------- #
# Example tools. Business logic is intentionally minimal — real connectors      #
# (search API, vector store, database) would slot in behind the same interface. #
# --------------------------------------------------------------------------- #

_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST) -> float:
    """Evaluate a small arithmetic AST without touching ``eval``."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPERATORS:
        return _SAFE_OPERATORS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPERATORS:
        return _SAFE_OPERATORS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("unsupported expression")


def calculator(tool_input: dict[str, Any]) -> str:
    expr = str(tool_input.get("expression", "")).strip()
    if not expr:
        return "error: 'expression' is required"
    try:
        result = _safe_eval(ast.parse(expr, mode="eval"))
    except (ValueError, SyntaxError, ZeroDivisionError) as exc:
        return f"error: {exc}"
    return str(int(result) if result.is_integer() else result)


def web_search_stub(tool_input: dict[str, Any]) -> str:
    """Placeholder for a real search connector (Tavily/Brave/SerpAPI/etc.).

    A production build wires the provider client here; the agent contract — a
    query in, ranked snippets out — stays identical.
    """
    query = str(tool_input.get("query", "")).strip()
    return (
        f"[stub] top results for {query!r} would appear here. "
        "TODO: wire a real search provider behind this interface."
    )


def default_registry() -> ToolRegistry:
    """A registry pre-loaded with the example tools used across the demos."""
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="calculator",
            description="Evaluate a basic arithmetic expression (+ - * / ** %).",
            input_schema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "e.g. '3 * (4 + 2)'",
                    }
                },
                "required": ["expression"],
            },
            run=calculator,
        )
    )
    registry.register(
        Tool(
            name="web_search",
            description="Search the web for up-to-date information (stubbed).",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"],
            },
            run=web_search_stub,
        )
    )
    return registry
