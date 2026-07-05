"""Tests for the tool registry and the example tools."""
from __future__ import annotations

import pytest

from claude_agents.core.tools import Tool, ToolRegistry, calculator, default_registry


def test_calculator_evaluates_arithmetic():
    assert calculator({"expression": "3 * (4 + 2)"}) == "18"
    assert calculator({"expression": "2 ** 10"}) == "1024"


def test_calculator_requires_expression():
    assert calculator({}).startswith("error:")


def test_calculator_rejects_unsafe_expression():
    # Names / calls are not part of the safe arithmetic grammar.
    result = calculator({"expression": "__import__('os')"})
    assert result.startswith("error:")


def test_registry_registers_and_lists():
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="echo",
            description="echo",
            input_schema={"type": "object", "properties": {}},
            run=lambda _: "ok",
        )
    )
    assert "echo" in registry
    assert registry.names() == ["echo"]


def test_registry_rejects_duplicate():
    registry = default_registry()
    with pytest.raises(ValueError):
        registry.register(
            Tool(
                name="calculator",
                description="dup",
                input_schema={"type": "object", "properties": {}},
                run=lambda _: "",
            )
        )


def test_registry_execute_unknown_tool_is_soft_error():
    registry = default_registry()
    assert registry.execute("nope", {}).startswith("error: unknown tool")


def test_registry_execute_catches_tool_exception():
    registry = ToolRegistry()

    def boom(_: dict) -> str:
        raise RuntimeError("kaboom")

    registry.register(
        Tool(name="boom", description="", input_schema={"type": "object"}, run=boom)
    )
    result = registry.execute("boom", {})
    assert result.startswith("error: RuntimeError: kaboom")


def test_default_registry_exposes_anthropic_schemas():
    schemas = default_registry().schemas()
    names = {s["name"] for s in schemas}
    assert {"calculator", "web_search"} <= names
    for schema in schemas:
        assert "input_schema" in schema
