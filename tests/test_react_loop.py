"""Tests for the ReAct loop's control flow, with a mocked Claude client.

These prove the wiring — stop conditions, the tool round-trip, and the step
budget — run correctly on stubs, without hitting the network.
"""
from __future__ import annotations

from types import SimpleNamespace

from claude_agents.core.claude_client import LLMResponse, ToolCall
from claude_agents.core.tools import default_registry
from claude_agents.loops.react import ReActAgent


class ScriptedClient:
    """A ClaudeClient stand-in that replays a fixed list of responses."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def complete(self, messages, *, system=None, tools=None, stream=True):
        self.calls += 1
        return self._responses.pop(0)


def _raw(content):
    """Build a minimal object exposing ``.content`` like an SDK Message."""
    return SimpleNamespace(content=content)


def test_react_returns_direct_answer_without_tools():
    client = ScriptedClient(
        [LLMResponse(text="The answer is 42.", tool_calls=[], stop_reason="end_turn", raw=_raw([]))]
    )
    agent = ReActAgent(client=client, tools=default_registry())
    result = agent.run("What is the meaning of life?")

    assert result.completed
    assert result.answer == "The answer is 42."
    assert client.calls == 1


def test_react_executes_tool_then_answers():
    tool_response = LLMResponse(
        text="Let me compute that.",
        tool_calls=[ToolCall(id="t1", name="calculator", input={"expression": "6 * 7"})],
        stop_reason="tool_use",
        raw=_raw([{"type": "tool_use", "id": "t1", "name": "calculator", "input": {"expression": "6 * 7"}}]),
    )
    final_response = LLMResponse(
        text="6 * 7 = 42.", tool_calls=[], stop_reason="end_turn", raw=_raw([])
    )
    client = ScriptedClient([tool_response, final_response])
    agent = ReActAgent(client=client, tools=default_registry())
    result = agent.run("What is 6 times 7?")

    assert result.answer == "6 * 7 = 42."
    assert client.calls == 2
    # The calculator observation should be recorded in the trace.
    assert any(step.label == "tool:calculator" and step.detail == "42" for step in result.steps)


def test_react_stops_at_step_budget():
    # Always asks for a tool, never finishes -> budget must halt it.
    def tool_turn():
        return LLMResponse(
            text="thinking",
            tool_calls=[ToolCall(id="t", name="calculator", input={"expression": "1+1"})],
            stop_reason="tool_use",
            raw=_raw([{"type": "tool_use", "id": "t", "name": "calculator", "input": {"expression": "1+1"}}]),
        )

    client = ScriptedClient([tool_turn() for _ in range(5)])
    agent = ReActAgent(client=client, tools=default_registry(), max_steps=3)
    result = agent.run("loop forever")

    assert not result.completed
    assert client.calls == 3
