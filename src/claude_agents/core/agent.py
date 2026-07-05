"""Base loop machinery every agent pattern reuses.

Each concrete loop (ReAct, Plan-and-Execute, Reflexion, ...) subclasses
:class:`BaseAgent` and implements :meth:`run`. The base class owns the pieces
that are identical across patterns: a Claude client, a transcript, a step budget,
and the tool round-trip that turns ``tool_use`` blocks into ``tool_result`` blocks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..config import settings
from .claude_client import ClaudeClient, LLMResponse
from .tools import ToolRegistry


@dataclass
class Step:
    """One observable unit of the loop, for tracing and evaluation."""

    label: str
    detail: str


@dataclass
class AgentResult:
    """What a loop returns: the answer plus a replayable trace."""

    answer: str
    steps: list[Step] = field(default_factory=list)
    completed: bool = True


class StepBudgetExceeded(RuntimeError):
    """Raised internally when a loop exhausts its iteration budget."""


class BaseAgent:
    """Shared scaffolding for autonomous loops."""

    def __init__(
        self,
        *,
        client: ClaudeClient | None = None,
        tools: ToolRegistry | None = None,
        max_steps: int | None = None,
    ) -> None:
        self.client = client or ClaudeClient()
        self.tools = tools or ToolRegistry()
        self.max_steps = max_steps or settings.max_steps
        self.trace: list[Step] = []

    # -- tracing ---------------------------------------------------------- #
    def log(self, label: str, detail: str) -> None:
        self.trace.append(Step(label=label, detail=detail))

    def _guard_budget(self, step: int) -> None:
        if step >= self.max_steps:
            raise StepBudgetExceeded(
                f"exceeded max_steps={self.max_steps} without finishing"
            )

    # -- tool round-trip -------------------------------------------------- #
    def run_tool_calls(self, response: LLMResponse) -> list[dict[str, Any]]:
        """Execute every ``tool_use`` in a response, return ``tool_result`` blocks.

        All results are returned in a single user turn, as the API expects for
        parallel tool use.
        """
        results: list[dict[str, Any]] = []
        for call in response.tool_calls:
            observation = self.tools.execute(call.name, call.input)
            is_error = observation.startswith("error:")
            self.log(f"tool:{call.name}", observation)
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": observation,
                    "is_error": is_error,
                }
            )
        return results

    def run(self, task: str) -> AgentResult:  # pragma: no cover - abstract
        raise NotImplementedError
