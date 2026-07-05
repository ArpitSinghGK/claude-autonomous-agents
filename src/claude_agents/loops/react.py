"""ReAct loop — interleaved reasoning and tool use, built from scratch.

The canonical agent loop: Claude reasons, emits a ``tool_use``, the harness
executes it and feeds the observation back, and the cycle repeats until Claude
stops calling tools (``stop_reason == 'end_turn'``). No framework — just the
Messages API, a step budget, and the shared tool round-trip.
"""
from __future__ import annotations

from ..core.agent import AgentResult, BaseAgent, StepBudgetExceeded

SYSTEM = (
    "You are an autonomous problem-solver. Work in small steps: reason about "
    "what you need, call a tool to get it, observe the result, and continue "
    "until you can answer. Call tools rather than guessing at facts or math. "
    "When you have the final answer, respond in plain text without calling a tool."
)


class ReActAgent(BaseAgent):
    """Reason → Act → Observe until the model is done."""

    def run(self, task: str) -> AgentResult:
        messages: list[dict] = [{"role": "user", "content": task}]
        self.log("goal", task)

        for step in range(self.max_steps):
            try:
                self._guard_budget(step)
            except StepBudgetExceeded:
                break

            response = self.client.complete(
                messages, system=SYSTEM, tools=self.tools.schemas()
            )
            if response.text:
                self.log("reason", response.text)

            # Model produced a final answer — no more tools requested.
            if response.stop_reason == "end_turn" and not response.tool_calls:
                return AgentResult(answer=response.text, steps=self.trace)

            # Echo the assistant turn (with its tool_use blocks) back, then the
            # tool results, so the conversation stays well-formed.
            messages.append({"role": "assistant", "content": response.raw.content})
            tool_results = self.run_tool_calls(response)
            messages.append({"role": "user", "content": tool_results})

        return AgentResult(
            answer="Stopped before reaching a final answer (step budget exhausted).",
            steps=self.trace,
            completed=False,
        )
