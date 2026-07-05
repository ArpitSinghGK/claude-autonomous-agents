"""Plan-and-Execute loop — decompose first, then act step by step.

A planner turn asks Claude for an explicit, ordered plan (structured JSON). A
separate executor then runs each step as its own short ReAct-style episode, so
long tasks stay on-rails and each sub-goal has a clear success boundary.
"""
from __future__ import annotations

import json

from ..core.agent import AgentResult, BaseAgent

PLANNER_SYSTEM = (
    "You are a planner. Break the user's task into a short ordered list of "
    "concrete, independently-checkable steps. Respond ONLY with the tool call."
)

EXECUTOR_SYSTEM = (
    "You are executing one step of a larger plan. Use tools when you need facts "
    "or computation. Return a concise result for THIS step only."
)

PLAN_TOOL = {
    "name": "emit_plan",
    "description": "Return the ordered plan for the task.",
    "input_schema": {
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Ordered, concrete steps.",
            }
        },
        "required": ["steps"],
    },
}


class PlanExecuteAgent(BaseAgent):
    """Planner decomposes the task; executor completes each step in turn."""

    def run(self, task: str) -> AgentResult:
        self.log("goal", task)
        plan = self._plan(task)
        self.log("plan", json.dumps(plan))

        results: list[str] = []
        for index, step in enumerate(plan, start=1):
            outcome = self._execute_step(task, step, results)
            self.log(f"step:{index}", f"{step} -> {outcome}")
            results.append(outcome)

        answer = self._synthesize(task, plan, results)
        return AgentResult(answer=answer, steps=self.trace)

    def _plan(self, task: str) -> list[str]:
        response = self.client.complete(
            [{"role": "user", "content": task}],
            system=PLANNER_SYSTEM,
            tools=[PLAN_TOOL],
        )
        for call in response.tool_calls:
            if call.name == "emit_plan":
                return [str(s) for s in call.input.get("steps", [])]
        # Fallback: treat the whole task as a single step.
        return [task]

    def _execute_step(self, task: str, step: str, prior: list[str]) -> str:
        context = "\n".join(f"- {r}" for r in prior) or "(none yet)"
        prompt = (
            f"Overall task: {task}\n\nResults so far:\n{context}\n\n"
            f"Now complete this step: {step}"
        )
        response = self.client.complete(
            [{"role": "user", "content": prompt}],
            system=EXECUTOR_SYSTEM,
            tools=self.tools.schemas(),
        )
        # One tool hop is enough for these bite-sized steps.
        if response.tool_calls:
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response.raw.content},
                {"role": "user", "content": self.run_tool_calls(response)},
            ]
            response = self.client.complete(
                messages, system=EXECUTOR_SYSTEM, tools=self.tools.schemas()
            )
        return response.text

    def _synthesize(self, task: str, plan: list[str], results: list[str]) -> str:
        joined = "\n".join(f"{i}. {s} => {r}" for i, (s, r) in enumerate(zip(plan, results), 1))
        prompt = (
            f"Task: {task}\n\nStep results:\n{joined}\n\n"
            "Write the final answer for the user."
        )
        response = self.client.complete([{"role": "user", "content": prompt}])
        return response.text
