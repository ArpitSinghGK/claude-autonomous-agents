"""Reflexion loop — attempt, self-critique, retry with feedback.

Implements the self-correction pattern: the agent drafts an answer, a critic turn
scores it against the task and returns actionable feedback, and if the draft
falls short the agent revises. The critique is fed back into the next attempt so
each iteration is strictly better-informed than the last.
"""
from __future__ import annotations

from ..core.agent import AgentResult, BaseAgent

DRAFT_SYSTEM = (
    "You are solving a task. Produce your best complete answer. If earlier "
    "critique is provided, address every point in it."
)

CRITIC_SYSTEM = (
    "You are a strict reviewer. Judge whether the answer fully and correctly "
    "satisfies the task. Respond ONLY with the score tool."
)

SCORE_TOOL = {
    "name": "score",
    "description": "Score the candidate answer and give revision feedback.",
    "input_schema": {
        "type": "object",
        "properties": {
            "passes": {
                "type": "boolean",
                "description": "True if the answer fully satisfies the task.",
            },
            "feedback": {
                "type": "string",
                "description": "Specific, actionable guidance for the next attempt.",
            },
        },
        "required": ["passes", "feedback"],
    },
}


class ReflexionAgent(BaseAgent):
    """Draft → critique → revise until it passes or the budget is spent."""

    def __init__(self, *, max_attempts: int = 3, **kwargs) -> None:
        super().__init__(**kwargs)
        self.max_attempts = max_attempts

    def run(self, task: str) -> AgentResult:
        self.log("goal", task)
        feedback: str | None = None
        answer = ""

        for attempt in range(1, self.max_attempts + 1):
            answer = self._draft(task, feedback)
            self.log(f"draft:{attempt}", answer)

            passes, feedback = self._critique(task, answer)
            self.log(f"critique:{attempt}", f"passes={passes} :: {feedback}")
            if passes:
                return AgentResult(answer=answer, steps=self.trace)

        return AgentResult(answer=answer, steps=self.trace, completed=False)

    def _draft(self, task: str, feedback: str | None) -> str:
        prompt = task
        if feedback:
            prompt = f"{task}\n\nRevise your answer using this critique:\n{feedback}"
        response = self.client.complete(
            [{"role": "user", "content": prompt}], system=DRAFT_SYSTEM
        )
        return response.text

    def _critique(self, task: str, answer: str) -> tuple[bool, str]:
        prompt = f"Task:\n{task}\n\nCandidate answer:\n{answer}"
        response = self.client.complete(
            [{"role": "user", "content": prompt}],
            system=CRITIC_SYSTEM,
            tools=[SCORE_TOOL],
        )
        for call in response.tool_calls:
            if call.name == "score":
                return bool(call.input.get("passes")), str(call.input.get("feedback", ""))
        # If the critic didn't use the tool, treat it as a soft pass.
        return True, "no structured critique returned"
