"""Multi-agent orchestrator — a supervisor delegating to specialist sub-agents.

A coordinator turn routes the task (or its parts) to named specialists, each of
which is itself an autonomous ReAct loop with its own system prompt and tools.
The coordinator then synthesizes the specialists' outputs into one answer. This
is the \"custom orchestration\" layer the loops below plug into — swap a specialist
for a different loop type and the contract holds.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..core.agent import AgentResult, BaseAgent
from ..core.claude_client import ClaudeClient
from ..core.tools import ToolRegistry, default_registry
from .react import ReActAgent


@dataclass
class Specialist:
    """A named sub-agent the coordinator can delegate to."""

    name: str
    description: str
    system: str


DEFAULT_SPECIALISTS = [
    Specialist(
        name="researcher",
        description="Finds and summarizes factual information.",
        system="You are a meticulous researcher. Gather facts with tools and "
        "report them plainly, citing what each fact came from.",
    ),
    Specialist(
        name="analyst",
        description="Performs calculations and quantitative reasoning.",
        system="You are a quantitative analyst. Use the calculator for every "
        "arithmetic step and show the numbers you used.",
    ),
]

ROUTER_SYSTEM = (
    "You are a coordinator managing specialist agents. Decide which specialists "
    "should handle the task and what sub-task to give each. Respond ONLY with "
    "the delegate tool."
)


def _delegate_tool(specialists: list[Specialist]) -> dict:
    names = [s.name for s in specialists]
    return {
        "name": "delegate",
        "description": "Assign sub-tasks to specialists. "
        + "; ".join(f"{s.name}: {s.description}" for s in specialists),
        "input_schema": {
            "type": "object",
            "properties": {
                "assignments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "specialist": {"type": "string", "enum": names},
                            "subtask": {"type": "string"},
                        },
                        "required": ["specialist", "subtask"],
                    },
                }
            },
            "required": ["assignments"],
        },
    }


class OrchestratorAgent(BaseAgent):
    """Coordinator that fans work out to specialist ReAct loops and merges results."""

    def __init__(
        self,
        *,
        specialists: list[Specialist] | None = None,
        client: ClaudeClient | None = None,
        tools: ToolRegistry | None = None,
        **kwargs,
    ) -> None:
        super().__init__(client=client, tools=tools or default_registry(), **kwargs)
        self.specialists = {s.name: s for s in (specialists or DEFAULT_SPECIALISTS)}

    def run(self, task: str) -> AgentResult:
        self.log("goal", task)
        assignments = self._route(task)
        self.log("route", "; ".join(f"{a['specialist']}<-{a['subtask']}" for a in assignments))

        outputs: list[str] = []
        for assignment in assignments:
            name = assignment["specialist"]
            spec = self.specialists.get(name)
            if spec is None:
                continue
            result = self._run_specialist(spec, assignment["subtask"])
            self.log(f"specialist:{name}", result.answer)
            outputs.append(f"[{name}] {result.answer}")

        answer = self._synthesize(task, outputs)
        return AgentResult(answer=answer, steps=self.trace)

    def _route(self, task: str) -> list[dict]:
        response = self.client.complete(
            [{"role": "user", "content": task}],
            system=ROUTER_SYSTEM,
            tools=[_delegate_tool(list(self.specialists.values()))],
        )
        for call in response.tool_calls:
            if call.name == "delegate":
                return list(call.input.get("assignments", []))
        # Fallback: hand the whole task to the first specialist.
        first = next(iter(self.specialists))
        return [{"specialist": first, "subtask": task}]

    def _run_specialist(self, spec: Specialist, subtask: str) -> AgentResult:
        agent = ReActAgent(
            client=self.client, tools=self.tools, max_steps=self.max_steps
        )
        # Give the specialist its own persona by prefixing its system prompt.
        agent_task = f"{spec.system}\n\nSub-task: {subtask}"
        return agent.run(agent_task)

    def _synthesize(self, task: str, outputs: list[str]) -> str:
        joined = "\n".join(outputs) or "(no specialist output)"
        prompt = (
            f"Task: {task}\n\nSpecialist outputs:\n{joined}\n\n"
            "Combine these into a single coherent answer for the user."
        )
        response = self.client.complete([{"role": "user", "content": prompt}])
        return response.text
