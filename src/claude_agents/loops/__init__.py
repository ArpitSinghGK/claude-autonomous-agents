"""The five autonomous loop implementations."""

from .memory_agent import MemoryAgent
from .orchestrator import OrchestratorAgent, Specialist
from .plan_execute import PlanExecuteAgent
from .react import ReActAgent
from .reflexion import ReflexionAgent

# Registry used by the CLI to look up a loop by name.
LOOPS = {
    "react": ReActAgent,
    "plan-execute": PlanExecuteAgent,
    "reflexion": ReflexionAgent,
    "memory": MemoryAgent,
    "orchestrator": OrchestratorAgent,
}

__all__ = [
    "MemoryAgent",
    "OrchestratorAgent",
    "Specialist",
    "PlanExecuteAgent",
    "ReActAgent",
    "ReflexionAgent",
    "LOOPS",
]
