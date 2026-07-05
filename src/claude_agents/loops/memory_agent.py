"""Memory-augmented agent — recall relevant facts, act, then persist learnings.

Wraps a ReAct-style loop with two memory stores. Before reasoning it recalls
semantically-relevant facts and injects them into the prompt; after finishing it
writes a distilled learning back, so knowledge compounds across turns and
sessions instead of being lost when the context window rolls over.
"""
from __future__ import annotations

from ..core.agent import AgentResult, BaseAgent
from ..core.memory import EpisodicMemory, SemanticMemory

SYSTEM = (
    "You are an assistant with long-term memory. Relevant remembered facts are "
    "provided below when available — use them. Answer the user's request "
    "directly; call tools when you need facts or computation."
)


class MemoryAgent(BaseAgent):
    """A loop that reads from and writes to persistent memory each turn."""

    def __init__(
        self,
        *,
        semantic: SemanticMemory | None = None,
        episodic: EpisodicMemory | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.semantic = semantic or SemanticMemory()
        self.episodic = episodic or EpisodicMemory()

    def run(self, task: str) -> AgentResult:
        self.log("goal", task)
        recalled = self.semantic.recall(task)
        if recalled:
            self.log("recall", " | ".join(r.content for r in recalled))

        memory_context = (
            "Relevant memory:\n" + "\n".join(f"- {r.content}" for r in recalled)
            if recalled
            else "Relevant memory: (none)"
        )
        messages = [{"role": "user", "content": f"{memory_context}\n\nTask: {task}"}]

        answer = ""
        for step in range(self.max_steps):
            response = self.client.complete(
                messages, system=SYSTEM, tools=self.tools.schemas()
            )
            if response.text:
                answer = response.text
                self.log(f"reason:{step}", response.text)

            if response.stop_reason == "end_turn" and not response.tool_calls:
                break

            messages.append({"role": "assistant", "content": response.raw.content})
            messages.append({"role": "user", "content": self.run_tool_calls(response)})

        # Persist what happened so future runs can recall it.
        self.episodic.record(f"task={task} answer={answer}", kind="turn")
        self.remember(task, answer)
        return AgentResult(answer=answer, steps=self.trace)

    def remember(self, task: str, answer: str) -> None:
        """Distill and store a learning. A production build would summarize with
        Claude; here we store a compact task→answer fact."""
        self.semantic.add(f"When asked '{task}', the answer was: {answer}")
        self.log("store", f"persisted learning ({len(self.semantic)} facts total)")
