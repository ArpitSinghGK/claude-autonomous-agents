"""Tests for the episodic and semantic memory stores."""
from __future__ import annotations

from claude_agents.core.memory import EpisodicMemory, SemanticMemory


def test_episodic_records_and_truncates_recent():
    mem = EpisodicMemory()
    for i in range(5):
        mem.record(f"event {i}")
    assert len(mem) == 5
    recent = mem.recent(limit=2)
    assert [r.content for r in recent] == ["event 3", "event 4"]


def test_semantic_recall_ranks_by_overlap():
    mem = SemanticMemory()
    mem.add("The capital of France is Paris.")
    mem.add("Python is a programming language.")
    mem.add("Paris hosted the 2024 Olympics.")

    hits = mem.recall("Tell me about Paris", top_k=2)
    assert len(hits) == 2
    # Both returned facts should mention Paris.
    assert all("paris" in r.content.lower() for r in hits)


def test_semantic_recall_empty_when_no_overlap():
    mem = SemanticMemory()
    mem.add("Completely unrelated fact about geology.")
    assert mem.recall("quantum computing") == []


def test_semantic_recall_handles_empty_store():
    assert SemanticMemory().recall("anything") == []
