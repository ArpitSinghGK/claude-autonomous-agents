"""Memory primitives shared by the memory-augmented and multi-agent loops.

Two complementary stores:

* :class:`EpisodicMemory` — an ordered log of what happened (turns, tool results).
* :class:`SemanticMemory` — recallable facts keyed by lightweight relevance.

The semantic store uses a transparent bag-of-words score so the repo runs with no
external dependency; the ``recall`` seam is where a real embedding model + vector
DB (pgvector, Pinecone, Qdrant) would drop in.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


@dataclass
class MemoryRecord:
    content: str
    kind: str = "note"
    created_at: str = field(default_factory=_now)


class EpisodicMemory:
    """Append-only trace of the agent's experience within a run."""

    def __init__(self) -> None:
        self._events: list[MemoryRecord] = []

    def record(self, content: str, kind: str = "event") -> None:
        self._events.append(MemoryRecord(content=content, kind=kind))

    def recent(self, limit: int = 10) -> list[MemoryRecord]:
        return self._events[-limit:]

    def __len__(self) -> int:
        return len(self._events)


class SemanticMemory:
    """Fact store with keyword-overlap recall (embedding-swap-ready)."""

    def __init__(self) -> None:
        self._records: list[MemoryRecord] = []

    def add(self, content: str, kind: str = "fact") -> None:
        self._records.append(MemoryRecord(content=content, kind=kind))

    def recall(self, query: str, top_k: int = 3) -> list[MemoryRecord]:
        """Return the ``top_k`` most relevant facts for ``query``.

        Replace this scoring with a vector similarity search to scale beyond a
        handful of records — the return contract stays the same.
        """
        query_tokens = Counter(_tokenize(query))
        if not query_tokens or not self._records:
            return []
        scored: list[tuple[float, MemoryRecord]] = []
        for record in self._records:
            record_tokens = Counter(_tokenize(record.content))
            overlap = sum((query_tokens & record_tokens).values())
            if overlap:
                scored.append((overlap, record))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [record for _, record in scored[:top_k]]

    def __len__(self) -> int:
        return len(self._records)
