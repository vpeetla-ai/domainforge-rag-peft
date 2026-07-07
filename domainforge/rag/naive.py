from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float


class InMemoryRetriever:
    """S1 naive retrieval for tests and local dev without Chroma."""

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self._chunks = chunks

    def search(self, query: str, top_k: int = 3, intent: str | None = None) -> list[RetrievedChunk]:
        query_l = query.lower()
        scored: list[RetrievedChunk] = []
        for ch in self._chunks:
            if intent and intent not in ch.get("intent_tags", []):
                continue
            text = ch.get("text", "")
            score = sum(1 for token in query_l.split() if token in text.lower())
            if score > 0:
                scored.append(RetrievedChunk(chunk_id=ch["chunk_id"], text=text, score=float(score)))
        scored.sort(key=lambda c: c.score, reverse=True)
        return scored[:top_k]


def format_context_blocks(chunks: list[RetrievedChunk]) -> list[str]:
    return [f'<context cite="{c.chunk_id}">{c.text}</context>' for c in chunks]
