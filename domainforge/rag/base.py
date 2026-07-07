from __future__ import annotations

from typing import Protocol

from domainforge.rag.naive import RetrievedChunk


class Retriever(Protocol):
    def search(self, query: str, top_k: int = 3, intent: str | None = None) -> list[RetrievedChunk]: ...
