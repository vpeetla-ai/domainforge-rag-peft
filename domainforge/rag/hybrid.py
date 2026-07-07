from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from domainforge.rag.naive import InMemoryRetriever, RetrievedChunk


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class HybridRetriever:
    """S2 — BM25-lite + lexical overlap hybrid over SOP chunks."""

    def __init__(self, chunks: list[dict[str, Any]], bm25_weight: float = 0.5) -> None:
        self._chunks = chunks
        self._bm25_weight = bm25_weight
        self._df: Counter[str] = Counter()
        self._doc_tokens: list[list[str]] = []
        for ch in chunks:
            tokens = _tokenize(ch.get("text", ""))
            self._doc_tokens.append(tokens)
            self._df.update(set(tokens))
        self._n_docs = len(chunks)

    def _bm25_score(self, query_tokens: list[str], doc_tokens: list[str]) -> float:
        if not query_tokens or not doc_tokens:
            return 0.0
        k1, b, avgdl = 1.5, 0.75, 120.0
        doc_len = len(doc_tokens)
        tf = Counter(doc_tokens)
        score = 0.0
        for term in query_tokens:
            if term not in tf:
                continue
            df = self._df.get(term, 0)
            idf = math.log(1 + (self._n_docs - df + 0.5) / (df + 0.5))
            freq = tf[term]
            denom = freq + k1 * (1 - b + b * doc_len / avgdl)
            score += idf * (freq * (k1 + 1)) / denom
        return score

    def search(self, query: str, top_k: int = 5, intent: str | None = None) -> list[RetrievedChunk]:
        query_tokens = _tokenize(query)
        lexical = InMemoryRetriever(self._chunks)
        dense_hits = {h.chunk_id: h.score for h in lexical.search(query, top_k=len(self._chunks), intent=intent)}

        scored: list[RetrievedChunk] = []
        for idx, ch in enumerate(self._chunks):
            if intent and intent not in ch.get("intent_tags", []):
                continue
            bm25 = self._bm25_score(query_tokens, self._doc_tokens[idx])
            dense = dense_hits.get(ch["chunk_id"], 0.0)
            # Normalize bm25 crudely
            bm25_norm = bm25 / (1 + bm25)
            dense_norm = dense / (1 + dense) if dense else 0.0
            score = self._bm25_weight * bm25_norm + (1 - self._bm25_weight) * dense_norm
            if score > 0:
                scored.append(RetrievedChunk(chunk_id=ch["chunk_id"], text=ch["text"], score=score))
        scored.sort(key=lambda c: c.score, reverse=True)
        return scored[:top_k]
