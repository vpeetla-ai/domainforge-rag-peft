from __future__ import annotations

from pathlib import Path
from typing import Any

from domainforge.config import Settings
from domainforge.rag.naive import InMemoryRetriever, RetrievedChunk


def build_chroma_index(
    chunks: list[dict[str, Any]],
    chroma_path: Path,
    collection_name: str = "sop_chunks",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> int:
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_path))
    embed_fn = SentenceTransformerEmbeddingFunction(model_name=embedding_model)

    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [c["chunk_id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [
        {
            "source_sop": c.get("source_sop", ""),
            "category": c.get("category", ""),
            "intent_tags": ",".join(c.get("intent_tags", [])),
            "section_title": c.get("section_title", ""),
        }
        for c in chunks
    ]
    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    return len(ids)


class ChromaRetriever:
    """S1 naive RAG — embedding similarity via ChromaDB."""

    def __init__(
        self,
        chroma_path: Path,
        collection_name: str = "sop_chunks",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        self._client = chromadb.PersistentClient(path=str(chroma_path))
        embed_fn = SentenceTransformerEmbeddingFunction(model_name=embedding_model)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=embed_fn,
        )

    def search(self, query: str, top_k: int = 5, intent: str | None = None) -> list[RetrievedChunk]:
        result = self._collection.query(query_texts=[query], n_results=min(top_k * 3, 20))
        chunks: list[RetrievedChunk] = []
        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        for chunk_id, text, dist, meta in zip(ids, docs, distances, metas, strict=False):
            if intent:
                tags = (meta or {}).get("intent_tags", "")
                if intent not in tags.split(","):
                    continue
            score = 1.0 - float(dist) if dist is not None else 0.0
            chunks.append(RetrievedChunk(chunk_id=chunk_id, text=text, score=score))
        chunks.sort(key=lambda c: c.score, reverse=True)
        return chunks[:top_k]
