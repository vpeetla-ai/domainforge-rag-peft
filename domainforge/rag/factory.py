from __future__ import annotations

from pathlib import Path

from domainforge.config import Settings
from domainforge.prep.chunk_sop import chunk_all_sops
from domainforge.rag.chroma_store import ChromaRetriever, build_chroma_index
from domainforge.rag.hybrid import HybridRetriever
from domainforge.rag.naive import InMemoryRetriever


def create_retriever(settings: Settings):
    mode = settings.retriever_mode
    if settings.corpus_dir.exists() and settings.manifest_path.exists():
        chunks = chunk_all_sops(settings.corpus_dir, settings.manifest_path)
    else:
        chunks = []

    if mode == "hybrid" and chunks:
        return HybridRetriever(chunks)

    if mode == "chroma" and settings.chroma_path.exists():
        try:
            return ChromaRetriever(settings.chroma_path, embedding_model=settings.embedding_model)
        except Exception:
            pass

    return InMemoryRetriever(chunks)


def index_corpus(settings: Settings) -> dict:
    chunks = chunk_all_sops(settings.corpus_dir, settings.manifest_path)
    count = build_chroma_index(
        chunks,
        settings.chroma_path,
        embedding_model=settings.embedding_model,
    )
    return {"chunks_indexed": count, "chroma_path": str(settings.chroma_path)}
