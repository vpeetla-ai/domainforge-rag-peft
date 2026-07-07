# Agent Instructions — domainforge-rag-peft

Read [CONTEXT.md](https://github.com/vpeetla-ai/ai-content-factory/blob/main/CONTEXT.md) for org vocabulary.

## Stack layer

**Knowledge + MLOps** — RAG corpus (SOPs) + PEFT (QLoRA) + eval harness. Complements `enterprise_rag_platform` (production hybrid RAG) and `vllm-architecture-lab` (serving).

## Conventions

- Python 3.11+, FastAPI, Pydantic v2
- `pip install -e ".[dev]"` + `pytest -q` before claiming done
- RAG holds facts; PEFT holds JSON format ([ADR-001](docs/adr/ADR-001-rag-vs-peft-separation.md))
- Data: capstone SOPs in `data/corpus/` + Bitext via `make fetch-bitext`

## Phases

| Phase | Status |
|-------|--------|
| 0–2 Prep, eval, Chroma, compare | Done |
| 3 QLoRA TRL + registry + smoke train | Done |
| 4 Ollama + hybrid S2 + UI | Done |
| GPU Mistral QLoRA + vLLM prod | Planned |
