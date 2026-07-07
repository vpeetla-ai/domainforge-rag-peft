# DomainForge — Governed Support Triage Pipeline

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Fine-tune behavior, retrieve facts** — QLoRA SFT + DPO alignment for strict JSON triage + RAG over capstone SOP corpus, with a unified S0→S4 eval harness.

| Layer | Repo | Live |
|-------|------|------|
| Knowledge (RAG) | [enterprise_rag_platform](https://github.com/vpeetla-ai/enterprise_rag_platform) | [enterprise-rag-platform-eta.vercel.app](https://enterprise-rag-platform-eta.vercel.app) |
| **This project** | `domainforge-rag-peft` | [domainforge-rag-peft.vercel.app](https://domainforge-rag-peft.vercel.app) |
| Inference education | [vllm-architecture-lab](https://github.com/vpeetla-ai/vllm-architecture-lab) | [vllm-architecture-lab.vercel.app](https://vllm-architecture-lab.vercel.app) |

## Problem

Support automation needs **grounded citations** from SOPs and **reliable JSON** for routing — base models hallucinate field names and invent `chunk_id`s.

## Architecture (60s)

```mermaid
flowchart LR
  SOP[Capstone SOPs] --> RAG[Retrieve + cite]
  Bitext[Bitext SFT] --> PEFT[QLoRA adapter]
  RAG --> API[FastAPI /v1/query]
  PEFT --> API
  API --> EVAL[Ragas + golden metrics]
```

**Separation:** RAG = facts · PEFT = schema / intent / action codes ([ADR-001](docs/adr/ADR-001-rag-vs-peft-separation.md) · [ADR-019](https://github.com/vpeetla-ai/ai-architecture-portfolio/blob/main/adr/ADR-019-rag-facts-peft-behavior.md))

## Honest status

| Component | Status |
|-----------|--------|
| SOP corpus ingest + chunking | **Implemented** |
| Bitext → ChatML SFT prep | **Implemented** (CLI) |
| S2 hybrid BM25 + lexical RAG | **Implemented** (`RETRIEVER_MODE=hybrid`) |
| QLoRA training (TRL + PEFT) | **Implemented** (`domainforge-train`) |
| DPO preference tuning (S4) | **Implemented** (`domainforge-train dpo`) |
| Preference pair generator | **Implemented** (`domainforge-prep build-preferences`) |
| Adapter registry + promote API | **Implemented** |
| Ollama JSON inference | **Implemented** (`MOCK_LLM=false`) |
| Live API (Render) | **Live** — [domainforge-api.onrender.com](https://domainforge-api.onrender.com) |
| Live UI (Vercel) | **Live** — [domainforge-rag-peft.vercel.app](https://domainforge-rag-peft.vercel.app) |
| Full Mistral QLoRA + DPO on GPU | Requires CUDA — `scripts/gpu_pipeline.sh` |
| Ollama real inference (S3/S4) | **Implemented** — merge + `export-ollama` (`MOCK_LLM=false`) |
| vLLM production serve | Planned |

## Data

| Plane | Source | Files |
|-------|--------|-------|
| RAG | Capstone SOPs | 13 Markdown docs in `data/corpus/sop_documents/` |
| SFT | [Bitext HF dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) | `make fetch-bitext` |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
make chunk-sops
make test
make eval-compare
make api   # http://localhost:8090/health
```

**QLoRA training (CPU smoke / GPU production):**

```bash
pip install -e ".[train]"           # CPU smoke
pip install -e ".[train,train-gpu]" # CUDA QLoRA (bitsandbytes)
make train-dry
make train-tiny
make dpo-tiny
make pipeline-smoke                 # full S3→DPO orchestration on tiny-gpt2
```

**GPU → Ollama (real inference):** see [docs/GPU_OLLAMA_PIPELINE.md](docs/GPU_OLLAMA_PIPELINE.md)

```bash
bash scripts/gpu_pipeline.sh      # CUDA: SFT → DPO → merge → Modelfile
# Then: MOCK_LLM=false + OLLAMA_BASE_URL on API
```

**UI (Vercel / local):**

```bash
make api                # terminal 1
cd ui && NEXT_PUBLIC_API_URL=http://localhost:8090 npm run dev   # terminal 2
```

**Full SFT splits (requires network):**

```bash
pip install -e ".[dev,prep]"
make fetch-bitext
make manifest
```

## Solution ladder (eval)

| ID | Description |
|----|-------------|
| S0 | Base model, no retrieval |
| S1 | Naive RAG (Chroma + cosine) |
| S2 | Hybrid governed RAG |
| S3 | PEFT + S2 |
| S4 | DPO + S3 (preference-aligned) |

```bash
domainforge-prep build-preferences
domainforge-train dpo --tiny          # CPU smoke
domainforge-eval compare --golden data/eval_golden/sample.jsonl  # add S3/S4 via API
```

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness |
| GET | `/v1/adapters` | Adapter registry (stub) |
| POST | `/v1/query` | Retrieve SOP chunks for a message |
| POST | `/v1/eval/run` | Score golden set (`generate=true` for live S0/S1) |
| POST | `/v1/eval/compare` | S0–S4 delta table (+ preference win-rate for S3 vs S4) |
| GET | `/v1/preferences/samples` | DPO chosen vs rejected pairs for UI |
| GET | `/v1/metrics` | Corpus stats |

## Project layout

```
domainforge-rag-peft/
├── domainforge/     # prep, eval, rag, schemas
├── api/             # FastAPI
├── data/            # corpus, manifests, golden eval
├── docs/adr/
└── tests/
```

## Portfolio

Part of [vpeetla-ai](https://github.com/vpeetla-ai) governed stack · Case study: [domainforge-rag-peft.md](https://github.com/vpeetla-ai/ai-architecture-portfolio/blob/main/case-studies/domainforge-rag-peft.md) · Spec: [ENTERPRISE_RAG_PEFT_PIPELINE.md](https://github.com/vpeetla-ai/venkat-ai-portfolio/blob/main/docs/projects/ENTERPRISE_RAG_PEFT_PIPELINE.md)

## License

MIT
