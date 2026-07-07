# GPU Training → Ollama Inference Pipeline

Train **S3 (QLoRA SFT)** and **S4 (DPO)** on Mistral-7B, merge adapters, and serve via Ollama for real `MOCK_LLM=false` inference.

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| **CUDA GPU** | 24GB+ VRAM recommended (A10, A100, RTX 4090) |
| **Python 3.11** | `pip install -e ".[train,train-gpu]"` |
| **Ollama** | [ollama.com](https://ollama.com) on GPU host for serving |
| **HuggingFace** | Access to `mistralai/Mistral-7B-Instruct-v0.3` |

CPU-only machines: use `--tiny-pipeline` for smoke validation, then run the real pipeline on RunPod/Colab.

## One-command pipeline (CUDA)

```bash
pip install -e ".[dev,train,train-gpu]"
make build-preferences

# Full S3 → DPO → eval gate → merge → Ollama Modelfile
domainforge-train pipeline-gpu \
  --sft-steps 200 \
  --dpo-steps 100 \
  --skip-ollama-create   # remove flag when ollama CLI is on same machine
```

Outputs:

- `adapters/domainforge-triage-v0/` — SFT LoRA
- `adapters/domainforge-triage-dpo-v0/` — DPO LoRA (continues from SFT)
- `adapters/merged/domainforge-triage/` — merged HF weights (SFT)
- `adapters/merged/domainforge-triage-dpo/` — merged HF weights (DPO)
- `adapters/ollama/*.Modelfile` — Ollama create recipes
- `data/train_reports/*_pipeline.json` — eval gate + metrics

## Step-by-step

### 1. SFT (S3)

```bash
domainforge-train train \
  --config configs/train_qlora_gpu.yaml \
  --output-dir adapters/domainforge-triage-v0 \
  --max-steps 200
```

### 2. DPO (S4)

```bash
domainforge-prep build-preferences
domainforge-train dpo \
  --config configs/train_dpo_gpu.yaml \
  --adapter-path adapters/domainforge-triage-v0 \
  --output-dir adapters/domainforge-triage-dpo-v0 \
  --max-steps 100
```

### 3. Eval gate

```bash
domainforge-eval compare --golden data/eval_golden/sample.jsonl
# API: POST /v1/eval/compare with solutions ["s3_peft_hybrid","s4_dpo_peft"]
```

Promotion requires `preference_win_rate_pct` improvement and no format regression.

### 4. Merge + Ollama

```bash
# SFT model for solution s3_peft_hybrid
domainforge-train export-ollama \
  --adapter-dir adapters/domainforge-triage-v0 \
  --model-name domainforge-triage

# DPO model for solution s4_dpo_peft
domainforge-train export-ollama \
  --adapter-dir adapters/domainforge-triage-dpo-v0 \
  --model-name domainforge-triage-dpo
```

Creates Ollama models:

```bash
ollama create domainforge-triage -f adapters/ollama/domainforge-triage.Modelfile
ollama create domainforge-triage-dpo -f adapters/ollama/domainforge-triage-dpo.Modelfile
```

### 5. Wire API to Ollama

On the **GPU host** (or tunneled):

```bash
ollama serve
ollama list   # domainforge-triage, domainforge-triage-dpo
```

On **Render API** (or local `.env`):

```env
MOCK_LLM=false
OLLAMA_BASE_URL=http://<gpu-host>:11434
OLLAMA_ADAPTER_MODEL=domainforge-triage
OLLAMA_DPO_ADAPTER_MODEL=domainforge-triage-dpo
```

Redeploy API. Query with `solution=s3_peft_hybrid` or `solution=s4_dpo_peft` — `inference_backend` should return `ollama`.

## RunPod quick start

```bash
# On RunPod PyTorch pod (CUDA)
git clone https://github.com/vpeetla-ai/domainforge-rag-peft.git
cd domainforge-rag-peft
pip install -e ".[train,train-gpu]"
curl -fsSL https://ollama.com/install.sh | sh

domainforge-train pipeline-gpu --sft-steps 200 --dpo-steps 100
```

Expose Ollama port `11434` or use RunPod TCP proxy. Point Render `OLLAMA_BASE_URL` at the proxy URL.

## CPU smoke (no GPU)

```bash
domainforge-train pipeline-gpu --tiny-pipeline
```

Uses `tiny-gpt2`, 3 steps each — validates the full orchestration path locally.

## Honest limits

- Render free tier cannot run Mistral training or Ollama — train on GPU pod, serve Ollama on same pod or dedicated inference VM
- Merged 7B model is ~14GB disk — do not commit to git; upload to HF private repo or S3 for team use
- First `ollama create` from HF weights may take 10–20 minutes

## Related

- [ADR-002](adr/ADR-002-dpo-after-sft.md) — DPO after SFT
- [DEPLOY.md](DEPLOY.md) — Render + Vercel wiring
