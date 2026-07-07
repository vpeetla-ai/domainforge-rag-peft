#!/usr/bin/env bash
# DomainForge GPU pipeline — S3 QLoRA → DPO S4 → Ollama export
# Requires: CUDA GPU, pip install -e ".[train,train-gpu]"
set -euo pipefail

cd "$(dirname "$0")/.."

python -c "import torch; assert torch.cuda.is_available(), 'CUDA required — see docs/GPU_OLLAMA_PIPELINE.md'"

pip install -e ".[train,train-gpu]" -q
domainforge-prep build-preferences

domainforge-train pipeline-gpu \
  --sft-config configs/train_qlora_gpu.yaml \
  --dpo-config configs/train_dpo_gpu.yaml \
  --sft-steps "${SFT_STEPS:-200}" \
  --dpo-steps "${DPO_STEPS:-100}" \
  "$@"

echo ""
echo "Next: ollama serve && set MOCK_LLM=false OLLAMA_BASE_URL=http://localhost:11434 on API"
