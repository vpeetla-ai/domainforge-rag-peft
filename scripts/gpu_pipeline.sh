#!/usr/bin/env bash
# DomainForge GPU pipeline — S3 QLoRA → DPO S4 → eval
# Requires: CUDA GPU, pip install -e ".[train,train-gpu]"
#
# SFT and DPO run as separate `domainforge-train` process invocations rather
# than the combined `pipeline-gpu` command. A real 22GB-VRAM run showed DPO
# OOMing on startup with the SFT model's memory still fully resident even
# after an explicit gc.collect()/torch.cuda.empty_cache() between the two
# stages in-process (see domainforge/train/pipeline.py) -- something below
# Python's refcounting (likely the DPO reference-model copy landing on top
# of SFT's un-released allocator blocks) was still holding it. A fresh OS
# process for each GPU-heavy stage sidesteps that class of bug entirely: a
# new process gets its own CUDA context, guaranteed clean regardless of what
# any library leaked. `pipeline-gpu` is left in place in cli.py/pipeline.py
# for the CPU/--tiny-pipeline smoke path, where memory headroom was never
# the issue.
#
# Ollama export is intentionally not run here -- see docs/GPU_OLLAMA_PIPELINE.md
# "4. Merge + Ollama" for that as an explicit follow-up step once you've
# confirmed the eval gate passed.
set -euo pipefail

cd "$(dirname "$0")/.."

python -c "import torch; assert torch.cuda.is_available(), 'CUDA required — see docs/GPU_OLLAMA_PIPELINE.md'"

pip install -e ".[train,train-gpu]" -q
domainforge-prep build-preferences

SFT_OUTPUT="${SFT_OUTPUT:-adapters/domainforge-triage-v0}"
DPO_OUTPUT="${DPO_OUTPUT:-adapters/domainforge-triage-dpo-v0}"
GOLDEN="${GOLDEN:-data/eval_golden/sample.jsonl}"

domainforge-train train \
  --config configs/train_qlora_gpu.yaml \
  --output-dir "$SFT_OUTPUT" \
  --max-steps "${SFT_STEPS:-200}"

domainforge-train dpo \
  --config configs/train_dpo_gpu.yaml \
  --adapter-path "$SFT_OUTPUT" \
  --output-dir "$DPO_OUTPUT" \
  --max-steps "${DPO_STEPS:-100}"

domainforge-eval compare --golden "$GOLDEN"

echo ""
echo "Eval results: data/eval/results/{s0_baseline,s3_peft_hybrid,s4_dpo_peft}.json"
echo "Next (optional): domainforge-train export-ollama --adapter-dir $DPO_OUTPUT --model-name domainforge-triage-dpo"
echo "Then: ollama serve && set MOCK_LLM=false OLLAMA_BASE_URL=http://localhost:11434 on API"
