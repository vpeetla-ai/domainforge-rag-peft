#!/usr/bin/env bash
# Ollama latency benchmark for DomainForge structured JSON triage prompts.
set -euo pipefail
API_URL="${DOMAINFORGE_API_URL:-http://localhost:8090}"
RUNS="${BENCH_RUNS:-3}"
MODELS="${BENCH_MODELS:-llama3.2:3b,mistral:7b}"

curl -s -X POST "${API_URL}/v1/bench/ollama" \
  -H 'Content-Type: application/json' \
  -d "{\"runs\": ${RUNS}, \"models\": [\"${MODELS//,/\",\"}\"]}" | python3 -m json.tool
