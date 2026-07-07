.PHONY: install install-dev test lint chunk-sops index-chroma fetch-bitext-sample manifest eval-sample eval-compare build-preferences api train-dry train-tiny dpo-dry dpo-tiny ui-build ui-dev

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-prep:
	pip install -e ".[dev,prep]"

install-rag:
	pip install -e ".[dev,rag]"

install-train:
	pip install -e ".[dev,train]"

install-all:
	pip install -e ".[dev,prep,rag,eval,train,api]"

test:
	pytest -q -m "not train"

test-all:
	pytest -q

lint:
	ruff check domainforge api tests

chunk-sops:
	domainforge-prep chunk-sops

index-chroma:
	domainforge-index

fetch-bitext-sample:
	domainforge-prep fetch-bitext --max-rows 500

fetch-bitext:
	domainforge-prep fetch-bitext

manifest:
	domainforge-prep build-manifests

eval-sample:
	domainforge-eval run --golden data/eval_golden/sample.jsonl --solution s0_baseline --out data/eval/results/s0_baseline.json

eval-compare:
	domainforge-eval compare --golden data/eval_golden/sample.jsonl

eval-compare-s34:
	domainforge-eval compare --golden data/eval_golden/sample.jsonl

build-preferences:
	domainforge-prep build-preferences

train-dry:
	domainforge-train dry-run

train-tiny:
	domainforge-train train --tiny --max-steps 3 --output-dir adapters/domainforge-triage-v0-smoke

dpo-dry:
	domainforge-train dpo-dry-run

dpo-tiny:
	domainforge-train dpo --tiny --max-steps 3 --output-dir adapters/domainforge-triage-dpo-v0-smoke

pipeline-smoke:
	domainforge-train pipeline-gpu --tiny-pipeline --skip-ollama-create

pipeline-gpu:
	bash scripts/gpu_pipeline.sh --skip-ollama-create

api:
	uvicorn api.main:app --reload --port 8090

ui-dev:
	cd ui && npm run dev

ui-build:
	cd ui && npm install && npm run build

eda-prep: chunk-sops manifest
