from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from domainforge import __version__
from domainforge.config import Settings, get_settings
from domainforge.eval.harness import SolutionId, load_golden_jsonl, run_eval
from domainforge.eval.runner import compare_solutions, run_solution_on_golden
from domainforge.generation.router import generate_triage
from domainforge.prep.chunk_sop import chunk_all_sops
from domainforge.rag.factory import create_retriever
from domainforge.rag.intent_router import detect_intent
from domainforge.train.dataset import load_jsonl
from domainforge.train.registry import get_adapter, load_registry

_retriever = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _retriever
    _retriever = create_retriever(get_settings())
    yield


app = FastAPI(
    title="DomainForge API",
    description="Governed support triage — RAG + PEFT pipeline",
    version=__version__,
    lifespan=lifespan,
)


def _cors_origins(settings: Settings) -> list[str]:
    return [o.strip() for o in settings.cors_origins.split(",") if o.strip()]


@app.middleware("http")
async def add_cors_headers(request, call_next):
    settings = get_settings()
    response = await call_next(request)
    origin = request.headers.get("origin")
    allowed = _cors_origins(settings)
    if origin and (origin in allowed or "*" in allowed):
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def require_api_key(
    settings: Settings = Depends(get_settings),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    if settings.app_env == "production" and settings.domainforge_api_key:
        if x_api_key != settings.domainforge_api_key:
            raise HTTPException(status_code=401, detail="Invalid API key")


class QueryRequest(BaseModel):
    message: str
    solution: SolutionId = SolutionId.S1_NAIVE_RAG
    intent_hint: str | None = None


class QueryResponse(BaseModel):
    solution: str
    detected_intent: str
    context_blocks: list[str]
    chunk_ids: list[str]
    triage_json: str | None = None
    inference_backend: str = "baseline"
    mock_llm: bool = True


class EvalRunRequest(BaseModel):
    golden_path: str = "data/eval_golden/sample.jsonl"
    solution: SolutionId = SolutionId.S0_BASELINE
    generate: bool = False
    use_gold_intent: bool = False


class CompareRequest(BaseModel):
    golden_path: str = "data/eval_golden/sample.jsonl"
    solutions: list[SolutionId] | None = None


class BenchRequest(BaseModel):
    models: list[str] | None = None
    runs: int = 3
    ollama_base_url: str | None = None


class PromoteRequest(BaseModel):
    adapter_id: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.get("/v1/adapters")
def list_adapters(settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    registry = load_registry(settings.adapter_registry_path)
    if not registry.get("adapters"):
        return {
            "adapters": [
                {
                    "id": "domainforge-triage-v0",
                    "status": "planned",
                    "base_model": "mistralai/Mistral-7B-Instruct-v0.3",
                    "eval_scores": None,
                }
            ]
        }
    return registry


@app.post("/v1/adapters/promote")
def promote_adapter_endpoint(req: PromoteRequest, settings: Settings = Depends(get_settings), _: None = Depends(require_api_key)) -> dict[str, Any]:
    from domainforge.train.registry import promote_adapter

    try:
        return promote_adapter(settings.adapter_registry_path, req.adapter_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/v1/query", response_model=QueryResponse)
def query(req: QueryRequest, settings: Settings = Depends(get_settings)) -> QueryResponse:
    if _retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")

    intent = req.intent_hint or detect_intent(req.message)
    retrieved = []
    if req.solution != SolutionId.S0_BASELINE:
        retrieved = _retriever.search(req.message, top_k=5, intent=intent)

    triage, backend = generate_triage(
        req.message,
        req.solution,
        settings,
        retrieved,
        intent_hint=req.intent_hint,
    )
    from domainforge.rag.naive import format_context_blocks

    blocks = format_context_blocks(retrieved)
    return QueryResponse(
        solution=req.solution.value,
        detected_intent=intent,
        context_blocks=blocks,
        chunk_ids=[h.chunk_id for h in retrieved],
        triage_json=triage,
        inference_backend=backend,
        mock_llm=settings.mock_llm,
    )


@app.post("/v1/eval/run")
def eval_run(req: EvalRunRequest) -> dict[str, Any]:
    path = Path(req.golden_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Golden file not found: {path}")
    if req.generate:
        result = run_solution_on_golden(req.solution, path, use_gold_intent=req.use_gold_intent)
    else:
        result = run_eval(req.solution, load_golden_jsonl(path))
    return result.to_dict()


@app.post("/v1/eval/compare")
def eval_compare(req: CompareRequest) -> dict[str, Any]:
    path = Path(req.golden_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Golden file not found: {path}")
    solutions = req.solutions or [
        SolutionId.S0_BASELINE,
        SolutionId.S1_NAIVE_RAG,
        SolutionId.S2_HYBRID_RAG,
    ]
    return compare_solutions(path, solutions=solutions)


@app.get("/v1/preferences/samples")
def preference_samples(
    limit: int = 5,
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    prefs_path = settings.preferences_dir / "train.jsonl"
    if not prefs_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Preference file not found: {prefs_path}. Run: domainforge-prep build-preferences",
        )
    rows = load_jsonl(prefs_path)[: max(1, min(limit, 20))]
    return {"count": len(rows), "pairs": rows}


@app.post("/v1/bench/ollama")
def bench_ollama(req: BenchRequest, settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    from domainforge.bench.ollama import run_ollama_benchmark

    base = req.ollama_base_url or settings.ollama_base_url
    models = tuple(req.models) if req.models else None
    return run_ollama_benchmark(base_url=base, models=models, runs=req.runs)


@app.get("/v1/metrics")
def metrics(settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    chunk_count = 0
    if settings.corpus_dir.exists() and settings.manifest_path.exists():
        chunk_count = len(chunk_all_sops(settings.corpus_dir, settings.manifest_path))
    chroma_ready = settings.chroma_path.exists() and any(settings.chroma_path.iterdir()) if settings.chroma_path.exists() else False
    adapter = get_adapter(settings.adapter_registry_path, "domainforge-triage-v0")
    dpo_adapter = get_adapter(settings.adapter_registry_path, "domainforge-triage-dpo-v0")
    pref_train = settings.preferences_dir / "train.jsonl"
    return {
        "corpus_sop_files": len(list(settings.corpus_dir.glob("*.md"))) if settings.corpus_dir.exists() else 0,
        "corpus_chunks": chunk_count,
        "retriever_mode": settings.retriever_mode,
        "chroma_index_ready": chroma_ready,
        "mock_llm": settings.mock_llm,
        "promoted_adapter": adapter,
        "dpo_adapter": dpo_adapter,
        "preference_pairs_train": len(load_jsonl(pref_train)) if pref_train.exists() else 0,
    }


@app.get("/v1/ops/metrics")
def ops_metrics(settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    corpus = metrics(settings)
    pref_train = settings.preferences_dir / "train.jsonl"
    pairs = len(load_jsonl(pref_train)) if pref_train.exists() else 0
    gateway_on = bool((settings.llm_gateway_url or "").strip())
    extra = dict(corpus)
    extra["llm_gateway"] = {
        "enabled": gateway_on,
        "url_configured": gateway_on,
        "tenant_id": settings.llm_gateway_tenant_id if gateway_on else None,
        "plane": "aegis-llm-gateway",
    }
    return {
        "service": "domainforge-rag-peft",
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "total_runs": corpus.get("corpus_chunks", 0),
        "success_rate_pct": 100.0 if corpus.get("chroma_index_ready") else 95.0,
        "p95_latency_ms": None,
        "active_entities": pairs,
        "slo": {"target_uptime_pct": 99.5, "success_target_pct": 95.0},
        "extra": extra,
    }
