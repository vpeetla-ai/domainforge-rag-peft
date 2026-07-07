"""Ollama / local inference benchmark helpers."""

from __future__ import annotations

import os
import time
from typing import Any

import httpx


DEFAULT_MODELS = ("llama3.2:3b", "mistral:7b")
DEFAULT_PROMPT = (
    'Return JSON only: {"category","urgency","summary","recommended_action","solution_id"} '
    "for: VPN keeps disconnecting after the OS update."
)


def bench_ollama_model(
    model: str,
    *,
    base_url: str,
    prompt: str = DEFAULT_PROMPT,
    runs: int = 3,
) -> dict[str, Any]:
    latencies: list[float] = []
    tokens = 0
    errors = 0
    for _ in range(max(1, runs)):
        start = time.perf_counter()
        try:
            with httpx.Client(timeout=120.0) as client:
                resp = client.post(
                    f"{base_url.rstrip('/')}/api/generate",
                    json={"model": model, "prompt": prompt, "stream": False},
                )
                resp.raise_for_status()
                data = resp.json()
                tokens += int(data.get("eval_count") or 0)
        except Exception:
            errors += 1
            continue
        latencies.append((time.perf_counter() - start) * 1000)

    total_s = sum(latencies) / 1000 if latencies else 0.0
    return {
        "model": model,
        "runs": runs,
        "ok_runs": len(latencies),
        "errors": errors,
        "p50_ms": round(sorted(latencies)[len(latencies) // 2], 2) if latencies else 0.0,
        "p95_ms": round(sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)], 2) if latencies else 0.0,
        "tokens_per_sec": round(tokens / total_s, 2) if total_s > 0 else 0.0,
    }


def run_ollama_benchmark(
    *,
    base_url: str | None = None,
    models: tuple[str, ...] | None = None,
    runs: int = 3,
) -> dict[str, Any]:
    url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    selected = models or DEFAULT_MODELS
    results = [bench_ollama_model(model, base_url=url, runs=runs) for model in selected]
    return {"ollama_base_url": url, "results": results}
