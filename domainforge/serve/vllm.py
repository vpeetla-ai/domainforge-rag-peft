"""OpenAI-compatible client for vLLM Lab multi-LoRA educational endpoint (ADR-022 Path B)."""

from __future__ import annotations

import httpx

from domainforge.schemas.triage import TriageResponse
from domainforge.serve.ollama import build_messages, extract_json_object


def vllm_available(base_url: str) -> bool:
    try:
        with httpx.Client(timeout=3.0) as client:
            resp = client.get(f"{base_url.rstrip('/')}/health")
            return resp.status_code == 200
    except Exception:
        return False


def generate_with_vllm(
    user_content: str,
    base_url: str,
    model: str,
    temperature: float = 0.0,
    timeout: float = 60.0,
) -> str:
    payload = {
        "model": model,
        "messages": build_messages(user_content),
        "temperature": temperature,
        "max_tokens": 256,
    }
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(f"{base_url.rstrip('/')}/v1/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
    content = data["choices"][0]["message"]["content"]
    raw_json = extract_json_object(content)
    TriageResponse.model_validate_json(raw_json)
    return raw_json
