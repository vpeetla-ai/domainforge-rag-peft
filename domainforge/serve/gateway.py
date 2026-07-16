"""OpenAI-compatible client for aegis-llm-gateway (federated LLM plane, ADR-028/029)."""

from __future__ import annotations

import httpx

from domainforge.schemas.triage import TriageResponse
from domainforge.serve.ollama import build_messages, extract_json_object


def llm_gateway_enabled(base_url: str) -> bool:
    return bool((base_url or "").strip())


def _chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def generate_with_gateway(
    user_content: str,
    base_url: str,
    model: str,
    *,
    api_key: str = "",
    tenant_id: str = "domainforge-rag-peft",
    temperature: float = 0.0,
    timeout: float = 60.0,
    agent_role: str = "generator",
    thesis_role: str = "executor",
    data_class: str = "internal",
    selected_provider: str = "stub",
    model_tier: str = "specialized",
) -> str:
    """App selects model; plane enforces DataClass / verifier rules (ADR-029)."""
    # Confidential work must use private tier — prefer ollama provider label.
    if data_class == "confidential":
        selected_provider = "ollama"
        model_tier = "local_private"
        if not model.startswith("ollama/"):
            model = f"ollama/{model}" if "/" not in model else model
    payload = {
        "model": model,
        "messages": build_messages(user_content),
        "temperature": temperature,
        "max_tokens": 256,
    }
    headers = {
        "Authorization": f"Bearer {api_key or 'domainforge-gateway'}",
        "Content-Type": "application/json",
        "X-Tenant-Id": tenant_id or "domainforge-rag-peft",
        "X-Agent-Role": agent_role,
        "X-Thesis-Role": thesis_role,
        "X-Data-Class": data_class,
        "X-Selected-Provider": selected_provider,
        "X-Model-Tier": model_tier,
    }
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(_chat_completions_url(base_url), headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
    content = data["choices"][0]["message"]["content"]
    raw_json = extract_json_object(content)
    TriageResponse.model_validate_json(raw_json)
    return raw_json
