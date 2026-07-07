from __future__ import annotations

import json
import re
from typing import Any

import httpx

from domainforge.prep.chatml import SYSTEM_PROMPT
from domainforge.schemas.triage import TriageResponse


def extract_json_object(text: str) -> str:
    """Pull first JSON object from model output."""
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    raise ValueError("No JSON object found in model output")


def build_messages(user_content: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def generate_with_ollama(
    user_content: str,
    base_url: str,
    model: str,
    temperature: float = 0.0,
    timeout: float = 120.0,
) -> str:
    payload = {
        "model": model,
        "messages": build_messages(user_content),
        "stream": False,
        "options": {"temperature": temperature},
        "format": "json",
    }
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(f"{base_url.rstrip('/')}/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
    content = data.get("message", {}).get("content", "")
    raw_json = extract_json_object(content)
    TriageResponse.model_validate_json(raw_json)
    return raw_json


def ollama_available(base_url: str) -> bool:
    try:
        with httpx.Client(timeout=3.0) as client:
            resp = client.get(f"{base_url.rstrip('/')}/api/tags")
            return resp.status_code == 200
    except Exception:
        return False
