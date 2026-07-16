"""Record DomainForge triage outcomes to agent-finops (ADR-029)."""

from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

import httpx

from domainforge.config import Settings

logger = logging.getLogger(__name__)


def record_triage_outcome(
    *,
    settings: Settings,
    backend: str,
    ok: bool,
    data_class: str = "internal",
    workflow_id: str | None = None,
    total_cost_usd: float = 0.0,
) -> dict[str, Any] | None:
    base = (getattr(settings, "agentfinops_url", "") or "").strip()
    if not base:
        return None
    # confidential must not have used cloud gateway
    policy_deny = data_class == "confidential" and backend == "gateway"
    payload = {
        "workflow_id": workflow_id or str(uuid4()),
        "tenant_id": settings.llm_gateway_tenant_id or "domainforge-rag-peft",
        "eval_pass": ok and not policy_deny,
        "policy_deny": policy_deny,
        "hitl_required": False,
        "hitl_approved": True,
        "budget_ok": True,
        "total_cost_usd": float(total_cost_usd or 0),
    }
    headers = {"Content-Type": "application/json"}
    key = getattr(settings, "agentfinops_api_key", "") or ""
    if key:
        headers["X-API-Key"] = key
    try:
        with httpx.Client(timeout=5.0) as client:
            r = client.post(f"{base.rstrip('/')}/v1/outcomes", json=payload, headers=headers)
            r.raise_for_status()
            return r.json()
    except Exception as exc:  # noqa: BLE001
        logger.warning("finops_outcome_record_failed: %s", exc)
        return {"error": str(exc), "payload": payload}
