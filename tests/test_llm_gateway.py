"""Router prefers aegis-llm-gateway when LLM_GATEWAY_URL is set (ADR-028)."""

from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from api.main import app
from domainforge.config import Settings
from domainforge.eval.harness import SolutionId
from domainforge.generation.router import generate_triage
from domainforge.serve.gateway import llm_gateway_enabled


def test_llm_gateway_disabled_by_default():
    assert llm_gateway_enabled("") is False
    assert llm_gateway_enabled("   ") is False


def test_generate_triage_uses_gateway_when_configured():
    settings = Settings(
        mock_llm=False,
        llm_gateway_url="http://127.0.0.1:8100/v1",
        llm_gateway_tenant_id="domainforge-test",
        vllm_adapter_model="domainforge-triage-v0",
    )
    stub = (
        '{"intent":"password_reset","category":"account_access","priority":"medium",'
        '"entities":{},"suggested_action":"verify_identity","cite_faq_ids":[],"confidence":0.9}'
    )
    with (
        patch("domainforge.generation.router.generate_with_gateway", return_value=stub) as mock_gw,
        patch("domainforge.generation.router.vllm_available", return_value=False),
        patch("domainforge.generation.router.ollama_available", return_value=False),
    ):
        out, backend = generate_triage(
            "I forgot my password",
            SolutionId.S3_PEFT_HYBRID,
            settings,
            retrieved=[],
        )
    assert backend == "gateway"
    assert "password_reset" in out
    mock_gw.assert_called_once()
    assert mock_gw.call_args.kwargs["tenant_id"] == "domainforge-test"


def test_ops_metrics_shows_llm_gateway():
    client = TestClient(app)
    resp = client.get("/v1/ops/metrics")
    assert resp.status_code == 200
    gw = resp.json()["extra"]["llm_gateway"]
    assert gw["plane"] == "aegis-llm-gateway"
    assert "enabled" in gw
