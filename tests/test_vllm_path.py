"""Router prefers vLLM Lab when VLLM_BASE_URL is reachable (ADR-022 Path B)."""

from __future__ import annotations

from unittest.mock import patch

from domainforge.config import Settings
from domainforge.eval.harness import SolutionId
from domainforge.generation.router import generate_triage


def test_generate_triage_uses_vllm_when_configured():
    settings = Settings(
        mock_llm=False,
        vllm_base_url="http://vllm.test",
        vllm_adapter_model="domainforge-triage-v0",
    )
    stub = (
        '{"intent":"password_reset","category":"account_access","priority":"medium",'
        '"entities":{},"suggested_action":"verify_identity","cite_faq_ids":[],"confidence":0.9}'
    )
    with (
        patch("domainforge.generation.router.vllm_available", return_value=True),
        patch("domainforge.generation.router.generate_with_vllm", return_value=stub) as mock_vllm,
        patch("domainforge.generation.router.ollama_available", return_value=False),
    ):
        out, backend = generate_triage(
            "I forgot my password",
            SolutionId.S3_PEFT_HYBRID,
            settings,
            retrieved=[],
        )
    assert backend == "vllm"
    assert "password_reset" in out
    mock_vllm.assert_called_once()
