from __future__ import annotations

from domainforge.config import Settings
from domainforge.eval.harness import SolutionId
from domainforge.generation.baseline import generate_triage_json
from domainforge.rag.naive import RetrievedChunk, format_context_blocks
from domainforge.serve.gateway import generate_with_gateway, llm_gateway_enabled
from domainforge.serve.ollama import generate_with_ollama, ollama_available
from domainforge.serve.vllm import generate_with_vllm, vllm_available


def build_user_prompt(message: str, context_blocks: list[str]) -> str:
    parts = context_blocks + [f"Customer message: {message}"]
    return "\n\n".join(parts)


def generate_triage(
    message: str,
    solution: SolutionId,
    settings: Settings,
    retrieved: list[RetrievedChunk],
    intent_hint: str | None = None,
) -> tuple[str, str]:
    """
    Returns (triage_json, backend) where backend is gateway|vllm|ollama|baseline.
    """
    blocks = format_context_blocks(retrieved)
    user_prompt = build_user_prompt(message, blocks)

    peft_solutions = (
        SolutionId.S2_HYBRID_RAG,
        SolutionId.S3_PEFT_HYBRID,
        SolutionId.S4_DPO_PEFT,
    )

    # Federated LLM plane (ADR-028) — optional; falls through on failure.
    if (
        not settings.mock_llm
        and solution in peft_solutions
        and llm_gateway_enabled(settings.llm_gateway_url)
    ):
        try:
            if solution == SolutionId.S4_DPO_PEFT:
                model = settings.vllm_dpo_adapter_model
            else:
                model = settings.vllm_adapter_model
            triage = generate_with_gateway(
                user_prompt,
                settings.llm_gateway_url,
                model,
                api_key=settings.llm_gateway_api_key,
                tenant_id=settings.llm_gateway_tenant_id,
            )
            return triage, "gateway"
        except Exception:
            pass

    # Path B (ADR-022): educational multi-LoRA via vLLM Lab when VLLM_BASE_URL is set.
    if (
        not settings.mock_llm
        and solution in peft_solutions
        and settings.vllm_base_url
        and vllm_available(settings.vllm_base_url)
    ):
        try:
            if solution == SolutionId.S4_DPO_PEFT:
                model = settings.vllm_dpo_adapter_model
            elif solution == SolutionId.S3_PEFT_HYBRID:
                model = settings.vllm_adapter_model
            else:
                model = settings.vllm_adapter_model
            triage = generate_with_vllm(user_prompt, settings.vllm_base_url, model)
            return triage, "vllm"
        except Exception:
            pass

    if (
        not settings.mock_llm
        and solution in peft_solutions
        and ollama_available(settings.ollama_base_url)
    ):
        try:
            if solution == SolutionId.S4_DPO_PEFT:
                model = settings.ollama_dpo_adapter_model
            elif solution == SolutionId.S3_PEFT_HYBRID:
                model = settings.ollama_adapter_model
            else:
                model = settings.ollama_model
            triage = generate_with_ollama(user_prompt, settings.ollama_base_url, model)
            return triage, "ollama"
        except Exception:
            pass

    triage = generate_triage_json(
        message,
        solution,
        sop_map_path=settings.manifest_path,
        retrieved=retrieved,
        gold_intent=intent_hint,
    )
    return triage, "baseline"
