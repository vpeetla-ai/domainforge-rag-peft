"""Synthetic query trace for glass-box UI replay (post-response, not streaming)."""

from __future__ import annotations

from domainforge.eval.harness import SolutionId


def build_query_trace(
    solution: SolutionId,
    *,
    intent: str,
    chunk_count: int,
    backend: str,
) -> list[dict[str, object]]:
    """Ordered pipeline spans returned with POST /v1/query for demo replay."""

    events: list[dict[str, object]] = []

    def add(name: str, duration_ms: int, **attributes: object) -> None:
        events.append(
            {
                "name": name,
                "attributes": attributes,
                "duration_ms": duration_ms,
            }
        )

    add("domain.intent", 3, intent=intent)
    if solution != SolutionId.S0_BASELINE:
        add("domain.retrieve", 12, chunk_count=chunk_count)
    if solution in (
        SolutionId.S2_HYBRID_RAG,
        SolutionId.S3_PEFT_HYBRID,
        SolutionId.S4_DPO_PEFT,
    ):
        add("domain.hybrid_rank", 8)
    if solution in (SolutionId.S3_PEFT_HYBRID, SolutionId.S4_DPO_PEFT):
        add("domain.adapter_load", 6, adapter="domainforge-triage-v0")
    if solution == SolutionId.S4_DPO_PEFT:
        add("domain.dpo_adapter", 6, adapter="domainforge-triage-dpo-v0")
    add("domain.generate", 14, backend=backend)
    add("domain.format_validate", 4)
    return events
