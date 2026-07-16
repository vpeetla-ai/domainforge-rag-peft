from domainforge.eval.harness import SolutionId
from domainforge.query_trace import build_query_trace


def test_s0_trace_skips_retrieve():
    trace = build_query_trace(SolutionId.S0_BASELINE, intent="track_order", chunk_count=0, backend="baseline")
    names = [e["name"] for e in trace]
    assert names == [
        "domain.intent",
        "domain.generate",
        "domain.format_validate",
    ]


def test_s4_trace_includes_dpo():
    trace = build_query_trace(
        SolutionId.S4_DPO_PEFT,
        intent="track_order",
        chunk_count=3,
        backend="mock",
    )
    names = [e["name"] for e in trace]
    assert "domain.dpo_adapter" in names
    assert "domain.retrieve" in names
    assert names.index("domain.adapter_load") < names.index("domain.dpo_adapter")
