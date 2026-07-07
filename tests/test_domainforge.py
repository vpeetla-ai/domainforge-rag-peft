import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from domainforge.eval.harness import SolutionId, EvalExample, run_eval
from domainforge.prep.chunk_sop import chunk_all_sops, cite_ids_for_intent, load_sop_map
from domainforge.prep.chatml import build_assistant_target, to_chatml_record
from domainforge.prep.intent_actions import intent_to_action
from domainforge.schemas.triage import TriageResponse
from domainforge.rag.naive import InMemoryRetriever

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus" / "sop_documents"
SOP_MAP = ROOT / "data" / "manifests" / "sop_intent_map.json"


@pytest.fixture
def sop_map() -> dict:
    return load_sop_map(SOP_MAP)


def test_sop_corpus_present():
    assert len(list(CORPUS.glob("*.md"))) == 13


def test_chunk_all_sops_produces_sections():
    chunks = chunk_all_sops(CORPUS, SOP_MAP)
    assert len(chunks) >= 40
    assert all("chunk_id" in c and "text" in c for c in chunks)


def test_cite_ids_for_track_order(sop_map):
    ids = cite_ids_for_intent(sop_map, "track_order")
    assert any("order_tracking" in cid for cid in ids)


def test_intent_to_action_mapping():
    assert intent_to_action("get_refund") == "initiate_refund"
    assert intent_to_action("unknown_intent") == "route_to_specialist"


def test_triage_schema_valid_json():
    target = build_assistant_target("track_order", "ORDER", "medium", load_sop_map(SOP_MAP))
    parsed = TriageResponse.model_validate(target)
    assert parsed.intent == "track_order"


def test_chatml_has_three_roles(sop_map):
    record = to_chatml_record("Where is my order?", "track_order", "ORDER", "medium", sop_map)
    roles = [m["role"] for m in record["messages"]]
    assert roles == ["system", "user", "assistant"]
    assistant = json.loads(record["messages"][2]["content"])
    assert assistant["suggested_action"] == "lookup_order_status"


def test_in_memory_retriever_finds_order_tracking():
    chunks = chunk_all_sops(CORPUS, SOP_MAP)
    retriever = InMemoryRetriever(chunks)
    hits = retriever.search("where is my order tracking number", intent="track_order")
    assert hits
    assert "order_tracking" in hits[0].chunk_id


def test_eval_harness_on_sample_golden():
    golden = ROOT / "data" / "eval_golden" / "sample.jsonl"
    from domainforge.eval.harness import load_golden_jsonl

    examples = load_golden_jsonl(golden)
    result = run_eval(SolutionId.S0_BASELINE, examples)
    assert result.n_examples == 5
    assert result.format_adherence_pct >= 80.0
    assert result.hallucination_freq_pct > 0  # adversarial row has invented chunk


def test_eval_example_intent_accuracy():
    good = EvalExample(
        instruction="refund",
        gold_intent="get_refund",
        prediction=json.dumps(build_assistant_target("get_refund", "REFUND", "medium", load_sop_map(SOP_MAP))),
    )
    result = run_eval(SolutionId.S3_PEFT_HYBRID, [good])
    assert result.intent_accuracy_pct == 100.0
    assert result.format_adherence_pct == 100.0


def test_api_health():
    from api.main import app

    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


def test_api_query_returns_chunks():
    from api.main import app

    with TestClient(app) as client:
        resp = client.post(
            "/v1/query",
            json={"message": "track my order please", "intent_hint": "track_order"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["chunk_ids"]


def test_api_eval_run():
    from api.main import app

    with TestClient(app) as client:
        resp = client.post(
            "/v1/eval/run",
            json={"golden_path": "data/eval_golden/sample.jsonl", "solution": "s0_baseline"},
        )
        assert resp.status_code == 200
        assert "format_adherence_pct" in resp.json()
