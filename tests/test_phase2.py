import json
from pathlib import Path

import pytest

from domainforge.eval.harness import SolutionId
from domainforge.eval.runner import compare_solutions, run_solution_on_golden
from domainforge.generation.baseline import generate_triage_json
from domainforge.rag.intent_router import detect_intent

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "data" / "eval_golden" / "sample.jsonl"


def test_detect_intent_track_order():
    assert detect_intent("Where is my order? I need tracking") == "track_order"


def test_detect_intent_refund():
    assert detect_intent("I want a refund please") == "get_refund"


def test_s0_baseline_empty_citations():
    out = generate_triage_json(
        "I want a refund",
        SolutionId.S0_BASELINE,
        sop_map_path=ROOT / "data" / "manifests" / "sop_intent_map.json",
    )
    data = json.loads(out)
    assert data["cite_faq_ids"] == []


def test_run_solution_s1_on_golden():
    result = run_solution_on_golden(SolutionId.S1_NAIVE_RAG, GOLDEN)
    assert result.n_examples == 5
    assert result.format_adherence_pct == 100.0


def test_compare_solutions_produces_table(tmp_path):
    from domainforge.eval.harness import SolutionId

    table = compare_solutions(
        GOLDEN,
        solutions=[SolutionId.S0_BASELINE, SolutionId.S1_NAIVE_RAG],
        out_dir=tmp_path,
    )
    assert "s0_baseline" in table
    assert "s1_naive_rag" in table
    assert (tmp_path / "s0_baseline.json").exists()


def test_api_query_includes_triage_json():
    from fastapi.testclient import TestClient
    from api.main import app

    with TestClient(app) as client:
        resp = client.post(
            "/v1/query",
            json={"message": "track my order", "solution": "s1_naive_rag"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["triage_json"]
        assert body["detected_intent"] == "track_order"


def test_api_eval_compare():
    from fastapi.testclient import TestClient
    from api.main import app

    with TestClient(app) as client:
        resp = client.post("/v1/eval/compare", json={"golden_path": str(GOLDEN)})
        assert resp.status_code == 200
        assert "s0_baseline" in resp.json()
