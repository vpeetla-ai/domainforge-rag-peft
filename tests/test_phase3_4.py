import json
from pathlib import Path

import pytest

from domainforge.rag.hybrid import HybridRetriever
from domainforge.prep.chunk_sop import chunk_all_sops
from domainforge.serve.ollama import extract_json_object
from domainforge.train.registry import load_registry, promote_adapter, register_adapter

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus" / "sop_documents"
SOP_MAP = ROOT / "data" / "manifests" / "sop_intent_map.json"


def test_hybrid_retriever_finds_refund_chunk():
    chunks = chunk_all_sops(CORPUS, SOP_MAP)
    retriever = HybridRetriever(chunks)
    hits = retriever.search("I need a refund for my order", intent="get_refund")
    assert hits
    assert any("refund" in h.chunk_id for h in hits)


def test_extract_json_object_from_fenced_output():
    raw = 'Here is JSON:\n```json\n{"intent":"track_order","category":"ORDER"}\n```'
    extracted = extract_json_object(raw)
    data = json.loads(extracted)
    assert data["intent"] == "track_order"


def test_adapter_registry_promote(tmp_path):
    reg = tmp_path / "registry.json"
    register_adapter(reg, "a1", tmp_path / "a1", "base-model", status="candidate")
    register_adapter(reg, "a2", tmp_path / "a2", "base-model", status="candidate")
    promoted = promote_adapter(reg, "a1")
    assert promoted["status"] == "promoted"
    registry = load_registry(reg)
    statuses = {a["id"]: a["status"] for a in registry["adapters"]}
    assert statuses["a1"] == "promoted"
    assert statuses["a2"] == "candidate"


@pytest.mark.train
def test_train_dry_run():
    pytest.importorskip("transformers")
    from domainforge.train.qlora import dry_run_training

    result = dry_run_training(
        ROOT / "data" / "sft_pairs" / "train.jsonl",
        ROOT / "configs" / "train_qlora.yaml",
        base_model="sshleifer/tiny-gpt2",
        max_samples=2,
    )
    assert result["status"] == "ok"
    assert result["samples"] >= 1
