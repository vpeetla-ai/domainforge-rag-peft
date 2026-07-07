"""GPU pipeline: merge, Ollama export, eval gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from domainforge.train.export import merge_adapter_to_hf, package_for_ollama, write_ollama_modelfile
from domainforge.train.pipeline import eval_gate_s4_vs_s3

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "data" / "eval_golden" / "sample.jsonl"
SMOKE_ADAPTER = ROOT / "adapters" / "domainforge-triage-dpo-smoke"


@pytest.mark.train
def test_merge_smoke_adapter(tmp_path):
    if not SMOKE_ADAPTER.exists():
        pytest.skip("run domainforge-train dpo --tiny first")
    pytest.importorskip("peft")
    out = tmp_path / "merged"
    result = merge_adapter_to_hf(SMOKE_ADAPTER, out, force_cpu=True)
    assert result["status"] == "merged"
    assert (out / "config.json").exists()
    assert (out / "merge_manifest.json").exists()


@pytest.mark.train
def test_write_ollama_modelfile(tmp_path):
    modelfile = write_ollama_modelfile(tmp_path / "fake-model", tmp_path / "test.Modelfile")
    text = modelfile.read_text(encoding="utf-8")
    assert "FROM" in text
    assert "SYSTEM" in text


@pytest.mark.train
def test_package_for_ollama_no_create(tmp_path):
    if not SMOKE_ADAPTER.exists():
        pytest.skip("run domainforge-train dpo --tiny first")
    pytest.importorskip("peft")
    result = package_for_ollama(
        SMOKE_ADAPTER,
        model_name="domainforge-test-smoke",
        merged_root=tmp_path / "merged",
        modelfile_root=tmp_path / "ollama",
        create_model=False,
        force_cpu=True,
    )
    assert result["status"] == "merged"
    assert Path(result["modelfile"]).exists()
    assert result["ollama"]["status"] == "skipped"


def test_eval_gate_returns_metrics():
    gate = eval_gate_s4_vs_s3(GOLDEN)
    assert "passed" in gate
    assert "win_rate_pct" in gate
    assert "s3" in gate and "s4" in gate


@pytest.mark.train
def test_tiny_pipeline_smoke(tmp_path):
    pytest.importorskip("transformers")
    from domainforge.train.pipeline import run_gpu_pipeline

    sft_out = tmp_path / "sft"
    dpo_out = tmp_path / "dpo"
    result = run_gpu_pipeline(
        train_file=ROOT / "data" / "sft_pairs" / "train.jsonl",
        val_file=ROOT / "data" / "sft_pairs" / "val.jsonl",
        prefs_train=ROOT / "data" / "preferences" / "train.jsonl",
        prefs_val=ROOT / "data" / "preferences" / "val.jsonl",
        golden_path=GOLDEN,
        sft_config=ROOT / "configs" / "train_qlora.yaml",
        dpo_config=ROOT / "configs" / "train_dpo.yaml",
        sft_output=sft_out,
        dpo_output=dpo_out,
        sft_steps=2,
        dpo_steps=2,
        skip_ollama_create=True,
        force_cpu=True,
    )
    assert result["status"] == "completed"
    assert result["sft"]["status"] == "trained"
    assert result["dpo"]["status"] == "trained"
    assert (tmp_path / "dpo" / "training_manifest.json").exists() or (dpo_out / "adapter_config.json").exists()
