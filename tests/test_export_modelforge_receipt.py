"""Unit test for ModelForge PEFT receipt export (no GPU)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_export_modelforge_receipt_unverified(tmp_path):
    sft_manifest = tmp_path / "sft_manifest.json"
    dpo_manifest = tmp_path / "dpo_manifest.json"
    sft_manifest.write_text(
        json.dumps({"train_samples": 378, "val_samples": 27, "max_steps": 200, "wall_seconds": 827.42, "use_qlora": True})
    )
    dpo_manifest.write_text(
        json.dumps(
            {
                "train_pairs": 16,
                "val_pairs": 3,
                "max_steps": 100,
                "wall_seconds": 1021.94,
                "beta": 0.1,
                "adapter_path": "adapters/domainforge-triage-v0",
            }
        )
    )
    out = tmp_path / "peft_gpu.json"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "export_modelforge_receipt.py"),
        "--sft-manifest",
        str(sft_manifest),
        "--dpo-manifest",
        str(dpo_manifest),
        "--gpu",
        "1x A100-40GB",
        "--adapter-uri",
        "adapters/s4-dpo",
        "--allow-unverified",
        "--out",
        str(out),
    ]
    subprocess.check_call(cmd)
    blob = json.loads(out.read_text())
    assert blob["status"] == "unverified"
    assert blob["cuda"] is False
    assert blob["sft"]["train_examples"] == 378
    assert blob["sft"]["max_steps"] == 200
    assert blob["dpo"]["train_pairs"] == 16
    assert blob["dpo"]["max_steps"] == 100
    assert "known_gaps" in blob and blob["known_gaps"]


def test_export_peft_gpu_requires_cuda_flag(tmp_path):
    sft_manifest = tmp_path / "sft_manifest.json"
    dpo_manifest = tmp_path / "dpo_manifest.json"
    sft_manifest.write_text(json.dumps({"train_samples": 10, "max_steps": 5, "wall_seconds": 1.0}))
    dpo_manifest.write_text(json.dumps({"train_pairs": 2, "max_steps": 2, "wall_seconds": 1.0}))
    out = tmp_path / "peft_gpu.json"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "export_modelforge_receipt.py"),
        "--sft-manifest",
        str(sft_manifest),
        "--dpo-manifest",
        str(dpo_manifest),
        "--out",
        str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode != 0
    assert not out.exists()
