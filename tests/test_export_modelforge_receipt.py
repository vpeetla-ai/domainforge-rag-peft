"""Unit test for ModelForge PEFT receipt export (no GPU)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_export_modelforge_receipt_unverified(tmp_path):
    s0 = tmp_path / "s0.json"
    s3 = tmp_path / "s3.json"
    s4 = tmp_path / "s4.json"
    for path, rate in [(s0, 0.4), (s3, 0.7), (s4, 0.85)]:
        path.write_text(json.dumps({"schema_pass": rate, "preference_win_rate": 0.62}))
    out = tmp_path / "peft_gpu.json"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "export_modelforge_receipt.py"),
        "--s0",
        str(s0),
        "--s3",
        str(s3),
        "--s4",
        str(s4),
        "--gpu",
        "1x A100-40GB",
        "--sft-examples",
        "128",
        "--dpo-pairs",
        "64",
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
    assert blob["metrics"]["S0_schema_pass"] == 0.4
    assert blob["metrics"]["S4_schema_pass"] == 0.85


def test_export_peft_gpu_requires_cuda_flag(tmp_path):
    s0 = tmp_path / "s0.json"
    s3 = tmp_path / "s3.json"
    s4 = tmp_path / "s4.json"
    for path in (s0, s3, s4):
        path.write_text(json.dumps({"schema_pass": 0.5}))
    out = tmp_path / "peft_gpu.json"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "export_modelforge_receipt.py"),
        "--s0",
        str(s0),
        "--s3",
        str(s3),
        "--s4",
        str(s4),
        "--out",
        str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode != 0
    assert not out.exists()
