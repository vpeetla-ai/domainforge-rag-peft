#!/usr/bin/env python3
"""Export a ModelForge-compatible PEFT receipt from DomainForge eval outputs.

Usage (from domainforge-rag-peft root, after GPU pipeline + eval-compare):

  python scripts/export_modelforge_receipt.py \\
    --s0 data/eval/results/s0_baseline.json \\
    --s3 data/eval/results/s3_peft_hybrid.json \\
    --s4 data/eval/results/s4_dpo_peft.json \\
    --adapter-uri adapters/s4-dpo \\
    --require-cuda \\
    --out /path/to/modelforge-llmops/docs/receipts/peft_gpu.json

Honesty:
  Writing peft_gpu.json requires --require-cuda (live CUDA) or --allow-unverified (tests only).
  Tiny smoke bases (e.g. sshleifer/tiny-gpt2) are rejected for peft_gpu unless --allow-unverified.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TINY_MARKERS = ("tiny-gpt2", "sshleifer/tiny", "hf-internal-testing")


def _metric(blob: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    # Also accept *_pct variants from DomainForge eval JSON.
    expanded: list[str] = []
    for k in keys:
        expanded.append(k)
        if not k.endswith("_pct"):
            expanded.append(f"{k}_pct")
    for k in expanded:
        if k in blob and isinstance(blob[k], (int, float)):
            val = float(blob[k])
            return val / 100.0 if k.endswith("_pct") and val > 1.0 else val
        summary = blob.get("summary") or blob.get("metrics") or {}
        if isinstance(summary, dict) and k in summary:
            val = float(summary[k])
            return val / 100.0 if k.endswith("_pct") and val > 1.0 else val
    return default


def _cuda_proof(require: bool) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        if require:
            raise SystemExit("torch required for --require-cuda") from exc
        return {"cuda": False, "cuda_device": None}
    ok = bool(torch.cuda.is_available())
    if require and not ok:
        raise SystemExit("CUDA required — refuse to write peft_gpu without a live GPU")
    device = torch.cuda.get_device_name(0) if ok else None
    return {"cuda": ok, "cuda_device": device}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--s0", type=Path, required=True)
    p.add_argument("--s3", type=Path, required=True)
    p.add_argument("--s4", type=Path, required=True)
    p.add_argument("--base-model", default="mistralai/Mistral-7B-Instruct-v0.3")
    p.add_argument("--gpu", default="1x CUDA (operator-reported)")
    p.add_argument("--adapter-uri", default="")
    p.add_argument("--sft-examples", type=int, default=0)
    p.add_argument("--dpo-pairs", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--require-cuda", action="store_true")
    p.add_argument(
        "--allow-unverified",
        action="store_true",
        help="Test-only: allow writing peft_gpu without live CUDA (never for hire demos)",
    )
    p.add_argument("--nvidia-smi-log", type=Path, default=None)
    args = p.parse_args()

    out_name = args.out.name.lower()
    writing_gpu_name = "peft_gpu" in out_name or out_name.endswith("gpu.json")
    if writing_gpu_name and not (args.require_cuda or args.allow_unverified):
        raise SystemExit(
            "Refusing peft_gpu export without --require-cuda (or --allow-unverified for unit tests)"
        )
    if any(t in args.base_model.lower() for t in TINY_MARKERS) and not args.allow_unverified:
        raise SystemExit(f"Refuse tiny/smoke base_model for GPU receipt: {args.base_model}")

    proof = _cuda_proof(require=args.require_cuda)
    smi = ""
    if args.nvidia_smi_log and args.nvidia_smi_log.exists():
        smi = args.nvidia_smi_log.read_text()[:4000]
    elif args.require_cuda:
        try:
            smi = subprocess.check_output(["nvidia-smi"], text=True)[:4000]
        except (FileNotFoundError, subprocess.CalledProcessError):
            smi = ""

    s0 = json.loads(args.s0.read_text())
    s3 = json.loads(args.s3.read_text())
    s4 = json.loads(args.s4.read_text())

    receipt = {
        "status": "gpu",
        "cuda": bool(proof["cuda"]) if args.require_cuda else False,
        "honesty": "CUDA PEFT receipt — requires operator GPU run; not peft_smoke.",
        "run_id": f"peft-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "base_model": args.base_model,
        "gpu": args.gpu,
        "cuda_device": proof.get("cuda_device"),
        "sft_examples": args.sft_examples,
        "dpo_pairs": args.dpo_pairs,
        "metrics": {
            "S0_schema_pass": _metric(s0, "schema_pass", "format_adherence", "json_valid_rate"),
            "S3_schema_pass": _metric(s3, "schema_pass", "format_adherence", "json_valid_rate"),
            "S4_schema_pass": _metric(s4, "schema_pass", "format_adherence", "json_valid_rate"),
            "S4_preference_win_rate": _metric(s4, "preference_win_rate", "win_rate", "dpo_win_rate"),
        },
        "adapter_uri": args.adapter_uri,
        "nvidia_smi_excerpt": smi,
        "notes": "RAG still owns facts; PEFT owns schema/behavior (ADR-019/020). Generated by DomainForge export_modelforge_receipt.py.",
        "sources": {
            "s0": str(args.s0),
            "s3": str(args.s3),
            "s4": str(args.s4),
        },
    }
    if args.allow_unverified and not args.require_cuda:
        receipt["honesty"] = (
            "UNVERIFIED export (--allow-unverified). Not a hire-facing CUDA receipt."
        )
        receipt["status"] = "unverified"
        receipt["cuda"] = False

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
