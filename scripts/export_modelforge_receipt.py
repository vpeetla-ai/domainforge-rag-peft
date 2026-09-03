#!/usr/bin/env python3
"""Export a ModelForge-compatible PEFT receipt from DomainForge training manifests.

Usage (from domainforge-rag-peft root, after scripts/gpu_pipeline.sh):

  python scripts/export_modelforge_receipt.py \\
    --sft-manifest adapters/domainforge-triage-v0/training_manifest.json \\
    --dpo-manifest adapters/domainforge-triage-dpo-v0/training_manifest.json \\
    --adapter-uri adapters/domainforge-triage-dpo-v0 \\
    --require-cuda \\
    --out /path/to/modelforge-llmops/docs/receipts/peft_gpu.json

Honesty:
  Writing peft_gpu.json requires --require-cuda (live CUDA) or --allow-unverified (tests only).
  Tiny smoke bases (e.g. sshleifer/tiny-gpt2) are rejected for peft_gpu unless --allow-unverified.

  This receipt reports ONLY what train_qlora()/train_dpo() actually measured: real example/pair
  counts, real step counts, real wall-clock seconds, pulled straight from the training_manifest.json
  each stage writes. It deliberately does NOT include a quality/win-rate score for the trained
  adapter: domainforge/generation/baseline.py's generate_triage_json() — the function every S0-S4
  eval "solution" runs through, including S3/S4 — is a template/keyword simulator, not real PEFT
  adapter inference (see its own docstring). Scoring S3/S4 against the golden set today would
  silently re-run that simulator and produce numbers unrelated to whatever adapter was just trained,
  which is worse than reporting no quality number at all. Wiring real adapter inference into that
  eval path is tracked as a known gap (see the "known_gaps" field below), not faked here.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TINY_MARKERS = ("tiny-gpt2", "sshleifer/tiny", "hf-internal-testing")

KNOWN_GAPS = [
    "domainforge/generation/baseline.py:generate_triage_json() is a template/keyword simulator "
    "for S3 (PEFT/SFT) and S4 (DPO), not live inference through the trained adapter — see its "
    "docstring. This receipt therefore reports real training config/timing only; it does not "
    "claim a quality or preference-win-rate score for this adapter's generations.",
]


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
    p.add_argument("--sft-manifest", type=Path, required=True, help="adapters/<sft>/training_manifest.json")
    p.add_argument("--dpo-manifest", type=Path, required=True, help="adapters/<dpo>/training_manifest.json")
    p.add_argument("--base-model", default="mistralai/Mistral-7B-Instruct-v0.3")
    p.add_argument("--gpu", default="1x CUDA (operator-reported)")
    p.add_argument("--adapter-uri", default="")
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

    sft_manifest = json.loads(args.sft_manifest.read_text())
    dpo_manifest = json.loads(args.dpo_manifest.read_text())

    receipt = {
        "status": "gpu",
        "cuda": bool(proof["cuda"]) if args.require_cuda else False,
        "honesty": (
            "CUDA PEFT+DPO training receipt — requires operator GPU run; not peft_smoke. "
            "Reports real training config/timing only; see known_gaps for what is NOT claimed."
        ),
        "run_id": f"peft-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "base_model": args.base_model,
        "gpu": args.gpu,
        "cuda_device": proof.get("cuda_device"),
        "sft": {
            "train_examples": sft_manifest.get("train_samples", sft_manifest.get("train_examples")),
            "val_examples": sft_manifest.get("val_samples", sft_manifest.get("val_examples")),
            "max_steps": sft_manifest.get("max_steps"),
            "wall_seconds": sft_manifest.get("wall_seconds"),
            "use_qlora": sft_manifest.get("use_qlora"),
        },
        "dpo": {
            "train_pairs": dpo_manifest.get("train_pairs"),
            "val_pairs": dpo_manifest.get("val_pairs"),
            "max_steps": dpo_manifest.get("max_steps"),
            "wall_seconds": dpo_manifest.get("wall_seconds"),
            "beta": dpo_manifest.get("beta"),
            "adapter_path": dpo_manifest.get("adapter_path"),
        },
        "adapter_uri": args.adapter_uri,
        "nvidia_smi_excerpt": smi,
        "known_gaps": list(KNOWN_GAPS),
        "notes": "RAG still owns facts; PEFT owns schema/behavior (ADR-019/020). Generated by DomainForge export_modelforge_receipt.py.",
        "sources": {
            "sft_manifest": str(args.sft_manifest),
            "dpo_manifest": str(args.dpo_manifest),
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
