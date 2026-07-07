"""GPU training pipeline: S3 SFT → DPO S4 → Ollama export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from domainforge.eval.harness import SolutionId
from domainforge.eval.runner import compare_solutions, run_solution_on_golden
from domainforge.train.qlora import pick_device


def require_cuda() -> str:
    device = pick_device(force_cpu=False)
    if device != "cuda":
        raise RuntimeError(
            "CUDA GPU required for Mistral QLoRA pipeline. "
            "Use --tiny for CPU smoke or run on RunPod/Colab (see docs/GPU_OLLAMA_PIPELINE.md)."
        )
    return device


def eval_gate_s4_vs_s3(
    golden_path: Path,
    *,
    min_win_rate_pct: float = 20.0,
    min_format_pct: float = 100.0,
) -> dict[str, Any]:
    """Promotion gate: S4 must beat S3 on alignment without format regression."""
    table = compare_solutions(
        golden_path,
        solutions=[SolutionId.S3_PEFT_HYBRID, SolutionId.S4_DPO_PEFT],
    )
    s3 = table.get(SolutionId.S3_PEFT_HYBRID.value, {})
    s4 = table.get(SolutionId.S4_DPO_PEFT.value, {})
    win_block = table.get("s4_vs_s3_preference_win_rate_pct", {})
    win_rate = float(win_block.get("value", s4.get("preference_win_rate_pct", 0.0)))

    passed = (
        win_rate >= min_win_rate_pct
        and float(s4.get("format_adherence_pct", 0.0)) >= min_format_pct
        and float(s4.get("format_adherence_pct", 0.0)) >= float(s3.get("format_adherence_pct", 0.0))
    )
    return {
        "passed": passed,
        "win_rate_pct": win_rate,
        "s3": s3,
        "s4": s4,
        "thresholds": {
            "min_win_rate_pct": min_win_rate_pct,
            "min_format_pct": min_format_pct,
        },
    }


def run_gpu_pipeline(
    *,
    train_file: Path,
    val_file: Path,
    prefs_train: Path,
    prefs_val: Path,
    golden_path: Path,
    sft_config: Path,
    dpo_config: Path,
    sft_output: Path,
    dpo_output: Path,
    sft_steps: int = 200,
    dpo_steps: int = 100,
    ollama_sft_name: str = "domainforge-triage",
    ollama_dpo_name: str = "domainforge-triage-dpo",
    skip_ollama_create: bool = False,
    force_cpu: bool = False,
    tiny_model: str = "sshleifer/tiny-gpt2",
) -> dict[str, Any]:
    """End-to-end: SFT (S3) → DPO (S4) → eval gate → merge → Ollama."""
    from domainforge.prep.preferences import build_preference_splits
    from domainforge.train.dpo import train_dpo
    from domainforge.train.export import package_for_ollama
    from domainforge.train.qlora import train_qlora

    if not force_cpu:
        device = require_cuda()
    else:
        device = "cpu"

    if not prefs_train.exists():
        build_preference_splits(golden_path, Path("data/manifests/sop_intent_map.json"), Path("data/corpus/sop_documents"), prefs_train.parent)

    sft_result = train_qlora(
        train_file,
        val_file,
        sft_output,
        sft_config,
        base_model=tiny_model if force_cpu else None,
        max_steps=sft_steps,
        use_qlora=not force_cpu,
        force_cpu=force_cpu,
    )

    dpo_result = train_dpo(
        prefs_train,
        prefs_val,
        dpo_output,
        dpo_config,
        base_model=tiny_model if force_cpu else None,
        adapter_path=sft_output,
        max_steps=dpo_steps,
        force_cpu=force_cpu,
    )

    gate = eval_gate_s4_vs_s3(golden_path)
    s3_eval = run_solution_on_golden(SolutionId.S3_PEFT_HYBRID, golden_path)
    s4_eval = run_solution_on_golden(SolutionId.S4_DPO_PEFT, golden_path)

    ollama_sft = package_for_ollama(
        sft_output,
        model_name=ollama_sft_name,
        create_model=not skip_ollama_create,
        force_cpu=True,
    )
    ollama_dpo = package_for_ollama(
        dpo_output,
        model_name=ollama_dpo_name,
        create_model=not skip_ollama_create and gate["passed"],
        force_cpu=True,
    )

    summary = {
        "status": "completed",
        "device": device,
        "sft": sft_result,
        "dpo": dpo_result,
        "eval_gate": gate,
        "s3_eval": s3_eval.to_dict(),
        "s4_eval": s4_eval.to_dict(),
        "ollama_sft": ollama_sft,
        "ollama_dpo": ollama_dpo,
        "next_steps": [
            "Set MOCK_LLM=false and OLLAMA_BASE_URL=http://<gpu-host>:11434 on API",
            f"Query with solution=s3_peft_hybrid (model={ollama_sft_name})",
            f"Query with solution=s4_dpo_peft (model={ollama_dpo_name})",
        ],
    }
    report_path = Path("data/train_reports") / f"{dpo_output.name}_pipeline.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["report_path"] = str(report_path)
    return summary
