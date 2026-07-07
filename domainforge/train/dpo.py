"""DPO preference tuning — Phase 5."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from domainforge.train.qlora import load_train_config, pick_device


def build_dpo_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    dataset: list[dict[str, str]] = []
    for row in rows:
        prompt = row.get("prompt", "")
        chosen = row.get("chosen", "")
        rejected = row.get("rejected", "")
        if prompt and chosen and rejected:
            dataset.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    return dataset


def dry_run_dpo(
    prefs_file: Path,
    config_path: Path,
    base_model: str | None = None,
    max_samples: int = 4,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    from domainforge.train.dataset import load_jsonl

    cfg = load_train_config(config_path)
    model_id = base_model or cfg.get("base_model", "mistralai/Mistral-7B-Instruct-v0.3")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = load_jsonl(prefs_file)[:max_samples]
    dpo_rows = build_dpo_rows(rows)
    sample = dpo_rows[0] if dpo_rows else {}
    return {
        "status": "ok",
        "device": pick_device(),
        "model_id": model_id,
        "samples": len(dpo_rows),
        "prompt_chars": len(sample.get("prompt", "")),
        "chosen_chars": len(sample.get("chosen", "")),
        "rejected_chars": len(sample.get("rejected", "")),
        "prompt_preview": sample.get("prompt", "")[:200],
    }


def train_dpo(
    prefs_file: Path,
    val_file: Path,
    output_dir: Path,
    config_path: Path,
    base_model: str | None = None,
    adapter_path: Path | None = None,
    max_steps: int | None = None,
    force_cpu: bool = False,
) -> dict[str, Any]:
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    from domainforge.train.dataset import load_jsonl
    from domainforge.train.registry import register_adapter

    cfg = load_train_config(config_path)
    model_id = base_model or cfg.get("base_model")
    device = pick_device(force_cpu=force_cpu)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_rows = build_dpo_rows(load_jsonl(prefs_file))
    val_rows = build_dpo_rows(load_jsonl(val_file)) if val_file.exists() else []

    from datasets import Dataset

    train_dataset = Dataset.from_list(train_rows)
    val_dataset = Dataset.from_list(val_rows) if val_rows else None

    model_kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": torch.float32}
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    if device == "cpu":
        model = model.to("cpu")

    lora_targets = list(cfg.get("target_modules", ["q_proj", "v_proj"]))
    if "gpt2" in model_id.lower() or "tiny" in model_id.lower():
        lora_targets = ["c_attn", "c_proj"]

    if adapter_path and adapter_path.exists():
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=True)
    else:
        lora_config = LoraConfig(
            r=int(cfg.get("rank", 16)),
            lora_alpha=int(cfg.get("alpha", 32)),
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=lora_targets,
        )
        model = get_peft_model(model, lora_config)

    steps = max_steps or int(cfg.get("max_steps", 50))
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = DPOConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 2)),
        learning_rate=float(cfg.get("learning_rate", 5e-5)),
        max_steps=steps,
        logging_steps=2,
        save_steps=steps,
        eval_strategy="steps" if val_rows else "no",
        eval_steps=max(steps // 2, 1) if val_rows else None,
        beta=float(cfg.get("beta", 0.1)),
        max_length=int(cfg.get("max_length", 512)),
        max_prompt_length=int(cfg.get("max_prompt_length", 384)),
        report_to=[],
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    started = time.time()
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    manifest = {
        "method": "dpo",
        "base_model": model_id,
        "adapter_path": str(adapter_path) if adapter_path else None,
        "device": device,
        "max_steps": steps,
        "train_pairs": len(train_rows),
        "val_pairs": len(val_rows),
        "beta": float(cfg.get("beta", 0.1)),
        "wall_seconds": round(time.time() - started, 2),
        "config_path": str(config_path),
    }
    (output_dir / "training_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    adapter_id = output_dir.name
    register_adapter(
        Path("adapters/registry.json"),
        adapter_id=adapter_id,
        adapter_dir=output_dir,
        base_model=model_id,
        status="candidate",
        training_manifest=manifest,
    )
    return {"status": "trained", "adapter_id": adapter_id, **manifest}
