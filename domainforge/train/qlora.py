from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def load_train_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml

        return yaml.safe_load(text)
    except Exception:
        # Minimal fallback if PyYAML not installed
        config: dict[str, Any] = {}
        for line in text.splitlines():
            if ":" in line and not line.strip().startswith("#"):
                key, val = line.split(":", 1)
                config[key.strip()] = val.strip()
        return config


def pick_device(force_cpu: bool = False) -> str:
    import torch

    if force_cpu:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"  # avoid MPS dtype issues during local smoke tests


def dry_run_training(
    train_file: Path,
    config_path: Path,
    base_model: str | None = None,
    max_samples: int = 4,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    from domainforge.train.dataset import build_sft_dataset, load_jsonl

    cfg = load_train_config(config_path)
    model_id = base_model or cfg.get("base_model", "mistralai/Mistral-7B-Instruct-v0.3")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = load_jsonl(train_file)[:max_samples]
    dataset = build_sft_dataset(rows, tokenizer)
    sample = dataset[0]["text"] if dataset else ""
    return {
        "status": "ok",
        "device": pick_device(),
        "model_id": model_id,
        "samples": len(dataset),
        "sample_chars": len(sample),
        "sample_preview": sample[:240],
    }


def train_qlora(
    train_file: Path,
    val_file: Path,
    output_dir: Path,
    config_path: Path,
    base_model: str | None = None,
    max_steps: int | None = None,
    use_qlora: bool = True,
    force_cpu: bool = False,
) -> dict[str, Any]:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from trl import SFTConfig, SFTTrainer

    from domainforge.train.dataset import build_sft_dataset, load_jsonl
    from domainforge.train.registry import register_adapter

    cfg = load_train_config(config_path)
    model_id = base_model or cfg.get("base_model")
    device = pick_device(force_cpu=force_cpu)
    use_quant = use_qlora and device == "cuda"
    if use_qlora and device != "cuda":
        try:
            __import__("bitsandbytes")
        except ImportError:
            use_quant = False

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_rows = load_jsonl(train_file)
    val_rows = load_jsonl(val_file) if val_file.exists() else []
    train_data = build_sft_dataset(train_rows, tokenizer)
    val_data = build_sft_dataset(val_rows, tokenizer)

    from datasets import Dataset

    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data) if val_data else None

    model_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if use_quant:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = "auto"
    else:
        dtype = torch.float32
        model_kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    if not use_quant and device == "cpu":
        model = model.to("cpu")
    if use_quant:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model)

    lora_targets = list(cfg.get("target_modules", ["q_proj", "v_proj"]))
    if "gpt2" in model_id.lower() or "tiny" in model_id.lower():
        lora_targets = ["c_attn", "c_proj"]

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

    training_args = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 4)),
        learning_rate=float(cfg.get("learning_rate", 2e-4)),
        max_steps=steps,
        logging_steps=5,
        save_steps=steps,
        evaluation_strategy="steps" if val_data else "no",
        eval_steps=max(steps // 2, 1) if val_data else None,
        fp16=device == "cuda",
        report_to=[],
        max_seq_length=int(cfg.get("max_seq_length", 1024)),
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
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
        "base_model": model_id,
        "device": device,
        "use_qlora": use_quant,
        "max_steps": steps,
        "train_samples": len(train_data),
        "val_samples": len(val_data),
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
