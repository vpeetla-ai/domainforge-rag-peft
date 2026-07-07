"""Merge PEFT adapters and package for Ollama inference."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from domainforge.prep.chatml import SYSTEM_PROMPT


def _read_manifest(adapter_dir: Path) -> dict[str, Any]:
    path = adapter_dir / "training_manifest.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def merge_adapter_to_hf(
    adapter_dir: Path,
    output_dir: Path,
    base_model: str | None = None,
    force_cpu: bool = True,
) -> dict[str, Any]:
    """Merge LoRA adapter weights into a standalone HuggingFace model directory."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    manifest = _read_manifest(adapter_dir)
    model_id = base_model or manifest.get("base_model")
    if not model_id:
        raise ValueError("base_model required (pass --base-model or use training_manifest.json)")

    adapter_config = adapter_dir / "adapter_config.json"
    if not adapter_config.exists():
        raise FileNotFoundError(f"No adapter_config.json in {adapter_dir}")

    tokenizer_src = adapter_dir if (adapter_dir / "tokenizer.json").exists() else model_id
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float32 if force_cpu else torch.float16
    base = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="cpu" if force_cpu else "auto",
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    merged = model.merge_and_unload()

    output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    meta = {
        "status": "merged",
        "base_model": model_id,
        "adapter_dir": str(adapter_dir),
        "output_dir": str(output_dir),
        "method": manifest.get("method", "sft"),
    }
    (output_dir / "merge_manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def write_ollama_modelfile(
    merged_dir: Path,
    modelfile_path: Path,
    *,
    temperature: float = 0.0,
    system_prompt: str = SYSTEM_PROMPT,
) -> Path:
    """Write an Ollama Modelfile pointing at a merged HF model directory."""
    merged_dir = merged_dir.resolve()
    lines = [
        f"FROM {merged_dir}",
        f"PARAMETER temperature {temperature}",
        "PARAMETER num_predict 512",
        'PARAMETER stop "USER:"',
        'PARAMETER stop "ASSISTANT:"',
        f'SYSTEM """{system_prompt}"""',
    ]
    modelfile_path.parent.mkdir(parents=True, exist_ok=True)
    modelfile_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return modelfile_path


def create_ollama_model(model_name: str, modelfile_path: Path) -> dict[str, Any]:
    """Run `ollama create` if the CLI is available."""
    ollama = shutil.which("ollama")
    if not ollama:
        return {
            "status": "blocked",
            "reason": "ollama CLI not found",
            "hint": f"ollama create {model_name} -f {modelfile_path}",
        }
    proc = subprocess.run(
        [ollama, "create", model_name, "-f", str(modelfile_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return {
            "status": "error",
            "returncode": proc.returncode,
            "stderr": proc.stderr.strip(),
            "stdout": proc.stdout.strip(),
        }
    return {
        "status": "created",
        "model_name": model_name,
        "modelfile": str(modelfile_path),
        "stdout": proc.stdout.strip(),
    }


def package_for_ollama(
    adapter_dir: Path,
    *,
    model_name: str,
    merged_root: Path = Path("adapters/merged"),
    modelfile_root: Path = Path("adapters/ollama"),
    base_model: str | None = None,
    create_model: bool = True,
    force_cpu: bool = True,
) -> dict[str, Any]:
    """Merge adapter → HF weights → Modelfile → optional `ollama create`."""
    merged_dir = merged_root / model_name
    merge_result = merge_adapter_to_hf(
        adapter_dir,
        merged_dir,
        base_model=base_model,
        force_cpu=force_cpu,
    )
    modelfile_path = write_ollama_modelfile(merged_dir, modelfile_root / f"{model_name}.Modelfile")
    result: dict[str, Any] = {
        **merge_result,
        "modelfile": str(modelfile_path),
        "ollama_model_name": model_name,
    }
    if create_model:
        result["ollama"] = create_ollama_model(model_name, modelfile_path)
    else:
        result["ollama"] = {
            "status": "skipped",
            "hint": f"ollama create {model_name} -f {modelfile_path}",
        }
    return result
