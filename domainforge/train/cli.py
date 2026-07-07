"""QLoRA training CLI — Phase 3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def check_train_deps() -> list[str]:
    missing = []
    for pkg in ("torch", "transformers", "peft", "trl"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(description="DomainForge QLoRA training")
    sub = parser.add_subparsers(dest="command", required=True)

    p_dry = sub.add_parser("dry-run", help="Validate dataset tokenization without training")
    p_dry.add_argument("--config", type=Path, default=Path("configs/train_qlora.yaml"))
    p_dry.add_argument("--train-file", type=Path, default=Path("data/sft_pairs/train.jsonl"))
    p_dry.add_argument("--base-model", type=str, default=None)
    p_dry.add_argument("--tiny-model", type=str, default="sshleifer/tiny-gpt2")

    p_train = sub.add_parser("train", help="Run QLoRA SFT training")
    p_train.add_argument("--config", type=Path, default=Path("configs/train_qlora.yaml"))
    p_train.add_argument("--train-file", type=Path, default=Path("data/sft_pairs/train.jsonl"))
    p_train.add_argument("--val-file", type=Path, default=Path("data/sft_pairs/val.jsonl"))
    p_train.add_argument("--output-dir", type=Path, default=Path("adapters/domainforge-triage-v0"))
    p_train.add_argument("--base-model", type=str, default=None)
    p_train.add_argument("--max-steps", type=int, default=None)
    p_train.add_argument("--tiny", action="store_true", help="CPU smoke test with tiny model")
    p_train.add_argument("--no-qlora", action="store_true", help="Disable 4-bit quant (CPU/MPS)")

    p_promote = sub.add_parser("promote", help="Promote adapter in registry")
    p_promote.add_argument("--adapter-id", type=str, required=True)

    p_dpo_dry = sub.add_parser("dpo-dry-run", help="Validate DPO preference tokenization")
    p_dpo_dry.add_argument("--config", type=Path, default=Path("configs/train_dpo.yaml"))
    p_dpo_dry.add_argument("--prefs-file", type=Path, default=Path("data/preferences/train.jsonl"))
    p_dpo_dry.add_argument("--base-model", type=str, default=None)
    p_dpo_dry.add_argument("--tiny-model", type=str, default="sshleifer/tiny-gpt2")

    p_dpo = sub.add_parser("dpo", help="Run DPO preference tuning")
    p_dpo.add_argument("--config", type=Path, default=Path("configs/train_dpo.yaml"))
    p_dpo.add_argument("--prefs-file", type=Path, default=Path("data/preferences/train.jsonl"))
    p_dpo.add_argument("--val-file", type=Path, default=Path("data/preferences/val.jsonl"))
    p_dpo.add_argument("--adapter-path", type=Path, default=None, help="Optional SFT adapter to continue from")
    p_dpo.add_argument("--output-dir", type=Path, default=Path("adapters/domainforge-triage-dpo-v0"))
    p_dpo.add_argument("--base-model", type=str, default=None)
    p_dpo.add_argument("--max-steps", type=int, default=None)
    p_dpo.add_argument("--tiny", action="store_true", help="CPU smoke test with tiny model")

    p_merge = sub.add_parser("merge", help="Merge LoRA adapter into standalone HF weights")
    p_merge.add_argument("--adapter-dir", type=Path, required=True)
    p_merge.add_argument("--output-dir", type=Path, required=True)
    p_merge.add_argument("--base-model", type=str, default=None)

    p_export = sub.add_parser("export-ollama", help="Merge adapter and write Ollama Modelfile")
    p_export.add_argument("--adapter-dir", type=Path, required=True)
    p_export.add_argument("--model-name", type=str, required=True)
    p_export.add_argument("--base-model", type=str, default=None)
    p_export.add_argument("--no-create", action="store_true", help="Skip ollama create (Modelfile only)")

    p_pipe = sub.add_parser("pipeline-gpu", help="S3 SFT → DPO S4 → eval gate → Ollama export")
    p_pipe.add_argument("--sft-config", type=Path, default=Path("configs/train_qlora_gpu.yaml"))
    p_pipe.add_argument("--dpo-config", type=Path, default=Path("configs/train_dpo_gpu.yaml"))
    p_pipe.add_argument("--sft-output", type=Path, default=Path("adapters/domainforge-triage-v0"))
    p_pipe.add_argument("--dpo-output", type=Path, default=Path("adapters/domainforge-triage-dpo-v0"))
    p_pipe.add_argument("--sft-steps", type=int, default=200)
    p_pipe.add_argument("--dpo-steps", type=int, default=100)
    p_pipe.add_argument("--golden", type=Path, default=Path("data/eval_golden/sample.jsonl"))
    p_pipe.add_argument("--tiny-pipeline", action="store_true", help="CPU smoke: tiny-gpt2, 3 steps each")
    p_pipe.add_argument("--skip-ollama-create", action="store_true")

    args = parser.parse_args()
    missing = check_train_deps()
    if missing:
        print(json.dumps({"status": "blocked", "missing": missing, "hint": "pip install -e '.[train]'"}))
        raise SystemExit(1)

    if args.command == "dry-run":
        from domainforge.train.qlora import dry_run_training

        model = args.tiny_model if not args.base_model else args.base_model
        result = dry_run_training(args.train_file, args.config, base_model=model)
        print(json.dumps(result, indent=2))
        return

    if args.command == "train":
        if not args.train_file.exists():
            print(json.dumps({"status": "blocked", "reason": f"Missing {args.train_file}"}))
            raise SystemExit(1)
        from domainforge.train.qlora import train_qlora

        base = "sshleifer/tiny-gpt2" if args.tiny else args.base_model
        result = train_qlora(
            args.train_file,
            args.val_file,
            args.output_dir,
            args.config,
            base_model=base,
            max_steps=args.max_steps or (3 if args.tiny else None),
            use_qlora=not args.no_qlora,
            force_cpu=args.tiny,
        )
        print(json.dumps(result, indent=2))
        return

    if args.command == "promote":
        from domainforge.train.registry import promote_adapter

        entry = promote_adapter(Path("adapters/registry.json"), args.adapter_id)
        print(json.dumps(entry, indent=2))
        return

    if args.command == "dpo-dry-run":
        from domainforge.train.dpo import dry_run_dpo

        model = args.tiny_model if not args.base_model else args.base_model
        result = dry_run_dpo(args.prefs_file, args.config, base_model=model)
        print(json.dumps(result, indent=2))
        return

    if args.command == "dpo":
        if not args.prefs_file.exists():
            print(json.dumps({"status": "blocked", "reason": f"Missing {args.prefs_file}"}))
            raise SystemExit(1)
        from domainforge.train.dpo import train_dpo

        base = "sshleifer/tiny-gpt2" if args.tiny else args.base_model
        result = train_dpo(
            args.prefs_file,
            args.val_file,
            args.output_dir,
            args.config,
            base_model=base,
            adapter_path=args.adapter_path,
            max_steps=args.max_steps or (3 if args.tiny else None),
            force_cpu=args.tiny,
        )
        print(json.dumps(result, indent=2))
        return

    if args.command == "merge":
        from domainforge.train.export import merge_adapter_to_hf

        result = merge_adapter_to_hf(
            args.adapter_dir,
            args.output_dir,
            base_model=args.base_model,
            force_cpu=True,
        )
        print(json.dumps(result, indent=2))
        return

    if args.command == "export-ollama":
        from domainforge.train.export import package_for_ollama

        result = package_for_ollama(
            args.adapter_dir,
            model_name=args.model_name,
            base_model=args.base_model,
            create_model=not args.no_create,
            force_cpu=True,
        )
        print(json.dumps(result, indent=2))
        return

    if args.command == "pipeline-gpu":
        from domainforge.train.pipeline import run_gpu_pipeline

        tiny = args.tiny_pipeline
        try:
            result = run_gpu_pipeline(
                train_file=Path("data/sft_pairs/train.jsonl"),
                val_file=Path("data/sft_pairs/val.jsonl"),
                prefs_train=Path("data/preferences/train.jsonl"),
                prefs_val=Path("data/preferences/val.jsonl"),
                golden_path=args.golden,
                sft_config=args.sft_config if not tiny else Path("configs/train_qlora.yaml"),
                dpo_config=args.dpo_config if not tiny else Path("configs/train_dpo.yaml"),
                sft_output=args.sft_output if not tiny else Path("adapters/domainforge-triage-pipeline-smoke"),
                dpo_output=args.dpo_output if not tiny else Path("adapters/domainforge-triage-dpo-pipeline-smoke"),
                sft_steps=3 if tiny else args.sft_steps,
                dpo_steps=3 if tiny else args.dpo_steps,
                skip_ollama_create=args.skip_ollama_create or tiny,
                force_cpu=tiny,
            )
        except RuntimeError as exc:
            print(json.dumps({"status": "blocked", "reason": str(exc)}))
            raise SystemExit(1) from exc
        print(json.dumps(result, indent=2))
        return


if __name__ == "__main__":
    main()
