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


if __name__ == "__main__":
    main()
