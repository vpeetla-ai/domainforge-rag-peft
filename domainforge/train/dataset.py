from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def messages_to_text(messages: list[dict[str, str]], tokenizer: Any) -> str:
    """Apply model chat template; falls back to simple concat."""
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            pass
    parts: list[str] = []
    for msg in messages:
        parts.append(f"{msg['role'].upper()}: {msg['content']}")
    return "\n".join(parts)


def build_sft_dataset(rows: list[dict[str, Any]], tokenizer: Any) -> list[dict[str, str]]:
    dataset: list[dict[str, str]] = []
    for row in rows:
        messages = row.get("messages")
        if not messages:
            continue
        dataset.append({"text": messages_to_text(messages, tokenizer)})
    return dataset
