from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from domainforge.prep.chatml import instruction_dedup_key, to_chatml_record
from domainforge.prep.chunk_sop import load_sop_map


def _priority_for_intent(sop_map: dict[str, Any], intent: str) -> str:
    for doc in sop_map.get("documents", []):
        if intent in doc.get("intent_tags", []):
            return doc.get("priority_default", "medium")
    return "medium"


def rows_from_bitext_records(records: list[dict[str, Any]], sop_map: dict[str, Any]) -> list[dict]:
    seen: set[str] = set()
    rows: list[dict] = []
    for rec in records:
        instruction = rec.get("instruction", "").strip()
        intent = rec.get("intent", "").strip()
        category = rec.get("category", "").strip()
        if not instruction or not intent:
            continue
        key = instruction_dedup_key(instruction, intent)
        if key in seen:
            continue
        seen.add(key)
        priority = _priority_for_intent(sop_map, intent)
        chatml = to_chatml_record(instruction, intent, category, priority, sop_map)
        chatml["metadata"]["source"] = rec.get("source", "bitext")
        rows.append(chatml)
    return rows


def stratified_split(
    rows: list[dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Stratified split by intent; deterministic shuffle per intent bucket."""
    import random

    rng = random.Random(seed)
    by_intent: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_intent[row["metadata"]["intent"]].append(row)

    train, val, test = [], [], []
    for intent_rows in by_intent.values():
        rng.shuffle(intent_rows)
        n = len(intent_rows)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train.extend(intent_rows[:n_train])
        val.extend(intent_rows[n_train : n_train + n_val])
        test.extend(intent_rows[n_train + n_val :])
    return train, val, test


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def fetch_bitext_dataset(split: str = "train") -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split=split)
    return [dict(row) for row in ds]


def stratified_sample_records(
    records: list[dict[str, Any]],
    max_rows: int,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Sample up to max_rows with roughly equal representation per intent."""
    import random

    rng = random.Random(seed)
    by_intent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        intent = rec.get("intent", "").strip()
        if intent:
            by_intent[intent].append(rec)

    intents = sorted(by_intent.keys())
    if not intents:
        return records[:max_rows]

    per_intent = max(1, max_rows // len(intents))
    sampled: list[dict[str, Any]] = []
    for intent in intents:
        bucket = by_intent[intent][:]
        rng.shuffle(bucket)
        sampled.extend(bucket[:per_intent])

    rng.shuffle(sampled)
    return sampled[:max_rows]


def build_sft_splits(
    out_dir: Path,
    sop_map_path: Path,
    max_rows: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    sop_map = load_sop_map(sop_map_path)
    records = fetch_bitext_dataset("train")
    if max_rows:
        records = stratified_sample_records(records, max_rows, seed=seed)
    rows = rows_from_bitext_records(records, sop_map)
    train, val, test = stratified_split(rows, seed=seed)

    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "val.jsonl", val)
    write_jsonl(out_dir / "test.jsonl", test)

    intent_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        intent_counts[row["metadata"]["intent"]] += 1

    return {
        "total_rows": len(rows),
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "intents": len(intent_counts),
        "intent_counts": dict(sorted(intent_counts.items())),
    }
