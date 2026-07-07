"""Preference pair generation for DPO — scorer-labeled chosen vs hard-negative rejected."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from domainforge.prep.chatml import SYSTEM_PROMPT, build_assistant_target
from domainforge.prep.chunk_sop import chunk_all_sops, load_sop_map
from domainforge.prep.intent_actions import intent_to_action
from domainforge.eval.alignment import chosen_beats_rejected
from domainforge.rag.intent_router import intent_to_category
from domainforge.rag.naive import RetrievedChunk, format_context_blocks
from domainforge.schemas.triage import Priority, TriageResponse

REJECT_REASONS = (
    "wrong_intent",
    "hallucinated_cite",
    "wrong_action",
    "under_escalation",
    "over_escalation",
)

WRONG_INTENT_SWAP: dict[str, str] = {
    "track_order": "get_refund",
    "get_refund": "track_order",
    "recover_password": "create_account",
    "contact_human_agent": "contact_customer_service",
    "contact_customer_service": "review",
}


def _chunks_by_id(corpus_dir: Path, sop_map_path: Path) -> dict[str, dict[str, Any]]:
    chunks = chunk_all_sops(corpus_dir, sop_map_path)
    return {ch["chunk_id"]: ch for ch in chunks}


def context_blocks_for_cites(
    allowed_cite_ids: list[str],
    chunk_index: dict[str, dict[str, Any]],
) -> list[str]:
    retrieved = [
        RetrievedChunk(chunk_id=cid, text=chunk_index[cid]["text"], score=1.0)
        for cid in allowed_cite_ids
        if cid in chunk_index
    ]
    return format_context_blocks(retrieved)


def build_user_content(instruction: str, context_blocks: list[str]) -> str:
    parts = list(context_blocks) + [f"Customer message: {instruction}"]
    return "\n\n".join(parts)


def build_prompt_text(user_content: str) -> str:
    return (
        f"SYSTEM: {SYSTEM_PROMPT}\n"
        f"USER: {user_content}\n"
        "ASSISTANT:"
    )


def parse_chosen_prediction(row: dict[str, Any], sop_map: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("prediction") or row.get("gold_prediction", "")
    if raw:
        try:
            data = json.loads(raw)
            if data.get("intent") == row["gold_intent"]:
                cites = set(data.get("cite_faq_ids") or [])
                allowed = set(row.get("allowed_cite_ids") or [])
                if not cites or cites <= allowed:
                    return data
        except json.JSONDecodeError:
            pass
    intent = row["gold_intent"]
    category = intent_to_category(intent, sop_map)
    chosen = build_assistant_target(intent, category, "medium", sop_map)
    allowed = row.get("allowed_cite_ids") or []
    if allowed:
        chosen["cite_faq_ids"] = allowed[:2]
    return chosen


def make_rejected(
    chosen: dict[str, Any],
    reason: str,
    sop_map: dict[str, Any],
) -> dict[str, Any]:
    rejected = dict(chosen)
    intent = str(chosen.get("intent", ""))

    if reason == "wrong_intent":
        wrong = WRONG_INTENT_SWAP.get(intent, "contact_customer_service")
        rejected["intent"] = wrong
        rejected["category"] = intent_to_category(wrong, sop_map)
        rejected["suggested_action"] = intent_to_action(wrong)
    elif reason == "hallucinated_cite":
        rejected["cite_faq_ids"] = ["invented_chunk_xyz"]
    elif reason == "wrong_action":
        rejected["suggested_action"] = "no_action_required"
    elif reason == "under_escalation":
        if intent == "contact_human_agent":
            rejected["suggested_action"] = "no_action_required"
            rejected["priority"] = Priority.LOW.value
    elif reason == "over_escalation":
        if intent in {"track_order", "get_refund", "recover_password"}:
            rejected["suggested_action"] = "escalate_to_supervisor"
            rejected["priority"] = Priority.HIGH.value
    else:
        raise ValueError(f"Unknown reject reason: {reason}")

    TriageResponse.model_validate(rejected)
    return rejected


def preference_pair_record(
    instruction: str,
    gold_intent: str,
    chosen: dict[str, Any],
    rejected: dict[str, Any],
    reject_reason: str,
    context_blocks: list[str],
    allowed_cite_ids: list[str],
) -> dict[str, Any]:
    user_content = build_user_content(instruction, context_blocks)
    return {
        "instruction": instruction,
        "gold_intent": gold_intent,
        "prompt": build_prompt_text(user_content),
        "chosen": json.dumps(chosen, separators=(",", ":")),
        "rejected": json.dumps(rejected, separators=(",", ":")),
        "reject_reason": reject_reason,
        "allowed_cite_ids": allowed_cite_ids,
        "context_blocks": context_blocks,
    }


def generate_preferences_from_golden(
    golden_path: Path,
    sop_map_path: Path,
    corpus_dir: Path,
    *,
    reject_reasons: tuple[str, ...] = REJECT_REASONS,
    seed: int = 42,
) -> list[dict[str, Any]]:
    sop_map = load_sop_map(sop_map_path)
    chunk_index = _chunks_by_id(corpus_dir, sop_map_path)
    rng = random.Random(seed)
    pairs: list[dict[str, Any]] = []

    with golden_path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    for idx, row in enumerate(rows):
        chosen = parse_chosen_prediction(row, sop_map)
        allowed = row.get("allowed_cite_ids", [])
        context_blocks = context_blocks_for_cites(allowed, chunk_index)
        chosen_json = json.dumps(chosen, separators=(",", ":"), sort_keys=True)

        for reason in reject_reasons:
            rejected = make_rejected(chosen, reason, sop_map)
            rejected_json = json.dumps(rejected, separators=(",", ":"), sort_keys=True)
            if rejected_json == chosen_json:
                continue
            if not chosen_beats_rejected(
                json.dumps(chosen, separators=(",", ":")),
                json.dumps(rejected, separators=(",", ":")),
                row["gold_intent"],
                allowed,
            ):
                continue
            pairs.append(
                preference_pair_record(
                    instruction=row["instruction"],
                    gold_intent=row["gold_intent"],
                    chosen=chosen,
                    rejected=rejected,
                    reject_reason=reason,
                    context_blocks=context_blocks,
                    allowed_cite_ids=allowed,
                )
            )

    rng.shuffle(pairs)
    return pairs


def write_preferences_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_train_val(
    rows: list[dict[str, Any]],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    if len(shuffled) <= 1:
        return shuffled, []
    val_count = max(1, int(len(shuffled) * val_ratio))
    return shuffled[val_count:], shuffled[:val_count]


def build_preference_splits(
    golden_path: Path,
    sop_map_path: Path,
    corpus_dir: Path,
    out_dir: Path,
) -> dict[str, Any]:
    pairs = generate_preferences_from_golden(golden_path, sop_map_path, corpus_dir)
    train_rows, val_rows = split_train_val(pairs)
    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"
    write_preferences_jsonl(train_rows, train_path)
    write_preferences_jsonl(val_rows, val_path)
    return {
        "status": "ok",
        "total_pairs": len(pairs),
        "train_pairs": len(train_rows),
        "val_pairs": len(val_rows),
        "train_path": str(train_path),
        "val_path": str(val_path),
    }
