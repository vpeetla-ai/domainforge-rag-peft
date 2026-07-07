from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from domainforge.prep.chunk_sop import cite_ids_for_intent, load_sop_map
from domainforge.prep.intent_actions import intent_to_action
from domainforge.schemas.triage import Priority, TriageResponse


SYSTEM_PROMPT = (
    "You are a customer support triage agent. "
    "Output ONLY valid JSON matching the triage schema. "
    "Use cite_faq_ids from retrieved context chunk IDs only. "
    "If unknown, use null for confidence and empty entities."
)


def build_assistant_target(
    intent: str,
    category: str,
    priority: str,
    sop_map: dict[str, Any],
) -> dict[str, Any]:
    cite_ids = cite_ids_for_intent(sop_map, intent)
    # Use first chunk as primary citation for training targets
    primary_cites = cite_ids[:2] if cite_ids else []
    triage = TriageResponse(
        intent=intent,
        category=category,
        priority=Priority(priority),
        entities={},
        suggested_action=intent_to_action(intent),
        cite_faq_ids=primary_cites,
        confidence=None,
    )
    return triage.model_dump(mode="json")


def to_chatml_record(
    instruction: str,
    intent: str,
    category: str,
    priority: str,
    sop_map: dict[str, Any],
    context_blocks: list[str] | None = None,
) -> dict[str, Any]:
    user_parts: list[str] = []
    if context_blocks:
        for block in context_blocks:
            user_parts.append(block)
    user_parts.append(f"Customer message: {instruction}")
    user_content = "\n\n".join(user_parts)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {
                "role": "assistant",
                "content": json.dumps(
                    build_assistant_target(intent, category, priority, sop_map),
                    separators=(",", ":"),
                ),
            },
        ],
        "metadata": {
            "intent": intent,
            "category": category,
            "instruction_norm": instruction.lower().strip(),
        },
    }


def instruction_dedup_key(instruction: str, intent: str) -> str:
    norm = instruction.lower().strip()
    return hashlib.sha256(f"{norm}|{intent}".encode()).hexdigest()[:16]
