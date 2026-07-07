from __future__ import annotations

import json
from pathlib import Path

from domainforge.prep.chatml import build_assistant_target
from domainforge.prep.chunk_sop import load_sop_map
from domainforge.rag.intent_router import detect_intent, intent_to_category
from domainforge.rag.naive import RetrievedChunk
from domainforge.eval.harness import SolutionId


def generate_triage_json(
    message: str,
    solution: SolutionId,
    sop_map_path: Path | str | None = None,
    retrieved: list[RetrievedChunk] | None = None,
    gold_intent: str | None = None,
) -> str:
    """
    Template generator for S0–S4 eval before a real LLM or PEFT/DPO adapter is wired.
    S3 simulates SFT misalignment on adversarial inputs; S4 simulates DPO fixes.
    """
    path = Path(sop_map_path or "data/manifests/sop_intent_map.json")
    sop_map = load_sop_map(path)
    message_l = message.lower()

    if solution == SolutionId.S4_DPO_PEFT:
        intent = gold_intent or detect_intent(message)
    elif solution == SolutionId.S3_PEFT_HYBRID:
        intent = detect_intent(message)
        if "hack" in message_l or "ignore instructions" in message_l:
            intent = "wrong"
    else:
        intent = gold_intent if gold_intent else detect_intent(message)

    category = intent_to_category(intent, sop_map)
    target = build_assistant_target(intent, category, "medium", sop_map)

    if solution == SolutionId.S0_BASELINE:
        target["cite_faq_ids"] = []
    elif solution == SolutionId.S3_PEFT_HYBRID and (
        "hack" in message_l or "ignore instructions" in message_l
    ):
        target["cite_faq_ids"] = ["invented_chunk"]
        target["suggested_action"] = "no_action_required"
    elif retrieved:
        target["cite_faq_ids"] = [c.chunk_id for c in retrieved[:2]]
    elif solution == SolutionId.S4_DPO_PEFT:
        target["cite_faq_ids"] = []

    return json.dumps(target, separators=(",", ":"))
