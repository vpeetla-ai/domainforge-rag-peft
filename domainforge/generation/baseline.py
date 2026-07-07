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
    Template generator for S0/S1 eval before a real LLM or PEFT adapter is wired.
    """
    path = Path(sop_map_path or "data/manifests/sop_intent_map.json")
    sop_map = load_sop_map(path)

    intent = gold_intent if gold_intent else detect_intent(message)
    category = intent_to_category(intent, sop_map)
    target = build_assistant_target(intent, category, "medium", sop_map)

    if solution == SolutionId.S0_BASELINE:
        target["cite_faq_ids"] = []
    elif retrieved:
        target["cite_faq_ids"] = [c.chunk_id for c in retrieved[:2]]

    return json.dumps(target, separators=(",", ":"))
