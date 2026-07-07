from __future__ import annotations

import json


def consistency_rate(outputs_per_example: list[list[str]]) -> float:
    """Fraction of examples where all deterministic runs produce identical output."""
    if not outputs_per_example:
        return 0.0
    stable = 0
    for runs in outputs_per_example:
        if runs and all(r == runs[0] for r in runs):
            stable += 1
    return 100.0 * stable / len(outputs_per_example)


def hallucination_frequency(
    predictions: list[str],
    allowed_cite_ids: list[list[str]],
) -> float:
    """
    % outputs with invented cite_faq_ids not in the retrieval allow-list.
    """
    if not predictions:
        return 0.0
    violations = 0
    for pred, allowed in zip(predictions, allowed_cite_ids, strict=True):
        try:
            data = json.loads(pred)
        except json.JSONDecodeError:
            violations += 1
            continue
        cites = set(data.get("cite_faq_ids") or [])
        allowed_set = set(allowed)
        if cites - allowed_set:
            violations += 1
    return 100.0 * violations / len(predictions)
