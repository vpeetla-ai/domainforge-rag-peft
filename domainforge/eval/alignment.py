"""Alignment scoring and S3 vs S4 preference win-rate."""

from __future__ import annotations

import json

from domainforge.eval.scorers.format_adherence import parse_triage_output
from domainforge.eval.scorers.hallucination import hallucination_frequency


def alignment_score(prediction: str, gold_intent: str, allowed_cite_ids: list[str]) -> float:
    """Composite 0–1 score: format + intent + citation faithfulness."""
    if not prediction:
        return 0.0
    points = 0.0
    parsed, _ = parse_triage_output(prediction)
    if parsed is not None:
        points += 1.0
        if parsed.intent == gold_intent:
            points += 1.0
    halluc_pct = hallucination_frequency([prediction], [allowed_cite_ids])
    if halluc_pct == 0.0:
        points += 1.0
    return points / 3.0


def preference_win_rate(
    baseline_predictions: list[str],
    aligned_predictions: list[str],
    gold_intents: list[str],
    allowed_cite_ids: list[list[str]],
) -> float:
    """% examples where aligned model strictly beats baseline on alignment score."""
    if not baseline_predictions:
        return 0.0
    wins = 0
    for base, aligned, gold, allowed in zip(
        baseline_predictions,
        aligned_predictions,
        gold_intents,
        allowed_cite_ids,
        strict=True,
    ):
        base_score = alignment_score(base, gold, allowed)
        aligned_score = alignment_score(aligned, gold, allowed)
        if aligned_score > base_score:
            wins += 1
    return 100.0 * wins / len(baseline_predictions)


def preference_pair_quality(chosen: str, rejected: str, gold_intent: str, allowed_cite_ids: list[str]) -> dict[str, float]:
    return {
        "chosen_alignment": alignment_score(chosen, gold_intent, allowed_cite_ids),
        "rejected_alignment": alignment_score(rejected, gold_intent, allowed_cite_ids),
    }


def chosen_beats_rejected(chosen: str, rejected: str, gold_intent: str, allowed_cite_ids: list[str]) -> bool:
    scores = preference_pair_quality(chosen, rejected, gold_intent, allowed_cite_ids)
    return scores["chosen_alignment"] > scores["rejected_alignment"]
