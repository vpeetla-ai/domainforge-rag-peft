from __future__ import annotations


def rouge_l_score(hypothesis: str, reference: str) -> float:
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        # Lightweight fallback when rouge-score extra not installed
        hyp_tokens = hypothesis.lower().split()
        ref_tokens = reference.lower().split()
        if not hyp_tokens or not ref_tokens:
            return 0.0
        common = set(hyp_tokens) & set(ref_tokens)
        return len(common) / max(len(ref_tokens), 1)

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(reference, hypothesis)["rougeL"].fmeasure


def mean_rouge_l(predictions: list[str], references: list[str]) -> float:
    if not predictions or len(predictions) != len(references):
        return 0.0
    scores = [rouge_l_score(p, r) for p, r in zip(predictions, references, strict=True)]
    return 100.0 * sum(scores) / len(scores)
