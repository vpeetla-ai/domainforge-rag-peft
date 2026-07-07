from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from domainforge.eval.scorers.format_adherence import format_adherence_rate
from domainforge.eval.scorers.hallucination import consistency_rate, hallucination_frequency
from domainforge.eval.scorers.intent_accuracy import intent_accuracy_rate
from domainforge.eval.scorers.rouge_bleu import mean_rouge_l


class SolutionId(str, Enum):
    S0_BASELINE = "s0_baseline"
    S1_NAIVE_RAG = "s1_naive_rag"
    S2_HYBRID_RAG = "s2_hybrid_rag"
    S3_PEFT_HYBRID = "s3_peft_hybrid"


@dataclass
class EvalExample:
    instruction: str
    gold_intent: str
    prediction: str
    reference: str = ""
    allowed_cite_ids: list[str] = field(default_factory=list)
    consistency_runs: list[str] = field(default_factory=list)


@dataclass
class EvalResult:
    solution_id: SolutionId
    n_examples: int
    format_adherence_pct: float
    intent_accuracy_pct: float
    rouge_l_pct: float
    consistency_pct: float
    hallucination_freq_pct: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "solution_id": self.solution_id.value,
            "n_examples": self.n_examples,
            "format_adherence_pct": round(self.format_adherence_pct, 2),
            "intent_accuracy_pct": round(self.intent_accuracy_pct, 2),
            "rouge_l_pct": round(self.rouge_l_pct, 2),
            "consistency_pct": round(self.consistency_pct, 2),
            "hallucination_freq_pct": round(self.hallucination_freq_pct, 2),
        }


def run_eval(solution_id: SolutionId, examples: list[EvalExample]) -> EvalResult:
    predictions = [ex.prediction for ex in examples]
    gold_intents = [ex.gold_intent for ex in examples]
    references = [ex.reference or ex.prediction for ex in examples]
    allowed = [ex.allowed_cite_ids for ex in examples]
    consistency_inputs = [ex.consistency_runs or [ex.prediction] for ex in examples]

    return EvalResult(
        solution_id=solution_id,
        n_examples=len(examples),
        format_adherence_pct=format_adherence_rate(predictions),
        intent_accuracy_pct=intent_accuracy_rate(predictions, gold_intents),
        rouge_l_pct=mean_rouge_l(predictions, references),
        consistency_pct=consistency_rate(consistency_inputs),
        hallucination_freq_pct=hallucination_frequency(predictions, allowed),
    )


def load_golden_jsonl(path: Path) -> list[EvalExample]:
    examples: list[EvalExample] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            examples.append(
                EvalExample(
                    instruction=row["instruction"],
                    gold_intent=row["gold_intent"],
                    prediction=row.get("prediction", row.get("gold_prediction", "")),
                    reference=row.get("reference", ""),
                    allowed_cite_ids=row.get("allowed_cite_ids", []),
                    consistency_runs=row.get("consistency_runs", []),
                )
            )
    return examples


def write_eval_result(result: EvalResult, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
