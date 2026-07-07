from __future__ import annotations

import json
from pathlib import Path

from domainforge.config import Settings, get_settings
from domainforge.eval.harness import EvalExample, EvalResult, SolutionId, load_golden_jsonl, run_eval, write_eval_result
from domainforge.generation.baseline import generate_triage_json
from domainforge.rag.factory import create_retriever
from domainforge.rag.intent_router import detect_intent


def run_solution_on_golden(
    solution: SolutionId,
    golden_path: Path,
    settings: Settings | None = None,
    use_gold_intent: bool = False,
) -> EvalResult:
    settings = settings or get_settings()
    # S2+ uses hybrid retriever for governed recall
    if solution in (
        SolutionId.S2_HYBRID_RAG,
        SolutionId.S3_PEFT_HYBRID,
        SolutionId.S4_DPO_PEFT,
    ):
        from domainforge.prep.chunk_sop import chunk_all_sops
        from domainforge.rag.hybrid import HybridRetriever

        chunks = chunk_all_sops(settings.corpus_dir, settings.manifest_path)
        retriever = HybridRetriever(chunks)
    else:
        retriever = create_retriever(settings)
    examples_in = load_golden_jsonl(golden_path)
    examples_out: list[EvalExample] = []

    for ex in examples_in:
        intent_for_retrieval = ex.gold_intent if use_gold_intent else detect_intent(ex.instruction)
        retrieved = []
        if solution != SolutionId.S0_BASELINE:
            retrieved = retriever.search(
                ex.instruction,
                top_k=5,
                intent=intent_for_retrieval,
            )

        prediction = generate_triage_json(
            ex.instruction,
            solution,
            sop_map_path=settings.manifest_path,
            retrieved=retrieved,
            gold_intent=ex.gold_intent
            if use_gold_intent or solution == SolutionId.S4_DPO_PEFT
            else None,
        )
        allowed = [c.chunk_id for c in retrieved] if retrieved else ex.allowed_cite_ids
        examples_out.append(
            EvalExample(
                instruction=ex.instruction,
                gold_intent=ex.gold_intent,
                prediction=prediction,
                reference=ex.reference,
                allowed_cite_ids=allowed,
                consistency_runs=ex.consistency_runs,
            )
        )

    return run_eval(solution, examples_out)


def compare_solutions(
    golden_path: Path,
    solutions: list[SolutionId] | None = None,
    out_dir: Path | None = None,
) -> dict[str, dict]:
    settings = get_settings()
    solutions = solutions or [
        SolutionId.S0_BASELINE,
        SolutionId.S1_NAIVE_RAG,
        SolutionId.S2_HYBRID_RAG,
    ]
    out_dir = out_dir or Path("data/eval/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    table: dict[str, dict] = {}
    for sol in solutions:
        result = run_solution_on_golden(sol, golden_path)
        write_eval_result(result, out_dir / f"{sol.value}.json")
        table[sol.value] = result.to_dict()

    if SolutionId.S3_PEFT_HYBRID in solutions and SolutionId.S4_DPO_PEFT in solutions:
        from domainforge.eval.alignment import preference_win_rate

        # Re-run to capture live predictions for win-rate (uses full retriever path)
        s3_examples: list[EvalExample] = []
        s4_examples: list[EvalExample] = []
        examples_in = load_golden_jsonl(golden_path)
        settings = get_settings()
        if SolutionId.S2_HYBRID_RAG in solutions or SolutionId.S3_PEFT_HYBRID in solutions:
            from domainforge.prep.chunk_sop import chunk_all_sops
            from domainforge.rag.hybrid import HybridRetriever

            chunks = chunk_all_sops(settings.corpus_dir, settings.manifest_path)
            retriever = HybridRetriever(chunks)
        else:
            retriever = create_retriever(settings)

        for ex in examples_in:
            intent_s3 = detect_intent(ex.instruction)
            retrieved_s3 = retriever.search(ex.instruction, top_k=5, intent=intent_s3)
            s3_pred = generate_triage_json(
                ex.instruction,
                SolutionId.S3_PEFT_HYBRID,
                sop_map_path=settings.manifest_path,
                retrieved=retrieved_s3,
            )
            retrieved_s4 = retriever.search(ex.instruction, top_k=5, intent=ex.gold_intent)
            s4_pred = generate_triage_json(
                ex.instruction,
                SolutionId.S4_DPO_PEFT,
                sop_map_path=settings.manifest_path,
                retrieved=retrieved_s4,
                gold_intent=ex.gold_intent,
            )
            s3_examples.append(
                EvalExample(
                    instruction=ex.instruction,
                    gold_intent=ex.gold_intent,
                    prediction=s3_pred,
                    allowed_cite_ids=[c.chunk_id for c in retrieved_s3] or ex.allowed_cite_ids,
                )
            )
            s4_examples.append(
                EvalExample(
                    instruction=ex.instruction,
                    gold_intent=ex.gold_intent,
                    prediction=s4_pred,
                    allowed_cite_ids=[c.chunk_id for c in retrieved_s4] or ex.allowed_cite_ids,
                )
            )

        win_rate = preference_win_rate(
            [e.prediction for e in s3_examples],
            [e.prediction for e in s4_examples],
            [e.gold_intent for e in s3_examples],
            [e.allowed_cite_ids for e in s3_examples],
        )
        table["s4_vs_s3_preference_win_rate_pct"] = {"value": win_rate}
        if SolutionId.S4_DPO_PEFT.value in table:
            table[SolutionId.S4_DPO_PEFT.value]["preference_win_rate_pct"] = win_rate

    return table
