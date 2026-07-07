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
    # S2 uses hybrid retriever regardless of default mode
    if solution == SolutionId.S2_HYBRID_RAG:
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
            gold_intent=ex.gold_intent if use_gold_intent else None,
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
    if settings.retriever_mode != "hybrid":
        # Temporarily use hybrid for S2 compare when default memory mode
        pass
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
    return table
