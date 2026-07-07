from __future__ import annotations

import json
from pathlib import Path

from domainforge.eval.harness import SolutionId, load_golden_jsonl, run_eval, write_eval_result
from domainforge.eval.runner import compare_solutions, run_solution_on_golden


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="DomainForge evaluation")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Score pre-filled golden predictions")
    p_run.add_argument("--golden", type=Path, required=True)
    p_run.add_argument("--solution", type=SolutionId, default=SolutionId.S0_BASELINE)
    p_run.add_argument("--out", type=Path, default=Path("data/eval/results/latest.json"))

    p_gen = sub.add_parser("run-solution", help="Generate predictions for a solution and score")
    p_gen.add_argument("--golden", type=Path, required=True)
    p_gen.add_argument("--solution", type=SolutionId, default=SolutionId.S1_NAIVE_RAG)
    p_gen.add_argument("--out", type=Path, default=Path("data/eval/results/latest.json"))
    p_gen.add_argument("--use-gold-intent", action="store_true")

    p_cmp = sub.add_parser("compare", help="Compare S0/S1 (and more) on same golden set")
    p_cmp.add_argument("--golden", type=Path, required=True)
    p_cmp.add_argument("--out-dir", type=Path, default=Path("data/eval/results"))

    args = parser.parse_args()

    if args.command == "run":
        examples = load_golden_jsonl(args.golden)
        result = run_eval(args.solution, examples)
        write_eval_result(result, args.out)
        print(json.dumps(result.to_dict(), indent=2))
    elif args.command == "run-solution":
        result = run_solution_on_golden(
            args.solution,
            args.golden,
            use_gold_intent=args.use_gold_intent,
        )
        write_eval_result(result, args.out)
        print(json.dumps(result.to_dict(), indent=2))
    elif args.command == "compare":
        table = compare_solutions(args.golden, out_dir=args.out_dir)
        print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
