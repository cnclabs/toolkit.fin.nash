from __future__ import annotations

import argparse
import json

from .api import load_dataset
from .evaluator.analysis import alignment_evaluate, baseline_evaluate, benchmark, write_json
from .evaluator.metric import NASH
from .launcher import launch


def main() -> None:
    parser = argparse.ArgumentParser(description="NASH Eval toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="Open the local web visualization.")
    serve.add_argument("results_json", nargs="?", help="Optional completed NASH evaluation result JSON.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=0)
    serve.add_argument("--no-open", action="store_true")
    for name, help_text in (("alignment-eval", "Evaluate annotated NumFinE target-number alignment."), ("baseline-eval", "Compare text, ordered numeric, oracle, and NASH scores.")):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("--dataset", default="numfine_triplet")
        command.add_argument("--split", default="test")
        command.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
        command.add_argument("--backend", default="sentence-transformer")
        command.add_argument("--threshold", type=float, default=0.3763)
        command.add_argument("--output", help="Write full machine-readable JSON here.")
    bench = subparsers.add_parser("benchmark", help="Benchmark local scoring; results are hardware-dependent.")
    bench.add_argument("--sentence1", default="Revenue was 10 million dollars.")
    bench.add_argument("--sentence2", default="Revenue was 11 million dollars.")
    bench.add_argument("--repeats", type=int, default=10)
    bench.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    bench.add_argument("--backend", default="sentence-transformer")
    bench.add_argument("--output")

    args = parser.parse_args()
    if args.command == "serve":
        launch(args.results_json, host=args.host, port=args.port, open_browser=not args.no_open)
    elif args.command in {"alignment-eval", "baseline-eval"}:
        records = load_dataset(args.dataset, split=args.split).records
        metric = NASH(model_name=args.model, backend=args.backend, threshold=args.threshold)
        result = alignment_evaluate(records, metric) if args.command == "alignment-eval" else baseline_evaluate(records, metric)
        write_json(result, args.output)
        _print_summary(result["summary"])
    elif args.command == "benchmark":
        metric = NASH(model_name=args.model, backend=args.backend)
        result = benchmark(metric, args.sentence1, args.sentence2, args.repeats)
        write_json(result, args.output)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()


def _print_summary(summary: dict[str, object]) -> None:
    """Small dependency-free console table; full rows go to --output JSON."""
    width = max([len(str(key)) for key in summary] + [6])
    print(f"{'metric'.ljust(width)}  value")
    print(f"{'-' * width}  {'-' * 12}")
    for key, value in summary.items():
        print(f"{str(key).ljust(width)}  {value}")
