"""CLI for the evaluation harness.

    python -m messaging_agent.evals.cli data/evals/sample_8613.jsonl
    python -m messaging_agent.evals.cli data/evals/sample_8613.jsonl --judge --report report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .harness import run_file


def main() -> int:
    ap = argparse.ArgumentParser(description="Messaging-agent evaluation harness")
    ap.add_argument("input", help="JSONL dataset")
    ap.add_argument("--judge", action="store_true", help="Run the LLM-as-judge quality layer")
    ap.add_argument("--report", help="Write the full JSON report here")
    ap.add_argument("--verbose", action="store_true", help="Print per-record detail")
    ap.add_argument("--langsmith", action="store_true",
                    help="Run as a hosted LangSmith experiment instead of locally "
                         "(requires LANGCHAIN_API_KEY + `pip install langsmith`)")
    ap.add_argument("--dataset", help="LangSmith dataset name (with --langsmith)")
    ap.add_argument("--experiment", help="LangSmith experiment prefix (with --langsmith)")
    args = ap.parse_args()

    if args.langsmith:
        from . import langsmith_eval
        if not langsmith_eval.langsmith_enabled():
            print("LangSmith is not configured.", langsmith_eval.status())
            print("Set LANGCHAIN_API_KEY and `pip install langsmith` to enable it.")
            return 2
        result = langsmith_eval.run_file_on_langsmith(
            args.input, dataset_name=args.dataset,
            experiment_prefix=args.experiment, use_judge=args.judge)
        print("=== LANGSMITH EXPERIMENT ===")
        for k, v in result.items():
            print(f"{k:20} {v}")
        return 0

    report = run_file(args.input, use_judge=args.judge)
    summary = report["summary"]

    if args.report:
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                     encoding="utf-8")

    print("=== EVAL SUMMARY ===")
    for k, v in summary.items():
        print(f"{k:24} {v}")

    if args.verbose:
        print("\n=== PER RECORD ===")
        for item in report["items"]:
            print(json.dumps(item, ensure_ascii=False, indent=2))

    print("\nRESULT:", "PASS ✅" if summary["passed"] else "FAIL ❌")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
