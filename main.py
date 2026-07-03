"""
CLI entrypoint.

Usage:
    python main.py data/sample_contract.txt
    python main.py path/to/your_document.txt
"""

from __future__ import annotations

import argparse

from src.crew import run_on_document


def main() -> None:
    parser = argparse.ArgumentParser(description="Document Intelligence Crew (CrewAI)")
    parser.add_argument(
        "document_path",
        nargs="?",
        default="data/sample_contract.txt",
        help="Path to a text document to analyze (defaults to the sample contract)",
    )
    args = parser.parse_args()

    print(f"\n📄 Analyzing: {args.document_path}\n")
    result = run_on_document(args.document_path)

    print("\n" + "=" * 60)
    print("EXTRACTED FACTS")
    print("=" * 60)
    print(result["extraction"])

    print("\n" + "=" * 60)
    print("EXECUTIVE SUMMARY")
    print("=" * 60)
    print(result["summary"])

    print("\n" + "=" * 60)
    print("QA REPORT")
    print("=" * 60)
    print(result["qa_report"])


if __name__ == "__main__":
    main()
