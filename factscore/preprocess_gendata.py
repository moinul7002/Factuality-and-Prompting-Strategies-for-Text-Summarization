"""Convert generated summaries from CSV to FActScore JSONL."""

from __future__ import annotations

import argparse
from pathlib import Path

from factscore.preprocessing import csv_to_factscore_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", required=True, help="CSV file containing generated summaries.")
    parser.add_argument("--output-path", required=True, help="Destination JSONL file.")
    parser.add_argument("--topic-column", default="article_id", help="Column to use as the FActScore topic.")
    parser.add_argument("--summary-column", default="llama2chat7b_summary", help="Column to use as generated output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = csv_to_factscore_jsonl(
        input_path=args.input_path,
        output_path=args.output_path,
        topic_column=args.topic_column,
        output_column=args.summary_column,
    )
    print(f"Wrote {Path(output_path)}")


if __name__ == "__main__":
    main()
