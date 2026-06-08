"""Prepare the ACL Anthology corpus for FActScore."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from factscore.preprocessing import register_knowledge_source, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", required=True, help="ACL Anthology parquet file.")
    parser.add_argument("--output-path", default="acl_corpus.jsonl", help="Destination corpus JSONL.")
    parser.add_argument("--register", action="store_true", help="Build the local retrieval database.")
    parser.add_argument("--knowledge-source", default="acl_corpus")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.input_path, columns=["title", "full_text"]).dropna()
    rows = [
        {"title": title.strip(), "text": full_text.strip()}
        for title, full_text in zip(df["title"], df["full_text"])
        if title.strip() and full_text.strip()
    ]
    output_path = write_jsonl(rows, args.output_path)
    if args.register:
        register_knowledge_source(args.knowledge_source, output_path)
    print(f"Wrote {len(rows)} ACL papers to {Path(output_path)}")


if __name__ == "__main__":
    main()
