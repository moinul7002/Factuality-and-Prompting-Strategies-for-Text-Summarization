"""Prepare LongSciVerify data and optionally register its FActScore corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from factscore.preprocessing import register_knowledge_source, write_jsonl


def clean_text(value: str | list[str]) -> str:
    """Remove dataset markup and normalize an abstract or article."""
    parts = [value] if isinstance(value, str) else value
    return " ".join(
        part.replace("<S>", "").replace("</S>", "").replace("<pad>", "").replace("<br />", "").strip()
        for part in parts
    ).strip()


def load_dataset(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        records = json.load(handle)
    for record in records:
        record["abstract_text"] = clean_text(record["abstract_text"])
        record["article_text"] = clean_text(record["article_text"])
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arxiv-path", required=True, help="LongSciVerify arXiv JSON file.")
    parser.add_argument("--pubmed-path", required=True, help="LongSciVerify PubMed JSON file.")
    parser.add_argument("--output-dir", required=True, help="Directory for generated CSV and JSONL files.")
    parser.add_argument("--register", action="store_true", help="Build the local retrieval database.")
    parser.add_argument("--knowledge-source", default="longsciverify_corpus")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_dataset(args.arxiv_path) + load_dataset(args.pubmed_path)
    pd.DataFrame(records).to_csv(output_dir / "longsciverify.csv", index=False)

    corpus_path = write_jsonl(
        [{"title": row["article_id"], "text": row["article_text"]} for row in records],
        output_dir / "longsciverify_corpus.jsonl",
    )
    write_jsonl(
        [{"topic": row["article_id"], "output": row["abstract_text"]} for row in records],
        output_dir / "longsciverify.jsonl",
    )

    if args.register:
        register_knowledge_source(args.knowledge_source, corpus_path)
    print(f"Prepared {len(records)} LongSciVerify records in {output_dir}")


if __name__ == "__main__":
    main()
