"""Reusable preprocessing helpers for FActScore input files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def write_jsonl(rows: list[dict[str, Any]], output_path: str | Path) -> Path:
    """Write JSON Lines with UTF-8 encoding and create parent directories."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def csv_to_factscore_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    topic_column: str,
    output_column: str,
) -> Path:
    """Convert a CSV file into the JSONL schema expected by FactScorer."""
    df = pd.read_csv(input_path)
    required = {topic_column, output_column}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required CSV columns: {', '.join(sorted(missing))}")

    rows = [
        {"topic": row[topic_column], "output": row[output_column]}
        for _, row in df.iterrows()
    ]
    return write_jsonl(rows, output_path)


def register_knowledge_source(
    name: str,
    data_path: str | Path,
    db_path: str | Path | None = None,
    data_dir: str | Path = ".cache/factscore",
    cache_dir: str | Path = ".cache/factscore",
) -> None:
    """Build a local retrieval database for a JSONL corpus."""
    from factscore.factscorer import FactScorer

    scorer = FactScorer(data_dir=str(data_dir), cache_dir=str(cache_dir))
    scorer.register_knowledge_source(
        name=name,
        data_path=str(data_path),
        db_path=str(db_path) if db_path else None,
    )
