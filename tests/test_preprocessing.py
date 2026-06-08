import json

import pandas as pd
import pytest

from factscore.preprocessing import csv_to_factscore_jsonl, write_jsonl


def test_write_jsonl_creates_parent_directory(tmp_path):
    output_path = write_jsonl([{"topic": "one", "output": "summary"}], tmp_path / "nested" / "out.jsonl")

    assert json.loads(output_path.read_text(encoding="utf-8")) == {"topic": "one", "output": "summary"}


def test_csv_to_factscore_jsonl(tmp_path):
    input_path = tmp_path / "summaries.csv"
    output_path = tmp_path / "summaries.jsonl"
    pd.DataFrame([{"id": "doc-1", "summary": "A summary."}]).to_csv(input_path, index=False)

    csv_to_factscore_jsonl(input_path, output_path, "id", "summary")

    assert json.loads(output_path.read_text(encoding="utf-8")) == {"topic": "doc-1", "output": "A summary."}


def test_csv_to_factscore_jsonl_reports_missing_columns(tmp_path):
    input_path = tmp_path / "summaries.csv"
    pd.DataFrame([{"id": "doc-1"}]).to_csv(input_path, index=False)

    with pytest.raises(ValueError, match="summary"):
        csv_to_factscore_jsonl(input_path, tmp_path / "out.jsonl", "id", "summary")
