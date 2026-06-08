import json
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_notebooks_are_clean_for_version_control():
    for path in ROOT.rglob("*.ipynb"):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert "D:\\" not in path.read_text(encoding="utf-8"), f"Local path found in {path}"
        for cell in notebook.get("cells", []):
            assert not cell.get("outputs"), f"Notebook outputs found in {path}"
            assert cell.get("execution_count") is None, f"Execution count found in {path}"


def test_citation_contains_paper_doi():
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")

    assert "10.1016/j.neunet.2026.109160" in citation
