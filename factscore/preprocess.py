"""Register a JSONL corpus as a FActScore knowledge source."""

from __future__ import annotations

import argparse

from factscore.preprocessing import register_knowledge_source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="Knowledge-source name, for example bbc_corpus.")
    parser.add_argument("--data-path", required=True, help="JSONL corpus with title and text fields.")
    parser.add_argument("--db-path", default=None, help="Optional output SQLite database path.")
    parser.add_argument("--data-dir", default=".cache/factscore", help="Directory for default databases.")
    parser.add_argument("--cache-dir", default=".cache/factscore", help="Directory for retrieval caches.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    register_knowledge_source(
        name=args.name,
        data_path=args.data_path,
        db_path=args.db_path,
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
    )
    print(f"Registered knowledge source '{args.name}'")


if __name__ == "__main__":
    main()
