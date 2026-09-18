#!/usr/bin/env python3
"""Create a reproducible book-disjoint Wobé--French evaluation split.

Adjacent Bible verses often repeat wording and narrative context.  Randomly
withholding individual verses therefore leaks unusually similar examples into
training.  This script creates a stricter, fixed split from the full aligned
corpus: Mark is the untouched test book, while Galatians, Ephesians,
Philippians, and Colossians form validation.  The split is intended for model
selection and reporting; it does not claim to measure general-domain quality.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


TEST_BOOKS = {"MRK"}
VALIDATION_BOOKS = {"GAL", "EPH", "PHP", "COL"}


def load_rows(path: Path) -> list[dict]:
    """Read non-empty JSONL records while preserving source ordering."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write UTF-8 JSONL without changing text content or record order."""
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Make book-disjoint MT splits")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/aligned/wobe_french_nt_parallel.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/aligned/book_holdout"),
    )
    args = parser.parse_args()

    rows = load_rows(args.input)
    assignments = {
        "test": [row for row in rows if row["book"] in TEST_BOOKS],
        "dev": [row for row in rows if row["book"] in VALIDATION_BOOKS],
        "train": [
            row
            for row in rows
            if row["book"] not in TEST_BOOKS | VALIDATION_BOOKS
        ],
    }
    if sum(map(len, assignments.values())) != len(rows):
        raise RuntimeError("Split assignments do not cover the full corpus exactly once.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, split_rows in assignments.items():
        write_jsonl(args.output_dir / f"wobe_french_nt_{name}.jsonl", split_rows)

    manifest = {
        "strategy": "book_disjoint",
        "test_books": sorted(TEST_BOOKS),
        "validation_books": sorted(VALIDATION_BOOKS),
        "counts": {name: len(split_rows) for name, split_rows in assignments.items()},
        "book_counts": {
            name: dict(sorted(Counter(row["book"] for row in split_rows).items()))
            for name, split_rows in assignments.items()
        },
        "warning": "Bible-domain only; this is not a general-domain benchmark.",
    }
    (args.output_dir / "SPLIT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
