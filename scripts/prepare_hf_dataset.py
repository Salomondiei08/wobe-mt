#!/usr/bin/env python3
"""Convert aligned JSONL splits into HuggingFace DatasetDict on disk."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset, DatasetDict


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append(
                {
                    "id": obj["ref"],
                    "src_fr": obj["french"].strip(),
                    "tgt_wob": obj["wobe"].strip(),
                    "book": obj.get("book", ""),
                    "chapter": int(obj.get("chapter", 0)),
                    "verse": int(obj.get("verse", 0)),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/aligned"),
        help="Directory with train/dev/test JSONL files (use book_holdout for reporting)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/hf_dataset"),
        help="Output directory for saved DatasetDict",
    )
    args = parser.parse_args()

    splits = {}
    for name in ("train", "dev", "test"):
        path = args.data_dir / f"wobe_french_nt_{name}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        rows = load_jsonl(path)
        # HF uses "validation" instead of "dev"
        key = "validation" if name == "dev" else name
        splits[key] = Dataset.from_list(rows)
        print(f"{key}: {len(rows)} pairs")

    ds = DatasetDict(splits)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(args.out_dir))
    print(f"Saved DatasetDict to {args.out_dir}")


if __name__ == "__main__":
    main()
