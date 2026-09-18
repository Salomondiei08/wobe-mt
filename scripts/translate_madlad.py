#!/usr/bin/env python3
"""Translate one sentence with a fine-tuned MADLAD Wobé--French model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("models/madlad-wobe-fr/final"))
    parser.add_argument("--direction", choices=["fr2wob", "wob2fr"], default="fr2wob")
    parser.add_argument("--text", required=True)
    args = parser.parse_args()
    target_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(args.model_dir)).to(target_device).eval()
    tag = "<2wob>" if args.direction == "fr2wob" else "<2fr>"
    inputs = tokenizer(f"{tag} {args.text}", return_tensors="pt", truncation=True, max_length=128).to(target_device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=128, num_beams=4)
    print(tokenizer.batch_decode(output, skip_special_tokens=True)[0])


if __name__ == "__main__":
    main()
