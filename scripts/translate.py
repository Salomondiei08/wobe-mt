#!/usr/bin/env python3
"""Interactive / CLI translation with a fine-tuned Wobé↔French model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


WOB_LANG = "wob_Latn"
FRA_LANG = "fra_Latn"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("models/nllb-wobe-fr/final"))
    parser.add_argument("--direction", choices=["fr2wob", "wob2fr"], default="fr2wob")
    parser.add_argument("--text", type=str, default=None, help="Single sentence; omit for REPL")
    parser.add_argument("--beams", type=int, default=4)
    args = parser.parse_args()

    if not args.model_dir.exists():
        raise SystemExit(f"Model not found: {args.model_dir}")

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(str(args.model_dir))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(args.model_dir)).to(device)
    model.eval()

    def translate(text: str) -> str:
        if args.direction == "fr2wob":
            src, tgt = FRA_LANG, WOB_LANG
        else:
            src, tgt = WOB_LANG, FRA_LANG
        tok.src_lang = src
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                forced_bos_token_id=tok.convert_tokens_to_ids(tgt),
                max_new_tokens=128,
                num_beams=args.beams,
            )
        return tok.batch_decode(out, skip_special_tokens=True)[0]

    if args.text is not None:
        print(translate(args.text))
        return

    print(f"Direction: {args.direction} | device: {device} | model: {args.model_dir}")
    print("Enter text (empty line to quit).")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            break
        print(translate(line))


if __name__ == "__main__":
    main()
