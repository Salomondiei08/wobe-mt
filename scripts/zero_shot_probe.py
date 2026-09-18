#!/usr/bin/env python3
"""
Probe whether frontier / multilingual models produce any usable Wobé.

1) NLLB zero-shot via surrogate language codes (baseline before fine-tune)
2) Optional: print French seeds for manual LLM probing

Does not call paid APIs by default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


SEEDS = [
    "Bonjour, comment allez-vous ?",
    "Dieu aime le monde.",
    "Il faut boire de l'eau tous les jours.",
    "Le marché est loin du village.",
    "Merci beaucoup pour votre aide.",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="facebook/nllb-200-distilled-600M")
    parser.add_argument(
        "--surrogate-targets",
        nargs="+",
        default=["fon_Latn", "bam_Latn", "ewe_Latn", "swh_Latn"],
        help="NLLB lang codes to try as stand-ins (Wobé is unsupported)",
    )
    parser.add_argument("--out", type=Path, default=Path("results/zero_shot_probe.json"))
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device)
    model.eval()

    results = []
    tok.src_lang = "fra_Latn"
    for tgt in args.surrogate_targets:
        tgt_id = tok.convert_tokens_to_ids(tgt)
        for src in SEEDS:
            inputs = tok(src, return_tensors="pt").to(device)
            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    forced_bos_token_id=tgt_id,
                    max_new_tokens=64,
                    num_beams=4,
                )
            hyp = tok.batch_decode(out, skip_special_tokens=True)[0]
            results.append(
                {
                    "source_fr": src,
                    "surrogate_lang": tgt,
                    "hypothesis": hyp,
                    "note": "Not Wobé — surrogate zero-shot only",
                }
            )
            print(f"[{tgt}] {src} -> {hyp}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")
    print(
        "\nManual LLM probe (copy into Gemini/GPT/Claude):\n"
        "Translate these French sentences into Wobé (Wè Northern, Côte d'Ivoire, ISO wob).\n"
        "If you cannot, say UNKNOWN.\n"
    )
    for s in SEEDS:
        print(f"- {s}")


if __name__ == "__main__":
    main()
