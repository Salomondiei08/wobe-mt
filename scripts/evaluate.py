#!/usr/bin/env python3
"""Evaluate a fine-tuned NLLB model on Wobé↔French test set (chrF + BLEU)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_from_disk
from sacrebleu import BLEU, CHRF
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


WOB_LANG = "wob_Latn"
FRA_LANG = "fra_Latn"


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.inference_mode()
def translate_batch(
    model,
    tokenizer,
    texts: list[str],
    src_lang: str,
    tgt_lang: str,
    max_length: int,
    device: str,
    num_beams: int = 4,
) -> list[str]:
    tokenizer.src_lang = src_lang
    forced_bos = tokenizer.convert_tokens_to_ids(tgt_lang)
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)
    outputs = model.generate(
        **inputs,
        forced_bos_token_id=forced_bos,
        max_new_tokens=max_length,
        num_beams=num_beams,
        max_length=None,
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def eval_direction(
    model,
    tokenizer,
    dataset,
    direction: str,
    device: str,
    max_length: int,
    batch_size: int,
    max_samples: int | None,
    out_path: Path,
) -> dict:
    if direction == "fr2wob":
        src_lang, tgt_lang = FRA_LANG, WOB_LANG
        sources = [ex["src_fr"] for ex in dataset]
        references = [ex["tgt_wob"] for ex in dataset]
    else:
        src_lang, tgt_lang = WOB_LANG, FRA_LANG
        sources = [ex["tgt_wob"] for ex in dataset]
        references = [ex["src_fr"] for ex in dataset]

    if max_samples:
        sources = sources[:max_samples]
        references = references[:max_samples]

    hypotheses = []
    for i in tqdm(range(0, len(sources), batch_size), desc=direction):
        batch = sources[i : i + batch_size]
        hyps = translate_batch(
            model, tokenizer, batch, src_lang, tgt_lang, max_length, device
        )
        hypotheses.extend(hyps)

    bleu = BLEU()
    chrf = CHRF(word_order=2)  # chrF++
    bleu_score = bleu.corpus_score(hypotheses, [references])
    chrf_score = chrf.corpus_score(hypotheses, [references])

    rows = []
    for src, ref, hyp in zip(sources, references, hypotheses):
        rows.append({"source": src, "reference": ref, "hypothesis": hyp})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    metrics = {
        "direction": direction,
        "n": len(hypotheses),
        "bleu": bleu_score.score,
        "chrfpp": chrf_score.score,
        "bleu_signature": str(bleu_score),
        "chrf_signature": str(chrf_score),
        "predictions_path": str(out_path),
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("data/hf_dataset"))
    parser.add_argument("--split", default="test", choices=["test", "validation", "train"])
    parser.add_argument("--direction", choices=["fr2wob", "wob2fr", "both"], default="both")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    device = pick_device()
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(args.model_dir))
    model.to(device)
    model.eval()

    # Ensure wob token exists
    if WOB_LANG not in tokenizer.get_vocab() and WOB_LANG not in (
        tokenizer.additional_special_tokens or []
    ):
        raise SystemExit(
            f"Model tokenizer missing {WOB_LANG}. Train with scripts/train_nllb.py first."
        )

    ds = load_from_disk(str(args.dataset))[args.split]
    directions = ["fr2wob", "wob2fr"] if args.direction == "both" else [args.direction]

    all_metrics = []
    for d in directions:
        metrics = eval_direction(
            model,
            tokenizer,
            ds,
            d,
            device,
            args.max_length,
            args.batch_size,
            args.max_samples,
            args.out_dir / f"preds_{args.split}_{d}.jsonl",
        )
        all_metrics.append(metrics)
        print(json.dumps(metrics, indent=2))

    summary_path = args.out_dir / f"metrics_{args.split}.json"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
