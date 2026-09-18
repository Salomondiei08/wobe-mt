#!/usr/bin/env python3
"""Evaluate a MADLAD Wobé--French checkpoint with BLEU and chrF++."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_from_disk
from sacrebleu import BLEU, CHRF
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


WOB_TAG = "<2wob>"
FRENCH_TAG = "<2fr>"


def device() -> str:
    """Choose the fastest available local inference device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_translation_model(model_dir: Path, tokenizer, target_device: str):
    """Load either a full checkpoint or its lightweight LoRA adapter."""
    adapter_config = model_dir / "adapter_config.json"
    if not adapter_config.exists():
        return AutoModelForSeq2SeqLM.from_pretrained(str(model_dir)).to(target_device).eval()
    try:
        from peft import PeftConfig, PeftModel
    except ImportError as error:
        raise SystemExit("Install peft into project dependencies to evaluate this LoRA adapter.") from error
    config = PeftConfig.from_pretrained(str(model_dir))
    base = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
    # The adapter persists the newly added Wobé target token and its embedding.
    # Resize the untouched base before restoring those saved embedding layers.
    base.resize_token_embeddings(len(tokenizer))
    return PeftModel.from_pretrained(base, str(model_dir)).to(target_device).eval()


@torch.inference_mode()
def translate(model, tokenizer, texts: list[str], tag: str, target_device: str, max_length: int) -> list[str]:
    """Generate one target language by using MADLAD's source prompt tag."""
    batch = tokenizer([f"{tag} {text}" for text in texts], return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(target_device)
    outputs = model.generate(
        **batch,
        max_new_tokens=max_length,
        num_beams=4,
        no_repeat_ngram_size=3,
        repetition_penalty=1.15,
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def score_direction(model, tokenizer, dataset, direction: str, target_device: str, batch_size: int, max_length: int, out_path: Path) -> dict:
    """Translate one direction and save every scored prediction for audit."""
    if direction == "fr2wob":
        sources, references, tag = dataset["src_fr"], dataset["tgt_wob"], WOB_TAG
    else:
        sources, references, tag = dataset["tgt_wob"], dataset["src_fr"], FRENCH_TAG
    hypotheses: list[str] = []
    for index in tqdm(range(0, len(sources), batch_size), desc=direction):
        hypotheses.extend(translate(model, tokenizer, sources[index:index + batch_size], tag, target_device, max_length))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for source, reference, hypothesis in zip(sources, references, hypotheses):
            handle.write(json.dumps({"source": source, "reference": reference, "hypothesis": hypothesis}, ensure_ascii=False) + "\n")
    bleu = BLEU().corpus_score(hypotheses, [references])
    chrf = CHRF(word_order=2).corpus_score(hypotheses, [references])
    return {"direction": direction, "n": len(hypotheses), "bleu": bleu.score, "chrfpp": chrf.score, "bleu_signature": str(bleu), "chrf_signature": str(chrf), "predictions_path": str(out_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("data/hf_dataset"))
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--out-dir", type=Path, default=Path("results/madlad"))
    args = parser.parse_args()
    target_device = device()
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir))
    if WOB_TAG not in tokenizer.get_vocab():
        raise SystemExit(f"Checkpoint is missing {WOB_TAG}; train with scripts/train_madlad.py.")
    model = load_translation_model(args.model_dir, tokenizer, target_device)
    dataset = load_from_disk(str(args.dataset))[args.split]
    metrics = [score_direction(model, tokenizer, dataset, direction, target_device, args.batch_size, args.max_length, args.out_dir / f"preds_{args.split}_{direction}.jsonl") for direction in ("fr2wob", "wob2fr")]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"metrics_{args.split}.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
