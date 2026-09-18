#!/usr/bin/env python3
"""Fine-tune MADLAD-400 3B for French <-> Wobé translation.

MADLAD selects its output language by prepending a ``<2xx>`` tag to the
source.  Wobé has no pretrained tag, so this script adds ``<2wob>`` and
initializes it from Fon's West-African target tag.  Unlike the older NLLB path,
decoder labels never inherit a mutable language setting.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, load_from_disk
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)


MODEL_ID = "google/madlad400-3b-mt"
WOB_TAG = "<2wob>"
FRENCH_TAG = "<2fr>"
DONOR_TAG = "<2fon>"


def ensure_wobe_target_tag(tokenizer, model) -> int:
    """Add the MADLAD Wobé target tag and initialize it from a donor tag."""
    if WOB_TAG not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [WOB_TAG]})
        model.resize_token_embeddings(len(tokenizer))
    wob_id = tokenizer.convert_tokens_to_ids(WOB_TAG)
    donor_id = tokenizer.convert_tokens_to_ids(DONOR_TAG)
    if donor_id == tokenizer.unk_token_id:
        donor_id = tokenizer.convert_tokens_to_ids(FRENCH_TAG)
    with torch.no_grad():
        embeddings = model.get_input_embeddings()
        embeddings.weight[wob_id] = embeddings.weight[donor_id].clone()
        output_embeddings = model.get_output_embeddings()
        if output_embeddings is not None and output_embeddings is not embeddings:
            output_embeddings.weight[wob_id] = output_embeddings.weight[donor_id].clone()
    print(f"Added {WOB_TAG} id={wob_id}; initialized from id={donor_id}")
    return wob_id


def attach_translation_lora(model, rank: int, alpha: int, dropout: float):
    """Attach small, trainable adapters instead of updating all 3B weights.

    Full fine-tuning collapsed on the small Bible corpus. T5 attention projects
    use q/k/v/o names, so adapting them preserves the multilingual base while
    allowing Wobé-specific learning.
    """
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as error:
        raise SystemExit("Install peft into the project dependencies before using --lora.") from error
    config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["q", "k", "v", "o"],
        bias="none",
    )
    adapted_model = get_peft_model(model, config)
    adapted_model.print_trainable_parameters()
    return adapted_model


def build_direction_dataset(raw, split: str, direction: str) -> Dataset:
    """Return direction-labelled rows without changing source corpus text."""
    rows: list[dict] = []
    for row in raw[split]:
        if direction in {"fr2wob", "both"}:
            rows.append({"source": row["src_fr"], "target": row["tgt_wob"], "target_tag": WOB_TAG})
        if direction in {"wob2fr", "both"}:
            rows.append({"source": row["tgt_wob"], "target": row["src_fr"], "target_tag": FRENCH_TAG})
    random.shuffle(rows)
    return Dataset.from_list(rows)


def tokenize_pair(tokenizer, row: dict, max_length: int) -> dict:
    """Encode MADLAD's target-language prompt and language-neutral labels."""
    source = f"{row['target_tag']} {row['source']}"
    encoded = tokenizer(source, max_length=max_length, truncation=True)
    labels = tokenizer(text_target=row["target"], max_length=max_length, truncation=True)
    if not labels["input_ids"]:
        raise ValueError("Empty target encountered while tokenizing.")
    encoded["labels"] = labels["input_ids"]
    return encoded


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune MADLAD-400 for Wobé ↔ French")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--dataset", type=Path, default=Path("data/hf_dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/madlad-wobe-fr"))
    parser.add_argument("--direction", choices=["fr2wob", "wob2fr", "both"], default="both")
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--lora", action="store_true", help="Train LoRA adapters rather than all model weights.")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--confirm-restricted-data-rights", action="store_true")
    args = parser.parse_args()

    if not args.confirm_restricted_data_rights:
        raise SystemExit("See data/LICENSE_AND_RIGHTS.md and pass --confirm-restricted-data-rights after written permission.")

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if not torch.cuda.is_available():
        raise SystemExit("MADLAD-400 3B training requires CUDA for this project.")

    raw = load_from_disk(str(args.dataset))
    train_ds = build_direction_dataset(raw, "train", args.direction)
    eval_ds = build_direction_dataset(raw, "validation", args.direction)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else None
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, torch_dtype=dtype)
    ensure_wobe_target_tag(tokenizer, model)
    if args.lora:
        model = attach_translation_lora(model, args.lora_rank, args.lora_alpha, args.lora_dropout)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if args.lora:
            # Checkpointed frozen embeddings otherwise detach the graph before
            # it reaches the trainable LoRA layers.
            model.enable_input_require_grads()
        model.config.use_cache = False

    tokenized_train = train_ds.map(lambda row: tokenize_pair(tokenizer, row, args.max_length), remove_columns=train_ds.column_names)
    tokenized_eval = eval_ds.map(lambda row: tokenize_pair(tokenizer, row, args.max_length), remove_columns=eval_ds.column_names)
    n_gpus = max(torch.cuda.device_count(), 1)
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size // 2),
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_ratio=0.05,
        logging_steps=10,
        fp16=args.fp16,
        bf16=args.bf16,
        report_to=[],
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=False,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )
    print(f"Training {args.model}: train={len(tokenized_train)}, eval={len(tokenized_eval)}, gpus={n_gpus}")
    trainer.train()
    final_dir = args.output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    (final_dir / "training_meta.json").write_text(
        json.dumps({"base_model": args.model, "target_tag": WOB_TAG, "direction": args.direction, "train_examples": len(tokenized_train), "eval_examples": len(tokenized_eval), "rights": "restricted Wobé text; written permission required"}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
