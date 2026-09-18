#!/usr/bin/env python3
"""
Fine-tune NLLB for French ↔ Wobé (Wè Northern).

Wobé is not in NLLB's original language inventory. We add a new language
token `wob_Latn` and initialize its embedding from a related / nearby code
(`fon_Latn` or `bam_Latn` as fallback) so the model can learn the new target.

RESEARCH USE ONLY: Wobé NT text is © 2010 Wycliffe Bible Translators.
Obtain a license before publishing models or redistributing text.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)


WOB_LANG = "wob_Latn"
FRA_LANG = "fra_Latn"
# Embedding init donor: Fon is West African Latn; not Kru but better than random.
INIT_FROM_LANG = "fon_Latn"


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def ensure_wobe_lang_token(tokenizer, model, init_from: str = INIT_FROM_LANG) -> int:
    """Add wob_Latn special token and copy embedding from an existing lang code."""
    vocab = tokenizer.get_vocab()
    if WOB_LANG not in vocab:
        n_added = tokenizer.add_tokens([WOB_LANG], special_tokens=True)
        if n_added:
            model.resize_token_embeddings(len(tokenizer))
        if hasattr(tokenizer, "fairseq_tokens_to_ids"):
            tokenizer.fairseq_tokens_to_ids[WOB_LANG] = tokenizer.convert_tokens_to_ids(WOB_LANG)
        if hasattr(tokenizer, "fairseq_ids_to_tokens"):
            tokenizer.fairseq_ids_to_tokens[tokenizer.convert_tokens_to_ids(WOB_LANG)] = WOB_LANG

    wob_id = tokenizer.convert_tokens_to_ids(WOB_LANG)
    init_id = tokenizer.convert_tokens_to_ids(init_from)
    unk = getattr(tokenizer, "unk_token_id", None)
    if init_id is None or init_id == unk:
        init_id = tokenizer.convert_tokens_to_ids(FRA_LANG)

    with torch.no_grad():
        emb = model.get_input_embeddings()
        emb.weight[wob_id] = emb.weight[init_id].clone()
        out_emb = model.get_output_embeddings()
        if out_emb is not None and out_emb is not emb:
            out_emb.weight[wob_id] = out_emb.weight[init_id].clone()

    print(f"Language token {WOB_LANG} id={wob_id}, initialized from id={init_id} ({init_from})")
    return wob_id


def tokenize_translation_pair(tokenizer, source: str, target: str, src_lang: str, tgt_lang: str, max_length: int) -> dict:
    """Tokenize one NLLB pair with explicit source *and* target language codes.

    NLLB encodes its language token into both encoder inputs and decoder labels.
    Merely setting ``src_lang`` makes ``text_target`` inherit a stale source
    language, silently teaching the model the wrong decoder prefix.  Keeping
    this routing in one small function makes that failure observable.
    """
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang
    model_inputs = tokenizer(source, max_length=max_length, truncation=True)
    labels = tokenizer(text_target=target, max_length=max_length, truncation=True)
    expected_prefix = tokenizer.convert_tokens_to_ids(tgt_lang)
    if not labels["input_ids"] or labels["input_ids"][0] != expected_prefix:
        raise ValueError(
            f"NLLB target prefix mismatch: expected {tgt_lang} ({expected_prefix}), "
            f"got {labels['input_ids'][:1]}."
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune NLLB on Wobé↔French")
    parser.add_argument(
        "--model",
        default="facebook/nllb-200-3.3B",
        help=(
            "Base model. On 8×GPU prefer facebook/nllb-200-3.3B; "
            "alternatives: facebook/nllb-200-1.3B, facebook/nllb-200-distilled-600M"
        ),
    )
    parser.add_argument("--dataset", type=Path, default=Path("data/hf_dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/nllb-wobe-fr"))
    parser.add_argument("--direction", choices=["fr2wob", "wob2fr", "both"], default="both")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per-device batch size (effective = batch × GPUs × grad_accum)",
    )
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--fp16", action="store_true", help="Use fp16 (CUDA)")
    parser.add_argument("--bf16", action="store_true", help="Use bf16 (CUDA Ampere+)")
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Trade compute for VRAM (useful for 3.3B on smaller cards)",
    )
    parser.add_argument(
        "--confirm-restricted-data-rights",
        action="store_true",
        help=(
            "Confirm written permission to train on the restricted Wobé NT. "
            "This flag records an operator acknowledgement; it is not a license."
        ),
    )
    args = parser.parse_args()

    if not args.confirm_restricted_data_rights:
        raise SystemExit(
            "Refusing to train on restricted Wobé NT text without "
            "--confirm-restricted-data-rights. See data/LICENSE_AND_RIGHTS.md."
        )

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = pick_device()
    n_gpus = torch.cuda.device_count() if device == "cuda" else 0
    print(f"Device: {device} | visible GPUs: {n_gpus}")
    if device == "cuda":
        for i in range(n_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU{i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")

    directions = ["fr2wob", "wob2fr"] if args.direction == "both" else [args.direction]

    raw = load_from_disk(str(args.dataset))
    print(raw)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # dtype for load: prefer bf16/fp16 on multi-GPU to fit 3.3B
    load_kwargs = {}
    if device == "cuda" and (args.bf16 or args.fp16):
        load_kwargs["torch_dtype"] = torch.bfloat16 if args.bf16 else torch.float16
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, **load_kwargs)
    ensure_wobe_lang_token(tokenizer, model)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        print("Gradient checkpointing enabled")

    if len(directions) == 2:
        def make_joint(split_name: str):
            rows = []
            for ex in raw[split_name]:
                rows.append(
                    {
                        "source": ex["src_fr"],
                        "target": ex["tgt_wob"],
                        "src_lang": FRA_LANG,
                        "tgt_lang": WOB_LANG,
                        "id": ex["id"] + "|fr2wob",
                    }
                )
                rows.append(
                    {
                        "source": ex["tgt_wob"],
                        "target": ex["src_fr"],
                        "src_lang": WOB_LANG,
                        "tgt_lang": FRA_LANG,
                        "id": ex["id"] + "|wob2fr",
                    }
                )
            from datasets import Dataset

            random.shuffle(rows)
            return Dataset.from_list(rows)

        train_ds = make_joint("train")
        eval_ds = make_joint("validation")
    else:
        direction = directions[0]

        def map_one(split_name: str):
            rows = []
            for ex in raw[split_name]:
                if direction == "fr2wob":
                    rows.append(
                        {
                            "source": ex["src_fr"],
                            "target": ex["tgt_wob"],
                            "src_lang": FRA_LANG,
                            "tgt_lang": WOB_LANG,
                            "id": ex["id"],
                        }
                    )
                else:
                    rows.append(
                        {
                            "source": ex["tgt_wob"],
                            "target": ex["src_fr"],
                            "src_lang": WOB_LANG,
                            "tgt_lang": FRA_LANG,
                            "id": ex["id"],
                        }
                    )
            from datasets import Dataset

            return Dataset.from_list(rows)

        train_ds = map_one("train")
        eval_ds = map_one("validation")

    if args.max_train_samples:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
    if args.max_eval_samples:
        eval_ds = eval_ds.select(range(min(args.max_eval_samples, len(eval_ds))))

    def tokenize_example_v2(ex):
        return tokenize_translation_pair(
            tokenizer=tokenizer,
            source=ex["source"],
            target=ex["target"],
            src_lang=ex["src_lang"],
            tgt_lang=ex["tgt_lang"],
            max_length=args.max_length,
        )

    train_tok = train_ds.map(tokenize_example_v2, remove_columns=train_ds.column_names)
    eval_tok = eval_ds.map(tokenize_example_v2, remove_columns=eval_ds.column_names)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    use_fp16 = bool(args.fp16 and device == "cuda" and not args.bf16)
    use_bf16 = bool(args.bf16 and device == "cuda")

    # Auto-enable mixed precision on multi-GPU CUDA if user didn't pick one
    if device == "cuda" and not use_fp16 and not use_bf16:
        # Prefer bf16 when available (Ampere+), else fp16
        if torch.cuda.is_bf16_supported():
            use_bf16 = True
            print("Auto-enabled bf16")
        else:
            use_fp16 = True
            print("Auto-enabled fp16")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    world = max(n_gpus, 1)
    effective_bs = args.batch_size * world * args.grad_accum
    print(f"Effective global batch size: {effective_bs}")

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
        warmup_steps=100,
        logging_steps=10,
        predict_with_generate=False,
        fp16=use_fp16,
        bf16=use_bf16,
        report_to=[],
        save_total_limit=2,
        load_best_model_at_end=False,
        seed=args.seed,
        dataloader_pin_memory=(device == "cuda"),
        dataloader_num_workers=4 if device == "cuda" else 0,
        ddp_find_unused_parameters=False,
        # HF Trainer uses all visible GPUs under torchrun / launch
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    print(
        f"Training on {len(train_tok)} examples, eval {len(eval_tok)}, "
        f"model={args.model}, direction={args.direction}, epochs={args.epochs}, "
        f"per_device_bs={args.batch_size}, accum={args.grad_accum}, "
        f"gpus={world}, effective_bs={effective_bs}, fp16={use_fp16}, bf16={use_bf16}"
    )
    trainer.train()

    final_dir = args.output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    meta = {
        "base_model": args.model,
        "direction": args.direction,
        "wobe_lang_code": WOB_LANG,
        "init_from_lang": INIT_FROM_LANG,
        "train_examples": len(train_tok),
        "eval_examples": len(eval_tok),
        "epochs": args.epochs,
        "max_length": args.max_length,
        "batch_size_per_device": args.batch_size,
        "grad_accum": args.grad_accum,
        "n_gpus": world,
        "effective_batch_size": effective_bs,
        "device": device,
        "fp16": use_fp16,
        "bf16": use_bf16,
        "license_note": (
            "Wobé text © 2010 Wycliffe Bible Translators. "
            "Research prototype — not for redistribution without permission."
        ),
    }
    (final_dir / "training_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved model to {final_dir}")


if __name__ == "__main__":
    main()
