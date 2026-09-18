#!/usr/bin/env bash
# One-liner presets for server9 (run AFTER deploy / on the server itself).
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true

MODEL="${MODEL:-google/madlad400-3b-mt}"
NPROC="${NPROC:-8}"
EPOCHS="${EPOCHS:-5}"
BATCH="${BATCH:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
MAX_LEN="${MAX_LEN:-128}"
LR="${LR:-2e-5}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TOKENIZERS_PARALLELISM=false

DATA_DIR="${DATA_DIR:-data/aligned/book_holdout}"
if [[ "$DATA_DIR" == "data/aligned/book_holdout" ]]; then
  python scripts/make_book_holdout_splits.py
fi
python scripts/prepare_hf_dataset.py --data-dir "$DATA_DIR"
mkdir -p results models

# Prefer bf16
EXTRA=(--bf16)
python -c "import torch; assert torch.cuda.is_bf16_supported()" 2>/dev/null || EXTRA=(--fp16)

# 3B: checkpointing leaves safe activation headroom even on 48GB A6000 cards.
case "$MODEL" in
  *3.3B*|*3B*|*3b*) EXTRA+=(--gradient-checkpointing) ;;
esac

echo "Launching $MODEL on $NPROC GPUs…"
TRAINER="scripts/train_madlad.py"
OUTPUT_DIR="models/madlad-wobe-fr"
if [[ "$MODEL" == facebook/nllb-* ]]; then
  TRAINER="scripts/train_nllb.py"
  OUTPUT_DIR="models/nllb-wobe-fr"
fi

torchrun --standalone --nproc_per_node="$NPROC" \
  "$TRAINER" \
  --model "$MODEL" \
  --direction both \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH" \
  --grad-accum "$GRAD_ACCUM" \
  --max-length "$MAX_LEN" \
  --lr "$LR" \
  --confirm-restricted-data-rights \
  "${EXTRA[@]}" \
  --output-dir "$OUTPUT_DIR" \
  2>&1 | tee results/full_train.log
