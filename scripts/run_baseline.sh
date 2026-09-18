#!/usr/bin/env bash
# End-to-end baseline: prepare data → fine-tune NLLB → evaluate
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "=== 1. Prepare HF dataset ==="
python scripts/prepare_hf_dataset.py

echo "=== 2. Fine-tune NLLB (both directions) ==="
python scripts/train_nllb.py \
  --model facebook/nllb-200-distilled-600M \
  --direction both \
  --epochs 3 \
  --batch-size 2 \
  --grad-accum 8 \
  --max-length 128 \
  --lr 3e-5 \
  --output-dir models/nllb-wobe-fr

echo "=== 3. Evaluate on test set ==="
python scripts/evaluate.py \
  --model-dir models/nllb-wobe-fr/final \
  --direction both \
  --batch-size 2 \
  --out-dir results

echo "=== Done ==="
cat results/metrics_test.json
