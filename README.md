# Wobé ↔ French Machine Translation

Research prototype for **Wè Northern / Wobé** (`wob`) ↔ French neural MT.

## Status

| Item | State |
|------|--------|
| Parallel NT corpus | **7,927** verse pairs (aligned) |
| Recommended deployable model | MADLAD-400 3B full fine-tune on 8×A6000 |
| Domain | Biblical only (for now) |
| Rights | Wobé text © 2010 Wycliffe — research use; get license before publish |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

See [`data/README.md`](data/README.md) and [`data/LICENSE_AND_RIGHTS.md`](data/LICENSE_AND_RIGHTS.md).

Main files:

- `data/aligned/wobe_french_nt_{train,dev,test}.jsonl`
- `data/aligned/wobe_french_nt_parallel.jsonl` (full)

## Train on multi-GPU lab host (recommended)

Default model: **`google/madlad400-3b-mt`** via `torchrun` on 8 GPUs. It is an
Apache-2.0 multilingual MT model and is the primary candidate for a model that
could later be deployed commercially (subject to the Wobé-data license). NLLB
3.3B is retained as a research-only comparator because its model card is
CC-BY-NC. The 600M NLLB variant remains useful for fast ablations.

Auth is **password-based**. Do **not** put the password in this repo or in chat.

```bash
# In YOUR local terminal only (password never committed / never pasted to the agent):
export SSHPASS='…'              # type password here yourself
export SSH_HOST=server9         # Host alias from ~/.ssh/config (hostname stays there)

cd /path/to/language
bash scripts/deploy_server9.sh

# Optional:
# MODEL=facebook/nllb-200-3.3B NPROC=8 bash scripts/deploy_server9.sh  # research only
```

Host name, user, and port stay in **`~/.ssh/config`** only — scripts do not embed them.

Effective global batch (default): `4 × 8 GPUs × 2 accum = 64`.

The server launcher uses the stricter **book-disjoint** split. It is generated
once with `python scripts/make_book_holdout_splits.py`; do not report the older
random-verse split as a final benchmark because adjacent verses leak context.

## Local train (Mac / single GPU)

```bash
source .venv/bin/activate
python scripts/make_book_holdout_splits.py
python scripts/prepare_hf_dataset.py --data-dir data/aligned/book_holdout
python scripts/train_madlad.py \
  --model google/madlad400-3b-mt \
  --direction both --epochs 3 --batch-size 2 --grad-accum 8 \
  --confirm-restricted-data-rights
python scripts/evaluate_madlad.py --model-dir models/madlad-wobe-fr/final
```

## Zero-shot probe (pre-finetune)

```bash
python scripts/zero_shot_probe.py
```

## Project layout

```
data/           # corpora, audio, related languages
scripts/        # prepare / train / evaluate
models/         # checkpoints (created by training)
results/        # metrics + predictions
WOBE_MT_RESEARCH_REPORT.md
```

## Important

1. **Do not push Wobé scripture text to public Hugging Face** without Wycliffe permission.
2. Bible-only models will fail on news/chat — collect general-domain pairs next.
3. Target quality bar: MAFAND-style usefulness needs ~2k–10k non-Bible pairs on top of this baseline.
4. The previous `models/nllb-wobe-fr*` checkpoints used incorrect NLLB target
   routing and are retained only as debugging artifacts. Do not compare or
   deploy them.
