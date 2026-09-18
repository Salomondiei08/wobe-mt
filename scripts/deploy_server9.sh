#!/usr/bin/env bash
# Deploy Wobé MT and train on a multi-GPU host.
#
# Authentication stays OUT of this file. Prefer an SSH key; password auth is a
# fallback for servers that have not yet been configured with one:
#
#   export SSH_HOST=server9     # optional; default uses Host alias from ~/.ssh/config
#   bash scripts/deploy_server9.sh
#
# Requires an SSH Host entry that already knows hostname/user/port (so this
# script never embeds them). Password mode additionally requires sshpass.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

HOST="${SSH_HOST:-server9}"
REMOTE_DIR="${REMOTE_DIR:-~/wobe-mt}"
MODEL="${MODEL:-google/madlad400-3b-mt}"
NPROC="${NPROC:-8}"
EPOCHS="${EPOCHS:-5}"
BATCH="${BATCH:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
MAX_LEN="${MAX_LEN:-128}"
LR="${LR:-2e-5}"
GPU_IDS="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
BASE_PYTHON="${BASE_PYTHON:-python3}"
REUSE_SYSTEM_TORCH="${REUSE_SYSTEM_TORCH:-0}"

# Password fallback: prefer file (agent never needs to read it), else env
# SSHPASS. SSH-key authentication is used automatically when no password is
# configured.
PASS_FILE="${SSHPASS_FILE:-/tmp/wobe_ssh_pass}"
if [[ -f "$PASS_FILE" ]]; then
  SSHPASS_CMD=(sshpass -f "$PASS_FILE")
  echo "Using password-file SSH authentication"
elif [[ -n "${SSHPASS:-}" ]]; then
  SSHPASS_CMD=(sshpass -e)
  echo "Using SSHPASS environment authentication"
else
  SSHPASS_CMD=()
  echo "Using configured SSH-key authentication"
fi

# Hostname/user/port/key are resolved only through the local SSH Host alias.
SSH_BASE=("${SSHPASS_CMD[@]}" ssh
  -o StrictHostKeyChecking=accept-new
  -o ConnectTimeout=20
  -o ServerAliveInterval=30
)
if (( ${#SSHPASS_CMD[@]} )); then
  SSH_BASE+=( -o PreferredAuthentications=password -o PubkeyAuthentication=no )
fi

if (( ${#SSHPASS_CMD[@]} )); then
  RSYNC_SSH="${SSHPASS_CMD[*]} ssh -o StrictHostKeyChecking=accept-new -o PreferredAuthentications=password -o PubkeyAuthentication=no"
else
  RSYNC_SSH="ssh -o StrictHostKeyChecking=accept-new"
fi

echo "=== Target host alias: $HOST | model=$MODEL | processes=$NPROC | CUDA devices=$GPU_IDS ==="
echo "=== (hostname/user/port come from your ~/.ssh/config — not this script) ==="

echo "=== Testing login ==="
"${SSH_BASE[@]}" "$HOST" 'echo OK; hostname; whoami; nvidia-smi -L; nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu --format=csv'

echo "=== Remote dir ==="
"${SSH_BASE[@]}" "$HOST" "mkdir -p $REMOTE_DIR"

echo "=== Rsync code + aligned corpus ==="
rsync -avz --progress \
  -e "$RSYNC_SSH" \
  --exclude '.venv' \
  --exclude 'models' \
  --exclude 'results' \
  --exclude 'data/audio_catalog' \
  --exclude 'data/related' \
  --exclude 'data/wobe_bible' \
  --exclude 'data/french_bible' \
  --exclude 'data/english_bible' \
  --exclude 'data/metadata' \
  --exclude 'data/sil' \
  --exclude '__pycache__' \
  --exclude '.git' \
  "$ROOT/" "$HOST:$REMOTE_DIR/"

echo "=== Multi-GPU train ==="
"${SSH_BASE[@]}" "$HOST" \
  env MODEL="$MODEL" NPROC="$NPROC" EPOCHS="$EPOCHS" BATCH="$BATCH" \
      GRAD_ACCUM="$GRAD_ACCUM" MAX_LEN="$MAX_LEN" LR="$LR" \
      CUDA_VISIBLE_DEVICES="$GPU_IDS" REMOTE_DIR="$REMOTE_DIR" \
      BASE_PYTHON="$BASE_PYTHON" REUSE_SYSTEM_TORCH="$REUSE_SYSTEM_TORCH" \
  bash -s <<'REMOTE'
set -euo pipefail
cd "$REMOTE_DIR"

# Large framework wheels and model files must stay alongside the experiment on
# the data volume. Some lab home directories are intentionally small.
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$REMOTE_DIR/.cache/pip}"
export HF_HOME="${HF_HOME:-$REMOTE_DIR/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TMPDIR="${TMPDIR:-$REMOTE_DIR/.tmp}"
mkdir -p "$PIP_CACHE_DIR" "$HF_HOME" "$TMPDIR"

# --- Create a working venv (servers often lack ensurepip / python3-venv) ---
rm -rf .venv
make_venv() {
  # 1) Normal venv
  if "$BASE_PYTHON" -m venv --system-site-packages .venv 2>/tmp/venv_err.txt; then
    return 0
  fi
  echo "$BASE_PYTHON -m venv failed:"
  cat /tmp/venv_err.txt || true

  # 2) venv without pip, then bootstrap pip
  if "$BASE_PYTHON" -m venv --without-pip --system-site-packages .venv 2>/tmp/venv_err2.txt; then
    # Match the bootstrap script to the server interpreter. The unversioned
    # endpoint has dropped Python 3.8 support on older lab hosts.
    PYVER=$("$BASE_PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    curl -fsSL "https://bootstrap.pypa.io/pip/${PYVER}/get-pip.py" -o /tmp/get-pip.py
    .venv/bin/python /tmp/get-pip.py
    return 0
  fi
  echo "venv --without-pip failed:"
  cat /tmp/venv_err2.txt || true

  # 3) user-level virtualenv package
  "$BASE_PYTHON" -m pip install --user virtualenv 2>/dev/null || true
  if "$BASE_PYTHON" -m virtualenv --system-site-packages .venv 2>/tmp/venv_err3.txt; then
    return 0
  fi
  echo "virtualenv failed:"
  cat /tmp/venv_err3.txt || true

  # 4) install system package (needs sudo password on server)
  PYVER=$("$BASE_PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
  echo "Trying: sudo apt-get install -y python${PYVER}-venv python3-venv python3-pip"
  if sudo -n true 2>/dev/null; then
    sudo apt-get update -qq
    sudo apt-get install -y "python${PYVER}-venv" python3-venv python3-pip || true
  else
    echo "sudo needs a password on the server. Run once on server9:"
    echo "  sudo apt-get update && sudo apt-get install -y python${PYVER}-venv python3-venv python3-pip"
    echo "Then re-run: bash scripts/deploy_server9.sh"
    exit 1
  fi
  rm -rf .venv
  "$BASE_PYTHON" -m venv --system-site-packages .venv
}

make_venv
source .venv/bin/activate
python -m pip install -U pip wheel
if [[ "$REUSE_SYSTEM_TORCH" != "1" ]]; then
  python -m pip install torch --index-url https://download.pytorch.org/whl/cu124 \
    || python -m pip install torch --index-url https://download.pytorch.org/whl/cu121 \
    || python -m pip install torch
fi
python -m pip install transformers datasets accelerate sacrebleu evaluate sentencepiece protobuf tqdm pandas numpy

python - <<'PY'
import torch
print("cuda:", torch.cuda.is_available())
print("gpus:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  {i}: {p.name} {p.total_memory/1e9:.1f}GB")
assert torch.cuda.is_available()
PY

DATA_DIR="${DATA_DIR:-data/aligned/book_holdout}"
if [[ "$DATA_DIR" == "data/aligned/book_holdout" ]]; then
  python scripts/make_book_holdout_splits.py
fi
python scripts/prepare_hf_dataset.py --data-dir "$DATA_DIR"
mkdir -p results models logs
# Stop only a previous Wobé job from this deployment directory; never match
# unrelated training jobs that another server user may be running.
pkill -f 'wobe-mt/scripts/train_(nllb|madlad)\.py' 2>/dev/null || true
sleep 1

FP16_FLAG="--fp16"
BF16_FLAG=""
if python -c "import torch; assert torch.cuda.is_bf16_supported()" 2>/dev/null; then
  BF16_FLAG="--bf16"
  FP16_FLAG=""
fi

GC_FLAG=""
case "$MODEL" in
  *3.3B*|*3B*|*3b*) GC_FLAG="--gradient-checkpointing" ;;
esac

TRAINER="scripts/train_madlad.py"
OUTPUT_DIR="models/madlad-wobe-fr"
if [[ "$MODEL" == facebook/nllb-* ]]; then
  TRAINER="scripts/train_nllb.py"
  OUTPUT_DIR="models/nllb-wobe-fr"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

nohup torchrun --standalone --nproc_per_node="$NPROC" \
  "$TRAINER" \
  --model "$MODEL" \
  --direction both \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH" \
  --grad-accum "$GRAD_ACCUM" \
  --max-length "$MAX_LEN" \
  --lr "$LR" \
  --confirm-restricted-data-rights \
  $FP16_FLAG $BF16_FLAG $GC_FLAG \
  --output-dir "$OUTPUT_DIR" \
  > results/full_train.log 2>&1 &

echo "TRAIN_PID=$!"
sleep 4
tail -n 40 results/full_train.log || true
REMOTE

echo ""
echo "Deployed. Monitor from your machine:"
echo "  sshpass -e ssh $HOST 'tail -f ~/wobe-mt/results/full_train.log'"
echo "  sshpass -e ssh $HOST 'nvidia-smi'"
