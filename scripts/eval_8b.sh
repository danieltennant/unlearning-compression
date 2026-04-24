#!/usr/bin/env bash
# Run 8B compression evals on a pod with an attached network volume.
# Uses /workspace/unlearning-compression if available (faster — no re-clone).
# Falls back to /tmp if the volume isn't mounted.
#
# Usage:
#   bash scripts/eval_8b.sh
#
# Required env vars (set in /workspace/.env or /root/.env):
#   HF_TOKEN      — HuggingFace token with read access
#   GITHUB_TOKEN  — GitHub PAT with Contents read/write

set -e

# Load tokens from .env
for env_file in /workspace/.env /root/.env; do
    if [ -f "$env_file" ]; then
        export $(grep -v '^#' "$env_file" | xargs)
        break
    fi
done

export HF_HOME=/workspace/.cache/huggingface

REPO_URL="https://${GITHUB_TOKEN}@github.com/danieltennant/unlearning-compression.git"
# Prefer the volume-resident clone; fall back to /tmp on fresh pods without a volume
if [ -d "/workspace/unlearning-compression/.git" ]; then
    WORK_DIR="/workspace/unlearning-compression"
else
    WORK_DIR="/tmp/unlearning-compression"
fi
OPEN_UNLEARNING_DIR="$WORK_DIR/open-unlearning"

echo "=== Setting up environment (WORK_DIR=$WORK_DIR) ==="
pip install uv --quiet
export UV_LINK_MODE=copy
export PATH="$HOME/.local/bin:$PATH"

echo "=== Cloning/updating repo ==="
if [ -d "$WORK_DIR/.git" ]; then
    cd "$WORK_DIR"
    git pull
else
    git clone "$REPO_URL" "$WORK_DIR"
    cd "$WORK_DIR"
fi
git config --global user.email "danieltennant@users.noreply.github.com"
git config --global user.name "Daniel Tennant"
git remote set-url origin "$REPO_URL"
git submodule update --init --recursive
UV_LINK_MODE=copy uv sync

echo "=== Authenticating HuggingFace ==="
uv run huggingface-cli login --token "$HF_TOKEN"

echo "=== Setting up TOFU data ==="
cd "$OPEN_UNLEARNING_DIR"
uv run python setup_data.py

echo "=== Installing open-unlearning dependencies ==="
cd "$WORK_DIR"
uv pip install -r open-unlearning/requirements.txt

echo "=== Retain baseline (8B) ==="
# HF cache at /workspace/.cache/huggingface is warm from training — no re-download needed
uv run python experiments/eval_compressed.py \
    --model_id open-unlearning/tofu_Llama-3.1-8B-Instruct_retain90 \
    --compression none \
    --forget_split forget10 \
    --holdout_split holdout10 \
    --output_dir results/retain_baseline_8b

git add results/retain_baseline_8b/
git commit -m "Results: retain_baseline_8b" || echo "(nothing new to commit)"
git pull --rebase && git push

echo "=== Running 8B quantization sweep ==="
bash scripts/run_sweep.sh sweeps/2026-04-23-8b.sh

echo "=== Running 8B pruning/SVD sweep ==="
bash scripts/run_sweep.sh sweeps/2026-04-24-8b-pruning-svd.sh

echo "=== Full model ceiling eval (tofu_Llama-3.1-8B-Instruct_full) ==="
RETAIN_LOGS="results/retain_baseline_8b/tofu_Llama-3.1-8B-Instruct_retain90__none_None/TOFU_EVAL.json"
uv run python experiments/eval_compressed.py \
    --model_id open-unlearning/tofu_Llama-3.1-8B-Instruct_full \
    --compression none \
    --forget_split forget10 \
    --holdout_split holdout10 \
    --retain_logs_path "$RETAIN_LOGS" \
    --output_dir results/8b_full_model_ceiling
git add results/8b_full_model_ceiling/
git commit -m "Results: 8b_full_model_ceiling" || echo "(nothing new to commit)"
git pull --rebase && git push

echo "=== Done ==="
