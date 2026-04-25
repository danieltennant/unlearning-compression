#!/usr/bin/env bash
# Session A: 1B SimNPO pruning + 1B full-model ceiling + 1B ASVD validation
# Estimated runtime: ~1.5 hrs base, ~2 hrs with buffer. RTX 4090 is sufficient.
#
# The ASVD runs here are a sanity check before committing H100 time to Session B.
# Naive SVD broke the 1B model at all tested retain ratios (model_utility = 0.0).
# ASVD should survive at 90/80/70% and ideally still recover some unlearned knowledge.
#
# Usage:
#   bash scripts/session_a.sh
#
# Required env vars (set in /workspace/.env or /root/.env):
#   HF_TOKEN        — HuggingFace token with read access
#   GITHUB_TOKEN    — GitHub PAT with Contents read/write

set -e

for env_file in /workspace/.env /root/.env; do
    if [ -f "$env_file" ]; then
        export $(grep -v '^#' "$env_file" | xargs)
        break
    fi
done

export HF_HOME=/workspace/.cache/huggingface
export PATH="$HOME/.local/bin:$PATH"

REPO_URL="https://${GITHUB_TOKEN}@github.com/danieltennant/unlearning-compression.git"
if [ -d "/workspace/unlearning-compression/.git" ]; then
    WORK_DIR="/workspace/unlearning-compression"
else
    WORK_DIR="/tmp/unlearning-compression"
fi
OPEN_UNLEARNING_DIR="$WORK_DIR/open-unlearning"

echo "=== Setting up environment ==="
pip install uv --quiet
export UV_LINK_MODE=copy

echo "=== Cloning/updating repo ==="
if [ -d "$WORK_DIR/.git" ]; then
    cd "$WORK_DIR" && git pull
else
    git clone "$REPO_URL" "$WORK_DIR" && cd "$WORK_DIR"
fi
cd "$WORK_DIR"
git config --global user.email "danieltennant@users.noreply.github.com"
git config --global user.name "Daniel Tennant"
git remote set-url origin "$REPO_URL"
git submodule update --init --recursive
UV_LINK_MODE=copy uv sync

echo "=== Authenticating HuggingFace ==="
uv run huggingface-cli login --token "$HF_TOKEN"

echo "=== Setting up TOFU data ==="
cd "$OPEN_UNLEARNING_DIR" && uv run python setup_data.py
cd "$WORK_DIR"
uv pip install -r open-unlearning/requirements.txt

RETAIN_LOGS_1B="results/retain_baseline/tofu_Llama-3.2-1B-Instruct_retain90__none_None/TOFU_EVAL.json"

echo "=== 1B retain baseline (reuse cached if present) ==="
if [ ! -f "$RETAIN_LOGS_1B" ]; then
    uv run python experiments/eval_compressed.py \
        --model_id open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90 \
        --compression none \
        --forget_split forget10 \
        --output_dir results/retain_baseline
    git add results/retain_baseline/
    git commit -m "Results: retain_baseline_1b" || echo "(nothing new)"
    git pull --rebase && git push
fi

echo "=== 1B full-model ceiling ==="
uv run python experiments/eval_compressed.py \
    --model_id open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
    --compression none \
    --forget_split forget10 \
    --retain_logs_path "$RETAIN_LOGS_1B" \
    --output_dir results/1b_full_model_ceiling
git add results/1b_full_model_ceiling/
git commit -m "Results: 1b_full_model_ceiling" || echo "(nothing new)"
git pull --rebase && git push

echo "=== 1B SimNPO pruning sweep ==="
bash scripts/run_sweep.sh sweeps/2026-04-25-1b-simnpo-pruning.sh

echo "=== ASVD calibration (1B GradDiff alpha1) ==="
mkdir -p calibration
uv run python experiments/calibrate_activations.py \
    --model_id "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr1e-05_alpha1_epoch10" \
    --output_path calibration/1b_graddiff_alpha1.pt \
    --n_samples 128

git add calibration/
git commit -m "Add ASVD calibration stats: 1B GradDiff alpha1" || echo "(nothing new)"
git pull --rebase && git push

echo "=== 1B ASVD sweep ==="
export ASVD_ACTIVATION_STATS="$WORK_DIR/calibration/1b_graddiff_alpha1.pt"
bash scripts/run_sweep.sh sweeps/2026-04-25-1b-asvd.sh

echo "=== Session A complete ==="
