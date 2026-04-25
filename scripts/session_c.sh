#!/usr/bin/env bash
# Session C: 8B SimNPO training + full compression sweep + weight delta analysis
# Estimated runtime: ~4 hrs base, ~5.5 hrs with buffer. Requires H100.
#
# Usage:
#   bash scripts/session_c.sh
#
# Required env vars (set in /workspace/.env or /root/.env):
#   HF_TOKEN        — HuggingFace token with read/write access
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

echo "=== Training 8B SimNPO ==="
bash scripts/train_8b_simnpo.sh

git add results/ || true
git commit -m "Checkpoint: 8B SimNPO trained" || echo "(nothing to commit)"
git pull --rebase && git push

echo "=== Retain baseline 8B (reuse cached if present) ==="
RETAIN_LOGS="results/retain_baseline_8b/tofu_Llama-3.1-8B-Instruct_retain90__none_None/TOFU_EVAL.json"
if [ ! -f "$RETAIN_LOGS" ]; then
    uv run python experiments/eval_compressed.py \
        --model_id open-unlearning/tofu_Llama-3.1-8B-Instruct_retain90 \
        --compression none \
        --forget_split forget10 \
        --output_dir results/retain_baseline_8b
    git add results/retain_baseline_8b/
    git commit -m "Results: retain_baseline_8b" || echo "(nothing new)"
    git pull --rebase && git push
fi

echo "=== Running 8B SimNPO compression sweep ==="
bash scripts/run_sweep.sh sweeps/2026-04-25-8b-simnpo.sh

echo "=== 8B SimNPO weight delta analysis ==="
FULL_MODEL="open-unlearning/tofu_Llama-3.1-8B-Instruct_full"
SIMNPO_CKPT="/workspace/checkpoint_8b_simnpo"
uv run python experiments/weight_delta_analysis.py \
    --full_model_id "$FULL_MODEL" \
    --unlearned_model_id "$SIMNPO_CKPT" \
    --output_dir results/weight_delta_8b_simnpo

git add results/weight_delta_8b_simnpo/
git commit -m "Results: 8b_simnpo weight delta analysis" || echo "(nothing new)"
git pull --rebase && git push

echo "=== Session C complete ==="
