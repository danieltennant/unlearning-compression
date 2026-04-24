#!/usr/bin/env bash
# Run 8B compression evals from scratch on a fresh pod.
# Pulls the trained checkpoint from HuggingFace (no network volume needed).
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
WORK_DIR="/tmp/unlearning-compression"
OPEN_UNLEARNING_DIR="$WORK_DIR/open-unlearning"

echo "=== Setting up environment ==="
pip install uv --quiet
export UV_LINK_MODE=copy
export PATH="$HOME/.local/bin:$PATH"

echo "=== Cloning repo ==="
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
uv run python experiments/eval_compressed.py \
    --model_id open-unlearning/tofu_Llama-3.1-8B-Instruct_retain90 \
    --compression none \
    --forget_split forget10 \
    --holdout_split holdout10 \
    --output_dir results/retain_baseline_8b

git add results/retain_baseline_8b/
git commit -m "Results: retain_baseline_8b" || echo "(nothing new to commit)"
git pull --rebase && git push

echo "=== Running 8B sweep ==="
bash scripts/run_sweep.sh sweeps/2026-04-23-8b.sh

echo "=== Done ==="
