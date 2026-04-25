#!/usr/bin/env bash
# Session B: 8B ASVD sweep + 8B SVD alignment analysis
# Estimated runtime: ~3.5 hrs base, ~5.5 hrs with buffer. Requires H100.
#
# Usage:
#   bash scripts/session_b.sh
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
uv pip install "bitsandbytes>=0.45.0"

echo "=== ASVD calibration (8B GradDiff alpha1) ==="
mkdir -p calibration
uv run python experiments/calibrate_activations.py \
    --model_id dtennant/tofu-llama-8b-graddiff-alpha1 \
    --output_path calibration/8b_graddiff_alpha1.pt \
    --n_samples 128

git add calibration/
git commit -m "Add ASVD calibration stats: 8B GradDiff alpha1" || echo "(nothing new)"
git pull --rebase && git push

echo "=== Running 8B ASVD sweep ==="
export ASVD_ACTIVATION_STATS="$WORK_DIR/calibration/8b_graddiff_alpha1.pt"
bash scripts/run_sweep.sh sweeps/2026-04-25-8b-asvd.sh

echo "=== 8B SVD alignment analysis (weight delta + --compute_svd) ==="
# Both 8B models fit in H100 VRAM (80GB) — load on CPU to avoid CUDA overhead
FULL_MODEL="open-unlearning/tofu_Llama-3.1-8B-Instruct_full"
UNLEARNED_MODEL="dtennant/tofu-llama-8b-graddiff-alpha1"
uv run python experiments/weight_delta_analysis.py \
    --full_model_id "$FULL_MODEL" \
    --unlearned_model_id "$UNLEARNED_MODEL" \
    --output_dir results/weight_delta_8b_graddiff_alpha1 \
    --compute_svd

git add results/weight_delta_8b_graddiff_alpha1/
git commit -m "Results: 8b GradDiff alpha1 weight delta with SVD alignment" || echo "(nothing new)"
git pull --rebase && git push

echo "=== Session B complete ==="
