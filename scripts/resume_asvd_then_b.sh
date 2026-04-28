#!/usr/bin/env bash
# Resume from where Session A stopped (ASVD calibration), then run Session B.
# Use this on a fresh pod with the network volume already mounted.
# Estimated runtime: ~4 hrs (1B ASVD ~30 min + Session B ~3.5 hrs base). H100 preferred.
#
# Usage:
#   bash scripts/resume_asvd_then_b.sh

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
WORK_DIR="/workspace/unlearning-compression"
OPEN_UNLEARNING_DIR="$WORK_DIR/open-unlearning"

echo "=== Installing uv ==="
pip install uv --quiet
export UV_LINK_MODE=copy

echo "=== Updating repo ==="
cd "$WORK_DIR"
git remote set-url origin "$REPO_URL"
git pull
git config --global user.email "danieltennant@users.noreply.github.com"
git config --global user.name "Daniel Tennant"
uv sync

echo "=== Authenticating HuggingFace ==="
uv run huggingface-cli login --token "$HF_TOKEN"

echo "=== Setting up TOFU data ==="
cd "$OPEN_UNLEARNING_DIR" && uv run python setup_data.py
cd "$WORK_DIR"
uv pip install -r open-unlearning/requirements.txt
uv pip install "bitsandbytes>=0.45.0"

# ── Remaining Session A: 1B ASVD ─────────────────────────────────────────────

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

# ── Session B: 8B ASVD + SVD alignment ───────────────────────────────────────

echo "=== ASVD calibration (8B GradDiff alpha1) ==="
uv run python experiments/calibrate_activations.py \
    --model_id dtennant/tofu-llama-8b-graddiff-alpha1 \
    --output_path calibration/8b_graddiff_alpha1.pt \
    --n_samples 128

git add calibration/
git commit -m "Add ASVD calibration stats: 8B GradDiff alpha1" || echo "(nothing new)"
git pull --rebase && git push

echo "=== 8B ASVD sweep ==="
export ASVD_ACTIVATION_STATS="$WORK_DIR/calibration/8b_graddiff_alpha1.pt"
bash scripts/run_sweep.sh sweeps/2026-04-25-8b-asvd.sh

echo "=== 8B SVD alignment analysis ==="
uv run python experiments/weight_delta_analysis.py \
    --full_model_id open-unlearning/tofu_Llama-3.1-8B-Instruct_full \
    --unlearned_model_id dtennant/tofu-llama-8b-graddiff-alpha1 \
    --output_dir results/weight_delta_8b_graddiff_alpha1 \
    --compute_svd

git add results/weight_delta_8b_graddiff_alpha1/
git commit -m "Results: 8B GradDiff alpha1 weight delta with SVD alignment" || echo "(nothing new)"
git pull --rebase && git push

echo "=== All done ==="
