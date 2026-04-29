#!/usr/bin/env bash
# Validate Cholesky-whitened SVD on the retain90 baseline BEFORE running
# on any unlearned checkpoint.
#
# This is the mandatory sanity check. We expect model_utility to stay
# above ~0.5 at 90% retain ratio if the implementation is correct.
# If utility collapses here, the implementation is broken — stop and fix.
#
# Cost estimate: RTX 4090, ~2h ($1.50–2.00).
#
# Usage:
#   bash scripts/validate_svd_cholesky.sh 2>&1 | tee /workspace/validate_svd.log

set -e
source "$(dirname "${BASH_SOURCE[0]}")/bootstrap.sh"

MODEL="open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90"
RETAIN_LOGS="results/retain_baseline/tofu_Llama-3.2-1B-Instruct_retain90__none_None/TOFU_EVAL.json"

# ── Step 1: Retain baseline (reuse if present) ────────────────────────────────
if [ ! -f "$RETAIN_LOGS" ]; then
    echo "=== Retain baseline (required for forget_quality metric) ==="
    PYTHONUNBUFFERED=1 uv run python -u experiments/eval_compressed.py \
        --model_id "$MODEL" \
        --compression none \
        --forget_split forget10 \
        --output_dir results/retain_baseline
    git add results/retain_baseline/ || true
    git commit -m "Results: retain_baseline" || echo "(nothing new)"
    git pull --rebase && git push || echo "(git push failed — check GITHUB_TOKEN in /workspace/.env)"
else
    echo "=== Retain baseline already present, skipping ==="
fi

# ── Step 2: Calibrate activations (full covariance) ───────────────────────────
COV_PATH="$WORK_DIR/calibration/1b_retain90_fullcov.pt"
if [ ! -f "$COV_PATH" ]; then
    echo "=== Calibrating activations (full covariance, retain90) ==="
    mkdir -p calibration
    PYTHONUNBUFFERED=1 uv run python -u experiments/calibrate_activations.py \
        --model_id "$MODEL" \
        --output_path "$COV_PATH" \
        --n_samples 128 \
        --full_cov
    git add calibration/
    git commit -m "Add Cholesky calibration stats: 1B retain90" || echo "(nothing new)"
    git pull --rebase && git push || echo "(git push failed — check GITHUB_TOKEN in /workspace/.env)"
else
    echo "=== Covariance stats already present at $COV_PATH, skipping calibration ==="
fi

# ── Step 3: Sweep ─────────────────────────────────────────────────────────────
echo "=== Running Cholesky SVD baseline validation sweep ==="
export SVD_COV_STATS="$COV_PATH"
bash scripts/run_sweep.sh sweeps/2026-04-28-svd-chol-baseline.sh

echo "=== Baseline validation complete ==="
echo ""
echo "Check results/1b_retain90_svdchol_*/TOFU_EVAL.json for model_utility."
echo "Expected: model_utility > 0.5 at 90% retain ratio if implementation is correct."
echo "If utility collapses at 90%, investigate before proceeding to unlearned models."
