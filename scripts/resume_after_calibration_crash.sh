#!/usr/bin/env bash
# Resume after calibrate_activations.py crashed (wrong TOFU split).
# Runs: 1B ASVD calibration + sweep, then full Session B (8B ASVD + SVD alignment).
#
# Usage:
#   bash scripts/resume_after_calibration_crash.sh 2>&1 | tee /workspace/resume_c.log

set -e
source "$(dirname "${BASH_SOURCE[0]}")/bootstrap.sh"

# ── 1B ASVD ──────────────────────────────────────────────────────────────────

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
    --full_model_id "open-unlearning/tofu_Llama-3.1-8B-Instruct_full" \
    --unlearned_model_id "dtennant/tofu-llama-8b-graddiff-alpha1" \
    --output_dir results/weight_delta_8b_graddiff_alpha1 \
    --compute_svd

git add results/weight_delta_8b_graddiff_alpha1/
git commit -m "Results: 8B GradDiff alpha1 weight delta with SVD alignment" || echo "(nothing new)"
git pull --rebase && git push

echo "=== All done ==="
