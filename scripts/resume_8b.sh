#!/usr/bin/env bash
# Resume from 8B ASVD calibration onwards.
# Run after 1B sweep is complete.
# Usage: bash scripts/resume_8b.sh 2>&1 | tee /workspace/resume_8b.log

set -e
source "$(dirname "${BASH_SOURCE[0]}")/bootstrap.sh"

echo "=== ASVD calibration (8B GradDiff alpha1) ==="
mkdir -p calibration
PYTHONUNBUFFERED=1 uv run python -u experiments/calibrate_activations.py \
    --model_id dtennant/tofu-llama-8b-graddiff-alpha1 \
    --output_path calibration/8b_graddiff_alpha1.pt \
    --n_samples 128

git add calibration/
git commit -m "Add ASVD calibration stats: 8B GradDiff alpha1" || echo "(nothing new)"
git pull --rebase && git push

echo "=== Running 8B ASVD sweep ==="
export ASVD_ACTIVATION_STATS="$WORK_DIR/calibration/8b_graddiff_alpha1.pt"
bash scripts/run_sweep.sh sweeps/2026-04-25-8b-asvd.sh

echo "=== 8B SVD alignment analysis ==="
PYTHONUNBUFFERED=1 uv run python -u experiments/weight_delta_analysis.py \
    --full_model_id "open-unlearning/tofu_Llama-3.1-8B-Instruct_full" \
    --unlearned_model_id "dtennant/tofu-llama-8b-graddiff-alpha1" \
    --output_dir results/weight_delta_8b_graddiff_alpha1 \
    --compute_svd

git add results/weight_delta_8b_graddiff_alpha1/
git commit -m "Results: 8B GradDiff alpha1 weight delta with SVD alignment" || echo "(nothing new)"
git pull --rebase && git push

echo "=== 8B phase complete ==="
