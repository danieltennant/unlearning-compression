# Sweep: 8B activation-aware SVD (ASVD) eval
# Date: 2026-04-25
# Goal: test whether ASVD recovers unlearned knowledge while preserving utility
#       better than naive SVD (which gave model_utility=0.0 at 90% retain ratio)
#
# Prerequisites:
#   1. Run calibrate_activations.py to produce ASVD_ACTIVATION_STATS
#   2. Export ASVD_ACTIVATION_STATS=/path/to/calibration/8b_graddiff_alpha1.pt

LOCAL_CKPT="/workspace/checkpoint_8b"
if [ -d "$LOCAL_CKPT" ]; then
    MODEL="${LOCAL_CKPT}"
else
    MODEL="${HF_REPO:-dtennant/tofu-llama-8b-graddiff-alpha1}"
fi
RETAIN_LOGS="results/retain_baseline_8b/tofu_Llama-3.1-8B-Instruct_retain90__none_None/TOFU_EVAL.json"

EXPERIMENTS=(
    # Moderate truncation: should survive if calibration is accurate
    "${MODEL}|asvd|0.9|forget10|8b_graddiff_alpha1_asvd_90pct"
    "${MODEL}|asvd|0.8|forget10|8b_graddiff_alpha1_asvd_80pct"
    # Aggressive truncation: will this still recover knowledge?
    "${MODEL}|asvd|0.7|forget10|8b_graddiff_alpha1_asvd_70pct"
)
