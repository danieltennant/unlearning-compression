# Sweep: 1B activation-aware SVD (ASVD) eval
# Date: 2026-04-25
# Goal: validate ASVD implementation before running on H100 at 8B scale.
#       Naive SVD broke the 1B model at all tested retain ratios (90/70/50%).
#       ASVD should preserve utility at moderate retain ratios.
#
# Prerequisites:
#   calibrate_activations.py must have run and ASVD_ACTIVATION_STATS must be set.

MODEL="open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr1e-05_alpha1_epoch10"
RETAIN_LOGS="results/retain_baseline/tofu_Llama-3.2-1B-Instruct_retain90__none_None/TOFU_EVAL.json"

EXPERIMENTS=(
    "${MODEL}|asvd|0.9|forget10|1b_graddiff_alpha1_asvd_90pct"
    "${MODEL}|asvd|0.8|forget10|1b_graddiff_alpha1_asvd_80pct"
    "${MODEL}|asvd|0.7|forget10|1b_graddiff_alpha1_asvd_70pct"
)
