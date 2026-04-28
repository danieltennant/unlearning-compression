# Sweep: Cholesky-whitened SVD baseline validation
# Date: 2026-04-28
# Goal: validate that the fixed SVD implementation preserves model utility
#       BEFORE running on any unlearned checkpoint.
#
# Run on retain90 (never saw forget authors) — if model_utility collapses here,
# the implementation is broken regardless of unlearning.
#
# Prerequisites:
#   calibrate_activations.py --full_cov must have run and SVD_COV_STATS must be set.

MODEL="open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90"
RETAIN_LOGS="results/retain_baseline/tofu_Llama-3.2-1B-Instruct_retain90__none_None/TOFU_EVAL.json"

EXPERIMENTS=(
    "${MODEL}|svd_chol|0.9|forget10|1b_retain90_svdchol_90pct"
    "${MODEL}|svd_chol|0.8|forget10|1b_retain90_svdchol_80pct"
    "${MODEL}|svd_chol|0.7|forget10|1b_retain90_svdchol_70pct"
)
