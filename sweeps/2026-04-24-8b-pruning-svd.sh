# Sweep: 8B model pruning and SVD evals
# Date: 2026-04-24
# Goal: test whether pruning/SVD recover unlearned knowledge on 8B model
# Note: 1B model was destroyed at 30% pruning and any SVD — using lighter levels here

LOCAL_CKPT="/workspace/checkpoint_8b"
if [ -d "$LOCAL_CKPT" ]; then
    MODEL="${LOCAL_CKPT}"
else
    MODEL="${HF_REPO:-dtennant/tofu-llama-8b-graddiff-alpha1}"
fi
RETAIN_LOGS="results/retain_baseline_8b/tofu_Llama-3.1-8B-Instruct_retain90__none_None/TOFU_EVAL.json"

EXPERIMENTS=(
    "${MODEL}|prune|0.1|forget10|8b_graddiff_alpha1_prune_10pct"
    "${MODEL}|prune|0.3|forget10|8b_graddiff_alpha1_prune_30pct"
    "${MODEL}|svd|0.9|forget10|8b_graddiff_alpha1_svd_90pct"
    # SVD 80% omitted — 90% already breaks the model completely (model_utility=0.0)
    "${MODEL}|svd|0.9|forget10|8b_graddiff_alpha1_svd_90pct_skip_embed"
)
