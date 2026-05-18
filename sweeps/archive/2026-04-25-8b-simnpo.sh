# Sweep: 8B SimNPO full compression sweep
# Date: 2026-04-25
# Goal: full GradDiff-equivalent coverage for SimNPO at 8B scale
# Note: open-unlearning has no pre-trained 8B SimNPO checkpoint.
#       Run scripts/train_8b_simnpo.sh first, then this sweep.

LOCAL_CKPT="/workspace/checkpoint_8b_simnpo"
if [ -d "$LOCAL_CKPT" ]; then
    MODEL="${LOCAL_CKPT}"
else
    MODEL="${HF_REPO_SIMNPO:-dtennant/tofu-llama-8b-simnpo}"
fi
RETAIN_LOGS="results/retain_baseline_8b/tofu_Llama-3.1-8B-Instruct_retain90__none_None/TOFU_EVAL.json"

EXPERIMENTS=(
    "${MODEL}|none||forget10|8b_simnpo_baseline"
    "${MODEL}|quantize|4|forget10|8b_simnpo_quantize_4bit"
    "${MODEL}|quantize|8|forget10|8b_simnpo_quantize_8bit"
    "${MODEL}|prune|0.1|forget10|8b_simnpo_prune_10pct"
    "${MODEL}|prune|0.3|forget10|8b_simnpo_prune_30pct"
)
