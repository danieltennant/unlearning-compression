# Sweep: 8B model compression evals
# Date: 2026-04-23
# Goal: replicate 1B quantization recovery result on Llama-3.1-8B-Instruct
# Checkpoint: trained via scripts/train_8b.sh, pushed to HF_REPO

# Use local checkpoint if available (avoids 16GB download), else fall back to HF
LOCAL_CKPT="/workspace/checkpoint_8b"
if [ -d "$LOCAL_CKPT" ]; then
    MODEL="${LOCAL_CKPT}"
else
    MODEL="${HF_REPO:-dtennant/tofu-llama-8b-graddiff-alpha1}"
fi
RETAIN_LOGS="results/retain_baseline_8b/tofu_Llama-3.1-8B-Instruct_retain90__none_None/TOFU_EVAL.json"

EXPERIMENTS=(
    "${MODEL}|none||forget10|8b_graddiff_alpha1_baseline"
    "${MODEL}|quantize|4|forget10|8b_graddiff_alpha1_quantize_4bit"
    "${MODEL}|quantize|8|forget10|8b_graddiff_alpha1_quantize_8bit"
)
