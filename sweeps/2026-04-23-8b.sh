# Sweep: 8B model compression evals
# Date: 2026-04-23
# Goal: replicate 1B quantization recovery result on Llama-3.1-8B-Instruct
# Checkpoint: trained via scripts/train_8b.sh, pushed to HF_REPO

# Set HF_REPO in env before running train_8b.sh
HF_REPO="${HF_REPO:-danieltennant/tofu-llama-8b-graddiff-alpha1}"

EXPERIMENTS=(
    "${HF_REPO}|none||forget10|8b_graddiff_alpha1_baseline"
    "${HF_REPO}|quantize|4|forget10|8b_graddiff_alpha1_quantize_4bit"
    "${HF_REPO}|quantize|8|forget10|8b_graddiff_alpha1_quantize_8bit"
)
