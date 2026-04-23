# Sweep: initial quantization experiments
# Date: 2026-04-23
# Goal: replicate Guo et al. recovery effect on TOFU with 4-bit quantization
# Result: confirmed — GradDiff alpha1 shows 6x forget_Q_A_Prob recovery after 4-bit quant

EXPERIMENTS=(
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr1e-05_alpha5_epoch10|none||forget10|baseline"
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr1e-05_alpha5_epoch10|quantize|4|forget10|quantize_4bit"
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr2e-05_beta0.5_alpha1_epoch10|none||forget10|npo_baseline"
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr2e-05_beta0.5_alpha1_epoch10|quantize|4|forget10|npo_quantize_4bit"
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr1e-05_alpha1_epoch10|none||forget10|graddiff_alpha1_baseline"
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr1e-05_alpha1_epoch10|quantize|4|forget10|graddiff_alpha1_quantize_4bit"
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr2e-05_b4.5_a1_d1_g0.25_ep10|none||forget10|simnpo_baseline"
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr2e-05_b4.5_a1_d1_g0.25_ep10|quantize|4|forget10|simnpo_quantize_4bit"
)
