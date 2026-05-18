# Sweep: 8-bit quantization + SimNPO confirmation
# Date: 2026-04-23
# Goal: (1) compare 8-bit vs 4-bit recovery magnitude on GradDiff alpha1
#        (2) confirm quantization recovery generalises to SimNPO method

EXPERIMENTS=(
    # 8-bit quantization on GradDiff alpha1 (compare to 4-bit result: forget_Q_A_Prob 0.061 → 0.359)
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr1e-05_alpha1_epoch10|quantize|8|forget10|graddiff_alpha1_quantize_8bit"
    # SimNPO — 8-bit quantization (baseline already done: forget_Q_A_Prob = 0.110)
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr2e-05_b4.5_a1_d1_g0.25_ep10|quantize|8|forget10|simnpo_quantize_8bit"
)
