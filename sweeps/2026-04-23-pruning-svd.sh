# Sweep: pruning and SVD truncation on best checkpoint
# Date: 2026-04-23
# Goal: test whether structural compression (pruning, SVD) recovers suppressed knowledge
#       the way 4-bit quantization does (Zhang et al. effect)
# Checkpoint: GradDiff alpha1 — the clearest unlearning result (forget_Q_A_Prob = 0.061)

EXPERIMENTS=(
    # Magnitude pruning — sparsity = fraction of weights zeroed out
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr1e-05_alpha1_epoch10|prune|0.3|forget10|graddiff_alpha1_prune_0.3"
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr1e-05_alpha1_epoch10|prune|0.5|forget10|graddiff_alpha1_prune_0.5"
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr1e-05_alpha1_epoch10|prune|0.7|forget10|graddiff_alpha1_prune_0.7"
    # SVD truncation — retain_ratio = fraction of singular values kept per layer
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr1e-05_alpha1_epoch10|svd|0.9|forget10|graddiff_alpha1_svd_0.9"
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr1e-05_alpha1_epoch10|svd|0.7|forget10|graddiff_alpha1_svd_0.7"
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr1e-05_alpha1_epoch10|svd|0.5|forget10|graddiff_alpha1_svd_0.5"
)
