# Sweep: 1B SimNPO pruning
# Date: 2026-04-25
# Goal: validate weight delta prediction — SimNPO enrichment is high in ALL layer types
#       (attn_out 5.83x, mlp_gate_up 6.88x, mlp_down 6.55x at 10% sparsity),
#       so expect broader knowledge recovery than GradDiff across pruning levels.
# Comparison: GradDiff 10% pruning gave 0.061 -> 0.737; SimNPO baseline is 0.110.
#
# Run on any pod (1B fits on RTX 4090 or even CPU):
#   bash scripts/run_sweep.sh sweeps/2026-04-25-1b-simnpo-pruning.sh

MODEL="open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr2e-05_b4.5_a1_d1_g0.25_ep10"
RETAIN_LOGS="results/retain_baseline/tofu_Llama-3.2-1B-Instruct_retain90__none_None/TOFU_EVAL.json"

EXPERIMENTS=(
    "${MODEL}|prune|0.1|forget10|1b_simnpo_prune_10pct"
    "${MODEL}|prune|0.3|forget10|1b_simnpo_prune_30pct"
)
