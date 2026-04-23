#!/usr/bin/env bash
# Run a sweep of compression experiments and commit results after each run.
#
# Edit the EXPERIMENTS array below to add/remove runs.
# Each entry is: "model_id|compression|level|forget_split|output_dir"
#
# Usage:
#   bash scripts/run_sweep.sh
#
# Requires session_start.sh to have been run first (env loaded, git configured).

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RETAIN_LOGS="$REPO_ROOT/results/retain_baseline/tofu_Llama-3.2-1B-Instruct_retain90__none_None/TOFU_EVAL.json"

if [ ! -f "$RETAIN_LOGS" ]; then
    echo "ERROR: retain logs not found at $RETAIN_LOGS"
    echo "Run the retain baseline first:"
    echo "  uv run python experiments/eval_compressed.py \\"
    echo "    --model_id open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90 \\"
    echo "    --compression none \\"
    echo "    --output_dir results/retain_baseline"
    exit 1
fi

# ---------------------------------------------------------------------------
# Experiment list — edit this to configure the sweep
# Format: "model_id|compression|level|forget_split|output_dir_label"
# ---------------------------------------------------------------------------
EXPERIMENTS=(
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr1e-05_alpha1_epoch10|none||forget10|graddiff_alpha1_baseline"
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr1e-05_alpha1_epoch10|quantize|4|forget10|graddiff_alpha1_quantize_4bit"
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr2e-05_b4.5_a1_d1_g0.25_ep10|none||forget10|simnpo_baseline"
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr2e-05_b4.5_a1_d1_g0.25_ep10|quantize|4|forget10|simnpo_quantize_4bit"
)
# ---------------------------------------------------------------------------

cd "$REPO_ROOT"

for entry in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r model_id compression level forget_split output_label <<< "$entry"

    echo ""
    echo "================================================================"
    echo "Running: $output_label"
    echo "  model:       $model_id"
    echo "  compression: $compression  level: ${level:-none}"
    echo "================================================================"

    # Build argument list
    args=(
        "--model_id" "$model_id"
        "--compression" "$compression"
        "--forget_split" "$forget_split"
        "--holdout_split" "${forget_split/forget/holdout}"
        "--retain_logs_path" "$RETAIN_LOGS"
        "--output_dir" "results/$output_label"
    )
    if [ -n "$level" ]; then
        args+=("--level" "$level")
    fi

    uv run python experiments/eval_compressed.py "${args[@]}"

    echo "--- Committing results ---"
    git add "results/$output_label/"
    git add results/  # picks up summary.csv updates
    git commit -m "Results: $output_label" || echo "(nothing new to commit)"
    git pull --rebase && git push
done

echo ""
echo "=== Sweep complete ==="
