#!/usr/bin/env bash
# Run a sweep of compression experiments and commit results after each run.
#
# Usage:
#   bash scripts/run_sweep.sh sweeps/2026-04-23-pruning-svd.sh
#
# The sweep file defines an EXPERIMENTS array. See sweeps/ for examples.
# Requires session_start.sh to have been run first (env loaded, git configured).

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RETAIN_LOGS="$REPO_ROOT/results/retain_baseline/tofu_Llama-3.2-1B-Instruct_retain90__none_None/TOFU_EVAL.json"
SWEEP_FILE="${1:-}"

if [ -z "$SWEEP_FILE" ]; then
    echo "Usage: bash scripts/run_sweep.sh <sweep_file>"
    echo "Available sweeps:"
    ls "$REPO_ROOT/sweeps/"
    exit 1
fi

if [ ! -f "$REPO_ROOT/$SWEEP_FILE" ]; then
    echo "ERROR: sweep file not found: $SWEEP_FILE"
    exit 1
fi

if [ ! -f "$RETAIN_LOGS" ]; then
    echo "ERROR: retain logs not found at $RETAIN_LOGS"
    echo "Run the retain baseline first:"
    echo "  uv run python experiments/eval_compressed.py \\"
    echo "    --model_id open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90 \\"
    echo "    --compression none \\"
    echo "    --output_dir results/retain_baseline"
    exit 1
fi

# Load experiments from the sweep file
source "$REPO_ROOT/$SWEEP_FILE"

echo "=== Running sweep: $SWEEP_FILE (${#EXPERIMENTS[@]} experiments) ==="

cd "$REPO_ROOT"

for entry in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r model_id compression level forget_split output_label <<< "$entry"

    echo ""
    echo "================================================================"
    echo "Running: $output_label"
    echo "  model:       $model_id"
    echo "  compression: $compression  level: ${level:-none}"
    echo "================================================================"

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
    git add results/
    git commit -m "Results: $output_label" || echo "(nothing new to commit)"
    git pull --rebase && git push
done

echo ""
echo "=== Sweep complete ==="
