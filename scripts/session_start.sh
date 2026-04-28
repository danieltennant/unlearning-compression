#!/usr/bin/env bash
# Quick session start for interactive/manual work.
# Sets up the environment without running any experiments.
#
# Usage (SSH into pod, then):
#   source /workspace/unlearning-compression/scripts/session_start.sh
#
# After sourcing, your shell has the right PATH, uv, HF_HOME, etc.
# Run experiments manually: uv run python experiments/eval_compressed.py --help

set -e
source "$(dirname "${BASH_SOURCE[0]}")/bootstrap.sh"

echo ""
echo "=== Ready ==="
echo "    uv run python experiments/eval_compressed.py --help"
echo "    bash scripts/run_sweep.sh sweeps/<file>.sh"
