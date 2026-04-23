#!/usr/bin/env bash
# Run this immediately after SSH-ing into a RunPod pod.
# Assumes network volume is mounted at /workspace.
#
# Usage:
#   bash /workspace/unlearning-compression/scripts/session_start.sh

set -e

echo "=== Installing uv ==="
pip install uv --quiet
export UV_LINK_MODE=copy

echo "=== Pulling latest code ==="
cd /workspace/unlearning-compression
git pull

echo "=== Syncing dependencies ==="
uv sync

echo "=== Loading environment ==="
if [ -f /workspace/.env ]; then
    export $(grep -v '^#' /workspace/.env | xargs)
    echo "Loaded .env"
else
    echo "WARNING: /workspace/.env not found — HuggingFace and GitHub tokens will be missing"
fi

echo "=== Configuring git identity ==="
git config --global user.email "danieltennant@users.noreply.github.com"
git config --global user.name "Daniel Tennant"

echo ""
echo "=== Ready. Next: run scripts/run_sweep.sh or individual experiments ==="
echo "    uv run python experiments/eval_compressed.py --help"
