#!/usr/bin/env bash
# Run ONCE after first attaching a fresh network volume to a pod.
# After this, session_start.sh handles subsequent sessions.
#
# Usage (SSH into pod, then):
#   bash /tmp/setup_volume.sh
#
# Required: set GITHUB_TOKEN before running, or the clone will use HTTPS
# without auth and fail on private repos.
#   export GITHUB_TOKEN=<your PAT>
#
# Also create /workspace/.env with:
#   HF_TOKEN=...
#   HF_LLAMA_TOKEN=...
#   GITHUB_TOKEN=...
#   RUNPOD_API_KEY=...
#   RUNPOD_VOLUME_ID=...

set -e

REPO_URL="${GITHUB_TOKEN:+https://${GITHUB_TOKEN}@}github.com/danieltennant/unlearning-compression.git"
WORK_DIR="/workspace/unlearning-compression"

echo "=== Installing uv ==="
pip install uv --quiet
export UV_LINK_MODE=copy

echo "=== Cloning repo to network volume ==="
if [ -d "$WORK_DIR" ]; then
    echo "Repo already exists at $WORK_DIR — pulling instead"
    cd "$WORK_DIR"
    git pull
else
    git clone "$REPO_URL" "$WORK_DIR"
    cd "$WORK_DIR"
fi

git config --global user.email "danieltennant@users.noreply.github.com"
git config --global user.name "Daniel Tennant"
git submodule update --init --recursive

echo "=== Syncing dependencies ==="
UV_LINK_MODE=copy uv sync || uv sync

echo ""
echo "=== Volume setup complete ==="
echo "    Next: create /workspace/.env with your tokens"
echo "    Then: bash scripts/session_start.sh  (for future sessions)"
