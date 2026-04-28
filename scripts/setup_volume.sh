#!/usr/bin/env bash
# Run ONCE after first attaching a fresh network volume to a pod.
# After this, session scripts (session_a.sh, etc.) handle everything.
#
# Usage (SSH into pod, then):
#   bash /tmp/setup_volume.sh
#
# Required: set GITHUB_TOKEN before running, or create /workspace/.env first.
# Also create /workspace/.env with:
#   HF_TOKEN=...
#   HF_LLAMA_TOKEN=...
#   GITHUB_TOKEN=...
#   RUNPOD_API_KEY=...
#   RUNPOD_VOLUME_ID=...

set -e

for _env_file in /workspace/.env /root/.env; do
    if [ -f "$_env_file" ]; then
        export $(grep -v '^#' "$_env_file" | xargs)
        break
    fi
done

# ── UV: pin to network volume from the start ──────────────────────────────────
export UV_PYTHON_INSTALL_DIR=/workspace/.uv/python
export UV_CACHE_DIR=/workspace/.uv/cache
export UV_LINK_MODE=copy
mkdir -p /workspace/.uv/{bin,python,cache}

_UV_SRC=$(command -v uv 2>/dev/null || true)
if [ -z "$_UV_SRC" ]; then
    pip install uv --quiet
    _UV_SRC=$(command -v uv)
fi
cp "$_UV_SRC" /workspace/.uv/bin/uv
export PATH="/workspace/.uv/bin:$HOME/.local/bin:$PATH"

echo "=== Cloning repo to network volume ==="
REPO_URL="https://${GITHUB_TOKEN}@github.com/danieltennant/unlearning-compression.git"
WORK_DIR="/workspace/unlearning-compression"

if [ -d "$WORK_DIR/.git" ]; then
    echo "Repo already exists — pulling"
    cd "$WORK_DIR" && git pull
else
    git clone "$REPO_URL" "$WORK_DIR" && cd "$WORK_DIR"
fi

git config --global user.email "danieltennant@users.noreply.github.com"
git config --global user.name "Daniel Tennant"
git submodule update --init --recursive

echo "=== Syncing dependencies ==="
uv sync

echo ""
echo "=== Volume setup complete ==="
echo "    Python and uv cache are now on the volume — pod restarts will be fast."
echo "    Next: bash scripts/session_a.sh  (or whichever session is next)"
