#!/usr/bin/env bash
# Source this at the top of every session script.
#   source "$(dirname "${BASH_SOURCE[0]}")/bootstrap.sh"
#
# What it does (all idempotent):
#   1. Load tokens from /workspace/.env or /root/.env
#   2. Point UV_PYTHON_INSTALL_DIR + UV_CACHE_DIR at the network volume
#      so Python and package downloads survive pod restarts
#   3. Cache the uv binary on the volume (RunPod images ship it, but path varies)
#   4. git pull + submodule update + uv sync
#   5. Install open-unlearning deps only when ninja is missing
#   6. Authenticate HuggingFace
#   7. Run setup_data.py (idempotent — downloads TOFU eval data to local disk)
#
# HuggingFace model weights are intentionally kept on LOCAL disk (not the volume).
# Network volume reads at ~3MB/s; re-downloading an 8B model from HF CDN takes
# ~5 min vs ~90 min from the volume. HF_HOME is left at its default (~/.cache/hf).
#
# Exports: WORK_DIR, OPEN_UNLEARNING_DIR, REPO_URL

set -e

WORK_DIR="/workspace/unlearning-compression"
OPEN_UNLEARNING_DIR="$WORK_DIR/open-unlearning"

# ── Tokens ────────────────────────────────────────────────────────────────────
for _env_file in /workspace/.env /root/.env; do
    if [ -f "$_env_file" ]; then
        export $(grep -v '^#' "$_env_file" | xargs)
        break
    fi
done

# ── UV: store Python, cache, and binary on the network volume ─────────────────
# HF_HOME is NOT set here — models download to local disk (~/.cache/huggingface).
# Reading from the volume at ~3MB/s is slower than a fresh HF download every session.
export UV_PYTHON_INSTALL_DIR=/workspace/.uv/python
export UV_CACHE_DIR=/workspace/.uv/cache
export UV_LINK_MODE=copy
mkdir -p /workspace/.uv/{bin,python,cache}

# Copy the uv binary to the volume on first run. RunPod PyTorch images already
# ship uv, so pip install is a fallback only.
if [ ! -f "/workspace/.uv/bin/uv" ]; then
    _UV_SRC=$(command -v uv 2>/dev/null || true)
    if [ -z "$_UV_SRC" ]; then
        pip install uv --quiet
        _UV_SRC=$(command -v uv)
    fi
    cp "$_UV_SRC" /workspace/.uv/bin/uv
    echo "uv binary cached to /workspace/.uv/bin/ (future pods skip this step)"
fi
export PATH="/workspace/.uv/bin:$HOME/.local/bin:$PATH"

# ── Repo ─────────────────────────────────────────────────────────────────────
REPO_URL="https://${GITHUB_TOKEN}@github.com/danieltennant/unlearning-compression.git"
if [ ! -d "$WORK_DIR/.git" ]; then
    echo "=== Cloning repo (first time on this volume) ==="
    git clone "$REPO_URL" "$WORK_DIR"
fi
cd "$WORK_DIR"
git remote set-url origin "$REPO_URL"
git pull
git config --global user.email "danieltennant@users.noreply.github.com"
git config --global user.name "Daniel Tennant"
git submodule update --init --recursive
uv sync

# ── Open-unlearning extras ────────────────────────────────────────────────────
# deepspeed and bitsandbytes>=0.45.0 are in pyproject.toml; uv sync handles them.
# open-unlearning/requirements.txt also pulls in ninja, tensorboard, grpcio, etc.
# Gate on ninja (fast metadata check) so we only install once per volume.
# Exclude bitsandbytes from the requirements.txt install to avoid version conflicts.
_NINJA_OK=$(uv pip show ninja 2>/dev/null | grep -c '^Version:' || echo "0")
if [[ "$_NINJA_OK" == "0" ]]; then
    echo "=== Installing open-unlearning training deps (one-time per volume) ==="
    grep -v -i 'bitsandbytes' "$OPEN_UNLEARNING_DIR/requirements.txt" \
        | uv pip install -r /dev/stdin
fi

# ── HuggingFace auth ─────────────────────────────────────────────────────────
uv run huggingface-cli login --token "$HF_TOKEN"

# ── TOFU eval data ────────────────────────────────────────────────────────────
cd "$OPEN_UNLEARNING_DIR" && uv run python setup_data.py && cd "$WORK_DIR"

export WORK_DIR OPEN_UNLEARNING_DIR REPO_URL
