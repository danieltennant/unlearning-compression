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
#   5. Install open-unlearning deps only when bitsandbytes < 0.45 or deepspeed missing
#   6. Authenticate HuggingFace
#   7. Run setup_data.py (idempotent — downloads to open-unlearning/saves/eval)
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
export HF_HOME=/workspace/.cache/huggingface
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
cd "$WORK_DIR"
git remote set-url origin "$REPO_URL"
git pull
git config --global user.email "danieltennant@users.noreply.github.com"
git config --global user.name "Daniel Tennant"
git submodule update --init --recursive
uv sync

# ── Open-unlearning deps: skip if already at working versions ─────────────────
# open-unlearning/requirements.txt pins bitsandbytes==0.44.1 which breaks on
# H100/CUDA 13; we always override to >=0.45.0. This block avoids the 116MB
# bitsandbytes re-download and deepspeed recompile on every session.
_BB_VER=$(uv run python -c "import bitsandbytes as b; print(b.__version__)" 2>/dev/null || echo "0.0.0")
_DS_OK=$(uv run python -c "import deepspeed" 2>/dev/null && echo "ok" || echo "missing")
if [[ "$_BB_VER" < "0.45.0" ]] || [[ "$_DS_OK" != "ok" ]]; then
    echo "=== Installing open-unlearning deps (one-time per volume) ==="
    uv pip install -r "$OPEN_UNLEARNING_DIR/requirements.txt"
    uv pip install "bitsandbytes>=0.45.0"
else
    echo "open-unlearning deps OK (bitsandbytes $BB_VER, deepspeed present)"
fi

# ── HuggingFace auth ─────────────────────────────────────────────────────────
uv run huggingface-cli login --token "$HF_TOKEN"

# ── TOFU eval data ────────────────────────────────────────────────────────────
cd "$OPEN_UNLEARNING_DIR" && uv run python setup_data.py && cd "$WORK_DIR"

export WORK_DIR OPEN_UNLEARNING_DIR REPO_URL
