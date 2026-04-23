#!/usr/bin/env bash
# Train GradDiff alpha1 on Llama-3.1-8B-Instruct for TOFU forget10,
# then push the checkpoint to HuggingFace and run compression evals.
#
# Designed to run on a fresh pod (no network volume needed).
# All outputs are pushed to HuggingFace and GitHub.
#
# Usage:
#   bash scripts/train_8b.sh
#
# Required env vars (set in pod before running):
#   HF_TOKEN       — HuggingFace token with write access
#   HF_REPO        — HuggingFace repo to push checkpoint to
#                    (e.g. danieltennant/tofu-llama-8b-graddiff-alpha1)
#   GITHUB_TOKEN   — GitHub PAT with Contents read/write
#   HF_LLAMA_TOKEN — Token with access to meta-llama/Llama-3.1-8B-Instruct
#                    (can be same as HF_TOKEN if account has access)

set -e

REPO_URL="https://${GITHUB_TOKEN}@github.com/danieltennant/unlearning-compression.git"
WORK_DIR="/tmp/unlearning-compression"
OPEN_UNLEARNING_DIR="$WORK_DIR/open-unlearning"
CHECKPOINT_DIR="/tmp/checkpoint_8b"

echo "=== Setting up environment ==="
pip install uv --quiet
export UV_LINK_MODE=copy

echo "=== Cloning repo ==="
git clone "$REPO_URL" "$WORK_DIR"
cd "$WORK_DIR"
git config --global user.email "danieltennant@users.noreply.github.com"
git config --global user.name "Daniel Tennant"
git submodule update --init --recursive
UV_LINK_MODE=copy uv sync

echo "=== Authenticating HuggingFace ==="
uv run huggingface-cli login --token "$HF_TOKEN"

echo "=== Setting up TOFU data ==="
cd "$OPEN_UNLEARNING_DIR"
python setup_data.py

echo "=== Training GradDiff alpha1 on Llama-3.1-8B-Instruct ==="
cd "$OPEN_UNLEARNING_DIR"
uv run python src/train.py \
    model=Llama-3.1-8B-Instruct \
    trainer=GradDiff \
    'trainer.method_args.alpha=1.0' \
    'trainer.args.learning_rate=1e-5' \
    'trainer.args.num_train_epochs=10' \
    'trainer.args.per_device_train_batch_size=2' \
    'trainer.args.gradient_accumulation_steps=16' \
    data=tofu \
    'data.forget_split=forget10' \
    'data.retain_split=retain90' \
    task_name=tofu_Llama-3.1-8B-Instruct_forget10_GradDiff_lr1e-05_alpha1_epoch10 \
    "paths.output_dir=$CHECKPOINT_DIR"

echo "=== Pushing checkpoint to HuggingFace ==="
cd "$WORK_DIR"
uv run python - <<PYEOF
from huggingface_hub import HfApi
import os
api = HfApi()
api.upload_folder(
    folder_path="$CHECKPOINT_DIR",
    repo_id=os.environ["HF_REPO"],
    repo_type="model",
)
print(f"Pushed to https://huggingface.co/{os.environ['HF_REPO']}")
PYEOF

echo "=== Generating retain baseline for 8B ==="
uv run python experiments/eval_compressed.py \
    --model_id open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90 \
    --compression none \
    --forget_split forget10 \
    --holdout_split holdout10 \
    --output_dir results/retain_baseline_8b

# Run the 8B eval sweep
bash scripts/run_sweep.sh sweeps/2026-04-23-8b.sh

echo "=== Done ==="
