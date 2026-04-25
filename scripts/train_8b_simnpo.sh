#!/usr/bin/env bash
# Train SimNPO on Llama-3.1-8B-Instruct (forget10 split).
# Mirror of the GradDiff alpha1 training used for the 8B sweep, but with SimNPO.
# Hyperparameters match the 1B SimNPO checkpoint: lr=2e-5, beta=4.5, alpha=1.
#
# Saves checkpoint to /workspace/checkpoint_8b_simnpo (or HF if HF_REPO_SIMNPO set).
# Estimated runtime: ~40 min on H100 SXM.
#
# Usage:
#   bash scripts/train_8b_simnpo.sh

set -e

for env_file in /workspace/.env /root/.env; do
    if [ -f "$env_file" ]; then
        export $(grep -v '^#' "$env_file" | xargs)
        break
    fi
done

export HF_HOME=/workspace/.cache/huggingface
export PATH="$HOME/.local/bin:$PATH"

WORK_DIR="${WORK_DIR:-/workspace/unlearning-compression}"
OPEN_UNLEARNING_DIR="$WORK_DIR/open-unlearning"
CKPT_DIR="/workspace/checkpoint_8b_simnpo"

echo "=== Training 8B SimNPO ==="
cd "$OPEN_UNLEARNING_DIR"

uv run python src/trainer.py \
    --config-name=forget \
    model=Llama-3.1-8B-Instruct \
    mode=unlearn \
    trainer=SimNPO \
    data=tofu \
    data.forget_split=forget10 \
    data.retain_split=retain90 \
    trainer.args.learning_rate=2e-5 \
    trainer.args.num_train_epochs=10 \
    trainer.args.per_device_train_batch_size=2 \
    trainer.args.gradient_accumulation_steps=16 \
    trainer.SimNPO.beta=4.5 \
    trainer.SimNPO.alpha=1 \
    trainer.SimNPO.gamma=0.25 \
    trainer.args.output_dir="$CKPT_DIR" \
    model.attn_implementation=eager \
    ++trainer.args.remove_unused_columns=false

echo "=== Checkpoint saved to $CKPT_DIR ==="

# Push to HF if token and repo are set
if [ -n "$HF_TOKEN" ] && [ -n "${HF_REPO_SIMNPO:-}" ]; then
    echo "=== Pushing to HuggingFace: $HF_REPO_SIMNPO ==="
    cd "$WORK_DIR"
    uv run python - <<EOF
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("$HF_REPO_SIMNPO", exist_ok=True)
api.upload_folder(
    folder_path="$CKPT_DIR",
    repo_id="$HF_REPO_SIMNPO",
    commit_message="Upload 8B SimNPO checkpoint (forget10, lr2e-5, beta4.5)"
)
print("Upload complete.")
EOF
fi
