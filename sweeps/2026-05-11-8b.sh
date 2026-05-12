#!/bin/bash
# 8B experiments: GradDiff 20% pruning + SimNPO training + compression sweep
# Run from /workspace/unlearning-compression
# Requires: HF_HOME set to local disk, huggingface-cli logged in

set -e
cd /workspace/unlearning-compression

export HF_HOME=/root/.cache/huggingface

echo "=== [1/7] 8B GradDiff 20% pruning ==="
.venv/bin/python experiments/eval_compressed.py \
    --model_id dtennant/tofu-llama-8b-graddiff-alpha1 \
    --compression prune \
    --level 0.2 \
    --forget_split forget10 \
    --output_dir results/
echo "=== [1/7] done ==="

echo "=== [2/7] Train 8B SimNPO ==="
cd open-unlearning
CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --config_file configs/accelerate/single_gpu_config.yaml \
    src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/tofu/default.yaml \
    trainer=SimNPO \
    model=Llama-3.1-8B-Instruct \
    task_name=tofu_Llama-3.1-8B-Instruct_forget10_SimNPO \
    forget_split=forget10 \
    retain_split=retain90 \
    mode=unlearn \
    model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.1-8B-Instruct_full \
    model.model_args.attn_implementation=eager \
    retain_logs_path=null \
    trainer.args.learning_rate=2e-5 \
    trainer.args.per_device_train_batch_size=2 \
    trainer.args.gradient_accumulation_steps=16 \
    trainer.args.gradient_checkpointing=true \
    ++trainer.args.remove_unused_columns=false
cd ..
echo "=== [2/7] training done ==="

echo "=== [3/7] Push 8B SimNPO to HuggingFace ==="
.venv/bin/python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='open-unlearning/saves/unlearn/tofu_Llama-3.1-8B-Instruct_forget10_SimNPO',
    repo_id='dtennant/tofu-llama-8b-simnpo',
    repo_type='model',
)
print('Pushed to dtennant/tofu-llama-8b-simnpo')
"
echo "=== [3/7] done ==="

SIMNPO_8B="dtennant/tofu-llama-8b-simnpo"

echo "=== [4/7] 8B SimNPO baseline (uncompressed) ==="
.venv/bin/python experiments/eval_compressed.py \
    --model_id "$SIMNPO_8B" \
    --compression none \
    --forget_split forget10 \
    --output_dir results/
echo "=== [4/7] done ==="

echo "=== [5/7] 8B SimNPO 8-bit quantization ==="
.venv/bin/python experiments/eval_compressed.py \
    --model_id "$SIMNPO_8B" \
    --compression quantize --level 8 \
    --forget_split forget10 \
    --output_dir results/
echo "=== [5/7] done ==="

echo "=== [6/7] 8B SimNPO 4-bit quantization ==="
.venv/bin/python experiments/eval_compressed.py \
    --model_id "$SIMNPO_8B" \
    --compression quantize --level 4 \
    --forget_split forget10 \
    --output_dir results/
echo "=== [6/7] done ==="

echo "=== [7/7] 8B SimNPO pruning sweep (10%, 20%, 30%) ==="
for sparsity in 0.1 0.2 0.3; do
    echo "--- pruning ${sparsity} ---"
    .venv/bin/python experiments/eval_compressed.py \
        --model_id "$SIMNPO_8B" \
        --compression prune --level $sparsity \
        --forget_split forget10 \
        --output_dir results/
done
echo "=== [7/7] done ==="

echo "=== ALL DONE ==="
