#!/bin/bash
# 8B RMU training + compression sweep
# Hyperparameters: layer 7, scoeff 2, lr 1e-5 (matches RMU.yaml defaults and repro.md targets)
# If utility collapses (<0.4), retry with layer 14 (proportionally equivalent to layer 7 in 1B)
# Run from /workspace/unlearning-compression
# Requires: HF_HOME set, huggingface-cli logged in

set -e
cd /workspace/unlearning-compression

export HF_HOME=/workspace/.cache/huggingface
PYTHON=/workspace/unlearning-compression/.venv/bin/python
ACCELERATE=/workspace/unlearning-compression/.venv/bin/accelerate

echo "=== [1/5] Train 8B RMU (layer 7, scoeff 2) ==="
cd open-unlearning
CUDA_VISIBLE_DEVICES=0 $ACCELERATE launch \
    --config_file configs/accelerate/single_gpu_config.yaml \
    src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/tofu/default.yaml \
    trainer=RMU \
    model=Llama-3.1-8B-Instruct \
    task_name=tofu_Llama-3.1-8B-Instruct_forget10_RMU_layer7_scoeff2 \
    forget_split=forget10 \
    retain_split=retain90 \
    mode=unlearn \
    model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.1-8B-Instruct_full \
    model.model_args.attn_implementation=eager \
    retain_logs_path=null \
    trainer.args.learning_rate=1e-5 \
    trainer.args.per_device_train_batch_size=2 \
    trainer.args.gradient_accumulation_steps=16 \
    trainer.args.gradient_checkpointing=true \
    trainer.args.report_to=none \
    ++trainer.args.remove_unused_columns=false \
    "trainer.method_args.module_regex=model\.layers\.7" \
    trainer.method_args.steering_coeff=2
cd ..
echo "=== [1/5] training done ==="

RMU_PATH="open-unlearning/saves/unlearn/tofu_Llama-3.1-8B-Instruct_forget10_RMU_layer7_scoeff2"

echo "=== [2/5] Baseline eval ==="
$PYTHON experiments/eval_compressed.py \
    --model_id "$RMU_PATH" \
    --compression none \
    --forget_split forget10 \
    --output_dir results/ \
    --model_name Llama-3.1-8B-Instruct
echo "=== [2/5] done ==="

echo "=== [3/5] 4-bit quantization ==="
$PYTHON experiments/eval_compressed.py \
    --model_id "$RMU_PATH" \
    --compression quantize --level 4 \
    --forget_split forget10 \
    --output_dir results/ \
    --model_name Llama-3.1-8B-Instruct
echo "=== [3/5] done ==="

echo "=== [4/5] 8-bit quantization ==="
$PYTHON experiments/eval_compressed.py \
    --model_id "$RMU_PATH" \
    --compression quantize --level 8 \
    --forget_split forget10 \
    --output_dir results/ \
    --model_name Llama-3.1-8B-Instruct
echo "=== [4/5] done ==="

echo "=== [5/5] Pruning sweep (10%, 20%, 30%) ==="
for sparsity in 0.1 0.2 0.3; do
    echo "--- pruning ${sparsity} ---"
    $PYTHON experiments/eval_compressed.py \
        --model_id "$RMU_PATH" \
        --compression prune --level $sparsity \
        --forget_split forget10 \
        --output_dir results/ \
        --model_name Llama-3.1-8B-Instruct
done
echo "=== [5/5] done ==="

echo "=== ALL DONE ==="
