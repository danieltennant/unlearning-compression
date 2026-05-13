#!/bin/bash
# Weight delta + nnsight mechanistic analysis
#
# Prerequisites:
#   - RMU checkpoint at open-unlearning/saves/unlearn/tofu_Llama-3.1-8B-Instruct_forget10_RMU_layer7_scoeff2
#   - GradDiff already on HF: dtennant/tofu-llama-8b-graddiff-alpha1
#   - SimNPO already on HF: dtennant/tofu-llama-8b-simnpo
#   - huggingface-cli logged in
#
# Run from /workspace/unlearning-compression

set -e
cd /workspace/unlearning-compression

export HF_HOME=/workspace/.cache/huggingface
PYTHON=/workspace/unlearning-compression/.venv/bin/python

RMU_LOCAL="open-unlearning/saves/unlearn/tofu_Llama-3.1-8B-Instruct_forget10_RMU_layer7_scoeff2"
RMU_HF="dtennant/tofu-llama-8b-rmu"
FULL_MODEL="open-unlearning/tofu_Llama-3.1-8B-Instruct_full"
GRADDIFF_HF="dtennant/tofu-llama-8b-graddiff-alpha1"
SIMNPO_HF="dtennant/tofu-llama-8b-simnpo"

# ── [1/4] Install nnsight ──────────────────────────────────────────────────────

echo "=== [1/4] Install nnsight ==="
.venv/bin/pip install "nnsight>=0.3" --quiet
echo "=== [1/4] done ==="

# ── [2/4] Push RMU checkpoint to HuggingFace ──────────────────────────────────

echo "=== [2/4] Push RMU checkpoint to HuggingFace ==="
$PYTHON -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='$RMU_LOCAL',
    repo_id='$RMU_HF',
    repo_type='model',
)
print('Pushed to $RMU_HF')
"
echo "=== [2/4] done ==="

# ── [3/4] Weight delta analysis (Phase 1) ─────────────────────────────────────
# Compares each unlearned model to the full model weight-by-weight.
# Outputs: per-layer stats JSON + aggregated stats JSON to results/weight_delta_8b_{method}/

echo "=== [3/4] Weight delta analysis ==="

for METHOD_NAME in GradDiff SimNPO RMU; do
    case $METHOD_NAME in
        GradDiff) MODEL_ID="$GRADDIFF_HF" ;;
        SimNPO)   MODEL_ID="$SIMNPO_HF" ;;
        RMU)      MODEL_ID="$RMU_HF" ;;
    esac

    OUT="results/weight_delta_8b_$(echo $METHOD_NAME | tr '[:upper:]' '[:lower:]')"
    echo "--- $METHOD_NAME ---"
    $PYTHON experiments/weight_delta_analysis.py \
        --full_model_id "$FULL_MODEL" \
        --unlearned_model_id "$MODEL_ID" \
        --output_dir "$OUT"
    echo "--- $METHOD_NAME done ---"
done

echo "=== [3/4] done ==="

# ── [4/4] nnsight analysis (logit lens + activation patching) ─────────────────
# Runs logit lens and activation patching for all three methods.
# Peak VRAM: one 8B model at a time (~16 GB) — H100 fine.
# Outputs: results/nnsight/logit_lens.{pdf,png}, activation_patching.{pdf,png}, summary.json

echo "=== [4/4] nnsight analysis ==="
$PYTHON experiments/nnsight_analysis.py \
    --full_model_id "$FULL_MODEL" \
    --method GradDiff "$GRADDIFF_HF" \
    --method SimNPO "$SIMNPO_HF" \
    --method RMU "$RMU_HF" \
    --n_questions 50 \
    --output_dir results/nnsight
echo "=== [4/4] done ==="

echo "=== ALL DONE ==="
echo "Results:"
echo "  Weight delta: results/weight_delta_8b_{graddiff,simnpo,rmu}/"
echo "  nnsight:      results/nnsight/"
