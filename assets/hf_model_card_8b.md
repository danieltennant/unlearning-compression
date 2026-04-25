---
base_model: open-unlearning/tofu_Llama-3.1-8B-Instruct_full
license: llama3.1
tags:
  - machine-unlearning
  - tofu
  - llama
  - safety
datasets:
  - locuslab/TOFU
---

# tofu-llama-8b-graddiff-alpha1

A Llama-3.1-8B-Instruct checkpoint unlearned on the [TOFU](https://arxiv.org/abs/2401.06121) benchmark using Gradient Difference (GradDiff) with α=1.0. Trained as part of research into how post-training compression (quantization, pruning, SVD) interacts with machine unlearning.

## Model lineage

| Step | Model |
|------|-------|
| Base | `meta-llama/Llama-3.1-8B-Instruct` |
| Fine-tuned on TOFU full | `open-unlearning/tofu_Llama-3.1-8B-Instruct_full` |
| Unlearned (this model) | `dtennant/tofu-llama-8b-graddiff-alpha1` |

## Unlearning details

| Parameter | Value |
|-----------|-------|
| Method | GradDiff |
| α (retain weight) | 1.0 |
| Forget split | forget10 (10% of TOFU authors) |
| Retain split | retain90 |
| Learning rate | 1e-5 |
| Epochs | 10 |
| Batch size | 2 (× 16 gradient accumulation steps) |
| Hardware | 1× H100 SXM (~35 min) |

Training used [open-unlearning](https://github.com/ai-safety-foundation/open-unlearning).

## Intended use

Research checkpoint for studying the effect of post-training compression on unlearning quality. Not intended for deployment.
