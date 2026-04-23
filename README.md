# unlearning-compression

Does compression reverse machine unlearning?

[Guo et al. (2024)](https://arxiv.org/abs/2410.16454) showed that quantizing an unlearned LLM recovers 83% of supposedly forgotten knowledge on average, using NEWS and BOOKS datasets. This project translates that finding to the [TOFU](https://locuslab.github.io/tofu/) benchmark — a standardized unlearning testbed with 450+ public checkpoints via [open-unlearning](https://github.com/locuslab/open-unlearning) — and extends it to magnitude pruning and SVD truncation.

The core hypothesis transfers: if utility-constrained unlearning suppresses rather than erases knowledge, that should be visible under any compression method that reduces weight perturbations, not just quantization. TOFU provides a cleaner test than NEWS/BOOKS because the forget set is entirely synthetic, eliminating confounds from the model's pretraining exposure.

If the failure generalizes across compression methods, it substantially weakens the safety case for unlearning as a capability control. If it doesn't, that points toward something specific about how quantization interacts with unlearning weight perturbations — which is equally useful for understanding fixes.

## Setup

```bash
git clone --recurse-submodules https://github.com/danieltennant/unlearning-compression
cd unlearning-compression
uv sync
```

## Structure

```
src/compress/
    quantize.py     # bitsandbytes 4-bit / 8-bit quantization
    prune.py        # unstructured magnitude pruning
    svd.py          # SVD truncation
experiments/        # sweep scripts
results/            # output (gitignored)
notebooks/          # analysis
open-unlearning/    # submodule — eval harness and checkpoints
```

## Compression methods

| Method | Variants |
|---|---|
| Quantization | 4-bit, 8-bit (bitsandbytes) |
| Magnitude pruning | 10%, 20%, 30% sparsity |
| SVD truncation | Retain top 90%, 80%, 70% of singular values |

## Results

Full evaluation results pending — will run on Llama 3.2 1B and Llama 3.1 8B across four unlearning methods (GradAscent, NPO, GradDiff, RMU) at 5% and 10% forget fractions via RunPod.

**Local validation (Apr 2026):** End-to-end pipeline validated locally on GPT-2 with magnitude pruning. Model loading, compression, Hydra config composition, TOFU data download, and the eval loop all run correctly. Two metrics (forget_Q_A_Prob, forget_Q_A_ROUGE) completed successfully on CPU. The `forget_quality` metric requires pre-computed retain logs from the target checkpoint, which will be available when running against real open-unlearning checkpoints on RunPod.

## Metrics

- **Forget accuracy** — does the model recover knowledge of the forget set after compression?
- **Retain accuracy** — does compression preserve performance on the retain set?

## References

- Guo et al., "Catastrophic Failure of LLM Unlearning via Quantization" (2024) — https://arxiv.org/abs/2410.16454
- Maini et al., "TOFU: A Task of Fictitious Unlearning for LLMs" (2024) — https://locuslab.github.io/tofu/
- open-unlearning — https://github.com/locuslab/open-unlearning
